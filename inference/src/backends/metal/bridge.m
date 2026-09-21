// Model-independent Metal boundary. The bridge owns device, queue, compiled
// pipelines, and buffers; it knows nothing about layers, encodings, or shapes.
// Zig records dispatches between nu_metal_begin and nu_metal_commit; commit
// waits for completion, so CPU access to shared memory after it is race-free
// and destruction cannot overlap submitted work.
//
// Profiling (opt-in, nu_metal_profile_enable): Apple GPUs sample timestamps only
// at encoder stage boundaries, never per dispatch, so in profile mode every
// dispatch is recorded in its own compute encoder whose start/end timestamps
// land in a counter sample buffer. Commit resolves them into seconds per
// dispatch, in recording order; Zig knows which kernel each index was.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

// Observed device limit for one counter sample buffer: 32768 bytes of 8-byte
// timestamps. Two samples per dispatch, so 2048 dispatches per buffer.
static const uint32_t samples_per_buffer = 4096;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableArray<id<MTLComputePipelineState>> * pipelines;
    NSMutableArray<id<MTLBuffer>> * buffers;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder; // nil while profiling: one encoder per dispatch
    double gpu_seconds; // accumulated GPU busy time of completed command buffers
    uint32_t dispatches; // recorded in the current command buffer
    // Profiling state; `samples` is nil when profiling is off.
    NSMutableArray<id<MTLCounterSampleBuffer>> * samples;
    uint32_t sample_capacity; // dispatches that can be timed per command buffer
    uint32_t sampled; // timed dispatches in the current command buffer
    uint32_t resolved; // entries valid in `durations` after the last commit
    double * durations; // seconds per timed dispatch, `sample_capacity` entries
} NuMetal;

// Timestamp counters are undocumented in unit; measured on Apple M4 Pro (macOS
// 26) they are nanoseconds on the same timeline as MTLCommandBuffer.GPUStartTime
// (the first encoder's start stamp / 1e9 equals GPUStartTime to the microsecond).
static const double seconds_per_timestamp = 1e-9;

typedef struct { uint32_t buffer; size_t offset; } NuBinding;

static void message(char * error, size_t capacity, NSString * text) {
    if (capacity) snprintf(error, capacity, "%s", text.UTF8String ?: "Metal failure");
}

void nu_metal_destroy(void * opaque) {
    if (!opaque) return;
    NuMetal * m = opaque;
    if (m->encoder) { [m->encoder endEncoding]; [m->encoder release]; }
    if (m->command) { [m->command waitUntilCompleted]; [m->command release]; }
    [m->samples release]; free(m->durations);
    [m->buffers release]; [m->pipelines release]; [m->library release];
    [m->queue release]; [m->device release];
    free(m);
}

void * nu_metal_create(const char * source, size_t length, char * error, size_t capacity) {
    @autoreleasepool {
        NuMetal * m = calloc(1, sizeof(*m));
        if (!m) { message(error, capacity, @"Host allocation failed"); return NULL; }
        m->device = MTLCreateSystemDefaultDevice();
        if (!m->device) { message(error, capacity, @"Metal device unavailable"); nu_metal_destroy(m); return NULL; }
        m->queue = [m->device newCommandQueue];
        NSString * text = [[NSString alloc] initWithBytes:source length:length encoding:NSUTF8StringEncoding];
        MTLCompileOptions * options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe; // no reassociation or contraction: comparable with the CPU reference
        NSError * failure = nil;
        m->library = [m->device newLibraryWithSource:text options:options error:&failure];
        [text release]; [options release];
        m->pipelines = [NSMutableArray new];
        m->buffers = [NSMutableArray new];
        if (!m->queue || !m->library || !m->pipelines || !m->buffers) {
            message(error, capacity, failure.localizedDescription ?: @"Metal preparation failed");
            nu_metal_destroy(m); return NULL;
        }
        return m;
    }
}

// Compiles one kernel function into a pipeline; the returned id is stable.
int nu_metal_pipeline(void * opaque, const char * name, uint32_t * out_id, char * error, size_t capacity) {
    @autoreleasepool {
        NuMetal * m = opaque;
        id<MTLFunction> function = [m->library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!function) { message(error, capacity, [NSString stringWithFormat:@"missing kernel %s", name]); return 1; }
        NSError * failure = nil;
        id<MTLComputePipelineState> pipeline = [m->device newComputePipelineStateWithFunction:function error:&failure];
        [function release];
        if (!pipeline) { message(error, capacity, failure.localizedDescription ?: @"pipeline creation failed"); return 1; }
        *out_id = (uint32_t)m->pipelines.count;
        [m->pipelines addObject:pipeline]; [pipeline release];
        return 0;
    }
}

size_t nu_metal_max_buffer_length(void * opaque) { return ((NuMetal *)opaque)->device.maxBufferLength; }

// Zero-filled shared buffer owned by the bridge for the handle's lifetime.
int nu_metal_buffer_create(void * opaque, size_t length, uint32_t * out_id) {
    @autoreleasepool {
        NuMetal * m = opaque;
        if (length == 0 || length > m->device.maxBufferLength || m->buffers.count >= UINT32_MAX) return 1;
        id<MTLBuffer> buffer = [m->device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (!buffer) return 1;
        memset(buffer.contents, 0, length);
        *out_id = (uint32_t)m->buffers.count;
        [m->buffers addObject:buffer]; [buffer release];
        return 0;
    }
}

// Borrowed memory must outlive this handle. The no-copy buffer covers whole
// pages; the returned offset restores the exact start. Nothing unmaps it.
int nu_metal_buffer_wrap(void * opaque, const void * bytes, size_t length, uint32_t * out_id, size_t * offset) {
    @autoreleasepool {
        NuMetal * m = opaque;
        size_t page = (size_t)getpagesize();
        uintptr_t base = (uintptr_t)bytes & ~(page - 1);
        *offset = (uintptr_t)bytes - base;
        if (length > SIZE_MAX - *offset - (page - 1)) return 1;
        size_t size = (length + *offset + page - 1) & ~(page - 1);
        if (size > m->device.maxBufferLength || m->buffers.count >= UINT32_MAX) return 1;
        id<MTLBuffer> buffer = [m->device newBufferWithBytesNoCopy:(void *)base length:size options:MTLResourceStorageModeShared deallocator:nil];
        if (!buffer) return 1;
        *out_id = (uint32_t)m->buffers.count;
        [m->buffers addObject:buffer]; [buffer release];
        return 0;
    }
}

void * nu_metal_buffer_contents(void * opaque, uint32_t index) {
    NuMetal * m = opaque;
    if (index >= m->buffers.count) return NULL;
    return m->buffers[index].contents;
}

// Opens a command buffer with one serial compute pass: dispatches execute in
// order and each one's writes are visible to the next. Separate encoders in
// one command buffer (profile mode) keep the same ordering guarantee.
int nu_metal_begin(void * opaque) {
    @autoreleasepool {
        NuMetal * m = opaque;
        if (m->command || m->encoder) return 1;
        m->command = [[m->queue commandBuffer] retain];
        if (!m->command) return 1;
        if (!m->samples) {
            m->encoder = [[m->command computeCommandEncoder] retain];
            if (!m->encoder) { [m->command release]; m->command = nil; return 1; }
        }
        m->dispatches = 0;
        m->sampled = 0;
        return 0;
    }
}

// Profile mode: an encoder for exactly one dispatch, timestamped at its start
// and end. Beyond the sample capacity the dispatch still runs, untimed.
static id<MTLComputeCommandEncoder> profiledEncoder(NuMetal * m) {
    if (m->sampled >= m->sample_capacity) return [m->command computeCommandEncoder];
    uint32_t sample = m->sampled * 2;
    MTLComputePassDescriptor * pass = [MTLComputePassDescriptor computePassDescriptor];
    MTLComputePassSampleBufferAttachmentDescriptor * attachment = pass.sampleBufferAttachments[0];
    attachment.sampleBuffer = m->samples[sample / samples_per_buffer];
    attachment.startOfEncoderSampleIndex = sample % samples_per_buffer;
    attachment.endOfEncoderSampleIndex = sample % samples_per_buffer + 1;
    id<MTLComputeCommandEncoder> encoder = [m->command computeCommandEncoderWithDescriptor:pass];
    if (encoder) m->sampled += 1;
    return encoder;
}

// Records one kernel: buffers at indices 0..count-1, constants at index 7.
int nu_metal_dispatch(void * opaque, uint32_t pipeline, const NuBinding * bindings, uint32_t count,
                      const void * constants, size_t constants_length,
                      uint32_t groups_x, uint32_t groups_y, uint32_t threads_x, uint32_t threads_y) {
    @autoreleasepool {
        NuMetal * m = opaque;
        if (!m->command || (!m->encoder && !m->samples)) return 1;
        if (pipeline >= m->pipelines.count || count > 7 || constants_length > 4096) return 1;
        if (groups_x == 0 || groups_y == 0 || threads_x == 0 || threads_y == 0) return 1;
        id<MTLComputePipelineState> state = m->pipelines[pipeline];
        if (threads_x * threads_y > state.maxTotalThreadsPerThreadgroup) return 1;
        for (uint32_t i = 0; i < count; ++i) {
            if (bindings[i].buffer >= m->buffers.count) return 1;
            id<MTLBuffer> buffer = m->buffers[bindings[i].buffer];
            if (bindings[i].offset >= buffer.length) return 1;
        }
        id<MTLComputeCommandEncoder> encoder = m->encoder ?: profiledEncoder(m);
        if (!encoder) return 1;
        [encoder setComputePipelineState:state];
        for (uint32_t i = 0; i < count; ++i)
            [encoder setBuffer:m->buffers[bindings[i].buffer] offset:bindings[i].offset atIndex:i];
        if (constants_length) [encoder setBytes:constants length:constants_length atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1) threadsPerThreadgroup:MTLSizeMake(threads_x, threads_y, 1)];
        if (!m->encoder) [encoder endEncoding];
        m->dispatches += 1;
        return 0;
    }
}

// Reads the resolved timestamps of the completed command buffer into
// `durations`. A dispatch whose timestamps the GPU did not report gets -1.
static void resolveProfile(NuMetal * m) {
    m->resolved = 0;
    for (uint32_t i = 0; i < m->sampled; i += samples_per_buffer / 2) {
        uint32_t in_buffer = MIN(m->sampled - i, samples_per_buffer / 2);
        NSData * data = [m->samples[i * 2 / samples_per_buffer] resolveCounterRange:NSMakeRange(0, in_buffer * 2)];
        const MTLCounterResultTimestamp * stamps = data.bytes;
        for (uint32_t j = 0; j < in_buffer; ++j) {
            double seconds = -1;
            if (stamps && data.length >= (j * 2 + 2) * sizeof(*stamps)) {
                MTLTimestamp start = stamps[j * 2].timestamp, end = stamps[j * 2 + 1].timestamp;
                if (start != MTLCounterErrorValue && end != MTLCounterErrorValue && end >= start)
                    seconds = (double)(end - start) * seconds_per_timestamp;
            }
            m->durations[i + j] = seconds;
        }
    }
    m->resolved = m->sampled;
}

// Ends the pass, submits, waits, and accounts GPU time. Returns 0 on completion.
// With a `tick`, the wait is a semaphore the completion handler signals, timed
// out every `tick_interval_ns` to call back: completion still returns at once,
// so only a wait longer than the interval ever pays for the callback.
static const int64_t tick_interval_ns = 100 * 1000 * 1000;

int nu_metal_commit(void * opaque, void (*tick)(void *), void * tick_context) {
    @autoreleasepool {
        NuMetal * m = opaque;
        if (!m->command || (!m->encoder && !m->samples)) return 1;
        if (m->encoder) { [m->encoder endEncoding]; [m->encoder release]; m->encoder = nil; }
        id<MTLCommandBuffer> command = m->command;
        m->command = nil;
        m->resolved = 0;
        int status = 0;
        if (m->dispatches) {
            if (tick) {
                dispatch_semaphore_t done = dispatch_semaphore_create(0);
                [command addCompletedHandler:^(id<MTLCommandBuffer> finished) { (void)finished; dispatch_semaphore_signal(done); }];
                [command commit];
                while (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, tick_interval_ns)) != 0) tick(tick_context);
                dispatch_release(done);
            } else {
                [command commit];
                [command waitUntilCompleted];
            }
            if (command.status != MTLCommandBufferStatusCompleted) status = 1;
            else {
                m->gpu_seconds += command.GPUEndTime - command.GPUStartTime;
                if (m->samples) resolveProfile(m);
            }
        }
        [command release];
        return status;
    }
}

double nu_metal_gpu_seconds(void * opaque) { return ((NuMetal *)opaque)->gpu_seconds; }

// Enables per-dispatch timing for all later command buffers; up to
// `max_dispatches` per command buffer are timed. Once, while idle.
int nu_metal_profile_enable(void * opaque, uint32_t max_dispatches, char * error, size_t capacity) {
    @autoreleasepool {
        NuMetal * m = opaque;
        if (m->command || m->samples || max_dispatches == 0 || max_dispatches > (1u << 20)) {
            message(error, capacity, @"profiling is enabled once, outside a command buffer, for at most 2^20 dispatches");
            return 1;
        }
        if (![m->device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            message(error, capacity, @"device does not sample counters at encoder boundaries");
            return 1;
        }
        id<MTLCounterSet> timestamps = nil;
        for (id<MTLCounterSet> set in m->device.counterSets)
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamps = set;
        if (!timestamps) { message(error, capacity, @"device has no timestamp counter set"); return 1; }
        MTLCounterSampleBufferDescriptor * descriptor = [MTLCounterSampleBufferDescriptor new];
        descriptor.counterSet = timestamps;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = samples_per_buffer;
        NSMutableArray<id<MTLCounterSampleBuffer>> * samples = [NSMutableArray new];
        double * durations = calloc(max_dispatches, sizeof(double));
        uint32_t buffers = (max_dispatches * 2 + samples_per_buffer - 1) / samples_per_buffer;
        NSError * failure = nil;
        for (uint32_t i = 0; durations && i < buffers; ++i) {
            id<MTLCounterSampleBuffer> buffer = [m->device newCounterSampleBufferWithDescriptor:descriptor error:&failure];
            if (!buffer) break;
            [samples addObject:buffer]; [buffer release];
        }
        [descriptor release];
        if (!durations || samples.count != buffers) {
            message(error, capacity, failure.localizedDescription ?: @"counter sample buffer allocation failed");
            [samples release]; free(durations);
            return 1;
        }
        m->samples = samples;
        m->durations = durations;
        m->sample_capacity = max_dispatches;
        return 0;
    }
}

// Copies the timed dispatches of the last completed command buffer, in
// recording order, and returns how many were written.
uint32_t nu_metal_profile_read(void * opaque, double * out, uint32_t capacity) {
    NuMetal * m = opaque;
    uint32_t count = MIN(m->resolved, capacity);
    if (count) memcpy(out, m->durations, count * sizeof(double));
    return count;
}
