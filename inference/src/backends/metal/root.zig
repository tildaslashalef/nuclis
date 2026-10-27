//! Metal backend: a thin Zig layer over the recording bridge. `Backend` owns
//! the device handle and compiled pipelines, hands out `Buffer` handles, and
//! records typed kernel dispatches between `begin()` and `commit()`. It knows
//! kernel contracts and shapes, not model semantics: layer schedules belong to
//! model adapters that call these encoders.
//!
//! Ownership: created buffers live until `deinit`. Wrapped buffers borrow the
//! caller's page-backed memory, which must outlive the backend. `commit()`
//! waits for completion, so CPU access to any buffer after it is race-free;
//! `discard()` drops a recording on an error path without running it.
//!
//! Profiling (`enableProfiling`) times every dispatch with GPU timestamps and
//! accumulates them per kernel and shape in `Profile`. It is a diagnostic
//! mode: each dispatch then runs in its own encoder, which costs GPU time that
//! the report exposes rather than hides.
//!
//! Layout: `kernels.zig` assembles the Metal source and names the kernels,
//! `profile.zig` holds the timing types, and each `ops/<kind>.zig` encodes the
//! kernels of `kernels/<kind>.metal`; `Backend` aliases their functions, so a
//! new encoder is its function plus one alias line below.
const std = @import("std");
const cpu = @import("../cpu/root.zig");
const quant = @import("../../quant/decode.zig");
/// Storage precision of an attention cache the attention encoders read.
pub const Precision = @import("../../runtime/session.zig").Precision;
pub const enabled = @import("build_options").metal;

const Binding = extern struct { buffer: u32, offset: usize };
extern fn nu_metal_create([*]const u8, usize, [*]u8, usize) ?*anyopaque;
extern fn nu_metal_destroy(*anyopaque) void;
extern fn nu_metal_pipeline(*anyopaque, [*:0]const u8, *u32, [*]u8, usize) c_int;
extern fn nu_metal_pipeline_stats(*anyopaque, u32, *[3]u32) c_int;
extern fn nu_metal_max_buffer_length(*anyopaque) usize;
extern fn nu_metal_buffer_create(*anyopaque, usize, *u32) c_int;
extern fn nu_metal_buffer_wrap(*anyopaque, [*]const u8, usize, *u32, *usize) c_int;
extern fn nu_metal_buffer_contents(*anyopaque, u32) ?[*]u8;
extern fn nu_metal_buffer_release(*anyopaque, u32) c_int;
extern fn nu_metal_begin(*anyopaque) c_int;
extern fn nu_metal_dispatch(*anyopaque, u32, [*]const Binding, u32, ?*const anyopaque, usize, u32, u32, u32, u32) c_int;
extern fn nu_metal_commit(*anyopaque, ?*const fn (?*anyopaque) callconv(.c) void, ?*anyopaque) c_int;
extern fn nu_metal_discard(*anyopaque) c_int;
extern fn nu_metal_gpu_seconds(*anyopaque) f64;
extern fn nu_metal_profile_enable(*anyopaque, u32, [*]u8, usize) c_int;
extern fn nu_metal_profile_read(*anyopaque, [*]f64, u32) u32;
extern fn nu_metal_capture_next(*anyopaque, [*:0]const u8, [*]u8, usize) c_int;

const kernels = @import("kernels.zig");
const source = kernels.source;
const kernel_names = kernels.kernel_names;
pub const Kernel = kernels.Kernel;
const profile_mod = @import("profile.zig");
pub const Profile = profile_mod.Profile;
pub const Tick = profile_mod.Tick;
pub const PipelineStats = profile_mod.PipelineStats;
const Shape = profile_mod.Shape;

/// A GPU-visible byte range. `slice` derives sub-ranges without new bindings.
pub const Buffer = struct {
    id: u32,
    offset: usize,
    len: usize,
    /// CPU pointer for created buffers; wrapped buffers return the borrowed
    /// memory. Valid to read after `commit()` and to write before `begin()`.
    host: [*]u8,

    pub fn slice(self: Buffer, byte_offset: usize, len: usize) Buffer {
        std.debug.assert(byte_offset + len <= self.len);
        return .{ .id = self.id, .offset = self.offset + byte_offset, .len = len, .host = self.host + byte_offset };
    }
    pub fn floats(self: Buffer) []f32 {
        return @as([*]f32, @ptrCast(@alignCast(self.host)))[0 .. self.len / 4];
    }
    fn binding(self: Buffer) Binding {
        return .{ .buffer = self.id, .offset = self.offset };
    }
    /// Whether `a_len` bytes at `a` and `b_len` bytes at `b` share a byte.
    pub fn overlaps(a: Buffer, a_len: usize, b: Buffer, b_len: usize) bool {
        if (a.id != b.id) return false;
        // Subtraction avoids overflow when validating caller-provided offsets.
        return if (a.offset <= b.offset) b.offset - a.offset < a_len else a.offset - b.offset < b_len;
    }
};

pub const Backend = struct {
    alloc: std.mem.Allocator,
    handle: *anyopaque,
    /// Called from `commit` while a submitted buffer runs; null waits silently.
    tick: ?Tick = null,
    pipelines: [kernel_names.len]u32,
    wrapped: std.AutoHashMapUnmanaged(usize, Buffer) = .empty,
    /// Row-partial buffer of the split-K matvec, created on first use.
    split_scratch: ?Buffer = null,
    recording: bool = false,
    /// Diagnostic: record the generic `nu_matvec` / `nu_matmul` even when a
    /// specialized kernel applies. `metal-check` compares both paths;
    /// production leaves it off.
    generic_only: bool = false,
    /// Counts at or below this take the register-reuse chunk body on the
    /// 256-wide value geometry it was measured on: 11–16 % faster on the
    /// verify-shaped batches (1–8 rows, which the models send to
    /// `attentionVerify` instead) and 4–13 % at 64 rows, while the
    /// row-split body keeps the 256-row prefill chunks (0–5 % apart, either
    /// way). Zero forces the row-split body everywhere; other widths keep it
    /// because the column split leaves SIMD groups idle below 256.
    attention_reuse_max_rows: usize = 64,
    /// Whether the post-norm and q/k-norm fusions dispatch (`rmsNormAdd`,
    /// `addRmsNorm`, `rmsNormRope`). False makes each helper run the pair it
    /// replaces, so a benchmark can take the interleaved unfused control.
    fused_norms: bool = true,
    /// Present after `enableProfiling`; read it after `commit()`.
    profile: ?Profile = null,
    /// Bytes of the buffers `create` made and `release` has not freed: the
    /// device memory this backend owns, beside the wrapped (borrowed) ranges.
    created_bytes: u64 = 0,

    pub fn init(alloc: std.mem.Allocator, diagnostic: []u8) !Backend {
        if (!enabled) return error.MetalNotEnabled;
        const handle = nu_metal_create(source.ptr, source.len, diagnostic.ptr, diagnostic.len) orelse return error.MetalInitializationFailed;
        var self: Backend = .{ .alloc = alloc, .handle = handle, .pipelines = undefined };
        errdefer nu_metal_destroy(handle);
        for (kernel_names, &self.pipelines) |name, *id| {
            if (nu_metal_pipeline(handle, name.ptr, id, diagnostic.ptr, diagnostic.len) != 0) return error.MetalInitializationFailed;
        }
        return self;
    }
    pub fn deinit(self: *Backend) void {
        if (enabled) nu_metal_destroy(self.handle);
        if (self.profile) |*p| p.deinit(self.alloc);
        self.wrapped.deinit(self.alloc);
        self.* = undefined;
    }
    pub fn gpuSeconds(self: *const Backend) f64 {
        if (!enabled) return 0;
        return nu_metal_gpu_seconds(self.handle);
    }
    /// Times every later dispatch; at most `max_dispatches` per command buffer
    /// are sampled (a Qwen token records about 1,240). Once, while not
    /// recording. `diagnostic` receives the device's reason on failure.
    pub fn enableProfiling(self: *Backend, max_dispatches: u32, diagnostic: []u8) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (self.recording or self.profile != null or max_dispatches == 0) return error.MetalAlreadyRecording;
        const durations = try self.alloc.alloc(f64, max_dispatches);
        errdefer self.alloc.free(durations);
        if (nu_metal_profile_enable(self.handle, max_dispatches, diagnostic.ptr, diagnostic.len) != 0) return error.MetalProfilingUnavailable;
        self.profile = .{ .durations = durations };
    }
    /// Records the next submitted command buffer into a new `.gputrace`
    /// document at `path` for Xcode's Metal debugger (counters, limiters,
    /// occupancy). Needs `MTL_CAPTURE_ENABLED=1`; the reason for a refusal is
    /// written to `diagnostic`. Idle only.
    pub fn captureNext(self: *Backend, path: [:0]const u8, diagnostic: []u8) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (self.recording) return error.MetalAlreadyRecording;
        if (nu_metal_capture_next(self.handle, path.ptr, diagnostic.ptr, diagnostic.len) != 0) return error.MetalCaptureUnavailable;
    }
    /// Pairs the bridge's per-dispatch durations with the recorded kernels.
    /// Runs after every commit while profiling, including failed ones, so the
    /// pending list never leaks into the next command buffer.
    fn account(self: *Backend, gpu_before: f64, completed: bool) !void {
        const p = &self.profile.?;
        defer p.pending.clearRetainingCapacity();
        if (!completed or p.paused or p.pending.items.len == 0) return;
        p.command_buffers += 1;
        p.gpu_seconds += nu_metal_gpu_seconds(self.handle) - gpu_before;
        const timed = nu_metal_profile_read(self.handle, p.durations.ptr, @intCast(p.durations.len));
        for (p.pending.items, 0..) |item, i| {
            if (i >= timed or p.durations[i] < 0) {
                p.unsampled += 1;
                continue;
            }
            const entry = try p.totals.getOrPutValue(self.alloc, item.key, .{});
            entry.value_ptr.dispatches += 1;
            entry.value_ptr.seconds += p.durations[i];
            entry.value_ptr.bytes += item.bytes;
        }
    }
    pub const pipeline_count = kernel_names.len;
    /// The compiled pipeline `index` (0 ..< `pipeline_count`, the order the
    /// kernels are compiled in) as the Metal compiler sized it.
    pub fn pipelineStats(self: *const Backend, index: usize) PipelineStats {
        if (!enabled) unreachable; // no Backend exists without Metal
        var out: [3]u32 = undefined;
        if (nu_metal_pipeline_stats(self.handle, self.pipelines[index], &out) != 0) unreachable;
        return .{ .name = kernel_names[index], .max_threads = out[0], .execution_width = out[1], .threadgroup_memory = out[2] };
    }
    pub fn maxBufferLength(self: *const Backend) usize {
        if (!enabled) return 0;
        return nu_metal_max_buffer_length(self.handle);
    }

    /// Zero-filled GPU buffer owned by the backend.
    pub fn create(self: *Backend, len: usize) !Buffer {
        if (!enabled) return error.MetalNotEnabled;
        if (len == 0) return error.InvalidShape;
        var id: u32 = undefined;
        if (nu_metal_buffer_create(self.handle, len, &id) != 0) return error.MetalBufferFailed;
        const host = nu_metal_buffer_contents(self.handle, id) orelse return error.MetalBufferFailed;
        self.created_bytes += len;
        return .{ .id = id, .offset = 0, .len = len, .host = host };
    }
    /// Frees a buffer `create` returned (whole, not a slice) while nothing is
    /// recording; every slice of it becomes invalid.
    pub fn release(self: *Backend, buffer: Buffer) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (self.recording or buffer.offset != 0) return error.InvalidShape;
        if (nu_metal_buffer_release(self.handle, buffer.id) != 0) return error.MetalBufferFailed;
        self.created_bytes -|= buffer.len;
    }
    /// Wraps borrowed page-backed memory; repeated wraps of the same start
    /// address reuse the buffer and must request the same length.
    pub fn wrap(self: *Backend, bytes: []const u8) !Buffer {
        if (!enabled) return error.MetalNotEnabled;
        if (bytes.len == 0) return error.InvalidShape;
        const entry = try self.wrapped.getOrPut(self.alloc, @intFromPtr(bytes.ptr));
        if (!entry.found_existing) {
            errdefer _ = self.wrapped.remove(@intFromPtr(bytes.ptr));
            var id: u32 = undefined;
            var offset: usize = undefined;
            if (nu_metal_buffer_wrap(self.handle, bytes.ptr, bytes.len, &id, &offset) != 0) return error.MetalBufferFailed;
            entry.value_ptr.* = .{ .id = id, .offset = offset, .len = bytes.len, .host = @constCast(bytes.ptr) };
        }
        if (entry.value_ptr.len != bytes.len) return error.InvalidShape;
        return entry.value_ptr.*;
    }
    /// Forgets a wrapped range so its memory can be freed and later wrapped
    /// again at another length (a plan's session on `deinit`). Only
    /// valid after the last `commit` that used it, which the synchronous
    /// backend guarantees. The bridge keeps its `MTLBuffer` object (a no-copy
    /// view that owns nothing) until the backend is destroyed.
    pub fn unwrap(self: *Backend, bytes: []const u8) void {
        _ = self.wrapped.remove(@intFromPtr(bytes.ptr));
    }

    pub fn begin(self: *Backend) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (self.recording) return error.MetalAlreadyRecording;
        if (nu_metal_begin(self.handle) != 0) return error.MetalExecutionFailed;
        self.recording = true;
    }
    /// Submits and waits. After it returns, all recorded writes are visible.
    pub fn commit(self: *Backend) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (!self.recording) return error.MetalNotRecording;
        self.recording = false;
        const gpu_before = nu_metal_gpu_seconds(self.handle);
        const completed = nu_metal_commit(self.handle, if (self.tick != null) tickTrampoline else null, self) == 0;
        if (self.profile != null) try self.account(gpu_before, completed);
        if (!completed) return error.MetalExecutionFailed;
    }

    /// Drops the open command buffer unsubmitted, so none of its dispatches
    /// run; a no-op when idle. The error-path counterpart of `commit`: model
    /// plans `errdefer` it after `begin`, and the backend can `begin` again.
    pub fn discard(self: *Backend) void {
        if (!enabled or !self.recording) return;
        self.recording = false;
        _ = nu_metal_discard(self.handle);
        if (self.profile) |*p| p.pending.clearRetainingCapacity();
    }

    fn tickTrampoline(context: ?*anyopaque) callconv(.c) void {
        const self: *Backend = @ptrCast(@alignCast(context.?));
        if (self.tick) |t| t.call(t.context);
    }

    /// Records one dispatch of `kernel` over `groups` threadgroups of `threads`.
    /// Public for the encoders under `ops/`, not for model plans.
    pub fn dispatch(self: *Backend, kernel: Kernel, buffers: []const Buffer, params: anytype, groups: u32, threads: u32, shape: Shape) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (!self.recording) return error.MetalNotRecording;
        var bindings: [7]Binding = undefined;
        if (buffers.len > bindings.len) return error.InvalidShape;
        for (buffers, bindings[0..buffers.len]) |b, *out| out.* = b.binding();
        if (nu_metal_dispatch(self.handle, self.pipelines[@backingInt(kernel)], &bindings, @intCast(buffers.len), @ptrCast(&params), @sizeOf(@TypeOf(params)), groups, 1, threads, 1) != 0) return error.MetalExecutionFailed;
        // Recorded only after the bridge accepted the dispatch, so pending[i]
        // is the i-th dispatch the GPU will time.
        if (self.profile) |*p| try p.pending.append(self.alloc, .{ .key = .{ .kernel = kernel, .encoding = shape.encoding, .rows = shape.rows, .columns = shape.columns }, .bytes = shape.bytes });
    }
    /// `dispatch` over a two-dimensional grid of threadgroups; for `ops/` too.
    pub fn dispatch2(self: *Backend, kernel: Kernel, buffers: []const Buffer, params: anytype, groups_x: u32, groups_y: u32, threads: u32) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (!self.recording) return error.MetalNotRecording;
        var bindings: [7]Binding = undefined;
        if (buffers.len > bindings.len) return error.InvalidShape;
        for (buffers, bindings[0..buffers.len]) |b, *out| out.* = b.binding();
        if (nu_metal_dispatch(self.handle, self.pipelines[@backingInt(kernel)], &bindings, @intCast(buffers.len), @ptrCast(&params), @sizeOf(@TypeOf(params)), groups_x, groups_y, threads, 1) != 0) return error.MetalExecutionFailed;
        if (self.profile) |*p| try p.pending.append(self.alloc, .{ .key = .{ .kernel = kernel, .encoding = null, .rows = 0, .columns = 0 }, .bytes = 0 });
    }
    /// Threadgroups of 256 covering `count` elements.
    pub fn perElement(count: usize) u32 {
        return @intCast((count + 255) / 256);
    }

    // ----- Kernel encoders, one file per kind of operation under `ops/`. -----
    const matvec_ops = @import("ops/matvec.zig");
    const matmul_ops = @import("ops/matmul.zig");
    const experts_ops = @import("ops/experts.zig");
    const rowwise_ops = @import("ops/rowwise.zig");
    const encoders_ops = @import("ops/encoders.zig");
    const audio_ops = @import("ops/audio.zig");
    const asr_ops = @import("ops/asr.zig");
    const recurrent_ops = @import("ops/recurrent.zig");
    const attention_ops = @import("ops/attention.zig");
    const sampling_ops = @import("ops/sampling.zig");
    const tts_ops = @import("ops/tts.zig");
    pub const MatvecParams = matvec_ops.MatvecParams;
    pub const MatvecBlockParams = matvec_ops.MatvecBlockParams;
    pub const rows_per_simdgroup = matvec_ops.rows_per_simdgroup;
    pub const matvec = matvec_ops.matvec;
    pub const matvecSplits = matvec_ops.matvecSplits;
    pub const matvecRowsPerSimdgroup = matvec_ops.matvecRowsPerSimdgroup;
    pub const matvecSimdgroups = matvec_ops.matvecSimdgroups;
    pub const split_k_max = matvec_ops.split_k_max;
    pub const split_k_rows = matvec_ops.split_k_rows;
    pub const matvecSplitKernel = matvec_ops.matvecSplitKernel;
    pub const MatvecSplitParams = matvec_ops.MatvecSplitParams;
    pub const ReduceSplitsParams = matvec_ops.ReduceSplitsParams;
    pub const MatvecRowsParams = matvec_ops.MatvecRowsParams;
    pub const small_batch_rows = matvec_ops.small_batch_rows;
    pub const route_small_batch = matvec_ops.route_small_batch;
    pub const matvec_rows_max = matvec_ops.matvec_rows_max;
    pub const matvec_rows_per_simdgroup = matvec_ops.matvec_rows_per_simdgroup;
    pub const usesMatvecRows = matvec_ops.usesMatvecRows;
    pub const matvecRows = matvec_ops.matvecRows;
    pub const specializedMatvecRows = matvec_ops.specializedMatvecRows;
    pub const specializedMatvec = matvec_ops.specializedMatvec;
    pub const Segment = matvec_ops.Segment;
    pub const SegmentMode = matvec_ops.SegmentMode;
    pub const matvecSegments = matvec_ops.matvecSegments;
    pub const mergeDeclined = matvec_ops.mergeDeclined;
    pub const matvecSegmentsSplits = matvec_ops.matvecSegmentsSplits;
    pub const MatmulParams = matmul_ops.MatmulParams;
    pub const matmul_tile = matmul_ops.matmul_tile;
    pub const small_chunk_tokens = matmul_ops.small_chunk_tokens;
    pub const small_matrix_rows = matmul_ops.small_matrix_rows;
    pub const matmulPadded = matmul_ops.matmulPadded;
    pub const MatmulGeometry = matmul_ops.MatmulGeometry;
    pub const matmulGeometry = matmul_ops.matmulGeometry;
    pub const matmul = matmul_ops.matmul;
    pub const matmulTile = matmul_ops.matmulTile;
    pub const matmulTile32 = matmul_ops.matmulTile32;
    pub const matmulKernel = matmul_ops.matmulKernel;
    pub const specializedMatmulWide = matmul_ops.specializedMatmulWide;
    pub const specializedTile = matmul_ops.specializedTile;
    pub const specializedMatmulFrag = matmul_ops.specializedMatmulFrag;
    pub const fragmentLayout = matmul_ops.fragmentLayout;
    pub const specializedMatmul = matmul_ops.specializedMatmul;
    pub const RouteParams = experts_ops.RouteParams;
    pub const route_max_experts = experts_ops.route_max_experts;
    pub const route_max_k = experts_ops.route_max_k;
    pub const route = experts_ops.route;
    pub const ExpertMatvecParams = experts_ops.ExpertMatvecParams;
    pub const matvecExperts = experts_ops.matvecExperts;
    pub const CombineParams = experts_ops.CombineParams;
    pub const CombineShape = experts_ops.CombineShape;
    pub const combineExperts = experts_ops.combineExperts;
    pub const GeluRowsParams = experts_ops.GeluRowsParams;
    pub const geluMulRows = experts_ops.geluMulRows;
    pub const ExpertListsParams = experts_ops.ExpertListsParams;
    pub const expert_tile = experts_ops.expert_tile;
    pub const expertTileBound = experts_ops.expertTileBound;
    pub const ExpertListsLayout = experts_ops.ExpertListsLayout;
    pub const expertListsLayout = experts_ops.expertListsLayout;
    pub const expertLists = experts_ops.expertLists;
    pub const MatmulExpertsParams = experts_ops.MatmulExpertsParams;
    pub const matmulExperts = experts_ops.matmulExperts;
    pub const EmbedParams = rowwise_ops.EmbedParams;
    pub const embed = rowwise_ops.embed;
    pub const NormParams = rowwise_ops.NormParams;
    pub const Norm = rowwise_ops.Norm;
    pub const rmsNorm = rowwise_ops.rmsNorm;
    pub const NormAddParams = rowwise_ops.NormAddParams;
    pub const rmsNormAdd = rowwise_ops.rmsNormAdd;
    pub const AddNormParams = rowwise_ops.AddNormParams;
    pub const addRmsNorm = rowwise_ops.addRmsNorm;
    pub const NormRopeParams = rowwise_ops.NormRopeParams;
    pub const rmsNormRope = rowwise_ops.rmsNormRope;
    pub const HadamardParams = rowwise_ops.HadamardParams;
    pub const hadamard_block = rowwise_ops.hadamard_block;
    pub const hadamard = rowwise_ops.hadamard;
    pub const GatherParams = rowwise_ops.GatherParams;
    pub const gatherRows = rowwise_ops.gatherRows;
    pub const L2Params = rowwise_ops.L2Params;
    pub const l2Norm = rowwise_ops.l2Norm;
    pub const l2NormRows = rowwise_ops.l2NormRows;
    pub const Pairing = rowwise_ops.Pairing;
    pub const RopeParams = rowwise_ops.RopeParams;
    pub const rope = rowwise_ops.rope;
    pub const RopeRowsParams = rowwise_ops.RopeRowsParams;
    pub const ropeRows = rowwise_ops.ropeRows;
    pub const ropeTable = rowwise_ops.ropeTable;
    pub const CountParams = rowwise_ops.CountParams;
    pub const add = rowwise_ops.add;
    pub const copy = rowwise_ops.copy;
    pub const siluMul = rowwise_ops.siluMul;
    pub const DflashConvParams = rowwise_ops.DflashConvParams;
    pub const dflashConv = rowwise_ops.dflashConv;
    pub const silu = rowwise_ops.silu;
    pub const geluMul = rowwise_ops.geluMul;
    pub const geluQuickMul = rowwise_ops.geluQuickMul;
    pub const geluErf = rowwise_ops.geluErf;
    pub const geluErfMulRows = rowwise_ops.geluErfMulRows;
    pub const ScaleParams = rowwise_ops.ScaleParams;
    pub const scale = rowwise_ops.scale;
    pub const addScale = rowwise_ops.addScale;
    pub const softcap = rowwise_ops.softcap;
    pub const ClampParams = rowwise_ops.ClampParams;
    pub const clamp = rowwise_ops.clamp;
    pub const DeltaGatesParams = rowwise_ops.DeltaGatesParams;
    pub const deltaGates = rowwise_ops.deltaGates;
    pub const deltaGatesRows = rowwise_ops.deltaGatesRows;
    pub const GateParams = rowwise_ops.GateParams;
    pub const sigmoidGate = rowwise_ops.sigmoidGate;
    pub const LayerNormParams = encoders_ops.LayerNormParams;
    pub const layerNorm = encoders_ops.layerNorm;
    pub const BiasRowsParams = encoders_ops.BiasRowsParams;
    pub const addBiasRows = encoders_ops.addBiasRows;
    pub const gelu = encoders_ops.gelu;
    pub const AttentionFullParams = encoders_ops.AttentionFullParams;
    pub const AttentionFullShape = encoders_ops.AttentionFullShape;
    pub const attention_full_max_rows = encoders_ops.attention_full_max_rows;
    pub const attentionFull = encoders_ops.attentionFull;
    pub const AttentionSegmentsParams = encoders_ops.AttentionSegmentsParams;
    pub const AttentionSegmentsShape = encoders_ops.AttentionSegmentsShape;
    pub const attentionSegments = encoders_ops.attentionSegments;
    pub const AttentionGemmParams = encoders_ops.AttentionGemmParams;
    pub const AttentionSegmentsGroupedShape = encoders_ops.AttentionSegmentsGroupedShape;
    pub const AttentionTile = encoders_ops.AttentionTile;
    pub const attention_tile_rows = encoders_ops.attention_tile_rows;
    pub const AttentionChunk = encoders_ops.AttentionChunk;
    pub const attentionTiles = encoders_ops.attentionTiles;
    pub const chunkTiles = encoders_ops.chunkTiles;
    pub const attentionSegmentsGrouped = encoders_ops.attentionSegmentsGrouped;
    pub const AudioAttentionParams = audio_ops.AudioAttentionParams;
    pub const audioAttention = audio_ops.audioAttention;
    pub const GluConvParams = audio_ops.GluConvParams;
    pub const gluConv = audio_ops.gluConv;
    pub const AsrConvParams = asr_ops.AsrConvParams;
    pub const AsrConv = asr_ops.AsrConv;
    pub const asrConv2d = asr_ops.asrConv2d;
    pub const AsrRowsParams = asr_ops.AsrRowsParams;
    pub const asrBiasRelu = asr_ops.asrBiasRelu;
    pub const AsrShiftParams = asr_ops.AsrShiftParams;
    pub const asrShift = asr_ops.asrShift;
    pub const AsrAttentionParams = asr_ops.AsrAttentionParams;
    pub const AsrAttention = asr_ops.AsrAttention;
    pub const asrAttention = asr_ops.asrAttention;
    pub const asrGlu = asr_ops.asrGlu;
    pub const AsrDepthwiseParams = asr_ops.AsrDepthwiseParams;
    pub const asrDepthwise = asr_ops.asrDepthwise;
    pub const asrJoint = asr_ops.asrJoint;
    pub const AsrScaleParams = asr_ops.AsrScaleParams;
    pub const asrAddScaled = asr_ops.asrAddScaled;
    pub const AsrRowsMatmulParams = asr_ops.AsrRowsMatmulParams;
    pub const AsrEpilogue = asr_ops.AsrEpilogue;
    pub const asr_matmul_rows_max = asr_ops.asr_matmul_rows_max;
    pub const AsrProduct = asr_ops.AsrProduct;
    pub const asrMatmulRows = asr_ops.asrMatmulRows;
    pub const DeltaParams = recurrent_ops.DeltaParams;
    pub const DeltaShape = recurrent_ops.DeltaShape;
    pub const delta = recurrent_ops.delta;
    pub const DeltaChunkParams = recurrent_ops.DeltaChunkParams;
    pub const DeltaChunkShape = recurrent_ops.DeltaChunkShape;
    pub const deltaChunkRows = recurrent_ops.deltaChunkRows;
    pub const deltaChunk = recurrent_ops.deltaChunk;
    pub const DeltaRowsShape = recurrent_ops.DeltaRowsShape;
    pub const DeltaRowsParams = recurrent_ops.DeltaRowsParams;
    pub const DeltaReplayParams = recurrent_ops.DeltaReplayParams;
    pub const deltaTapeFloats = recurrent_ops.deltaTapeFloats;
    pub const deltaRows = recurrent_ops.deltaRows;
    pub const deltaReplay = recurrent_ops.deltaReplay;
    pub const ConvParams = recurrent_ops.ConvParams;
    pub const convolution = recurrent_ops.convolution;
    pub const ConvRowsParams = recurrent_ops.ConvRowsParams;
    pub const convolutionRows = recurrent_ops.convolutionRows;
    pub const convolutionHistory = recurrent_ops.convolutionHistory;
    pub const AttentionParams = attention_ops.AttentionParams;
    pub const AttentionShape = attention_ops.AttentionShape;
    pub const attention = attention_ops.attention;
    pub const AttentionDecodeParams = attention_ops.AttentionDecodeParams;
    pub const decode_splits_max = attention_ops.decode_splits_max;
    pub const decode_split_rows = attention_ops.decode_split_rows;
    pub const attentionDecodeSplits = attention_ops.attentionDecodeSplits;
    pub const attentionDecodePartials = attention_ops.attentionDecodePartials;
    pub const attentionDecodeHeadGroups = attention_ops.attentionDecodeHeadGroups;
    pub const attentionDecode = attention_ops.attentionDecode;
    pub const verify_max_rows = attention_ops.verify_max_rows;
    pub const attentionVerifyPartials = attention_ops.attentionVerifyPartials;
    pub const attentionVerify = attention_ops.attentionVerify;
    pub const PackParams = attention_ops.PackParams;
    pub const Pack = attention_ops.Pack;
    pub const packHalf = attention_ops.packHalf;
    pub const packQ8 = attention_ops.packQ8;
    pub const unpackQ8 = attention_ops.unpackQ8;
    pub const storeRows = attention_ops.storeRows;
    pub const AttentionChunkParams = attention_ops.AttentionChunkParams;
    pub const Span = attention_ops.Span;
    pub const AttentionChunkShape = attention_ops.AttentionChunkShape;
    pub const max_chunk_rows = attention_ops.max_chunk_rows;
    pub const attentionChunkRows = attention_ops.attentionChunkRows;
    pub const attentionChunk = attention_ops.attentionChunk;
    pub const attentionChunkReuse = attention_ops.attentionChunkReuse;
    pub const TopKParams = sampling_ops.TopKParams;
    pub const topk_partials = sampling_ops.topk_partials;
    pub const topk_max_count = sampling_ops.topk_max_count;
    pub const TopKBuffers = sampling_ops.TopKBuffers;
    pub const topkBuffers = sampling_ops.topkBuffers;
    pub const topk = sampling_ops.topk;
    pub const PenalizeParams = sampling_ops.PenalizeParams;
    pub const penalize = sampling_ops.penalize;
    pub const ArgmaxParams = sampling_ops.ArgmaxParams;
    pub const argmax_partials = sampling_ops.argmax_partials;
    pub const argmax = sampling_ops.argmax;
    pub const TtsCodebookParams = tts_ops.TtsCodebookParams;
    pub const ttsCodebookSum = tts_ops.ttsCodebookSum;
    pub const TtsGatherParams = tts_ops.TtsGatherParams;
    pub const ttsGather = tts_ops.ttsGather;
    pub const TtsDepthwiseParams = tts_ops.TtsDepthwiseParams;
    pub const ttsDepthwise = tts_ops.ttsDepthwise;
    pub const TtsSnakeParams = tts_ops.TtsSnakeParams;
    pub const ttsSnake = tts_ops.ttsSnake;
    pub const TtsScaledAddParams = tts_ops.TtsScaledAddParams;
    pub const ttsScaledAdd = tts_ops.ttsScaledAdd;
    pub const TtsWindowParams = tts_ops.TtsWindowParams;
    pub const tts_window_max = tts_ops.tts_window_max;
    pub const ttsWindowAttention = tts_ops.ttsWindowAttention;
    pub const TtsConvOutParams = tts_ops.TtsConvOutParams;
    pub const ttsConvOut = tts_ops.ttsConvOut;
    pub const TtsMatvecParams = tts_ops.TtsMatvecParams;
    pub const ttsMatvecBf16 = tts_ops.ttsMatvecBf16;
    pub const ttsMatmulBf16 = tts_ops.ttsMatmulBf16;
    pub const TtsEmbedRowParams = tts_ops.TtsEmbedRowParams;
    pub const ttsEmbedRow = tts_ops.ttsEmbedRow;
    pub const TtsSampleParams = tts_ops.TtsSampleParams;
    pub const tts_sample_max = tts_ops.tts_sample_max;
    pub const ttsSample = tts_ops.ttsSample;
    pub const TtsMatvec3Params = tts_ops.TtsMatvec3Params;
    pub const ttsMatvec3Bf16 = tts_ops.ttsMatvec3Bf16;
    pub const TtsGateUpParams = tts_ops.TtsGateUpParams;
    pub const ttsGateUpBf16 = tts_ops.ttsGateUpBf16;
};
