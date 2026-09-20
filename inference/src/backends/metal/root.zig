//! Metal backend: a thin Zig layer over the recording bridge. `Backend` owns
//! the device handle and compiled pipelines, hands out `Buffer` handles, and
//! records typed kernel dispatches between `begin()` and `commit()`. It knows
//! kernel contracts and shapes, not model semantics: layer schedules belong to
//! model adapters that call these encoders.
//!
//! Ownership: created buffers live until `deinit`. Wrapped buffers borrow the
//! caller's page-backed memory, which must outlive the backend. `commit()`
//! waits for completion, so CPU access to any buffer after it is race-free.
//!
//! Profiling (`enableProfiling`) times every dispatch with GPU timestamps and
//! accumulates them per kernel and shape in `Profile`. It is a diagnostic
//! mode: each dispatch then runs in its own encoder, which costs GPU time that
//! the report exposes rather than hides.
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
extern fn nu_metal_max_buffer_length(*anyopaque) usize;
extern fn nu_metal_buffer_create(*anyopaque, usize, *u32) c_int;
extern fn nu_metal_buffer_wrap(*anyopaque, [*]const u8, usize, *u32, *usize) c_int;
extern fn nu_metal_buffer_contents(*anyopaque, u32) ?[*]u8;
extern fn nu_metal_begin(*anyopaque) c_int;
extern fn nu_metal_dispatch(*anyopaque, u32, [*]const Binding, u32, ?*const anyopaque, usize, u32, u32, u32, u32) c_int;
extern fn nu_metal_commit(*anyopaque) c_int;
extern fn nu_metal_gpu_seconds(*anyopaque) f64;
extern fn nu_metal_profile_enable(*anyopaque, u32, [*]u8, usize) c_int;
extern fn nu_metal_profile_read(*anyopaque, [*]f64, u32) u32;

/// The IQ3_S codebook is emitted into the shader source at compile time from
/// the single Zig definition, so the CPU and GPU decoders share one table.
const iq3_grid_source = blk: {
    @setEvalBranchQuota(2_000_000);
    const grid = @import("../../quant/iq3-grid.zig").values;
    var text: []const u8 = "constant uint nu_iq3_grid[512] = {\n";
    for (grid, 0..) |value, i| {
        text = text ++ std.fmt.comptimePrint("{d}u,", .{value});
        if (i % 16 == 15) text = text ++ "\n";
    }
    break :blk text ++ "};\n";
};
const source = "#include <metal_stdlib>\nusing namespace metal;\n" ++ iq3_grid_source ++ @embedFile("dequant.metal") ++ "\n" ++ @embedFile("kernels.metal");

const kernel_names = [_][:0]const u8{
    "nu_matvec",                "nu_matvec_q4_k",           "nu_matvec_q5_k",           "nu_matvec_q6_k",           "nu_matvec_iq4_xs",         "nu_embed",
    "nu_rmsnorm",               "nu_l2norm",                "nu_rope",                  "nu_add",                   "nu_silu_mul",              "nu_silu_inplace",
    "nu_delta_gates",           "nu_sigmoid_gate",          "nu_delta",                 "nu_convolution",           "nu_attention_scores",      "nu_attention_softmax",
    "nu_attention_values",      "nu_penalize",              "nu_argmax_partial",        "nu_argmax_final",          "nu_matvec_q3_k",           "nu_matvec_iq3_s",
    "nu_matvec_segments",       "nu_topk_partial",          "nu_topk_final",            "nu_expsum_partial",        "nu_matmul",                "nu_rope_rows",
    "nu_copy",                  "nu_convolution_rows",      "nu_convolution_history",   "nu_attention_chunk",       "nu_delta_chunk",           "nu_matmul_q3_k",
    "nu_matmul_q4_k",           "nu_matmul_q5_k",           "nu_matmul_q6_k",           "nu_matmul_iq3_s",          "nu_matmul_iq4_xs",         "nu_matmul_q3_k_32",
    "nu_matmul_q4_k_32",        "nu_matmul_q5_k_32",        "nu_matmul_q6_k_32",        "nu_matmul_iq3_s_32",       "nu_matmul_iq4_xs_32",      "nu_attention_scores_h",
    "nu_attention_values_h",    "nu_attention_chunk_h",     "nu_pack_half",             "nu_attention_decode",      "nu_attention_decode_h",    "nu_attention_merge",
    "nu_gelu_mul",              "nu_scale",                 "nu_add_scale",             "nu_softcap",               "nu_attention_decode_w",    "nu_attention_decode_wh",
    "nu_matvec_q4_0",           "nu_matmul_q4_0",           "nu_matmul_q4_0_32",        "nu_matvec_experts",        "nu_route",                 "nu_combine_experts",
    "nu_gelu_mul_rows",         "nu_expert_lists",          "nu_matmul_experts",        "nu_matmul_experts_q4_0",   "nu_matvec_pq2_0",          "nu_matvec_ptq1_0",
    "nu_matmul_pq2_0",          "nu_matmul_ptq1_0",         "nu_matmul_pq2_0_32",       "nu_matmul_ptq1_0_32",      "nu_hadamard",              "nu_gather_rows",
    "nu_matmul_q3_k_8",         "nu_matmul_q4_k_8",         "nu_matmul_q5_k_8",         "nu_matmul_q6_k_8",         "nu_matmul_iq3_s_8",        "nu_matmul_iq4_xs_8",
    "nu_matmul_q4_0_8",         "nu_matmul_pq2_0_8",        "nu_matmul_ptq1_0_8",       "nu_matvec_rows",           "nu_matvec_rows_q4_k_t2",   "nu_matvec_rows_q4_k_t3",
    "nu_matvec_rows_q4_k_t4",   "nu_matvec_rows_q4_k_t5",   "nu_matvec_rows_q4_k_t6",   "nu_matvec_rows_q4_k_t7",   "nu_matvec_rows_q4_k_t8",   "nu_matvec_rows_q5_k_t2",
    "nu_matvec_rows_q5_k_t3",   "nu_matvec_rows_q5_k_t4",   "nu_matvec_rows_q5_k_t5",   "nu_matvec_rows_q5_k_t6",   "nu_matvec_rows_q5_k_t7",   "nu_matvec_rows_q5_k_t8",
    "nu_matvec_rows_q6_k_t2",   "nu_matvec_rows_q6_k_t3",   "nu_matvec_rows_q6_k_t4",   "nu_matvec_rows_q6_k_t5",   "nu_matvec_rows_q6_k_t6",   "nu_matvec_rows_q6_k_t7",
    "nu_matvec_rows_q6_k_t8",   "nu_matvec_rows_iq4_xs_t2", "nu_matvec_rows_iq4_xs_t3", "nu_matvec_rows_iq4_xs_t4", "nu_matvec_rows_iq4_xs_t5", "nu_matvec_rows_iq4_xs_t6",
    "nu_matvec_rows_iq4_xs_t7", "nu_matvec_rows_iq4_xs_t8", "nu_matmul_q3_k_w8",        "nu_matmul_q4_k_w8",        "nu_matmul_q5_k_w8",        "nu_matmul_q6_k_w8",
    "nu_matmul_iq3_s_w8",       "nu_matmul_iq4_xs_w8",      "nu_matmul_q4_0_w8",        "nu_matmul_pq2_0_w8",       "nu_matmul_ptq1_0_w8",
};
pub const Kernel = enum(u32) {
    matvec,
    matvec_q4_k,
    matvec_q5_k,
    matvec_q6_k,
    matvec_iq4_xs,
    embed,
    rmsnorm,
    l2norm,
    rope,
    add,
    silu_mul,
    silu_inplace,
    delta_gates,
    sigmoid_gate,
    delta,
    convolution,
    attention_scores,
    attention_softmax,
    attention_values,
    penalize,
    argmax_partial,
    argmax_final,
    matvec_q3_k,
    matvec_iq3_s,
    matvec_segments,
    topk_partial,
    topk_final,
    expsum_partial,
    matmul,
    rope_rows,
    copy,
    convolution_rows,
    convolution_history,
    attention_chunk,
    delta_chunk,
    matmul_q3_k,
    matmul_q4_k,
    matmul_q5_k,
    matmul_q6_k,
    matmul_iq3_s,
    matmul_iq4_xs,
    matmul_q3_k_32,
    matmul_q4_k_32,
    matmul_q5_k_32,
    matmul_q6_k_32,
    matmul_iq3_s_32,
    matmul_iq4_xs_32,
    attention_scores_h,
    attention_values_h,
    attention_chunk_h,
    pack_half,
    attention_decode,
    attention_decode_h,
    attention_merge,
    gelu_mul,
    scale,
    add_scale,
    softcap,
    attention_decode_w,
    attention_decode_wh,
    matvec_q4_0,
    matmul_q4_0,
    matmul_q4_0_32,
    matvec_experts,
    route,
    combine_experts,
    gelu_mul_rows,
    expert_lists,
    matmul_experts,
    matmul_experts_q4_0,
    matvec_pq2_0,
    matvec_ptq1_0,
    matmul_pq2_0,
    matmul_ptq1_0,
    matmul_pq2_0_32,
    matmul_ptq1_0_32,
    hadamard,
    gather_rows,
    matmul_q3_k_8,
    matmul_q4_k_8,
    matmul_q5_k_8,
    matmul_q6_k_8,
    matmul_iq3_s_8,
    matmul_iq4_xs_8,
    matmul_q4_0_8,
    matmul_pq2_0_8,
    matmul_ptq1_0_8,
    matvec_rows,
    matvec_rows_q4_k_t2,
    matvec_rows_q4_k_t3,
    matvec_rows_q4_k_t4,
    matvec_rows_q4_k_t5,
    matvec_rows_q4_k_t6,
    matvec_rows_q4_k_t7,
    matvec_rows_q4_k_t8,
    matvec_rows_q5_k_t2,
    matvec_rows_q5_k_t3,
    matvec_rows_q5_k_t4,
    matvec_rows_q5_k_t5,
    matvec_rows_q5_k_t6,
    matvec_rows_q5_k_t7,
    matvec_rows_q5_k_t8,
    matvec_rows_q6_k_t2,
    matvec_rows_q6_k_t3,
    matvec_rows_q6_k_t4,
    matvec_rows_q6_k_t5,
    matvec_rows_q6_k_t6,
    matvec_rows_q6_k_t7,
    matvec_rows_q6_k_t8,
    matvec_rows_iq4_xs_t2,
    matvec_rows_iq4_xs_t3,
    matvec_rows_iq4_xs_t4,
    matvec_rows_iq4_xs_t5,
    matvec_rows_iq4_xs_t6,
    matvec_rows_iq4_xs_t7,
    matvec_rows_iq4_xs_t8,
    matmul_q3_k_w8,
    matmul_q4_k_w8,
    matmul_q5_k_w8,
    matmul_q6_k_w8,
    matmul_iq3_s_w8,
    matmul_iq4_xs_w8,
    matmul_q4_0_w8,
    matmul_pq2_0_w8,
    matmul_ptq1_0_w8,
};

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
};

/// Per-dispatch GPU time accumulated by kernel and shape. Only dispatches the
/// GPU actually timestamped are added to `totals`; the rest are counted in
/// `unsampled` so a report can never attribute time it did not measure.
pub const Profile = struct {
    /// Matrix kernels carry their encoding and dimensions so one kernel's time
    /// splits by tensor shape (a 48-row projection is not a 17,408-row one);
    /// other kernels aggregate by name with `encoding = null` and zero dims.
    pub const Key = struct { kernel: Kernel, encoding: ?u32, rows: u32, columns: u32 };
    pub const Total = struct { dispatches: u64 = 0, seconds: f64 = 0, bytes: u64 = 0 };
    const Pending = struct { key: Key, bytes: u64 };

    totals: std.AutoHashMapUnmanaged(Key, Total) = .empty,
    /// Kernels recorded into the open command buffer, in dispatch order; paired
    /// with the bridge's durations at `commit`.
    pending: std.ArrayListUnmanaged(Pending) = .empty,
    durations: []f64 = &.{},
    /// Committed command buffers that contained at least one dispatch.
    command_buffers: u64 = 0,
    unsampled: u64 = 0,
    /// Whole-command-buffer GPU time while profiling. Exceeds the sum of the
    /// totals by the cost of the per-dispatch encoder boundaries.
    gpu_seconds: f64 = 0,

    pub fn deinit(self: *Profile, alloc: std.mem.Allocator) void {
        self.totals.deinit(alloc);
        self.pending.deinit(alloc);
        alloc.free(self.durations);
        self.* = undefined;
    }
    /// Discards accumulated totals (for example after warm-up runs); the
    /// bridge's sample buffers stay in place.
    pub fn clear(self: *Profile) void {
        self.totals.clearRetainingCapacity();
        self.pending.clearRetainingCapacity();
        self.command_buffers = 0;
        self.unsampled = 0;
        self.gpu_seconds = 0;
    }
};

/// What a dispatch is about, for profiling: the weight encoding and matrix
/// dimensions of a matrix kernel and the bytes it must read. Elementwise
/// kernels pass `.{}`.
const Shape = struct { encoding: ?u32 = null, rows: u32 = 0, columns: u32 = 0, bytes: u64 = 0 };

pub const Backend = struct {
    alloc: std.mem.Allocator,
    handle: *anyopaque,
    pipelines: [kernel_names.len]u32,
    wrapped: std.AutoHashMapUnmanaged(usize, Buffer) = .empty,
    recording: bool = false,
    /// Diagnostic: record the generic `nu_matvec` / `nu_matmul` even when a
    /// specialized kernel applies. `metal-check` compares both paths;
    /// production leaves it off.
    generic_only: bool = false,
    /// Present after `enableProfiling`; read it after `commit()`.
    profile: ?Profile = null,

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
    /// Pairs the bridge's per-dispatch durations with the recorded kernels.
    /// Runs after every commit while profiling, including failed ones, so the
    /// pending list never leaks into the next command buffer.
    fn account(self: *Backend, gpu_before: f64, completed: bool) !void {
        const p = &self.profile.?;
        defer p.pending.clearRetainingCapacity();
        if (!completed or p.pending.items.len == 0) return;
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
        return .{ .id = id, .offset = 0, .len = len, .host = host };
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
        const completed = nu_metal_commit(self.handle) == 0;
        if (self.profile != null) try self.account(gpu_before, completed);
        if (!completed) return error.MetalExecutionFailed;
    }

    fn dispatch(self: *Backend, kernel: Kernel, buffers: []const Buffer, params: anytype, groups: u32, threads: u32, shape: Shape) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (!self.recording) return error.MetalNotRecording;
        var bindings: [7]Binding = undefined;
        if (buffers.len > bindings.len) return error.InvalidShape;
        for (buffers, bindings[0..buffers.len]) |b, *out| out.* = b.binding();
        if (nu_metal_dispatch(self.handle, self.pipelines[@intFromEnum(kernel)], &bindings, @intCast(buffers.len), @ptrCast(&params), @sizeOf(@TypeOf(params)), groups, 1, threads, 1) != 0) return error.MetalExecutionFailed;
        // Recorded only after the bridge accepted the dispatch, so pending[i]
        // is the i-th dispatch the GPU will time.
        if (self.profile) |*p| try p.pending.append(self.alloc, .{ .key = .{ .kernel = kernel, .encoding = shape.encoding, .rows = shape.rows, .columns = shape.columns }, .bytes = shape.bytes });
    }
    fn perElement(count: usize) u32 {
        return @intCast((count + 255) / 256);
    }

    // ----- Kernel encoders. Each validates shapes before recording. -----

    pub const MatvecParams = extern struct { columns: u32, encoding: u32, stride: u32, rows: u32 };
    pub const MatvecBlockParams = extern struct { columns: u32, stride: u32, rows: u32, blocks: u32 };
    /// Rows each SIMD group of a specialized matvec accumulates; must equal the
    /// template argument of the `nu_matvec_*` instantiations in kernels.metal.
    pub const rows_per_simdgroup = 4;
    const simdgroups_per_matvec_group = 4;
    /// output[r] = sum_c weights[r,c] * input[c]; weights are an encoded matrix
    /// with `stride` bytes per row. Input/output are F32 vectors. Q4_K, Q5_K,
    /// Q3_K, Q6_K, IQ3_S, IQ4_XS, and Q4_0 use the specialized block kernels when the weight range
    /// and input are aligned for their vector loads (see `specializedMatvec`).
    pub fn matvec(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, output: Buffer) !void {
        if (matrix.rows == 0 or matrix.columns == 0 or matrix.columns % 16 != 0 or matrix.bytes.len % matrix.rows != 0) return error.InvalidShape;
        const stride = matrix.bytes.len / matrix.rows;
        try quant.validateRow(matrix.encoding, stride, matrix.columns);
        if (weights.len < matrix.bytes.len or input.len < matrix.columns * 4 or output.len < matrix.rows * 4) return error.InvalidShape;
        // Weight bytes are the traffic that bounds decode; the profile divides them by measured time.
        const shape: Shape = .{ .encoding = matrix.encoding, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = matrix.bytes.len };
        if (!self.generic_only) if (specializedMatvec(matrix.encoding, weights.offset, stride, input.offset)) |kernel| {
            const p: MatvecBlockParams = .{ .columns = @intCast(matrix.columns), .stride = @intCast(stride), .rows = @intCast(matrix.rows), .blocks = @intCast(matrix.columns / 256) };
            const rows_per_group = rows_per_simdgroup * simdgroups_per_matvec_group;
            try self.dispatch(kernel, &.{ weights, input, output }, p, @intCast((matrix.rows + rows_per_group - 1) / rows_per_group), 32 * simdgroups_per_matvec_group, shape);
            return;
        };
        const p: MatvecParams = .{ .columns = @intCast(matrix.columns), .encoding = matrix.encoding, .stride = @intCast(stride), .rows = @intCast(matrix.rows) };
        try self.dispatch(.matvec, &.{ weights, input, output }, p, @intCast(matrix.rows), 32, shape);
    }
    pub const MatvecRowsParams = extern struct { columns: u32, encoding: u32, stride: u32, rows: u32, tokens: u32, in_stride: u32, out_stride: u32 };
    /// Largest batch `matmul` routes to the multi-row matvec. The 2026-09-20
    /// sweep (metal-check `--matvec-rows-bench`) measured the scalar body at
    /// 143–182 GB/s (weight bytes) at 2 rows against the 16×8 tile's 88–116,
    /// but at 3 rows only Q6_K and IQ4_XS still win and at 5–8 the per-token
    /// input loads and scalar FMA cap it at 49–67. The single safe threshold
    /// is 2; the tile serves 3–24.
    pub const small_batch_rows = 2;
    /// Whether `matmul` routes a `tokens`-row batch to the multi-row matvec.
    /// On for 2-row batches: the replay of a short accepted prefix and the
    /// 2-token commit are the batches that win, and routing only the
    /// specialized encodings keeps the tile under the rest.
    pub const route_small_batch = true;
    /// Largest token count the multi-row kernels are instantiated for; the
    /// sweep measures every count to `matvec_rows_max` even though `matmul`
    /// routes only to `small_batch_rows`.
    pub const matvec_rows_max = 8;
    pub fn usesMatvecRows(tokens: usize) bool {
        return route_small_batch and tokens >= 2 and tokens <= small_batch_rows;
    }
    /// output[t][r] = Σ_c weights[r,c] · input[t,c] for `1 < tokens ≤
    /// matvec_rows_max` activation rows. Each weight block is decoded once
    /// and multiplied against every token row, so the weight bytes (the
    /// bandwidth bound of a small batch) are read once instead of once per
    /// token. The specialized encodings use their multi-row body when aligned,
    /// every other encoding the generic one. `input` and `output` hold
    /// `tokens` rows of `in_stride`/`out_stride` floats. `matmul` routes only
    /// the specialized encodings here; `metal-check` calls it directly for
    /// every encoding.
    pub fn matvecRows(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
        if (matrix.rows == 0 or matrix.columns == 0 or matrix.columns % 16 != 0 or matrix.bytes.len % matrix.rows != 0) return error.InvalidShape;
        const stride = matrix.bytes.len / matrix.rows;
        try quant.validateRow(matrix.encoding, stride, matrix.columns);
        if (tokens < 2 or tokens > matvec_rows_max or tokens > 64) return error.InvalidShape;
        if (in_stride < matrix.columns or out_stride < matrix.rows) return error.InvalidShape;
        if (weights.len < matrix.bytes.len or input.len < ((tokens - 1) * in_stride + matrix.columns) * 4 or output.len < ((tokens - 1) * out_stride + matrix.rows) * 4) return error.InvalidShape;
        if (input.offset % 16 != 0 or in_stride % 4 != 0 or output.offset % 4 != 0) return error.InvalidShape;
        const shape: Shape = .{ .encoding = matrix.encoding, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = matrix.bytes.len };
        const p: MatvecRowsParams = .{ .columns = @intCast(matrix.columns), .encoding = matrix.encoding, .stride = @intCast(stride), .rows = @intCast(matrix.rows), .tokens = @intCast(tokens), .in_stride = @intCast(in_stride), .out_stride = @intCast(out_stride) };
        if (!self.generic_only) if (specializedMatvecRows(matrix.encoding, tokens, weights.offset, stride, input.offset)) |kernel| {
            const rows_per_group = rows_per_simdgroup * simdgroups_per_matvec_group;
            try self.dispatch(kernel, &.{ weights, input, output }, p, @intCast((matrix.rows + rows_per_group - 1) / rows_per_group), 32 * simdgroups_per_matvec_group, shape);
            return;
        };
        try self.dispatch(.matvec_rows, &.{ weights, input, output }, p, @intCast(matrix.rows), 32, shape);
    }
    /// The multi-row bodies that exist (Q4_K, Q5_K, Q6_K, IQ4_XS), one per token
    /// count 2..8; every other encoding takes the generic `nu_matvec_rows` above.
    /// The token count is a template parameter so the body's accumulator loops
    /// are compile-time bound (see kernels.metal); the per-encoding kernels are
    /// contiguous in `Kernel` at `t2`.
    pub fn specializedMatvecRows(encoding: u32, tokens: usize, weight_offset: usize, stride: usize, input_offset: usize) ?Kernel {
        if (input_offset % 16 != 0 or tokens < 2 or tokens > matvec_rows_max or !blockAligned(encoding, weight_offset, stride)) return null;
        const base: u32 = switch (encoding) {
            12 => @intFromEnum(Kernel.matvec_rows_q4_k_t2),
            13 => @intFromEnum(Kernel.matvec_rows_q5_k_t2),
            14 => @intFromEnum(Kernel.matvec_rows_q6_k_t2),
            23 => @intFromEnum(Kernel.matvec_rows_iq4_xs_t2),
            else => return null,
        };
        return @enumFromInt(base + tokens - 2);
    }
    pub const MatmulParams = extern struct { columns: u32, encoding: u32, stride: u32, rows: u32, tokens: u32, in_stride: u32, out_stride: u32, row_tiles: u32 };
    /// Token padding of `matmul`: activation buffers hold a multiple of this
    /// many token rows (the largest token tile any instantiation uses).
    pub const matmul_tile = 64;
    /// Chunk length up to which the split-K 16×8 tile serves. Set to the
    /// measured crossover between the small tile and the 32×32 one.
    pub const small_chunk_tokens = 24;
    pub fn matmulPadded(tokens: usize) usize {
        return (tokens + matmul_tile - 1) / matmul_tile * matmul_tile;
    }
    /// Output tile of a matmul kernel and whether its operands are rounded to
    /// half; must match the instantiations in kernels.metal. The specialized
    /// tiles hold half operands and accumulate in F32: 64×64 tiles, 32×32 for
    /// chunks of at most 32 tokens, and a 16×8 split-K tile for chunks of at
    /// most `small_chunk_tokens`, where the grid's row parallelism matters more
    /// than the tile's K reuse; the generic tile keeps F32 operands in 32×32.
    pub const MatmulGeometry = struct { rows: usize, tokens: usize, half: bool };
    pub fn matmulGeometry(kernel: Kernel) MatmulGeometry {
        return switch (kernel) {
            .matmul => .{ .rows = 32, .tokens = 32, .half = false },
            .matmul_q3_k, .matmul_q4_k, .matmul_q5_k, .matmul_q6_k, .matmul_iq3_s, .matmul_iq4_xs, .matmul_q4_0, .matmul_pq2_0, .matmul_ptq1_0 => .{ .rows = 64, .tokens = 64, .half = true },
            .matmul_q3_k_8, .matmul_q4_k_8, .matmul_q5_k_8, .matmul_q6_k_8, .matmul_iq3_s_8, .matmul_iq4_xs_8, .matmul_q4_0_8, .matmul_pq2_0_8, .matmul_ptq1_0_8 => .{ .rows = 16, .tokens = 8, .half = true },
            .matmul_q3_k_w8, .matmul_q4_k_w8, .matmul_q5_k_w8, .matmul_q6_k_w8, .matmul_iq3_s_w8, .matmul_iq4_xs_w8, .matmul_q4_0_w8, .matmul_pq2_0_w8, .matmul_ptq1_0_w8 => .{ .rows = 32, .tokens = 8, .half = true },
            else => .{ .rows = 32, .tokens = 32, .half = true },
        };
    }
    /// out[t][r] = Σ_c weights[r,c] · input[t,c] for `tokens` activation rows
    /// (chunked prefill). `input` holds `matmulPadded(tokens)` rows of
    /// `in_stride` floats and `output` as many rows of `out_stride`; rows past
    /// `tokens` are scratch the caller must ignore. Requires columns % 64 == 0,
    /// rows % 8 == 0, and a float4-aligned input (offset and stride). Weights
    /// are read once per token tile. Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS,
    /// and Q4_0 go through their specialized half-operand tile when the weight
    /// range is aligned (`specializedMatmul`), every other case through the
    /// generic F32 tile (`matmulGeometry`).
    pub fn matmul(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
        return self.matmulImpl(weights, matrix, input, in_stride, output, out_stride, tokens, .auto);
    }
    /// Same contract as `matmul`, but always uses a matrix tile. Keeps benchmark
    /// controls independent of production small-batch routing.
    pub fn matmulTile(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
        return self.matmulImpl(weights, matrix, input, in_stride, output, out_stride, tokens, .tile);
    }
    /// Same contract, forcing the wide 32×8 split-K tile (KERN-14's
    /// candidate, measured against `matmulTile`'s 16×8 control). Rejected
    /// when the encoding or alignment has no wide body, or past its 8-token
    /// tile.
    pub fn matmulTile32(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
        return self.matmulImpl(weights, matrix, input, in_stride, output, out_stride, tokens, .wide);
    }
    const MatmulPolicy = enum { auto, tile, wide };
    fn matmulImpl(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize, policy: MatmulPolicy) !void {
        if (matrix.rows == 0 or matrix.rows % 8 != 0 or matrix.columns == 0 or matrix.columns % 64 != 0 or matrix.bytes.len % matrix.rows != 0) return error.InvalidShape;
        if (tokens == 0 or in_stride < matrix.columns or out_stride < matrix.rows) return error.InvalidShape;
        const stride = matrix.bytes.len / matrix.rows;
        try quant.validateRow(matrix.encoding, stride, matrix.columns);
        const padded = matmulPadded(tokens);
        if (weights.len < matrix.bytes.len or input.len < padded * in_stride * 4 or output.len < padded * out_stride * 4) return error.InvalidShape;
        if (input.offset % 16 != 0 or in_stride % 4 != 0 or output.offset % 4 != 0) return error.InvalidShape;
        // A 2-row batch is a matvec problem, not a tile one, and only the
        // specialized bodies beat the tile: the generic `nu_matvec_rows` is
        // slower there (the sweep measures both).
        if (policy == .auto and usesMatvecRows(tokens) and !self.generic_only and
            specializedMatvecRows(matrix.encoding, tokens, weights.offset, stride, input.offset) != null)
            return self.matvecRows(weights, matrix, input, in_stride, output, out_stride, tokens);
        const kernel = (if (self.generic_only) null else switch (policy) {
            .auto, .tile => specializedMatmul(matrix.encoding, weights.offset, stride, tokens),
            .wide => specializedMatmulWide(matrix.encoding, weights.offset, stride, tokens),
        }) orelse blk: {
            if (policy == .wide) return error.InvalidShape;
            break :blk .matmul;
        };
        const geometry = matmulGeometry(kernel);
        const row_tiles = (matrix.rows + geometry.rows - 1) / geometry.rows;
        // The buffers hold `padded` rows, a multiple of every token tile.
        const token_tiles = (tokens + geometry.tokens - 1) / geometry.tokens;
        const p: MatmulParams = .{ .columns = @intCast(matrix.columns), .encoding = matrix.encoding, .stride = @intCast(stride), .rows = @intCast(matrix.rows), .tokens = @intCast(tokens), .in_stride = @intCast(in_stride), .out_stride = @intCast(out_stride), .row_tiles = @intCast(row_tiles) };
        // Weights are read once per token tile, so the profile attributes that many bytes.
        const shape: Shape = .{ .encoding = matrix.encoding, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = matrix.bytes.len * token_tiles };
        try self.dispatch(kernel, &.{ weights, input, output }, p, @intCast(row_tiles * token_tiles), 128, shape);
    }
    /// Whether the row start (buffer offset) and the row stride are multiples
    /// of the vector-load width the encoding's block size allows (Q4_K/Q5_K
    /// 16, IQ4_XS 8, Q3_K/Q6_K/IQ3_S/Q4_0 2). False for encodings without a
    /// specialized kernel.
    fn blockAligned(encoding: u32, weight_offset: usize, stride: usize) bool {
        const alignment: usize = switch (encoding) {
            12, 13 => 16,
            23 => 8,
            143 => 4,
            2, 11, 14, 21, 142 => 2,
            else => return false,
        };
        return weight_offset % alignment == 0 and stride % alignment == 0;
    }
    /// Picks a specialized matvec when its vector loads are aligned
    /// (`blockAligned`) and the input vector is float4-aligned. Otherwise
    /// `null`: generic path. The K-quant kernels walk 256-value strides;
    /// the Q4_0 kernel walks 32-value blocks and the ternary kernels
    /// 128-value blocks (lanes past the last block idle), so any whole-block
    /// row serves those three.
    pub fn specializedMatvec(encoding: u32, weight_offset: usize, stride: usize, input_offset: usize) ?Kernel {
        if (input_offset % 16 != 0 or !blockAligned(encoding, weight_offset, stride)) return null;
        return switch (encoding) {
            2 => .matvec_q4_0,
            142 => .matvec_pq2_0,
            143 => .matvec_ptq1_0,
            11 => .matvec_q3_k,
            21 => .matvec_iq3_s,
            12 => .matvec_q4_k,
            13 => .matvec_q5_k,
            14 => .matvec_q6_k,
            23 => .matvec_iq4_xs,
            else => unreachable,
        };
    }
    /// Picks the wide 32×8 split-K tile for the specialized encodings, up to
    /// its 8-token tile. The KERN-14 experiment's candidate; nothing routes
    /// to it in production.
    pub fn specializedMatmulWide(encoding: u32, weight_offset: usize, stride: usize, tokens: usize) ?Kernel {
        if (tokens == 0 or tokens > 8 or !blockAligned(encoding, weight_offset, stride)) return null;
        return switch (encoding) {
            2 => .matmul_q4_0_w8,
            142 => .matmul_pq2_0_w8,
            143 => .matmul_ptq1_0_w8,
            11 => .matmul_q3_k_w8,
            21 => .matmul_iq3_s_w8,
            12 => .matmul_q4_k_w8,
            13 => .matmul_q5_k_w8,
            14 => .matmul_q6_k_w8,
            23 => .matmul_iq4_xs_w8,
            else => null,
        };
    }
    /// Picks a specialized matmul tile under the same weight alignment
    /// rules (the activation tile is staged through float4 loads, which
    /// `matmul` checks), first the 16×8 split-K instantiation for chunks of at
    /// most `small_chunk_tokens` tokens, then the 32×32 one for at most 32.
    pub fn specializedMatmul(encoding: u32, weight_offset: usize, stride: usize, tokens: usize) ?Kernel {
        if (!blockAligned(encoding, weight_offset, stride)) return null;
        if (tokens <= small_chunk_tokens) return switch (encoding) {
            2 => .matmul_q4_0_8,
            142 => .matmul_pq2_0_8,
            143 => .matmul_ptq1_0_8,
            11 => .matmul_q3_k_8,
            21 => .matmul_iq3_s_8,
            12 => .matmul_q4_k_8,
            13 => .matmul_q5_k_8,
            14 => .matmul_q6_k_8,
            23 => .matmul_iq4_xs_8,
            else => unreachable,
        };
        if (tokens <= 32) return switch (encoding) {
            2 => .matmul_q4_0_32,
            142 => .matmul_pq2_0_32,
            143 => .matmul_ptq1_0_32,
            11 => .matmul_q3_k_32,
            21 => .matmul_iq3_s_32,
            12 => .matmul_q4_k_32,
            13 => .matmul_q5_k_32,
            14 => .matmul_q6_k_32,
            23 => .matmul_iq4_xs_32,
            else => unreachable,
        };
        return switch (encoding) {
            2 => .matmul_q4_0,
            142 => .matmul_pq2_0,
            143 => .matmul_ptq1_0,
            11 => .matmul_q3_k,
            21 => .matmul_iq3_s,
            12 => .matmul_q4_k,
            13 => .matmul_q5_k,
            14 => .matmul_q6_k,
            23 => .matmul_iq4_xs,
            else => unreachable,
        };
    }
    /// Borrowed projection descriptors. All projections read the same input.
    /// Plain outputs must be disjoint; a gate pair (SiLU or tanh GELU) names
    /// the same output in both entries and writes only act(segment[0]) * segment[1].
    pub const Segment = struct { weights: Buffer, matrix: cpu.Matrix, output: Buffer };
    pub const SegmentMode = enum(u32) { plain, silu_mul_pair, gelu_mul_pair };
    const SegmentParams = extern struct { rows: u32, weight_slot: u32, weight_offset: u32, encoding: u32, stride: u32, output_slot: u32, output_offset: u32 };
    const MatvecSegments = extern struct { columns: u32, blocks: u32, count: u32, mode: u32, segments: [4]SegmentParams };

    fn overlaps(a: Buffer, a_len: usize, b: Buffer, b_len: usize) bool {
        if (a.id != b.id) return false;
        // Subtraction avoids overflow when validating caller-provided offsets.
        return if (a.offset <= b.offset) b.offset - a.offset < a_len else a.offset - b.offset < b_len;
    }
    /// Bind each underlying buffer once at offset zero. Segment byte offsets
    /// retain its original slice. This fits both four weights + packed DeltaNet
    /// outputs and three weights + attention outputs in six slots, plus input.
    /// The rebased descriptors are used only as Metal bindings, never as host views.
    fn segmentSlot(bindings: *[7]Buffer, used: *usize, buffer: Buffer) !u32 {
        for (1..used.*) |i| if (bindings[i].id == buffer.id) return @intCast(i);
        if (used.* == bindings.len) return error.InvalidShape;
        const index = used.*;
        bindings[index] = buffer;
        bindings[index].offset = 0;
        used.* += 1;
        return @intCast(index);
    }
    pub fn matvecSegments(self: *Backend, segments: []const Segment, input: Buffer, mode: SegmentMode) !void {
        // Shape first, then availability: with Metal compiled out the early
        // `MetalNotEnabled` return would otherwise make `InvalidShape`
        // unreachable, and the adapter's switch on it would not compile in
        // a CPU-only build.
        if (segments.len == 0 or segments.len > 4) return error.InvalidShape;
        if (!enabled) return error.MetalNotEnabled;
        const columns = segments[0].matrix.columns;
        if (columns == 0 or columns > std.math.maxInt(u32) / 4 or columns % 16 != 0 or input.len < columns * 4 or input.offset % 4 != 0) return error.InvalidShape;
        if (mode != .plain and (segments.len != 2 or segments[0].matrix.rows != segments[1].matrix.rows or segments[0].output.id != segments[1].output.id or segments[0].output.offset != segments[1].output.offset)) return error.InvalidShape;
        var bindings: [7]Buffer = @splat(input);
        var used: usize = 1;
        var p: MatvecSegments = std.mem.zeroes(MatvecSegments);
        p.columns = @intCast(columns);
        p.blocks = @intCast(columns / 256);
        p.count = @intCast(segments.len);
        p.mode = @intFromEnum(mode);
        var total_rows: u32 = 0;
        var bytes: u64 = 0;
        for (segments, 0..) |s, i| {
            const m = s.matrix;
            if (m.rows == 0 or m.rows > std.math.maxInt(u32) / 4 or m.rows % 16 != 0 or m.columns != columns or m.bytes.len % m.rows != 0) return error.InvalidShape;
            const stride = m.bytes.len / m.rows;
            try quant.validateRow(m.encoding, stride, columns);
            if (s.weights.len < m.bytes.len or s.output.len < m.rows * 4 or s.output.offset % 4 != 0) return error.InvalidShape;
            if (overlaps(s.output, m.rows * 4, input, columns * 4)) return error.InvalidShape;
            for (segments) |other| if (overlaps(s.output, m.rows * 4, other.weights, other.matrix.bytes.len)) return error.InvalidShape;
            if (mode == .plain) for (segments[0..i]) |other| {
                if (overlaps(s.output, m.rows * 4, other.output, other.matrix.rows * 4)) return error.InvalidShape;
            };
            total_rows = std.math.add(u32, total_rows, @intCast(m.rows)) catch return error.InvalidShape;
            bytes = std.math.add(u64, bytes, m.bytes.len) catch return error.InvalidShape;
            const specialized = !self.generic_only and specializedMatvec(m.encoding, s.weights.offset, stride, input.offset) != null;
            p.segments[i] = .{
                .rows = @intCast(m.rows),
                .weight_slot = try segmentSlot(&bindings, &used, s.weights),
                .weight_offset = std.math.cast(u32, s.weights.offset) orelse return error.InvalidShape,
                .encoding = m.encoding | (if (specialized) @as(u32, 0) else 0x80000000),
                .stride = std.math.cast(u32, stride) orelse return error.InvalidShape,
                .output_slot = try segmentSlot(&bindings, &used, s.output),
                .output_offset = std.math.cast(u32, s.output.offset) orelse return error.InvalidShape,
            };
        }
        // Unused slots bind the valid input buffer, because the kernel signature
        // declares all seven. No dispatch is recorded before all validation passes.
        const groups = if (mode == .plain) total_rows / 16 else p.segments[0].rows / 8;
        try self.dispatch(.matvec_segments, &bindings, p, groups, 128, .{ .rows = total_rows, .columns = p.columns, .bytes = bytes });
    }

    // ----- Mixture of experts: routing, gathered projections, combine. -----

    pub const RouteParams = extern struct { experts: u32, k: u32, rows: u32, in_stride: u32 };
    /// Bounds of `route`: one logit per thread of the 256-thread group, and
    /// the selected-probability scratch (`NU_ROUTE_MAX_K`).
    pub const route_max_experts = 256;
    pub const route_max_k = 64;
    /// For each of `rows` logit rows (`[row][in_stride]`), the `k` experts
    /// with the largest logits by (value desc, index asc) into `indices`
    /// (`[row][k]` u32) and their softmax probabilities renormalized to sum
    /// one into `weights` (`[row][k]` f32): `cpu.experts.route` per row.
    pub fn route(self: *Backend, logits: Buffer, experts: usize, k: usize, rows: usize, in_stride: usize, indices: Buffer, weights: Buffer) !void {
        if (experts == 0 or experts > route_max_experts or k == 0 or k > experts or k > route_max_k or rows == 0 or in_stride < experts) return error.InvalidShape;
        if (logits.len < ((rows - 1) * in_stride + experts) * 4 or indices.len < rows * k * 4 or weights.len < rows * k * 4 or indices.offset % 4 != 0 or weights.offset % 4 != 0) return error.InvalidShape;
        const p: RouteParams = .{ .experts = @intCast(experts), .k = @intCast(k), .rows = @intCast(rows), .in_stride = @intCast(in_stride) };
        try self.dispatch(.route, &.{ logits, indices, weights }, p, @intCast(rows), 256, .{});
    }

    pub const ExpertMatvecParams = extern struct { columns: u32, stride: u32, rows: u32, blocks: u32, encoding: u32, experts: u32, slots: u32, in_stride: u32, out_stride: u32, row_groups: u32 };
    /// Gathered projection: for each slot `s < slots`,
    /// `output[s·out_stride + r] = W[indices[s]][r] · input[s·in_stride]`
    /// over the expert matrices of `tensor` (`cpu.ExpertMatrix`, experts
    /// contiguous). `in_stride` 0 shares one input across the slots.
    /// Specialized bodies apply under `matvec`'s alignment rules (the input
    /// stride must keep every slot float4-aligned); otherwise the generic
    /// decoder. Reads only the selected experts' bytes.
    pub fn matvecExperts(self: *Backend, weights: Buffer, tensor: cpu.ExpertMatrix, indices: Buffer, slots: usize, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize) !void {
        const per_expert = tensor.expertBytes() catch return error.InvalidShape;
        const stride = per_expert / tensor.rows;
        if (tensor.columns % 16 != 0 or tensor.rows > std.math.maxInt(u32) / 4 or tensor.experts > std.math.maxInt(u32)) return error.InvalidShape;
        if (slots == 0 or slots > std.math.maxInt(u16) or out_stride < tensor.rows or (in_stride != 0 and in_stride < tensor.columns) or in_stride % 4 != 0) return error.InvalidShape;
        if (weights.len < tensor.bytes.len or indices.len < slots * 4 or indices.offset % 4 != 0) return error.InvalidShape;
        if (input.len < ((slots - 1) * in_stride + tensor.columns) * 4 or output.len < ((slots - 1) * out_stride + tensor.rows) * 4 or output.offset % 4 != 0) return error.InvalidShape;
        const specialized = !self.generic_only and specializedMatvec(tensor.encoding, weights.offset, stride, input.offset) != null;
        const row_groups = (tensor.rows + 15) / 16;
        const p: ExpertMatvecParams = .{
            .columns = @intCast(tensor.columns),
            .stride = std.math.cast(u32, stride) orelse return error.InvalidShape,
            .rows = @intCast(tensor.rows),
            .blocks = @intCast(tensor.columns / 256),
            .encoding = tensor.encoding | (if (specialized) @as(u32, 0) else 0x80000000),
            .experts = @intCast(tensor.experts),
            .slots = @intCast(slots),
            .in_stride = @intCast(in_stride),
            .out_stride = std.math.cast(u32, out_stride) orelse return error.InvalidShape,
            .row_groups = @intCast(row_groups),
        };
        // The bytes that bound the dispatch are the selected experts', read once each.
        const shape: Shape = .{ .encoding = tensor.encoding, .rows = @intCast(tensor.rows * slots), .columns = @intCast(tensor.columns), .bytes = @as(u64, per_expert) * slots };
        try self.dispatch(.matvec_experts, &.{ weights, input, output, indices }, p, @intCast(slots * row_groups), 32 * simdgroups_per_matvec_group, shape);
    }

    pub const CombineParams = extern struct { columns: u32, slots: u32, rows: u32, in_stride: u32, out_stride: u32, experts: u32, flags: u32 };
    pub const CombineShape = struct { columns: usize, slots: usize, rows: usize = 1, experts: usize, in_stride: usize, out_stride: usize };
    /// `output[row][c] = Σ_s weights[row][s] · scale[indices[row][s]] · values[row·slots + s][c]`
    /// (the scale factor only when `scales` is given): the weighted sum of
    /// the slots' down projections into one row per token.
    pub fn combineExperts(self: *Backend, values: Buffer, weights: Buffer, indices: Buffer, scales: ?Buffer, output: Buffer, s: CombineShape) !void {
        if (s.columns == 0 or s.slots == 0 or s.rows == 0 or s.experts == 0 or s.in_stride < s.columns or s.out_stride < s.columns) return error.InvalidShape;
        const slot_rows = s.rows * s.slots;
        if (values.len < ((slot_rows - 1) * s.in_stride + s.columns) * 4 or weights.len < slot_rows * 4 or indices.len < slot_rows * 4 or indices.offset % 4 != 0) return error.InvalidShape;
        if (output.len < ((s.rows - 1) * s.out_stride + s.columns) * 4) return error.InvalidShape;
        if (scales) |sc| if (sc.len < s.experts * 4) return error.InvalidShape;
        const p: CombineParams = .{ .columns = @intCast(s.columns), .slots = @intCast(s.slots), .rows = @intCast(s.rows), .in_stride = @intCast(s.in_stride), .out_stride = @intCast(s.out_stride), .experts = @intCast(s.experts), .flags = if (scales != null) 1 else 0 };
        try self.dispatch(.combine_experts, &.{ values, weights, indices, scales orelse weights, output }, p, perElement(s.columns * s.rows), 256, .{});
    }

    pub const GeluRowsParams = extern struct { width: u32, rows: u32, gate_stride: u32, up_stride: u32, out_stride: u32 };
    /// `output[r][i] = gelu(gate[r][i]) · up[r][i]` over `rows` strided rows
    /// of `width`; with a fused gate-up row, `up` is the gate buffer sliced
    /// at the up half. The output must not overlap the inputs.
    pub fn geluMulRows(self: *Backend, gate: Buffer, up: Buffer, output: Buffer, width: usize, rows: usize, gate_stride: usize, up_stride: usize, out_stride: usize) !void {
        if (width == 0 or rows == 0 or gate_stride < width or up_stride < width or out_stride < width) return error.InvalidShape;
        if (gate.len < ((rows - 1) * gate_stride + width) * 4 or up.len < ((rows - 1) * up_stride + width) * 4 or output.len < ((rows - 1) * out_stride + width) * 4) return error.InvalidShape;
        const out_len = ((rows - 1) * out_stride + width) * 4;
        if (overlaps(output, out_len, gate, ((rows - 1) * gate_stride + width) * 4) or overlaps(output, out_len, up, ((rows - 1) * up_stride + width) * 4)) return error.InvalidShape;
        const p: GeluRowsParams = .{ .width = @intCast(width), .rows = @intCast(rows), .gate_stride = @intCast(gate_stride), .up_stride = @intCast(up_stride), .out_stride = @intCast(out_stride) };
        try self.dispatch(.gelu_mul_rows, &.{ gate, up, output }, p, perElement(width * rows), 256, .{});
    }

    // Prefill: the chunk's slot rows grouped by expert, then gathered tiles.

    pub const ExpertListsParams = extern struct { experts: u32, n: u32, max_tiles: u32 };
    /// Slot rows per gathered tile; the tile bound below counts them.
    pub const expert_tile = 32;
    /// Tiles the lists of `n` slot rows over `experts` can hold at most:
    /// Σ ceil(count/32) ≤ n/32 + experts. This bounds the gathered matmul
    /// grid before the counts are known.
    pub fn expertTileBound(n: usize, experts: usize) usize {
        return (n + expert_tile - 1) / expert_tile + experts;
    }
    /// Words of the lists buffer, in order: the tile count, `n`, the
    /// `experts + 1` exclusive prefix offsets, `expertTileBound` tiles of
    /// (expert, first, count), and the `n` row list.
    pub const ExpertListsLayout = struct { tiles_at: usize, rows_at: usize, words: usize };
    pub fn expertListsLayout(n: usize, experts: usize) ExpertListsLayout {
        const tiles_at = experts + 3;
        const rows_at = tiles_at + 3 * expertTileBound(n, experts);
        return .{ .tiles_at = tiles_at, .rows_at = rows_at, .words = rows_at + n };
    }
    /// Groups the `rows · k` slot rows of `indices` (`[row][k]`, as `route`
    /// writes them) by expert into `lists` (`expertListsLayout(rows · k,
    /// experts).words` u32, layout above) for `matmulExperts`. One
    /// threadgroup; at most 256 experts.
    pub fn expertLists(self: *Backend, indices: Buffer, rows: usize, k: usize, experts: usize, lists: Buffer) !void {
        if (rows == 0 or k == 0 or experts == 0 or experts > route_max_experts) return error.InvalidShape;
        const n = std.math.mul(usize, rows, k) catch return error.InvalidShape;
        if (n > std.math.maxInt(u32) / 4) return error.InvalidShape;
        const layout = expertListsLayout(n, experts);
        if (indices.len < n * 4 or indices.offset % 4 != 0 or lists.len < layout.words * 4 or lists.offset % 4 != 0) return error.InvalidShape;
        const p: ExpertListsParams = .{ .experts = @intCast(experts), .n = @intCast(n), .max_tiles = @intCast(expertTileBound(n, experts)) };
        try self.dispatch(.expert_lists, &.{ indices, lists }, p, 1, 256, .{});
    }

    pub const MatmulExpertsParams = extern struct { columns: u32, encoding: u32, stride: u32, rows: u32, in_stride: u32, out_stride: u32, row_tiles: u32, experts: u32, in_group: u32, tiles_at: u32, rows_at: u32 };
    /// Gathered projection over the row lists: for each slot row `s` of the
    /// chunk (`rows · k` of them, expert `e_s` per the lists),
    /// `output[s·out_stride + r] = W[e_s][r] · input[(s / in_group)·in_stride]`.
    /// `in_group` = k shares a token's input across its slots (gate-up);
    /// 1 reads one input row per slot (down). Weights are read once per
    /// 32-row tile of an expert's slot rows; no padding rows are written,
    /// so `output` holds exactly `rows · k` rows. Same shape rules as
    /// `matmul` (columns % 64, rows % 8, float4-aligned input). Q4_0 takes
    /// the half tile under the matvec alignment rules, everything else the
    /// generic F32 tile. The profile attributes no bytes: the volume read
    /// depends on the routing.
    pub fn matmulExperts(self: *Backend, weights: Buffer, tensor: cpu.ExpertMatrix, lists: Buffer, rows: usize, k: usize, input: Buffer, in_stride: usize, in_group: usize, output: Buffer, out_stride: usize) !void {
        const per_expert = tensor.expertBytes() catch return error.InvalidShape;
        const stride = per_expert / tensor.rows;
        if (tensor.rows % 8 != 0 or tensor.columns % 64 != 0 or tensor.rows > std.math.maxInt(u32) / 4 or tensor.experts > route_max_experts) return error.InvalidShape;
        if (rows == 0 or k == 0 or in_group == 0 or in_stride < tensor.columns or out_stride < tensor.rows) return error.InvalidShape;
        const n = std.math.mul(usize, rows, k) catch return error.InvalidShape;
        if (n > std.math.maxInt(u32) / 4) return error.InvalidShape;
        const layout = expertListsLayout(n, tensor.experts);
        const input_rows = (n + in_group - 1) / in_group;
        if (weights.len < tensor.bytes.len or lists.len < layout.words * 4 or lists.offset % 4 != 0) return error.InvalidShape;
        if (input.len < ((input_rows - 1) * in_stride + tensor.columns) * 4 or input.offset % 16 != 0 or in_stride % 4 != 0) return error.InvalidShape;
        if (output.len < ((n - 1) * out_stride + tensor.rows) * 4 or output.offset % 4 != 0) return error.InvalidShape;
        const specialized = !self.generic_only and tensor.encoding == 2 and blockAligned(tensor.encoding, weights.offset, stride);
        const kernel: Kernel = if (specialized) .matmul_experts_q4_0 else .matmul_experts;
        const tile_rows: usize = if (specialized) 64 else 32;
        const row_tiles = (tensor.rows + tile_rows - 1) / tile_rows;
        const p: MatmulExpertsParams = .{
            .columns = @intCast(tensor.columns),
            .encoding = tensor.encoding,
            .stride = std.math.cast(u32, stride) orelse return error.InvalidShape,
            .rows = @intCast(tensor.rows),
            .in_stride = std.math.cast(u32, in_stride) orelse return error.InvalidShape,
            .out_stride = std.math.cast(u32, out_stride) orelse return error.InvalidShape,
            .row_tiles = @intCast(row_tiles),
            .experts = @intCast(tensor.experts),
            .in_group = std.math.cast(u32, in_group) orelse return error.InvalidShape,
            .tiles_at = @intCast(layout.tiles_at),
            .rows_at = @intCast(layout.rows_at),
        };
        const groups = std.math.cast(u32, row_tiles * expertTileBound(n, tensor.experts)) orelse return error.InvalidShape;
        try self.dispatch(kernel, &.{ weights, input, output, lists }, p, groups, 128, .{ .encoding = tensor.encoding, .rows = @intCast(tensor.rows), .columns = @intCast(tensor.columns) });
    }

    pub const EmbedParams = extern struct { columns: u32, encoding: u32, stride: u32, token: u32 };
    pub fn embed(self: *Backend, weights: Buffer, matrix: cpu.Matrix, token: u32, output: Buffer) !void {
        if (matrix.rows == 0 or matrix.columns == 0 or matrix.columns % 16 != 0 or matrix.bytes.len % matrix.rows != 0 or token >= matrix.rows) return error.InvalidShape;
        const stride = matrix.bytes.len / matrix.rows;
        try quant.validateRow(matrix.encoding, stride, matrix.columns);
        if (weights.len < matrix.bytes.len or output.len < matrix.columns * 4) return error.InvalidShape;
        const p: EmbedParams = .{ .columns = @intCast(matrix.columns), .encoding = matrix.encoding, .stride = @intCast(stride), .token = token };
        // One row is read, so the profiled shape is 1 x columns and `stride` bytes.
        try self.dispatch(.embed, &.{ weights, output }, p, perElement(matrix.columns / 16), 256, .{ .encoding = matrix.encoding, .rows = 1, .columns = @intCast(matrix.columns), .bytes = stride });
    }
    pub const NormParams = extern struct { width: u32, in_stride: u32, out_stride: u32, mult_stride: u32, eps: f32, flags: u32 };
    pub const Norm = struct {
        rows: usize,
        width: usize,
        in_stride: usize,
        out_stride: usize,
        eps: f32 = 1e-6,
        /// Optional per-row multiplier applied as silu(multiplier) after the norm.
        silu_multiplier: ?struct { buffer: Buffer, stride: usize } = null,
    };
    /// Weighted RMSNorm over `rows` rows of `width`; input and output may alias exactly.
    pub fn rmsNorm(self: *Backend, input: Buffer, weight: Buffer, output: Buffer, spec: Norm) !void {
        if (spec.rows == 0 or spec.width == 0 or spec.in_stride < spec.width or spec.out_stride < spec.width or !(spec.eps > 0)) return error.InvalidShape;
        if (input.len < ((spec.rows - 1) * spec.in_stride + spec.width) * 4 or output.len < ((spec.rows - 1) * spec.out_stride + spec.width) * 4 or weight.len < spec.width * 4) return error.InvalidShape;
        var mult = input;
        var mult_stride: usize = 0;
        var flags: u32 = 0;
        if (spec.silu_multiplier) |m| {
            if (m.stride < spec.width or m.buffer.len < ((spec.rows - 1) * m.stride + spec.width) * 4) return error.InvalidShape;
            mult = m.buffer;
            mult_stride = m.stride;
            flags = 1;
        }
        const p: NormParams = .{ .width = @intCast(spec.width), .in_stride = @intCast(spec.in_stride), .out_stride = @intCast(spec.out_stride), .mult_stride = @intCast(mult_stride), .eps = spec.eps, .flags = flags };
        try self.dispatch(.rmsnorm, &.{ input, weight, output, mult }, p, @intCast(spec.rows), 256, .{});
    }
    pub const HadamardParams = extern struct { width: u32, stride: u32, rows: u32, blocks: u32, inverse: u32 };
    /// The block the transform is written for (`cpu.hadamard`'s contract).
    pub const hadamard_block = 1024;
    /// In place over `rows` rows of `width` floats at `stride`: per 1,024-block,
    /// forward `x = H (signs ⊙ x)`, inverse `x = signs ⊙ (H x)`
    /// (`cpu.hadamard.forward` / `.inverse`). `signs` holds `width` floats of
    /// ±1; data and signs must be float4-aligned, `width` a multiple of the block.
    pub fn hadamard(self: *Backend, data: Buffer, signs: Buffer, width: usize, rows: usize, stride: usize, inverse: bool) !void {
        if (rows == 0 or width == 0 or width % hadamard_block != 0 or stride < width or stride % 4 != 0) return error.InvalidShape;
        if (data.offset % 16 != 0 or signs.offset % 16 != 0 or signs.len < width * 4 or data.len < ((rows - 1) * stride + width) * 4) return error.InvalidShape;
        const blocks = width / hadamard_block;
        if (rows * blocks > std.math.maxInt(u32)) return error.InvalidShape;
        const p: HadamardParams = .{ .width = @intCast(width), .stride = @intCast(stride), .rows = @intCast(rows), .blocks = @intCast(blocks), .inverse = @intFromBool(inverse) };
        try self.dispatch(.hadamard, &.{ data, signs }, p, @intCast(rows * blocks), 256, .{});
    }
    pub const GatherParams = extern struct { width: u32, groups: u32, rows: u32, in_stride: u32, out_stride: u32 };
    /// dst[r][g][0..width] = src[r][map[g]][0..width] over `rows` rows of
    /// `groups` vectors: a fixed permutation of the vectors inside every row.
    /// `map` holds `groups` u32 indices below `groups` (a bijection is the
    /// caller's contract); `dst` and `src` must not overlap.
    pub fn gatherRows(self: *Backend, dst: Buffer, src: Buffer, map: Buffer, width: usize, groups: usize, rows: usize, in_stride: usize, out_stride: usize) !void {
        if (width == 0 or groups == 0 or rows == 0 or in_stride < groups * width or out_stride < groups * width) return error.InvalidShape;
        const count = rows * groups * width;
        if (count > std.math.maxInt(u32) or map.len < groups * 4 or map.offset % 4 != 0) return error.InvalidShape;
        const src_len = ((rows - 1) * in_stride + groups * width) * 4;
        const dst_len = ((rows - 1) * out_stride + groups * width) * 4;
        if (src.len < src_len or dst.len < dst_len or overlaps(dst, dst_len, src, src_len)) return error.InvalidShape;
        const p: GatherParams = .{ .width = @intCast(width), .groups = @intCast(groups), .rows = @intCast(rows), .in_stride = @intCast(in_stride), .out_stride = @intCast(out_stride) };
        try self.dispatch(.gather_rows, &.{ dst, src, map }, p, perElement(count), 256, .{});
    }
    pub const L2Params = extern struct { width: u32, stride: u32, eps: f32, rows: u32, heads: u32, row_stride: u32 };
    pub fn l2Norm(self: *Backend, data: Buffer, rows: usize, width: usize, stride: usize, eps: f32) !void {
        if (rows == 0 or width == 0 or stride < width or !(eps > 0) or data.len < ((rows - 1) * stride + width) * 4) return error.InvalidShape;
        const p: L2Params = .{ .width = @intCast(width), .stride = @intCast(stride), .eps = eps, .rows = @intCast(rows), .heads = @intCast(rows), .row_stride = 0 };
        try self.dispatch(.l2norm, &.{data}, p, @intCast(rows), 32, .{});
    }
    /// `l2Norm` over `rows` token rows of `row_stride`, each holding `heads`
    /// vectors of `width` at `stride`.
    pub fn l2NormRows(self: *Backend, data: Buffer, rows: usize, heads: usize, width: usize, stride: usize, row_stride: usize, eps: f32) !void {
        if (rows == 0 or heads == 0 or width == 0 or stride < width or row_stride < (heads - 1) * stride + width or !(eps > 0)) return error.InvalidShape;
        if (data.len < ((rows - 1) * row_stride + (heads - 1) * stride + width) * 4) return error.InvalidShape;
        const p: L2Params = .{ .width = @intCast(width), .stride = @intCast(stride), .eps = eps, .rows = @intCast(rows * heads), .heads = @intCast(heads), .row_stride = @intCast(row_stride) };
        try self.dispatch(.l2norm, &.{data}, p, @intCast(rows * heads), 32, .{});
    }
    /// Which two channels form rotation pair `i`: `cpu.rope.Pairing`.
    pub const Pairing = cpu.rope.Pairing;
    pub const RopeParams = extern struct { heads: u32, head_stride: u32, dims: u32, position: u32, pairing: u32 };
    /// Rotates the leading `dims` channels of `heads` heads using a (cos, sin)
    /// table with `dims/2` entries per position (`ropeTable`), pairing the
    /// channels split-half or adjacently as `cpu.rope.apply` does.
    pub fn rope(self: *Backend, data: Buffer, table: Buffer, heads: usize, head_stride: usize, dims: usize, position: usize, pairing: Pairing) !void {
        if (heads == 0 or dims == 0 or dims % 2 != 0 or head_stride < dims or data.len < ((heads - 1) * head_stride + dims) * 4) return error.InvalidShape;
        if (table.len < (position + 1) * (dims / 2) * 8) return error.InvalidShape;
        const p: RopeParams = .{ .heads = @intCast(heads), .head_stride = @intCast(head_stride), .dims = @intCast(dims), .position = @intCast(position), .pairing = @intFromEnum(pairing) };
        try self.dispatch(.rope, &.{ data, table }, p, perElement(heads * dims / 2), 256, .{});
    }
    pub const RopeRowsParams = extern struct { heads: u32, head_stride: u32, dims: u32, position: u32, rows: u32, row_stride: u32, pairing: u32 };
    /// `rope` over `rows` token rows of `row_stride` floats; row t is rotated
    /// for position `position + t`.
    pub fn ropeRows(self: *Backend, data: Buffer, table: Buffer, heads: usize, head_stride: usize, dims: usize, position: usize, rows: usize, row_stride: usize, pairing: Pairing) !void {
        if (rows == 0 or heads == 0 or dims == 0 or dims % 2 != 0 or head_stride < dims or row_stride < (heads - 1) * head_stride + dims) return error.InvalidShape;
        if (data.len < ((rows - 1) * row_stride + (heads - 1) * head_stride + dims) * 4 or table.len < (position + rows) * (dims / 2) * 8) return error.InvalidShape;
        const p: RopeRowsParams = .{ .heads = @intCast(heads), .head_stride = @intCast(head_stride), .dims = @intCast(dims), .position = @intCast(position), .rows = @intCast(rows), .row_stride = @intCast(row_stride), .pairing = @intFromEnum(pairing) };
        try self.dispatch(.rope_rows, &.{ data, table }, p, perElement(rows * heads * dims / 2), 256, .{});
    }
    /// Fills `table` with F64-computed (cos, sin) pairs for `positions` positions
    /// of the unscaled rotation with `dims` rotary channels: the angles of
    /// `cpu.rope.apply` (the same for either pairing), including its optional
    /// per-pair `factors` (`dims / 2` finite positive divisors; a checkpoint's
    /// 1e30 entries leave a pair in place to F32 precision).
    pub fn ropeTable(table: Buffer, positions: usize, dims: usize, base: f64, factors: ?[]const f32) !void {
        if (dims == 0 or dims % 2 != 0 or table.len < positions * (dims / 2) * 8) return error.InvalidShape;
        const half = dims / 2;
        if (factors) |f| {
            if (f.len != half) return error.InvalidShape;
            for (f) |x| if (!std.math.isFinite(x) or x <= 0) return error.InvalidShape;
        }
        const out = table.floats();
        for (0..positions) |position| for (0..half) |i| {
            const factor: f64 = if (factors) |f| f[i] else 1;
            const theta = @as(f64, @floatFromInt(position)) * std.math.pow(f64, base, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half))) / factor;
            out[(position * half + i) * 2] = @floatCast(@cos(theta));
            out[(position * half + i) * 2 + 1] = @floatCast(@sin(theta));
        };
    }
    pub const CountParams = extern struct { count: u32 };
    pub fn add(self: *Backend, x: Buffer, y: Buffer, count: usize) !void {
        if (count == 0 or x.len < count * 4 or y.len < count * 4) return error.InvalidShape;
        try self.dispatch(.add, &.{ x, y }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
    }
    /// dst[0..count] = src[0..count]; the ranges must not overlap.
    pub fn copy(self: *Backend, dst: Buffer, src: Buffer, count: usize) !void {
        if (count == 0 or dst.len < count * 4 or src.len < count * 4) return error.InvalidShape;
        if (dst.id == src.id and dst.offset < src.offset + count * 4 and src.offset < dst.offset + count * 4) return error.InvalidShape;
        try self.dispatch(.copy, &.{ dst, src }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
    }
    pub fn siluMul(self: *Backend, gate: Buffer, up: Buffer, count: usize) !void {
        if (count == 0 or gate.len < count * 4 or up.len < count * 4) return error.InvalidShape;
        try self.dispatch(.silu_mul, &.{ gate, up }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
    }
    pub fn silu(self: *Backend, x: Buffer, count: usize) !void {
        if (count == 0 or x.len < count * 4) return error.InvalidShape;
        try self.dispatch(.silu_inplace, &.{x}, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
    }
    /// gate[i] = gelu(gate[i]) * up[i], the tanh GELU of `cpu.gelu`.
    pub fn geluMul(self: *Backend, gate: Buffer, up: Buffer, count: usize) !void {
        if (count == 0 or gate.len < count * 4 or up.len < count * 4) return error.InvalidShape;
        try self.dispatch(.gelu_mul, &.{ gate, up }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
    }
    pub const ScaleParams = extern struct { count: u32, factor: f32 };
    /// x[i] *= factor; the factor must be finite.
    pub fn scale(self: *Backend, x: Buffer, count: usize, factor: f32) !void {
        if (count == 0 or x.len < count * 4 or !std.math.isFinite(factor)) return error.InvalidShape;
        try self.dispatch(.scale, &.{x}, ScaleParams{ .count = @intCast(count), .factor = factor }, perElement(count), 256, .{});
    }
    /// x[i] = (x[i] + y[i]) * factor: a residual add with a per-layer output
    /// scale, rounded once after the add as the CPU reference does.
    pub fn addScale(self: *Backend, x: Buffer, y: Buffer, count: usize, factor: f32) !void {
        if (count == 0 or x.len < count * 4 or y.len < count * 4 or !std.math.isFinite(factor)) return error.InvalidShape;
        try self.dispatch(.add_scale, &.{ x, y }, ScaleParams{ .count = @intCast(count), .factor = factor }, perElement(count), 256, .{});
    }
    /// x[i] = cap * tanh(x[i] / cap), the final logit soft-cap; cap > 0.
    pub fn softcap(self: *Backend, x: Buffer, count: usize, cap: f32) !void {
        if (count == 0 or x.len < count * 4 or !std.math.isFinite(cap) or cap <= 0) return error.InvalidShape;
        try self.dispatch(.softcap, &.{x}, ScaleParams{ .count = @intCast(count), .factor = cap }, perElement(count), 256, .{});
    }
    pub const DeltaGatesParams = extern struct { count: u32, heads: u32 };
    pub fn deltaGates(self: *Backend, alpha: Buffer, beta: Buffer, a: Buffer, bias: Buffer, heads: usize) !void {
        try self.deltaGatesRows(alpha, beta, a, bias, heads, 1);
    }
    /// `deltaGates` over `rows` consecutive token rows of `heads` gates each.
    pub fn deltaGatesRows(self: *Backend, alpha: Buffer, beta: Buffer, a: Buffer, bias: Buffer, heads: usize, rows: usize) !void {
        const count = heads * rows;
        if (heads == 0 or rows == 0 or alpha.len < count * 4 or beta.len < count * 4 or a.len < heads * 4 or bias.len < heads * 4) return error.InvalidShape;
        try self.dispatch(.delta_gates, &.{ alpha, beta, a, bias }, DeltaGatesParams{ .count = @intCast(count), .heads = @intCast(heads) }, perElement(count), 256, .{});
    }
    pub const GateParams = extern struct { heads: u32, width: u32, gate_stride: u32, gate_offset: u32 };
    pub fn sigmoidGate(self: *Backend, out: Buffer, gates: Buffer, heads: usize, width: usize, gate_stride: usize, gate_offset: usize) !void {
        if (heads == 0 or width == 0 or out.len < heads * width * 4 or gate_stride < gate_offset + width or gates.len < ((heads - 1) * gate_stride + gate_offset + width) * 4) return error.InvalidShape;
        const p: GateParams = .{ .heads = @intCast(heads), .width = @intCast(width), .gate_stride = @intCast(gate_stride), .gate_offset = @intCast(gate_offset) };
        try self.dispatch(.sigmoid_gate, &.{ out, gates }, p, perElement(heads * width), 256, .{});
    }
    pub const DeltaParams = extern struct { qheads: u32, vheads: u32, keys: u32, values: u32, scale: f32 };
    pub const DeltaShape = struct { qheads: usize, vheads: usize, keys: usize, values: usize, scale: f32 };
    /// Nontransactional in-place state update: a failed command buffer leaves
    /// partially updated state; the owning session must be reset.
    pub fn delta(self: *Backend, state: Buffer, qkv: Buffer, decay_log: Buffer, beta: Buffer, output: Buffer, s: DeltaShape) !void {
        if (s.qheads == 0 or s.vheads == 0 or s.keys == 0 or s.values == 0 or s.vheads % s.qheads != 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
        if (state.len < s.vheads * s.values * s.keys * 4 or qkv.len < (2 * s.qheads * s.keys + s.vheads * s.values) * 4 or decay_log.len < s.vheads * 4 or beta.len < s.vheads * 4 or output.len < s.vheads * s.values * 4) return error.InvalidShape;
        const p: DeltaParams = .{ .qheads = @intCast(s.qheads), .vheads = @intCast(s.vheads), .keys = @intCast(s.keys), .values = @intCast(s.values), .scale = s.scale };
        try self.dispatch(.delta, &.{ state, qkv, decay_log, beta, output }, p, @intCast(s.vheads * s.values), 32, .{});
    }
    pub const DeltaChunkParams = extern struct { qheads: u32, vheads: u32, keys: u32, values: u32, count: u32, in_stride: u32, gate_stride: u32, out_stride: u32, row_states: u32, row_stride: u32, scale: f32 };
    /// `count` tokens of one layer: `qkv` rows `[token][in_stride]` laid out
    /// as `nu_delta`'s input (q, k, then v per head), `decay_log`/`beta` rows
    /// `[token][gate_stride]`, `output` rows `[token][out_stride]` at
    /// `head * values`. State `[head][value][key]` is updated in place. With
    /// `row_states > 0` (a verify batch, at most one sub-chunk) each row's
    /// state is also written to `slots`, `row_stride` floats apart.
    pub const DeltaChunkShape = struct { qheads: usize, vheads: usize, keys: usize, values: usize, count: usize, in_stride: usize, gate_stride: usize, out_stride: usize, row_states: usize = 0, row_stride: usize = 0, scale: f32 };
    /// Rows the `qkv` buffer must hold for `count` tokens (the kernel walks
    /// 32-token sub-chunks; rows past `count` are masked, never read).
    pub fn deltaChunkRows(count: usize) usize {
        return (count + 31) / 32 * 32;
    }
    /// Chunkwise DeltaNet: one dispatch per layer for a prefill chunk,
    /// threadgroup per (value head, 32 value rows), 32-token sub-chunks.
    /// Nontransactional like `delta`: a failed command buffer leaves partial
    /// state and the session must be reset.
    pub fn deltaChunk(self: *Backend, state: Buffer, qkv: Buffer, decay_log: Buffer, beta: Buffer, output: Buffer, slots: Buffer, s: DeltaChunkShape) !void {
        if (s.qheads == 0 or s.vheads == 0 or s.keys == 0 or s.values == 0 or s.vheads % s.qheads != 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
        if (s.keys % 8 != 0 or s.values % 32 != 0 or s.count == 0 or s.count > 4096) return error.InvalidShape;
        const qkv_row = 2 * s.qheads * s.keys + s.vheads * s.values;
        if (s.in_stride < qkv_row or s.gate_stride < s.vheads or s.out_stride < s.vheads * s.values) return error.InvalidShape;
        const rows = deltaChunkRows(s.count);
        if (state.len < s.vheads * s.values * s.keys * 4) return error.InvalidShape;
        if (qkv.len < rows * s.in_stride * 4 or decay_log.len < s.count * s.gate_stride * 4 or beta.len < s.count * s.gate_stride * 4 or output.len < s.count * s.out_stride * 4) return error.InvalidShape;
        // Row slots only for a single sub-chunk and a matrix-sized stride.
        if (s.row_states > 0) {
            if (s.row_states > s.count or s.row_states > 8 or s.row_stride < s.vheads * s.values * s.keys) return error.InvalidShape;
            if (slots.len < ((s.row_states - 1) * s.row_stride + s.vheads * s.values * s.keys) * 4) return error.InvalidShape;
        }
        const p: DeltaChunkParams = .{ .qheads = @intCast(s.qheads), .vheads = @intCast(s.vheads), .keys = @intCast(s.keys), .values = @intCast(s.values), .count = @intCast(s.count), .in_stride = @intCast(s.in_stride), .gate_stride = @intCast(s.gate_stride), .out_stride = @intCast(s.out_stride), .row_states = @intCast(s.row_states), .row_stride = @intCast(s.row_stride), .scale = s.scale };
        try self.dispatch(.delta_chunk, &.{ state, qkv, decay_log, beta, output, slots }, p, @intCast(s.vheads * (s.values / 32)), 128, .{});
    }
    pub const ConvParams = extern struct { channels: u32, taps: u32 };
    pub fn convolution(self: *Backend, history: Buffer, input: Buffer, weights: Buffer, output: Buffer, channels: usize, taps: usize) !void {
        if (channels == 0 or taps < 2 or taps > 32 or history.len < channels * (taps - 1) * 4 or input.len < channels * 4 or weights.len < channels * taps * 4 or output.len < channels * 4) return error.InvalidShape;
        try self.dispatch(.convolution, &.{ history, input, weights, output }, ConvParams{ .channels = @intCast(channels), .taps = @intCast(taps) }, perElement(channels), 256, .{});
    }
    pub const ConvRowsParams = extern struct { channels: u32, taps: u32, rows: u32, stride: u32, row_states: u32, row_stride: u32 };
    /// Per-row history slots: slot `r` (`stride` floats apart from `base`)
    /// receives the history after the chunk's first `r + 1` rows.
    pub const RowSlots = struct { base: Buffer, states: usize = 0, stride: usize = 0 };
    /// Causal convolution over `rows` token rows (stride `stride`) reading the
    /// pre-chunk inputs from `history`; bit-identical to `rows` sequential
    /// `convolution` calls. Does not touch `history`: call
    /// `convolutionHistory` afterwards.
    pub fn convolutionRows(self: *Backend, history: Buffer, input: Buffer, weights: Buffer, output: Buffer, channels: usize, taps: usize, rows: usize, stride: usize) !void {
        if (channels == 0 or taps < 2 or taps > 32 or rows == 0 or stride < channels) return error.InvalidShape;
        if (history.len < channels * (taps - 1) * 4 or input.len < ((rows - 1) * stride + channels) * 4 or weights.len < channels * taps * 4 or output.len < ((rows - 1) * stride + channels) * 4) return error.InvalidShape;
        const p: ConvRowsParams = .{ .channels = @intCast(channels), .taps = @intCast(taps), .rows = @intCast(rows), .stride = @intCast(stride), .row_states = 0, .row_stride = 0 };
        try self.dispatch(.convolution_rows, &.{ history, input, weights, output }, p, perElement(rows * channels), 256, .{});
    }
    /// Shifts the last `taps - 1` inputs of the chunk into `history`. With
    /// `slots.states > 0` (a verify batch, at most one sub-chunk) each row's
    /// history is also written to its slot.
    pub fn convolutionHistory(self: *Backend, history: Buffer, input: Buffer, slots: RowSlots, channels: usize, taps: usize, rows: usize, stride: usize) !void {
        if (channels == 0 or taps < 2 or taps > 32 or rows == 0 or stride < channels) return error.InvalidShape;
        if (history.len < channels * (taps - 1) * 4 or input.len < ((rows - 1) * stride + channels) * 4) return error.InvalidShape;
        if (slots.states > 0) {
            if (slots.states > rows or slots.states > 8 or slots.stride < channels * (taps - 1)) return error.InvalidShape;
            if (slots.base.len < ((slots.states - 1) * slots.stride + channels * (taps - 1)) * 4) return error.InvalidShape;
        }
        const p: ConvRowsParams = .{ .channels = @intCast(channels), .taps = @intCast(taps), .rows = @intCast(rows), .stride = @intCast(stride), .row_states = @intCast(slots.states), .row_stride = @intCast(slots.stride) };
        try self.dispatch(.convolution_history, &.{ history, input, slots.base }, p, perElement(channels), 256, .{});
    }
    pub const AttentionParams = extern struct { query_heads: u32, kv_heads: u32, key_width: u32, value_width: u32, visible: u32, scale: f32 };
    /// `precision` is how `keys`/`values` are stored; queries, scores,
    /// and the output are always F32.
    pub const AttentionShape = struct { query_heads: usize, kv_heads: usize, key_width: usize, value_width: usize, visible: usize, scale: f32, precision: Precision = .f32 };
    /// Three dispatches; `scores` needs query_heads * visible floats.
    pub fn attention(self: *Backend, keys: Buffer, values: Buffer, queries: Buffer, scores: Buffer, output: Buffer, s: AttentionShape) !void {
        if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or s.key_width == 0 or s.value_width == 0 or s.visible == 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
        const elem = s.precision.size();
        if (keys.len < s.visible * s.kv_heads * s.key_width * elem or values.len < s.visible * s.kv_heads * s.value_width * elem or queries.len < s.query_heads * s.key_width * 4 or scores.len < s.query_heads * s.visible * 4 or output.len < s.query_heads * s.value_width * 4) return error.InvalidShape;
        if (keys.offset % elem != 0 or values.offset % elem != 0) return error.InvalidShape;
        const p: AttentionParams = .{ .query_heads = @intCast(s.query_heads), .kv_heads = @intCast(s.kv_heads), .key_width = @intCast(s.key_width), .value_width = @intCast(s.value_width), .visible = @intCast(s.visible), .scale = s.scale };
        const half = s.precision == .f16;
        try self.dispatch(if (half) .attention_scores_h else .attention_scores, &.{ keys, queries, scores }, p, @intCast(s.query_heads * s.visible), 32, .{});
        try self.dispatch(.attention_softmax, &.{scores}, p, @intCast(s.query_heads), 32, .{});
        try self.dispatch(if (half) .attention_values_h else .attention_values, &.{ values, scores, output }, p, @intCast(s.query_heads * s.value_width), 32, .{});
    }
    pub const AttentionDecodeParams = extern struct { query_heads: u32, kv_heads: u32, key_width: u32, value_width: u32, visible: u32, splits: u32, scale: f32 };
    /// Most splits of the visible range one `attentionDecode` uses, and the
    /// rows each split holds before another split is added; must match the
    /// partial layout the kernels write.
    pub const decode_splits_max = 64;
    pub const decode_split_rows = 256;
    pub fn attentionDecodeSplits(visible: usize) usize {
        return @min(decode_splits_max, (visible + decode_split_rows - 1) / decode_split_rows);
    }
    /// Floats the partial buffer must hold: `[query_heads][splits][2 + value_width]`.
    pub fn attentionDecodePartials(query_heads: usize, value_width: usize) usize {
        return query_heads * decode_splits_max * (2 + value_width);
    }
    /// Query heads one threadgroup of the split pass covers: 8 in the
    /// instantiation for widths up to 256, 4 in the wide one (widths up to
    /// 512); a KV head with more query heads takes several threadgroups
    /// per split, each reading the slice once.
    pub fn attentionDecodeHeadGroups(query_heads: usize, kv_heads: usize, key_width: usize, value_width: usize) usize {
        const per_group: usize = if (key_width > 256 or value_width > 256) 4 else 8;
        const g = query_heads / kv_heads;
        return (g + per_group - 1) / per_group;
    }
    /// Flash-decoding attention: two dispatches (split pass, merge)
    /// and no score buffer; `partials` needs `attentionDecodePartials`
    /// floats. Same shape and precision contract as `attention`, plus
    /// widths of at most 512 (above 256 through the wide instantiation).
    pub fn attentionDecode(self: *Backend, keys: Buffer, values: Buffer, queries: Buffer, partials: Buffer, output: Buffer, s: AttentionShape) !void {
        if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or s.key_width == 0 or s.key_width > 512 or s.value_width == 0 or s.value_width > 512 or s.visible == 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
        const elem = s.precision.size();
        const splits = attentionDecodeSplits(s.visible);
        const wide = s.key_width > 256 or s.value_width > 256;
        const head_groups = attentionDecodeHeadGroups(s.query_heads, s.kv_heads, s.key_width, s.value_width);
        if (keys.len < s.visible * s.kv_heads * s.key_width * elem or values.len < s.visible * s.kv_heads * s.value_width * elem or queries.len < s.query_heads * s.key_width * 4 or partials.len < s.query_heads * splits * (2 + s.value_width) * 4 or output.len < s.query_heads * s.value_width * 4) return error.InvalidShape;
        if (keys.offset % elem != 0 or values.offset % elem != 0) return error.InvalidShape;
        const p: AttentionDecodeParams = .{ .query_heads = @intCast(s.query_heads), .kv_heads = @intCast(s.kv_heads), .key_width = @intCast(s.key_width), .value_width = @intCast(s.value_width), .visible = @intCast(s.visible), .splits = @intCast(splits), .scale = s.scale };
        // Cache traffic per dispatch, for the profile: every row once.
        const bytes: u64 = @intCast(s.visible * s.kv_heads * (s.key_width + s.value_width) * elem);
        const kernel: Kernel = if (wide) (if (s.precision == .f16) .attention_decode_wh else .attention_decode_w) else (if (s.precision == .f16) .attention_decode_h else .attention_decode);
        try self.dispatch(kernel, &.{ keys, values, queries, partials }, p, @intCast(s.kv_heads * head_groups * splits), 128, .{ .bytes = bytes });
        try self.dispatch(.attention_merge, &.{ partials, output }, p, @intCast(s.query_heads), 256, .{});
    }
    pub const PackParams = extern struct { count0: u32, count1: u32 };
    /// One F32 → F16 conversion: `count` floats of `src` into `dst`.
    pub const Pack = struct { dst: Buffer, src: Buffer, count: usize };
    /// Rounds one or two F32 vectors to F16 in one dispatch (round to
    /// nearest even, as `@floatCast`): the attention projections stay F32
    /// and this writes the F16 cache slot. Destinations must not overlap
    /// their sources.
    pub fn packHalf(self: *Backend, pairs: []const Pack) !void {
        if (pairs.len == 0 or pairs.len > 2) return error.InvalidShape;
        var total: usize = 0;
        for (pairs) |pair| {
            if (pair.count == 0 or pair.dst.len < pair.count * 2 or pair.src.len < pair.count * 4 or pair.dst.offset % 2 != 0 or pair.src.offset % 4 != 0) return error.InvalidShape;
            if (overlaps(pair.dst, pair.count * 2, pair.src, pair.count * 4)) return error.InvalidShape;
            total += pair.count;
        }
        const second = if (pairs.len == 2) pairs[1] else pairs[0];
        const p: PackParams = .{ .count0 = @intCast(pairs[0].count), .count1 = if (pairs.len == 2) @intCast(pairs[1].count) else 0 };
        try self.dispatch(.pack_half, &.{ pairs[0].dst, pairs[0].src, second.dst, second.src }, p, perElement(total), 256, .{});
    }
    pub const AttentionChunkParams = extern struct { query_heads: u32, kv_heads: u32, key_width: u32, value_width: u32, position: u32, count: u32, q_stride: u32, out_stride: u32, scale: f32, window: u32 };
    /// `count` query rows at cache positions `position..position + count`,
    /// each attending causally over the cache rows `[0, position + t]`, or
    /// `[position + t + 1 - window, position + t]` when `window` is nonzero
    /// (a sliding window; the caller may slice the cache so that row 0
    /// is the earliest key any row of the chunk can see).
    /// `queries` holds `[row][q_stride]` with the row's heads at
    /// `head * key_width`; `output` is written `[row][out_stride]` likewise.
    /// `precision` is how the cache **and the queries** are stored:
    /// the `f16` instantiation multiplies half operands into F32
    /// accumulators, so the caller packs its queries to half first
    /// (`packHalf`); the output is F32 either way.
    pub const AttentionChunkShape = struct { query_heads: usize, kv_heads: usize, key_width: usize, value_width: usize, position: usize, count: usize, q_stride: usize, out_stride: usize, scale: f32, precision: Precision = .f32, window: usize = 0 };
    /// Rows the query and output buffers must hold for `count` rows: the
    /// last SIMD-group tile computes and stores on the caller's padding.
    pub fn attentionChunkRows(count: usize) usize {
        return (count + 7) / 8 * 8;
    }
    /// One dispatch per layer for a prefill chunk: threadgroup per
    /// (query head, 32-query tile, 256 value columns), no score buffer.
    /// Reads at most `position + count` cache rows. Value widths above 256
    /// (up to 512) take one threadgroup per 256 columns.
    pub fn attentionChunk(self: *Backend, keys: Buffer, values: Buffer, queries: Buffer, output: Buffer, s: AttentionChunkShape) !void {
        if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
        if (s.key_width == 0 or s.key_width % 8 != 0 or s.value_width == 0 or s.value_width % 8 != 0 or s.value_width > 512) return error.InvalidShape;
        if (s.count == 0 or s.count > 4096 or s.position > 32768 - s.count) return error.InvalidShape;
        if (s.q_stride < s.query_heads * s.key_width or s.out_stride < s.query_heads * s.value_width) return error.InvalidShape;
        const total = s.position + s.count;
        const rows = attentionChunkRows(s.count);
        const elem = s.precision.size();
        if (keys.len < total * s.kv_heads * s.key_width * elem or values.len < total * s.kv_heads * s.value_width * elem) return error.InvalidShape;
        if (queries.len < rows * s.q_stride * elem or output.len < rows * s.out_stride * 4) return error.InvalidShape;
        if (keys.offset % elem != 0 or values.offset % elem != 0 or queries.offset % elem != 0) return error.InvalidShape;
        const p: AttentionChunkParams = .{ .query_heads = @intCast(s.query_heads), .kv_heads = @intCast(s.kv_heads), .key_width = @intCast(s.key_width), .value_width = @intCast(s.value_width), .position = @intCast(s.position), .count = @intCast(s.count), .q_stride = @intCast(s.q_stride), .out_stride = @intCast(s.out_stride), .scale = s.scale, .window = @intCast(s.window) };
        const tiles = (s.count + 31) / 32;
        const value_splits = (s.value_width + 255) / 256;
        try self.dispatch(if (s.precision == .f16) .attention_chunk_h else .attention_chunk, &.{ keys, values, queries, output }, p, @intCast(s.query_heads * tiles * value_splits), 128, .{});
    }
    pub const TopKParams = extern struct { count: u32, partials: u32, k: u32, temperature: f32 };
    pub const topk_partials = 64;
    /// Values each partial-pass thread keeps in registers (`NU_TOPK_LOCAL`);
    /// bounds the vocabulary size the kernels accept.
    const topk_local = 16;
    pub const topk_max_count = topk_partials * 256 * topk_local;
    /// Scratch and result buffers for `topk`, sized by `topkBuffers`.
    pub const TopKBuffers = struct {
        partial_values: Buffer, // partials * k floats
        partial_indices: Buffer, // partials * k u32
        values: Buffer, // k floats, sorted (value desc, index asc)
        indices: Buffer, // k u32
        sums: Buffer, // partials floats: F32 partial sums of exp((l - max) / T)
        flags: Buffer, // partials u32: nonzero when the partition saw a non-finite logit
    };
    pub fn topkBuffers(self: *Backend, k: usize) !TopKBuffers {
        return .{
            .partial_values = try self.create(topk_partials * k * 4),
            .partial_indices = try self.create(topk_partials * k * 4),
            .values = try self.create(k * 4),
            .indices = try self.create(k * 4),
            .sums = try self.create(topk_partials * 4),
            .flags = try self.create(topk_partials * 4),
        };
    }
    /// The `k` best of `count` logits by (value desc, index asc) into
    /// `scratch.values`/`indices`, and Σ exp((l − max) / temperature) as
    /// `topk_partials` F32 partial sums with per-partition non-finite flags.
    /// The CPU adds the partials in F64 (see `sampling.TopK`). Three dispatches.
    pub fn topk(self: *Backend, logits: Buffer, count: usize, k: usize, temperature: f32, scratch: TopKBuffers) !void {
        if (!enabled) return error.MetalNotEnabled;
        if (count == 0 or count > topk_max_count or k == 0 or k > 256 or logits.len < count * 4) return error.InvalidShape;
        if (!std.math.isFinite(temperature) or temperature <= 0) return error.InvalidShape;
        if (scratch.partial_values.len < topk_partials * k * 4 or scratch.partial_indices.len < topk_partials * k * 4 or
            scratch.values.len < k * 4 or scratch.indices.len < k * 4 or scratch.sums.len < topk_partials * 4 or scratch.flags.len < topk_partials * 4) return error.InvalidShape;
        const p: TopKParams = .{ .count = @intCast(count), .partials = topk_partials, .k = @intCast(k), .temperature = temperature };
        try self.dispatch(.topk_partial, &.{ logits, scratch.partial_values, scratch.partial_indices }, p, topk_partials, 256, .{});
        try self.dispatch(.topk_final, &.{ scratch.partial_values, scratch.partial_indices, scratch.values, scratch.indices }, p, 1, 256, .{});
        try self.dispatch(.expsum_partial, &.{ logits, scratch.values, scratch.sums, scratch.flags }, p, topk_partials, 256, .{});
    }

    pub const PenalizeParams = extern struct { count: u32, history_words: u32, repetition: f32, presence: f32 };
    /// Applies the history penalties to `logits[0..count]` in place:
    /// `l / r` for positive and `l · r` for negative logits of every token in
    /// `history` (bit `id` in word `id / 32`), then `l − presence`. One thread
    /// per logit; `repetition` must be positive and both values finite. The
    /// readback taken after this is the penalized logits, which is what the
    /// CPU sampler sorts (`sampling.Sampler.penalize`).
    pub fn penalize(self: *Backend, logits: Buffer, count: usize, history: Buffer, repetition: f32, presence: f32) !void {
        if (count == 0 or count > std.math.maxInt(u32) or logits.len < count * 4) return error.InvalidShape;
        if (!std.math.isFinite(repetition) or repetition <= 0 or !std.math.isFinite(presence)) return error.InvalidShape;
        const words = (count + 31) / 32;
        if (history.len < words * 4) return error.InvalidShape;
        const p: PenalizeParams = .{ .count = @intCast(count), .history_words = @intCast(words), .repetition = repetition, .presence = presence };
        try self.dispatch(.penalize, &.{ logits, history }, p, perElement(count), 256, .{});
    }

    pub const ArgmaxParams = extern struct { count: u32, partials: u32 };
    pub const argmax_partials = 64;
    /// Two-pass greedy selection with lowest-index tie breaking. `values` and
    /// `indices` hold `argmax_partials` scratch entries; `result` receives one u32.
    pub fn argmax(self: *Backend, logits: Buffer, count: usize, values: Buffer, indices: Buffer, result: Buffer) !void {
        if (count == 0 or logits.len < count * 4 or values.len < argmax_partials * 4 or indices.len < argmax_partials * 4 or result.len < 4) return error.InvalidShape;
        const p: ArgmaxParams = .{ .count = @intCast(count), .partials = argmax_partials };
        try self.dispatch(.argmax_partial, &.{ logits, values, indices }, p, argmax_partials, 256, .{});
        try self.dispatch(.argmax_final, &.{ values, indices, result }, p, 1, 32, .{});
    }
};
