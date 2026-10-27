//! Qwen3-TTS's codec decoder on Metal: the CPU reference's pipeline
//! (qwen3_codec.zig) as one command buffer per piece of at most
//! `max_frames` frames. `Decoder` holds the weights, copied from a CPU
//! `codec.Runtime` (the same tap-major convolution matrices, columns padded
//! with zeros to the matmul tile's 64), and the scratch; `Stream` holds one
//! clip's carried state (every convolution's history, the attention
//! window's keys and values) on the device. Every kernel sums an output in
//! an order fixed by the output's own index (the generic F32 matmul tile,
//! `tts.metal`'s window attention), so pieces of any size give one decode's
//! samples bit for bit. The decoder borrows its `Backend`, which must
//! outlive it; its buffers are released on `deinit`.
//! docs/models/qwen3-tts.md § The codec decoder.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const codec = @import("qwen3_codec.zig");

const Allocator = std.mem.Allocator;
const Backend = metal.Backend;
const Buffer = metal.Buffer;
const Frame = codec.Frame;
const groups = codec.groups;

/// Frames one command buffer decodes; `Stream.push` splits a longer push,
/// which changes no sample (the decode does not depend on the pieces).
pub const max_frames = 8;
/// RoPE positions the transformer's table covers: a stream's frames at most.
pub const max_positions = 16384;
const residual_kernel = 7;
const dwconv_kernel = 7;
const layer_norm_eps = 1e-6;

/// A matrix in a created buffer, `columns` padded to a multiple of 64.
const Mat = struct { buffer: Buffer, matrix: cpu.Matrix, bias: ?Buffer };
/// A causal convolution (`taps` 1 with `dilation` 1 is a plain matrix).
const Conv = struct { mat: Mat, in: usize, taps: usize, dilation: usize };
/// A transposed convolution: `stride` output rows per input row.
const Trans = struct { mat: Mat, in: usize, out: usize, stride: usize, taps: usize };
const Snake = struct { scale: Buffer, inv: Buffer };
const Layer = struct { input_norm: Buffer, post_norm: Buffer, q: Mat, k: Mat, v: Mat, o: Mat, gate: Mat, up: Mat, down: Mat, attn_scale: Buffer, mlp_scale: Buffer };
const NeXt = struct { dw: Buffer, dw_b: Buffer, ln_w: Buffer, ln_b: Buffer, gamma: Buffer, pw1: Mat, pw2: Mat };
const Up = struct { trans: Trans, next: NeXt };
const Unit = struct { act1: Snake, conv1: Conv, act2: Snake, conv2: Conv };
const Block = struct { act: Snake, trans: Trans, units: [3]Unit };

fn padded(columns: usize) usize {
    return (columns + 63) / 64 * 64;
}
fn tile(rows: usize) usize {
    return Backend.matmulPadded(rows);
}

/// The decoder's weights and scratch on the device.
pub const Decoder = struct {
    gpa: Allocator,
    backend: *Backend,
    config: codec.Config,
    buffers: std.ArrayList(Buffer) = .empty,
    codebooks: Buffer,
    project: [2]Mat,
    pre_conv: Conv,
    input: Mat,
    layers: []Layer,
    norm: Buffer,
    output: Mat,
    upsample: [2]Up,
    conv_in: Conv,
    blocks: [4]Block,
    act_out: Snake,
    out_w: Buffer,
    out_b: Buffer,
    rope: Buffer,
    // Scratch, sized for `max_frames`.
    codes: Buffer,
    first: Buffer,
    rest: Buffer,
    /// The pipeline's activations, four buffers of `act_floats` each.
    act: [4]Buffer,
    gathered: Buffer,
    joined: Buffer,
    tx: Buffer,
    th: Buffer,
    tq: Buffer,
    tk: Buffer,
    tv: Buffer,
    tatt: Buffer,
    to: Buffer,
    tgate: Buffer,
    tup: Buffer,
    jk: Buffer,
    jv: Buffer,
    samples: Buffer,
    /// Bytes of device memory created.
    bytes: usize = 0,

    /// Copies `runtime`'s weights to the device; `runtime` may be released
    /// afterwards. `backend` must outlive the decoder.
    pub fn init(gpa: Allocator, backend: *Backend, runtime: *const codec.Runtime) !Decoder {
        const c = runtime.config;
        if (c.head_dim != 64 or c.window > Backend.tts_window_max) return error.UnsupportedConfiguration;
        var self: Decoder = undefined;
        self.gpa = gpa;
        self.backend = backend;
        self.config = c;
        self.buffers = .empty;
        self.bytes = 0;
        errdefer self.releaseAll();
        self.codebooks = try self.upload(runtime.codebooks);
        for (&self.project, runtime.project) |*p, l| p.* = try self.linear(l.w, null, l.out, l.in);
        self.pre_conv = try self.causal(runtime.pre_conv.w, runtime.pre_conv.b, runtime.pre_conv.out, runtime.pre_conv.in, runtime.pre_conv.kernel, runtime.pre_conv.dilation);
        self.input = try self.linear(runtime.input.w, runtime.input.b, runtime.input.out, runtime.input.in);
        self.layers = try gpa.alloc(Layer, runtime.layers.len);
        errdefer gpa.free(self.layers);
        for (self.layers, runtime.layers) |*out, l| out.* = .{
            .input_norm = try self.upload(l.input_norm),
            .post_norm = try self.upload(l.post_norm),
            .q = try self.linear(l.q.w, null, l.q.out, l.q.in),
            .k = try self.linear(l.k.w, null, l.k.out, l.k.in),
            .v = try self.linear(l.v.w, null, l.v.out, l.v.in),
            .o = try self.linear(l.o.w, null, l.o.out, l.o.in),
            .gate = try self.linear(l.gate.w, null, l.gate.out, l.gate.in),
            .up = try self.linear(l.up.w, null, l.up.out, l.up.in),
            .down = try self.linear(l.down.w, null, l.down.out, l.down.in),
            .attn_scale = try self.upload(l.attn_scale),
            .mlp_scale = try self.upload(l.mlp_scale),
        };
        self.norm = try self.upload(runtime.norm);
        self.output = try self.linear(runtime.output.w, runtime.output.b, runtime.output.out, runtime.output.in);
        for (&self.upsample, runtime.upsample) |*u, s| u.* = .{
            .trans = try self.transposed(s.trans.w, s.trans.b, s.trans.in, s.trans.out, s.trans.stride, s.trans.taps),
            .next = .{
                .dw = try self.upload(s.next.dw),
                .dw_b = try self.upload(s.next.dw_b),
                .ln_w = try self.upload(s.next.ln_w),
                .ln_b = try self.upload(s.next.ln_b),
                .gamma = try self.upload(s.next.gamma),
                .pw1 = try self.linear(s.next.pw1.w, s.next.pw1.b, s.next.pw1.out, s.next.pw1.in),
                .pw2 = try self.linear(s.next.pw2.w, s.next.pw2.b, s.next.pw2.out, s.next.pw2.in),
            },
        };
        const ci = runtime.conv_in;
        self.conv_in = try self.causal(ci.w, ci.b, ci.out, ci.in, ci.kernel, ci.dilation);
        for (&self.blocks, runtime.blocks) |*out, blk| {
            out.act = try self.snake(blk.act);
            out.trans = try self.transposed(blk.trans.w, blk.trans.b, blk.trans.in, blk.trans.out, blk.trans.stride, blk.trans.taps);
            for (&out.units, blk.units) |*u, unit| u.* = .{
                .act1 = try self.snake(unit.act1),
                .conv1 = try self.causal(unit.conv1.w, unit.conv1.b, unit.conv1.out, unit.conv1.in, unit.conv1.kernel, unit.conv1.dilation),
                .act2 = try self.snake(unit.act2),
                .conv2 = try self.causal(unit.conv2.w, unit.conv2.b, unit.conv2.out, unit.conv2.in, unit.conv2.kernel, unit.conv2.dilation),
            };
        }
        self.act_out = try self.snake(runtime.act_out);
        if (runtime.conv_out.out != 1) return error.UnsupportedConfiguration;
        self.out_w = try self.upload(runtime.conv_out.w);
        self.out_b = try self.upload(runtime.conv_out.b);
        self.rope = try self.floats(max_positions * c.head_dim);
        try Backend.ropeTable(self.rope, max_positions, c.head_dim, c.rope_theta, null);

        // Scratch: every stage's rows at `max_frames`, padded for the tile.
        const n = max_frames;
        const width = c.heads * c.head_dim;
        self.codes = try self.floats(n * groups);
        self.first = try self.floats(tile(n) * c.codebook_dim);
        self.rest = try self.floats(tile(n) * c.codebook_dim);
        var act: usize = tile(n) * @max(c.latent, 2 * c.quantized);
        var gathered: usize = tile(n) * padded(self.pre_conv.taps * self.pre_conv.in);
        var joined: usize = (n + 2) * c.quantized;
        var rows: usize = n;
        for (self.upsample) |u| {
            act = @max(act, tile(rows) * u.trans.mat.matrix.rows);
            rows *= u.trans.stride;
            act = @max(act, tile(rows) * u.next.pw1.matrix.rows);
            joined = @max(joined, (rows + dwconv_kernel - 1) * c.latent);
        }
        act = @max(act, tile(rows) * c.decoder_dim);
        gathered = @max(gathered, tile(rows) * self.conv_in.mat.matrix.columns);
        joined = @max(joined, (rows + (residual_kernel - 1)) * c.latent);
        for (self.blocks) |blk| {
            gathered = @max(gathered, tile(rows) * blk.trans.mat.matrix.columns);
            joined = @max(joined, (rows + blk.trans.taps - 1) * blk.trans.in);
            act = @max(act, tile(rows) * @max(blk.trans.in, blk.trans.mat.matrix.rows));
            rows *= blk.trans.stride;
            act = @max(act, tile(rows) * @max(blk.trans.out, blk.units[0].conv2.mat.matrix.columns));
            for (blk.units) |u| {
                gathered = @max(gathered, tile(rows) * @max(u.conv1.mat.matrix.columns, u.conv2.mat.matrix.columns));
                joined = @max(joined, (rows + (u.conv1.taps - 1) * u.conv1.dilation) * blk.trans.out);
            }
        }
        joined = @max(joined, (rows + residual_kernel - 1) * c.blockChannels(4));
        for (&self.act) |*a| a.* = try self.floats(act);
        self.gathered = try self.floats(gathered);
        self.joined = try self.floats(joined);
        const t = tile(n);
        self.tx = try self.floats(t * c.hidden);
        self.th = try self.floats(t * c.hidden);
        self.tq = try self.floats(t * width);
        self.tk = try self.floats(t * width);
        self.tv = try self.floats(t * width);
        self.tatt = try self.floats(t * width);
        self.to = try self.floats(t * c.hidden);
        self.tgate = try self.floats(t * c.intermediate);
        self.tup = try self.floats(t * c.intermediate);
        self.jk = try self.floats((c.window - 1 + n) * width);
        self.jv = try self.floats((c.window - 1 + n) * width);
        self.samples = try self.floats(n * c.samplesPerFrame());
        return self;
    }

    pub fn deinit(self: *Decoder) void {
        self.releaseAll();
        self.gpa.free(self.layers);
        self.* = undefined;
    }

    fn releaseAll(self: *Decoder) void {
        for (self.buffers.items) |b| self.backend.release(b) catch {};
        self.buffers.deinit(self.gpa);
    }

    fn floats(self: *Decoder, count: usize) !Buffer {
        try self.buffers.ensureUnusedCapacity(self.gpa, 1);
        const b = try self.backend.create(count * 4);
        self.buffers.appendAssumeCapacity(b);
        self.bytes += count * 4;
        return b;
    }
    fn upload(self: *Decoder, values: []const f32) !Buffer {
        const b = try self.floats(values.len);
        @memcpy(b.floats()[0..values.len], values);
        return b;
    }
    /// `w` is `[rows][columns]`, copied with the columns padded to 64.
    fn linear(self: *Decoder, w: []const f32, bias: ?[]const f32, rows: usize, columns: usize) !Mat {
        const stride = padded(columns);
        const b = try self.floats(rows * stride);
        const dst = b.floats();
        for (0..rows) |r| @memcpy(dst[r * stride ..][0..columns], w[r * columns ..][0..columns]);
        return .{
            .buffer = b,
            .matrix = .{ .encoding = 0, .rows = rows, .columns = stride, .bytes = std.mem.sliceAsBytes(dst[0 .. rows * stride]) },
            .bias = if (bias) |v| try self.upload(v) else null,
        };
    }
    fn causal(self: *Decoder, w: []const f32, bias: []const f32, out: usize, in: usize, taps: usize, dilation: usize) !Conv {
        return .{ .mat = try self.linear(w, bias, out, taps * in), .in = in, .taps = taps, .dilation = dilation };
    }
    fn transposed(self: *Decoder, w: []const f32, bias: []const f32, in: usize, out: usize, stride: usize, taps: usize) !Trans {
        return .{ .mat = try self.linear(w, bias, stride * out, taps * in), .in = in, .out = out, .stride = stride, .taps = taps };
    }
    fn snake(self: *Decoder, s: codec.SnakeBeta) !Snake {
        return .{ .scale = try self.upload(s.scale), .inv = try self.upload(s.inv) };
    }

    // ---- recording helpers --------------------------------------------

    /// `output = m · input (+ bias)` over `rows` rows of `in_stride`.
    fn mm(self: *Decoder, m: Mat, input: Buffer, in_stride: usize, output: Buffer, rows: usize) !void {
        const b = self.backend;
        try b.matmulTile(m.buffer, m.matrix, input, in_stride, output, m.matrix.rows, rows);
        if (m.bias) |bias| try b.addBiasRows(output, bias, m.matrix.rows, rows, m.matrix.rows);
    }

    /// `joined = history ‖ x` (`history` has `history_rows` rows of `width`).
    fn join(self: *Decoder, history: ?Buffer, history_rows: usize, x: Buffer, rows: usize, width: usize) !void {
        const b = self.backend;
        if (history_rows > 0) try b.copy(self.joined, history.?, history_rows * width);
        try b.copy(self.joined.slice(history_rows * width * 4, rows * width * 4), x, rows * width);
    }
    /// `history = the last rows of joined` after a `join` of `rows` new rows.
    fn keep(self: *Decoder, history: Buffer, history_rows: usize, rows: usize, width: usize) !void {
        if (history_rows > 0) try self.backend.copy(history, self.joined.slice(rows * width * 4, history_rows * width * 4), history_rows * width);
    }

    /// A causal convolution of `rows` rows of `x` into `y` through `history`.
    fn convolve(self: *Decoder, conv: Conv, history: ?Buffer, x: Buffer, rows: usize, y: Buffer) !void {
        const b = self.backend;
        const h = (conv.taps - 1) * conv.dilation;
        const columns = conv.mat.matrix.columns;
        if (h == 0 and columns == conv.in) return self.mm(conv.mat, x, conv.in, y, rows);
        var source = x;
        if (h > 0) {
            try self.join(history, h, x, rows, conv.in);
            source = self.joined;
        }
        try b.ttsGather(source, self.gathered, rows, conv.in, conv.taps, columns, 0, @intCast(conv.dilation));
        try self.mm(conv.mat, self.gathered, columns, y, rows);
        if (h > 0) try self.keep(history.?, h, rows, conv.in);
    }

    /// A transposed convolution of `rows` input rows into `rows · stride` rows of `y`.
    fn transpose(self: *Decoder, t: Trans, history: ?Buffer, x: Buffer, rows: usize, y: Buffer) !void {
        const h = t.taps - 1;
        if (h == 0 and t.mat.matrix.columns == t.in) return self.mm(t.mat, x, t.in, y, rows);
        try self.join(history, h, x, rows, t.in);
        try self.backend.ttsGather(self.joined, self.gathered, rows, t.in, t.taps, t.mat.matrix.columns, h, -1);
        try self.mm(t.mat, self.gathered, t.mat.matrix.columns, y, rows);
        try self.keep(history.?, h, rows, t.in);
    }
};

/// One clip's decode: push frames as they come. Borrows the decoder.
pub const Stream = struct {
    decoder: *Decoder,
    /// Frames pushed so far: the next frame's RoPE position.
    position: usize,
    /// Rows in the attention caches (at most window − 1).
    cached: usize,
    /// Every carried history and cache, one buffer (zeroed by `reset`).
    state: Buffer,
    pre_conv: Buffer,
    keys: []Buffer,
    values: []Buffer,
    dw: [2]Buffer,
    conv_in: Buffer,
    trans: [4]Buffer,
    units: [4][3]Buffer,
    conv_out: Buffer,

    pub fn init(decoder: *Decoder) !Stream {
        const c = decoder.config;
        const gpa = decoder.gpa;
        var self: Stream = undefined;
        self.decoder = decoder;
        const width = c.heads * c.head_dim;
        // Sizes in floats, in the order sliced below.
        var total: usize = 2 * c.quantized + 2 * c.layers * (c.window - 1) * width + 2 * (dwconv_kernel - 1) * c.latent + (residual_kernel - 1) * c.latent;
        for (decoder.blocks) |blk| {
            total += (blk.trans.taps - 1) * blk.trans.in;
            for (blk.units) |u| total += (u.conv1.taps - 1) * u.conv1.dilation * blk.trans.out;
        }
        total += (residual_kernel - 1) * c.blockChannels(4);
        self.state = try decoder.backend.create(total * 4);
        errdefer decoder.backend.release(self.state) catch {};
        self.keys = try gpa.alloc(Buffer, c.layers);
        errdefer gpa.free(self.keys);
        self.values = try gpa.alloc(Buffer, c.layers);
        var at: usize = 0;
        const slice = struct {
            fn take(state: Buffer, offset: *usize, count: usize) Buffer {
                defer offset.* += count;
                return state.slice(offset.* * 4, count * 4);
            }
        }.take;
        self.pre_conv = slice(self.state, &at, 2 * c.quantized);
        for (self.keys, self.values) |*k, *v| {
            k.* = slice(self.state, &at, (c.window - 1) * width);
            v.* = slice(self.state, &at, (c.window - 1) * width);
        }
        for (&self.dw) |*d| d.* = slice(self.state, &at, (dwconv_kernel - 1) * c.latent);
        self.conv_in = slice(self.state, &at, (residual_kernel - 1) * c.latent);
        for (&self.trans, &self.units, decoder.blocks) |*t, *units, blk| {
            t.* = slice(self.state, &at, (blk.trans.taps - 1) * blk.trans.in);
            for (units, blk.units) |*u, unit| u.* = slice(self.state, &at, (unit.conv1.taps - 1) * unit.conv1.dilation * blk.trans.out);
        }
        self.conv_out = slice(self.state, &at, (residual_kernel - 1) * c.blockChannels(4));
        self.reset();
        return self;
    }

    pub fn deinit(self: *Stream) void {
        self.decoder.backend.release(self.state) catch {};
        self.decoder.gpa.free(self.keys);
        self.decoder.gpa.free(self.values);
        self.* = undefined;
    }

    /// Back to the start of a clip. Valid while no command buffer is in
    /// flight, which every `push` guarantees on return.
    pub fn reset(self: *Stream) void {
        @memset(self.state.floats(), 0);
        self.position = 0;
        self.cached = 0;
    }

    /// Decodes the next `codes` into `out` (`codes.len · samplesPerFrame()`
    /// samples). A code past the codebook is `error.InvalidCode`, before any
    /// state changes.
    pub fn push(self: *Stream, codes: []const Frame, out: []f32) !void {
        const c = self.decoder.config;
        const per = c.samplesPerFrame();
        if (out.len != codes.len * per) return error.InvalidShape;
        for (codes) |frame| for (frame) |code| if (code >= c.codebook_size) return error.InvalidCode;
        if (self.position + codes.len > max_positions) return error.StreamTooLong;
        var at: usize = 0;
        while (at < codes.len) {
            const n = @min(max_frames, codes.len - at);
            try self.piece(codes[at..][0..n], out[at * per ..][0 .. n * per]);
            at += n;
        }
    }

    fn piece(self: *Stream, codes: []const Frame, out: []f32) !void {
        const d = self.decoder;
        const b = d.backend;
        const c = d.config;
        const n = codes.len;
        const ids: [*]u32 = @ptrCast(@alignCast(d.codes.host));
        for (codes, 0..) |frame, t| for (frame, 0..) |code, g| {
            ids[t * groups + g] = code;
        };
        try b.begin();
        errdefer b.discard();

        // Quantizer: the two projected sums, then the input convolution.
        const a = d.act;
        try b.ttsCodebookSum(d.codebooks, d.codes, d.first, d.rest, n, groups, c.codebook_size, c.codebook_dim);
        try d.mm(d.project[0], d.first, c.codebook_dim, a[0], n);
        try d.mm(d.project[1], d.rest, c.codebook_dim, a[1], n);
        try b.add(a[0], a[1], n * c.quantized);
        try d.convolve(d.pre_conv, self.pre_conv, a[0], n, a[1]);
        try self.transformer(a[1], n, a[0]);

        // Upsampling: transposed convolution, then ConvNeXt, twice.
        var x = a[0];
        var spare = a[1];
        var rows = n;
        for (d.upsample, &self.dw) |u, history| {
            try d.transpose(u.trans, null, x, rows, spare);
            rows *= u.trans.stride;
            std.mem.swap(Buffer, &x, &spare);
            try self.convNeXt(u.next, history, x, rows);
        }
        try d.convolve(d.conv_in, self.conv_in, x, rows, spare);
        std.mem.swap(Buffer, &x, &spare);

        // Decoder blocks: SnakeBeta, the transposed convolution, three residual units.
        for (d.blocks, &self.trans, &self.units) |blk, history, *units| {
            const channels = blk.trans.in;
            try b.ttsSnake(x, blk.act.scale, blk.act.inv, spare, rows * channels, channels);
            try d.transpose(blk.trans, history, spare, rows, x);
            rows *= blk.trans.stride;
            const out_channels = blk.trans.out;
            const count = rows * out_channels;
            for (blk.units, units) |unit, unit_history| {
                try b.ttsSnake(x, unit.act1.scale, unit.act1.inv, a[2], count, out_channels);
                try d.convolve(unit.conv1, unit_history, a[2], rows, a[3]);
                try b.ttsSnake(a[3], unit.act2.scale, unit.act2.inv, a[3], count, out_channels);
                try d.convolve(unit.conv2, null, a[3], rows, a[2]);
                try b.add(x, a[2], count);
            }
        }
        const last = c.blockChannels(4);
        try b.ttsSnake(x, d.act_out.scale, d.act_out.inv, spare, rows * last, last);
        try d.join(self.conv_out, residual_kernel - 1, spare, rows, last);
        try b.ttsConvOut(d.joined, d.out_w, d.out_b, d.samples, rows, last, residual_kernel);
        try d.keep(self.conv_out, residual_kernel - 1, rows, last);
        try b.commit();
        @memcpy(out, d.samples.floats()[0..out.len]);
        self.position += n;
    }

    /// The pre-transformer over `n` rows of `y` (`[n][latent]`) into `out`.
    fn transformer(self: *Stream, y: Buffer, n: usize, out: Buffer) !void {
        const d = self.decoder;
        const b = d.backend;
        const c = d.config;
        const width = c.heads * c.head_dim;
        const norm: Backend.Norm = .{ .rows = n, .width = c.hidden, .in_stride = c.hidden, .out_stride = c.hidden, .eps = c.rms_eps };
        const scale: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.head_dim))));
        const past = self.cached;
        const total = past + n;
        const kept = @min(c.window - 1, total);
        try d.mm(d.input, y, c.latent, d.tx, n);
        for (d.layers, self.keys, self.values) |layer, keys, values| {
            try b.rmsNorm(d.tx, layer.input_norm, d.th, norm);
            try d.mm(layer.q, d.th, c.hidden, d.tq, n);
            try d.mm(layer.k, d.th, c.hidden, d.tk, n);
            try d.mm(layer.v, d.th, c.hidden, d.tv, n);
            try b.ropeRows(d.tq, d.rope, c.heads, c.head_dim, c.head_dim, self.position, n, width, .split_half);
            try b.ropeRows(d.tk, d.rope, c.heads, c.head_dim, c.head_dim, self.position, n, width, .split_half);
            if (past > 0) {
                try b.copy(d.jk, keys, past * width);
                try b.copy(d.jv, values, past * width);
            }
            try b.copy(d.jk.slice(past * width * 4, n * width * 4), d.tk, n * width);
            try b.copy(d.jv.slice(past * width * 4, n * width * 4), d.tv, n * width);
            try b.ttsWindowAttention(d.tq, d.jk, d.jv, d.tatt, n, past, c.heads, c.window, scale);
            try b.copy(keys, d.jk.slice((total - kept) * width * 4, kept * width * 4), kept * width);
            try b.copy(values, d.jv.slice((total - kept) * width * 4, kept * width * 4), kept * width);
            try d.mm(layer.o, d.tatt, width, d.to, n);
            try b.ttsScaledAdd(d.tx, layer.attn_scale, d.to, n * c.hidden, c.hidden);
            try b.rmsNorm(d.tx, layer.post_norm, d.th, norm);
            try d.mm(layer.gate, d.th, c.hidden, d.tgate, n);
            try d.mm(layer.up, d.th, c.hidden, d.tup, n);
            try b.siluMul(d.tgate, d.tup, n * c.intermediate);
            try d.mm(layer.down, d.tgate, c.intermediate, d.to, n);
            try b.ttsScaledAdd(d.tx, layer.mlp_scale, d.to, n * c.hidden, c.hidden);
        }
        self.cached = kept;
        try b.rmsNorm(d.tx, d.norm, d.th, norm);
        try d.mm(d.output, d.th, c.hidden, out, n);
    }

    /// ConvNeXt in place over `x` (`[rows][latent]`), through `history`.
    fn convNeXt(self: *Stream, next: NeXt, history: Buffer, x: Buffer, rows: usize) !void {
        const d = self.decoder;
        const b = d.backend;
        const channels = d.config.latent;
        const a = d.act;
        // `x` is act[0] or act[1]; the other two hold the intermediates.
        const dw = a[2];
        const wide = a[3];
        try d.join(history, dwconv_kernel - 1, x, rows, channels);
        try b.ttsDepthwise(d.joined, next.dw, next.dw_b, dw, rows, channels, dwconv_kernel);
        try d.keep(history, dwconv_kernel - 1, rows, channels);
        try b.layerNorm(dw, next.ln_w, next.ln_b, dw, .{ .rows = rows, .width = channels, .in_stride = channels, .out_stride = channels, .eps = layer_norm_eps });
        try d.mm(next.pw1, dw, channels, wide, rows);
        try b.geluErf(wide, rows * next.pw1.matrix.rows);
        try d.mm(next.pw2, wide, next.pw1.matrix.rows, dw, rows);
        try b.ttsScaledAdd(x, next.gamma, dw, rows * channels, channels);
    }
};
