//! Fish's codec decoder (Drama 3) on Metal: fish_codec.zig's pipeline as
//! one command buffer per piece of at most `max_frames` frames. `Decoder`
//! holds the weights, copied in F32 from a CPU `fish_codec.Runtime` (its
//! tap-major convolution matrices, columns padded with zeros to the matmul
//! tile's 64; the file's F16 widens exactly, and the re-laid convolutions
//! could not be wrapped in place), and the scratch; `Stream` holds one
//! clip's carried state (every convolution's history, the window's keys
//! and values) on the device. Every kernel sums an output in an order fixed
//! by the output's own index, so pieces of any size give one decode's
//! samples bit for bit. The decoder borrows its `Backend`, which must
//! outlive it. docs/models/drama3.md § The codec on Metal.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const codec = @import("fish_codec.zig");
const q3 = @import("qwen3_codec.zig");

const Allocator = std.mem.Allocator;
const Backend = metal.Backend;
const Buffer = metal.Buffer;
const Frame = codec.Frame;
const books = codec.codebooks;
const latent = codec.latent;
const width = codec.heads * codec.head_dim;

/// Frames one command buffer decodes; `Stream.push` splits a longer push,
/// which changes no sample.
pub const max_frames = 8;
/// RoPE positions the table covers: a stream's frames at most.
pub const max_positions = 16384;
const dwconv_kernel = 7;
const out_kernel = 7;
/// fish_codec.zig's (and qwen3_codec.zig's ConvNeXt) constants.
const rms_eps = 1e-5;
const layer_norm_eps = 1e-6;
const rope_base = 1e4;

/// A matrix in a created buffer, `columns` padded to a multiple of 64.
const Mat = struct { buffer: Buffer, matrix: cpu.Matrix, bias: ?Buffer };
/// A causal convolution (`taps` 1 with `dilation` 1 is a plain matrix).
const Conv = struct { mat: Mat, in: usize, taps: usize, dilation: usize };
/// A transposed convolution: `stride` output rows per input row.
const Trans = struct { mat: Mat, in: usize, out: usize, stride: usize, taps: usize };
const Snake = struct { scale: Buffer, inv: Buffer };
const Layer = struct { attn_norm: Buffer, ffn_norm: Buffer, q: Mat, k: Mat, v: Mat, o: Mat, gate: Mat, up: Mat, down: Mat, attn_scale: Buffer, ffn_scale: Buffer };
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
fn bookSize(c: usize) usize {
    return if (c == 0) codec.semantic_size else codec.residual_size;
}
/// Where book `c` starts in `Decoder.books`, in floats.
fn bookOffset(c: usize) usize {
    return if (c == 0) 0 else (codec.semantic_size + (c - 1) * codec.residual_size) * codec.codebook_dim;
}

/// Fills `table` (`[positions][dims / 2]` cos, sin pairs) with fish_codec's
/// rope angles: computed in F32, cos and sin rounded to BF16.
pub fn ropeTable(table: []f32, positions: usize, dims: usize, base: f32) void {
    const half = dims / 2;
    for (0..positions) |position| for (0..half) |i| {
        const exponent: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dims));
        const freq: f32 = 1.0 / std.math.pow(f32, base, exponent);
        const angle: f32 = @as(f32, @floatFromInt(position)) * freq;
        table[(position * half + i) * 2] = codec.bf16(@cos(angle));
        table[(position * half + i) * 2 + 1] = codec.bf16(@sin(angle));
    };
}

/// The decoder's weights and scratch on the device.
pub const Decoder = struct {
    gpa: Allocator,
    backend: *Backend,
    buffers: std.ArrayList(Buffer) = .empty,
    /// The codebooks on the host (`bookOffset`): the lookup gathers there.
    books: []f32,
    /// The ten out_proj matrices side by side (`[latent][10 · 8]`), their
    /// biases summed: one product gives the summed projections.
    project: Mat,
    layers: []Layer,
    norm: Buffer,
    window: usize,
    upsample: [2]Up,
    conv_in: Conv,
    blocks: [4]Block,
    act_out: Snake,
    out_w: Buffer,
    out_b: Buffer,
    rope: Buffer,
    // Scratch, sized for `max_frames`.
    picked: Buffer,
    /// The pipeline's activations, four buffers of the largest stage each.
    act: [4]Buffer,
    gathered: Buffer,
    joined: Buffer,
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
        const post = runtime.post;
        if (post.window > Backend.tts_window_max or post.layers.len == 0 or runtime.conv_out.out != 1 or runtime.conv_out.kernel != out_kernel) return error.UnsupportedConfiguration;
        var self: Decoder = undefined;
        self.gpa = gpa;
        self.backend = backend;
        self.buffers = .empty;
        self.bytes = 0;
        self.window = post.window;
        errdefer self.releaseAll();

        self.books = try gpa.alloc(f32, bookOffset(books));
        errdefer gpa.free(self.books);
        for (runtime.books, 0..) |book, c| @memcpy(self.books[bookOffset(c)..][0 .. bookSize(c) * codec.codebook_dim], book);
        {
            const inner = books * codec.codebook_dim;
            const w = try gpa.alloc(f32, latent * inner);
            defer gpa.free(w);
            const bias = try gpa.alloc(f32, latent);
            defer gpa.free(bias);
            @memset(bias, 0);
            for (runtime.project, 0..) |p, c| {
                if (p.in != codec.codebook_dim or p.out != latent) return error.UnsupportedConfiguration;
                for (0..latent) |r| @memcpy(w[r * inner + c * codec.codebook_dim ..][0..codec.codebook_dim], p.w[r * codec.codebook_dim ..][0..codec.codebook_dim]);
                for (bias, p.b.?) |*a, v| a.* += v;
            }
            self.project = try self.linear(w, bias, latent, inner);
        }

        self.layers = try gpa.alloc(Layer, post.layers.len);
        errdefer gpa.free(self.layers);
        for (self.layers, post.layers) |*out, l| out.* = .{
            .attn_norm = try self.upload(l.attn_norm),
            .ffn_norm = try self.upload(l.ffn_norm),
            .q = try self.plain(l.q),
            .k = try self.plain(l.k),
            .v = try self.plain(l.v),
            .o = try self.plain(l.o),
            .gate = try self.plain(l.gate),
            .up = try self.plain(l.up),
            .down = try self.plain(l.down),
            .attn_scale = try self.upload(l.attn_scale),
            .ffn_scale = try self.upload(l.ffn_scale),
        };
        self.norm = try self.upload(post.norm);
        for (&self.upsample, runtime.upsample) |*u, s| u.* = .{
            .trans = try self.transposed(s.trans),
            .next = .{
                .dw = try self.upload(s.next.dw),
                .dw_b = try self.upload(s.next.dw_b),
                .ln_w = try self.upload(s.next.ln_w),
                .ln_b = try self.upload(s.next.ln_b),
                .gamma = try self.upload(s.next.gamma),
                .pw1 = try self.plain(s.next.pw1),
                .pw2 = try self.plain(s.next.pw2),
            },
        };
        self.conv_in = try self.causal(runtime.conv_in);
        for (&self.blocks, runtime.blocks) |*out, blk| {
            out.act = try self.snake(blk.act);
            out.trans = try self.transposed(blk.trans);
            for (&out.units, blk.units) |*u, unit| u.* = .{
                .act1 = try self.snake(unit.act1),
                .conv1 = try self.causal(unit.conv1),
                .act2 = try self.snake(unit.act2),
                .conv2 = try self.causal(unit.conv2),
            };
        }
        self.act_out = try self.snake(runtime.act_out);
        self.out_w = try self.upload(runtime.conv_out.w);
        self.out_b = try self.upload(runtime.conv_out.b);
        self.rope = try self.floats(max_positions * codec.head_dim);
        ropeTable(self.rope.floats(), max_positions, codec.head_dim, rope_base);

        // Scratch: every stage's rows at `max_frames`, padded for the tile.
        const n = max_frames;
        var act: usize = tile(n) * latent;
        var gathered: usize = 64;
        var joined: usize = 64;
        var rows: usize = n;
        for (self.upsample) |u| {
            gathered = @max(gathered, tile(rows) * u.trans.mat.matrix.columns);
            joined = @max(joined, (rows + u.trans.taps - 1) * u.trans.in);
            act = @max(act, tile(rows) * u.trans.mat.matrix.rows);
            rows *= u.trans.stride;
            act = @max(act, tile(rows) * u.next.pw1.matrix.rows);
            joined = @max(joined, (rows + dwconv_kernel - 1) * latent);
        }
        act = @max(act, tile(rows) * self.conv_in.mat.matrix.rows);
        gathered = @max(gathered, tile(rows) * self.conv_in.mat.matrix.columns);
        joined = @max(joined, (rows + (self.conv_in.taps - 1) * self.conv_in.dilation) * self.conv_in.in);
        for (self.blocks) |blk| {
            gathered = @max(gathered, tile(rows) * blk.trans.mat.matrix.columns);
            joined = @max(joined, (rows + blk.trans.taps - 1) * blk.trans.in);
            act = @max(act, tile(rows) * @max(blk.trans.in, blk.trans.mat.matrix.rows));
            rows *= blk.trans.stride;
            act = @max(act, tile(rows) * @max(blk.trans.out, blk.units[0].conv2.mat.matrix.rows));
            for (blk.units) |u| {
                gathered = @max(gathered, tile(rows) * @max(u.conv1.mat.matrix.columns, u.conv2.mat.matrix.columns));
                joined = @max(joined, (rows + (u.conv1.taps - 1) * u.conv1.dilation) * blk.trans.out);
            }
        }
        const last = self.blocks[3].trans.out;
        joined = @max(joined, (rows + out_kernel - 1) * last);
        self.picked = try self.floats(tile(n) * self.project.matrix.columns);
        for (&self.act) |*a| a.* = try self.floats(act);
        self.gathered = try self.floats(gathered);
        self.joined = try self.floats(joined);
        const t = tile(n);
        const intermediate = post.layers[0].gate.out;
        self.th = try self.floats(t * latent);
        self.tq = try self.floats(t * width);
        self.tk = try self.floats(t * width);
        self.tv = try self.floats(t * width);
        self.tatt = try self.floats(t * width);
        self.to = try self.floats(t * latent);
        self.tgate = try self.floats(t * intermediate);
        self.tup = try self.floats(t * intermediate);
        self.jk = try self.floats((self.window - 1 + n) * width);
        self.jv = try self.floats((self.window - 1 + n) * width);
        self.samples = try self.floats(n * codec.samples_per_frame);
        return self;
    }

    pub fn deinit(self: *Decoder) void {
        self.releaseAll();
        self.gpa.free(self.layers);
        self.gpa.free(self.books);
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
    /// `w` is `[rows][columns]`, copied with the columns padded to 64 (the
    /// buffer is created zeroed).
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
    fn plain(self: *Decoder, l: q3.Linear) !Mat {
        return self.linear(l.w, l.b, l.out, l.in);
    }
    fn causal(self: *Decoder, c: q3.CausalConv) !Conv {
        return .{ .mat = try self.linear(c.w, c.b, c.out, c.kernel * c.in), .in = c.in, .taps = c.kernel, .dilation = c.dilation };
    }
    fn transposed(self: *Decoder, t: q3.TransConv) !Trans {
        return .{ .mat = try self.linear(t.w, t.b, t.stride * t.out, t.taps * t.in), .in = t.in, .out = t.out, .stride = t.stride, .taps = t.taps };
    }
    fn snake(self: *Decoder, s: q3.SnakeBeta) !Snake {
        return .{ .scale = try self.upload(s.scale), .inv = try self.upload(s.inv) };
    }

    // ---- recording helpers --------------------------------------------

    /// `output = m · input (+ bias)` over `rows` rows of `in_stride`.
    fn mm(self: *Decoder, m: Mat, input: Buffer, in_stride: usize, output: Buffer, rows: usize) !void {
        const b = self.backend;
        try b.matmulTile(m.buffer, m.matrix, input, in_stride, output, m.matrix.rows, rows);
        if (m.bias) |bias| try b.addBiasRows(output, bias, m.matrix.rows, rows, m.matrix.rows);
    }

    /// `joined = history ‖ x` (`history` has `history_rows` rows of `w`).
    fn join(self: *Decoder, history: ?Buffer, history_rows: usize, x: Buffer, rows: usize, w: usize) !void {
        const b = self.backend;
        if (history_rows > 0) try b.copy(self.joined, history.?, history_rows * w);
        try b.copy(self.joined.slice(history_rows * w * 4, rows * w * 4), x, rows * w);
    }
    /// `history = the last rows of joined` after a `join` of `rows` new rows.
    fn keep(self: *Decoder, history: ?Buffer, history_rows: usize, rows: usize, w: usize) !void {
        if (history_rows > 0) try self.backend.copy(history.?, self.joined.slice(rows * w * 4, history_rows * w * 4), history_rows * w);
    }

    /// A causal convolution of `rows` rows of `x` into `y` through `history`.
    fn convolve(self: *Decoder, conv: Conv, history: ?Buffer, x: Buffer, rows: usize, y: Buffer) !void {
        const h = (conv.taps - 1) * conv.dilation;
        const columns = conv.mat.matrix.columns;
        if (h == 0 and columns == conv.in) return self.mm(conv.mat, x, conv.in, y, rows);
        var source = x;
        if (h > 0) {
            try self.join(history, h, x, rows, conv.in);
            source = self.joined;
        }
        try self.backend.ttsGather(source, self.gathered, rows, conv.in, conv.taps, columns, 0, @intCast(conv.dilation));
        try self.mm(conv.mat, self.gathered, columns, y, rows);
        try self.keep(history, h, rows, conv.in);
    }

    /// A transposed convolution of `rows` input rows into `rows · stride` rows of `y`.
    fn transpose(self: *Decoder, t: Trans, history: ?Buffer, x: Buffer, rows: usize, y: Buffer) !void {
        const h = t.taps - 1;
        if (h == 0 and t.mat.matrix.columns == t.in) return self.mm(t.mat, x, t.in, y, rows);
        try self.join(history, h, x, rows, t.in);
        try self.backend.ttsGather(self.joined, self.gathered, rows, t.in, t.taps, t.mat.matrix.columns, h, -1);
        try self.mm(t.mat, self.gathered, t.mat.matrix.columns, y, rows);
        try self.keep(history, h, rows, t.in);
    }
};

/// One clip's decode: push frames as they come. Borrows the decoder;
/// streams may share one if their pushes do not overlap (its scratch).
pub const Stream = struct {
    decoder: *Decoder,
    /// Frames pushed so far: the next frame's RoPE position.
    position: usize,
    /// Rows in the attention caches (at most window − 1).
    cached: usize,
    /// Every carried history and cache, one buffer (zeroed by `reset`).
    state: Buffer,
    keys: []Buffer,
    values: []Buffer,
    up_trans: [2]?Buffer,
    dw: [2]Buffer,
    conv_in: Buffer,
    trans: [4]Buffer,
    units: [4][3]Buffer,
    conv_out: Buffer,

    pub fn init(decoder: *Decoder) !Stream {
        const gpa = decoder.gpa;
        const layers = decoder.layers.len;
        const cache = (decoder.window - 1) * width;
        var self: Stream = undefined;
        self.decoder = decoder;
        // Sizes in floats, in the order sliced below.
        var total: usize = 2 * layers * cache + (decoder.conv_in.taps - 1) * decoder.conv_in.dilation * decoder.conv_in.in;
        for (decoder.upsample) |u| total += (u.trans.taps - 1) * u.trans.in + (dwconv_kernel - 1) * latent;
        for (decoder.blocks) |blk| {
            total += (blk.trans.taps - 1) * blk.trans.in;
            for (blk.units) |u| total += (u.conv1.taps - 1) * u.conv1.dilation * blk.trans.out;
        }
        total += (out_kernel - 1) * decoder.blocks[3].trans.out;
        self.state = try decoder.backend.create(total * 4);
        errdefer decoder.backend.release(self.state) catch {};
        self.keys = try gpa.alloc(Buffer, layers);
        errdefer gpa.free(self.keys);
        self.values = try gpa.alloc(Buffer, layers);
        var at: usize = 0;
        const take = struct {
            fn take(state: Buffer, offset: *usize, count: usize) Buffer {
                defer offset.* += count;
                return state.slice(offset.* * 4, count * 4);
            }
        }.take;
        for (self.keys, self.values) |*k, *v| {
            k.* = take(self.state, &at, cache);
            v.* = take(self.state, &at, cache);
        }
        for (&self.up_trans, &self.dw, decoder.upsample) |*t, *d, u| {
            const h = (u.trans.taps - 1) * u.trans.in;
            t.* = if (h > 0) take(self.state, &at, h) else null;
            d.* = take(self.state, &at, (dwconv_kernel - 1) * latent);
        }
        self.conv_in = take(self.state, &at, (decoder.conv_in.taps - 1) * decoder.conv_in.dilation * decoder.conv_in.in);
        for (&self.trans, &self.units, decoder.blocks) |*t, *units, blk| {
            t.* = take(self.state, &at, (blk.trans.taps - 1) * blk.trans.in);
            for (units, blk.units) |*u, unit| u.* = take(self.state, &at, (unit.conv1.taps - 1) * unit.conv1.dilation * blk.trans.out);
        }
        self.conv_out = take(self.state, &at, (out_kernel - 1) * decoder.blocks[3].trans.out);
        std.debug.assert(at == total);
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

    pub fn push(self: *Stream, codes: []const Frame, out: []f32) !void {
        return self.pushObserved(codes, out, null);
    }

    /// Decodes the next `codes` into `out` (`codes.len · 2048` samples);
    /// codes past a book are clamped to it, as the CPU reference does. With
    /// an observer, each stage's rows are shown as fish_codec's are (the
    /// piece commits at each stage to read them; no value changes).
    pub fn pushObserved(self: *Stream, codes: []const Frame, out: []f32, observer: ?codec.Observer) !void {
        if (out.len != codes.len * codec.samples_per_frame) return error.InvalidShape;
        if (self.position + codes.len > max_positions) return error.StreamTooLong;
        var at: usize = 0;
        while (at < codes.len) {
            const n: usize = @min(max_frames, codes.len - at);
            try self.piece(codes[at..][0..n], out[at * codec.samples_per_frame ..][0 .. n * codec.samples_per_frame], observer);
            at += n;
        }
    }

    /// Shows `count` floats of `buffer` as `stage`: commits what is recorded
    /// and begins again.
    fn see(self: *Stream, observer: ?codec.Observer, stage: codec.Stage, buffer: Buffer, count: usize) !void {
        const o = observer orelse return;
        const b = self.decoder.backend;
        try b.commit();
        o.record(o.context, stage, buffer.floats()[0..count]);
        try b.begin();
    }

    fn piece(self: *Stream, codes: []const Frame, out: []f32, observer: ?codec.Observer) !void {
        const d = self.decoder;
        const b = d.backend;
        const n = codes.len;
        // The lookup on the host, into the side-by-side matrix's input row.
        const stride = d.project.matrix.columns;
        const picked = d.picked.floats();
        for (codes, 0..) |frame, t| for (frame, 0..) |code, c| {
            const row = @min(@as(usize, code), bookSize(c) - 1);
            @memcpy(picked[t * stride + c * codec.codebook_dim ..][0..codec.codebook_dim], d.books[bookOffset(c) + row * codec.codebook_dim ..][0..codec.codebook_dim]);
        };
        try b.begin();
        errdefer b.discard();

        const a = d.act;
        try d.mm(d.project, d.picked, stride, a[0], n);
        try self.see(observer, .quantized, a[0], n * latent);
        try self.transformer(a[0], n, a[1]);
        try self.see(observer, .transformer, a[1], n * latent);

        // Upsampling: transposed convolution, then ConvNeXt, twice.
        var x = a[1];
        var spare = a[0];
        var rows = n;
        for (d.upsample, self.up_trans, self.dw, [2]codec.Stage{ .upsample_0, .upsample_1 }) |u, history, dw, label| {
            try d.transpose(u.trans, history, x, rows, spare);
            rows *= u.trans.stride;
            std.mem.swap(Buffer, &x, &spare);
            try self.convNeXt(u.next, dw, x, rows);
            try self.see(observer, label, x, rows * latent);
        }
        try d.convolve(d.conv_in, self.conv_in, x, rows, spare);
        std.mem.swap(Buffer, &x, &spare);
        try self.see(observer, .decoder_in, x, rows * d.conv_in.mat.matrix.rows);

        // Decoder blocks: Snake, the transposed convolution, three residual units.
        for (d.blocks, self.trans, &self.units, [4]codec.Stage{ .block_0, .block_1, .block_2, .block_3 }) |blk, history, *units, label| {
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
            try self.see(observer, label, x, count);
        }
        const last = d.blocks[3].trans.out;
        try b.ttsSnake(x, d.act_out.scale, d.act_out.inv, spare, rows * last, last);
        try d.join(self.conv_out, out_kernel - 1, spare, rows, last);
        try b.ttsConvOutTanh(d.joined, d.out_w, d.out_b, d.samples, rows, last, out_kernel);
        try d.keep(self.conv_out, out_kernel - 1, rows, last);
        try b.commit();
        @memcpy(out, d.samples.floats()[0..out.len]);
        if (observer) |o| o.record(o.context, .samples, out);
        self.position += n;
    }

    /// The windowed post-transformer over `n` rows of `x` (`[n][latent]`,
    /// the residual stream, changed in place); its final norm into `out`.
    fn transformer(self: *Stream, x: Buffer, n: usize, out: Buffer) !void {
        const d = self.decoder;
        const b = d.backend;
        const norm: Backend.Norm = .{ .rows = n, .width = latent, .in_stride = latent, .out_stride = latent, .eps = rms_eps };
        const scale: f32 = @floatCast(1 / @sqrt(@as(f64, codec.head_dim)));
        const past = self.cached;
        const total = past + n;
        const kept = @min(d.window - 1, total);
        for (d.layers, self.keys, self.values) |layer, keys, values| {
            const intermediate = layer.gate.matrix.rows;
            try b.rmsNorm(x, layer.attn_norm, d.th, norm);
            try d.mm(layer.q, d.th, latent, d.tq, n);
            try d.mm(layer.k, d.th, latent, d.tk, n);
            try d.mm(layer.v, d.th, latent, d.tv, n);
            try b.ropeRows(d.tq, d.rope, codec.heads, codec.head_dim, codec.head_dim, self.position, n, width, .adjacent);
            try b.ropeRows(d.tk, d.rope, codec.heads, codec.head_dim, codec.head_dim, self.position, n, width, .adjacent);
            if (past > 0) {
                try b.copy(d.jk, keys, past * width);
                try b.copy(d.jv, values, past * width);
            }
            try b.copy(d.jk.slice(past * width * 4, n * width * 4), d.tk, n * width);
            try b.copy(d.jv.slice(past * width * 4, n * width * 4), d.tv, n * width);
            try b.ttsWindowAttention(d.tq, d.jk, d.jv, d.tatt, n, past, codec.heads, d.window, scale);
            try b.copy(keys, d.jk.slice((total - kept) * width * 4, kept * width * 4), kept * width);
            try b.copy(values, d.jv.slice((total - kept) * width * 4, kept * width * 4), kept * width);
            try d.mm(layer.o, d.tatt, width, d.to, n);
            try b.ttsScaledAdd(x, layer.attn_scale, d.to, n * latent, latent);
            try b.rmsNorm(x, layer.ffn_norm, d.th, norm);
            try d.mm(layer.gate, d.th, latent, d.tgate, n);
            try d.mm(layer.up, d.th, latent, d.tup, n);
            try b.siluMul(d.tgate, d.tup, n * intermediate);
            try d.mm(layer.down, d.tgate, intermediate, d.to, n);
            try b.ttsScaledAdd(x, layer.ffn_scale, d.to, n * latent, latent);
        }
        self.cached = kept;
        try b.rmsNorm(x, d.norm, out, norm);
    }

    /// ConvNeXt in place over `x` (`[rows][latent]`), through `history`.
    fn convNeXt(self: *Stream, next: NeXt, history: Buffer, x: Buffer, rows: usize) !void {
        const d = self.decoder;
        const b = d.backend;
        const a = d.act;
        // `x` is act[0] or act[1]; the other two hold the intermediates.
        const dw = a[2];
        const wide = a[3];
        try d.join(history, dwconv_kernel - 1, x, rows, latent);
        try b.ttsDepthwise(d.joined, next.dw, next.dw_b, dw, rows, latent, dwconv_kernel);
        try d.keep(history, dwconv_kernel - 1, rows, latent);
        try b.layerNorm(dw, next.ln_w, next.ln_b, dw, .{ .rows = rows, .width = latent, .in_stride = latent, .out_stride = latent, .eps = layer_norm_eps });
        try d.mm(next.pw1, dw, latent, wide, rows);
        try b.geluErf(wide, rows * next.pw1.matrix.rows);
        try d.mm(next.pw2, wide, next.pw1.matrix.rows, dw, rows);
        try b.ttsScaledAdd(x, next.gamma, dw, rows * latent, latent);
    }
};

test "the rope table rotates as fish_codec.rope does" {
    const dims = 64;
    const positions = 300;
    const table = try std.testing.allocator.alloc(f32, positions * dims);
    defer std.testing.allocator.free(table);
    ropeTable(table, positions, dims, rope_base);
    for ([_]usize{ 0, 1, 7, 127, 299 }) |position| {
        var want: [dims]f32 = undefined;
        for (&want, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) - 1.5;
        var got = want;
        codec.rope(&want, position, rope_base);
        // The kernel's adjacent pairing: (2i, 2i + 1) by entry i.
        for (0..dims / 2) |i| {
            const c = table[(position * dims / 2 + i) * 2];
            const s = table[(position * dims / 2 + i) * 2 + 1];
            const a0 = got[2 * i];
            const b0 = got[2 * i + 1];
            got[2 * i] = a0 * c - b0 * s;
            got[2 * i + 1] = a0 * s + b0 * c;
        }
        try std.testing.expectEqualSlices(f32, &want, &got);
    }
}

test "the codebook offsets tile the books" {
    try std.testing.expectEqual(@as(usize, codec.semantic_size * codec.codebook_dim), bookOffset(1));
    try std.testing.expectEqual(bookOffset(books - 1) + codec.residual_size * codec.codebook_dim, bookOffset(books));
}
