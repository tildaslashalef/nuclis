//! Higgs Audio V2 codec decoder on Metal (`higgs_codec.zig`'s exact pipeline
//! executed in one command buffer on the GPU using shared `codec_metal.Device`
//! primitives and existing `ttsSnake`, `ttsGather`, `matmulTile`, `addBiasRows`,
//! `add`, and `copy` kernels).
//!
//! Non-causal symmetric `Conv1d` pads `p` zero rows on both ends into `joined`
//! and runs `ttsGather` + `matmulTile`. Non-causal `ConvTranspose1d` (for
//! `upsampling_ratios = [8, 5, 4, 2, 3]`, `k = 2*s`, `p = (s+1)/2`, `op = s%2`)
//! folds into a 3-tap gather `[q-1, q, q+1]` + one `matmulTile` (`[T, 3*C_in]`
//! → `[T, s*C_out]` = `[T*s, C_out]`) + `addBiasRows`, exact with zero new GPU
//! kernels. The decoder borrows its `Backend`, which must outlive it.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cm = @import("codec_metal.zig");
const hc = @import("higgs_codec.zig");

const Allocator = std.mem.Allocator;
const Backend = metal.Backend;
const Buffer = metal.Buffer;
const Mat = cm.Mat;
const Snake = cm.Snake;

const padded = cm.padded;
const tile = cm.tile;

const NonCausalConv = struct {
    mat: Mat,
    in: usize,
    out: usize,
    k: usize,
    padding: usize,
    dilation: usize,
};

const NonCausalTrans = struct {
    /// `[stride * out][padded(3 * in)]`, bias applied per `out` channel separately.
    mat: Mat,
    bias: ?Buffer,
    in: usize,
    out: usize,
    stride: usize,
};

const ResUnit = struct {
    snake1: Snake,
    conv1: NonCausalConv,
    snake2: Snake,
    conv2: NonCausalConv,
};

const Block = struct {
    snake1: Snake,
    conv_t1: NonCausalTrans,
    units: [3]ResUnit,
};

const Scratch = struct {
    frames_capacity: usize = 0,
    picked: ?Buffer = null,
    /// Ping-pong activation buffers + residual scratch (sized for max stage elements).
    act: [3]?Buffer = .{ null, null, null },
    joined: ?Buffer = null,
    gathered: ?Buffer = null,
    /// Pre-zeroed scratch for symmetric zero-padding via `backend.copy`.
    zeros: ?Buffer = null,
    out_raw: ?Buffer = null,

    fn release(self: *Scratch, backend: *Backend) void {
        const bufs = [_]?Buffer{
            self.picked, self.act[0], self.act[1], self.act[2], self.joined, self.gathered, self.zeros, self.out_raw,
        };
        for (bufs) |mb| if (mb) |b| backend.release(b) catch {};
        self.* = .{};
    }

    fn ensure(self: *Scratch, backend: *Backend, frames: usize) !void {
        if (frames <= self.frames_capacity) return;
        self.release(backend);

        // Largest stage activation (every Conv1d and ConvTranspose1d output padded to matmul_tile):
        // initial conv1: tile(T) * 1024
        // block_0 conv_t1: tile(T) * (8 * 512) = tile(T) * 4096
        // block_1 conv_t1: tile(8*T) * (5 * 256) = tile(8*T) * 1280
        // block_2 conv_t1: tile(40*T) * (4 * 128) = tile(40*T) * 512
        // block_3 conv_t1: tile(160*T) * (2 * 64) = tile(160*T) * 128
        // block_4 conv_t1: tile(320*T) * (3 * 32) = tile(320*T) * 96
        // block_4 convs:   tile(960*T) * 32
        const max_act = @max(
            @max(tile(frames) * 4096, tile(frames * 8) * 1280),
            @max(
                @max(tile(frames * 40) * 512, tile(frames * 160) * 128),
                @max(tile(frames * 320) * 96, tile(frames * 960) * 32),
            ),
        );
        // Max joined floats: dilation 9 has pad = 27 on each side: (T_stage + 54) * C.
        const max_joined = @max(
            @max((frames * 960 + 54) * 32, (frames * 320 + 54) * 64),
            @max((frames * 160 + 54) * 128, @max((frames * 40 + 54) * 256, (frames * 8 + 54) * 512)),
        );
        // Max gathered floats across all Conv1d (k=7) and ConvTranspose1d (3-tap) stages:
        const max_gathered = @max(
            @max(tile(frames) * padded(3 * 1024), tile(frames * 8) * padded(7 * 512)),
            @max(
                @max(tile(frames * 40) * padded(7 * 256), tile(frames * 160) * padded(7 * 128)),
                @max(tile(frames * 320) * padded(7 * 64), tile(frames * 960) * padded(7 * 32)),
            ),
        );
        // Max padding zeros needed: 27 rows * 512 ch = 13,824 floats (always <= 16,384).
        const max_zeros: usize = 27 * 1024;

        self.picked = try backend.create(tile(frames) * hc.codebooks * hc.codebook_dim * 4);
        for (&self.act) |*a| a.* = try backend.create(max_act * 4);
        self.joined = try backend.create(max_joined * 4);
        self.gathered = try backend.create(max_gathered * 4);
        self.zeros = try backend.create(max_zeros * 4); // backend.create returns zero-filled memory
        self.out_raw = try backend.create(tile(frames * hc.samples_per_frame) * 8 * 4);
        self.frames_capacity = frames;
    }
};

pub const Decoder = struct {
    gpa: Allocator,
    device: cm.Device,
    /// Host codebook table `[8][1024][64]` for fast host gather before GPU matmul.
    books: []f32,
    /// All 8 `project_out` matrices stacked side by side `[1024][512]` with summed bias.
    project: Mat,
    fc2: Mat,
    conv1: NonCausalConv,
    blocks: [5]Block,
    snake_out: Snake,
    /// Final `conv2` (`32 -> 1`, k=7, pad=3) padded to 8 output channels (rows 1..7 zero)
    /// so `matmulTile` executes unclamped F32 directly.
    conv2_mat: Mat,
    conv2_bias: f32,
    scratch: Scratch = .{},

    /// Copies F32 weights from CPU `hc.Runtime` onto Metal device; `cpu_rt`
    /// may be released afterwards. `backend` must outlive `Decoder`.
    pub fn init(gpa: Allocator, backend: *Backend, cpu_rt: *const hc.Runtime) !Decoder {
        if (!metal.enabled) return error.MetalNotEnabled;

        var device: cm.Device = .{ .gpa = gpa, .backend = backend };
        errdefer device.releaseAll();

        const total_book_floats = hc.codebooks * hc.codebook_size * hc.codebook_dim;
        const books = try gpa.alloc(f32, total_book_floats);
        errdefer gpa.free(books);
        for (0..hc.codebooks) |c| {
            @memcpy(books[c * hc.codebook_size * hc.codebook_dim ..][0 .. hc.codebook_size * hc.codebook_dim], cpu_rt.books[c]);
        }

        // Stack all 8 `project_out` `[1024][64]` matrices into `[1024][512]` with summed bias.
        const proj_in = hc.codebooks * hc.codebook_dim; // 512
        const proj_w = try gpa.alloc(f32, hc.latent_dim * proj_in);
        defer gpa.free(proj_w);
        const proj_b = try gpa.alloc(f32, hc.latent_dim);
        defer gpa.free(proj_b);
        @memset(proj_b, 0);
        for (0..hc.codebooks) |c| {
            const l = cpu_rt.project_out[c];
            for (0..hc.latent_dim) |r| {
                @memcpy(proj_w[r * proj_in + c * hc.codebook_dim ..][0..hc.codebook_dim], l.w[r * hc.codebook_dim ..][0..hc.codebook_dim]);
            }
            if (l.b) |bias| {
                for (proj_b, bias) |*dst, bv| dst.* += bv;
            }
        }
        const project = try device.linear(proj_w, proj_b, hc.latent_dim, proj_in);
        const fc2 = try device.linear(cpu_rt.fc2.w, cpu_rt.fc2.b, cpu_rt.fc2.out, cpu_rt.fc2.in);
        const conv1 = try uploadConv(&device, cpu_rt.conv1);

        var blocks: [5]Block = undefined;
        for (&blocks, cpu_rt.blocks) |*dst, src| {
            dst.snake1 = try uploadSnake(&device, gpa, src.snake1);
            dst.conv_t1 = try uploadTrans(&device, gpa, src.conv_t1);
            for (&dst.units, src.units) |*du, su| {
                du.* = .{
                    .snake1 = try uploadSnake(&device, gpa, su.snake1),
                    .conv1 = try uploadConv(&device, su.conv1),
                    .snake2 = try uploadSnake(&device, gpa, su.snake2),
                    .conv2 = try uploadConv(&device, su.conv2),
                };
            }
        }
        const snake_out = try uploadSnake(&device, gpa, cpu_rt.snake_out);

        // Final conv2 (`1 x 224`) padded to `8 x 224` so `matmulTile` (rows % 8 == 0) runs unclamped.
        const c2_cols = cpu_rt.conv2.k * cpu_rt.conv2.in; // 224
        const c2_w = try gpa.alloc(f32, 8 * c2_cols);
        defer gpa.free(c2_w);
        @memset(c2_w, 0);
        @memcpy(c2_w[0..c2_cols], cpu_rt.conv2.w[0..c2_cols]);
        const conv2_mat = try device.linear(c2_w, null, 8, c2_cols);
        const conv2_bias: f32 = if (cpu_rt.conv2.b) |b| b[0] else 0.0;

        return .{
            .gpa = gpa,
            .device = device,
            .books = books,
            .project = project,
            .fc2 = fc2,
            .conv1 = conv1,
            .blocks = blocks,
            .snake_out = snake_out,
            .conv2_mat = conv2_mat,
            .conv2_bias = conv2_bias,
        };
    }

    pub fn deinit(self: *Decoder) void {
        self.scratch.release(self.device.backend);
        self.device.releaseAll();
        self.gpa.free(self.books);
        self.* = undefined;
    }

    /// Decodes `frames` (`[T][8]u16`) into 24 kHz mono F32 samples (`T * 960`)
    /// in ONE Metal command buffer. Caller owns returned slice.
    pub fn decode(self: *Decoder, gpa: Allocator, frames: []const hc.Frame) ![]f32 {
        if (frames.len == 0 or frames.len > hc.max_decode_frames) return error.InvalidLength;
        for (frames) |frame| {
            for (frame) |code| if (code >= hc.codebook_size) return error.InvalidCode;
        }

        const t_len = frames.len;
        try self.scratch.ensure(self.device.backend, t_len);
        const b = self.device.backend;

        // 1. Gather codebook embeddings on host into `picked` (`[T][512]`).
        const picked_buf = self.scratch.picked.?;
        const picked = picked_buf.floats();
        const proj_in = hc.codebooks * hc.codebook_dim;
        for (frames, 0..) |frame, t| {
            for (0..hc.codebooks) |c| {
                const src_off = (c * hc.codebook_size + frame[c]) * hc.codebook_dim;
                @memcpy(picked[t * proj_in + c * hc.codebook_dim ..][0..hc.codebook_dim], self.books[src_off..][0..hc.codebook_dim]);
            }
        }

        const a0 = self.scratch.act[0].?;
        const a1 = self.scratch.act[1].?;
        const a2 = self.scratch.act[2].?;
        const joined = self.scratch.joined.?;
        const gathered = self.scratch.gathered.?;
        const zeros = self.scratch.zeros.?;
        const out_raw = self.scratch.out_raw.?;

        try b.begin();
        errdefer b.discard();

        // 2. RVQ projection + fc2 + conv1.
        try self.device.mm(self.project, picked_buf, proj_in, a0, t_len); // [T, 1024]
        try self.device.mm(self.fc2, a0, hc.latent_dim, a1, t_len); // [T, 256]
        try self.runNonCausalConv(self.conv1, a1, t_len, a0, joined, gathered, zeros); // [T, 1024]

        var cur = a0;
        var alt = a1;
        var cur_t = t_len;

        // 3. 5 DAC decoder blocks.
        for (self.blocks) |blk| {
            try b.ttsSnake(cur, blk.snake1.scale, blk.snake1.inv, cur, cur_t * blk.conv_t1.in, blk.conv_t1.in);
            try self.runNonCausalTrans(blk.conv_t1, cur, cur_t, alt, joined, gathered, zeros);
            cur_t *= blk.conv_t1.stride;
            const tmp = cur;
            cur = alt;
            alt = tmp;

            for (blk.units) |u| {
                const ch = u.conv1.in;
                try b.ttsSnake(cur, u.snake1.scale, u.snake1.inv, alt, cur_t * ch, ch);
                try self.runNonCausalConv(u.conv1, alt, cur_t, a2, joined, gathered, zeros);
                try b.ttsSnake(a2, u.snake2.scale, u.snake2.inv, a2, cur_t * ch, ch);
                try self.runNonCausalConv(u.conv2, a2, cur_t, alt, joined, gathered, zeros);
                try b.add(cur, alt, cur_t * ch);
            }
        }

        // 4. Final Snake1d + conv2 (32 -> 8 padded, unclamped).
        const final_ch: usize = 32;
        try b.ttsSnake(cur, self.snake_out.scale, self.snake_out.inv, cur, cur_t * final_ch, final_ch);
        const pad_floats = 3 * final_ch;
        try b.copy(joined, zeros, pad_floats);
        try b.copy(joined.slice(pad_floats * 4, cur_t * final_ch * 4), cur, cur_t * final_ch);
        try b.copy(joined.slice((pad_floats + cur_t * final_ch) * 4, pad_floats * 4), zeros, pad_floats);
        const c2_cols = self.conv2_mat.matrix.columns; // 256
        try b.ttsGather(joined, gathered, cur_t, final_ch, 7, c2_cols, 0, 1);
        try b.matmulTile(self.conv2_mat.buffer, self.conv2_mat.matrix, gathered, c2_cols, out_raw, 8, cur_t);

        try b.commit();

        const raw_floats = out_raw.floats();
        const samples = try gpa.alloc(f32, cur_t);
        const bias = self.conv2_bias;
        for (samples, 0..) |*dst, t| dst.* = raw_floats[t * 8] + bias;
        return samples;
    }

    fn runNonCausalConv(
        self: *Decoder,
        c: NonCausalConv,
        x: Buffer,
        t_len: usize,
        y: Buffer,
        joined: Buffer,
        gathered: Buffer,
        zeros: Buffer,
    ) !void {
        const b = self.device.backend;
        if (c.k == 1 and c.padding == 0 and c.mat.matrix.columns == c.in) {
            return self.device.mm(c.mat, x, c.in, y, t_len);
        }
        if (c.padding == 0) {
            try b.ttsGather(x, gathered, t_len, c.in, c.k, c.mat.matrix.columns, 0, @intCast(c.dilation));
            return self.device.mm(c.mat, gathered, c.mat.matrix.columns, y, t_len);
        }
        const pad_floats = c.padding * c.in;
        const x_floats = t_len * c.in;
        try b.copy(joined, zeros, pad_floats);
        try b.copy(joined.slice(pad_floats * 4, x_floats * 4), x, x_floats);
        try b.copy(joined.slice((pad_floats + x_floats) * 4, pad_floats * 4), zeros, pad_floats);
        try b.ttsGather(joined, gathered, t_len, c.in, c.k, c.mat.matrix.columns, 0, @intCast(c.dilation));
        try self.device.mm(c.mat, gathered, c.mat.matrix.columns, y, t_len);
    }

    fn runNonCausalTrans(
        self: *Decoder,
        tr: NonCausalTrans,
        x: Buffer,
        t_in: usize,
        y: Buffer,
        joined: Buffer,
        gathered: Buffer,
        zeros: Buffer,
    ) !void {
        const b = self.device.backend;
        const pad_floats = tr.in; // 1 zero row left, 1 zero row right
        const x_floats = t_in * tr.in;
        try b.copy(joined, zeros, pad_floats);
        try b.copy(joined.slice(pad_floats * 4, x_floats * 4), x, x_floats);
        try b.copy(joined.slice((pad_floats + x_floats) * 4, pad_floats * 4), zeros, pad_floats);
        try b.ttsGather(joined, gathered, t_in, tr.in, 3, tr.mat.matrix.columns, 0, 1);
        try b.matmulTile(tr.mat.buffer, tr.mat.matrix, gathered, tr.mat.matrix.columns, y, tr.stride * tr.out, t_in);
        if (tr.bias) |bias| {
            try b.addBiasRows(y, bias, tr.out, t_in * tr.stride, tr.out);
        }
    }
};

fn uploadSnake(device: *cm.Device, gpa: Allocator, alpha: []const f32) !Snake {
    const inv = try gpa.alloc(f32, alpha.len);
    defer gpa.free(inv);
    for (inv, alpha) |*d, a| d.* = 1.0 / (a + hc.snake_eps);
    return .{ .scale = try device.upload(alpha), .inv = try device.upload(inv) };
}

fn uploadConv(device: *cm.Device, c: hc.Conv1d) !NonCausalConv {
    const cols = c.k * c.in;
    return .{
        .mat = try device.linear(c.w, c.b, c.out, cols),
        .in = c.in,
        .out = c.out,
        .k = c.k,
        .padding = c.padding,
        .dilation = c.dilation,
    };
}

/// Folds `ConvTranspose1d` (`k = 2*s`, `p = (s+1)/2`, `op = s % 2`, `cpu.w` is
/// `[k * out][in]`) into a 3-tap gather matrix `[s * out][3 * in]` so one
/// `ttsGather(taps=3)` + one `matmulTile` computes all `t_in * s` output rows.
fn uploadTrans(device: *cm.Device, gpa: Allocator, c: hc.ConvTranspose1d) !NonCausalTrans {
    const s = c.stride;
    const rows = s * c.out;
    const cols = 3 * c.in;
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    @memset(w, 0);

    for (0..s) |r| {
        const u = r + c.padding; // 1 <= u < 2*s
        for (0..c.out) |o| {
            const dst_row = w[(r * c.out + o) * cols ..][0..cols];
            // tap 0 (q - 1): kernel tap u + s (when u < s)
            if (u < s) {
                const src = c.w[((u + s) * c.out + o) * c.in ..][0..c.in];
                @memcpy(dst_row[0..c.in], src);
            }
            // tap 1 (q): kernel tap u (always valid: 0 <= u < 2*s)
            {
                const src = c.w[(u * c.out + o) * c.in ..][0..c.in];
                @memcpy(dst_row[c.in .. 2 * c.in], src);
            }
            // tap 2 (q + 1): kernel tap u - s (when u >= s)
            if (u >= s) {
                const src = c.w[((u - s) * c.out + o) * c.in ..][0..c.in];
                @memcpy(dst_row[2 * c.in .. 3 * c.in], src);
            }
        }
    }

    return .{
        .mat = try device.linear(w, null, rows, cols),
        .bias = if (c.b) |bias| try device.upload(bias) else null,
        .in = c.in,
        .out = c.out,
        .stride = s,
    };
}
