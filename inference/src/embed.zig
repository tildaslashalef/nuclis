//! Embeddings: one input in, one unit vector out. `Embedder` opens an
//! embedding-family GGUF file (today EmbeddingGemma 2) and, optionally, its
//! mmproj; it turns an input's ordered parts into rows (tokenization,
//! BOS/EOS framing, each medium's begin marker, projector rows and end
//! marker, the `max_tokens` limit) and runs the model's forward. Text is
//! literal: a control token's spelling in a part is text, never the token
//! (docs/models/embeddinggemma.md § The input contract). Images and audio
//! are processed as Google's processor does, not as the chat path's
//! reference does. The Embedder lives on the heap because its encoder and
//! runtimes hold pointers into it.
const std = @import("std");
const gguf = @import("formats/gguf.zig");
const weights = @import("runtime/weights.zig");
const vocabulary = @import("tokenizer/vocabulary.zig");
const Encoder = @import("tokenizer/encode.zig").Encoder;
const model = @import("models/embeddinggemma.zig");
const runtime = @import("models/embeddinggemma_runtime.zig");
const metal_plan = @import("models/embeddinggemma_metal.zig");
const vision = @import("vision/root.zig");
const gemma4v = vision.gemma4;
const preprocess = vision.preprocess;
const audio = @import("audio/root.zig");
const gemma4a = audio.gemma4a;
const mel = audio.mel;

/// `general.architecture` of the files an `Embedder` opens.
pub const architecture = model.architecture;
pub const max_tokens = model.max_tokens;
/// The vector width the model writes.
pub const dimensions = model.config.embedding_out;
/// The Matryoshka widths the model is trained for, widest first.
pub const widths = [_]usize{ 768, 512, 256, 128 };
/// Soft tokens per image: Google's default, and the budgets its processor accepts.
pub const default_image_tokens: u32 = 280;
pub const image_budgets = gemma4v.google_budgets;
/// The longest audio part: Google's feature extractor cuts a clip at 30 s
/// (750 rows).
pub const max_audio_seconds = 30;
const max_audio_samples = max_audio_seconds * mel.sample_rate;

/// Rows an input of `tokens` tokens takes in a packed batch.
pub const batchRows = metal_plan.batchRows;

pub const Stage = runtime.Stage;
/// Where the forward runs: the F32 CPU reference, or the Metal plan (packed
/// batches, F32 activations; embeddinggemma_metal.zig).
pub const Backend = enum { cpu, metal };
pub const Observer = runtime.Observer;
pub const Span = vision.Span;

pub const Part = union(enum) {
    text: []const u8,
    /// An encoded image (any format `vision.image.decode` reads), borrowed.
    image: []const u8,
    /// An encoded clip (any format `audio.audio.decode` reads), borrowed.
    audio: []const u8,
};
pub const Input = struct { parts: []const Part };

pub const Options = struct {
    /// Cut an input over `max_tokens` to fit, keeping its final EOS, instead
    /// of refusing it. A media part that would be cut is dropped whole; an
    /// audio part over `max_audio_seconds` keeps its beginning, as Google's
    /// processor does, instead of being refused.
    truncate: bool = false,
    /// Soft tokens per image, one of `image_budgets`.
    image_tokens: u32 = default_image_tokens,
};

pub const OpenOptions = struct {
    /// The mmproj file whose encoders turn image and audio parts into rows;
    /// without one such a part is `NoProjector`.
    projector: ?[]const u8 = null,
};

/// An input's rows, owning its tokens and media rows; it borrows nothing
/// from the input.
pub const Prepared = struct {
    /// A media span's tokens are placeholders for its `features` rows.
    tokens: []u32,
    /// The token count before truncation; equal to `tokens.len` unless cut.
    length: usize,
    spans: []Span = &.{},
    /// Each span's rows in span order, `model.config.embedding` wide.
    features: []f32 = &.{},
    /// A clip over `max_audio_seconds` was cut (part of `length − tokens.len`).
    clipped: bool = false,

    pub fn truncated(self: Prepared) bool {
        return self.length != self.tokens.len;
    }
    pub fn deinit(self: *Prepared, alloc: std.mem.Allocator) void {
        alloc.free(self.tokens);
        alloc.free(self.spans);
        alloc.free(self.features);
        self.* = undefined;
    }
    fn rows(self: Prepared) runtime.Input {
        return .{ .tokens = self.tokens, .spans = self.spans, .features = self.features };
    }
};

/// The framing tokens of an image, looked up by their text.
const Markers = struct { begin: u32, placeholder: u32, end: u32 };

/// The mmproj, its vision encoder built on the first image, and its audio
/// encoder (when the file has one) built on the first clip.
const Projector = struct {
    mapped: weights.Mapped,
    binding: gemma4v.Binding,
    engine: ?union(Backend) {
        cpu: gemma4v.Runtime,
        metal: gemma4v.Plan,
    } = null,
    audio: ?gemma4a.Binding,
    listener: ?Listener = null,
};

/// The audio encoder and its mel bank, owned by the projector.
const Listener = struct {
    bank: *mel.Bank,
    engine: union(Backend) {
        cpu: gemma4a.Runtime,
        metal: gemma4a.Plan,
    },
};

pub const Embedder = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mapped: weights.Mapped,
    binding: model.Binding,
    vocab: vocabulary.Vocabulary,
    encoder: Encoder,
    engine: union(Backend) {
        cpu: runtime.Runtime,
        /// Owns its Metal backend and every device buffer; `gpa`'s.
        metal: *metal_plan.Plan,
    },
    bos: ?u32,
    eos: ?u32,
    image_markers: Markers,
    audio_markers: Markers,
    projector: ?Projector,

    /// Opens an EmbeddingGemma 2 file for `backend`, with `options.projector`
    /// bound but its encoders not built. A file of another architecture is
    /// `NotAnEmbeddingModel`, a projector of another family
    /// `UnsupportedProjector`. Caller owns the result; `deinit` frees it.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8, backend: Backend, options: OpenOptions) !*Embedder {
        const self = try gpa.create(Embedder);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
        self.projector = null;
        self.mapped = try weights.Mapped.open(gpa, io, path);
        errdefer self.mapped.deinit(io);
        const doc = &self.mapped.document;
        if (!std.mem.eql(u8, doc.string("general.architecture") orelse "", model.architecture)) return error.NotAnEmbeddingModel;
        self.binding = try model.bind(gpa, doc);
        self.vocab = try vocabulary.load(gpa, doc.*, self.mapped.mapping.memory[0..@intCast(doc.directory_bytes)], .{});
        errdefer self.vocab.deinit();
        self.encoder = try Encoder.init(gpa, &self.vocab);
        errdefer self.encoder.deinit();
        self.bos = try added(doc, "tokenizer.ggml.add_bos_token", self.vocab.bos);
        self.eos = try added(doc, "tokenizer.ggml.add_eos_token", self.vocab.eos);
        self.image_markers = .{
            .begin = self.vocab.tokenId("<|image>") orelse return error.MissingSpecialToken,
            .placeholder = self.vocab.tokenId("<|image|>") orelse return error.MissingSpecialToken,
            .end = self.vocab.tokenId("<image|>") orelse return error.MissingSpecialToken,
        };
        self.audio_markers = .{
            .begin = self.vocab.tokenId("<|audio>") orelse return error.MissingSpecialToken,
            .placeholder = self.vocab.tokenId("<|audio|>") orelse return error.MissingSpecialToken,
            .end = self.vocab.tokenId("<audio|>") orelse return error.MissingSpecialToken,
        };
        if (options.projector) |projector_path| {
            var mapped = try weights.Mapped.open(gpa, io, projector_path);
            errdefer mapped.deinit(io);
            if (gemma4v.kindOf(&mapped.document) != .siglip) return error.UnsupportedProjector;
            var binding = try gemma4v.bind(gpa, &mapped.document);
            if (binding.output_width != model.config.embedding) return error.UnsupportedProjector;
            // Google's vision config, which the file does not record.
            binding.activation = .gelu_tanh;
            var heard: ?gemma4a.Binding = null;
            if (gemma4a.present(&mapped.document)) {
                heard = try gemma4a.bind(gpa, &mapped.document);
                if (heard.?.output_width != model.config.embedding) return error.UnsupportedProjector;
            }
            self.projector = .{ .mapped = mapped, .binding = binding, .audio = heard };
        }
        errdefer if (self.projector) |*p| p.mapped.deinit(io);
        self.engine = switch (backend) {
            .cpu => .{ .cpu = try .init(gpa, io, self.mapped.view(), &self.binding) },
            .metal => .{ .metal = try metal_plan.Plan.create(gpa, self.mapped.view(), &self.binding) },
        };
        return self;
    }

    pub fn deinit(self: *Embedder) void {
        const gpa = self.gpa;
        if (self.projector) |*p| {
            if (p.engine) |*e| switch (e.*) {
                inline else => |*x| x.deinit(),
            };
            if (p.listener) |*l| {
                switch (l.engine) {
                    inline else => |*x| x.deinit(),
                }
                gpa.destroy(l.bank);
            }
            p.mapped.deinit(self.io);
        }
        switch (self.engine) {
            .cpu => |*r| r.deinit(),
            .metal => |plan| plan.destroy(),
        }
        self.encoder.deinit();
        self.vocab.deinit();
        self.mapped.deinit(self.io);
        gpa.destroy(self);
    }

    /// Whether image parts can be embedded (a projector was opened).
    pub fn hasVision(self: *const Embedder) bool {
        return self.projector != null;
    }

    /// Whether audio parts can be embedded (the projector has an audio encoder).
    pub fn hasAudio(self: *const Embedder) bool {
        return if (self.projector) |p| p.audio != null else false;
    }

    /// Bytes the media encoders hold once built (the vision encoder by the
    /// first image, the audio encoder by the first clip): their weights,
    /// and on Metal the buffers their plans created; 0 before.
    pub fn mediaBytes(self: *const Embedder) u64 {
        const projector = if (self.projector) |*p| p else return 0;
        var total: u64 = 0;
        if (projector.engine) |engine| total += projector.binding.bytes + switch (engine) {
            .cpu => 0,
            .metal => |plan| plan.created_bytes,
        };
        if (projector.listener) |l| total += projector.audio.?.bytes + switch (l.engine) {
            .cpu => 0,
            .metal => |plan| plan.created_bytes,
        };
        return total;
    }

    /// The input's rows, framed: BOS, each run of adjacent text parts
    /// tokenized as one string, each image as BOI, its rows, EOI, then
    /// EOS. Images are encoded here, so this is not `const`: the first one
    /// builds the vision encoder. Over `max_tokens` it is `InputTooLong`
    /// unless `options.truncate`. Caller owns the result.
    pub fn prepare(self: *Embedder, alloc: std.mem.Allocator, input: Input, options: Options) !Prepared {
        var tokens: std.ArrayList(u32) = .empty;
        defer tokens.deinit(alloc);
        var spans: std.ArrayList(Span) = .empty;
        defer spans.deinit(alloc);
        var features: std.ArrayList(f32) = .empty;
        defer features.deinit(alloc);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(alloc);
        // Rows of audio past `max_audio_seconds` that truncation dropped.
        var cut: usize = 0;
        if (self.bos) |b| try tokens.append(alloc, b);
        for (input.parts) |part| switch (part) {
            .text => |t| try text.appendSlice(alloc, t),
            .image => |bytes| {
                try self.flushText(alloc, &text, &tokens);
                const rows = try self.encodeImage(alloc, bytes, options.image_tokens);
                defer alloc.free(rows);
                const count = rows.len / model.config.embedding;
                try tokens.append(alloc, self.image_markers.begin);
                try spans.append(alloc, .{ .start = tokens.items.len, .count = count, .width_tokens = 0, .height_tokens = 0 });
                try tokens.appendNTimes(alloc, self.image_markers.placeholder, count);
                try tokens.append(alloc, self.image_markers.end);
                try features.appendSlice(alloc, rows);
            },
            .audio => |bytes| {
                try self.flushText(alloc, &text, &tokens);
                const heard = try self.encodeAudio(alloc, bytes, options.truncate);
                defer alloc.free(heard.rows);
                cut += heard.cut;
                const count = heard.rows.len / model.config.embedding;
                try tokens.append(alloc, self.audio_markers.begin);
                try spans.append(alloc, .{ .start = tokens.items.len, .count = count, .width_tokens = 0, .height_tokens = 0 });
                try tokens.appendNTimes(alloc, self.audio_markers.placeholder, count);
                try tokens.append(alloc, self.audio_markers.end);
                try features.appendSlice(alloc, heard.rows);
            },
        };
        try self.flushText(alloc, &text, &tokens);
        var prepared = try finish(alloc, tokens.items, spans.items, features.items, self.eos, options);
        prepared.length += cut;
        prepared.clipped = cut != 0;
        return prepared;
    }

    fn flushText(self: *Embedder, alloc: std.mem.Allocator, text: *std.ArrayList(u8), tokens: *std.ArrayList(u32)) !void {
        if (text.items.len == 0) return;
        const body = try self.encoder.encode(alloc, text.items, false, .{});
        defer alloc.free(body);
        try tokens.appendSlice(alloc, body);
        text.clearRetainingCapacity();
    }

    /// The projector rows of one encoded image under `budget` soft tokens,
    /// `model.config.embedding` wide. Caller owns the result.
    pub fn encodeImage(self: *Embedder, alloc: std.mem.Allocator, bytes: []const u8, budget: u32) ![]f32 {
        if (std.mem.indexOfScalar(u32, &image_budgets, budget) == null) return error.UnsupportedImageTokens;
        const projector = if (self.projector) |*p| p else return error.NoProjector;
        var decoded = try vision.image.decode(alloc, bytes);
        defer decoded.deinit(alloc);
        const grid = try gemma4v.googleGrid(.{ .width = decoded.width, .height = decoded.height }, budget);
        const target = grid.pixels();
        const resized = try preprocess.resizeWith(alloc, decoded, target, .bicubic, .torchvision);
        defer alloc.free(resized);
        // F32 patches of 2x − 1, as Google's embedder reads them.
        var options = gemma4v.patchOptions(.siglip, projector.binding.mean, projector.binding.std);
        options.half = false;
        var patches = try preprocess.patches(alloc, resized, target, options);
        defer patches.deinit(alloc);
        const rows = try alloc.alloc(f32, grid.tokens() * model.config.embedding);
        errdefer alloc.free(rows);
        try self.encodePatches(patches, grid, rows);
        return rows;
    }

    pub const Heard = struct {
        /// `model.config.embedding` wide, owned by the caller.
        rows: []f32,
        /// Rows the clip would have given past `max_audio_seconds`.
        cut: usize,
    };

    /// The projector rows of one encoded clip: 16 kHz mono samples, the
    /// log-mel frames, the audio encoder. A clip over `max_audio_seconds` is
    /// `AudioTooLong` unless `truncate`, which keeps its beginning.
    pub fn encodeAudio(self: *Embedder, alloc: std.mem.Allocator, bytes: []const u8, cut_long: bool) !Heard {
        const ear = try self.listener();
        var pcm = try audio.audio.decode(alloc, bytes, max_audio_samples);
        defer pcm.deinit(alloc);
        if (pcm.total > pcm.samples.len and !cut_long) return error.AudioTooLong;
        const frames = mel.frameCount(pcm.samples.len);
        if (frames == 0) return error.AudioTooShort;
        const features = try alloc.alloc(f32, frames * mel.filters);
        defer alloc.free(features);
        try ear.bank.features(pcm.samples, features);
        const count = gemma4a.tokensFor(frames);
        const rows = try alloc.alloc(f32, count * model.config.embedding);
        errdefer alloc.free(rows);
        try self.encodeFeatures(features, frames, rows);
        const whole = gemma4a.tokensFor(mel.frameCount(@intCast(@min(pcm.total, std.math.maxInt(usize)))));
        return .{ .rows = rows, .cut = whole -| count };
    }

    /// Runs the audio encoder (built on first use) over log-mel frames.
    pub fn encodeFeatures(self: *Embedder, features: []const f32, frames: usize, rows: []f32) !void {
        const ear = try self.listener();
        switch (ear.engine) {
            inline else => |*e| try e.encode(features, frames, rows),
        }
    }

    /// The audio encoder, built on first use.
    fn listener(self: *Embedder) !*Listener {
        const projector = if (self.projector) |*p| p else return error.NoProjector;
        const binding = if (projector.audio) |*b| b else return error.NoAudioEncoder;
        if (projector.listener) |*l| return l;
        const bank = try self.gpa.create(mel.Bank);
        errdefer self.gpa.destroy(bank);
        bank.init();
        projector.listener = .{ .bank = bank, .engine = switch (self.engine) {
            .cpu => .{ .cpu = try .init(self.gpa, self.io, projector.mapped.view(), binding) },
            .metal => |plan| .{ .metal = try .init(self.gpa, self.io, &plan.backend, projector.mapped.view(), binding) },
        } };
        return &projector.listener.?;
    }

    /// Runs the vision encoder (built on first use) over prepared patches.
    pub fn encodePatches(self: *Embedder, patches: preprocess.Patches, grid: gemma4v.Grid, rows: []f32) !void {
        const projector = if (self.projector) |*p| p else return error.NoProjector;
        if (projector.engine == null) projector.engine = switch (self.engine) {
            .cpu => .{ .cpu = try .init(self.gpa, self.io, projector.mapped.view(), &projector.binding) },
            .metal => |plan| .{ .metal = try .init(self.gpa, &plan.backend, projector.mapped.view(), &projector.binding) },
        };
        switch (projector.engine.?) {
            inline else => |*e| try e.encode(patches, grid, rows),
        }
    }

    /// Writes the unit vector (`dimensions` values) of a prepared input;
    /// `observer` sees each stage's rows.
    pub fn embed(self: *Embedder, prepared: Prepared, vector: []f32, observer: ?Observer) !void {
        switch (self.engine) {
            .cpu => |*r| try r.embed(prepared.rows(), vector, observer),
            .metal => |plan| try plan.run(&.{prepared.rows()}, &.{vector}, observer),
        }
    }

    /// `embed` of every input, `vectors[i]` for `inputs[i]`. The Metal plan
    /// packs consecutive inputs into batches of at most `metal_plan.max_rows`
    /// rows (`batchRows` each), which changes no vector; the CPU runs them
    /// one at a time.
    pub fn embedBatch(self: *Embedder, inputs: []const Prepared, vectors: []const []f32) !void {
        if (inputs.len != vectors.len) return error.InvalidShape;
        switch (self.engine) {
            .cpu => |*r| for (inputs, vectors) |p, v| try r.embed(p.rows(), v, null),
            .metal => |plan| {
                const rows = try self.gpa.alloc(runtime.Input, inputs.len);
                defer self.gpa.free(rows);
                for (rows, inputs) |*r, p| r.* = p.rows();
                var first: usize = 0;
                while (first < inputs.len) {
                    var last = first;
                    var used: usize = 0;
                    while (last < inputs.len and used + metal_plan.batchRows(inputs[last].tokens.len) <= metal_plan.max_rows) : (last += 1)
                        used += metal_plan.batchRows(inputs[last].tokens.len);
                    if (last == first) return error.InputTooLong;
                    try plan.run(rows[first..last], vectors[first..last], null);
                    first = last;
                }
            },
        }
    }
};

/// The id `key` adds, or null when the file says it adds none.
fn added(doc: *const gguf.Document, key: []const u8, id: ?u32) !?u32 {
    const adds = switch (doc.get(key) orelse return null) {
        .boolean => |b| b,
        else => return error.InvalidMetadata,
    };
    if (!adds) return null;
    return id orelse error.MissingSpecialToken;
}

/// Takes `tokens` (BOS and the body), `spans` and their `features`, and
/// returns them owned with EOS appended. Over `max_tokens` with
/// `options.truncate`, the body is cut to fit, keeping EOS; a span whose
/// framed run (its begin marker to its end marker) the cut would split is
/// dropped whole, with everything after it.
fn finish(alloc: std.mem.Allocator, tokens: []const u32, spans: []const Span, features: []const f32, eos: ?u32, options: Options) !Prepared {
    const closing: usize = @intFromBool(eos != null);
    const length = tokens.len + closing;
    var keep = tokens.len;
    var kept_spans = spans.len;
    if (length > max_tokens) {
        if (!options.truncate) return error.InputTooLong;
        keep = max_tokens - closing;
        for (spans, 0..) |s, i| {
            // The framed run is [start − 1, start + count + 1).
            if (s.start + s.count + 1 <= keep) continue;
            keep = @min(keep, s.start - 1);
            kept_spans = i;
            break;
        }
    }
    var rows: usize = 0;
    for (spans[0..kept_spans]) |s| rows += s.count;
    const width = model.config.embedding;
    if (rows * width > features.len) return error.InvalidShape;
    const out = try alloc.alloc(u32, keep + closing);
    errdefer alloc.free(out);
    @memcpy(out[0..keep], tokens[0..keep]);
    if (eos) |e| out[keep] = e;
    const out_spans = try alloc.dupe(Span, spans[0..kept_spans]);
    errdefer alloc.free(out_spans);
    const out_features = try alloc.dupe(f32, features[0 .. rows * width]);
    return .{ .tokens = out, .length = length, .spans = out_spans, .features = out_features };
}

/// Keeps the first `width` values of a unit vector and renormalizes them in
/// place (Matryoshka). `width` must be one of `widths`.
pub fn truncate(vector: []f32, width: usize) ![]f32 {
    if (std.mem.indexOfScalar(usize, &widths, width) == null or vector.len < width) return error.UnsupportedDimensions;
    const kept = vector[0..width];
    var norm: f64 = 0;
    for (kept) |v| norm += @as(f64, v) * v;
    norm = @sqrt(norm);
    if (!(norm > 0)) return error.NonFiniteOutput;
    for (kept) |*v| v.* = @floatCast(v.* / norm);
    return kept;
}

test "framing adds EOS, refuses an input over the limit, and truncation keeps EOS" {
    const alloc = std.testing.allocator;
    var short = try finish(alloc, &.{ 2, 10, 11 }, &.{}, &.{}, 1, .{});
    defer short.deinit(alloc);
    try std.testing.expectEqualSlices(u32, &.{ 2, 10, 11, 1 }, short.tokens);
    try std.testing.expect(!short.truncated());

    const body = try alloc.alloc(u32, max_tokens);
    defer alloc.free(body);
    @memset(body, 7);
    body[0] = 2;
    try std.testing.expectError(error.InputTooLong, finish(alloc, body, &.{}, &.{}, 1, .{}));
    var cut = try finish(alloc, body, &.{}, &.{}, 1, .{ .truncate = true });
    defer cut.deinit(alloc);
    try std.testing.expectEqual(@as(usize, max_tokens), cut.tokens.len);
    try std.testing.expectEqual(@as(usize, max_tokens + 1), cut.length);
    try std.testing.expectEqual(@as(u32, 1), cut.tokens[max_tokens - 1]);
    try std.testing.expect(cut.truncated());

    // Exactly at the limit is accepted as is.
    var full = try finish(alloc, body[0 .. max_tokens - 1], &.{}, &.{}, 1, .{});
    defer full.deinit(alloc);
    try std.testing.expect(!full.truncated());
}

test "truncation keeps a span that fits and drops one the cut would split" {
    const alloc = std.testing.allocator;
    const width = model.config.embedding;
    const body = try alloc.alloc(u32, max_tokens + 50);
    defer alloc.free(body);
    @memset(body, 7);
    // Span 0 at 10..20 fits; span 1's run (begin marker at max_tokens − 21) crosses the limit.
    const spans = [_]Span{
        .{ .start = 10, .count = 10, .width_tokens = 0, .height_tokens = 0 },
        .{ .start = max_tokens - 20, .count = 30, .width_tokens = 0, .height_tokens = 0 },
    };
    const features = try alloc.alloc(f32, 40 * width);
    defer alloc.free(features);
    @memset(features, 0.5);
    var cut = try finish(alloc, body, &spans, features, 1, .{ .truncate = true });
    defer cut.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), cut.spans.len);
    try std.testing.expectEqual(@as(usize, 10 * width), cut.features.len);
    // Cut at span 1's begin marker, then EOS.
    try std.testing.expectEqual(@as(usize, max_tokens - 21 + 1), cut.tokens.len);
    try std.testing.expectEqual(@as(u32, 1), cut.tokens[cut.tokens.len - 1]);
    // Without truncation the same input is refused.
    try std.testing.expectError(error.InputTooLong, finish(alloc, body, &spans, features, 1, .{}));
}

test "Matryoshka truncation keeps the leading values at unit length; other widths are refused" {
    var vector: [dimensions]f32 = undefined;
    for (&vector, 0..) |*v, i| v.* = @floatFromInt(i % 7 + 1);
    const kept = try truncate(&vector, 256);
    try std.testing.expectEqual(@as(usize, 256), kept.len);
    var norm: f64 = 0;
    for (kept) |v| norm += @as(f64, v) * v;
    try std.testing.expectApproxEqAbs(@as(f64, 1), norm, 1e-6);
    // Direction is kept: the ratio of two values is unchanged.
    try std.testing.expectApproxEqRel(@as(f32, 2), kept[1] / kept[0], 1e-6);
    try std.testing.expectError(error.UnsupportedDimensions, truncate(&vector, 300));
    try std.testing.expectError(error.UnsupportedDimensions, truncate(vector[0..100], 128));
}
