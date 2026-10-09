//! Embeddings: one input in, one unit vector out. `Embedder` opens an
//! embedding-family GGUF file (today EmbeddingGemma 2), turns an input's
//! ordered parts into its rows (tokenization, BOS/EOS framing, the
//! `max_tokens` limit), and runs the model's forward. Text is literal: a
//! control token's spelling in a part is text, never the token
//! (docs/models/embeddinggemma.md § The input contract). The Embedder lives
//! on the heap because its encoder and runtime hold pointers into it.
const std = @import("std");
const gguf = @import("formats/gguf.zig");
const weights = @import("runtime/weights.zig");
const vocabulary = @import("tokenizer/vocabulary.zig");
const Encoder = @import("tokenizer/encode.zig").Encoder;
const model = @import("models/embeddinggemma.zig");
const runtime = @import("models/embeddinggemma_runtime.zig");
const metal_plan = @import("models/embeddinggemma_metal.zig");

pub const max_tokens = model.max_tokens;
/// The vector width the model writes.
pub const dimensions = model.config.embedding_out;
/// The Matryoshka widths the model is trained for, widest first.
pub const widths = [_]usize{ 768, 512, 256, 128 };

pub const Stage = runtime.Stage;
/// Where the forward runs: the F32 CPU reference, or the Metal plan (packed
/// batches, F32 activations; embeddinggemma_metal.zig).
pub const Backend = enum { cpu, metal };
pub const Observer = runtime.Observer;

pub const Part = union(enum) {
    text: []const u8,
};
pub const Input = struct { parts: []const Part };

pub const Options = struct {
    /// Cut an input over `max_tokens` to fit, keeping its final EOS, instead
    /// of refusing it.
    truncate: bool = false,
};

/// An input's rows, owning its tokens; it borrows nothing from the input.
pub const Prepared = struct {
    tokens: []u32,
    /// The token count before truncation; equal to `tokens.len` unless cut.
    length: usize,

    pub fn truncated(self: Prepared) bool {
        return self.length != self.tokens.len;
    }
    pub fn deinit(self: *Prepared, alloc: std.mem.Allocator) void {
        alloc.free(self.tokens);
        self.* = undefined;
    }
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

    /// Opens an EmbeddingGemma 2 file for `backend`. A file of another
    /// architecture is `NotAnEmbeddingModel`. Caller owns the result;
    /// `deinit` frees it.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8, backend: Backend) !*Embedder {
        const self = try gpa.create(Embedder);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
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
        self.engine = switch (backend) {
            .cpu => .{ .cpu = try .init(gpa, io, self.mapped.view(), &self.binding) },
            .metal => .{ .metal = try metal_plan.Plan.create(gpa, self.mapped.view(), &self.binding) },
        };
        return self;
    }

    pub fn deinit(self: *Embedder) void {
        const gpa = self.gpa;
        switch (self.engine) {
            .cpu => |*r| r.deinit(),
            .metal => |plan| plan.destroy(),
        }
        self.encoder.deinit();
        self.vocab.deinit();
        self.mapped.deinit(self.io);
        gpa.destroy(self);
    }

    /// The input's tokens, framed: BOS, each run of adjacent text parts
    /// tokenized as one string, EOS. Over `max_tokens` it is `InputTooLong`
    /// unless `options.truncate`. Caller owns the result.
    pub fn prepare(self: *const Embedder, alloc: std.mem.Allocator, input: Input, options: Options) !Prepared {
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(alloc);
        for (input.parts) |part| switch (part) {
            .text => |t| try text.appendSlice(alloc, t),
        };
        const body = try self.encoder.encode(alloc, text.items, false, .{});
        defer alloc.free(body);
        return frame(alloc, body, self.bos, self.eos, options);
    }

    /// Writes the unit vector (`dimensions` values) of a prepared input;
    /// `observer` sees each stage's rows.
    pub fn embed(self: *Embedder, prepared: Prepared, vector: []f32, observer: ?Observer) !void {
        switch (self.engine) {
            .cpu => |*r| try r.embed(.{ .tokens = prepared.tokens }, vector, observer),
            .metal => |plan| try plan.run(&.{.{ .tokens = prepared.tokens }}, &.{vector}, observer),
        }
    }

    /// `embed` of every input, `vectors[i]` for `inputs[i]`. The Metal plan
    /// packs consecutive inputs into batches of at most `metal_plan.max_rows`
    /// rows (`batchRows` each), which changes no vector; the CPU runs them
    /// one at a time.
    pub fn embedBatch(self: *Embedder, inputs: []const Prepared, vectors: []const []f32) !void {
        if (inputs.len != vectors.len) return error.InvalidShape;
        switch (self.engine) {
            .cpu => |*r| for (inputs, vectors) |p, v| try r.embed(.{ .tokens = p.tokens }, v, null),
            .metal => |plan| {
                const rows = try self.gpa.alloc(runtime.Input, inputs.len);
                defer self.gpa.free(rows);
                for (rows, inputs) |*r, p| r.* = .{ .tokens = p.tokens };
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

/// BOS, `body`, EOS, cut to `max_tokens` (keeping EOS) when `options.truncate`.
fn frame(alloc: std.mem.Allocator, body: []const u32, bos: ?u32, eos: ?u32, options: Options) !Prepared {
    const framing: usize = @as(usize, @intFromBool(bos != null)) + @intFromBool(eos != null);
    const length = body.len + framing;
    if (length > max_tokens and !options.truncate) return error.InputTooLong;
    const kept = @min(body.len, max_tokens - framing);
    const tokens = try alloc.alloc(u32, kept + framing);
    var at: usize = 0;
    if (bos) |b| {
        tokens[0] = b;
        at = 1;
    }
    @memcpy(tokens[at..][0..kept], body[0..kept]);
    if (eos) |e| tokens[tokens.len - 1] = e;
    return .{ .tokens = tokens, .length = length };
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

test "framing adds BOS and EOS, refuses an input over the limit, and truncation keeps EOS" {
    const alloc = std.testing.allocator;
    var short = try frame(alloc, &.{ 10, 11 }, 2, 1, .{});
    defer short.deinit(alloc);
    try std.testing.expectEqualSlices(u32, &.{ 2, 10, 11, 1 }, short.tokens);
    try std.testing.expect(!short.truncated());

    const body = try alloc.alloc(u32, max_tokens - 1);
    defer alloc.free(body);
    @memset(body, 7);
    try std.testing.expectError(error.InputTooLong, frame(alloc, body, 2, 1, .{}));
    var cut = try frame(alloc, body, 2, 1, .{ .truncate = true });
    defer cut.deinit(alloc);
    try std.testing.expectEqual(@as(usize, max_tokens), cut.tokens.len);
    try std.testing.expectEqual(@as(usize, max_tokens + 1), cut.length);
    try std.testing.expectEqual(@as(u32, 1), cut.tokens[max_tokens - 1]);
    try std.testing.expect(cut.truncated());

    // Exactly at the limit is accepted as is.
    var full = try frame(alloc, body[0 .. max_tokens - 2], 2, 1, .{});
    defer full.deinit(alloc);
    try std.testing.expect(!full.truncated());
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
