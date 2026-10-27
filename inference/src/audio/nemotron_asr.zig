//! Nemotron 3.5 ASR assembled from its checkpoint directory: the encoder
//! (nemotron_encoder.zig), the transducer (nemotron_rnnt.zig), the piece
//! table (tokenizer/pieces.zig) and the processor's language prompts. A
//! `Model` is shared and immutable once open (and its Metal plan built); a
//! `Transcriber` is one utterance stream: samples in, tokens out as each
//! chunk of rows completes, the text and the detected language read back at
//! any time, on the CPU reference or on Metal (`nemotron_metal.zig`, once
//! `Model.openMetal`). docs/models/nemotron-asr.md.
const std = @import("std");
const safetensors = @import("../formats/safetensors.zig");
const pieces = @import("../tokenizer/pieces.zig");
const encoder = @import("nemotron_encoder.zig");
const rnnt = @import("nemotron_rnnt.zig");
const nemotron_metal = @import("nemotron_metal.zig");

/// Bytes read from each JSON file of the directory, at most.
const max_json_bytes = 16 << 20;

pub const Prompt = struct { locale: []const u8, id: usize };

pub const Model = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    checkpoint: safetensors.Checkpoint,
    encoder_binding: encoder.Binding,
    transducer_binding: rnnt.Binding,
    encoder: encoder.Encoder,
    table: pieces.Table,
    /// `processor_config.json`'s `prompt_dictionary`, sorted by locale.
    prompts: []const Prompt,
    /// `config.json`'s `default_prompt_id` (`auto`).
    default_prompt: usize,
    /// By token id: the locale of a special language-tag piece, else null.
    tags: []const ?[]const u8,
    /// Owns `prompts` and `tags`.
    arena: std.heap.ArenaAllocator,
    /// The device weights, once `openMetal` built them.
    metal: ?*nemotron_metal.Plan = null,

    /// Opens the checkpoint directory (`config.json`, `processor_config.json`,
    /// `tokenizer.json`, `model.safetensors`), refusing what is not this
    /// architecture. The result is heap-allocated: the encoder points into it.
    pub fn open(alloc: std.mem.Allocator, io: std.Io, directory: []const u8) !*Model {
        const self = try alloc.create(Model);
        errdefer alloc.destroy(self);
        self.alloc = alloc;
        self.io = io;
        self.arena = .init(alloc);
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();
        var scratch: std.heap.ArenaAllocator = .init(alloc);
        defer scratch.deinit();
        const s = scratch.allocator();
        const dir = std.Io.Dir.cwd();

        const config_text = try dir.readFileAlloc(io, try std.fs.path.join(s, &.{ directory, "config.json" }), s, .limited(max_json_bytes));
        try expectArchitecture(s, config_text);
        const encoder_config = try encoder.parseConfig(alloc, config_text);
        const transducer_config = try rnnt.parseConfig(alloc, config_text);
        if (transducer_config.encoder != encoder_config.hidden) return error.UnsupportedConfig;
        const Defaults = struct { default_prompt_id: u32 };
        const defaults = std.json.parseFromSliceLeaky(Defaults, s, config_text, .{ .ignore_unknown_fields = true }) catch return error.InvalidConfig;
        if (defaults.default_prompt_id >= encoder_config.prompts) return error.InvalidConfig;
        self.default_prompt = defaults.default_prompt_id;

        const processor_text = try dir.readFileAlloc(io, try std.fs.path.join(s, &.{ directory, "processor_config.json" }), s, .limited(max_json_bytes));
        self.prompts = try parsePrompts(arena, processor_text, encoder_config.prompts);

        const tokenizer_text = try dir.readFileAlloc(io, try std.fs.path.join(s, &.{ directory, "tokenizer.json" }), s, .limited(max_json_bytes));
        self.table = try pieces.parse(alloc, tokenizer_text, .{});
        errdefer self.table.deinit();
        // The joint's outputs are the table's ids; the blank is its `<pad>`.
        if (self.table.count() < transducer_config.vocab) return error.VocabularyMismatch;
        const tags = try arena.alloc(?[]const u8, self.table.count());
        for (tags, 0..) |*t, id| t.* = if (self.table.isSpecial(@intCast(id))) rnnt.languageTag(self.table.piece(@intCast(id)).?) else null;
        self.tags = tags;

        self.checkpoint = try safetensors.Checkpoint.open(alloc, io, try std.fs.path.join(s, &.{ directory, "model.safetensors" }), .{});
        errdefer self.checkpoint.deinit(io);
        self.encoder_binding = try encoder.bind(alloc, &self.checkpoint, encoder_config);
        errdefer self.encoder_binding.deinit();
        self.transducer_binding = try rnnt.bind(alloc, &self.checkpoint, transducer_config);
        errdefer self.transducer_binding.deinit();
        self.encoder = try encoder.Encoder.init(alloc, io, &self.encoder_binding.weights);
        self.metal = null;
        return self;
    }

    /// Builds the Metal plan on a new backend, once: what a `.metal`
    /// transcriber runs on.
    pub fn openMetal(self: *Model) !void {
        if (self.metal != null) return;
        self.metal = try nemotron_metal.Plan.create(self.alloc, self.io, &self.encoder, &self.transducer_binding.weights);
    }

    pub fn close(self: *Model) void {
        const alloc = self.alloc;
        if (self.metal) |plan| plan.destroy();
        self.encoder.deinit();
        self.transducer_binding.deinit();
        self.encoder_binding.deinit();
        self.checkpoint.deinit(self.io);
        self.table.deinit();
        self.arena.deinit();
        alloc.destroy(self);
    }

    /// Device memory the Metal plan created (its converted weights and
    /// scratch), beside the mapped checkpoint; 0 on the CPU.
    pub fn deviceBytes(self: *const Model) u64 {
        return if (self.metal) |plan| plan.backend.created_bytes else 0;
    }

    /// The prompt id of a `prompt_dictionary` key (`de-DE`, `de`, `auto`).
    pub fn promptId(self: *const Model, locale: []const u8) ?usize {
        for (self.prompts) |p| if (std.mem.eql(u8, p.locale, locale)) return p.id;
        return null;
    }
};

fn expectArchitecture(scratch: std.mem.Allocator, config_text: []const u8) !void {
    const Json = struct { architectures: []const []const u8 = &.{} };
    const j = std.json.parseFromSliceLeaky(Json, scratch, config_text, .{ .ignore_unknown_fields = true }) catch return error.InvalidConfig;
    for (j.architectures) |a| if (std.mem.eql(u8, a, "Nemotron3_5AsrForRNNT")) return;
    return error.NotAnAsrModel;
}

/// `prompt_dictionary`: locale → prompt id, each id below `limit`. The
/// locales are copied into `arena`; `text` may be freed after.
fn parsePrompts(arena: std.mem.Allocator, text: []const u8, limit: usize) ![]const Prompt {
    const Json = struct { prompt_dictionary: std.json.ArrayHashMap(u32) };
    const j = std.json.parseFromSliceLeaky(Json, arena, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch return error.InvalidConfig;
    const map = j.prompt_dictionary.map;
    if (map.count() == 0 or map.count() > 4096) return error.InvalidConfig;
    const out = try arena.alloc(Prompt, map.count());
    for (map.keys(), map.values(), out) |k, v, *p| {
        if (v >= limit) return error.InvalidConfig;
        p.* = .{ .locale = k, .id = v };
    }
    std.mem.sortUnstable(Prompt, out, {}, struct {
        fn less(_: void, a: Prompt, b: Prompt) bool {
            return std.mem.lessThan(u8, a.locale, b.locale);
        }
    }.less);
    return out;
}

/// Where a transcriber runs.
pub const Engine = enum { cpu, metal };

/// One utterance stream over a shared `Model`, on either engine (they gave
/// the same tokens on every clip measured: nemotron-asr.md § On Metal). Not
/// thread-safe; on Metal, one transcriber of a model records at a time.
pub const Transcriber = struct {
    alloc: std.mem.Allocator,
    model: *const Model,
    lookahead: usize,
    prompt: usize,
    run: Run,
    /// Every token emitted since the last reset, language tags included.
    tokens: std.ArrayList(rnnt.Token) = .empty,
    /// The encoder's rows of the push in progress (CPU).
    rows: std.ArrayList(f32) = .empty,

    pub const Run = union(Engine) {
        cpu: struct { stream: encoder.Stream, decoder: rnnt.Decoder },
        metal: nemotron_metal.Stream,
    };

    /// `.metal` needs `model.openMetal()` first.
    pub fn init(alloc: std.mem.Allocator, model: *const Model, lookahead: usize, prompt: usize, engine: Engine) !Transcriber {
        const run: Run = switch (engine) {
            .cpu => blk: {
                var stream = try encoder.Stream.init(alloc, &model.encoder, lookahead, prompt);
                errdefer stream.deinit();
                break :blk .{ .cpu = .{ .stream = stream, .decoder = try rnnt.Decoder.init(alloc, model.io, &model.transducer_binding.weights) } };
            },
            .metal => .{ .metal = try nemotron_metal.Stream.init(alloc, model.metal orelse return error.MetalNotOpen, lookahead, prompt) },
        };
        return .{ .alloc = alloc, .model = model, .lookahead = lookahead, .prompt = prompt, .run = run };
    }

    pub fn deinit(self: *Transcriber) void {
        switch (self.run) {
            .cpu => |*r| {
                r.stream.deinit();
                r.decoder.deinit();
            },
            .metal => |*s| s.deinit(),
        }
        self.tokens.deinit(self.alloc);
        self.rows.deinit(self.alloc);
        self.* = undefined;
    }

    /// Shows each encoder stage's rows and each greedy step as they happen.
    pub fn observe(self: *Transcriber, stages: ?encoder.Observer, steps: ?rnnt.Observer) void {
        switch (self.run) {
            .cpu => |*r| {
                r.stream.observer = stages;
                r.decoder.observer = steps;
            },
            .metal => |*s| {
                s.observer = stages;
                s.decoder.observer = steps;
            },
        }
    }

    /// Appends 16 kHz samples; returns how many tokens the rows they
    /// completed emitted.
    pub fn push(self: *Transcriber, pcm: []const f32) !usize {
        const before = self.tokens.items.len;
        switch (self.run) {
            .cpu => |*r| {
                self.rows.clearRetainingCapacity();
                try r.stream.push(pcm, &self.rows);
                try r.decoder.push(self.rows.items, &self.tokens);
            },
            .metal => |*s| try s.push(pcm, &self.tokens),
        }
        return self.tokens.items.len - before;
    }

    /// Ends the utterance as a whole-clip pass ends; returns the tokens of
    /// the last rows. Further pushes need a `reset`.
    pub fn finish(self: *Transcriber) !usize {
        const before = self.tokens.items.len;
        switch (self.run) {
            .cpu => |*r| {
                self.rows.clearRetainingCapacity();
                try r.stream.finish(&self.rows);
                try r.decoder.push(self.rows.items, &self.tokens);
            },
            .metal => |*s| try s.finish(&self.tokens),
        }
        return self.tokens.items.len - before;
    }

    /// A new utterance: fresh encoder caches, the decoder from its start,
    /// no tokens.
    pub fn reset(self: *Transcriber) !void {
        switch (self.run) {
            .cpu => |*r| {
                var fresh = try encoder.Stream.init(self.alloc, &self.model.encoder, self.lookahead, self.prompt);
                fresh.observer = r.stream.observer;
                r.stream.deinit();
                r.stream = fresh;
                try r.decoder.reset();
            },
            .metal => |*s| try s.reset(),
        }
        self.tokens.clearRetainingCapacity();
    }

    /// The language prompt from the next chunk on; the caches stay.
    pub fn setPrompt(self: *Transcriber, prompt: usize) !void {
        switch (self.run) {
            .cpu => |*r| {
                if (prompt >= self.model.encoder.weights.config.prompts) return error.UnsupportedPrompt;
                r.stream.prompt = prompt;
            },
            .metal => |*s| try s.setPrompt(prompt),
        }
        self.prompt = prompt;
    }

    /// Encoder rows decoded so far (80 ms each).
    pub fn rowsDecoded(self: *const Transcriber) usize {
        return switch (self.run) {
            .cpu => |r| r.decoder.row,
            .metal => |s| s.decoder.row,
        };
    }

    /// The text so far: special tokens (the language tags) skipped.
    pub fn writeText(self: *const Transcriber, out: *std.Io.Writer) std.Io.Writer.Error!void {
        var started = false;
        for (self.tokens.items) |t| try self.model.table.decodeNext(t.id, &started, out);
    }

    /// The locale of the last language tag emitted, or null.
    pub fn language(self: *const Transcriber) ?[]const u8 {
        var i = self.tokens.items.len;
        while (i > 0) {
            i -= 1;
            const id = self.tokens.items[i].id;
            if (id < self.model.tags.len) if (self.model.tags[id]) |tag| return tag;
        }
        return null;
    }
};

test "prompt_dictionary: sorted, bounded by the prompt count" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // The locales outlive the text they were read from.
    const text = try std.testing.allocator.dupe(u8, "{\"prompt_dictionary\":{\"de-DE\":9,\"auto\":101,\"en\":0},\"num_prompts\":128}");
    const prompts = try parsePrompts(a, text, 128);
    @memset(text, 'x');
    std.testing.allocator.free(text);
    try std.testing.expectEqual(@as(usize, 3), prompts.len);
    try std.testing.expectEqualStrings("de-DE", prompts[1].locale);
    try std.testing.expectEqualStrings("auto", prompts[0].locale);
    try std.testing.expectEqual(@as(usize, 9), prompts[1].id);
    try std.testing.expectError(error.InvalidConfig, parsePrompts(a, "{\"prompt_dictionary\":{\"x\":128}}", 128));
    try std.testing.expectError(error.InvalidConfig, parsePrompts(a, "{\"prompt_dictionary\":{}}", 128));
}

test "only Nemotron3_5AsrForRNNT's config is taken" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try expectArchitecture(arena.allocator(), "{\"architectures\":[\"Nemotron3_5AsrForRNNT\"]}");
    try std.testing.expectError(error.NotAnAsrModel, expectArchitecture(arena.allocator(), "{\"architectures\":[\"Qwen3TTSForConditionalGeneration\"]}"));
}
