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
const rnnt_boost = @import("boost.zig");
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

    /// `phrases` against this model's pieces at `bonus` (owned; `deinit`).
    pub fn boost(self: *const Model, alloc: std.mem.Allocator, phrases: []const []const u8, bonus: f32, follow: f32) rnnt_boost.Error!rnnt_boost.Boost {
        return rnnt_boost.Boost.init(alloc, phrases, bonus, follow, &self.table);
    }

    /// The prompt id of a `prompt_dictionary` key (`de-DE`, `de`, `auto`).
    pub fn promptId(self: *const Model, locale: []const u8) ?usize {
        for (self.prompts) |p| if (std.mem.eql(u8, p.locale, locale)) return p.id;
        return null;
    }

    /// The prompts of a comma-separated list of locales (`ar,en-US,auto`)
    /// into `out`; their count. Each is a `prompt_dictionary` key, named
    /// once; at most `max_languages`; `auto` only last.
    pub fn languages(self: *const Model, list: []const u8, out: *[max_languages]usize) error{ UnknownLanguage, TooManyLanguages, DuplicateLanguage, FallbackNotLast }!usize {
        var n: usize = 0;
        var names = std.mem.splitScalar(u8, list, ',');
        while (names.next()) |raw| {
            const name = std.mem.trim(u8, raw, " ");
            const p = self.promptId(name) orelse return error.UnknownLanguage;
            if (n == max_languages) return error.TooManyLanguages;
            if (std.mem.indexOfScalar(usize, out[0..n], p) != null) return error.DuplicateLanguage;
            if (n > 0 and out[n - 1] == self.default_prompt) return error.FallbackNotLast;
            out[n] = p;
            n += 1;
        }
        return n;
    }

    /// The language tag's locale (`ar-AR`) that names `prompt`, or null.
    pub fn localeOf(self: *const Model, prompt: usize) ?[]const u8 {
        for (self.tags) |tag| if (tag) |t| if (self.promptId(t) == prompt) return t;
        return null;
    }

    /// Whether token `id` writes text: not special, not only spaces.
    pub fn writesText(self: *const Model, id: u32) bool {
        if (self.table.isSpecial(id)) return false;
        var rest = self.table.piece(id) orelse return false;
        while (std.mem.startsWith(u8, rest, self.table.replacement)) rest = rest[self.table.replacement.len..];
        return std.mem.trim(u8, rest, " ").len > 0;
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

/// Languages one transcriber listens for at once (`setLanguages`).
pub const max_languages = nemotron_metal.max_prompts;
/// Rows without text from the language that has the words, after which the
/// next words choose again: 1.04 s, the transcript segmenter's pause.
pub const choose_again_rows = 13;
/// Rows after a settled stretch's last text within which another
/// language's word is the same word, heard a row or two later: 320 ms.
pub const echo_rows = 4;

/// One utterance stream over a shared `Model`, on either engine (they gave
/// the same tokens on every clip measured: nemotron-asr.md § On Metal). Not
/// thread-safe; on Metal, one transcriber of a model records at a time.
pub const Transcriber = struct {
    alloc: std.mem.Allocator,
    model: *const Model,
    lookahead: usize,
    /// The first language's prompt.
    prompt: usize,
    run: Run,
    /// Every token emitted since the last reset, language tags included.
    /// With several languages, the tokens of the language chosen for each
    /// stretch of words (`setLanguages`).
    tokens: std.ArrayList(rnnt.Token) = .empty,
    /// The encoder's rows of the push in progress (CPU).
    rows: std.ArrayList(f32) = .empty,
    /// With several languages: the first one's tokens, the CPU's other
    /// decoders (Metal's are its stream's), one prompt's rows, the choice.
    own: std.ArrayList(rnnt.Token) = .empty,
    alternates: std.ArrayList(Alternate) = .empty,
    prompted: std.ArrayList(f32) = .empty,
    choice: Choice = .{},

    pub const Run = union(Engine) {
        cpu: Cpu,
        metal: nemotron_metal.Stream,
    };
    const Cpu = struct { stream: encoder.Stream, decoder: rnnt.Decoder };

    const Alternate = struct {
        prompt: usize,
        decoder: rnnt.Decoder,
        tokens: std.ArrayList(rnnt.Token) = .empty,
    };

    /// The words' language (docs/models/nemotron-asr.md § Several
    /// languages). A stretch is words up to a pause; its language may change
    /// until the pause settles it.
    const Choice = struct {
        /// The open stretch's language, once one wrote text in it.
        chosen: ?usize = null,
        /// The one chosen last.
        last: ?usize = null,
        /// Per language: where its tokens of the open stretch begin.
        start: [max_languages]usize = @splat(0),
        /// Per language: its decoder's score when the stretch opened.
        since: [max_languages]f64 = @splat(0),
        /// `tokens` before this never change.
        settled: usize = 0,
    };

    const Hypothesis = struct { prompt: usize, decoder: *rnnt.Decoder, tokens: *std.ArrayList(rnnt.Token) };

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
        self.dropAlternates();
        self.alternates.deinit(self.alloc);
        self.tokens.deinit(self.alloc);
        self.rows.deinit(self.alloc);
        self.own.deinit(self.alloc);
        self.prompted.deinit(self.alloc);
        self.* = undefined;
    }

    fn dropAlternates(self: *Transcriber) void {
        for (self.alternates.items) |*a| {
            a.decoder.deinit();
            a.tokens.deinit(self.alloc);
        }
        self.alternates.clearRetainingCapacity();
    }

    /// Shows each encoder stage's rows and each greedy step as they happen
    /// (the first language's).
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

    /// The languages it listens for, decoded over one encoder pass.
    pub fn languageCount(self: *const Transcriber) usize {
        return 1 + switch (self.run) {
            .cpu => self.alternates.items.len,
            .metal => |s| s.alternates.items.len,
        };
    }

    fn hypothesis(self: *Transcriber, i: usize) Hypothesis {
        if (i == 0) return .{ .prompt = self.prompt, .decoder = self.firstDecoder(), .tokens = &self.own };
        switch (self.run) {
            .cpu => {
                const a = &self.alternates.items[i - 1];
                return .{ .prompt = a.prompt, .decoder = &a.decoder, .tokens = &a.tokens };
            },
            .metal => |*s| {
                const a = &s.alternates.items[i - 1];
                return .{ .prompt = a.prompt, .decoder = &a.decoder, .tokens = &a.tokens };
            },
        }
    }

    fn firstDecoder(self: *Transcriber) *rnnt.Decoder {
        return switch (self.run) {
            .cpu => |*r| &r.decoder,
            .metal => |*s| &s.decoder,
        };
    }

    /// Appends 16 kHz samples; returns how many tokens the rows they
    /// completed emitted.
    pub fn push(self: *Transcriber, pcm: []const f32) !usize {
        const before = self.tokens.items.len;
        const several = self.languageCount() > 1;
        const out = if (several) &self.own else &self.tokens;
        switch (self.run) {
            .cpu => |*r| {
                self.rows.clearRetainingCapacity();
                try r.stream.push(pcm, &self.rows);
                try self.decodeCpu(r, out);
            },
            .metal => |*s| try s.push(pcm, out),
        }
        if (several) return if (try self.merge(false)) @max(1, self.tokens.items.len -| before) else 0;
        return self.tokens.items.len - before;
    }

    /// Ends the utterance as a whole-clip pass ends; returns the tokens of
    /// the last rows. Further pushes need a `reset`.
    pub fn finish(self: *Transcriber) !usize {
        const before = self.tokens.items.len;
        const several = self.languageCount() > 1;
        const out = if (several) &self.own else &self.tokens;
        switch (self.run) {
            .cpu => |*r| {
                self.rows.clearRetainingCapacity();
                try r.stream.finish(&self.rows);
                try self.decodeCpu(r, out);
            },
            .metal => |*s| try s.finish(out),
        }
        if (several) return if (try self.merge(true)) @max(1, self.tokens.items.len -| before) else 0;
        return self.tokens.items.len - before;
    }

    /// The CPU's rows through every language's prompt block and decoder.
    fn decodeCpu(self: *Transcriber, r: *Cpu, out: *std.ArrayList(rnnt.Token)) !void {
        if (!r.stream.raw) return r.decoder.push(self.rows.items, out);
        if (self.rows.items.len == 0) return;
        try self.prompted.resize(self.alloc, self.rows.items.len);
        try self.model.encoder.applyPrompt(self.alloc, self.rows.items, self.prompt, self.prompted.items);
        try r.decoder.push(self.prompted.items, out);
        for (self.alternates.items) |*a| {
            try self.model.encoder.applyPrompt(self.alloc, self.rows.items, a.prompt, self.prompted.items);
            try a.decoder.push(self.prompted.items, &a.tokens);
        }
    }

    /// The open stretch's tokens from the language that wrote them best:
    /// of the listed ones that wrote text, the highest score gain since
    /// the stretch opened; the fallback (`auto`) when none did, or when it
    /// gained more writing the same script as that one. A stretch settles
    /// `choose_again_rows` after its language's last text, or at the end.
    /// Returns whether the open stretch's tokens changed.
    fn merge(self: *Transcriber, final: bool) !bool {
        const c = &self.choice;
        const k = self.pick() orelse {
            if (final) self.settle(self.rowsDecoded());
            return false;
        };
        const list = self.hypothesis(k).tokens.items[c.start[k]..];
        const open = self.tokens.items.len - c.settled;
        const changed = c.chosen != k or list.len != open;
        if (c.chosen != k) {
            self.tokens.shrinkRetainingCapacity(c.settled);
            try self.tokens.appendSlice(self.alloc, list);
        } else try self.tokens.appendSlice(self.alloc, list[open..]);
        c.chosen = k;
        c.last = k;
        const spoke = self.lastText(list).?;
        if (final or self.rowsDecoded() >= spoke + choose_again_rows) self.settle(spoke);
        return changed;
    }

    fn pick(self: *Transcriber) ?usize {
        const c = &self.choice;
        var best: ?usize = null;
        var gain: f64 = 0;
        var fallback: ?usize = null;
        for (0..self.languageCount()) |i| {
            const h = self.hypothesis(i);
            if (self.lastText(h.tokens.items[c.start[i]..]) == null) continue;
            if (h.prompt == self.model.default_prompt) {
                fallback = i;
                continue;
            }
            const g = h.decoder.score - c.since[i];
            if (best == null or g > gain) {
                best = i;
                gain = g;
            }
        }
        const f = fallback orelse return best;
        const b = best orelse return f;
        const fh = self.hypothesis(f);
        if (fh.decoder.score - c.since[f] <= gain) return b;
        const script = self.firstScript(self.hypothesis(b).tokens.items[c.start[b]..]);
        return if (script == self.firstScript(fh.tokens.items[c.start[f]..])) f else b;
    }

    /// Settles the open stretch; every language's next stretch begins at
    /// its first word after row `spoke + echo_rows`.
    fn settle(self: *Transcriber, spoke: usize) void {
        const c = &self.choice;
        c.settled = self.tokens.items.len;
        c.chosen = null;
        const table = &self.model.table;
        for (0..self.languageCount()) |i| {
            const h = self.hypothesis(i);
            const list = h.tokens.items;
            var at = c.start[i];
            while (at < list.len and list[at].row <= spoke + echo_rows) at += 1;
            // Not into a word: its pieces up to the next word's go too.
            while (at < list.len and !table.isSpecial(list[at].id) and !std.mem.startsWith(u8, table.piece(list[at].id) orelse "", table.replacement)) at += 1;
            c.start[i] = at;
            c.since[i] = h.decoder.score;
        }
    }

    /// The row of the last token in `tokens` that writes text.
    fn lastText(self: *const Transcriber, tokens: []const rnnt.Token) ?usize {
        var i = tokens.len;
        while (i > 0) {
            i -= 1;
            if (self.model.writesText(tokens[i].id)) return tokens[i].row;
        }
        return null;
    }

    /// The script of the first letter `tokens` write.
    fn firstScript(self: *const Transcriber, tokens: []const rnnt.Token) Script {
        for (tokens) |t| {
            if (self.model.table.isSpecial(t.id)) continue;
            const s = scriptOf(self.model.table.piece(t.id) orelse continue);
            if (s != .none) return s;
        }
        return .none;
    }

    /// `tokens` up to here no longer change: the words before the open
    /// stretch with several languages, every token with one.
    pub fn settled(self: *const Transcriber) usize {
        return if (self.languageCount() > 1) self.choice.settled else self.tokens.items.len;
    }

    /// A new utterance: fresh encoder caches, the decoders from their
    /// start, no tokens.
    pub fn reset(self: *Transcriber) !void {
        switch (self.run) {
            .cpu => |*r| {
                var fresh = try encoder.Stream.init(self.alloc, &self.model.encoder, self.lookahead, self.prompt);
                fresh.observer = r.stream.observer;
                fresh.raw = r.stream.raw;
                r.stream.deinit();
                r.stream = fresh;
                try r.decoder.reset();
                for (self.alternates.items) |*a| {
                    try a.decoder.reset();
                    a.tokens.clearRetainingCapacity();
                }
            },
            .metal => |*s| try s.reset(),
        }
        self.tokens.clearRetainingCapacity();
        self.own.clearRetainingCapacity();
        self.choice = .{};
    }

    /// The language prompt from the next chunk on; the caches stay.
    pub fn setPrompt(self: *Transcriber, prompt: usize) !void {
        return self.setLanguages(&.{prompt});
    }

    /// The languages to listen for from the next chunk on, at most
    /// `max_languages` distinct prompts: each decodes the same encoder rows
    /// and the words take the one that writes them (`merge`). The model's
    /// default prompt (`auto`) may only come last, as the fallback. The
    /// caches and the tokens so far stay; the next words choose afresh.
    pub fn setLanguages(self: *Transcriber, prompts: []const usize) !void {
        if (prompts.len == 0 or prompts.len > max_languages) return error.TooManyLanguages;
        const limit = self.model.encoder.weights.config.prompts;
        for (prompts, 0..) |p, i| {
            if (p >= limit) return error.UnsupportedPrompt;
            if (std.mem.indexOfScalar(usize, prompts[0..i], p) != null) return error.DuplicateLanguage;
            if (p == self.model.default_prompt and i + 1 != prompts.len) return error.FallbackNotLast;
        }
        const boost = self.firstDecoder().boost;
        switch (self.run) {
            .cpu => |*r| {
                r.stream.prompt = prompts[0];
                r.stream.raw = prompts.len > 1;
                self.dropAlternates();
                for (prompts[1..]) |p| {
                    var decoder = try rnnt.Decoder.init(self.alloc, self.model.io, &self.model.transducer_binding.weights);
                    errdefer decoder.deinit();
                    decoder.row = r.decoder.row;
                    try self.alternates.append(self.alloc, .{ .prompt = p, .decoder = decoder });
                }
            },
            .metal => |*s| {
                try s.setPrompt(prompts[0]);
                try s.setAlternates(prompts[1..]);
            },
        }
        self.prompt = prompts[0];
        self.own.clearRetainingCapacity();
        self.choice = .{ .settled = self.tokens.items.len };
        for (0..self.languageCount()) |i| self.choice.since[i] = self.hypothesis(i).decoder.score;
        self.setBoost(boost);
    }

    /// Phrases to lean toward from the next row on, or none; `boost` is
    /// borrowed and must outlive its use here.
    pub fn setBoost(self: *Transcriber, boost: ?*const rnnt_boost.Boost) void {
        for (0..self.languageCount()) |i| {
            const d = self.hypothesis(i).decoder;
            d.boost = boost;
            d.boosting.reset();
        }
    }

    /// Encoder rows decoded so far (80 ms each).
    pub fn rowsDecoded(self: *const Transcriber) usize {
        return switch (self.run) {
            .cpu => |r| r.decoder.row,
            .metal => |s| s.decoder.row,
        };
    }

    /// The greedy path's log probability since the last reset: the
    /// language chosen last's, else the first's.
    pub fn score(self: *const Transcriber) f64 {
        const i = self.choice.last orelse 0;
        return @constCast(self).hypothesis(i).decoder.score;
    }

    /// The text so far: special tokens (the language tags) skipped.
    pub fn writeText(self: *const Transcriber, out: *std.Io.Writer) std.Io.Writer.Error!void {
        var started = false;
        for (self.tokens.items) |t| try self.model.table.decodeNext(t.id, &started, out);
    }

    /// The locale of the last language tag emitted; with several
    /// languages and no tag, the locale of the one chosen last; else null.
    pub fn language(self: *const Transcriber) ?[]const u8 {
        var i = self.tokens.items.len;
        while (i > 0) {
            i -= 1;
            const id = self.tokens.items[i].id;
            if (id < self.model.tags.len) if (self.model.tags[id]) |tag| return tag;
        }
        if (self.languageCount() == 1) return null;
        const last = self.choice.last orelse return null;
        return self.model.localeOf(@constCast(self).hypothesis(last).prompt);
    }
};
/// A writing system, as far as telling one language's transcript from
/// another's needs.
pub const Script = enum { none, latin, greek, cyrillic, hebrew, arabic, indic, thai, hangul, cjk, other };

/// The script of the first letter in `text` (UTF-8); `.none` without one.
pub fn scriptOf(text: []const u8) Script {
    var it = (std.unicode.Utf8View.init(text) catch return .none).iterator();
    while (it.nextCodepoint()) |cp| {
        const s: Script = switch (cp) {
            'A'...'Z', 'a'...'z', 0xC0...0xD6, 0xD8...0xF6, 0xF8...0x24F, 0x1E00...0x1EFF => .latin,
            0x370...0x3FF => .greek,
            0x400...0x52F => .cyrillic,
            0x590...0x5FF => .hebrew,
            0x600...0x6FF, 0x750...0x77F, 0xFB50...0xFDFF, 0xFE70...0xFEFF => .arabic,
            0x900...0xDFF => .indic,
            0xE00...0xE7F => .thai,
            0x1100...0x11FF, 0x3130...0x318F, 0xAC00...0xD7AF => .hangul,
            0x3040...0x30FF, 0x4E00...0x9FFF => .cjk,
            // Digits, punctuation, symbols.
            else => if (cp < 0xC0 or cp == 0xD7 or cp == 0xF7 or (cp >= 0x2000 and cp <= 0x2BFF)) .none else .other,
        };
        if (s != .none) return s;
    }
    return .none;
}

test "a transcript's script is its first letter's" {
    try std.testing.expectEqual(Script.arabic, scriptOf("\u{2581}إزيك"));
    try std.testing.expectEqual(Script.latin, scriptOf("\u{2581}¿qué"));
    try std.testing.expectEqual(Script.cyrillic, scriptOf("Привет"));
    try std.testing.expectEqual(Script.indic, scriptOf("पकड़ने"));
    try std.testing.expectEqual(Script.hangul, scriptOf("몸ون"));
    try std.testing.expectEqual(Script.cjk, scriptOf("今日"));
    try std.testing.expectEqual(Script.none, scriptOf("\u{2581}, 42?"));
}

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
