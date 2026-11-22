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
        if (transducer_config.encoder != encoder_config.hidden) return error.UnsupportedConfiguration;
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
        const word_start = try self.transducer_binding.arena.allocator().alloc(bool, transducer_config.vocab);
        for (word_start, 0..) |*w, id| w.* = !self.table.isSpecial(@intCast(id)) and std.mem.startsWith(u8, self.table.piece(@intCast(id)) orelse "", self.table.replacement);
        self.transducer_binding.weights.word_start = word_start;
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

    /// A stable locale name for `prompt`: `"auto"` for the default, else the
    /// tag's locale, else the first `prompt_dictionary` key with this id.
    pub fn promptName(self: *const Model, prompt: usize) []const u8 {
        if (prompt == self.default_prompt) return "auto";
        if (self.localeOf(prompt)) |locale| return locale;
        for (self.prompts) |p| if (p.id == prompt) return p.locale;
        return "unknown";
    }

    /// Whether token `id` writes text: not special, not only spaces.
    pub fn writesText(self: *const Model, id: u32) bool {
        if (self.table.isSpecial(id)) return false;
        var rest = self.table.piece(id) orelse return false;
        while (std.mem.startsWith(u8, rest, self.table.replacement)) rest = rest[self.table.replacement.len..];
        return std.mem.trim(u8, rest, " ").len > 0;
    }

    /// Decodes `tokens` into `out`, skipping special tokens.
    pub fn writeTokensText(self: *const Model, tokens: []const rnnt.Token, out: *std.Io.Writer) std.Io.Writer.Error!void {
        var started = false;
        for (tokens) |t| try self.table.decodeNext(t.id, &started, out);
    }

    fn tokenInfo(self: *const Model, id: u32) Arbiter.TokenInfo {
        if (self.table.isSpecial(id)) return .{ .writes_text = false, .continuation = false, .script = .none };
        const piece = self.table.piece(id) orelse return .{ .writes_text = false, .continuation = false, .script = .none };
        return .{
            .writes_text = self.writesText(id),
            .continuation = !std.mem.startsWith(u8, piece, self.table.replacement),
            .script = scriptOf(piece),
        };
    }
};

/// Distinct encoder rows on which `tokens` wrote text.
pub fn countTextRows(model: *const Model, tokens: []const rnnt.Token) usize {
    var rows: usize = 0;
    var last: ?u32 = null;
    for (tokens) |t| {
        if (!model.writesText(t.id)) continue;
        if (last != t.row) {
            rows += 1;
            last = t.row;
        }
    }
    return rows;
}

/// One listed language's reading of the utterance (`Transcriber.alternatives`).
pub const Alternative = struct {
    /// The listed prompt's locale (`de-DE`, `ar-AR`, `en-US`, or `auto`).
    language: []const u8,
    /// Decoded text of this language's tokens, trimmed.
    text: []const u8,
    /// Non-special tokens emitted by this language.
    tokens: usize,
    /// Distinct encoder rows on which this language wrote text.
    text_rows: usize,
    /// For `auto`, the last language tag it emitted, if any.
    tag: ?[]const u8 = null,
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
/// Coverage margin for a later-listed language to replace an earlier one:
/// at least `coverage_num / coverage_den`× (1.5×) as many text rows and at
/// least `coverage_min_diff` (3) more text rows.
pub const coverage_num: usize = 3;
pub const coverage_den: usize = 2;
pub const coverage_min_diff: usize = 3;

/// Pure row-by-row multi-language arbiter over per-language token lists
/// (docs/models/nemotron-asr.md § Prompt profile). List order is priority:
/// the earliest listed language with text in the open stretch keeps it unless
/// a later one out-covers it by `(coverage_num/coverage_den, coverage_min_diff)`;
/// `auto` (last) replaces only while writing the same script. Advancing by any
/// partition of rows produces the exact same output.
pub const Arbiter = struct {
    /// The open stretch's language, once one wrote text in it.
    chosen: ?usize = null,
    /// The one chosen last.
    last: ?usize = null,
    /// Per language: where its tokens of the open stretch begin.
    start: [max_languages]usize = @splat(0),
    /// Output tokens before this never change.
    settled: usize = 0,
    /// Rows already arbitrated (tokens with `row < arbitrated` have been seen).
    arbitrated: usize = 0,

    pub const TokenInfo = struct {
        writes_text: bool,
        continuation: bool,
        script: Script,
    };

    /// Replays row by row from `arbitrated` to `total_rows`, settling stretches
    /// that pause `choose_again_rows` (or on `final`), and updates `out`.
    /// Returns whether `out` changed.
    pub fn advance(
        self: *Arbiter,
        alloc: std.mem.Allocator,
        lists: []const []const rnnt.Token,
        has_fallback: bool,
        total_rows: usize,
        final: bool,
        ctx: anytype,
        comptime infoFn: fn (@TypeOf(ctx), u32) TokenInfo,
        out: *std.ArrayList(rnnt.Token),
    ) !bool {
        std.debug.assert(lists.len > 0 and lists.len <= max_languages);
        const before_len = out.items.len;
        const before_settled = self.settled;
        const before_chosen = self.chosen;
        var changed = false;

        var rows = self.arbitrated;
        while (rows < total_rows) {
            rows += 1;
            const k = self.pickAt(lists, has_fallback, rows, ctx, infoFn) orelse continue;
            self.chosen = k;
            self.last = k;
            const slice = openSlice(lists[k], self.start[k], rows);
            const spoke = lastTextRow(slice, ctx, infoFn).?;
            if (rows >= spoke + choose_again_rows) {
                try replaceTail(alloc, out, self.settled, slice);
                self.settleAt(lists, spoke, rows, out.items.len, ctx, infoFn);
                changed = true;
            }
        }
        self.arbitrated = total_rows;

        if (final) {
            if (self.chosen) |k| {
                const slice = openSlice(lists[k], self.start[k], total_rows);
                const spoke = lastTextRow(slice, ctx, infoFn).?;
                try replaceTail(alloc, out, self.settled, slice);
                self.settleAt(lists, spoke, total_rows, out.items.len, ctx, infoFn);
                changed = true;
            } else {
                self.settleAt(lists, total_rows, total_rows, out.items.len, ctx, infoFn);
            }
        } else if (self.chosen) |k| {
            const slice = openSlice(lists[k], self.start[k], total_rows);
            const open = out.items.len - self.settled;
            if (changed or before_chosen != k or before_settled != self.settled or slice.len != open) {
                try replaceTail(alloc, out, self.settled, slice);
                changed = true;
            }
        }
        return changed or out.items.len != before_len;
    }

    fn outcovers(candidate: usize, incumbent: usize) bool {
        return candidate * coverage_den >= incumbent * coverage_num and candidate >= incumbent + coverage_min_diff;
    }

    fn pickAt(
        self: *const Arbiter,
        lists: []const []const rnnt.Token,
        has_fallback: bool,
        rows: usize,
        ctx: anytype,
        comptime infoFn: fn (@TypeOf(ctx), u32) TokenInfo,
    ) ?usize {
        const listed = if (has_fallback) lists.len - 1 else lists.len;
        var best: ?usize = null;
        var best_rows: usize = 0;
        for (0..listed) |i| {
            const slice = openSlice(lists[i], self.start[i], rows);
            const r = sliceTextRows(slice, ctx, infoFn);
            if (r == 0) continue;
            if (best == null or outcovers(r, best_rows)) {
                best = i;
                best_rows = r;
            }
        }
        if (!has_fallback) return best;
        const f = lists.len - 1;
        const f_slice = openSlice(lists[f], self.start[f], rows);
        const f_rows = sliceTextRows(f_slice, ctx, infoFn);
        if (f_rows == 0) return best;
        const b = best orelse return f;
        if (!outcovers(f_rows, best_rows)) return b;
        const b_slice = openSlice(lists[b], self.start[b], rows);
        return if (firstScriptOf(b_slice, ctx, infoFn) == firstScriptOf(f_slice, ctx, infoFn)) f else b;
    }

    fn settleAt(
        self: *Arbiter,
        lists: []const []const rnnt.Token,
        spoke: usize,
        rows: usize,
        settled_len: usize,
        ctx: anytype,
        comptime infoFn: fn (@TypeOf(ctx), u32) TokenInfo,
    ) void {
        self.settled = settled_len;
        self.chosen = null;
        for (lists, 0..) |list, i| {
            const end = self.start[i] + openSlice(list, self.start[i], rows).len;
            var at = self.start[i];
            while (at < end and list[at].row <= spoke + echo_rows) at += 1;
            // Not into a word: its pieces up to the next word's go too, bounded to rows <= R.
            while (at < end and infoFn(ctx, list[at].id).continuation) at += 1;
            self.start[i] = at;
        }
    }

    fn openSlice(list: []const rnnt.Token, start: usize, rows: usize) []const rnnt.Token {
        if (start >= list.len) return &.{};
        var end = start;
        while (end < list.len and list[end].row < rows) end += 1;
        return list[start..end];
    }

    fn sliceTextRows(tokens: []const rnnt.Token, ctx: anytype, comptime infoFn: fn (@TypeOf(ctx), u32) TokenInfo) usize {
        var count: usize = 0;
        var last: ?u32 = null;
        for (tokens) |t| {
            if (!infoFn(ctx, t.id).writes_text) continue;
            if (last != t.row) {
                count += 1;
                last = t.row;
            }
        }
        return count;
    }

    fn lastTextRow(tokens: []const rnnt.Token, ctx: anytype, comptime infoFn: fn (@TypeOf(ctx), u32) TokenInfo) ?usize {
        var i = tokens.len;
        while (i > 0) {
            i -= 1;
            if (infoFn(ctx, tokens[i].id).writes_text) return tokens[i].row;
        }
        return null;
    }

    fn firstScriptOf(tokens: []const rnnt.Token, ctx: anytype, comptime infoFn: fn (@TypeOf(ctx), u32) TokenInfo) Script {
        for (tokens) |t| {
            const s = infoFn(ctx, t.id).script;
            if (s != .none) return s;
        }
        return .none;
    }

    fn replaceTail(alloc: std.mem.Allocator, out: *std.ArrayList(rnnt.Token), settled: usize, slice: []const rnnt.Token) !void {
        out.shrinkRetainingCapacity(settled);
        try out.appendSlice(alloc, slice);
    }
};

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
    arbiter: Arbiter = .{},

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

    /// Replays the row-by-row arbiter up to `rowsDecoded()` (settling on
    /// `final`). Returns whether `self.tokens` changed.
    fn merge(self: *Transcriber, final: bool) !bool {
        const n = self.languageCount();
        var lists: [max_languages][]const rnnt.Token = undefined;
        for (0..n) |i| lists[i] = self.languageTokens(i);
        const has_fallback = self.languagePrompt(n - 1) == self.model.default_prompt;
        return self.arbiter.advance(self.alloc, lists[0..n], has_fallback, self.rowsDecoded(), final, self.model, Model.tokenInfo, &self.tokens);
    }

    /// `tokens` up to here no longer change: the words before the open
    /// stretch with several languages, every token with one.
    pub fn settled(self: *const Transcriber) usize {
        return if (self.languageCount() > 1) self.arbiter.settled else self.tokens.items.len;
    }

    /// Rows up to which `tokens[0..settled()]` is complete: `rowsDecoded()`
    /// unless an open multi-language stretch holds back later tokens.
    pub fn settledRows(self: *const Transcriber) usize {
        if (self.languageCount() > 1 and self.arbiter.chosen != null and self.arbiter.settled < self.tokens.items.len) {
            return self.tokens.items[self.arbiter.settled].row;
        }
        return self.rowsDecoded();
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
        self.arbiter = .{};
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
        self.arbiter = .{ .settled = self.tokens.items.len, .arbitrated = self.rowsDecoded() };
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
        const i = self.arbiter.last orelse 0;
        return @constCast(self).hypothesis(i).decoder.score;
    }

    /// The text so far: special tokens (the language tags) skipped.
    pub fn writeText(self: *const Transcriber, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.model.writeTokensText(self.tokens.items, out);
    }

    /// The prompt id of the `i`-th listed language.
    pub fn languagePrompt(self: *const Transcriber, i: usize) usize {
        if (i == 0) return self.prompt;
        return switch (self.run) {
            .cpu => self.alternates.items[i - 1].prompt,
            .metal => |s| s.alternates.items[i - 1].prompt,
        };
    }

    /// Every token emitted by the `i`-th listed language since the last reset.
    pub fn languageTokens(self: *const Transcriber, i: usize) []const rnnt.Token {
        if (i == 0) return if (self.languageCount() > 1) self.own.items else self.tokens.items;
        return switch (self.run) {
            .cpu => self.alternates.items[i - 1].tokens.items,
            .metal => |s| s.alternates.items[i - 1].tokens.items,
        };
    }

    /// Each listed language's reading of the utterance, in list order; strings
    /// live in `arena`.
    pub fn alternatives(self: *const Transcriber, arena: std.mem.Allocator) ![]const Alternative {
        const n = self.languageCount();
        const out = try arena.alloc(Alternative, n);
        for (0..n) |i| {
            const prompt = self.languagePrompt(i);
            const toks = self.languageTokens(i);
            var buf: std.Io.Writer.Allocating = .init(arena);
            try self.model.writeTokensText(toks, &buf.writer);
            var count: usize = 0;
            var tag: ?[]const u8 = null;
            for (toks) |t| {
                if (self.model.table.isSpecial(t.id)) {
                    if (prompt == self.model.default_prompt and t.id < self.model.tags.len) {
                        if (self.model.tags[t.id]) |tg| tag = tg;
                    }
                } else count += 1;
            }
            out[i] = .{
                .language = self.model.promptName(prompt),
                .text = std.mem.trim(u8, buf.written(), " "),
                .tokens = count,
                .text_rows = countTextRows(self.model, toks),
                .tag = tag,
            };
        }
        return out;
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
        const last = self.arbiter.last orelse return null;
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

// ----- Arbiter unit tests (pure token lists, no model file) -----

fn mockTokenInfo(_: void, id: u32) Arbiter.TokenInfo {
    return switch (id) {
        // 1..99: Latin word starts; 100..199: Latin continuations.
        1...99 => .{ .writes_text = true, .continuation = false, .script = .latin },
        100...199 => .{ .writes_text = true, .continuation = true, .script = .latin },
        // 200..299: Arabic word starts.
        200...299 => .{ .writes_text = true, .continuation = false, .script = .arabic },
        // 300+: special / non-text tokens (e.g. language tags).
        else => .{ .writes_text = false, .continuation = false, .script = .none },
    };
}

fn runArbiter(
    alloc: std.mem.Allocator,
    lists: []const []const rnnt.Token,
    has_fallback: bool,
    total_rows: usize,
    step: usize,
) ![]rnnt.Token {
    var arbiter: Arbiter = .{};
    var out: std.ArrayList(rnnt.Token) = .empty;
    errdefer out.deinit(alloc);
    var cur: usize = 0;
    while (cur < total_rows) {
        cur = @min(total_rows, cur + step);
        _ = try arbiter.advance(alloc, lists, has_fallback, cur, false, {}, mockTokenInfo, &out);
    }
    _ = try arbiter.advance(alloc, lists, has_fallback, total_rows, true, {}, mockTokenInfo, &out);
    try std.testing.expectEqual(out.items.len, arbiter.settled);
    return out.toOwnedSlice(alloc);
}

test "arbiter: pushes of 1, 3, 4, 7, 25 rows equal one push" {
    const alloc = std.testing.allocator;
    // Stretch 1 (rows 1..4): lang 0 writes 4 rows, lang 1 writes an echo at row 6
    // plus a continuation piece at row 20 (past settle row 17) and a second stretch
    // at rows 24..30; lang 2 (auto) writes both stretches.
    const l0 = [_]rnnt.Token{
        .{ .id = 1, .row = 1 },
        .{ .id = 2, .row = 2 },
        .{ .id = 3, .row = 3 },
        .{ .id = 4, .row = 4 },
        .{ .id = 300, .row = 5 }, // tag
    };
    const l1 = [_]rnnt.Token{
        .{ .id = 10, .row = 6 }, // within echo_rows (4 + 4 = 8)
        .{ .id = 110, .row = 20 }, // continuation past settle row 17: must NOT be skipped
        .{ .id = 11, .row = 24 },
        .{ .id = 12, .row = 25 },
        .{ .id = 13, .row = 26 },
        .{ .id = 14, .row = 27 },
        .{ .id = 15, .row = 28 },
    };
    const l2 = [_]rnnt.Token{
        .{ .id = 20, .row = 1 },
        .{ .id = 21, .row = 2 },
        .{ .id = 22, .row = 3 },
        .{ .id = 23, .row = 4 },
        .{ .id = 24, .row = 24 },
        .{ .id = 25, .row = 25 },
        .{ .id = 26, .row = 26 },
    };
    const lists = [_][]const rnnt.Token{ &l0, &l1, &l2 };
    const total_rows: usize = 50;

    const whole = try runArbiter(alloc, &lists, true, total_rows, total_rows);
    defer alloc.free(whole);
    // Stretch 1 goes to l0 (4 rows vs l2's 4); stretch 2 goes to l1 (rows 20, 24..28).
    try std.testing.expectEqual(@as(usize, 11), whole.len);
    try std.testing.expectEqual(@as(u32, 1), whole[0].id);
    try std.testing.expectEqual(@as(u32, 110), whole[5].id);
    try std.testing.expectEqual(@as(u32, 15), whole[10].id);

    for ([_]usize{ 1, 3, 4, 7, 25 }) |step| {
        const piece = try runArbiter(alloc, &lists, true, total_rows, step);
        defer alloc.free(piece);
        try std.testing.expectEqualSlices(rnnt.Token, whole, piece);
    }
}

test "arbiter: list order flips an equal-coverage choice" {
    const alloc = std.testing.allocator;
    const de = [_]rnnt.Token{
        .{ .id = 1, .row = 1 },
        .{ .id = 2, .row = 2 },
        .{ .id = 3, .row = 3 },
        .{ .id = 4, .row = 4 },
    };
    const en = [_]rnnt.Token{
        .{ .id = 51, .row = 1 },
        .{ .id = 52, .row = 2 },
        .{ .id = 53, .row = 3 },
        .{ .id = 54, .row = 4 },
    };

    const de_first = try runArbiter(alloc, &.{ &de, &en }, false, 20, 4);
    defer alloc.free(de_first);
    try std.testing.expectEqualSlices(rnnt.Token, &de, de_first);

    const en_first = try runArbiter(alloc, &.{ &en, &de }, false, 20, 4);
    defer alloc.free(en_first);
    try std.testing.expectEqualSlices(rnnt.Token, &en, en_first);
}

test "arbiter: a 3-row prefix loses to a continuing language before row 13" {
    const alloc = std.testing.allocator;
    // Lang 0 writes a 3-row prefix; lang 1 continues through row 10.
    const prefix = [_]rnnt.Token{
        .{ .id = 1, .row = 0 },
        .{ .id = 2, .row = 1 },
        .{ .id = 3, .row = 2 },
    };
    const full = [_]rnnt.Token{
        .{ .id = 50, .row = 0 },
        .{ .id = 51, .row = 1 },
        .{ .id = 52, .row = 2 },
        .{ .id = 53, .row = 3 },
        .{ .id = 54, .row = 4 },
        .{ .id = 55, .row = 5 },
        .{ .id = 56, .row = 6 },
        .{ .id = 57, .row = 7 },
        .{ .id = 58, .row = 8 },
        .{ .id = 59, .row = 9 },
        .{ .id = 60, .row = 10 },
    };
    const lists = [_][]const rnnt.Token{ &prefix, &full };

    // At row 6 (well before row 13), lang 1 has 6 rows (>= 1.5*3 and >= 3+3) and takes over.
    var arbiter: Arbiter = .{};
    var out: std.ArrayList(rnnt.Token) = .empty;
    defer out.deinit(alloc);
    _ = try arbiter.advance(alloc, &lists, false, 6, false, {}, mockTokenInfo, &out);
    try std.testing.expectEqual(@as(?usize, 1), arbiter.chosen);
    try std.testing.expectEqual(@as(usize, 0), arbiter.settled);

    // Finish the clip: full wins the whole stretch, nothing dropped.
    _ = try arbiter.advance(alloc, &lists, false, 25, true, {}, mockTokenInfo, &out);
    try std.testing.expectEqualSlices(rnnt.Token, &full, out.items);
}

test "arbiter: auto fallback respects script and coverage margin" {
    const alloc = std.testing.allocator;
    // Arabic listed (3 rows across 1..7) vs auto writing Latin (7 rows across 1..7):
    // different script, so Arabic keeps the stretch despite auto's coverage margin.
    const ar = [_]rnnt.Token{
        .{ .id = 200, .row = 1 },
        .{ .id = 201, .row = 4 },
        .{ .id = 202, .row = 7 },
    };
    const auto_latin = [_]rnnt.Token{
        .{ .id = 1, .row = 1 },
        .{ .id = 2, .row = 2 },
        .{ .id = 3, .row = 3 },
        .{ .id = 4, .row = 4 },
        .{ .id = 5, .row = 5 },
        .{ .id = 6, .row = 6 },
        .{ .id = 7, .row = 7 },
    };
    const kept_ar = try runArbiter(alloc, &.{ &ar, &auto_latin }, true, 25, 4);
    defer alloc.free(kept_ar);
    try std.testing.expectEqualSlices(rnnt.Token, &ar, kept_ar);

    // Latin listed (3 rows across 1..7) vs auto writing Latin (7 rows across 1..7):
    // same script + coverage margin (7 >= 1.5*3 and 7 >= 3+3), so auto replaces.
    const en_short = [_]rnnt.Token{
        .{ .id = 50, .row = 1 },
        .{ .id = 51, .row = 4 },
        .{ .id = 52, .row = 7 },
    };
    const took_auto = try runArbiter(alloc, &.{ &en_short, &auto_latin }, true, 25, 4);
    defer alloc.free(took_auto);
    try std.testing.expectEqualSlices(rnnt.Token, &auto_latin, took_auto);

    // No listed language wrote: auto wins regardless of script.
    const empty = [_]rnnt.Token{};
    const only_auto = try runArbiter(alloc, &.{ &empty, &auto_latin }, true, 25, 4);
    defer alloc.free(only_auto);
    try std.testing.expectEqualSlices(rnnt.Token, &auto_latin, only_auto);
}
