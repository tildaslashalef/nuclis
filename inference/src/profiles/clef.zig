//! clef-flash's input contract, as Cloudflare's `joint_schema_model.py`
//! defines it at the pinned revision: question validation and option
//! rendering, and the one sequence that holds a state and every question
//! (system prompt, `STATE:`, the state, `SCHEMA FIELDS:` with one field per
//! question, the assistant's empty thinking block). Each piece is tokenized
//! on its own and concatenated, as the reference does, and the spans the
//! head averages are recorded while the schema is built. Pure: tokenizing
//! goes through the backbone GGUF's own tokenizer (the same vocabulary as
//! the repository's `tokenizer.json`), nothing here reads files or runs the
//! model. Contract: docs/reference/clef.md.
const std = @import("std");
const Encoder = @import("../tokenizer/encode.zig").Encoder;
const decision = @import("decision.zig");

pub const Kind = decision.Kind;
pub const Question = decision.Question;
pub const Diagnostic = decision.Diagnostic;
pub const QuestionError = decision.QuestionError;

pub const system_prompt = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
/// The reference's `max_length`: the whole sequence, state cut first.
pub const max_length = 16384;
pub const image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>";
/// `processor_config.json`'s `shortest_edge`: an image smaller than this
/// many pixels is scaled up (64 tokens of 32×32 pixels).
pub const min_image_pixels = 65536;

/// The tokens images become after the prefix: per image
/// `<|vision_start|>`, one `<|image_pad|>` per token, `<|vision_end|>`, then
/// the newline the reference's placeholder text ends with. `counts[i]` is
/// image `i`'s token count; empty `counts` gives no tokens.
pub fn mediaIds(arena: std.mem.Allocator, tokenizer: *const Encoder, counts: []const usize) ![]u32 {
    if (counts.len == 0) return &.{};
    const v = tokenizer.vocab;
    const start = v.tokenId("<|vision_start|>") orelse return error.MissingImageToken;
    const pad = v.tokenId("<|image_pad|>") orelse return error.MissingImageToken;
    const end = v.tokenId("<|vision_end|>") orelse return error.MissingImageToken;
    var out: std.ArrayList(u32) = .empty;
    for (counts) |n| {
        try out.append(arena, start);
        try out.appendNTimes(arena, pad, n);
        try out.append(arena, end);
    }
    try out.appendSlice(arena, try tokens(arena, tokenizer, "\n"));
    return out.items;
}

/// The head's type embedding row for a question kind.
pub fn typeId(kind: Kind) u32 {
    return switch (kind) {
        .noul => 0,
        .choice => 1,
        .score => 2,
    };
}

/// `render(value)`: a string as it is, anything else as
/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`.
pub fn render(arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |s| s,
        else => decision.pythonJsonStyled(arena, value, .{ .compact = true, .sort_keys = true }),
    };
}

/// Validates a question definition (`{"type", "instructions", "criteria"}`)
/// as `systemone` does and renders its options. The model reads them as
/// `question_options` orders them (noul `true`, `false`; choice keys sorted;
/// score levels by index), `model_order`.
/// `arena` owns the result; failures name `id` in `diag`.
pub fn parseQuestion(arena: std.mem.Allocator, id: []const u8, value: std.json.Value, diag: *Diagnostic) QuestionError!Question {
    const object = switch (value) {
        .object => |o| o,
        else => return fail(diag, "question {s}: definition must be an object", .{id}),
    };
    const kind_text = switch (object.get("type") orelse .null) {
        .string => |s| s,
        else => return fail(diag, "{s}: type must be noul, choice, or score", .{id}),
    };
    const kind = std.meta.stringToEnum(Kind, kind_text) orelse
        return fail(diag, "{s}: type must be noul, choice, or score", .{id});
    const instructions = switch (object.get("instructions") orelse .null) {
        .null => id,
        .string => |s| if (s.len == 0) id else s,
        else => |v| try render(arena, v),
    };
    const criteria = object.get("criteria") orelse .null;
    // Keys in the answer's order (criteria order; noul false, true);
    // `order` lists them as the model reads them.
    var keys: std.ArrayList([]const u8) = .empty;
    var descriptions: std.ArrayList(std.json.Value) = .empty;
    var order: []u32 = &.{};
    var legend: []const std.json.Value = &.{};
    switch (kind) {
        .noul => {
            var given: [2]std.json.Value = .{ .{ .string = "The proposition is false or the answer is no." }, .{ .string = "The proposition is true or the answer is yes." } };
            switch (criteria) {
                .null => {},
                // `dict.update`: other keys are accepted and unused.
                .object => |o| for ([_][]const u8{ "false", "true" }, &given) |key, *slot| {
                    if (o.get(key)) |d| slot.* = d;
                },
                else => return fail(diag, "{s}: a noul question takes \"criteria\" as an object with optional \"true\"/\"false\" descriptions, or omits it", .{id}),
            }
            try keys.appendSlice(arena, &.{ "false", "true" });
            try descriptions.appendSlice(arena, &given);
            order = try arena.dupe(u32, &.{ 1, 0 });
        },
        .choice => switch (criteria) {
            .object => |o| {
                if (o.count() == 0) return fail(diag, "{s}: criteria must not be empty", .{id});
                try keys.appendSlice(arena, o.keys());
                try descriptions.appendSlice(arena, o.values());
                order = try arena.alloc(u32, o.count());
                for (order, 0..) |*slot, i| slot.* = @intCast(i);
                std.mem.sort(u32, order, o.keys(), struct {
                    fn less(k: []const []const u8, a: u32, b: u32) bool {
                        return std.mem.lessThan(u8, k[a], k[b]);
                    }
                }.less);
            },
            else => return fail(diag, "{s}: a choice question takes \"criteria\" as a non-empty object of option -> description", .{id}),
        },
        .score => switch (criteria) {
            .array => |a| {
                if (a.items.len == 0) return fail(diag, "{s}: criteria must not be empty", .{id});
                for (a.items, 0..) |level, i| {
                    try keys.append(arena, try std.fmt.allocPrint(arena, "{d}", .{i}));
                    try descriptions.append(arena, level);
                }
                legend = a.items;
            },
            else => return fail(diag, "{s}: a score question takes \"criteria\" as a non-empty list of level descriptions, index 0 first", .{id}),
        },
    }
    const texts = try arena.alloc([]const u8, keys.items.len);
    for (texts, keys.items, descriptions.items) |*text, key, description| {
        var semantics: std.json.ObjectMap = .empty;
        try semantics.put(arena, "option_id", .{ .string = key });
        if (description != .null) try semantics.put(arena, "description", description);
        text.* = try render(arena, .{ .object = semantics });
    }
    return .{ .id = id, .kind = kind, .instructions = instructions, .keys = keys.items, .texts = texts, .model_order = order, .legend = legend };
}

fn fail(diag: *Diagnostic, comptime fmt: []const u8, args: anytype) error{InvalidQuestion} {
    diag.set(fmt, args);
    return error.InvalidQuestion;
}

pub const Span = struct { start: u32, end: u32 };

/// One question's place in a sequence: the instruction tokens and each
/// option's semantics tokens, the spans the head averages.
pub const Field = struct {
    kind: Kind,
    span: Span,
    options: []const Span,
};

/// The prompt around the state, tokenized once per model.
pub const Frame = struct {
    prefix: []const u32,
    suffix: []const u32,

    pub fn init(arena: std.mem.Allocator, tokenizer: *const Encoder) !Frame {
        return .{
            .prefix = try tokens(arena, tokenizer, "<|im_start|>system\n" ++ system_prompt ++ "<|im_end|>\n<|im_start|>user\nSTATE:\n"),
            .suffix = try tokens(arena, tokenizer, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"),
        };
    }
};

/// Every question of a call as tokens, spans relative to the schema's start;
/// shared by every state the call asks about.
pub const Schema = struct {
    ids: []const u32,
    fields: []const Field,
};

pub fn schema(arena: std.mem.Allocator, tokenizer: *const Encoder, ids: []const []const u8, questions: []const Question) !Schema {
    var out: std.ArrayList(u32) = .empty;
    const fields = try arena.alloc(Field, questions.len);
    try append(arena, tokenizer, &out, "\n\nSCHEMA FIELDS:\n");
    for (questions, ids, fields, 0..) |q, id, *field, qi| {
        try append(arena, tokenizer, &out, try std.fmt.allocPrint(arena, "\nFIELD {d}\nID: {s}\nTYPE: {s}\nINSTRUCTION: ", .{ qi + 1, id, q.kind.name() }));
        const start = out.items.len;
        try append(arena, tokenizer, &out, q.instructions);
        field.span = .{ .start = @intCast(start), .end = @intCast(out.items.len) };
        field.kind = q.kind;
        try append(arena, tokenizer, &out, "\nALLOWED OPTIONS:\n");
        const options = try arena.alloc(Span, q.texts.len);
        for (options, 0..) |*span, oi| {
            const text = q.texts[if (q.model_order.len > 0) q.model_order[oi] else oi];
            try append(arena, tokenizer, &out, try std.fmt.allocPrint(arena, "OPTION {d}: ", .{oi + 1}));
            const option_start = out.items.len;
            try append(arena, tokenizer, &out, text);
            span.* = .{ .start = @intCast(option_start), .end = @intCast(out.items.len) };
            try append(arena, tokenizer, &out, "\n");
        }
        field.options = options;
        try append(arena, tokenizer, &out, "END FIELD\n");
    }
    return .{ .ids = out.items, .fields = fields };
}

fn append(arena: std.mem.Allocator, tokenizer: *const Encoder, out: *std.ArrayList(u32), text: []const u8) !void {
    try out.appendSlice(arena, try tokens(arena, tokenizer, text));
}

/// `tokenizer(text, add_special_tokens=False)`: added tokens spelled in the
/// text are matched, nothing is inserted.
pub fn tokens(arena: std.mem.Allocator, tokenizer: *const Encoder, text: []const u8) ![]u32 {
    return tokenizer.encode(arena, text, true, .{});
}

/// One state's model input with every question of its call.
pub const Sequence = struct {
    ids: []const u32,
    /// Spans absolute in `ids`.
    fields: []const Field,
    /// Where the state's tokens start (after the prefix and any media tokens).
    state_offset: usize,
    state_tokens: usize,
    truncated: bool,
};

pub const AssembleError = std.mem.Allocator.Error || error{SchemaExceedsBudget};

/// `prefix ‖ media ‖ state ‖ schema ‖ suffix`, the state cut at its end to
/// `max_state_tokens` and then to what `max_length` leaves (the reference
/// keeps a state's head whatever its shape). `media` are the image tokens
/// that follow the prefix, empty for text.
pub fn assemble(arena: std.mem.Allocator, frame: Frame, call: Schema, media: []const u32, state: []const u32, max_state_tokens: ?usize) AssembleError!Sequence {
    const fixed = frame.prefix.len + media.len + call.ids.len + frame.suffix.len;
    if (fixed > max_length) return error.SchemaExceedsBudget;
    var kept = state.len;
    if (max_state_tokens) |limit| kept = @min(kept, limit);
    kept = @min(kept, max_length - fixed);
    const ids = try arena.alloc(u32, fixed + kept);
    var at: usize = 0;
    for ([_][]const u32{ frame.prefix, media, state[0..kept], call.ids, frame.suffix }) |part| {
        @memcpy(ids[at..][0..part.len], part);
        at += part.len;
    }
    const offset: u32 = @intCast(frame.prefix.len + media.len + kept);
    const fields = try arena.alloc(Field, call.fields.len);
    for (fields, call.fields) |*f, c| {
        const options = try arena.alloc(Span, c.options.len);
        for (options, c.options) |*o, s| o.* = .{ .start = s.start + offset, .end = s.end + offset };
        f.* = .{ .kind = c.kind, .span = .{ .start = c.span.start + offset, .end = c.span.end + offset }, .options = options };
    }
    return .{ .ids = ids, .fields = fields, .state_offset = frame.prefix.len + media.len, .state_tokens = state.len, .truncated = kept < state.len };
}

test "questions render as the reference renders them" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: Diagnostic = .{};
    const parse = struct {
        fn f(a: std.mem.Allocator, text: []const u8, d: *Diagnostic) !Question {
            return parseQuestion(a, "q", try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}), d);
        }
    }.f;
    const choice = try parse(arena, "{\"type\":\"choice\",\"instructions\":\"Which?\",\"criteria\":{\"paid\":\"Invoice is paid.\",\"draft\":null}}", &diag);
    try std.testing.expectEqualStrings("paid", choice.keys[0]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, choice.model_order);
    try std.testing.expectEqualStrings("{\"option_id\":\"draft\"}", choice.texts[1]);
    try std.testing.expectEqualStrings("{\"description\":\"Invoice is paid.\",\"option_id\":\"paid\"}", choice.texts[0]);
    const noul = try parse(arena, "{\"type\":\"noul\",\"instructions\":\"\",\"criteria\":{\"false\":\"no\"}}", &diag);
    try std.testing.expectEqualStrings("q", noul.instructions);
    try std.testing.expectEqualStrings("false", noul.keys[0]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, noul.model_order);
    try std.testing.expectEqualStrings("{\"description\":\"no\",\"option_id\":\"false\"}", noul.texts[0]);
    const score = try parse(arena, "{\"type\":\"score\",\"instructions\":{\"b\":1,\"a\":[2.0]},\"criteria\":[\"low\",\"high\"]}", &diag);
    try std.testing.expectEqualStrings("{\"a\":[2.0],\"b\":1}", score.instructions);
    try std.testing.expectEqualStrings("1", score.keys[1]);
    try std.testing.expectEqual(@as(usize, 2), score.legend.len);
    for ([_][]const u8{
        "[]",
        "{\"type\":\"rank\"}",
        "{\"type\":\"choice\",\"criteria\":{}}",
        "{\"type\":\"choice\",\"criteria\":[\"a\"]}",
        "{\"type\":\"score\",\"criteria\":[]}",
        "{\"type\":\"noul\",\"criteria\":[\"yes\"]}",
    }) |bad| try std.testing.expectError(error.InvalidQuestion, parse(arena, bad, &diag));
}

test "assemble: spans shift past the state, the state is cut to the budget" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const frame: Frame = .{ .prefix = &.{ 1, 2 }, .suffix = &.{9} };
    const call: Schema = .{ .ids = &.{ 5, 6, 7 }, .fields = &.{.{ .kind = .noul, .span = .{ .start = 0, .end = 1 }, .options = &.{.{ .start = 1, .end = 3 }} }} };
    const s = try assemble(arena, frame, call, &.{}, &.{ 3, 4, 4, 4 }, 2);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 7, 9 }, s.ids);
    try std.testing.expect(s.truncated);
    try std.testing.expectEqual(@as(u32, 4), s.fields[0].span.start);
    try std.testing.expectEqual(@as(u32, 7), s.fields[0].options[0].end);
}
