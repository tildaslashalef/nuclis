//! Laya's input contract and calibration, as the `laya` 0.3.20 package
//! defines them: question validation and option rendering, the sequence
//! `[CLS] <type> question: <instructions> [SEP] ([MASK] option)… [SEP] state
//! [SEP]` within its budgets, `rl_agent_config.json`, and the calibrated
//! answer. Pure: tokenizing goes through a borrowed Hugging Face tokenizer,
//! nothing here reads files or runs the model. Structured values are
//! rendered as Python's `json.dumps(…, ensure_ascii=False)` renders them,
//! since the model read that text in training. Contract: docs/reference/laya.md.
const std = @import("std");
const hf = @import("../tokenizer/hf_json.zig");

pub const Kind = enum(u2) {
    choice = 0,
    score = 1,
    noul = 2,

    pub fn name(self: Kind) []const u8 {
        return @tagName(self);
    }
};

/// A validation failure's reason, naming the question.
pub const Diagnostic = struct {
    buffer: [512]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buffer[0..self.len];
    }

    pub fn set(self: *Diagnostic, comptime fmt: []const u8, args: anytype) void {
        self.len = if (std.fmt.bufPrint(&self.buffer, fmt, args)) |text| text.len else |_| self.buffer.len;
    }
};

pub const QuestionError = std.mem.Allocator.Error || error{InvalidQuestion};

/// One question, validated and rendered.
pub const Question = struct {
    kind: Kind,
    /// As the model reads it: non-string instructions are JSON.
    instructions: []const u8,
    /// Answer keys in option order: choice keys, score `"0"`…, noul `"false"`, `"true"`.
    keys: []const []const u8,
    /// Option texts in the same order (`"key: description"`, `"level i: …"`, `"false: …"`).
    texts: []const []const u8,
    /// A score question's criteria as given (the answer's `legend`); empty otherwise.
    legend: []const std.json.Value = &.{},
};

/// Validates a question definition (`{"type", "instructions", "criteria",
/// "labels"}`) as the package does and renders its options; `arena` owns
/// the result. Failures name `id` in `diag`.
pub fn parseQuestion(arena: std.mem.Allocator, id: []const u8, value: std.json.Value, diag: *Diagnostic) QuestionError!Question {
    const object = switch (value) {
        .object => |o| o,
        else => return fail(diag, "question {s}: definition must be an object", .{id}),
    };
    const kind_text = switch (object.get("type") orelse .null) {
        .string => |s| s,
        else => return fail(diag, "question {s}: \"type\" must be one of choice, noul, score", .{id}),
    };
    const kind = std.meta.stringToEnum(Kind, kind_text) orelse
        return fail(diag, "question {s}: unknown type \"{s}\"; use one of choice, noul, score", .{ id, kind_text });
    const instructions_value = object.get("instructions") orelse
        return fail(diag, "question {s}: no \"instructions\"; add the text the model should answer", .{id});
    const instructions = switch (instructions_value) {
        .string => |s| s,
        else => try pythonJson(arena, instructions_value),
    };
    const criteria = object.get("criteria") orelse .null;
    if (object.get("labels") != null and kind != .noul)
        return fail(diag, "question {s}: \"labels\" is only supported for noul questions", .{id});
    var keys: std.ArrayList([]const u8) = .empty;
    var texts: std.ArrayList([]const u8) = .empty;
    var legend: []const std.json.Value = &.{};
    switch (kind) {
        .choice => switch (criteria) {
            .object => |o| {
                if (o.count() == 0) return fail(diag, "question {s}: a choice question needs at least one criterion", .{id});
                var it = o.iterator();
                while (it.next()) |entry| {
                    try keys.append(arena, entry.key_ptr.*);
                    const v = entry.value_ptr.*;
                    const bare = v == .null or (v == .string and v.string.len == 0);
                    try texts.append(arena, if (bare) entry.key_ptr.* else try std.fmt.allocPrint(arena, "{s}: {s}", .{ entry.key_ptr.*, try criterion(arena, v) }));
                }
            },
            .array => |a| {
                if (a.items.len == 0) return fail(diag, "question {s}: a choice question needs at least one criterion", .{id});
                for (a.items) |item| {
                    const label = switch (item) {
                        .string => |s| s,
                        else => return fail(diag, "question {s}: a choice question's criteria list holds labels (strings)", .{id}),
                    };
                    try keys.append(arena, label);
                    try texts.append(arena, label);
                }
            },
            else => return fail(diag, "question {s}: a choice question takes \"criteria\" as an object of label -> description, or a list of labels", .{id}),
        },
        .score => switch (criteria) {
            .array => |a| {
                if (a.items.len == 0) return fail(diag, "question {s}: a score question needs at least one level", .{id});
                for (a.items, 0..) |level, i| {
                    try keys.append(arena, try std.fmt.allocPrint(arena, "{d}", .{i}));
                    try texts.append(arena, try std.fmt.allocPrint(arena, "level {d}: {s}", .{ i, try criterion(arena, level) }));
                }
                legend = a.items;
            },
            else => return fail(diag, "question {s}: a score question takes \"criteria\" as a list of level descriptions, index 0 first", .{id}),
        },
        .noul => {
            var descriptions: [2]std.json.Value = .{ .null, .null };
            switch (criteria) {
                .null => {},
                .object => |o| {
                    var it = o.iterator();
                    while (it.next()) |entry| {
                        const slot = noulSlot(entry.key_ptr.*) orelse return fail(diag, "question {s}: a noul question takes \"criteria\" keyed only \"true\"/\"false\" (either or both), got \"{s}\"; set \"labels\" to word the answer differently", .{ id, entry.key_ptr.* });
                        descriptions[slot] = entry.value_ptr.*;
                    }
                },
                else => return fail(diag, "question {s}: a noul question takes \"criteria\" as an object with optional \"true\"/\"false\" descriptions, or omits it", .{id}),
            }
            var labels: [2][]const u8 = .{ "false", "true" };
            if (object.get("labels")) |l| labels = noulLabels(l) orelse
                return fail(diag, "question {s}: noul labels must map exactly \"false\" and \"true\" to distinct non-empty strings", .{id});
            const defaults = [2][]const u8{ "no, the statement does not hold", "yes, the statement holds" };
            for (0..2) |slot| {
                const d = descriptions[slot];
                const bare = d == .null or (d == .string and d.string.len == 0);
                try keys.append(arena, if (slot == 0) "false" else "true");
                try texts.append(arena, try std.fmt.allocPrint(arena, "{s}: {s}", .{ labels[slot], if (bare) defaults[slot] else try criterion(arena, d) }));
            }
        },
    }
    return .{ .kind = kind, .instructions = instructions, .keys = keys.items, .texts = texts.items, .legend = legend };
}

fn fail(diag: *Diagnostic, comptime fmt: []const u8, args: anytype) error{InvalidQuestion} {
    diag.set(fmt, args);
    return error.InvalidQuestion;
}

/// `str(k).lower()` is `"true"` or `"false"`.
fn noulSlot(key: []const u8) ?usize {
    if (std.ascii.eqlIgnoreCase(key, "false")) return 0;
    if (std.ascii.eqlIgnoreCase(key, "true")) return 1;
    return null;
}

fn noulLabels(value: std.json.Value) ?[2][]const u8 {
    const o = switch (value) {
        .object => |o| o,
        else => return null,
    };
    if (o.count() != 2) return null;
    var labels: [2][]const u8 = undefined;
    for ([_][]const u8{ "false", "true" }, &labels) |key, *label| {
        const text = switch (o.get(key) orelse return null) {
            .string => |s| std.mem.trim(u8, s, &std.ascii.whitespace),
            else => return null,
        };
        if (text.len == 0) return null;
        label.* = text;
    }
    if (std.mem.eql(u8, labels[0], labels[1])) return null;
    return labels;
}

/// A criterion value as text: strings pass through, anything else is JSON.
fn criterion(arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |s| s,
        else => pythonJson(arena, value),
    };
}

/// `json.dumps(value, ensure_ascii=False)`: separators `", "` and `": "`,
/// non-ASCII kept, floats in Python's `repr` form, keys in their order.
pub fn pythonJson(arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    writePythonJson(&out.writer, value) catch return error.OutOfMemory;
    return out.written();
}

pub fn writePythonJson(w: *std.Io.Writer, value: std.json.Value) std.Io.Writer.Error!void {
    switch (value) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .float => |f| try writePythonFloat(w, f),
        .number_string => |s| try w.writeAll(s),
        .string => |s| try writePythonString(w, s),
        .array => |a| {
            try w.writeByte('[');
            for (a.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try writePythonJson(w, item);
            }
            try w.writeByte(']');
        },
        .object => |o| {
            try w.writeByte('{');
            var it = o.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try w.writeAll(", ");
                first = false;
                try writePythonString(w, entry.key_ptr.*);
                try w.writeAll(": ");
                try writePythonJson(w, entry.value_ptr.*);
            }
            try w.writeByte('}');
        },
    }
}

fn writePythonString(w: *std.Io.Writer, s: []const u8) std.Io.Writer.Error!void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0x08 => try w.writeAll("\\b"),
        0x0c => try w.writeAll("\\f"),
        0...0x07, 0x0b, 0x0e...0x1f => try w.print("\\u{x:0>4}", .{c}),
        else => try w.writeByte(c),
    };
    try w.writeByte('"');
}

/// Python's `repr(float)`: the shortest round-trip digits, positional for
/// decimal exponents −4…15 (always with a fraction), else `d.ddde±XX`.
pub fn writePythonFloat(w: *std.Io.Writer, x: f64) std.Io.Writer.Error!void {
    if (!std.math.isFinite(x)) return w.writeAll(if (std.math.isNan(x)) "NaN" else if (x > 0) "Infinity" else "-Infinity");
    var buffer: [64]u8 = undefined;
    const scientific = std.fmt.bufPrint(&buffer, "{e}", .{@abs(x)}) catch unreachable;
    const e_at = std.mem.indexOfScalar(u8, scientific, 'e').?;
    const exponent = std.fmt.parseInt(i32, scientific[e_at + 1 ..], 10) catch unreachable;
    var digits_buffer: [32]u8 = undefined;
    var n: usize = 0;
    for (scientific[0..e_at]) |c| if (c != '.') {
        digits_buffer[n] = c;
        n += 1;
    };
    const digits = digits_buffer[0..n];
    if (std.math.signbit(x)) try w.writeByte('-');
    if (exponent < -4 or exponent >= 16) {
        try w.writeByte(digits[0]);
        if (digits.len > 1) try w.print(".{s}", .{digits[1..]});
        return w.print("e{c}{d:0>2}", .{ @as(u8, if (exponent < 0) '-' else '+'), @abs(exponent) });
    }
    if (exponent < 0) {
        try w.writeAll("0.");
        for (0..@intCast(-exponent - 1)) |_| try w.writeByte('0');
        return w.writeAll(digits);
    }
    const whole: usize = @intCast(exponent + 1);
    if (whole >= digits.len) {
        try w.writeAll(digits);
        for (0..whole - digits.len) |_| try w.writeByte('0');
        return w.writeAll(".0");
    }
    try w.print("{s}.{s}", .{ digits[0..whole], digits[whole..] });
}

/// `text` with every `[MASK]` spelled in it replaced by a space, as the
/// package does before tokenizing instructions, options, and states.
pub fn unmask(arena: std.mem.Allocator, text: []const u8) ![]const u8 {
    if (std.mem.indexOf(u8, text, mask_text) == null) return text;
    return std.mem.replaceOwned(u8, arena, text, mask_text, " ");
}

const mask_text = "[MASK]";

pub const Budget = struct {
    max_len: usize = 512,
    head_max_len: usize = 192,
};

/// A question tokenized once, for every state it is asked of.
pub const Prepared = struct {
    kind: Kind,
    head: []const u32,
    /// Each option's `[MASK]` and its first 48 tokens.
    options: []const []const u32,
};

pub const Specials = struct {
    cls: u32,
    sep: u32,
    mask: u32,

    pub fn find(tokenizer: *const hf.Tokenizer) error{MissingSpecialToken}!Specials {
        const v = &tokenizer.vocab;
        return .{
            .cls = v.tokenId("[CLS]") orelse return error.MissingSpecialToken,
            .sep = v.tokenId("[SEP]") orelse return error.MissingSpecialToken,
            .mask = v.tokenId(mask_text) orelse return error.MissingSpecialToken,
        };
    }
};

pub fn prepare(arena: std.mem.Allocator, tokenizer: *const hf.Tokenizer, specials: Specials, question: Question) !Prepared {
    const head_text = try std.fmt.allocPrint(arena, "{s} question: {s}", .{ question.kind.name(), try unmask(arena, question.instructions) });
    const head = try tokenizer.encode(arena, head_text, .{});
    const options = try arena.alloc([]const u32, question.texts.len);
    for (options, question.texts) |*o, text| {
        const spaced = try std.fmt.allocPrint(arena, " {s}", .{try unmask(arena, text)});
        const ids = try tokenizer.encode(arena, spaced, .{});
        const kept = ids[0..@min(ids.len, 48)];
        const option = try arena.alloc(u32, 1 + kept.len);
        option[0] = specials.mask;
        @memcpy(option[1..], kept);
        o.* = option;
    }
    return .{ .kind = question.kind, .head = head, .options = options };
}

pub const Truncate = enum {
    /// Keep the state's first tokens (text and object states).
    tail,
    /// Keep its last tokens (a list state: a conversation, newest last).
    head,
};

pub const Sequence = struct {
    ids: []const u32,
    markers: []const usize,
    /// How many of the state's tokens the sequence kept.
    state_kept: usize,
    truncated: bool,
    /// The question text's tokens kept (the budget may cut them).
    head_kept: usize,
    /// Tokens each option kept, its `[MASK]` included.
    option_kept: []const usize,
};

pub const AssembleError = std.mem.Allocator.Error || error{OptionsExceedBudget};

/// The sequence for one question and one tokenized state, budgets applied
/// in the package's order: options capped (to `max(4, (head_max_len − 16) /
/// count)` each when they leave under 16 tokens), the question text to
/// what remains (at least 8), the state to the rest of `max_len`.
pub fn assemble(arena: std.mem.Allocator, specials: Specials, prepared: Prepared, state: []const u32, truncate: Truncate, budget: Budget) AssembleError!Sequence {
    const count = prepared.options.len;
    const option_kept = try arena.alloc(usize, count);
    var used: usize = 0;
    for (prepared.options, option_kept) |o, *kept| {
        kept.* = o.len;
        used += o.len;
    }
    var option_budget: isize = @as(isize, @intCast(budget.head_max_len)) - @as(isize, @intCast(used));
    if (option_budget < 16) {
        const per = @max(4, (budget.head_max_len -| 16) / @max(1, count));
        used = 0;
        for (option_kept) |*kept| {
            kept.* = @min(kept.*, per);
            used += kept.*;
        }
        option_budget = @as(isize, @intCast(budget.head_max_len)) - @as(isize, @intCast(used));
    }
    const head_kept = @min(prepared.head.len, @as(usize, @intCast(@max(8, option_budget))));
    var ids: std.ArrayList(u32) = .empty;
    try ids.ensureTotalCapacity(arena, 3 + head_kept + used + state.len);
    ids.appendAssumeCapacity(specials.cls);
    ids.appendSliceAssumeCapacity(prepared.head[0..head_kept]);
    ids.appendAssumeCapacity(specials.sep);
    var markers: std.ArrayList(usize) = .empty;
    for (prepared.options, option_kept) |o, kept| {
        try markers.append(arena, ids.items.len);
        ids.appendSliceAssumeCapacity(o[0..kept]);
    }
    ids.appendAssumeCapacity(specials.sep);
    const room = budget.max_len -| (ids.items.len + 1);
    const state_kept = @min(state.len, room);
    const kept = switch (truncate) {
        .tail => state[0..state_kept],
        .head => state[state.len - state_kept ..],
    };
    ids.appendSliceAssumeCapacity(kept);
    ids.appendAssumeCapacity(specials.sep);
    const final = ids.items[0..@min(ids.items.len, budget.max_len)];
    for (markers.items) |m| if (m >= budget.max_len) return error.OptionsExceedBudget;
    return .{ .ids = final, .markers = markers.items, .state_kept = state_kept, .truncated = state_kept < state.len, .head_kept = head_kept, .option_kept = option_kept };
}

/// `rl_agent_config.json`, the parts inference reads.
pub const AgentConfig = struct {
    budget: Budget = .{},
    /// Per question type, as shipped and as applied (clamped).
    temperature_raw: [3]f64 = .{ 1, 1, 1 },
    temperature: [3]f64 = .{ 1, 1, 1 },
    /// Per bucket (`"choice:3-5"`), as applied; `by_options_raw` as shipped.
    by_options: []const Bucketed = &.{},
    by_options_raw: []const Bucketed = &.{},

    pub const Bucketed = struct { bucket: []const u8, temperature: f64 };

    /// The temperature applied to a question of `kind` with `k` options.
    pub fn temperatureFor(self: *const AgentConfig, kind: Kind, k: usize) f64 {
        var buffer: [16]u8 = undefined;
        const b = bucket(&buffer, kind, k);
        for (self.by_options) |entry| if (std.mem.eql(u8, entry.bucket, b)) return entry.temperature;
        return self.temperature[@intFromEnum(kind)];
    }
};

pub const temperature_min = 0.5;
pub const temperature_max = 5.0;

/// The package refuses a temperature that sharpens hard or is not a number:
/// clamped to [0.5, 5.0], and 1.0 when not finite.
pub fn clampTemperature(t: f64) f64 {
    if (!std.math.isFinite(t)) return 1.0;
    return std.math.clamp(t, temperature_min, temperature_max);
}

/// `"<type>:<2|3-5|6-10|11+>"`.
pub fn bucket(buffer: []u8, kind: Kind, k: usize) []const u8 {
    const size = if (k <= 2) "2" else if (k <= 5) "3-5" else if (k <= 10) "6-10" else "11+";
    return std.fmt.bufPrint(buffer, "{s}:{s}", .{ kind.name(), size }) catch unreachable;
}

pub const ConfigError = std.mem.Allocator.Error || error{InvalidAgentConfig};

/// Parses `rl_agent_config.json`; `arena` owns the result. Missing keys take
/// the package's defaults; a temperature that is not a number applies as 1.0.
pub fn parseAgentConfig(arena: std.mem.Allocator, bytes: []const u8) ConfigError!AgentConfig {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{}) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidAgentConfig,
    };
    const o = switch (root) {
        .object => |o| o,
        else => return error.InvalidAgentConfig,
    };
    var config: AgentConfig = .{};
    config.budget.max_len = try positive(o.get("max_len"), config.budget.max_len);
    config.budget.head_max_len = try positive(o.get("head_max_len"), config.budget.head_max_len);
    // 8192 is the encoder's position limit; the head must leave room for the state.
    if (config.budget.max_len > 8192 or config.budget.head_max_len + 3 > config.budget.max_len) return error.InvalidAgentConfig;
    if (o.get("temperature")) |t| {
        const a = switch (t) {
            .array => |a| a,
            else => return error.InvalidAgentConfig,
        };
        if (a.items.len != 3) return error.InvalidAgentConfig;
        for (a.items, 0..) |item, i| {
            config.temperature_raw[i] = number(item);
            config.temperature[i] = clampTemperature(config.temperature_raw[i]);
        }
    }
    if (o.get("temperature_by_options")) |t| {
        const map = switch (t) {
            .object => |m| m,
            else => return error.InvalidAgentConfig,
        };
        const raw = try arena.alloc(AgentConfig.Bucketed, map.count());
        const applied = try arena.alloc(AgentConfig.Bucketed, map.count());
        var it = map.iterator();
        var i: usize = 0;
        while (it.next()) |entry| : (i += 1) {
            raw[i] = .{ .bucket = entry.key_ptr.*, .temperature = number(entry.value_ptr.*) };
            applied[i] = .{ .bucket = entry.key_ptr.*, .temperature = clampTemperature(raw[i].temperature) };
        }
        config.by_options_raw = raw;
        config.by_options = applied;
    }
    return config;
}

fn number(value: std.json.Value) f64 {
    return switch (value) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

fn positive(value: ?std.json.Value, default: usize) error{InvalidAgentConfig}!usize {
    const v = value orelse return default;
    return switch (v) {
        .integer => |i| if (i > 0) @intCast(i) else error.InvalidAgentConfig,
        else => error.InvalidAgentConfig,
    };
}

/// An answer's numbers. Probabilities are F32 as the package computes them
/// (NumPy on the F32 logits); `value` is the choice index, the expected
/// score, or P(true).
pub const Calibrated = struct {
    probabilities: []const f32,
    temperature: f64,
    best: usize,
    value: f64,
    /// Normalized entropy, `1 − H(p) / ln k` (choice and score); for noul
    /// `max(p1, 1 − p1)`.
    confidence: f64,
    /// `max(p)`, the calibrated one on every type.
    answer_confidence: f64,
};

/// Calibrates one question's logits (`temperature` 1 when uncalibrated).
pub fn calibrate(arena: std.mem.Allocator, kind: Kind, logits: []const f32, temperature: f64) !Calibrated {
    const k = logits.len;
    const p = try arena.alloc(f32, k);
    if (k == 0) return .{ .probabilities = p, .temperature = temperature, .best = 0, .value = 0, .confidence = 1, .answer_confidence = 1 };
    const t: f32 = @floatCast(temperature);
    var max: f32 = -std.math.inf(f32);
    for (logits) |z| max = @max(max, z / t);
    var total: f32 = 0;
    for (p, logits) |*x, z| {
        x.* = @exp(z / t - max);
        total += x.*;
    }
    var best: usize = 0;
    for (p, 0..) |*x, i| {
        x.* /= total;
        if (x.* > p[best]) best = i;
    }
    const top: f64 = p[best];
    const entropy_confidence: f64 = if (k < 2) 1 else blk: {
        var entropy: f32 = 0;
        for (p) |x| {
            const clipped = std.math.clamp(x, 1e-12, 1);
            entropy -= x * @log(clipped);
        }
        break :blk std.math.clamp(1 - @as(f64, entropy) / @log(@as(f64, @floatFromInt(k))), 0, 1);
    };
    return switch (kind) {
        .choice => .{ .probabilities = p, .temperature = temperature, .best = best, .value = @floatFromInt(best), .confidence = entropy_confidence, .answer_confidence = top },
        .score => blk: {
            var expected: f64 = 0;
            for (p, 0..) |x, i| expected += @as(f64, @floatFromInt(i)) * x;
            break :blk .{ .probabilities = p, .temperature = temperature, .best = best, .value = expected, .confidence = entropy_confidence, .answer_confidence = top };
        },
        .noul => blk: {
            const yes: f64 = if (k > 1) p[1] else 0;
            break :blk .{ .probabilities = p, .temperature = temperature, .best = best, .value = yes, .confidence = @max(yes, 1 - yes), .answer_confidence = top };
        },
    };
}

/// Python's `round(x, 4)`: the double nearest the exact value rounded half
/// to even at four places.
pub fn round4(x: f64) f64 {
    var buffer: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d:.4}", .{x}) catch return x;
    return std.fmt.parseFloat(f64, text) catch x;
}

test "Python JSON: separators, escapes, non-ASCII, and float repr" {
    const gpa = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena,
        \\{"customer": "Zoë \"Z\"\n\u0001", "seats": 12, "ratio": 1.0, "tiny": 1e-05, "big": 1e16,
        \\ "list": [true, false, null, 0.1, -2.5, 123456789012345678.0, 0.0001, 1.5e300], "empty": {}}
    , .{});
    try std.testing.expectEqualStrings(
        \\{"customer": "Zoë \"Z\"\n\u0001", "seats": 12, "ratio": 1.0, "tiny": 1e-05, "big": 1e+16, "list": [true, false, null, 0.1, -2.5, 1.2345678901234568e+17, 0.0001, 1.5e+300], "empty": {}}
    , try pythonJson(arena, parsed));
    var buffer: [32]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    try writePythonFloat(&w, -0.0);
    try w.writeAll(" ");
    try writePythonFloat(&w, 1234.5);
    try std.testing.expectEqualStrings("-0.0 1234.5", w.buffered());
}

fn questionFrom(arena: std.mem.Allocator, text: []const u8, diag: *Diagnostic) !Question {
    const value = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
    return parseQuestion(arena, "q", value, diag);
}

test "questions render as the package renders them" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: Diagnostic = .{};
    const choice = try questionFrom(arena,
        \\{"type": "choice", "instructions": "Which?", "criteria": {"billing": "invoices", "other": null, "zero": 0, "rubric": {"desc": "x"}}}
    , &diag);
    try std.testing.expectEqualStrings("billing: invoices", choice.texts[0]);
    try std.testing.expectEqualStrings("other", choice.texts[1]);
    try std.testing.expectEqualStrings("zero: 0", choice.texts[2]);
    try std.testing.expectEqualStrings("rubric: {\"desc\": \"x\"}", choice.texts[3]);
    const labels = try questionFrom(arena, "{\"type\": \"choice\", \"instructions\": [1, \"a\"], \"criteria\": [\"a\", \"b\"]}", &diag);
    try std.testing.expectEqualStrings("[1, \"a\"]", labels.instructions);
    try std.testing.expectEqualStrings("b", labels.texts[1]);
    const score = try questionFrom(arena, "{\"type\": \"score\", \"instructions\": \"How?\", \"criteria\": [\"low\", \"high\"]}", &diag);
    try std.testing.expectEqualStrings("level 1: high", score.texts[1]);
    try std.testing.expectEqualStrings("1", score.keys[1]);
    const noul = try questionFrom(arena,
        \\{"type": "noul", "instructions": "Open?", "criteria": {"True": "still open"}, "labels": {"false": " B ", "true": "A"}}
    , &diag);
    try std.testing.expectEqualStrings("B: no, the statement does not hold", noul.texts[0]);
    try std.testing.expectEqualStrings("A: still open", noul.texts[1]);
    const cases = .{
        .{ "{\"type\": \"rank\", \"instructions\": \"x\"}", "unknown type" },
        .{ "{\"type\": \"noul\"}", "no \"instructions\"" },
        .{ "{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": {}}", "at least one criterion" },
        .{ "{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": {\"a\": 1}}", "list of level descriptions" },
        .{ "{\"type\": \"noul\", \"instructions\": \"x\", \"criteria\": {\"yes\": \"a\"}}", "keyed only" },
        .{ "{\"type\": \"noul\", \"instructions\": \"x\", \"labels\": {\"false\": \"a\", \"true\": \"a\"}}", "distinct" },
        .{ "{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": [\"a\"], \"labels\": {}}", "only supported for noul" },
    };
    inline for (cases) |case| {
        try std.testing.expectError(error.InvalidQuestion, questionFrom(arena, case[0], &diag));
        try std.testing.expect(std.mem.indexOf(u8, diag.message(), case[1]) != null);
        try std.testing.expect(std.mem.startsWith(u8, diag.message(), "question q: "));
    }
}

test "assembly applies the budgets in the package's order" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const specials: Specials = .{ .cls = 100, .sep = 101, .mask = 102 };
    const head = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const option = [_]u32{ 102, 20, 21, 22, 23, 24 };
    const prepared: Prepared = .{ .kind = .choice, .head = &head, .options = &.{ &option, &option } };
    const state = [_]u32{ 30, 31, 32, 33, 34, 35 };
    // Everything fits.
    const whole = try assemble(arena, specials, prepared, &state, .tail, .{ .max_len = 64, .head_max_len = 40 });
    try std.testing.expectEqualSlices(usize, &.{ 12, 18 }, whole.markers);
    try std.testing.expectEqual(@as(usize, 6), whole.state_kept);
    try std.testing.expect(!whole.truncated);
    // Options leave under 16: each is cut to max(4, (20 − 16) / 2) = 4, the head to 20 − 8 = 12 (all 10).
    const shrunk = try assemble(arena, specials, prepared, &state, .tail, .{ .max_len = 64, .head_max_len = 20 });
    try std.testing.expectEqualSlices(usize, &.{ 4, 4 }, shrunk.option_kept);
    try std.testing.expectEqualSlices(u32, &.{ 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 101, 102, 20, 21, 22, 102, 20, 21, 22, 101, 30, 31, 32, 33, 34, 35, 101 }, shrunk.ids);
    // State cut at the tail or, for a conversation, at the head.
    const tail = try assemble(arena, specials, prepared, &state, .tail, .{ .max_len = 27, .head_max_len = 20 });
    try std.testing.expectEqualSlices(u32, &.{ 30, 31, 32, 33, 34 }, tail.ids[21..26]);
    try std.testing.expect(tail.truncated);
    const head_cut = try assemble(arena, specials, prepared, &state, .head, .{ .max_len = 27, .head_max_len = 20 });
    try std.testing.expectEqualSlices(u32, &.{ 31, 32, 33, 34, 35 }, head_cut.ids[21..26]);
    // Options that do not fit in max_len at all.
    try std.testing.expectError(error.OptionsExceedBudget, assemble(arena, specials, prepared, &state, .tail, .{ .max_len = 16, .head_max_len = 20 }));
}

test "agent config clamps temperatures; calibration and rounding" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try parseAgentConfig(arena,
        \\{"max_len": 512, "head_max_len": 192, "temperature": [1.6, 1.25, 1.98],
        \\ "temperature_by_options": {"choice:11+": 0.1006, "choice:2": 1.9, "noul:2": 9}}
    );
    try std.testing.expectEqual(@as(f64, 0.5), config.temperatureFor(.choice, 20));
    try std.testing.expectEqual(@as(f64, 1.9), config.temperatureFor(.choice, 2));
    try std.testing.expectEqual(@as(f64, 5.0), config.temperatureFor(.noul, 2));
    try std.testing.expectEqual(@as(f64, 1.6), config.temperatureFor(.choice, 4));
    try std.testing.expectEqual(@as(f64, 0.1006), config.by_options_raw[0].temperature);
    try std.testing.expectError(error.InvalidAgentConfig, parseAgentConfig(arena, "{\"temperature\": [1, 2]}"));
    try std.testing.expectError(error.InvalidAgentConfig, parseAgentConfig(arena, "{\"max_len\": 100, \"head_max_len\": 192}"));
    const noul = try calibrate(arena, .noul, &.{ 0, @log(@as(f32, 3)) }, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), noul.value, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), noul.answer_confidence, 1e-6);
    const score = try calibrate(arena, .score, &.{ 1, 1, 1, 1 }, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), score.value, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0), score.confidence, 1e-6);
    try std.testing.expectEqual(@as(f64, 0.1235), round4(0.12345678));
    try std.testing.expectEqual(@as(f64, 0.9865), round4(0.98649999));
}
