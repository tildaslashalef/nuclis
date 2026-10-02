//! What every decision model shares: the question as the request defines
//! it (`Question`, its `Kind`, validation `Diagnostic`s), and Python's
//! `json.dumps` text, which both families render structured values in since
//! their models read that text in training. Each family parses a question
//! definition into a `Question` (profiles/laya.zig, profiles/clef.zig).
const std = @import("std");

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

/// One question, validated and rendered by its family's profile.
pub const Question = struct {
    /// The request's name for it (clef's schema reads it).
    id: []const u8 = "",
    kind: Kind,
    /// As the model reads it: non-string instructions are JSON.
    instructions: []const u8,
    /// Answer keys in the family's option order (Laya: choice keys, score
    /// `"0"`…, noul `"false"`, `"true"`).
    keys: []const []const u8,
    /// Option texts in the same order, as the family renders them.
    texts: []const []const u8,
    /// The order the model reads the options in, as indices into `keys`;
    /// empty when it is `keys`' own order. Logits come back in this order.
    model_order: []const u32 = &.{},
    /// A score question's criteria as given (the answer's `legend`); empty otherwise.
    legend: []const std.json.Value = &.{},
};

/// `json.dumps`'s arguments that change the text; `ensure_ascii=False` always.
pub const Style = struct {
    /// `separators=(",", ":")` instead of `", "` and `": "`.
    compact: bool = false,
    /// `sort_keys=True`: object keys in code-point (UTF-8 byte) order
    /// (`pythonJsonStyled` only; the writer keeps the order it is given).
    sort_keys: bool = false,
};

/// `json.dumps(value, ensure_ascii=False)`: separators `", "` and `": "`,
/// non-ASCII kept, floats in Python's `repr` form, keys in their order.
pub fn pythonJson(arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    return pythonJsonStyled(arena, value, .{});
}

pub fn pythonJsonStyled(arena: std.mem.Allocator, value: std.json.Value, style: Style) ![]const u8 {
    const ordered = if (style.sort_keys) try sortedCopy(arena, value) else value;
    var out: std.Io.Writer.Allocating = .init(arena);
    writePythonJsonStyled(&out.writer, ordered, style) catch return error.OutOfMemory;
    return out.written();
}

/// `value` with every object's keys in code-point (UTF-8 byte) order, in
/// `arena`; the input is not modified.
fn sortedCopy(arena: std.mem.Allocator, value: std.json.Value) std.mem.Allocator.Error!std.json.Value {
    switch (value) {
        .array => |a| {
            var items = try std.json.Array.initCapacity(arena, a.items.len);
            for (a.items) |item| items.appendAssumeCapacity(try sortedCopy(arena, item));
            return .{ .array = items };
        },
        .object => |o| {
            const order = try arena.alloc(usize, o.count());
            for (order, 0..) |*slot, i| slot.* = i;
            const keys = o.keys();
            std.mem.sort(usize, order, keys, struct {
                fn less(k: []const []const u8, a: usize, b: usize) bool {
                    return std.mem.lessThan(u8, k[a], k[b]);
                }
            }.less);
            var map: std.json.ObjectMap = .empty;
            try map.ensureTotalCapacity(arena, order.len);
            for (order) |i| map.putAssumeCapacity(keys[i], try sortedCopy(arena, o.values()[i]));
            return .{ .object = map };
        },
        else => return value,
    }
}

pub fn writePythonJson(w: *std.Io.Writer, value: std.json.Value) std.Io.Writer.Error!void {
    return writePythonJsonStyled(w, value, .{});
}

pub fn writePythonJsonStyled(w: *std.Io.Writer, value: std.json.Value, style: Style) std.Io.Writer.Error!void {
    const item_separator = if (style.compact) "," else ", ";
    const key_separator = if (style.compact) ":" else ": ";
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
                if (i > 0) try w.writeAll(item_separator);
                try writePythonJsonStyled(w, item, style);
            }
            try w.writeByte(']');
        },
        .object => |o| {
            try w.writeByte('{');
            for (o.keys(), o.values(), 0..) |key, item, n| {
                if (n > 0) try w.writeAll(item_separator);
                try writePythonString(w, key);
                try w.writeAll(key_separator);
                try writePythonJsonStyled(w, item, style);
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

test "json.dumps styles: compact separators and sorted keys" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const value = try std.json.parseFromSliceLeaky(std.json.Value, arena, "{\"b\":[1,2.5,\"ü\"],\"a\":{\"z\":null,\"y\":true}}", .{});
    try std.testing.expectEqualStrings("{\"b\": [1, 2.5, \"ü\"], \"a\": {\"z\": null, \"y\": true}}", try pythonJson(arena, value));
    try std.testing.expectEqualStrings("{\"a\":{\"y\":true,\"z\":null},\"b\":[1,2.5,\"ü\"]}", try pythonJsonStyled(arena, value, .{ .compact = true, .sort_keys = true }));
}
