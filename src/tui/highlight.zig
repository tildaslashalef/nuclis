//! Heuristic syntax highlighting for fenced code blocks. One line at a time
//! with a small carry-over state (block comments), classified into four
//! token kinds: keywords (per-language tables), strings, comments, numbers.
//! Tokens are painted over the code-block baseline: each span re-stamps the
//! baseline afterwards so the block background and dimming survive, and the
//! caller closes the line with `theme.reset`.
//!
//! This is a readability aid, not a parser: unclosed strings end at the line
//! end, escapes are honored, and unknown languages still get
//! strings/numbers/comments.
const std = @import("std");
const theme = @import("theme.zig");

pub const Language = enum { generic, zig, c, python, javascript, rust, go, shell };

/// Maps a fence info string (text after ```) to a language; unknown or empty
/// info strings fall back to the generic rules.
pub fn detect(info: []const u8) Language {
    if (info.len == 0) return .generic;
    const names = [_]struct { tag: []const u8, lang: Language }{
        .{ .tag = "zig", .lang = .zig },
        .{ .tag = "c", .lang = .c },
        .{ .tag = "h", .lang = .c },
        .{ .tag = "cpp", .lang = .c },
        .{ .tag = "c++", .lang = .c },
        .{ .tag = "py", .lang = .python },
        .{ .tag = "python", .lang = .python },
        .{ .tag = "js", .lang = .javascript },
        .{ .tag = "ts", .lang = .javascript },
        .{ .tag = "javascript", .lang = .javascript },
        .{ .tag = "typescript", .lang = .javascript },
        .{ .tag = "rs", .lang = .rust },
        .{ .tag = "rust", .lang = .rust },
        .{ .tag = "go", .lang = .go },
        .{ .tag = "sh", .lang = .shell },
        .{ .tag = "bash", .lang = .shell },
        .{ .tag = "zsh", .lang = .shell },
        .{ .tag = "shell", .lang = .shell },
    };
    for (names) |entry| {
        if (std.ascii.eqlIgnoreCase(entry.tag, info)) return entry.lang;
    }
    return .generic;
}

/// State carried across the lines of one code block.
pub const State = struct {
    block_comment: bool = false,
};

/// Languages whose line comments start with `#`.
fn hashComments(lang: Language) bool {
    return lang == .python or lang == .shell;
}

fn keywords(lang: Language) []const []const u8 {
    return switch (lang) {
        .generic => &.{},
        .zig => &.{ "pub", "const", "var", "fn", "if", "else", "while", "for", "switch", "break", "continue", "return", "defer", "errdefer", "try", "catch", "orelse", "and", "or", "not", "struct", "enum", "union", "error", "test", "comptime", "inline", "export", "extern", "unreachable", "async", "await", "suspend", "resume", "volatile", "true", "false", "null", "undefined" },
        .c => &.{ "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "class", "namespace", "template", "typename", "public", "private", "protected", "virtual", "override", "new", "delete", "nullptr", "true", "false", "using" },
        .python => &.{ "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "match", "case", "nonlocal", "not", "or", "pass", "raise", "return", "while", "with", "yield", "True", "False", "None" },
        .javascript => &.{ "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do", "else", "export", "extends", "finally", "for", "from", "function", "get", "if", "import", "in", "instanceof", "let", "new", "null", "of", "return", "set", "static", "super", "switch", "this", "throw", "try", "typeof", "var", "void", "while", "with", "yield", "true", "false", "undefined" },
        .rust => &.{ "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait", "type", "unsafe", "use", "where", "while", "true", "false" },
        .go => &.{ "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "for", "func", "go", "goto", "if", "import", "interface", "map", "package", "range", "return", "select", "struct", "switch", "type", "var", "nil", "true", "false" },
        .shell => &.{ "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac", "function", "select", "in", "return", "exit", "local", "readonly", "declare", "export", "unset", "shift", "trap" },
    };
}

fn isKeyword(lang: Language, word: []const u8) bool {
    for (keywords(lang)) |k| {
        if (std.mem.eql(u8, k, word)) return true;
    }
    return false;
}

/// Renders one code line into a styled string. The baseline style is assumed
/// active on entry and re-stamped after every token span.
pub fn line(alloc: std.mem.Allocator, text: []const u8, lang: Language, state: *State, th: theme.Theme) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    const base = th.paint(.code_block);
    var i: usize = 0;
    while (i < text.len) {
        if (state.block_comment) {
            if (std.mem.indexOfPos(u8, text, i, "*/")) |j| {
                try span(alloc, &out, th, .comment, text[i .. j + 2], base);
                i = j + 2;
                state.block_comment = false;
            } else {
                try span(alloc, &out, th, .comment, text[i..], base);
                i = text.len;
            }
            continue;
        }
        const c = text[i];
        if (c == '/' and i + 1 < text.len and text[i + 1] == '/') {
            try span(alloc, &out, th, .comment, text[i..], base);
            break;
        }
        if (c == '#' and hashComments(lang)) {
            try span(alloc, &out, th, .comment, text[i..], base);
            break;
        }
        if (c == '/' and i + 1 < text.len and text[i + 1] == '*') {
            state.block_comment = true;
            continue;
        }
        if (c == '"' or c == '\'') {
            var j = i + 1;
            while (j < text.len and text[j] != c) : (j += 1) {
                if (text[j] == '\\' and j + 1 < text.len) j += 1;
            }
            const end = @min(j + 1, text.len);
            try span(alloc, &out, th, .string, text[i..end], base);
            i = end;
            continue;
        }
        if (std.ascii.isDigit(c)) {
            var j = i + 1;
            while (j < text.len and (std.ascii.isAlphanumeric(text[j]) or text[j] == '.')) j += 1;
            try span(alloc, &out, th, .number, text[i..j], base);
            i = j;
            continue;
        }
        if (std.ascii.isAlphabetic(c) or c == '_') {
            var j = i + 1;
            while (j < text.len and (std.ascii.isAlphanumeric(text[j]) or text[j] == '_')) j += 1;
            const word = text[i..j];
            if (isKeyword(lang, word)) {
                try span(alloc, &out, th, .keyword, word, base);
            } else {
                try out.appendSlice(alloc, word);
            }
            i = j;
            continue;
        }
        try out.append(alloc, c);
        i += 1;
    }
    return out.toOwnedSlice(alloc);
}

/// One painted token; closes back to the code-block baseline so following
/// text keeps the block's dim color and background (fg_default clears only
/// the foreground, `22` clears any bold from the token style).
fn span(alloc: std.mem.Allocator, out: *std.ArrayList(u8), th: theme.Theme, style: theme.Style, text: []const u8, base: []const u8) !void {
    try out.appendSlice(alloc, th.paint(style));
    try out.appendSlice(alloc, text);
    try out.appendSlice(alloc, theme.fg_default);
    try out.appendSlice(alloc, "\x1b[22m");
    try out.appendSlice(alloc, base);
}

test "keywords, strings, comments, numbers per language" {
    const th = theme.Theme{ .kind = .plain };
    var state: State = .{};
    const styled = try line(std.testing.allocator, "const x = 42; // done", .zig, &state, th);
    defer std.testing.allocator.free(styled);
    try std.testing.expect(std.mem.indexOf(u8, styled, "\x1b[1mconst\x1b[39m\x1b[22m") != null);
    try std.testing.expect(std.mem.indexOf(u8, styled, "42") != null);
    try std.testing.expect(std.mem.indexOf(u8, styled, "// done") != null);
    // Unrecognized words pass through unstyled.
    try std.testing.expect(std.mem.indexOf(u8, styled, "x = ") != null);
}

test "block comment state carries across lines, hash comments for python" {
    const a = std.testing.allocator;
    const th = theme.Theme{ .kind = .plain };
    var state: State = .{};
    const first = try line(a, "/* start", .c, &state, th);
    defer a.free(first);
    try std.testing.expect(state.block_comment);
    const second = try line(a, "end */ x = 1", .c, &state, th);
    defer a.free(second);
    try std.testing.expect(!state.block_comment);
    try std.testing.expect(std.mem.indexOf(u8, second, "*/") != null);
    var py: State = .{};
    const py_line = try line(a, "s = 'hi'  # note", .python, &py, th);
    defer a.free(py_line);
    try std.testing.expect(std.mem.indexOf(u8, py_line, "\x1b[2m# note") != null); // plain level: the attribute carries it
}
