//! Path and gate-name globs. Paths: `**` spans directories, `*` stays inside
//! one component. Names: `*` and `?`, as in a shell.
const std = @import("std");

/// True when `path` (repository-relative, `/`-separated) matches `glob`.
pub fn matches(glob: []const u8, path: []const u8) bool {
    if (glob.len == 0) return path.len == 0;
    if (std.mem.startsWith(u8, glob, "**/")) {
        const rest = glob[3..];
        if (matches(rest, path)) return true;
        for (path, 0..) |byte, i| {
            if (byte == '/' and matches(rest, path[i + 1 ..])) return true;
        }
        return false;
    }
    if (std.mem.startsWith(u8, glob, "**")) {
        const rest = glob[2..];
        for (0..path.len + 1) |i| {
            if (matches(rest, path[i..])) return true;
        }
        return false;
    }
    if (glob[0] == '*') {
        const rest = glob[1..];
        var i: usize = 0;
        while (true) : (i += 1) {
            if (matches(rest, path[i..])) return true;
            if (i == path.len or path[i] == '/') return false;
        }
    }
    return path.len > 0 and path[0] == glob[0] and matches(glob[1..], path[1..]);
}

pub fn matchesAny(globs: []const []const u8, path: []const u8) bool {
    for (globs) |glob| {
        if (matches(glob, path)) return true;
    }
    return false;
}

/// A gate-name pattern: `*` any run of bytes, `?` any one byte.
pub fn matchesName(pattern: []const u8, name: []const u8) bool {
    if (pattern.len == 0) return name.len == 0;
    if (pattern[0] == '*') {
        for (0..name.len + 1) |i| {
            if (matchesName(pattern[1..], name[i..])) return true;
        }
        return false;
    }
    if (name.len == 0) return false;
    if (pattern[0] != '?' and pattern[0] != name[0]) return false;
    return matchesName(pattern[1..], name[1..]);
}

const expect = std.testing.expect;

test "double star spans directories, single star stays in a component" {
    try expect(matches("inference/src/quant/**", "inference/src/quant/q4k.zig"));
    try expect(matches("inference/src/quant/**", "inference/src/quant/fixtures/x.json"));
    try expect(matches("inference/src/models/a*.zig", "inference/src/models/a_metal.zig"));
    try expect(!matches("inference/src/models/a*.zig", "inference/src/models/b.zig"));
    try expect(!matches("inference/src/models/a*.zig", "inference/src/models/fixtures/a.zig"));
    try expect(!matches("inference/src/quant/**", "src/quant/x.zig"));
}

test "a leading double-star-slash also matches at the root" {
    try expect(matches("**/*.zig", "build.zig"));
    try expect(matches("**/*.zig", "src/tui/editor.zig"));
    try expect(!matches("**/*.zig", "src/tui/editor.md"));
    try expect(matches("**/fixtures/**", "inference/src/models/fixtures/a.json"));
    try expect(matches("tests/fixtures/embeddinggemma-*/**", "tests/fixtures/embeddinggemma-inputs/a.wav"));
}

test "gate names take shell patterns" {
    try expect(matchesName("qwen38-*", "qwen38-trace-f16"));
    try expect(matchesName("*-metal", "laya-metal"));
    try expect(matchesName("laya-?etal", "laya-metal"));
    try expect(matchesName("laya-metal", "laya-metal"));
    try expect(!matchesName("qwen38-*", "gemma4-trace"));
    try expect(!matchesName("laya", "laya-metal"));
}
