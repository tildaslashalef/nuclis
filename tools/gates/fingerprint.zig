//! A file's fingerprint: what a gate's result can depend on. Zig sources hash
//! their tokens without doc comments, so comment and whitespace edits leave
//! it unchanged (line comments are not tokens); anything else hashes raw.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Digest = [Sha256.digest_length]u8;

pub fn ofFile(gpa: std.mem.Allocator, path: []const u8, bytes: []const u8) std.mem.Allocator.Error!Digest {
    if (!std.mem.endsWith(u8, path, ".zig")) {
        var digest: Digest = undefined;
        Sha256.hash(bytes, &digest, .{});
        return digest;
    }
    const source = try gpa.dupeSentinel(u8, bytes, 0);
    defer gpa.free(source);
    var hasher = Sha256.init(.{});
    var tokens = std.zig.Tokenizer.init(source);
    while (true) {
        const token = tokens.next();
        switch (token.tag) {
            .eof => break,
            .doc_comment, .container_doc_comment => continue,
            else => {},
        }
        hasher.update(source[token.loc.start..token.loc.end]);
        hasher.update(&.{0});
    }
    return hasher.finalResult();
}

fn same(a: []const u8, b: []const u8) !bool {
    const gpa = std.testing.allocator;
    return std.mem.eql(u8, &try ofFile(gpa, "a.zig", a), &try ofFile(gpa, "a.zig", b));
}

test "comments and whitespace leave a Zig fingerprint unchanged" {
    const base = "const a = 1;\npub fn f() u32 {\n    return a;\n}\n";
    try std.testing.expect(try same(base, "// note\nconst a = 1;\n\n\npub fn f() u32 { return a; }\n"));
    try std.testing.expect(try same(base, "//! module\n/// contract\nconst a = 1;\n/// more\npub fn f() u32 {\n    return a; // why\n}\n"));
}

test "code and string edits change a Zig fingerprint" {
    const base = "const a = 1;\nconst s = \"x y\";\n";
    try std.testing.expect(!try same(base, "const a = 2;\nconst s = \"x y\";\n"));
    try std.testing.expect(!try same(base, "const a = 1;\nconst s = \"x  y\";\n"));
}

test "a non-Zig file hashes its bytes" {
    const gpa = std.testing.allocator;
    const a = try ofFile(gpa, "k.metal", "// a\nkernel\n");
    const b = try ofFile(gpa, "k.metal", "// b\nkernel\n");
    try std.testing.expect(!std.mem.eql(u8, &a, &b));
}
