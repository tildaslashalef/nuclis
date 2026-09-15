//! Hydrates a directory inventory fixture (`fixtures/*.json`, captured by
//! `scripts/gguf-inventory.py` from the real artifact) into a
//! `gguf.Document`, so adapter tests bind the pinned file's exact metadata
//! and tensor list without a model download. No weights exist: tensor
//! offsets are laid out as the file would, and any attempt to read data
//! fails at the view. Shared by every adapter; the JSON is an
//! independent capture, never generated from a binder's expectations.
//!
//! Fixture value forms: JSON integers become unsigned metadata, floats
//! `float32`, booleans `boolean`, strings `string`; an object with `type`
//! and `count` is an array descriptor whose optional `values` are retained
//! (integers as `signed`, booleans as `boolean`, matching what the parser
//! keeps for the same element types); an object with `sha256` describes a
//! long string the fixture omits, and is skipped (no key). Tensors are
//! `[name, dimensions, encoding_id]`.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const encoding = @import("../tensor/encoding.zig");

pub fn document(gpa: std.mem.Allocator, json_text: []const u8) !gguf.Document {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    const json = try std.json.parseFromSliceLeaky(std.json.Value, alloc, json_text, .{ .allocate = .alloc_always });
    var metadata: std.ArrayList(gguf.Metadata) = .empty;
    var entries = json.object.get("metadata").?.object.iterator();
    while (entries.next()) |entry| {
        const value = entry.value_ptr.*;
        const converted: struct { kind: gguf.Type, value: gguf.Value } = switch (value) {
            .integer => |n| .{ .kind = .uint32, .value = .{ .unsigned = @intCast(n) } },
            .float => |n| .{ .kind = .float32, .value = .{ .float = n } },
            .bool => |b| .{ .kind = .boolean, .value = .{ .boolean = b } },
            .string => |s| .{ .kind = .string, .value = .{ .string = s } },
            .object => |object| blk: {
                if (object.get("sha256") != null) continue;
                const element_type: gguf.Type = @enumFromInt(object.get("type").?.integer);
                var array: gguf.Array = .{
                    .element_type = element_type,
                    .count = @intCast(object.get("count").?.integer),
                    .file_offset = 0,
                };
                if (object.get("values")) |values| {
                    const retained = try alloc.alloc(gguf.Value, values.array.items.len);
                    for (retained, values.array.items) |*out, n| out.* = switch (element_type) {
                        .boolean => .{ .boolean = n.integer != 0 },
                        .float32, .float64 => .{ .float = switch (n) {
                            .float => |f| f,
                            .integer => |i| @floatFromInt(i),
                            else => unreachable,
                        } },
                        .uint8, .uint16, .uint32, .uint64 => .{ .unsigned = @intCast(n.integer) },
                        else => .{ .signed = n.integer },
                    };
                    array.values = retained;
                }
                break :blk .{ .kind = .array, .value = .{ .array = array } };
            },
            else => unreachable,
        };
        try metadata.append(alloc, .{ .key = entry.key_ptr.*, .kind = converted.kind, .value = converted.value });
    }
    const records = json.object.get("tensors").?.array.items;
    const tensors = try alloc.alloc(gguf.Tensor, records.len);
    var offset: u64 = 0;
    for (tensors, records) |*tensor, record| {
        const fields = record.array.items;
        const dimensions = try alloc.alloc(u64, fields[1].array.items.len);
        var elements: u64 = 1;
        for (dimensions, fields[1].array.items) |*dim, value| {
            dim.* = @intCast(value.integer);
            elements *= dim.*;
        }
        const id: u32 = @intCast(fields[2].integer);
        const bytes = try encoding.layout(id).?.byteCount(dimensions);
        tensor.* = .{
            .name = fields[0].string,
            .dimensions = dimensions,
            .encoding_id = id,
            .offset = offset,
            .bytes = bytes,
            .elements = elements,
        };
        offset = std.mem.alignForward(u64, offset + bytes, 32);
    }
    return .{
        .storage = arena,
        .version = 3,
        .file_bytes = offset,
        .directory_bytes = 0,
        .data_offset = 0,
        .alignment = 32,
        .metadata = try metadata.toOwnedSlice(alloc),
        .tensors = tensors,
    };
}

test "fixture forms hydrate to the parser's value kinds" {
    var doc = try document(std.testing.allocator,
        \\{"metadata": {"a.count": 3, "a.eps": 0.5, "a.flag": true, "general.architecture": "x",
        \\  "a.kv": {"count": 2, "offset": 9, "type": 5, "values": [8, 1]},
        \\  "a.swa": {"count": 2, "offset": 9, "type": 7, "values": [1, 0]},
        \\  "a.big": {"count": 100, "offset": 9, "type": 8},
        \\  "tokenizer.chat_template": {"length": 5, "offset": 1, "sha256": "ab", "head": "x"}},
        \\ "tensors": [["t.weight", [4], 0], ["m.weight", [256, 2], 12]]}
    );
    defer doc.deinit();
    try std.testing.expectEqual(@as(u64, 3), doc.get("a.count").?.unsigned);
    try std.testing.expectEqual(@as(f64, 0.5), doc.get("a.eps").?.float);
    try std.testing.expect(doc.get("a.flag").?.boolean);
    try std.testing.expectEqualStrings("x", doc.string("general.architecture").?);
    try std.testing.expectEqual(@as(i64, 1), doc.get("a.kv").?.array.values.?[1].signed);
    try std.testing.expect(doc.get("a.swa").?.array.values.?[0].boolean and !doc.get("a.swa").?.array.values.?[1].boolean);
    try std.testing.expect(doc.get("a.big").?.array.values == null);
    try std.testing.expect(doc.get("tokenizer.chat_template") == null);
    try std.testing.expectEqual(@as(usize, 2), doc.tensors.len);
    try std.testing.expectEqual(@as(u64, 32), doc.tensors[1].offset);
    try std.testing.expectEqual(@as(u64, 256 * 2 / 256 * 144), doc.tensors[1].bytes);
}
