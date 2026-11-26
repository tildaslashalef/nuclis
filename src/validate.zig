//! CLI rendering for structural validation. Execution capability stays explicit
//! even after every required tensor has a valid name, shape, and storage layout.
const std = @import("std");
const Summary = @import("inference").models.Summary;
const kit = @import("ui.zig");

pub fn render(arena: std.mem.Allocator, path: []const u8, summary: Summary, out: *std.Io.Writer, json: bool, ui: kit.Ui) !void {
    if (json) {
        try std.json.Stringify.value(.{
            .schema_version = @as(u32, 2),
            .model_path = path,
            .model = summary,
        }, .{ .whitespace = .indent_2 }, out);
        return out.writeByte('\n');
    }
    const v = kit.values;
    const clean_path = try kit.text.clean(arena, path);
    try kit.blocks.header(out, ui, std.fs.path.basename(clean_path), summary.profile);
    var kinds: std.ArrayList(u8) = .empty;
    for (summary.layer_kinds, 0..) |kind, i| try kinds.print(arena, "{s}{d} {s}", .{ if (i > 0) ", " else "", kind.count, kind.kind });
    var fields: std.ArrayList(kit.blocks.Field) = .empty;
    try fields.append(arena, .{ .key = "Validation", .value = summary.validation });
    try fields.append(arena, .{ .key = "Text layers", .value = try std.fmt.allocPrint(arena, "{d} ({s})", .{ summary.decoder_layers, kinds.items }) });
    try fields.append(arena, .{ .key = "Text weights", .value = try std.fmt.allocPrint(arena, "{f} · {f}", .{ v.count(summary.text_tensors, "tensor"), v.size(summary.text_tensor_bytes) }) });
    if (summary.auxiliary_prediction_layers > 0)
        try fields.append(arena, .{ .key = "Draft head", .value = try std.fmt.allocPrint(arena, "embedded: {f}, {f} · {f}; runs on CPU and Metal", .{ v.count(summary.auxiliary_prediction_layers, "prediction layer"), v.count(summary.auxiliary_tensors, "tensor"), v.size(summary.auxiliary_tensor_bytes) }) });
    if (summary.rotated_basis) |basis| try fields.append(arena, .{ .key = "Rotated basis", .value = try std.fmt.allocPrint(arena, "{s} (the runtimes transform every projection input)", .{basis}) });
    try kit.blocks.detail(out, ui, &.{.{ .fields = fields.items }});
    try out.writeByte('\n');
    if (summary.inference_available and summary.metal_available)
        try kit.blocks.result(out, ui, .ok, "structure matches the text profile", "`generate` runs it on CPU and Metal")
    else if (summary.inference_available)
        try kit.blocks.result(out, ui, .ok, "structure matches the text profile", "`generate` runs it on the CPU only: no Metal plan yet")
    else
        try kit.blocks.result(out, ui, .warning, "structure matches the text profile", "no executor runs it: `generate` refuses it");
}

test "validation JSON reports structural check kind and executable binding" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    try render(arena_state.allocator(), "model.gguf", .{
        .profile = "qwen35_27b",
        .decoder_layers = 64,
        .layer_kinds = &.{ .{ .kind = "full_attention", .count = 16 }, .{ .kind = "delta_net", .count = 48 } },
        .auxiliary_prediction_layers = 1,
        .text_tensors = 851,
        .auxiliary_tensors = 15,
        .text_tensor_bytes = 10,
        .auxiliary_tensor_bytes = 2,
    }, &out.writer, true, .plain);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const model = parsed.value.object.get("model").?.object;
    try std.testing.expectEqualStrings("structure_only", model.get("validation").?.string);
    try std.testing.expect(model.get("inference_available").?.bool);
}

test "a binding with no executor is a warning, and one without a Metal plan says CPU only" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const cases = [_]struct { inference: bool, metal: bool, says: []const u8 }{
        .{ .inference = false, .metal = false, .says = "`generate` refuses it" },
        .{ .inference = true, .metal = false, .says = "on the CPU only" },
    };
    for (cases) |case| {
        var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer out.deinit();
        try render(arena_state.allocator(), "model.gguf", .{
            .profile = "synthetic",
            .decoder_layers = 2,
            .layer_kinds = &.{.{ .kind = "full_attention", .count = 2 }},
            .text_tensors = 10,
            .auxiliary_tensors = 0,
            .text_tensor_bytes = 10,
            .auxiliary_tensor_bytes = 0,
            .inference_available = case.inference,
            .metal_available = case.metal,
        }, &out.writer, false, .plain);
        try std.testing.expect(std.mem.indexOf(u8, out.written(), case.says) != null);
    }
}
