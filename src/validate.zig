//! CLI rendering for structural validation. Execution capability stays explicit
//! even after every required tensor has a valid name, shape, and storage layout.
const std = @import("std");
const Summary = @import("inference").models.Summary;
const style = @import("tui/style.zig");

pub fn render(path: []const u8, summary: Summary, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
    if (json) {
        try std.json.Stringify.value(.{
            .schema_version = @as(u32, 2),
            .model_path = path,
            .model = summary,
        }, .{ .whitespace = .indent_2 }, out);
        return out.writeByte('\n');
    }
    const label = sty.on(.label);
    const number = sty.on(.number);
    const off = sty.off();
    try out.print("{s}Profile:{s} {s}{s}{s}\n{s}Validation:{s} {s}\n", .{ label, off, sty.on(.code), summary.profile, off, label, off, summary.validation });
    try out.print("{s}Text layers:{s} {s}{d}{s} (", .{ label, off, number, summary.decoder_layers, off });
    for (summary.layer_kinds, 0..) |kind, i| try out.print("{s}{s}{d}{s} {s}", .{ if (i > 0) ", " else "", number, kind.count, off, kind.kind });
    try out.writeAll(")\n");
    try out.print("{s}Text weights:{s} {s}{d}{s} tensors, {s}{d}{s} bytes\n", .{ label, off, number, summary.text_tensors, off, number, summary.text_tensor_bytes, off });
    try out.print("{s}Auxiliary prediction:{s} {s}{d}{s} layer, {s}{d}{s} tensors, {s}{d}{s} bytes (excluded from text binding)\n", .{ label, off, number, summary.auxiliary_prediction_layers, off, number, summary.auxiliary_tensors, off, number, summary.auxiliary_tensor_bytes, off });
    if (summary.rotated_basis) |basis| try out.print("{s}Rotated basis:{s} {s} (the runtimes transform every projection input)\n", .{ label, off, basis });
    try out.print("{s}Structure matches the text profile executed by `generate` (CPU and Metal backends).{s}\n", .{ sty.on(.success), off });
}

test "validation JSON reports structural check kind and executable binding" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try render("model.gguf", .{
        .profile = "qwen35_27b",
        .decoder_layers = 64,
        .layer_kinds = &.{ .{ .kind = "full_attention", .count = 16 }, .{ .kind = "delta_net", .count = 48 } },
        .auxiliary_prediction_layers = 1,
        .text_tensors = 851,
        .auxiliary_tensors = 15,
        .text_tensor_bytes = 10,
        .auxiliary_tensor_bytes = 2,
    }, &out.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const model = parsed.value.object.get("model").?.object;
    try std.testing.expectEqualStrings("structure_only", model.get("validation").?.string);
    try std.testing.expect(model.get("inference_available").?.bool);
}
