//! The session file's entry for each loop record: one mapping for both
//! drivers (the chat surface and print mode), so they write the same file.
//! An entry borrows its record and the caller's `Scratch` for the call.
const std = @import("std");
const loop = @import("loop.zig");
const session = @import("session.zig");
const editor = @import("../tui/editor.zig");

/// Room for what an entry formats or gathers: a call id, a turn's images
/// and clips (at most the editor's bounds), and the clips' hex digests.
pub const Scratch = struct {
    call: [16]u8 = undefined,
    images: [editor.max_images]session.ImageEntry = undefined,
    clips: [editor.max_clips]session.AudioEntry = undefined,
    digests: [editor.max_clips][64]u8 = undefined,
};

pub fn entry(scratch: *Scratch, record: loop.Record) session.Entry {
    return switch (record) {
        .user => |u| .{ .user = .{ .text = u.text, .images = imageEntries(&scratch.images, u.images), .audio = audioEntries(&scratch.clips, &scratch.digests, u.images) } },
        .assistant => |step| .{ .assistant = .{
            .thinking = step.thinking,
            .answer = step.answer,
            .tool_calls = step.calls,
            .stop = @tagName(step.stop),
            .stats = .{
                .prompt_tokens = step.prompt_tokens,
                .generated = step.generated,
                .prefill_seconds = step.prefill_seconds,
                .decode_seconds = step.decode_seconds,
                .thinking_seconds = step.thinking_seconds,
                .replayed = step.replay != null,
                .replay = if (step.replay) |r| @tagName(r.cause) else "",
                .restored_tokens = if (step.replay) |r| r.restored else 0,
                .reasoning_cut = step.reasoning_cut,
                .reasoning_tokens = step.reasoning_tokens,
            },
            .boundary = if (step.boundary) |b| .{ .bytes = b.bytes, .digest = b.digest } else null,
        } },
        .compaction => |c| .{ .compaction = .{ .first_kept = c.first_kept, .reason = c.reason } },
        .tool_result => |result| .{ .tool_result = .{
            .call = std.fmt.bufPrint(&scratch.call, "{d}", .{result.call}) catch "?",
            .text = result.text,
            .truncated = result.truncated,
            .is_error = result.is_error,
            .summary = result.summary,
        } },
    };
}

/// A turn's images: path, size, and rows.
fn imageEntries(buffer: *[editor.max_images]session.ImageEntry, media: []const loop.Image) []const session.ImageEntry {
    var n: usize = 0;
    for (media) |image| {
        if (image.kind != .image or n == buffer.len) continue;
        buffer[n] = .{
            .path = image.path,
            .width = image.prepared.width,
            .height = image.prepared.height,
            .width_tokens = image.prepared.width_tokens,
            .height_tokens = image.prepared.height_tokens,
        };
        n += 1;
    }
    return buffer[0..n];
}

/// A turn's clips: path, digest (hex, written into `digests`), length, and
/// rows.
fn audioEntries(buffer: *[editor.max_clips]session.AudioEntry, digests: *[editor.max_clips][64]u8, media: []const loop.Image) []const session.AudioEntry {
    var n: usize = 0;
    for (media) |item| {
        if (item.kind != .audio or n == buffer.len) continue;
        digests[n] = std.fmt.bytesToHex(item.digest, .lower);
        buffer[n] = .{ .path = item.path, .sha256 = &digests[n], .samples = item.prepared.width, .tokens = item.prepared.width_tokens };
        n += 1;
    }
    return buffer[0..n];
}

test "a tool result's call id is written in decimal" {
    var scratch: Scratch = .{};
    const mapped = entry(&scratch, .{ .tool_result = .{ .call = 4294967295, .name = "grep", .text = "x", .truncated = false, .is_error = false, .summary = "" } });
    try std.testing.expectEqualStrings("4294967295", mapped.tool_result.call);
}
