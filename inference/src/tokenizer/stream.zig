//! Incremental UTF-8 presentation of decoded token bytes. Keeps at most one
//! incomplete scalar between pieces. Malformed bytes and an unfinished final
//! scalar become U+FFFD; exact token IDs remain the lossless representation.
const std = @import("std");
pub const Stream = struct {
    pending: [4]u8 = undefined,
    count: u3 = 0,
    needed: u3 = 0,

    pub fn write(self: *Stream, bytes: []const u8, out: *std.Io.Writer) !void {
        for (bytes) |byte| {
            if (self.count != 0 and byte & 0xc0 != 0x80) {
                try out.writeAll("�");
                self.count = 0;
            }
            if (self.count == 0) {
                self.needed = std.unicode.utf8ByteSequenceLength(byte) catch {
                    try out.writeAll("�");
                    continue;
                };
            }
            self.pending[self.count] = byte;
            self.count += 1;
            if (self.count == self.needed) {
                const scalar = self.pending[0..self.count];
                if (std.unicode.utf8ValidateSlice(scalar)) try out.writeAll(scalar) else try out.writeAll("�");
                self.count = 0;
            }
        }
    }
    pub fn finish(self: *Stream, out: *std.Io.Writer) !void {
        if (self.count != 0) try out.writeAll("�");
        self.count = 0;
    }
};

test "stream handles every split of multibyte text and malformed suffixes" {
    const text = "aé中🙂z";
    for (0..text.len + 1) |split| {
        var buffer: [64]u8 = undefined;
        var out: std.Io.Writer = .fixed(&buffer);
        var stream: Stream = .{};
        try stream.write(text[0..split], &out);
        try stream.write(text[split..], &out);
        try stream.finish(&out);
        try std.testing.expectEqualStrings(text, out.buffered());
    }
    var buffer: [64]u8 = undefined;
    var out: std.Io.Writer = .fixed(&buffer);
    var stream: Stream = .{};
    try stream.write("\xff\xe2x\xed\xa0\x80ok\xf0\x9f", &out);
    try stream.finish(&out);
    try std.testing.expectEqualStrings("��x�ok�", out.buffered());
}
