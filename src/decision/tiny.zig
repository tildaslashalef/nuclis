//! A whole tiny decision checkpoint for tests: `laya.writeTiny`'s weights
//! with an embedding for a byte-level vocabulary (every byte a token, no
//! merges, then `[CLS]`, `[SEP]`, `[MASK]`), so any text tokenizes, plus the
//! tokenizer and agent configs `Decider.open` reads. Every logit is 0.25.
const std = @import("std");
const inference = @import("inference");

const laya = inference.models.laya;

pub const vocabulary = 259;

const config =
    \\{"model_type":"modernbert","hidden_size":64,"num_attention_heads":8,"num_hidden_layers":2,
    \\ "intermediate_size":64,"vocab_size":259,"local_attention":4,
    \\ "layer_types":["full_attention","sliding_attention"],
    \\ "rope_parameters":{"full_attention":{"rope_theta":160000.0},"sliding_attention":{"rope_theta":10000.0}}}
;

/// Writes the checkpoint into `dir`.
pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    const embedding = "encoder.embeddings.tok_embeddings.weight";
    try laya.writeTiny(gpa, io, dir, embedding, .{ .name = embedding, .shape = &.{ vocabulary, 64 } });
    try dir.writeFile(io, .{ .sub_path = "encoder/config.json", .data = config });
    try dir.writeFile(io, .{ .sub_path = "rl_agent_config.json", .data = "{\"max_len\":256,\"head_max_len\":96}" });
    try dir.createDirPath(io, "tokenizer");
    try dir.writeFile(io, .{ .sub_path = "tokenizer/tokenizer_config.json", .data = "{\"cls_token\":\"[CLS]\",\"sep_token\":\"[SEP]\",\"mask_token\":\"[MASK]\"}" });
    var json: std.Io.Writer.Allocating = .init(gpa);
    defer json.deinit();
    try tokenizer(&json.writer);
    try dir.writeFile(io, .{ .sub_path = "tokenizer/tokenizer.json", .data = json.written() });
}

/// GPT-2's byte-to-character map: printable bytes stand for themselves, the
/// rest take code points from 256 up, in byte order.
fn byteChar(b: u8) u21 {
    const printable = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or b >= 0xAE;
    if (printable) return b;
    var n: u21 = 0;
    for (0..b) |x| {
        const c: u8 = @intCast(x);
        if (!((c >= '!' and c <= '~') or (c >= 0xA1 and c <= 0xAC) or c >= 0xAE)) n += 1;
    }
    return 256 + n;
}

fn tokenizer(out: *std.Io.Writer) !void {
    try out.writeAll(
        \\{"version":"1.0","truncation":null,"padding":null,"added_tokens":[
        \\{"id":256,"content":"[CLS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        \\{"id":257,"content":"[SEP]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        \\{"id":258,"content":"[MASK]","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":true}],
        \\"normalizer":null,
        \\"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
        \\"post_processor":null,
        \\"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        \\"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,
        \\"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":
    );
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    for (0..256) |b| {
        var utf8: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(byteChar(@intCast(b)), &utf8);
        try s.objectField(utf8[0..len]);
        try s.write(b);
    }
    try s.endObject();
    try out.writeAll(",\"merges\":[]}}");
}

test "the tiny checkpoint opens and answers any text on the CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(gpa, io, tmp.dir);
    const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(path);
    var decider = try inference.decide.Decider.open(gpa, io, .{ .directory = path }, .cpu);
    defer decider.deinit(io);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var why: inference.profiles.laya.Diagnostic = .{};
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, "{\"type\":\"noul\",\"instructions\":\"Is it urgent? ü\"}", .{});
    const q = try inference.profiles.laya.parseQuestion(arena, "urgent", parsed, &why);
    var timings: inference.decide.Timings = .{};
    const results = try decider.decide(arena, io, &.{.{ .text = "The site is down — fix it now!" }}, &.{q}, .{}, &timings);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.25 }, results[0].answers[0].logits);
}
