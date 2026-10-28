//! Phrases a transducer's greedy decoding leans toward: names the recognizer
//! was never taught. A phrase is matched as text against the pieces the
//! decoder may emit next, so every segmentation of it counts and no encoder
//! is needed; a piece that starts or continues a phrase has `bonus` added to
//! its logit before the argmax. Decoding without phrases is unchanged.
//! docs/models/nemotron-asr.md § Phrase boosting.
const std = @import("std");

/// The bounds a client's list is held to.
pub const max_phrases = 64;
pub const max_phrase_bytes = 48;
/// Logits added to a piece that starts a phrase and to one that continues
/// it, chosen by measurement (nemotron-asr.md § Phrase boosting).
pub const default_bonus: f32 = 2.0;
pub const default_follow: f32 = 6.0;

const space = "\u{2581}";
/// Phrase positions followed at once; more phrases sharing a prefix than
/// this lose the latest starts, never a crash.
const max_active = 16;

pub const Error = std.mem.Allocator.Error || error{InvalidPhrase};

pub const Boost = struct {
    alloc: std.mem.Allocator,
    /// Added to a piece that starts a phrase.
    bonus: f32,
    /// Added to a piece that continues one: larger, since a phrase begun
    /// and left is a misspelled word.
    follow: f32,
    /// Each phrase as the pieces spell it: `▁` before every word.
    phrases: [][]u8,
    /// `steps[i][off]`: the pieces that match phrase `i` at byte `off`.
    steps: [][][]Step,

    pub const Step = struct { token: u32, len: u16 };

    /// `pieces` answers `count()` and `piece(id) ?[]const u8`, and
    /// `isSpecial(id)`; it is read here only. A phrase is trimmed, its
    /// spaces collapsed; an empty one, one over `max_phrase_bytes`, one with
    /// a control character, or more than `max_phrases` is `InvalidPhrase`.
    pub fn init(alloc: std.mem.Allocator, phrases: []const []const u8, bonus: f32, follow: f32, pieces: anytype) Error!Boost {
        if (phrases.len > max_phrases) return error.InvalidPhrase;
        const spelled_all = try alloc.alloc([]u8, phrases.len);
        errdefer alloc.free(spelled_all);
        const steps_all = try alloc.alloc([][]Step, phrases.len);
        errdefer alloc.free(steps_all);
        var self: Boost = .{ .alloc = alloc, .bonus = bonus, .follow = follow, .phrases = spelled_all, .steps = steps_all };
        var built: usize = 0;
        errdefer for (self.phrases[0..built], self.steps[0..built]) |p, s| free(alloc, p, s);
        for (phrases) |raw| {
            const spelled = try spell(alloc, raw);
            errdefer alloc.free(spelled);
            const steps = try alloc.alloc([]Step, spelled.len);
            var filled: usize = 0;
            errdefer {
                for (steps[0..filled]) |s| alloc.free(s);
                alloc.free(steps);
            }
            for (steps, 0..) |*at, off| {
                var list: std.ArrayList(Step) = .empty;
                errdefer list.deinit(alloc);
                const rest = spelled[off..];
                for (0..pieces.count()) |id| {
                    const token: u32 = @intCast(id);
                    if (pieces.isSpecial(token)) continue;
                    const piece = pieces.piece(token) orelse continue;
                    if (piece.len == 0 or piece.len > rest.len or !std.mem.startsWith(u8, rest, piece)) continue;
                    try list.append(alloc, .{ .token = token, .len = @intCast(piece.len) });
                }
                at.* = try list.toOwnedSlice(alloc);
                filled += 1;
            }
            self.phrases[built] = spelled;
            self.steps[built] = steps;
            built += 1;
        }
        return self;
    }

    pub fn deinit(self: *Boost) void {
        for (self.phrases, self.steps) |p, s| free(self.alloc, p, s);
        self.alloc.free(self.phrases);
        self.alloc.free(self.steps);
        self.* = undefined;
    }

    fn free(alloc: std.mem.Allocator, phrase: []u8, steps: [][]Step) void {
        for (steps) |s| alloc.free(s);
        alloc.free(steps);
        alloc.free(phrase);
    }

    /// `▁word▁word`, owned.
    fn spell(alloc: std.mem.Allocator, raw: []const u8) Error![]u8 {
        if (!valid(raw)) return error.InvalidPhrase;
        const text = std.mem.trim(u8, raw, " \t\r\n");
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(alloc);
        var words = std.mem.tokenizeAny(u8, text, " \t");
        while (words.next()) |word| {
            try out.appendSlice(alloc, space);
            try out.appendSlice(alloc, word);
        }
        return out.toOwnedSlice(alloc);
    }
};

/// Whether `raw` can be a phrase: some text once trimmed, at most
/// `max_phrase_bytes`, UTF-8 with no control character but spaces and tabs.
pub fn valid(raw: []const u8) bool {
    const text = std.mem.trim(u8, raw, " \t\r\n");
    if (text.len == 0 or text.len > max_phrase_bytes or !std.unicode.utf8ValidateSlice(text)) return false;
    for (text) |c| if ((c < 0x20 and c != '\t') or c == 0x7f) return false;
    return true;
}

/// Where a decoder stands in the phrases: the positions its emitted pieces
/// have reached. Reset with the decoder.
pub const State = struct {
    active: [max_active]Pos = undefined,
    len: usize = 0,

    const Pos = struct { phrase: u16, off: u16 };

    pub fn reset(self: *State) void {
        self.len = 0;
    }

    /// The token to emit: `best` (the plain argmax) unless a piece that
    /// starts or continues a phrase scores higher with the bonus. A bare
    /// `▁` never gets it: it begins every word.
    pub fn choose(self: *const State, boost: *const Boost, logits: []const f32, best: u32) u32 {
        var token = best;
        var value = logits[best];
        for (self.active[0..self.len]) |pos| {
            for (boost.steps[pos.phrase][pos.off]) |step| consider(logits, boost.follow, step.token, &token, &value);
        }
        for (boost.steps) |steps| for (steps[0]) |step| {
            if (step.len == space.len) continue;
            consider(logits, boost.bonus, step.token, &token, &value);
        };
        return token;
    }

    fn consider(logits: []const f32, bonus: f32, candidate: u32, token: *u32, value: *f32) void {
        if (candidate >= logits.len) return;
        const v = logits[candidate] + bonus;
        if (v > value.*) {
            token.* = candidate;
            value.* = v;
        }
    }

    /// Moves past an emitted `token`: positions it continues advance (a
    /// phrase it completes is done), the others are dropped, and phrases it
    /// starts begin.
    pub fn advance(self: *State, boost: *const Boost, token: u32) void {
        var next: [max_active]Pos = undefined;
        var n: usize = 0;
        for (self.active[0..self.len]) |pos| {
            for (boost.steps[pos.phrase][pos.off]) |step| if (step.token == token) {
                const off = pos.off + step.len;
                if (off < boost.phrases[pos.phrase].len and n < max_active) {
                    next[n] = .{ .phrase = pos.phrase, .off = @intCast(off) };
                    n += 1;
                }
            };
        }
        for (boost.steps, 0..) |steps, i| for (steps[0]) |step| if (step.token == token) {
            if (step.len < boost.phrases[i].len and n < max_active) {
                next[n] = .{ .phrase = @intCast(i), .off = step.len };
                n += 1;
            }
        };
        self.active = next;
        self.len = n;
    }
};

const TestPieces = struct {
    list: []const []const u8,
    fn count(self: TestPieces) usize {
        return self.list.len;
    }
    fn piece(self: TestPieces, id: u32) ?[]const u8 {
        return self.list[id];
    }
    fn isSpecial(self: TestPieces, id: u32) bool {
        return std.mem.startsWith(u8, self.list[id], "<");
    }
};

test "a phrase is matched in any segmentation; its pieces win by the bonus, others do not" {
    const alloc = std.testing.allocator;
    // 0 ▁, 1 ▁S, 2 io, 3 b, 4 ha, 5 n, 6 ▁Si, 7 ob, 8 ▁the, 9 <blank>, 10 bhan
    const pieces: TestPieces = .{ .list = &.{ "▁", "▁S", "io", "b", "ha", "n", "▁Si", "ob", "▁the", "<blank>", "bhan" } };
    var boost = try Boost.init(alloc, &.{" Siobhan "}, 3.0, 3.0, pieces);
    defer boost.deinit();
    try std.testing.expectEqualStrings("▁Siobhan", boost.phrases[0]);

    var state: State = .{};
    var logits: [11]f32 = @splat(0);
    // The blank leads by 2: a phrase start (▁S or ▁Si) wins with the bonus,
    // the bare ▁ and an unrelated piece do not.
    logits[9] = 2;
    logits[1] = 0.5;
    logits[0] = 1.5;
    logits[8] = 1.9;
    try std.testing.expectEqual(@as(u32, 1), state.choose(&boost, &logits, 9));
    // A lead beyond the bonus stays.
    logits[9] = 4;
    try std.testing.expectEqual(@as(u32, 9), state.choose(&boost, &logits, 9));

    // After ▁S, "io" continues it; "ob" (which continues ▁Si) does not.
    state.advance(&boost, 1);
    @memset(&logits, 0);
    logits[9] = 2;
    logits[7] = 1.9;
    logits[2] = 0.1;
    try std.testing.expectEqual(@as(u32, 2), state.choose(&boost, &logits, 9));
    state.advance(&boost, 2);
    state.advance(&boost, 10); // "bhan" completes it
    try std.testing.expectEqual(@as(usize, 0), state.len);
    // A piece off the phrase drops the position.
    state.advance(&boost, 6);
    try std.testing.expectEqual(@as(usize, 1), state.len);
    state.advance(&boost, 8);
    try std.testing.expectEqual(@as(usize, 0), state.len);
}

test "phrases are bounded and must be text" {
    const alloc = std.testing.allocator;
    const pieces: TestPieces = .{ .list = &.{"▁a"} };
    try std.testing.expectError(error.InvalidPhrase, Boost.init(alloc, &.{"  "}, 1, 1, pieces));
    try std.testing.expectError(error.InvalidPhrase, Boost.init(alloc, &.{"a\x00b"}, 1, 1, pieces));
    const long: [max_phrase_bytes + 1]u8 = @splat('x');
    try std.testing.expectError(error.InvalidPhrase, Boost.init(alloc, &.{&long}, 1, 1, pieces));
    const many: [max_phrases + 1][]const u8 = @splat("a");
    try std.testing.expectError(error.InvalidPhrase, Boost.init(alloc, &many, 1, 1, pieces));
    var two = try Boost.init(alloc, &.{ "New  York", "a" }, 1, 1, pieces);
    defer two.deinit();
    try std.testing.expectEqualStrings("▁New▁York", two.phrases[0]);
}

test "a failed allocation frees what was built" {
    const pieces: TestPieces = .{ .list = &.{ "▁a", "b", "▁c" } };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(alloc: std.mem.Allocator, p: TestPieces) !void {
            var b = try Boost.init(alloc, &.{ "ab", "c" }, 1, 1, p);
            b.deinit();
        }
    }.run, .{pieces});
}
