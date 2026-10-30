//! Drama 3's last sentences kept between speech requests: a finished
//! request's last sentence (its text and codes) under the id its response
//! sent as `x-speech-context`, so a later request in the same model and
//! voice takes it as history (docs/guide/api.md § Drama 3). Each lives
//! `ttl_ns`; at most `capacity`, the oldest dropped first. Thread-safe.
const std = @import("std");
const inference = @import("inference");

const Codes = inference.models.drama3_text.Codes;

pub const ttl_ns: u64 = 5 * 60 * std.time.ns_per_s;
pub const capacity = 32;
pub const Id = [36]u8;

/// A kept sentence, owned by the allocator that `lookup` was given.
pub const Turn = struct {
    text: []u8,
    codes: []Codes,

    pub fn deinit(self: *Turn, gpa: std.mem.Allocator) void {
        gpa.free(self.text);
        gpa.free(self.codes);
        self.* = undefined;
    }
};

/// What a request's `context` found; the request log's words.
pub const Outcome = enum { hit, missing, expired, other_voice };

pub const Lookup = union(Outcome) {
    hit: Turn,
    missing,
    expired,
    other_voice,
};

const Entry = struct {
    id: Id,
    /// The model's directory and the voice's name, owned.
    model: []u8,
    voice: []u8,
    turn: Turn,
    at: std.Io.Timestamp,

    fn deinit(self: *Entry, gpa: std.mem.Allocator) void {
        gpa.free(self.model);
        gpa.free(self.voice);
        self.turn.deinit(gpa);
    }
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    entries: std.ArrayList(Entry) = .empty,

    pub fn init(gpa: std.mem.Allocator) Store {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Store) void {
        for (self.entries.items) |*e| e.deinit(self.gpa);
        self.entries.deinit(self.gpa);
        self.* = undefined;
    }

    /// Keeps a copy of `text` and `codes` under `id`, made at `now`.
    pub fn put(self: *Store, io: std.Io, now: std.Io.Timestamp, id: Id, model: []const u8, voice: []const u8, text: []const u8, codes: []const Codes) !void {
        var entry: Entry = .{ .id = id, .model = try self.gpa.dupe(u8, model), .voice = undefined, .turn = undefined, .at = now };
        errdefer self.gpa.free(entry.model);
        entry.voice = try self.gpa.dupe(u8, voice);
        errdefer self.gpa.free(entry.voice);
        entry.turn.text = try self.gpa.dupe(u8, text);
        errdefer self.gpa.free(entry.turn.text);
        entry.turn.codes = try self.gpa.dupe(Codes, codes);
        errdefer self.gpa.free(entry.turn.codes);

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        var i: usize = 0;
        while (i < self.entries.items.len) {
            const old = &self.entries.items[i];
            if (std.mem.eql(u8, &old.id, &id) or expired(old.at, now)) {
                var gone = self.entries.orderedRemove(i);
                gone.deinit(self.gpa);
            } else i += 1;
        }
        if (self.entries.items.len == capacity) {
            var gone = self.entries.orderedRemove(0);
            gone.deinit(self.gpa);
        }
        try self.entries.append(self.gpa, entry);
    }

    /// The sentence under `id` for this model and voice at `now`, copied
    /// into `gpa` on a hit. An entry stays for later lookups.
    pub fn lookup(self: *Store, gpa: std.mem.Allocator, io: std.Io, now: std.Io.Timestamp, id: []const u8, model: []const u8, voice: []const u8) !Lookup {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        for (self.entries.items) |*e| {
            if (!std.mem.eql(u8, &e.id, id)) continue;
            if (expired(e.at, now)) return .expired;
            if (!std.mem.eql(u8, e.model, model) or !std.mem.eql(u8, e.voice, voice)) return .other_voice;
            const text = try gpa.dupe(u8, e.turn.text);
            errdefer gpa.free(text);
            return .{ .hit = .{ .text = text, .codes = try gpa.dupe(Codes, e.turn.codes) } };
        }
        return .missing;
    }
};

fn expired(at: std.Io.Timestamp, now: std.Io.Timestamp) bool {
    return at.durationTo(now).nanoseconds > ttl_ns;
}

test "a kept sentence is found for its model and voice until it expires" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var store: Store = .init(gpa);
    defer store.deinit();
    const t0: std.Io.Timestamp = .{ .nanoseconds = 1_000 };
    const id: Id = @splat('a');
    const codes = [_]Codes{@splat(7)};
    try store.put(io, t0, id, "/m/drama", "anna", "يا عم.", &codes);

    var hit = try store.lookup(gpa, io, t0, &id, "/m/drama", "anna");
    try std.testing.expectEqualStrings("يا عم.", hit.hit.text);
    try std.testing.expectEqual(@as(u16, 7), hit.hit.codes[0][3]);
    hit.hit.deinit(gpa);

    try std.testing.expectEqual(Outcome.other_voice, try store.lookup(gpa, io, t0, &id, "/m/drama", "ben"));
    try std.testing.expectEqual(Outcome.missing, try store.lookup(gpa, io, t0, &@as(Id, @splat('b')), "/m/drama", "anna"));
    const later: std.Io.Timestamp = .{ .nanoseconds = t0.nanoseconds + ttl_ns + 1 };
    try std.testing.expectEqual(Outcome.expired, try store.lookup(gpa, io, later, &id, "/m/drama", "anna"));
}

test "the store keeps at most capacity, dropping the oldest" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var store: Store = .init(gpa);
    defer store.deinit();
    const codes = [_]Codes{@splat(1)};
    for (0..capacity + 1) |i| {
        var id: Id = @splat('0');
        id[0] = @intCast('A' + i);
        try store.put(io, .{ .nanoseconds = @intCast(i) }, id, "m", "v", "x", &codes);
    }
    try std.testing.expectEqual(@as(usize, capacity), store.entries.items.len);
    var first: Id = @splat('0');
    first[0] = 'A';
    try std.testing.expectEqual(Outcome.missing, try store.lookup(gpa, io, .{ .nanoseconds = 40 }, &first, "m", "v"));
}
