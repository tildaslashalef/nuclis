//! Xet reconstruction and HTTPS fallback.
//!
//! A `Session` is the per-file transport state shared by every worker: the
//! pooled HTTP client (keep-alive connections to the Hub, the CAS, and the
//! xorb store) and the CAS read token, fetched once and reused until it nears
//! expiry or a worker invalidates it after a 401/403.
//!
//! A Xet file is fetched span by span (256 MiB of output per reconstruction
//! call). The reconstruction names *terms* (runs of chunks of one xorb, in
//! file order) and, per xorb, signed URL *units* (a byte range holding a
//! range of chunks). The CAS cuts units on its own fixed boundaries, so a
//! term usually needs part of one or two units and neighbouring terms share
//! them. `Plan` dedups the units and records each one's last user; `execute`
//! fetches every unit exactly once through a sliding window of concurrent
//! futures and decodes chunks straight into the sink in file order, skipping
//! chunks outside the term by their headers. Fetching per output window
//! instead (the previous design) pulled every unit that a window touched and
//! the next window pulled it again: 1.5× the file in CDN traffic.
//!
//! Work is bounded by `concurrency` units in flight (each up to 64 MiB of
//! xorb bytes, typically ~8 MB) plus units held for a later term, and two
//! chunk buffers (128 KiB each).
const std = @import("std");
const http = @import("http.zig");
const hub = @import("hub.zig");
const xorb = @import("xorb.zig");
const A = std.mem.Allocator;
const Io = std.Io;
const Value = std.json.Value;
/// Output bytes per reconstruction call (Xet). Units keep flowing only
/// within a span; at a boundary the window drains before the next plan is
/// requested, so spans are large (a 256 MiB plan is ~100 KB of JSON).
pub const span_size = 256 * 1024 * 1024;
pub const window_size = 8 * 1024 * 1024;
const max_xorb = 64 * 1024 * 1024;
/// A serialized xorb: 64 MiB of chunk data plus chunk headers and LZ4 framing.
const max_unit = max_xorb + 1024 * 1024;
/// Bytes per fetch: a unit is fetched as pieces of this size, in parallel,
/// each landing in its place in the unit's buffer.
const piece_size = window_size;
pub const max_concurrency = 16;

pub const Session = struct {
    client: std.http.Client,
    hub_token: ?[]const u8,
    /// Borrowed for the session's lifetime (repo and pinned revision).
    catalog: hub.Catalog,
    mutex: Io.Mutex = .init,
    /// Credential strings live in this arena until deinit; a refresh appends
    /// rather than frees, so a worker holding an older copy stays valid.
    arena: std.heap.ArenaAllocator,
    cached: ?Credential = null,

    pub const Credential = struct { access_token: []const u8, cas_url: []const u8, expires: u64 };

    pub fn init(allocator: A, io: Io, hub_token: ?[]const u8, catalog: hub.Catalog) Session {
        return .{ .client = http.client(allocator, io), .hub_token = hub_token, .catalog = catalog, .arena = .init(allocator) };
    }
    /// All requests made through the session must have completed (workers joined).
    pub fn deinit(s: *Session) void {
        s.client.deinit();
        s.arena.deinit();
        s.* = undefined;
    }

    /// The CAS read token for the pinned revision, shared by all workers.
    /// Refreshed under the mutex when within a minute of its expiry.
    fn credential(s: *Session) !Credential {
        try s.mutex.lock(s.client.io);
        defer s.mutex.unlock(s.client.io);
        const now = Io.Clock.real.now(s.client.io).toSeconds();
        if (s.cached) |c| if (now >= 0 and c.expires > @as(u64, @intCast(now)) + 60) return c;
        var temp = std.heap.ArenaAllocator.init(s.client.allocator);
        defer temp.deinit();
        const url = try std.fmt.allocPrint(temp.allocator(), "https://huggingface.co/api/models/{s}/xet-read-token/{s}", .{ s.catalog.repo_id, s.catalog.revision });
        var auth = try http.get(&s.client, .{ .url = url, .token = s.hub_token, .max_bytes = 256 * 1024 });
        defer auth.deinit();
        try hub.hubStatus(auth.status, s.hub_token);
        const Token = struct { accessToken: []const u8, casUrl: []const u8, exp: u64 };
        const parsed = try std.json.parseFromSlice(Token, temp.allocator(), auth.body, .{ .ignore_unknown_fields = true });
        _ = try http.validateUrl(parsed.value.casUrl);
        const a = s.arena.allocator();
        const c: Credential = .{
            .access_token = try a.dupe(u8, parsed.value.accessToken),
            .cas_url = try a.dupe(u8, std.mem.trimEnd(u8, parsed.value.casUrl, "/")),
            .expires = parsed.value.exp,
        };
        s.cached = c;
        return c;
    }
    fn invalidate(s: *Session) void {
        s.mutex.lockUncancelable(s.client.io);
        defer s.mutex.unlock(s.client.io);
        s.cached = null;
    }
};

pub const Remote = struct {
    /// Owned by the allocator passed to resolve; other metadata stays in Catalog.
    url: []const u8,
    xet_hash: ?[64]u8,
    pub fn deinit(r: Remote, a: A) void {
        a.free(r.url);
    }
};

pub fn resolve(s: *Session, file: hub.File) !Remote {
    const a = s.client.allocator;
    const encoded = try hub.encode(a, file.name, true);
    defer a.free(encoded);
    const url = try std.fmt.allocPrint(a, "https://huggingface.co/{s}/resolve/{s}/{s}", .{ s.catalog.repo_id, s.catalog.revision, encoded });
    errdefer a.free(url);
    var response = try http.get(&s.client, .{ .url = url, .token = s.hub_token, .head = true });
    defer response.deinit();
    if (response.status != 302 and response.status != 307) try hub.hubStatus(response.status, s.hub_token);
    const commit = response.header("x-repo-commit") orelse return error.MissingRevision;
    if (!std.mem.eql(u8, commit, &s.catalog.revision)) return error.RevisionMismatch;
    var hash: ?[64]u8 = null;
    if (response.header("x-xet-hash")) |h| {
        if (h.len != 64 or !hub.isHex(h)) return error.InvalidMetadata;
        hash = h[0..64].*;
    }
    return .{ .url = url, .xet_hash = hash };
}

fn transient(err: anyerror) bool {
    return switch (err) {
        error.RateLimited, error.ServiceUnavailable, error.TimedOut, error.Unauthorized, error.Forbidden => true,
        else => false,
    };
}

/// Streams bytes [start, start + count) of the file into `sink.write(bytes)`
/// in order, `concurrency` fetches at a time. A transient failure (rate
/// limit, 5xx, timeout, expired token or signed URL) resumes from the last
/// byte delivered with fresh authorization, up to three attempts in a row.
/// The sink sees whole chunks only; bytes are never delivered twice.
pub fn fetch(s: *Session, remote: Remote, start: u64, count: u64, concurrency: u8, sink: anytype) !void {
    if (concurrency == 0 or concurrency > max_concurrency) return error.InvalidConcurrency;
    var position = start;
    const end = start + count;
    var attempt: usize = 0;
    while (position < end) {
        var delivered: u64 = 0;
        const limit: u64 = if (remote.xet_hash != null) span_size else window_size;
        const span = @min(limit, end - position);
        const outcome = if (remote.xet_hash) |hash| xetSpan(s, &hash, position, span, concurrency, sink, &delivered) else directSpan(s, remote.url, position, span, sink, &delivered);
        position += delivered;
        outcome catch |err| {
            if (!transient(err) or attempt == 2) return err;
            if (err == error.Unauthorized or err == error.Forbidden) s.invalidate();
            try s.client.io.sleep(.fromSeconds(@as(i64, 1) << @intCast(attempt)), .awake);
            attempt += 1;
            continue;
        };
        attempt = 0;
    }
}

fn xetSpan(s: *Session, hash: *const [64]u8, start: u64, count: u64, concurrency: u8, sink: anytype, delivered: *u64) !void {
    const credential = try s.credential();
    var temp = std.heap.ArenaAllocator.init(s.client.allocator);
    defer temp.deinit();
    const range = try std.fmt.allocPrint(temp.allocator(), "bytes={d}-{d}", .{ start, start + count - 1 });
    var version: u8 = 2;
    var reconstruction: http.Response = while (true) {
        const url = try std.fmt.allocPrint(temp.allocator(), "{s}/v{d}/reconstructions/{s}", .{ credential.cas_url, version, hash });
        var r = try http.get(&s.client, .{ .url = url, .token = credential.access_token, .range = range });
        if (version == 2 and (r.status == 404 or r.status == 501)) {
            r.deinit();
            version = 1;
            continue;
        }
        break r;
    };
    defer reconstruction.deinit();
    try http.statusError(reconstruction.status);
    var plan = try parsePlan(s.client.allocator, reconstruction.body, version, count);
    defer plan.deinit();
    try execute(s.client.allocator, s.client.io, plan, count, concurrency, UnitFetcher{ .client = &s.client }, sink, delivered);
}

const Range = struct { start: u64, end: u64 };
/// A signed byte range of one xorb holding chunks [chunks.start, chunks.end).
const Unit = struct { url: []const u8, bytes: Range, chunks: Range, last_use: usize };
/// A run of chunks of one xorb, in file order, and the units that hold them
/// in chunk order (each unit covers a prefix of what remains of the run).
const Term = struct { units: []const usize, chunks: Range, unpacked: u64 };
const Plan = struct {
    arena: std.heap.ArenaAllocator,
    skip: u64,
    terms: []const Term,
    units: []const Unit,
    fn deinit(p: *Plan) void {
        p.arena.deinit();
        p.* = undefined;
    }
};

/// Turns a reconstruction response into terms over deduplicated units.
/// `count` is the span the response was requested for; the terms may run past
/// it by less than one chunk (the CAS cuts on chunk boundaries).
fn parsePlan(gpa: A, body: []const u8, version: u8, count: u64) !Plan {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var json = std.heap.ArenaAllocator.init(gpa);
    defer json.deinit();
    const parsed = try std.json.parseFromSlice(Value, json.allocator(), body, .{});
    const root = parsed.value;
    const raw_terms = try array(try field(root, "terms"));
    const fetches = try field(root, if (version == 2) "xorbs" else "fetch_info");
    if (raw_terms.len == 0 or raw_terms.len > 65_536) return error.InvalidReconstruction;
    const skip = try number(try field(root, "offset_into_first_range"));
    if (skip >= try number(try field(raw_terms[0], "unpacked_length"))) return error.InvalidReconstruction;
    var units: std.ArrayList(Unit) = .empty;
    var terms: std.ArrayList(Term) = .empty;
    var total_unpacked: u64 = 0;
    for (raw_terms, 0..) |term, ti| {
        const key = try string(try field(term, "hash"));
        if (key.len != 64 or !hub.isHex(key)) return error.InvalidReconstruction;
        const tr = try chunkRange(try field(term, "range"));
        const unpacked = try number(try field(term, "unpacked_length"));
        if (unpacked == 0 or unpacked > max_xorb) return error.InvalidReconstruction;
        total_unpacked = std.math.add(u64, total_unpacked, unpacked) catch return error.InvalidReconstruction;
        if (total_unpacked > count + skip + 2 * xorb.max_chunk) return error.InvalidReconstruction;
        const entries = try array(try field(fetches, key));
        var term_units: std.ArrayList(usize) = .empty;
        var next = tr.start;
        while (next < tr.end) {
            // The first descriptor holding `next`; its chunk range must then
            // extend the run, so a term walks its units in chunk order.
            var found: ?Unit = null;
            for (entries) |entry| {
                const descriptors = if (version == 2) try array(try field(entry, "ranges")) else &[_]Value{entry};
                for (descriptors) |d| {
                    const chunks = try chunkRange(try field(d, if (version == 2) "chunks" else "range"));
                    if (chunks.start <= next and next < chunks.end) {
                        const bytes = try byteRange(try field(d, if (version == 2) "bytes" else "url_range"));
                        if (bytes.end - bytes.start >= max_unit) return error.InvalidReconstruction;
                        found = .{ .url = try string(try field(entry, "url")), .bytes = bytes, .chunks = chunks, .last_use = ti };
                        break;
                    }
                }
                if (found != null) break;
            }
            const unit = found orelse return error.InvalidReconstruction;
            var index: ?usize = null;
            for (units.items, 0..) |u, i| if (u.bytes.start == unit.bytes.start and u.bytes.end == unit.bytes.end and std.mem.eql(u8, u.url, unit.url)) {
                index = i;
                break;
            };
            if (index == null) {
                _ = try http.validateUrl(unit.url);
                index = units.items.len;
                try units.append(a, .{ .url = try a.dupe(u8, unit.url), .bytes = unit.bytes, .chunks = unit.chunks, .last_use = ti });
            }
            units.items[index.?].last_use = ti;
            try term_units.append(a, index.?);
            next = unit.chunks.end;
        }
        try terms.append(a, .{ .units = try term_units.toOwnedSlice(a), .chunks = tr, .unpacked = unpacked });
    }
    if (total_unpacked < skip + count) return error.InvalidReconstruction;
    return .{ .arena = arena, .skip = skip, .terms = try terms.toOwnedSlice(a), .units = try units.toOwnedSlice(a) };
}

const UnitFetcher = struct {
    client: *std.http.Client,
    /// One piece of a unit into its place in the unit's buffer. Signed URLs
    /// authorize themselves: neither Hub nor CAS token follows. Retried on
    /// rate limits, 5xx, and timeouts; an expired signature (401/403)
    /// surfaces so the caller re-plans with fresh URLs.
    fn fetch(f: UnitFetcher, url: []const u8, range: Range, dest: []u8) anyerror!void {
        var buffer: [64]u8 = undefined;
        const header = try std.fmt.bufPrint(&buffer, "bytes={d}-{d}", .{ range.start, range.end });
        var attempt: usize = 0;
        while (true) : (attempt += 1) {
            return fetchOnce(f, url, header, range, dest) catch |err| {
                if (attempt == 2 or !transient(err) or err == error.Unauthorized or err == error.Forbidden) return err;
                try f.client.io.sleep(.fromSeconds(@as(i64, 1) << @intCast(attempt)), .awake);
                continue;
            };
        }
    }
    fn fetchOnce(f: UnitFetcher, url: []const u8, header: []const u8, range: Range, dest: []u8) !void {
        var data = try http.get(f.client, .{ .url = url, .range = header, .max_bytes = dest.len, .into = dest });
        defer data.deinit();
        try http.statusError(data.status);
        if (data.status != 206) return error.InvalidRangeResponse;
        try checkContentRange(data.header("content-range") orelse return error.InvalidRangeResponse, range);
    }
};

const Piece = struct { unit: usize, offset: usize, len: usize };

/// Fetches the plan's units as pieces through a window of `concurrency`
/// futures in first-use order, each piece landing in its unit's buffer, and
/// decodes each term's chunks into `sink` in file order, counting bytes in
/// `delivered` as they go so a failure can resume exactly there. Every
/// future is awaited or canceled before returning. Buffers live from a unit's
/// first piece to its last use; with 64 MiB units that is about two or three
/// units at a time.
fn execute(gpa: A, io: Io, plan: Plan, count: u64, concurrency: u8, fetcher: anytype, sink: anytype, delivered: *u64) !void {
    // `Io.concurrent` needs a concrete signature; this binds the fetcher's type.
    const Task = struct {
        fn run(f: @TypeOf(fetcher), url: []const u8, range: Range, dest: []u8) anyerror!void {
            return f.fetch(url, range, dest);
        }
    };
    const Slot = struct { buffer: ?[]u8 = null, first_piece: usize = 0, pieces: usize = 0 };
    const slots = try gpa.alloc(Slot, plan.units.len);
    defer gpa.free(slots);
    @memset(slots, .{});
    var pieces: std.ArrayList(Piece) = .empty;
    defer pieces.deinit(gpa);
    for (plan.units, 0..) |unit, ui| {
        const len: usize = @intCast(unit.bytes.end - unit.bytes.start + 1);
        slots[ui].first_piece = pieces.items.len;
        var offset: usize = 0;
        while (offset < len) : (offset += piece_size) {
            try pieces.append(gpa, .{ .unit = ui, .offset = offset, .len = @min(piece_size, len - offset) });
            slots[ui].pieces += 1;
        }
    }
    const futures = try gpa.alloc(?Io.Future(anyerror!void), pieces.items.len);
    defer gpa.free(futures);
    @memset(futures, null);
    defer {
        for (futures) |*slot| if (slot.*) |*future| {
            _ = future.cancel(io) catch {};
            slot.* = null;
        };
        for (slots) |*slot| if (slot.buffer) |buffer| gpa.free(buffer);
    }
    var launched: usize = 0;
    var inflight: usize = 0;
    var awaited: usize = 0;
    var skip = plan.skip;
    var decoded: [xorb.max_chunk]u8 = undefined;
    var scratch: [xorb.max_chunk]u8 = undefined;
    for (plan.terms, 0..) |term, ti| {
        var next = term.chunks.start;
        var term_bytes: u64 = 0;
        for (term.units) |ui| {
            const unit = plan.units[ui];
            // Pieces complete in launch order; everything up to this unit's
            // last piece is awaited, refilling the window as it drains.
            while (awaited < slots[ui].first_piece + slots[ui].pieces) {
                while (inflight < concurrency and launched < pieces.items.len) : ({
                    launched += 1;
                    inflight += 1;
                }) {
                    const piece = pieces.items[launched];
                    const slot = &slots[piece.unit];
                    if (slot.buffer == null) slot.buffer = try gpa.alloc(u8, @intCast(plan.units[piece.unit].bytes.end - plan.units[piece.unit].bytes.start + 1));
                    const range: Range = .{ .start = plan.units[piece.unit].bytes.start + piece.offset, .end = plan.units[piece.unit].bytes.start + piece.offset + piece.len - 1 };
                    futures[launched] = try io.concurrent(Task.run, .{ fetcher, plan.units[piece.unit].url, range, slot.buffer.?[piece.offset..][0..piece.len] });
                }
                var future = futures[awaited] orelse return error.InvalidReconstruction;
                futures[awaited] = null;
                inflight -= 1;
                awaited += 1;
                try future.await(io);
            }
            var cursor: xorb.Cursor = .{ .bytes = slots[ui].buffer orelse return error.InvalidReconstruction };
            var index = unit.chunks.start;
            while (index < unit.chunks.end and next < term.chunks.end) : (index += 1) {
                try io.checkCancel();
                if (index < next) {
                    try xorb.skip(&cursor);
                    continue;
                }
                if (index != next) return error.InvalidReconstruction;
                const chunk = try xorb.decode(&cursor, &decoded, &scratch);
                next += 1;
                term_bytes += chunk.len;
                if (term_bytes > term.unpacked) return error.InvalidReconstruction;
                const drop: usize = @intCast(@min(skip, chunk.len));
                skip -= drop;
                const n: usize = @intCast(@min(chunk.len - drop, count - delivered.*));
                if (n > 0) {
                    try sink.write(chunk[drop..][0..n]);
                    delivered.* += n;
                }
                if (delivered.* == count) return;
            }
            if (unit.last_use == ti) {
                gpa.free(slots[ui].buffer.?);
                slots[ui].buffer = null;
            }
        }
        if (next != term.chunks.end or term_bytes != term.unpacked or skip != 0) return error.InvalidReconstruction;
    }
    return error.InvalidReconstruction; // Terms ended before the span was delivered.
}

fn directSpan(s: *Session, url: []const u8, start: u64, count: u64, sink: anytype, delivered: *u64) !void {
    const bytes = try directRange(s, url, start, @intCast(count));
    defer s.client.allocator.free(bytes);
    try sink.write(bytes);
    delivered.* += bytes.len;
}

fn byteRange(value: Value) !Range {
    const r: Range = .{ .start = try number(try field(value, "start")), .end = try number(try field(value, "end")) };
    if (r.end < r.start or r.end == std.math.maxInt(u64)) return error.InvalidReconstruction;
    return r;
}
fn chunkRange(value: Value) !Range {
    const r = try byteRange(value);
    if (r.start == r.end or r.end > 1_000_000) return error.InvalidReconstruction;
    return r;
}
fn field(v: Value, name: []const u8) !Value {
    if (v != .object) return error.InvalidReconstruction;
    return v.object.get(name) orelse error.InvalidReconstruction;
}
fn string(v: Value) ![]const u8 {
    return if (v == .string) v.string else error.InvalidReconstruction;
}
fn number(v: Value) !u64 {
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else error.InvalidReconstruction;
}
fn array(v: Value) ![]const Value {
    return if (v == .array) v.array.items else error.InvalidReconstruction;
}

fn checkContentRange(header: []const u8, expected: Range) !void {
    if (!std.mem.startsWith(u8, header, "bytes ")) return error.InvalidRangeResponse;
    const dash = std.mem.indexOfScalar(u8, header, '-') orelse return error.InvalidRangeResponse;
    const slash = std.mem.indexOfScalar(u8, header, '/') orelse return error.InvalidRangeResponse;
    if (dash < 6 or slash <= dash) return error.InvalidRangeResponse;
    const start = std.fmt.parseInt(u64, header[6..dash], 10) catch return error.InvalidRangeResponse;
    const end = std.fmt.parseInt(u64, header[dash + 1 .. slash], 10) catch return error.InvalidRangeResponse;
    if (start != expected.start or end != expected.end) return error.InvalidRangeResponse;
    if (!std.mem.eql(u8, header[slash + 1 ..], "*")) {
        const size = std.fmt.parseInt(u64, header[slash + 1 ..], 10) catch return error.InvalidRangeResponse;
        if (size <= end) return error.InvalidRangeResponse;
    }
}

/// One HTTP range of a non-Xet file, following redirects to the LFS store
/// (credentials never cross hosts). Caller owns the result.
pub fn directRange(s: *Session, initial_url: []const u8, start: u64, count: usize) ![]u8 {
    const a = s.client.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const temp = arena.allocator();
    var url = initial_url;
    var credential = s.hub_token;
    const range = try std.fmt.allocPrint(temp, "bytes={d}-{d}", .{ start, start + count - 1 });
    for (0..6) |_| {
        var r = try http.get(&s.client, .{ .url = url, .token = credential, .range = range, .max_bytes = count });
        defer r.deinit();
        if (r.status == 301 or r.status == 302 or r.status == 303 or r.status == 307 or r.status == 308) {
            const location = r.header("location") orelse return error.InvalidRedirect;
            const old = try http.validateUrl(url);
            const next_url = if (std.mem.startsWith(u8, location, "/") and !std.mem.startsWith(u8, location, "//"))
                try std.fmt.allocPrint(temp, "https://{s}{s}", .{ old.host.?.percent_encoded, location })
            else
                try temp.dupe(u8, location);
            const next = try http.validateUrl(next_url);
            // Once stripped, credentials never return, even on a later redirect back.
            if (!std.mem.eql(u8, old.host.?.percent_encoded, next.host.?.percent_encoded) or old.port != next.port) credential = null;
            url = next_url;
            continue;
        }
        try http.statusError(r.status);
        if (r.status != 206) return error.InvalidRangeResponse;
        try checkContentRange(r.header("content-range") orelse return error.InvalidRangeResponse, .{ .start = start, .end = start + count - 1 });
        if (r.body.len != count) return error.InvalidRangeResponse;
        return a.dupe(u8, r.body);
    }
    return error.TooManyRedirects;
}

test "range response rejects shifted and truncated data" {
    try checkContentRange("bytes 10-19/20", .{ .start = 10, .end = 19 });
    try std.testing.expectError(error.InvalidRangeResponse, checkContentRange("bytes 11-19/20", .{ .start = 10, .end = 19 }));
    try std.testing.expectError(error.InvalidRangeResponse, checkContentRange("bytes 10-19/19", .{ .start = 10, .end = 19 }));
}

// Two terms of one xorb sharing a unit that also holds a chunk neither needs
// (chunk 2), plus a second unit. Chunks: "GGUFa" "bcdef" "xyz" "ef" | "gh";
// terms cover chunks 0-1 and 3-4; with 3 leading bytes skipped the file
// bytes are "Fabcdef" + "efgh".
const fixture_v2 =
    \\{"offset_into_first_range":3,"terms":[{"hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","unpacked_length":10,"range":{"start":0,"end":2}},{"hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","unpacked_length":4,"range":{"start":3,"end":5}}],"xorbs":{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa":[{"url":"https://test.invalid/xorb","ranges":[{"chunks":{"start":0,"end":4},"bytes":{"start":0,"end":46}},{"chunks":{"start":4,"end":5},"bytes":{"start":47,"end":56}}]}]}}
;
const unit_a = [_]u8{ 0, 5, 0, 0, 0, 5, 0, 0, 'G', 'G', 'U', 'F', 'a', 0, 5, 0, 0, 0, 5, 0, 0, 'b', 'c', 'd', 'e', 'f', 0, 3, 0, 0, 0, 3, 0, 0, 'x', 'y', 'z', 0, 2, 0, 0, 0, 2, 0, 0, 'e', 'f' };
const unit_b = [_]u8{ 0, 2, 0, 0, 0, 2, 0, 0, 'g', 'h' };
const fixture_xorb = unit_a ++ unit_b;
const FixtureFetcher = struct {
    allocator: A,
    corrupt: bool = false,
    fetches: *usize,
    fn fetch(f: FixtureFetcher, url: []const u8, range: Range, dest: []u8) anyerror!void {
        try std.testing.expectEqualStrings("https://test.invalid/xorb", url);
        try std.testing.expectEqual(dest.len, range.end - range.start + 1);
        f.fetches.* += 1;
        if (f.corrupt) @memset(dest, 'b') else @memcpy(dest, fixture_xorb[@intCast(range.start)..][0..dest.len]);
    }
};
const Collect = struct {
    list: *std.ArrayList(u8),
    allocator: A,
    fn write(c: Collect, bytes: []const u8) !void {
        try c.list.appendSlice(c.allocator, bytes);
    }
};
fn fixtureRoundTrip(a: A) !void {
    var plan = try parsePlan(a, fixture_v2, 2, 11);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 2), plan.units.len);
    try std.testing.expectEqual(@as(usize, 1), plan.units[0].last_use);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var fetches: usize = 0;
    var delivered: u64 = 0;
    try execute(a, std.testing.io, plan, 11, 1, FixtureFetcher{ .allocator = a, .fetches = &fetches }, Collect{ .list = &out, .allocator = a }, &delivered);
    try std.testing.expectEqualStrings("Fabcdefefgh", out.items);
    try std.testing.expectEqual(@as(u64, 11), delivered);
    try std.testing.expectEqual(@as(usize, 2), fetches); // The shared unit was fetched once.
    // A shorter request stops inside the second term without touching unit B.
    out.clearRetainingCapacity();
    fetches = 0;
    delivered = 0;
    var short = try parsePlan(a, fixture_v2, 2, 9);
    defer short.deinit();
    try execute(a, std.testing.io, short, 9, 1, FixtureFetcher{ .allocator = a, .fetches = &fetches }, Collect{ .list = &out, .allocator = a }, &delivered);
    try std.testing.expectEqualStrings("Fabcdefef", out.items);
}

test "plan dedups units and execute skips leading bytes, foreign chunks, and the tail" {
    try fixtureRoundTrip(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, fixtureRoundTrip, .{});
}

test "reconstruction rejects malformed metadata and corrupt xorb bodies" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var fetches: usize = 0;
    var delivered: u64 = 0;
    {
        var plan = try parsePlan(a, fixture_v2, 2, 9);
        defer plan.deinit();
        try std.testing.expectError(error.InvalidChunk, execute(a, std.testing.io, plan, 9, 2, FixtureFetcher{ .allocator = a, .corrupt = true, .fetches = &fetches }, Collect{ .list = &out, .allocator = a }, &delivered));
        try std.testing.expectEqual(@as(u64, 0), delivered);
    }
    const wrong_length = try std.mem.replaceOwned(u8, a, fixture_v2, "\"unpacked_length\":10", "\"unpacked_length\":9");
    defer a.free(wrong_length);
    {
        var plan = try parsePlan(a, wrong_length, 2, 9);
        defer plan.deinit();
        try std.testing.expectError(error.InvalidReconstruction, execute(a, std.testing.io, plan, 9, 2, FixtureFetcher{ .allocator = a, .fetches = &fetches }, Collect{ .list = &out, .allocator = a }, &delivered));
    }
    const empty_range = try std.mem.replaceOwned(u8, a, fixture_v2, "\"start\":0,\"end\":2}", "\"start\":0,\"end\":0}");
    defer a.free(empty_range);
    try std.testing.expectError(error.InvalidReconstruction, parsePlan(a, empty_range, 2, 9));
    const uncovered = try std.mem.replaceOwned(u8, a, fixture_v2, "\"chunks\":{\"start\":4,\"end\":5}", "\"chunks\":{\"start\":5,\"end\":6}");
    defer a.free(uncovered);
    try std.testing.expectError(error.InvalidReconstruction, parsePlan(a, uncovered, 2, 9));
    try std.testing.expectError(error.InvalidReconstruction, parsePlan(a, "{}", 2, 9));
    try std.testing.expectError(error.InvalidReconstruction, parsePlan(a, fixture_v2, 2, 100)); // Terms too short for the span.
}
