//! The server's request log on disk and its requests in flight: a
//! `request.start` and a `request.end` JSON line a request, in one file a
//! UTC day under `<root>/logs/`, and the list `GET /v1/health` reads.
//! Fields are typed and named at compile time, and never a payload
//! (docs/guide/api.md § The request log). Every line is one `write` on an
//! `O_APPEND` descriptor, so servers sharing a file never interleave. A
//! `Record` lives in its connection's arena from `begin` to `finish`; one
//! mutex guards the list, every record's fields, and the file.
const std = @import("std");
const errors = @import("errors.zig");

pub const schema = 1;
pub const max_fields = 24;
/// A text field longer than this is cut, marked with `…`.
pub const max_text = 128;
pub const max_client_id = 128;
pub const max_line = 4096;
pub const part_bytes: u64 = 64 << 20;
pub const keep_days = 14;
pub const keep_bytes: u64 = 256 << 20;

pub const Value = union(enum) {
    int: i64,
    float: f64,
    text: []const u8,
    flag: bool,
};

pub const Field = struct {
    name: []const u8,
    value: Value,
};

/// One request from its head to its end line.
pub const Record = struct {
    id: [36]u8,
    client_id: []const u8 = "",
    client_id_refused: bool = false,
    /// Null for a head that never parsed or a refused connection.
    method: ?std.http.Method = null,
    path: []const u8 = "",
    started: std.Io.Timestamp,
    fields: [max_fields]Field = undefined,
    count: usize = 0,
    /// Text values are copied here so they outlive whatever produced them.
    pool: [max_fields * max_text]u8 = undefined,
    pool_used: usize = 0,
    /// Set when a field did not fit; the end line says so.
    dropped: bool = false,
    prev: ?*Record = null,
    next: ?*Record = null,

    pub fn init(io: std.Io) Record {
        return .{ .id = uuid7(io), .started = std.Io.Clock.awake.now(io) };
    }

    /// Keeps `value` when it is printable ASCII within the limit; otherwise
    /// marks it refused. `value` must outlive the record.
    pub fn setClientId(self: *Record, value: []const u8) void {
        if (validClientId(value)) self.client_id = value else self.client_id_refused = true;
    }

    fn put(self: *Record, name: []const u8, value: Value) void {
        var stored = value;
        if (value == .text) {
            const cut = if (value.text.len > max_text) cutUtf8(value.text, max_text - "…".len) else value.text;
            const extra: []const u8 = if (cut.len < value.text.len) "…" else "";
            const room = self.pool.len - self.pool_used;
            if (cut.len + extra.len > room) {
                self.dropped = true;
                return;
            }
            const into = self.pool[self.pool_used..][0 .. cut.len + extra.len];
            @memcpy(into[0..cut.len], cut);
            @memcpy(into[cut.len..], extra);
            self.pool_used += into.len;
            stored = .{ .text = into };
        }
        for (self.fields[0..self.count]) |*f| if (std.mem.eql(u8, f.name, name)) {
            f.value = stored;
            return;
        };
        if (self.count == max_fields) {
            self.dropped = true;
            return;
        }
        self.fields[self.count] = .{ .name = name, .value = stored };
        self.count += 1;
    }

    fn runningMs(self: *const Record, io: std.Io) f64 {
        return roundTo(nsToMs(self.started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds()), 10);
    }
};

/// What a handler fills for its request's end line. Copyable, so a stream
/// or a session keeps it until it ends; a null record discards (no log, or
/// a test). Names are snake_case with their unit, checked at compile time.
pub const Fields = struct {
    record: ?*Record = null,
    book: ?*Book = null,
    io: std.Io = undefined,

    pub fn int(self: Fields, comptime name: []const u8, value: anytype) void {
        comptime checkName(name);
        const v: i64 = std.math.cast(i64, value) orelse std.math.maxInt(i64);
        self.put(name, .{ .int = v });
    }

    /// `value` rounded to `decimals` places (0 to 3).
    pub fn float(self: Fields, comptime name: []const u8, value: f64, comptime decimals: u2) void {
        comptime checkName(name);
        self.put(name, .{ .float = roundTo(value, comptime std.math.pow(f64, 10, decimals)) });
    }

    /// Nanoseconds as milliseconds, to a tenth.
    pub fn ms(self: Fields, comptime name: []const u8, ns: u64) void {
        comptime checkName(name);
        self.put(name, .{ .float = roundTo(nsToMs(ns), 10) });
    }

    /// Never a payload: a name nuclis resolved, a code, a reason.
    pub fn text(self: Fields, comptime name: []const u8, value: []const u8) void {
        comptime checkName(name);
        self.put(name, .{ .text = value });
    }

    pub fn flag(self: Fields, comptime name: []const u8, value: bool) void {
        comptime checkName(name);
        self.put(name, .{ .flag = value });
    }

    /// The error's code and message.
    pub fn fail(self: Fields, err: errors.ApiError) void {
        self.text("error", err.code);
        self.text("message", err.message);
    }

    fn put(self: Fields, name: []const u8, value: Value) void {
        const record = self.record orelse return;
        const book = self.book orelse return record.put(name, value);
        book.mutex.lockUncancelable(self.io);
        defer book.mutex.unlock(self.io);
        record.put(name, value);
    }
};

/// How a request ended, beside its fields.
pub const End = struct {
    status: std.http.Status,
    bytes_in: usize = 0,
    bytes_out: usize = 0,
};

/// The in-flight list and, when opened, the day's file.
pub const Book = struct {
    mutex: std.Io.Mutex = .init,
    head: ?*Record = null,
    file: ?File = null,
    /// `GET` and `HEAD` of this path get no record: a poller's.
    quiet: []const u8 = "/v1/health",

    /// Opens the log under `<root>/logs`, creating it, and deletes what
    /// retention allows. Owns a copy of the path.
    pub fn open(self: *Book, gpa: std.mem.Allocator, io: std.Io, root: []const u8) !void {
        const dir = try std.fs.path.join(gpa, &.{ root, "logs" });
        errdefer gpa.free(dir);
        try std.Io.Dir.cwd().createDirPath(io, dir);
        prune(io, dir, epochDay(std.Io.Clock.real.now(io)));
        self.file = .{ .dir = dir };
    }

    pub fn deinit(self: *Book, gpa: std.mem.Allocator) void {
        if (self.file) |*f| {
            f.close();
            gpa.free(f.dir);
        }
        self.file = null;
    }

    /// Whether a request gets a record.
    pub fn logs(self: *const Book, method: std.http.Method, path: []const u8) bool {
        return !((method == .GET or method == .HEAD) and std.mem.eql(u8, path, self.quiet));
    }

    /// Links `record` into the in-flight list and writes its start line.
    pub fn begin(self: *Book, io: std.Io, record: *Record) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        record.next = self.head;
        record.prev = null;
        if (self.head) |h| h.prev = record;
        self.head = record;
        if (self.file) |*f| {
            var buffer: [max_line]u8 = undefined;
            f.append(io, startLine(&buffer, record, std.Io.Clock.real.now(io)));
        }
    }

    /// Writes the end line and unlinks `record`; it may then be freed.
    pub fn finish(self: *Book, io: std.Io, record: *Record, end: End) void {
        const duration_ns = record.started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (record.prev) |p| p.next = record.next else self.head = record.next;
        if (record.next) |n| n.prev = record.prev;
        record.prev = null;
        record.next = null;
        if (self.file) |*f| {
            var buffer: [max_line]u8 = undefined;
            f.append(io, endLine(&buffer, record, end, duration_ns, std.Io.Clock.real.now(io)));
        }
    }

    /// The requests in flight, newest first, copied into `arena`.
    pub fn inFlight(self: *Book, io: std.Io, arena: std.mem.Allocator) ![]Flight {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        var out: std.ArrayList(Flight) = .empty;
        var it = self.head;
        while (it) |r| : (it = r.next) {
            const fields = try arena.alloc(Field, r.count);
            for (fields, r.fields[0..r.count]) |*to, from| to.* = .{ .name = from.name, .value = switch (from.value) {
                .text => |t| .{ .text = try arena.dupe(u8, t) },
                else => from.value,
            } };
            try out.append(arena, .{
                .id = try arena.dupe(u8, &r.id),
                .client_id = try arena.dupe(u8, r.client_id),
                .method = if (r.method) |m| @tagName(m) else "",
                .path = try arena.dupe(u8, r.path),
                .running_ms = r.runningMs(io),
                .fields = fields,
            });
        }
        return out.items;
    }
};

pub const Flight = struct {
    id: []const u8,
    client_id: []const u8,
    method: []const u8,
    path: []const u8,
    running_ms: f64,
    fields: []const Field,

    pub fn write(self: Flight, s: *std.json.Stringify) std.Io.Writer.Error!void {
        try s.beginObject();
        try s.objectField("id");
        try s.write(self.id);
        if (self.client_id.len > 0) {
            try s.objectField("client_id");
            try s.write(self.client_id);
        }
        try s.objectField("method");
        try s.write(self.method);
        try s.objectField("path");
        try s.write(self.path);
        try s.objectField("running_ms");
        try s.write(self.running_ms);
        try writeFields(s, self.fields);
        try s.endObject();
    }
};

/// The day's open part: `serve-<date>.jsonl`, then `.1`, `.2`, … past
/// `part_bytes`. A failed open leaves it closed until the next day.
const File = struct {
    dir: []const u8,
    fd: std.c.fd_t = -1,
    day: i64 = -1,
    part: u32 = 0,
    size: u64 = 0,

    fn append(self: *File, io: std.Io, line: []const u8) void {
        const day = epochDay(std.Io.Clock.real.now(io));
        if (day != self.day) {
            self.close();
            self.day = day;
            self.part = 0;
            self.openPart(io);
        } else if (self.fd >= 0 and self.size >= part_bytes) {
            self.close();
            self.part += 1;
            self.openPart(io);
        }
        if (self.fd < 0) return;
        const n = std.c.write(self.fd, line.ptr, line.len);
        if (n > 0) self.size += @intCast(n);
    }

    /// The first part from `self.part` with room, as another server may
    /// have filled some.
    fn openPart(self: *File, io: std.Io) void {
        var name_buffer: [64]u8 = undefined;
        var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
        while (true) : (self.part += 1) {
            const name = partName(&name_buffer, self.day, self.part);
            const path = std.fmt.bufPrintSentinel(&path_buffer, "{s}/{s}", .{ self.dir, name }, 0) catch return;
            const size: u64 = if (std.Io.Dir.cwd().statFile(io, path, .{})) |st| st.size else |_| 0;
            if (size >= part_bytes) continue;
            const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true, .CLOEXEC = true }, @as(c_uint, 0o600));
            if (fd < 0) return;
            self.fd = fd;
            self.size = size;
            return;
        }
    }

    fn close(self: *File) void {
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.fd = -1;
    }
};

fn startLine(buffer: []u8, record: *const Record, now: std.Io.Timestamp) []const u8 {
    var out: std.Io.Writer = .fixed(buffer);
    writeStart(&out, record, now) catch unreachable; // within max_line: the path is at most the head
    return out.buffered();
}

fn writeStart(out: *std.Io.Writer, record: *const Record, now: std.Io.Timestamp) std.Io.Writer.Error!void {
    var s: std.json.Stringify = .{ .writer = out };
    try writeHead(&s, record, "request.start", now);
    if (record.client_id_refused) {
        try s.objectField("client_id_refused");
        try s.write(true);
    }
    if (record.method) |m| {
        try s.objectField("method");
        try s.write(@tagName(m));
        try s.objectField("path");
        try s.write(cutUtf8(record.path, 1024));
    }
    try s.endObject();
    try out.writeByte('\n');
}

/// The end line; fields that do not fit `buffer` are left out and marked.
fn endLine(buffer: []u8, record: *const Record, end: End, duration_ns: i96, now: std.Io.Timestamp) []const u8 {
    var out: std.Io.Writer = .fixed(buffer);
    if (writeEnd(&out, record, end, duration_ns, now, record.fields[0..record.count])) |_| return out.buffered() else |_| {}
    out = .fixed(buffer);
    writeEnd(&out, record, end, duration_ns, now, &.{}) catch unreachable;
    return out.buffered();
}

fn writeEnd(out: *std.Io.Writer, record: *const Record, end: End, duration_ns: i96, now: std.Io.Timestamp, fields: []const Field) std.Io.Writer.Error!void {
    var s: std.json.Stringify = .{ .writer = out };
    try writeHead(&s, record, "request.end", now);
    try s.objectField("status");
    try s.write(@backingInt(end.status));
    try s.objectField("duration_ms");
    try s.write(roundTo(nsToMs(duration_ns), 10));
    try s.objectField("bytes_in");
    try s.write(end.bytes_in);
    try s.objectField("bytes_out");
    try s.write(end.bytes_out);
    try writeFields(&s, fields);
    if (record.dropped or fields.len < record.count) {
        try s.objectField("fields_dropped");
        try s.write(true);
    }
    try s.endObject();
    try out.writeByte('\n');
}

fn writeHead(s: *std.json.Stringify, record: *const Record, event: []const u8, now: std.Io.Timestamp) std.Io.Writer.Error!void {
    try s.beginObject();
    try s.objectField("time");
    var time_buffer: [24]u8 = undefined;
    try s.write(rfc3339(&time_buffer, now));
    try s.objectField("schema");
    try s.write(schema);
    try s.objectField("event");
    try s.write(event);
    try s.objectField("id");
    try s.write(&record.id);
    if (record.client_id.len > 0) {
        try s.objectField("client_id");
        try s.write(record.client_id);
    }
}

fn writeFields(s: *std.json.Stringify, fields: []const Field) std.Io.Writer.Error!void {
    for (fields) |f| {
        try s.objectField(f.name);
        switch (f.value) {
            .int => |v| try s.write(v),
            .float => |v| if (std.math.isFinite(v)) try s.write(v) else try s.write(null),
            .text => |v| try s.write(v),
            .flag => |v| try s.write(v),
        }
    }
}

// ----- names, ids, and time -----

fn checkName(comptime name: []const u8) void {
    if (name.len == 0) @compileError("an empty field name");
    for (name) |c| if (!(std.ascii.isLower(c) or std.ascii.isDigit(c) or c == '_'))
        @compileError("a field name is snake_case: " ++ name);
    for ([_][]const u8{ "time", "schema", "event", "id", "client_id", "client_id_refused", "method", "path", "status", "duration_ms", "bytes_in", "bytes_out", "fields_dropped" }) |taken|
        if (std.mem.eql(u8, name, taken)) @compileError("a field the log writes itself: " ++ name);
}

/// Printable ASCII, 1 to `max_client_id` bytes.
pub fn validClientId(value: []const u8) bool {
    if (value.len == 0 or value.len > max_client_id) return false;
    for (value) |c| if (c < 0x20 or c > 0x7e) return false;
    return true;
}

/// A UUIDv7 (RFC 9562 § 5.7): 48 bits of Unix milliseconds, then random.
pub fn uuid7(io: std.Io) [36]u8 {
    var random: [10]u8 = undefined;
    io.random(&random);
    const unix_ms: u64 = @intCast(@max(0, @divFloor(std.Io.Clock.real.now(io).nanoseconds, std.time.ns_per_ms)));
    return formatUuid7(unix_ms, random);
}

fn formatUuid7(unix_ms: u64, random: [10]u8) [36]u8 {
    var bytes: [16]u8 = undefined;
    std.mem.writeInt(u48, bytes[0..6], @truncate(unix_ms), .big);
    bytes[6] = 0x70 | (random[0] & 0x0f);
    bytes[7] = random[1];
    bytes[8] = 0x80 | (random[2] & 0x3f);
    @memcpy(bytes[9..16], random[3..10]);
    var out: [36]u8 = undefined;
    var i: usize = 0;
    for (bytes, 0..) |b, k| {
        if (k == 4 or k == 6 or k == 8 or k == 10) {
            out[i] = '-';
            i += 1;
        }
        out[i] = std.fmt.hex_charset[b >> 4];
        out[i + 1] = std.fmt.hex_charset[b & 0x0f];
        i += 2;
    }
    return out;
}

/// `2026-10-30T09:12:04.318Z`.
fn rfc3339(buffer: *[24]u8, now: std.Io.Timestamp) []const u8 {
    const unix_ms: u64 = @intCast(@max(0, @divFloor(now.nanoseconds, std.time.ns_per_ms)));
    const seconds: std.time.epoch.EpochSeconds = .{ .secs = unix_ms / 1000 };
    const year_day = seconds.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_seconds = seconds.getDaySeconds();
    return std.fmt.bufPrint(buffer, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}Z", .{
        year_day.year,
        month_day.month.numeric(),
        @as(u32, month_day.day_index) + 1,
        day_seconds.getHoursIntoDay(),
        day_seconds.getMinutesIntoHour(),
        day_seconds.getSecondsIntoMinute(),
        unix_ms % 1000,
    }) catch unreachable;
}

fn epochDay(now: std.Io.Timestamp) i64 {
    return @intCast(@divFloor(now.nanoseconds, std.time.ns_per_day));
}

/// `serve-2026-10-30.jsonl`, or `serve-2026-10-30.<part>.jsonl`.
fn partName(buffer: []u8, day: i64, part: u32) []const u8 {
    const year_day = (std.time.epoch.EpochDay{ .day = @intCast(day) }).calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const date = .{ year_day.year, month_day.month.numeric(), @as(u32, month_day.day_index) + 1 };
    return if (part == 0)
        std.fmt.bufPrint(buffer, "serve-{d:0>4}-{d:0>2}-{d:0>2}.jsonl", date) catch unreachable
    else
        std.fmt.bufPrint(buffer, "serve-{d:0>4}-{d:0>2}-{d:0>2}.{d}.jsonl", date ++ .{part}) catch unreachable;
}

const Part = struct { day: i64, part: u32 };

/// The day and part a log file's name says, or null for any other file.
fn parsePart(name: []const u8) ?Part {
    if (!std.mem.startsWith(u8, name, "serve-") or !std.mem.endsWith(u8, name, ".jsonl")) return null;
    const middle = name["serve-".len .. name.len - ".jsonl".len];
    if (middle.len < 10 or middle[4] != '-' or middle[7] != '-') return null;
    const year = std.fmt.parseInt(i64, middle[0..4], 10) catch return null;
    const month = std.fmt.parseInt(u8, middle[5..7], 10) catch return null;
    const day = std.fmt.parseInt(u8, middle[8..10], 10) catch return null;
    if (month < 1 or month > 12 or day < 1 or day > 31) return null;
    const part: u32 = if (middle.len == 10) 0 else if (middle[10] == '.') std.fmt.parseInt(u32, middle[11..], 10) catch return null else return null;
    return .{ .day = daysFromCivil(year, month, day), .part = part };
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's
/// `days_from_civil`).
fn daysFromCivil(year: i64, month: u8, day: u8) i64 {
    const y = if (month <= 2) year - 1 else year;
    const era = @divFloor(y, 400);
    const yoe = y - era * 400;
    const m: i64 = month;
    const doy = @divFloor(153 * (if (m > 2) m - 3 else m + 9) + 2, 5) + day - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
}

/// Deletes the log files older than `keep_days`, then the oldest until the
/// rest fit `keep_bytes`; today's are kept. Failures are ignored: a log
/// that cannot be pruned still serves.
fn prune(io: std.Io, path: []const u8, today: i64) void {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch return;
    defer dir.close(io);
    const Entry = struct { part: Part, size: u64, name: [64]u8, len: usize };
    var kept: [512]Entry = undefined;
    var n: usize = 0;
    var total: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        const part = parsePart(entry.name) orelse continue;
        if (part.day <= today - keep_days) {
            dir.deleteFile(io, entry.name) catch {};
            continue;
        }
        if (n == kept.len or entry.name.len > 64) continue;
        const size = (dir.statFile(io, entry.name, .{}) catch continue).size;
        kept[n] = .{ .part = part, .size = size, .name = undefined, .len = entry.name.len };
        @memcpy(kept[n].name[0..entry.name.len], entry.name);
        n += 1;
        total += size;
    }
    std.mem.sort(Entry, kept[0..n], {}, struct {
        fn older(_: void, a: Entry, b: Entry) bool {
            return a.part.day < b.part.day or (a.part.day == b.part.day and a.part.part < b.part.part);
        }
    }.older);
    for (kept[0..n]) |e| {
        if (total <= keep_bytes or e.part.day >= today) break;
        dir.deleteFile(io, e.name[0..e.len]) catch continue;
        total -= e.size;
    }
}

fn nsToMs(ns: anytype) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

fn roundTo(value: f64, scale: f64) f64 {
    return @round(value * scale) / scale;
}

/// `text` cut to at most `limit` bytes on a UTF-8 boundary.
fn cutUtf8(text: []const u8, limit: usize) []const u8 {
    if (text.len <= limit) return text;
    var end = limit;
    while (end > 0 and text[end] & 0xc0 == 0x80) end -= 1;
    return text[0..end];
}

// ----- tests -----

const testing = std.testing;

test "a UUIDv7 carries its time, version, and variant" {
    const id = formatUuid7(0x019a34126f1e, .{ 0xfc, 0x2a, 0xdb, 0x4d, 0x0e, 0x5f, 0x6a, 0x7b, 0x8c, 0x9d });
    try testing.expectEqualStrings("019a3412-6f1e-7c2a-9b4d-0e5f6a7b8c9d", &id);
    const live = uuid7(testing.io);
    try testing.expectEqual('7', live[14]);
    try testing.expect(std.mem.indexOfScalar(u8, "89ab", live[19]) != null);
}

test "times, days, and file names" {
    var buffer: [24]u8 = undefined;
    const ns: i96 = (1_793_351_524 * std.time.ms_per_s + 318) * std.time.ns_per_ms;
    try testing.expectEqualStrings("2026-10-30T09:12:04.318Z", rfc3339(&buffer, .{ .nanoseconds = ns }));
    const day = epochDay(.{ .nanoseconds = ns });
    try testing.expectEqual(day, daysFromCivil(2026, 10, 30));
    try testing.expectEqual(@as(i64, 0), daysFromCivil(1970, 1, 1));
    try testing.expectEqual(@as(i64, 11016), daysFromCivil(2000, 2, 29));
    var name: [64]u8 = undefined;
    try testing.expectEqualStrings("serve-2026-10-30.jsonl", partName(&name, day, 0));
    try testing.expectEqualStrings("serve-2026-10-30.3.jsonl", partName(&name, day, 3));
    try testing.expectEqual(Part{ .day = day, .part = 0 }, parsePart("serve-2026-10-30.jsonl").?);
    try testing.expectEqual(Part{ .day = day, .part = 3 }, parsePart("serve-2026-10-30.3.jsonl").?);
    try testing.expectEqual(@as(?Part, null), parsePart("serve-2026-10-30.x.jsonl"));
    try testing.expectEqual(@as(?Part, null), parsePart("other.jsonl"));
}

test "client ids are printable ASCII within the limit" {
    try testing.expect(validClientId("turn-41"));
    try testing.expect(!validClientId(""));
    try testing.expect(!validClientId("a\nb"));
    try testing.expect(!validClientId("é"));
    try testing.expect(!validClientId(&@as([129]u8, @splat('x'))));
}

test "fields: replaced by name, texts cut and copied, overflow marked" {
    var record: Record = .init(testing.io);
    const fields: Fields = .{ .record = &record };
    var source = "qwen3-tts-0.6b-base".*;
    fields.text("model", &source);
    source[0] = 'X';
    fields.int("frames", @as(usize, 56));
    fields.int("frames", @as(usize, 57));
    fields.float("audio_s", 4.4567, 2);
    fields.flag("stream", true);
    fields.text("voice", &@as([300]u8, @splat('v')));
    try testing.expectEqual(@as(usize, 5), record.count);
    try testing.expectEqualStrings("qwen3-tts-0.6b-base", record.fields[0].value.text);
    try testing.expectEqual(@as(i64, 57), record.fields[1].value.int);
    try testing.expectEqual(@as(f64, 4.46), record.fields[2].value.float);
    try testing.expectEqual(max_text, record.fields[4].value.text.len);
    try testing.expect(std.mem.endsWith(u8, record.fields[4].value.text, "…"));
    const none: Fields = .{};
    none.int("frames", 1);
}

test "the two lines" {
    var record: Record = .init(testing.io);
    record.id = "019a3412-6f1e-7c2a-9b4d-0e5f6a7b8c9d".*;
    record.client_id = "turn-41";
    record.method = .POST;
    record.path = "/v1/audio/speech";
    const fields: Fields = .{ .record = &record };
    fields.text("model", "m");
    fields.float("audio_s", 4.46, 2);
    fields.fail(.init(.bad_request, "bad", "a \"quote\""));
    const now: std.Io.Timestamp = .{ .nanoseconds = (1_793_351_524 * std.time.ms_per_s + 318) * std.time.ns_per_ms };
    var buffer: [max_line]u8 = undefined;
    try testing.expectEqualStrings(
        "{\"time\":\"2026-10-30T09:12:04.318Z\",\"schema\":1,\"event\":\"request.start\",\"id\":\"019a3412-6f1e-7c2a-9b4d-0e5f6a7b8c9d\",\"client_id\":\"turn-41\",\"method\":\"POST\",\"path\":\"/v1/audio/speech\"}\n",
        startLine(&buffer, &record, now),
    );
    try testing.expectEqualStrings(
        "{\"time\":\"2026-10-30T09:12:04.318Z\",\"schema\":1,\"event\":\"request.end\",\"id\":\"019a3412-6f1e-7c2a-9b4d-0e5f6a7b8c9d\",\"client_id\":\"turn-41\",\"status\":400,\"duration_ms\":2586.2,\"bytes_in\":12,\"bytes_out\":34,\"model\":\"m\",\"audio_s\":4.46,\"error\":\"bad\",\"message\":\"a \\\"quote\\\"\"}\n",
        endLine(&buffer, &record, .{ .status = .bad_request, .bytes_in = 12, .bytes_out = 34 }, 2_586_234_000, now),
    );
    // Fields past the line's limit are left out, and the line says so.
    var big: Record = record;
    big.count = 0;
    big.pool_used = 0;
    const many: Fields = .{ .record = &big };
    inline for (0..max_fields) |k| many.text(std.fmt.comptimePrint("f{d}", .{k}), &@as([max_text]u8, @splat('"')));
    const line = endLine(&buffer, &big, .{ .status = .ok }, 0, now);
    try testing.expect(std.mem.endsWith(u8, line, "\"fields_dropped\":true}\n"));
}

test "a book: the in-flight list, the day's file, and retention" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = testing.io;
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    try tmp.dir.createDirPath(io, "logs");
    const today = epochDay(std.Io.Clock.real.now(io));
    var name: [64]u8 = undefined;
    var file: [64]u8 = undefined;
    for ([_]i64{ today - 30, today - 14, today - 13 }) |d|
        try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&file, "logs/{s}", .{partName(&name, d, 0)}), .data = "{}\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "logs/notes.txt", .data = "kept" });

    var book: Book = .{};
    try book.open(testing.allocator, io, root);
    defer book.deinit(testing.allocator);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, try std.fmt.bufPrint(&file, "logs/{s}", .{partName(&name, today - 30, 0)}), .{}));
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, try std.fmt.bufPrint(&file, "logs/{s}", .{partName(&name, today - 14, 0)}), .{}));
    _ = try tmp.dir.statFile(io, try std.fmt.bufPrint(&file, "logs/{s}", .{partName(&name, today - 13, 0)}), .{});
    _ = try tmp.dir.statFile(io, "logs/notes.txt", .{});

    try testing.expect(!book.logs(.GET, "/v1/health"));
    try testing.expect(book.logs(.POST, "/v1/health"));
    var a: Record = .init(io);
    a.method = .POST;
    a.path = "/v1/chat/completions";
    var b: Record = .init(io);
    b.method = .GET;
    b.path = "/v1/models";
    book.begin(io, &a);
    book.begin(io, &b);
    (Fields{ .record = &a, .book = &book, .io = io }).text("model", "gemma");
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const flying = try book.inFlight(io, arena_state.allocator());
    try testing.expectEqual(@as(usize, 2), flying.len);
    try testing.expectEqualStrings("/v1/models", flying[1 - 1].path);
    try testing.expectEqualStrings("gemma", flying[1].fields[0].value.text);
    book.finish(io, &b, .{ .status = .ok });
    book.finish(io, &a, .{ .status = .ok, .bytes_out = 5 });
    try testing.expectEqual(@as(?*Record, null), book.head);

    const path = try std.fmt.bufPrint(&file, "logs/{s}", .{partName(&name, today, 0)});
    const written = try tmp.dir.readFileAlloc(io, path, testing.allocator, .limited(1 << 16));
    defer testing.allocator.free(written);
    try testing.expectEqual(@as(usize, 4), std.mem.count(u8, written, "\n"));
    var expected: [80]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, written, try std.fmt.bufPrint(&expected, "\"event\":\"request.end\",\"id\":\"{s}\"", .{&a.id})) != null);
    const st = try tmp.dir.statFile(io, path, .{});
    try testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.permissions.toMode())) & 0o777);
}
