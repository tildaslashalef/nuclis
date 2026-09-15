//! `hf-downloader`: standalone exercise of the internal API, including its
//! progress sink, which it renders on stderr. Not wired into nuclis.
const std = @import("std");
const hf = @import("huggingface");

pub fn main(init: std.process.Init) void {
    run(init) catch |err| {
        std.log.err("{s}", .{@errorName(err)});
        switch (err) {
            error.TokenRequired => std.log.err("this repository needs authentication: set HF_TOKEN to a Hugging Face user access token (https://huggingface.co/settings/tokens)", .{}),
            error.TokenRejected => std.log.err("HF_TOKEN was refused by the Hub: check the token value and that it has not expired or been revoked", .{}),
            error.AccessDenied => std.log.err("access denied: if the repository is gated, accept its terms on huggingface.co with the account HF_TOKEN belongs to", .{}),
            error.ExistingFileMismatch => std.log.err("a different file already exists at the destination path; move it away to download this revision", .{}),
            else => {},
        }
        std.process.exit(1);
    };
}

const usage =
    \\hf-downloader REPO [--file EXACT.gguf] [--revision REF] [--local-dir DIR]
    \\  --concurrency N          signed xorb ranges in flight (1..16, default 8)
    \\  --list                   list GGUF files without downloading
    \\  --range START COUNT      print SHA-256 of a native download range (max 8 MiB)
    \\  --quiet                  no progress on stderr
    \\Repo-only downloads select a sole GGUF choice; otherwise list choices.
    \\Split filenames download the complete shard set. Xet is the default.
    \\Files land in <models dir>/<owner>/<repo>/<file>, where the models
    \\directory is: explicit --local-dir > NUCLIS_HOME/models > HOME/.nuclis/models.
    \\
;

fn run(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var buffer: [4096]u8 = undefined;
    var output = std.Io.File.stdout().writer(init.io, &buffer);
    const out = &output.interface;
    defer out.flush() catch {};
    if (args.len < 2 or std.mem.eql(u8, args[1], "--help")) {
        try out.writeAll(usage);
        return;
    }
    var request: hf.Request = .{ .repo_id = args[1] };
    var list = false;
    var quiet = false;
    var concurrency: u8 = 8;
    var range: ?struct { start: u64, count: usize } = null;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--list")) {
            list = true;
            continue;
        }
        if (std.mem.eql(u8, flag, "--quiet")) {
            quiet = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingArgument;
        i += 1;
        if (std.mem.eql(u8, flag, "--file")) request.filename = args[i] else if (std.mem.eql(u8, flag, "--revision")) request.revision = args[i] else if (std.mem.eql(u8, flag, "--local-dir")) request.local_dir = args[i] else if (std.mem.eql(u8, flag, "--concurrency")) concurrency = try std.fmt.parseInt(u8, args[i], 10) else if (std.mem.eql(u8, flag, "--range")) {
            const start = try std.fmt.parseInt(u64, args[i], 10);
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            range = .{ .start = start, .count = try std.fmt.parseInt(usize, args[i], 10) };
        } else return error.UnknownArgument;
    }
    if (list and range != null) return error.ConflictingArguments;
    var client = hf.Client.init(init.gpa, init.io, init.environ_map);
    client.concurrency = concurrency;
    var display: Display = .{ .io = init.io, .started = .now(init.io, .awake) };
    if (!quiet) client.progress = .{ .context = &display, .update = Display.update };
    if (list) {
        var catalog = try client.list(request);
        defer catalog.deinit();
        try out.print("revision {s}\n", .{catalog.revision});
        for (catalog.files) |f| try out.print("{s}\t{d}\n", .{ f.name, f.size });
    } else if (range) |r| {
        const bytes = try client.readRange(request, r.start, r.count);
        defer init.gpa.free(bytes);
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
        try out.print("range {d} {d} sha256 {x}\n", .{ r.start, bytes.len, &hash });
        if (r.start == 0 and bytes.len >= 4) try out.print("magic {s}\n", .{bytes[0..4]});
    } else {
        var result = try client.download(request);
        defer result.deinit();
        try out.print("revision {s}\n", .{result.revision});
        switch (result.outcome) {
            .selection_required => |files| {
                try out.writeAll("Select an exact filename with --file:\n");
                for (files) |f| try out.print("{s}\t{d}\n", .{ f.name, f.size });
            },
            .downloaded => |files| for (files) |f| try out.print("{s}\t{d}\t{s}\n", .{ f.path, f.size, @tagName(f.transport) }),
        }
    }
}

/// Renders progress events on stderr: one line per phase change, and during
/// a download a line rewritten in place at most every 200 ms with the file's
/// percentage, bytes, throughput since the download started, and the
/// estimated remaining time. Everything it needs arrives in the event; the
/// clock is the only state it keeps. Write failures are ignored: progress is
/// advisory and must never abort a transfer.
const Display = struct {
    io: std.Io,
    started: std.Io.Timestamp,
    last_line: ?std.Io.Timestamp = null,
    last_phase: ?@FieldType(hf.ProgressEvent, "phase") = null,
    download_started: ?std.Io.Timestamp = null,
    download_base: u64 = 0,
    open_line: bool = false,

    fn update(context: ?*anyopaque, event: hf.ProgressEvent) error{Canceled}!void {
        const d: *Display = @ptrCast(@alignCast(context.?));
        d.render(event) catch {};
    }

    fn render(d: *Display, event: hf.ProgressEvent) !void {
        var buffer: [512]u8 = undefined;
        // Streaming, not positional: a positional writer restarts at offset
        // zero on every call and overwrites a redirected log.
        var stderr = std.Io.File.stderr().writerStreaming(d.io, &buffer);
        const w = &stderr.interface;
        defer w.flush() catch {};
        const now: std.Io.Timestamp = .now(d.io, .awake);
        const name = event.filename orelse "";
        const phase_changed = d.last_phase == null or d.last_phase.? != event.phase;
        d.last_phase = event.phase;
        switch (event.phase) {
            .resolving => try w.writeAll("resolving revision and files\n"),
            .selection_required => try w.writeAll("several GGUF files: choose one with --file\n"),
            .verifying => if (event.reused) {
                try d.endLine(w);
                try w.print("{s}: existing file verified, reused\n", .{name});
            } else if (event.file_completed == 0) {
                try w.print("{s}: checking for an existing file\n", .{name});
            } else {
                try d.endLine(w);
                try w.print("{s}: verifying SHA-256 and publishing\n", .{name});
            },
            .downloading => {
                if (phase_changed or event.file_completed == 0) {
                    d.download_started = now;
                    d.download_base = event.file_completed;
                    if (event.file_count > 1) try w.print("file {d} of {d}\n", .{ event.file_index + 1, event.file_count });
                }
                if (d.last_line) |last| if (event.file_completed < event.file_total and last.durationTo(now).toMilliseconds() < 200) return;
                d.last_line = now;
                const elapsed_ms = d.download_started.?.durationTo(now).toMilliseconds();
                const moved = event.file_completed - d.download_base;
                const rate_bps = if (elapsed_ms > 0) @as(f64, @floatFromInt(moved)) * 1000 / @as(f64, @floatFromInt(elapsed_ms)) else 0;
                const remaining = event.file_total - event.file_completed;
                const eta_s: u64 = if (rate_bps > 0) @intFromFloat(@as(f64, @floatFromInt(remaining)) / rate_bps) else 0;
                const percent = if (event.file_total == 0) 0 else 100 * event.file_completed / event.file_total;
                try w.print("\r{s}: {d:>3}% {d:.1}/{d:.1} MB {d:.1} MB/s eta {d}s   ", .{ name, percent, mb(event.file_completed), mb(event.file_total), rate_bps / 1e6, eta_s });
                d.open_line = true;
            },
            .complete => {
                try d.endLine(w);
                const total_ms = d.started.durationTo(now).toMilliseconds();
                try w.print("done: {d:.1} MB in {d:.1} s ({d:.1} MB/s overall, verified reuse included)\n", .{ mb(event.completed_bytes), @as(f64, @floatFromInt(total_ms)) / 1000, if (total_ms > 0) mb(event.completed_bytes) * 1000 / @as(f64, @floatFromInt(total_ms)) else 0 });
            },
        }
    }

    fn endLine(d: *Display, w: *std.Io.Writer) !void {
        if (d.open_line) try w.writeAll("\n");
        d.open_line = false;
    }

    fn mb(bytes: u64) f64 {
        return @as(f64, @floatFromInt(bytes)) / 1e6;
    }
};
