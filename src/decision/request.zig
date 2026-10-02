//! The decision request's wire format: the Jev-shaped JSON (`questions`, and
//! `state` or `states`) into the questions and states `inference.decide`
//! answers. Shared by `nuclis decide --request` and the API; knows no HTTP.
//! A `{"file": path}` state is read only when the caller allows files.
//! docs/reference/laya.md § nuclis decide.
const std = @import("std");
const inference = @import("inference");
const config = @import("../config.zig");

const profile = inference.profiles.laya;
const decide = inference.decide;

/// One state as the request named it, ready for the decider.
pub const Labeled = struct { label: []const u8, state: decide.State };

pub const Request = struct {
    ids: []const []const u8,
    questions: []const profile.Question,
    states: []const Labeled,
};

/// Whether a `{"file": path}` state may be read: the CLI's caller owns the
/// files, a network client does not.
pub const Files = enum { allowed, refused };

pub const Error = error{ InvalidRequest, FileStateRefused, RequestTooLarge, StateTooLarge, TooManyOptions, MissingQuestions };

/// Reads a whole file, or standard input for `-`, up to `limit` bytes.
pub fn readBounded(arena: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize, diag: *config.Diagnostic) ![]u8 {
    if (std.mem.eql(u8, path, "-")) {
        var buffer: [4096]u8 = undefined;
        var reader = std.Io.File.stdin().readerStreaming(io, &buffer);
        return reader.interface.allocRemaining(arena, .limited(limit)) catch |err| {
            diag.set("standard input: {s} (at most {d} bytes)", .{ @errorName(err), limit });
            return err;
        };
    }
    return std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(limit)) catch |err| {
        diag.set("{s}: {s}", .{ path, @errorName(err) });
        return err;
    };
}

pub fn parseJson(arena: std.mem.Allocator, bytes: []const u8, what: []const u8, diag: *config.Diagnostic) !std.json.Value {
    return std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{}) catch |err| {
        diag.set("{s} is not valid JSON ({s})", .{ what, @errorName(err) });
        return error.InvalidRequest;
    };
}

/// A state value of a request: text, `{"file": path}` (read, as text, when
/// `files` allows), or any other JSON, rendered as the package renders it
/// (a list keeps its tail).
pub fn stateFromJson(arena: std.mem.Allocator, io: std.Io, value: std.json.Value, index: usize, files: Files, diag: *config.Diagnostic) !Labeled {
    switch (value) {
        .string => |s| return .{ .label = try std.fmt.allocPrint(arena, "state[{d}]", .{index}), .state = .{ .text = s } },
        .object => |o| if (o.count() == 1) if (o.get("file")) |f| if (f == .string) {
            if (files == .refused) {
                diag.set("state[{d}] names a file; this server reads no files, send the text", .{index});
                return error.FileStateRefused;
            }
            return .{ .label = f.string, .state = .{ .text = try readBounded(arena, io, f.string, decide.max_state_bytes, diag) } };
        },
        else => {},
    }
    return .{
        .label = try std.fmt.allocPrint(arena, "state[{d}]", .{index}),
        .state = .{ .text = try profile.pythonJson(arena, value), .truncate = if (value == .array) .head else .tail },
    };
}

/// Appends the questions of a `"questions"` object (id → definition).
pub fn questionsFromJson(arena: std.mem.Allocator, value: std.json.Value, what: []const u8, ids: *std.ArrayList([]const u8), questions: *std.ArrayList(profile.Question), diag: *config.Diagnostic) !void {
    const map = switch (value) {
        .object => |o| o,
        else => {
            diag.set("{s}: \"questions\" must be an object of id -> definition", .{what});
            return error.InvalidRequest;
        },
    };
    var it = map.iterator();
    while (it.next()) |entry| {
        var why: profile.Diagnostic = .{};
        const q = profile.parseQuestion(arena, entry.key_ptr.*, entry.value_ptr.*, &why) catch |err| {
            diag.set("{s}", .{why.message()});
            return err;
        };
        try ids.append(arena, entry.key_ptr.*);
        try questions.append(arena, q);
    }
}

/// A whole request object: `questions` and one of `state` or `states`.
/// Other fields (`model`) are the caller's to read; `what` names the source
/// in diagnostics.
pub fn fromJson(arena: std.mem.Allocator, io: std.Io, root: std.json.Value, what: []const u8, files: Files, diag: *config.Diagnostic) !Request {
    const body = switch (root) {
        .object => |b| b,
        else => {
            diag.set("{s}: a request is an object with \"questions\" and \"state\" or \"states\"", .{what});
            return error.InvalidRequest;
        },
    };
    var ids: std.ArrayList([]const u8) = .empty;
    var questions: std.ArrayList(profile.Question) = .empty;
    var states: std.ArrayList(Labeled) = .empty;
    try questionsFromJson(arena, body.get("questions") orelse .null, what, &ids, &questions, diag);
    const one = body.get("state");
    const many = body.get("states");
    if ((one == null) == (many == null)) {
        diag.set("{s}: give \"state\" or \"states\", one of them", .{what});
        return error.InvalidRequest;
    }
    if (one) |s| try states.append(arena, try stateFromJson(arena, io, s, 0, files, diag));
    if (many) |m| {
        if (m != .array or m.array.items.len == 0) {
            diag.set("{s}: \"states\" must be a non-empty list", .{what});
            return error.InvalidRequest;
        }
        // Checked before reading any file or rendering any state.
        if (m.array.items.len > decide.max_states) {
            diag.set("at most {d} states and {d} questions per call; this request has {d} and {d}", .{ decide.max_states, decide.max_questions, m.array.items.len, questions.items.len });
            return error.RequestTooLarge;
        }
        for (m.array.items, 0..) |s, i| try states.append(arena, try stateFromJson(arena, io, s, i, files, diag));
    }
    return .{ .ids = ids.items, .questions = questions.items, .states = states.items };
}

/// The host's request limits (`inference.decide`), with the counts named.
pub fn checkLimits(request: Request, diag: *config.Diagnostic) Error!void {
    if (request.questions.len == 0) {
        diag.set("the request has no questions", .{});
        return error.MissingQuestions;
    }
    if (request.states.len > decide.max_states or request.questions.len > decide.max_questions) {
        diag.set("at most {d} states and {d} questions per call; this request has {d} and {d}", .{ decide.max_states, decide.max_questions, request.states.len, request.questions.len });
        return error.RequestTooLarge;
    }
    for (request.ids, request.questions) |id, q| if (q.texts.len > decide.max_options) {
        diag.set("question {s}: {d} options, at most {d}", .{ id, q.texts.len, decide.max_options });
        return error.TooManyOptions;
    };
    for (request.states) |s| if (s.state.text.len > decide.max_state_bytes) {
        diag.set("{s}: {d} bytes, at most {d}", .{ s.label, s.state.text.len, decide.max_state_bytes });
        return error.StateTooLarge;
    };
}

/// The states as the decider takes them; `arena` owns the slice.
pub fn decisionStates(arena: std.mem.Allocator, request: Request) ![]decide.State {
    const states = try arena.alloc(decide.State, request.states.len);
    for (states, request.states) |*s, l| s.* = l.state;
    return states;
}

test "a request object: one state or many, files only when allowed" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    var diag: config.Diagnostic = .{};
    const body =
        \\{"model":"laya","questions":{"angry":{"type":"noul","instructions":"Angry?"}},
        \\ "states":["text",{"a":1},[{"role":"user","content":"hi"}]]}
    ;
    const request = try fromJson(arena, io, try parseJson(arena, body, "body", &diag), "body", .refused, &diag);
    try std.testing.expectEqual(@as(usize, 3), request.states.len);
    try std.testing.expectEqualStrings("angry", request.ids[0]);
    try std.testing.expectEqualStrings("state[1]", request.states[1].label);
    try std.testing.expectEqual(profile.Truncate.head, request.states[2].state.truncate);
    try checkLimits(request, &diag);

    const file = "{\"questions\":{\"a\":{\"type\":\"noul\",\"instructions\":\"x\"}},\"state\":{\"file\":\"/etc/hosts\"}}";
    try std.testing.expectError(error.FileStateRefused, fromJson(arena, io, try parseJson(arena, file, "body", &diag), "body", .refused, &diag));
    const cases = [_][]const u8{
        "[]",
        "{\"questions\":{\"a\":{\"type\":\"noul\",\"instructions\":\"x\"}}}",
        "{\"questions\":{\"a\":{\"type\":\"noul\",\"instructions\":\"x\"}},\"state\":\"s\",\"states\":[\"t\"]}",
        "{\"questions\":{\"a\":{\"type\":\"noul\",\"instructions\":\"x\"}},\"states\":[]}",
        "{\"questions\":[],\"state\":\"s\"}",
    };
    for (cases) |case| try std.testing.expectError(error.InvalidRequest, fromJson(arena, io, try parseJson(arena, case, "body", &diag), "body", .refused, &diag));
    try std.testing.expectError(error.InvalidRequest, parseJson(arena, "{", "body", &diag));
    const empty = try fromJson(arena, io, try parseJson(arena, "{\"questions\":{},\"state\":\"s\"}", "body", &diag), "body", .refused, &diag);
    try std.testing.expectError(error.MissingQuestions, checkLimits(empty, &diag));
}
