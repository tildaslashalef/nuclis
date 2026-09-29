//! `nuclis decide`: typed questions about states through a Laya checkpoint
//! (`inference.decide`). Three input tiers build the same request: a
//! Jev-shaped request file, a questions file with states from flags, or one
//! question inline. One state renders per question; several render ranked
//! by the first question, the filter shape (one question over many states).
//! `--json` writes one Jev response per state. docs/reference/laya.md.
const std = @import("std");
const inference = @import("inference");
const style = @import("tui/style.zig");
const config = @import("config.zig");
const model = @import("model.zig");
const paths = @import("paths.zig");

const profile = inference.profiles.laya;
const Decider = inference.decide.Decider;

pub const schema_version = 1;

const StateSource = union(enum) { text: []const u8, file: []const u8 };

const Inline = struct {
    kind: profile.Kind,
    text: []const u8,
    id: ?[]const u8 = null,
    options: std.ArrayList([]const u8) = .empty,
};

pub const Options = struct {
    request: ?[]const u8 = null,
    questions: ?[]const u8 = null,
    states: std.ArrayList(StateSource) = .empty,
    inline_questions: std.ArrayList(Inline) = .empty,
    model: ?[]const u8 = null,
    json: bool = false,
    explain: bool = false,
    uncalibrated: bool = false,
    truncate: ?profile.Truncate = null,
};

/// Parses the words after `decide`; `arena` owns the lists.
pub fn parseArgs(arena: std.mem.Allocator, args: []const []const u8, diag: *config.Diagnostic) !Options {
    var o: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--json")) {
            if (o.json) return error.DuplicateOption;
            o.json = true;
            continue;
        }
        if (std.mem.eql(u8, flag, "--explain")) {
            if (o.explain) return error.DuplicateOption;
            o.explain = true;
            continue;
        }
        if (std.mem.eql(u8, flag, "--uncalibrated")) {
            if (o.uncalibrated) return error.DuplicateOption;
            o.uncalibrated = true;
            continue;
        }
        const takes_value = for ([_][]const u8{ "--request", "--questions", "--state", "--state-file", "--choice", "--score", "--noul", "--option", "--level", "--id", "--model", "--truncate" }) |known| {
            if (std.mem.eql(u8, flag, known)) break true;
        } else false;
        if (!takes_value) {
            diag.set("{s} is not an option of decide (`nuclis decide --help`)", .{flag});
            return error.UnknownOption;
        }
        i += 1;
        if (i == args.len) {
            diag.set("{s} needs a value", .{flag});
            return error.MissingOptionValue;
        }
        const value = args[i];
        if (std.mem.eql(u8, flag, "--request")) {
            if (o.request != null) return error.DuplicateOption;
            o.request = value;
        } else if (std.mem.eql(u8, flag, "--questions")) {
            if (o.questions != null) return error.DuplicateOption;
            o.questions = value;
        } else if (std.mem.eql(u8, flag, "--state")) {
            try o.states.append(arena, .{ .text = value });
        } else if (std.mem.eql(u8, flag, "--state-file")) {
            try o.states.append(arena, .{ .file = value });
        } else if (std.mem.eql(u8, flag, "--model")) {
            if (o.model != null) return error.DuplicateOption;
            o.model = value;
        } else if (std.mem.eql(u8, flag, "--truncate")) {
            if (o.truncate != null) return error.DuplicateOption;
            o.truncate = std.meta.stringToEnum(profile.Truncate, value) orelse {
                diag.set("--truncate takes head or tail (the end of the state that is cut), not {s}", .{value});
                return error.InvalidOptionValue;
            };
        } else if (std.mem.eql(u8, flag, "--choice") or std.mem.eql(u8, flag, "--score") or std.mem.eql(u8, flag, "--noul")) {
            try o.inline_questions.append(arena, .{ .kind = std.meta.stringToEnum(profile.Kind, flag[2..]).?, .text = value });
        } else {
            // --option, --level, and --id attach to the latest inline question.
            if (o.inline_questions.items.len == 0) {
                diag.set("{s} follows the question it belongs to (--choice, --score, or --noul)", .{flag});
                return error.MisplacedOption;
            }
            const q = &o.inline_questions.items[o.inline_questions.items.len - 1];
            if (std.mem.eql(u8, flag, "--id")) {
                if (q.id != null) return error.DuplicateOption;
                q.id = value;
            } else {
                const wanted: profile.Kind = if (std.mem.eql(u8, flag, "--option")) .choice else .score;
                if (q.kind != wanted) {
                    diag.set("{s} belongs to a {s} question; this one is {s}", .{ flag, @tagName(wanted), @tagName(q.kind) });
                    return error.MisplacedOption;
                }
                try q.options.append(arena, value);
            }
        }
    }
    const sources = @as(u8, @intFromBool(o.request != null)) + @intFromBool(o.questions != null) + @intFromBool(o.inline_questions.items.len > 0);
    if (sources == 0) {
        diag.set("give the questions: --request <file>, --questions <file>, or --choice/--score/--noul inline", .{});
        return error.MissingQuestions;
    }
    if (sources > 1) {
        diag.set("--request, --questions, and inline questions are three ways to give the questions; use one", .{});
        return error.ConflictingOptions;
    }
    if (o.request != null and o.states.items.len > 0) {
        diag.set("a request carries its own state or states; --state and --state-file go with --questions or inline questions", .{});
        return error.ConflictingOptions;
    }
    if (o.request == null and o.states.items.len == 0) {
        diag.set("give at least one --state <text> or --state-file <path>", .{});
        return error.MissingState;
    }
    return o;
}

/// One state as the request named it, ready for the decider.
const Labeled = struct { label: []const u8, state: inference.decide.State };

const Request = struct {
    ids: []const []const u8,
    questions: []const profile.Question,
    states: []const Labeled,
};

fn readBounded(arena: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize, diag: *config.Diagnostic) ![]u8 {
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

fn parseJson(arena: std.mem.Allocator, bytes: []const u8, what: []const u8, diag: *config.Diagnostic) !std.json.Value {
    return std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{}) catch |err| {
        diag.set("{s} is not valid JSON ({s})", .{ what, @errorName(err) });
        return error.InvalidRequest;
    };
}

/// A state value of a request: text, `{"file": path}` (read, as text), or
/// any other JSON, rendered as the package renders it (a list keeps its tail).
fn stateFromJson(arena: std.mem.Allocator, io: std.Io, value: std.json.Value, index: usize, diag: *config.Diagnostic) !Labeled {
    switch (value) {
        .string => |s| return .{ .label = try std.fmt.allocPrint(arena, "state[{d}]", .{index}), .state = .{ .text = s } },
        .object => |o| if (o.count() == 1) if (o.get("file")) |f| if (f == .string) {
            return .{ .label = f.string, .state = .{ .text = try readBounded(arena, io, f.string, inference.decide.max_state_bytes, diag) } };
        },
        else => {},
    }
    return .{
        .label = try std.fmt.allocPrint(arena, "state[{d}]", .{index}),
        .state = .{ .text = try profile.pythonJson(arena, value), .truncate = if (value == .array) .head else .tail },
    };
}

fn questionsFromJson(arena: std.mem.Allocator, value: std.json.Value, what: []const u8, ids: *std.ArrayList([]const u8), questions: *std.ArrayList(profile.Question), diag: *config.Diagnostic) !void {
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

fn buildRequest(arena: std.mem.Allocator, io: std.Io, o: Options, diag: *config.Diagnostic) !Request {
    var ids: std.ArrayList([]const u8) = .empty;
    var questions: std.ArrayList(profile.Question) = .empty;
    var states: std.ArrayList(Labeled) = .empty;
    if (o.request) |request_path| {
        const path = if (std.mem.eql(u8, request_path, "-")) "standard input" else request_path;
        const root = try parseJson(arena, try readBounded(arena, io, request_path, 64 * 1024 * 1024, diag), path, diag);
        const body = switch (root) {
            .object => |b| b,
            else => {
                diag.set("{s}: a request is an object with \"questions\" and \"state\" or \"states\"", .{path});
                return error.InvalidRequest;
            },
        };
        try questionsFromJson(arena, body.get("questions") orelse .null, path, &ids, &questions, diag);
        const one = body.get("state");
        const many = body.get("states");
        if ((one == null) == (many == null)) {
            diag.set("{s}: give \"state\" or \"states\", one of them", .{path});
            return error.InvalidRequest;
        }
        if (one) |s| try states.append(arena, try stateFromJson(arena, io, s, 0, diag));
        if (many) |m| {
            if (m != .array or m.array.items.len == 0) {
                diag.set("{s}: \"states\" must be a non-empty list", .{path});
                return error.InvalidRequest;
            }
            for (m.array.items, 0..) |s, i| try states.append(arena, try stateFromJson(arena, io, s, i, diag));
        }
    } else {
        if (o.questions) |path| {
            const root = try parseJson(arena, try readBounded(arena, io, path, 1024 * 1024, diag), path, diag);
            // A whole request's "questions" field, or the map itself.
            const map = if (root == .object and root.object.get("questions") != null) root.object.get("questions").? else root;
            try questionsFromJson(arena, map, path, &ids, &questions, diag);
        }
        for (o.inline_questions.items) |q| {
            var definition: std.json.ObjectMap = .empty;
            try definition.put(arena, "type", .{ .string = @tagName(q.kind) });
            try definition.put(arena, "instructions", .{ .string = q.text });
            switch (q.kind) {
                .choice => {
                    var criteria: std.json.ObjectMap = .empty;
                    for (q.options.items) |option| {
                        const eq = std.mem.indexOfScalar(u8, option, '=');
                        try criteria.put(arena, if (eq) |e| option[0..e] else option, if (eq) |e| .{ .string = option[e + 1 ..] } else .null);
                    }
                    try definition.put(arena, "criteria", .{ .object = criteria });
                },
                .score => {
                    var levels: std.json.Array = .init(arena);
                    for (q.options.items) |level| try levels.append(.{ .string = level });
                    try definition.put(arena, "criteria", .{ .array = levels });
                },
                .noul => {},
            }
            const id = q.id orelse @tagName(q.kind);
            for (ids.items) |seen| if (std.mem.eql(u8, seen, id)) {
                diag.set("two questions are named {s}; name them with --id", .{id});
                return error.DuplicateQuestion;
            };
            var why: profile.Diagnostic = .{};
            const parsed = profile.parseQuestion(arena, id, .{ .object = definition }, &why) catch |err| {
                diag.set("{s}", .{why.message()});
                return err;
            };
            try ids.append(arena, id);
            try questions.append(arena, parsed);
        }
        for (o.states.items, 0..) |source, i| try states.append(arena, switch (source) {
            .text => |t| .{ .label = try std.fmt.allocPrint(arena, "state[{d}]", .{i}), .state = .{ .text = t } },
            .file => |f| .{ .label = f, .state = .{ .text = try readBounded(arena, io, f, inference.decide.max_state_bytes, diag) } },
        });
    }
    if (questions.items.len == 0) {
        diag.set("the request has no questions", .{});
        return error.MissingQuestions;
    }
    if (o.truncate) |t| for (states.items) |*s| {
        s.state.truncate = t;
    };
    return .{ .ids = ids.items, .questions = questions.items, .states = states.items };
}

/// The checkpoint directory `--model` or `decide.model` names: an absolute
/// or `./` path as given, else a path under `<root>/models`.
pub fn resolveDirectory(arena: std.mem.Allocator, root: ?[]const u8, name: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(name) or std.mem.startsWith(u8, name, "./") or std.mem.startsWith(u8, name, "../")) return name;
    const dir = root orelse return error.MissingHome;
    return std.fs.path.join(arena, &.{ dir, "models", name });
}

pub const Identity = struct { name: []const u8, repo: ?[]const u8 = null, revision: ?[]const u8 = null };

pub fn run(gpa: std.mem.Allocator, io: std.Io, directory: []const u8, identity: Identity, o: Options, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const request = try buildRequest(arena, io, o, diag);
    if (request.states.len > inference.decide.max_states or request.questions.len > inference.decide.max_questions) {
        diag.set("at most {d} states and {d} questions per call; this request has {d} and {d}", .{ inference.decide.max_states, inference.decide.max_questions, request.states.len, request.questions.len });
        return error.RequestTooLarge;
    }
    for (request.ids, request.questions) |id, q| if (q.texts.len > inference.decide.max_options) {
        diag.set("question {s}: {d} options, at most {d}", .{ id, q.texts.len, inference.decide.max_options });
        return error.TooManyOptions;
    };

    const started = std.Io.Clock.awake.now(io);
    var decider = Decider.open(gpa, io, directory, .cpu) catch |err| {
        diag.set("{s}: not a Laya checkpoint directory nuclis can run ({s})", .{ directory, @errorName(err) });
        return err;
    };
    defer decider.deinit(io);
    const load_ns: u64 = @intCast(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
    var timings: inference.decide.Timings = .{};
    const states = try arena.alloc(inference.decide.State, request.states.len);
    for (states, request.states) |*s, l| s.* = l.state;
    const results = decider.decide(arena, io, states, request.questions, .{ .uncalibrated = o.uncalibrated }, &timings) catch |err| switch (err) {
        error.OptionsExceedBudget => {
            diag.set("a question's options do not fit in {d} tokens; shorten them or ask fewer", .{decider.config.budget.max_len});
            return err;
        },
        else => return err,
    };
    const report: Report = .{ .request = request, .results = results, .identity = identity, .decider = &decider, .load_ns = load_ns, .timings = timings, .options = o };
    if (o.json) return report.writeJson(out);
    try report.writeText(arena, out, sty);
}

const Report = struct {
    request: Request,
    results: []const inference.decide.StateResult,
    identity: Identity,
    decider: *const Decider,
    load_ns: u64,
    timings: inference.decide.Timings,
    options: Options,

    fn ms(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
    }

    fn writeJson(self: Report, out: *std.Io.Writer) !void {
        var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
        try s.beginObject();
        try s.objectField("schema_version");
        try s.write(schema_version);
        try s.objectField("model");
        try s.write(self.identity.name);
        try s.objectField("repo");
        try s.write(self.identity.repo);
        try s.objectField("revision");
        try s.write(self.identity.revision);
        try s.objectField("timings_ms");
        try s.beginObject();
        try s.objectField("load");
        try s.write(round1(ms(self.load_ns)));
        try s.objectField("tokenize");
        try s.write(round1(ms(self.timings.tokenize_ns)));
        try s.objectField("encode");
        try s.write(round1(ms(self.timings.encode_ns)));
        try s.endObject();
        try s.objectField("results");
        try s.beginArray();
        for (self.results, self.request.states) |result, labeled| {
            try s.beginObject();
            try s.objectField("answers");
            try s.beginObject();
            for (self.request.ids, self.request.questions, result.answers) |id, q, a| {
                try s.objectField(id);
                try writeAnswer(&s, q, a, self.options.explain);
            }
            try s.endObject();
            try s.objectField("usage");
            try s.beginObject();
            try s.objectField("input_tokens");
            try s.write(result.input_tokens);
            try s.objectField("output_tokens");
            try s.write(0);
            try s.endObject();
            try s.objectField("nuclis");
            try s.beginObject();
            try s.objectField("state");
            try s.write(labeled.label);
            try s.objectField("state_tokens");
            try s.write(result.state_tokens);
            try s.objectField("truncated");
            try s.write(result.truncated);
            try s.endObject();
            try s.endObject();
        }
        try s.endArray();
        try s.endObject();
        try out.writeByte('\n');
    }

    fn writeText(self: Report, arena: std.mem.Allocator, out: *std.Io.Writer, sty: style.Style) !void {
        const off = sty.off();
        try out.print("{s}{s}{s} {s}· {d} state{s} · {d} question{s} · load {d:.1} s · tokenize {d:.0} ms · encode {d:.0} ms{s}\n", .{
            sty.on(.header),         self.identity.name,                     off,                        sty.on(.dim),
            self.results.len,        if (self.results.len == 1) "" else "s", self.request.ids.len,       if (self.request.ids.len == 1) "" else "s",
            ms(self.load_ns) / 1000, ms(self.timings.tokenize_ns),           ms(self.timings.encode_ns), off,
        });
        if (self.results.len == 1) {
            const result = self.results[0];
            if (result.truncated) try out.print("{s}the state is {d} tokens; the model read part of it (--truncate chooses the end cut){s}\n", .{ sty.on(.warning), result.state_tokens, off });
            for (self.request.ids, self.request.questions, result.answers) |id, q, a| {
                try out.writeByte('\n');
                try writeQuestion(out, sty, id, q, a);
                if (self.options.explain) try self.writeExplain(arena, out, sty, q, a);
            }
            return;
        }
        // Several states: ranked by the first question.
        const order = try arena.alloc(usize, self.results.len);
        for (order, 0..) |*x, i| x.* = i;
        const first = self.request.questions[0];
        const Rank = struct {
            results: []const inference.decide.StateResult,
            kind: profile.Kind,
            fn greater(ctx: @This(), x: usize, y: usize) bool {
                return rankKey(ctx.kind, ctx.results[x].answers[0]) > rankKey(ctx.kind, ctx.results[y].answers[0]);
            }
        };
        std.mem.sort(usize, order, Rank{ .results = self.results, .kind = first.kind }, Rank.greater);
        try out.print("\n{s}ranked by {s}{s} {s}({s}){s}\n", .{ sty.on(.label), self.request.ids[0], off, sty.on(.dim), rankMeaning(first), off });
        var widest: usize = 0;
        for (self.request.states) |l| widest = @max(widest, @min(l.label.len, 48));
        for (order, 1..) |index, rank| {
            const result = self.results[index];
            const a = result.answers[0];
            const label = self.request.states[index].label;
            try out.print("{d:>3}. {s}{s}{s}", .{ rank, sty.on(.code), label[0..@min(label.len, 48)], off });
            for (0..widest - @min(label.len, 48) + 2) |_| try out.writeByte(' ');
            try writeBar(out, sty, rankKey(first.kind, a) / rankScale(first));
            try out.print(" {s}{s}{s}", .{ sty.on(.number), try summary(arena, first, a), off });
            for (self.request.ids[1..], self.request.questions[1..], result.answers[1..]) |id, q, other|
                try out.print("  {s}{s}{s} {s}", .{ sty.on(.dim), id, off, try summary(arena, q, other) });
            if (result.truncated) try out.print("  {s}truncated ({d} tokens){s}", .{ sty.on(.warning), result.state_tokens, off });
            try out.writeByte('\n');
        }
    }

    fn writeExplain(self: Report, arena: std.mem.Allocator, out: *std.Io.Writer, sty: style.Style, q: profile.Question, a: inference.decide.Answer) !void {
        const off = sty.off();
        const seq = a.sequence;
        const decoded = try inference.bpe.decode(arena, &self.decider.tokenizer.vocab, seq.ids, true, .{});
        var options_kept: usize = 0;
        for (seq.option_kept) |k| options_kept += k;
        try out.print("   {s}sequence{s} {d} tokens: question {d}, options {d} ({d} each at most), state {d} of the budget {d}{s}\n", .{ sty.on(.label), off, seq.ids.len, seq.head_kept, options_kept, std.mem.max(usize, seq.option_kept), seq.state_kept, self.decider.config.budget.max_len, if (seq.truncated) ", cut" else "" });
        try out.print("   {s}temperature{s} {d:.4} ({s}{s}){s}\n", .{ sty.on(.label), off, a.calibrated.temperature, a.bucket, if (self.options.uncalibrated) ", uncalibrated" else "", "" });
        try out.print("   {s}logits{s}", .{ sty.on(.label), off });
        for (q.keys, a.logits) |key, z| try out.print(" {s}={d:.4}", .{ key, z });
        try out.print("\n   {s}decoded{s} {s}{s}{s}\n", .{ sty.on(.label), off, sty.on(.dim), decoded, off });
    }
};

fn round1(x: f64) f64 {
    return @round(x * 10) / 10;
}

/// What ranks states: P(true), the expected score, or the first option's probability.
fn rankKey(kind: profile.Kind, a: inference.decide.Answer) f64 {
    return switch (kind) {
        .noul, .score => a.calibrated.value,
        .choice => a.calibrated.probabilities[0],
    };
}

fn rankMeaning(q: profile.Question) []const u8 {
    return switch (q.kind) {
        .noul => "P(true)",
        .score => "expected score",
        .choice => "P(first option)",
    };
}

fn rankScale(q: profile.Question) f64 {
    return if (q.kind == .score) @floatFromInt(@max(1, q.keys.len - 1)) else 1;
}

fn summary(arena: std.mem.Allocator, q: profile.Question, a: inference.decide.Answer) ![]const u8 {
    const c = a.calibrated;
    return switch (q.kind) {
        .choice => std.fmt.allocPrint(arena, "{s} {d:.2}", .{ q.keys[c.best], c.probabilities[c.best] }),
        .score => std.fmt.allocPrint(arena, "{d:.2}/{d}", .{ c.value, q.keys.len - 1 }),
        .noul => std.fmt.allocPrint(arena, "{d:.2}", .{c.value}),
    };
}

const bar_width = 20;

fn writeBar(out: *std.Io.Writer, sty: style.Style, fraction: f64) !void {
    const eighths: usize = @intFromFloat(@round(std.math.clamp(fraction, 0, 1) * bar_width * 8));
    const blocks = [_][]const u8{ "", "▏", "▎", "▍", "▌", "▋", "▊", "▉" };
    try out.writeAll(sty.on(.accent));
    for (0..eighths / 8) |_| try out.writeAll("█");
    try out.writeAll(blocks[eighths % 8]);
    try out.writeAll(sty.off());
    for (0..bar_width - eighths / 8 - @intFromBool(eighths % 8 != 0)) |_| try out.writeByte(' ');
}

fn writeQuestion(out: *std.Io.Writer, sty: style.Style, id: []const u8, q: profile.Question, a: inference.decide.Answer) !void {
    const off = sty.off();
    const c = a.calibrated;
    try out.print("{s}{s}{s} ", .{ sty.on(.bold), id, off });
    if (!std.mem.eql(u8, id, @tagName(q.kind))) try out.print("{s}{s}{s} ", .{ sty.on(.dim), @tagName(q.kind), off });
    switch (q.kind) {
        .choice => try out.print("{s}{s}{s}", .{ sty.on(.success), q.keys[c.best], off }),
        .score => try out.print("{s}{d:.2}{s} of 0–{d}", .{ sty.on(.number), c.value, off, q.keys.len - 1 }),
        .noul => try out.print("{s}{d:.2}{s} {s}", .{ sty.on(.number), c.value, off, if (c.value >= 0.5) "true" else "false" }),
    }
    try out.print("  {s}confidence {d:.2}{s}\n", .{ sty.on(.dim), c.answer_confidence, off });
    var widest: usize = 0;
    for (q.keys, 0..) |key, i| widest = @max(widest, optionLabel(q, key, i).len);
    for (q.keys, c.probabilities, 0..) |key, p, i| {
        const label = optionLabel(q, key, i);
        try out.print("   {s}", .{label});
        for (0..widest - label.len + 2) |_| try out.writeByte(' ');
        try writeBar(out, sty, p);
        try out.print(" {s}{d:.2}{s}\n", .{ sty.on(.number), p, off });
    }
}

/// A score level shows its text beside its index; other keys stand alone.
fn optionLabel(q: profile.Question, key: []const u8, i: usize) []const u8 {
    if (q.kind == .score and i < q.legend.len and q.legend[i] == .string) return q.legend[i].string;
    return key;
}

fn writeAnswer(s: *std.json.Stringify, q: profile.Question, a: inference.decide.Answer, explain: bool) !void {
    const c = a.calibrated;
    try s.beginObject();
    try s.objectField("type");
    try s.write(@tagName(q.kind));
    switch (q.kind) {
        .choice => {
            try s.objectField("choice");
            try s.write(q.keys[c.best]);
        },
        .score => {
            try s.objectField("score");
            try s.write(profile.round4(c.value));
            try s.objectField("legend");
            try s.beginObject();
            for (q.keys, q.legend) |key, value| {
                try s.objectField(key);
                try s.write(value);
            }
            try s.endObject();
        },
        .noul => {
            try s.objectField("noul");
            try s.write(profile.round4(c.value));
        },
    }
    if (q.kind != .noul) {
        try s.objectField("probabilities");
        try s.beginObject();
        for (q.keys, c.probabilities) |key, p| {
            try s.objectField(key);
            try s.write(profile.round4(p));
        }
        try s.endObject();
    }
    try s.objectField("confidence");
    try s.write(profile.round4(c.confidence));
    try s.objectField("answer_confidence");
    try s.write(profile.round4(c.answer_confidence));
    try s.objectField("nuclis");
    try s.beginObject();
    try s.objectField("logits");
    try s.write(a.logits);
    try s.objectField("temperature");
    try s.write(c.temperature);
    try s.objectField("bucket");
    try s.write(a.bucket);
    if (explain) {
        try s.objectField("sequence_tokens");
        try s.write(a.sequence.ids.len);
        try s.objectField("state_kept");
        try s.write(a.sequence.state_kept);
    }
    try s.endObject();
    try s.endObject();
}

test "arguments: tiers, attachment, and conflicts" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const inline_args = [_][]const u8{ "--choice", "Which team?", "--option", "billing=invoices", "--option", "other", "--id", "team", "--noul", "Angry?", "--state", "text", "--state-file", "a.txt", "--json" };
    const o = try parseArgs(arena, &inline_args, &diag);
    try std.testing.expectEqual(@as(usize, 2), o.inline_questions.items.len);
    try std.testing.expectEqualStrings("team", o.inline_questions.items[0].id.?);
    try std.testing.expectEqual(@as(usize, 2), o.inline_questions.items[0].options.items.len);
    try std.testing.expectEqual(@as(usize, 2), o.states.items.len);
    try std.testing.expect(o.json);
    const cases = .{
        .{ &[_][]const u8{ "--state", "x" }, error.MissingQuestions },
        .{ &[_][]const u8{ "--choice", "x" }, error.MissingState },
        .{ &[_][]const u8{ "--option", "a", "--choice", "x", "--state", "s" }, error.MisplacedOption },
        .{ &[_][]const u8{ "--noul", "x", "--level", "a", "--state", "s" }, error.MisplacedOption },
        .{ &[_][]const u8{ "--request", "r.json", "--state", "s" }, error.ConflictingOptions },
        .{ &[_][]const u8{ "--request", "r.json", "--questions", "q.json" }, error.ConflictingOptions },
        .{ &[_][]const u8{ "--request", "r.json", "--truncate", "middle" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--request", "r.json", "--wat" }, error.UnknownOption },
        .{ &[_][]const u8{ "--request", "r.json", "--json", "--json" }, error.DuplicateOption },
        .{ &[_][]const u8{"--request"}, error.MissingOptionValue },
    };
    inline for (cases) |case| try std.testing.expectError(case[1], parseArgs(arena, case[0], &diag));
}

test "inline questions build the package's definitions" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const args = [_][]const u8{ "--choice", "Which?", "--option", "billing=invoices, refunds", "--option", "other", "--score", "How urgent?", "--level", "not urgent", "--level", "blocking", "--noul", "Cancel?", "--state", "hello" };
    const o = try parseArgs(arena, &args, &diag);
    const request = try buildRequest(arena, std.testing.io, o, &diag);
    try std.testing.expectEqualStrings("choice", request.ids[0]);
    try std.testing.expectEqualStrings("billing: invoices, refunds", request.questions[0].texts[0]);
    try std.testing.expectEqualStrings("other", request.questions[0].texts[1]);
    try std.testing.expectEqualStrings("level 1: blocking", request.questions[1].texts[1]);
    try std.testing.expectEqualStrings("true: yes, the statement holds", request.questions[2].texts[1]);
    const twice = [_][]const u8{ "--noul", "a", "--noul", "b", "--state", "s" };
    try std.testing.expectError(error.DuplicateQuestion, buildRequest(arena, std.testing.io, try parseArgs(arena, &twice, &diag), &diag));
}
