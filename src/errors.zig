//! What an error says to the person at the terminal. `main` prints
//! `nuclis: <message>` and never an error's name: the message is the
//! diagnostic a failure set, else one built here. A parse error is
//! explained from the argument the parser stopped at, and is a usage
//! error: `Run 'nuclis <command> --help' for usage.` follows it, and the
//! exit status is 2 (130 for an interrupt, 1 otherwise;
//! docs/app/command-output.md).
const std = @import("std");
const config = @import("config.zig");

/// The words of a command line and the index of the argument the parser
/// was reading when it failed.
pub const Cursor = struct {
    args: []const []const u8,
    at: usize = 0,

    fn arg(c: Cursor, i: usize) []const u8 {
        return if (i < c.args.len) c.args[i] else "";
    }

    /// `model pull`, `config set`, `generate`: the command and its action.
    fn command(c: Cursor, buf: []u8) []const u8 {
        if (c.args.len == 0) return "";
        const head = c.args[0];
        const has_action = for ([_][]const u8{ "model", "config", "cache", "agent", "voice" }) |n| {
            if (std.mem.eql(u8, head, n)) break true;
        } else false;
        if (has_action and c.args.len > 1 and !std.mem.startsWith(u8, c.args[1], "-") and c.at >= 1)
            return std.fmt.bufPrint(buf, "{s} {s}", .{ head, c.args[1] }) catch head;
        return head;
    }
};

/// Sets `diag` to what the parse error `err` means at `cursor`, and its
/// hint to the command's help page.
pub fn explainParse(diag: *config.Diagnostic, err: anyerror, cursor: Cursor) void {
    var buf: [64]u8 = undefined;
    const cmd = cursor.command(&buf);
    const flag = cursor.arg(cursor.at);
    const value = cursor.arg(cursor.at + 1);
    switch (err) {
        error.UnknownCommand => diag.set("'{s}' is not a nuclis command", .{flag}),
        error.UnknownOption => if (flag.len == 0)
            diag.set("'nuclis {s}' takes no further arguments", .{cmd})
        else if (std.mem.startsWith(u8, flag, "-"))
            diag.set("unknown option '{s}' for 'nuclis {s}'", .{ flag, cmd })
        else
            diag.set("unexpected argument '{s}' for 'nuclis {s}'", .{ flag, cmd }),
        error.DuplicateOption => diag.set("'{s}' is given twice", .{flag}),
        error.InvalidNumber => diag.set("'{s}' does not take '{s}'", .{ flag, value }),
        error.MissingOptionValue => diag.set("'{s}' needs a value", .{flag}),
        error.ConflictingOptions => diag.set("these options do not go together for 'nuclis {s}'", .{cmd}),
        error.UnknownRole => diag.set("'{s}' is not a companion role: mmproj, draft[=<variant>], codec, imatrix", .{value}),
        error.UnknownPromptProfile => diag.set("'{s}' is not a prompt profile", .{value}),
        error.UnsupportedBackend => diag.set("'{s}' is not a backend: cpu or metal", .{value}),
        error.UnsupportedKvPrecision => diag.set("'{s}' is not a cache precision: f16 or f32", .{value}),
        error.ConflictingPromptSources => diag.set("give the prompt one way: --prompt, --prompt-file, or --prompt-tokens", .{}),
        error.MissingPrompt => diag.set("'nuclis {s}' needs a prompt: --prompt <text> or --prompt-file <path>", .{cmd}),
        error.MissingModelPath => diag.set("'--model' needs a name or a path", .{}),
        error.MissingRepository => if (std.mem.eql(u8, cmd, "model pull"))
            diag.set("'nuclis model pull' needs a name: a catalogue name, a registry entry, owner/repo, or --catalogue", .{})
        else
            diag.set("'nuclis {s}' needs a name: a catalogue name, a registry entry, or owner/repo", .{cmd}),
        error.MissingModelAction => diag.set("'nuclis model' needs an action: ls, pull, inspect, adopt, or rm", .{}),
        error.UnknownModelAction => diag.set("'{s}' is not a model action: ls, pull, inspect, adopt, or rm", .{flag}),
        error.MissingConfigAction => diag.set("'nuclis config' needs an action: init, sync, show, or set", .{}),
        error.UnknownConfigAction => diag.set("'{s}' is not a config action: init, sync, show, or set", .{flag}),
        error.MissingSetArguments => diag.set("'nuclis config set' needs a key and a value", .{}),
        error.UnknownCacheAction => diag.set("'{s}' is not a cache action: ls or clear", .{flag}),
        error.MissingShell => diag.set("'nuclis completion' needs a shell: fish, bash, or zsh", .{}),
        error.UnknownShell => diag.set("'{s}' is not a shell nuclis completes: fish, bash, or zsh", .{flag}),
        error.MissingSessionId => diag.set("'nuclis {s}' needs a session id (`nuclis agent ls` lists them)", .{cmd}),
        error.MissingTextFile => diag.set("'nuclis eval' needs a text file: --file <path>", .{}),
        error.TooManyImages => diag.set("too many images for one prompt", .{}),
        error.TooManyClips => diag.set("too many audio clips for one prompt", .{}),
        else => diag.set("{s}", .{describe(err)}),
    }
    // Help pages are per command (`nuclis model --help`), not per action.
    if (err == error.UnknownCommand)
        diag.setHint("Run 'nuclis --help' for the commands.", .{})
    else
        diag.setHint("Run 'nuclis {s} --help' for usage.", .{cursor.arg(0)});
}

/// A sentence for an error that arrives without a diagnostic; the
/// name read as words when there is none here (`ContextFull` →
/// "context full").
pub fn describe(err: anyerror) []const u8 {
    return switch (err) {
        error.MissingHome => "no home directory: set HOME or NUCLIS_HOME",
        error.ContextFull => "the context is full; raise --ctx-size or shorten the prompt",
        error.EmptyPrompt => "the prompt is empty",
        error.OutOfMemory => "out of memory",
        error.FileNotFound => "a file it needs was not found",
        error.AccessDenied, error.PermissionDenied => "permission denied",
        error.MetalNotEnabled => "this build has no Metal backend; build with `make metal`",
        error.InferenceUnavailable => "the model's weights bind, but no executor runs its architecture",
        error.NoProjector => "the model has no projector for images or audio (pull it with --with mmproj)",
        error.ImagesNeedChatPrompt => "images go with a chat prompt, not --raw",
        error.ProfileRequiresMetal => "--profile needs the Metal backend",
        error.InvalidUtf8 => "the text is not valid UTF-8",
        error.BrokenPipe => "the output was closed",
        error.Cancelled, error.Canceled => "interrupted",
        else => "",
    };
}

/// `ContextFull` as "context full", into `buf`.
pub fn words(buf: []u8, name: []const u8) []const u8 {
    var n: usize = 0;
    for (name, 0..) |c, i| {
        if (n + 2 > buf.len) break;
        if (std.ascii.isUpper(c)) {
            if (i > 0) {
                buf[n] = ' ';
                n += 1;
            }
            buf[n] = std.ascii.toLower(c);
        } else buf[n] = c;
        n += 1;
    }
    return buf[0..n];
}

/// 130 for an interrupt, 2 for a usage error, 1 for the rest.
pub fn exitCode(err: anyerror, usage: bool) u8 {
    if (err == error.Cancelled or err == error.Canceled) return 130;
    return if (usage) 2 else 1;
}

const testing = std.testing;

fn expectParse(expected: []const u8, err: anyerror, args: []const []const u8, at: usize) !void {
    var diag: config.Diagnostic = .{};
    explainParse(&diag, err, .{ .args = args, .at = at });
    try testing.expectEqualStrings(expected, diag.message());
}

test "a parse error says which argument and which command" {
    try expectParse("unknown option '--frob' for 'nuclis model pull'", error.UnknownOption, &.{ "model", "pull", "x", "--frob" }, 3);
    try expectParse("'--with' is given twice", error.DuplicateOption, &.{ "model", "pull", "x", "--with", "a", "--with", "b" }, 5);
    try expectParse("'--draft-length' needs a value", error.MissingOptionValue, &.{ "generate", "--draft-length" }, 1);
    try expectParse("'frob' is not a nuclis command", error.UnknownCommand, &.{"frob"}, 0);
    try expectParse("'nuclis model pull' needs a name: a catalogue name, a registry entry, owner/repo, or --catalogue", error.MissingRepository, &.{ "model", "pull" }, 1);
    try expectParse("'--ctx-size' does not take 'big'", error.InvalidNumber, &.{ "generate", "--ctx-size", "big" }, 1);
    var diag: config.Diagnostic = .{};
    explainParse(&diag, error.UnknownOption, .{ .args = &.{ "config", "show", "--frob" }, .at = 2 });
    try testing.expectEqualStrings("Run 'nuclis config --help' for usage.", diag.hint());
}

test "an error without a diagnostic still reads as words" {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("context full", words(&buf, "ContextFull"));
    try testing.expectEqualStrings("interrupted", describe(error.Cancelled));
    try testing.expectEqual(@as(u8, 130), exitCode(error.Cancelled, false));
    try testing.expectEqual(@as(u8, 2), exitCode(error.UnknownOption, true));
    try testing.expectEqual(@as(u8, 1), exitCode(error.ContextFull, false));
}
