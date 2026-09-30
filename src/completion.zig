//! Shell completion: the command table, `nuclis __complete`, and the shims
//! `nuclis completion <shell>` prints.
//!
//! The shims are thin: on every Tab they run `nuclis __complete <words…>`
//! (the words after `nuclis`, the last one the word being completed, maybe
//! empty) and show what it prints: `candidate<TAB>description` lines, or
//! one directive line, `:file` or `:dir`, handing the word to the shell's
//! own path completion. So a new flag or a newly pulled model completes
//! without regenerating any script.
//!
//! `resolve` (words → what is being completed) and `candidates` (that and a
//! `State` snapshot → lines) are pure; `run` reads only the state the
//! position needs. The tests hold the table to `cli.parseArgs` and to each
//! command's `--help` page. A failure completes nothing: stdout is the
//! user's prompt line, never an error message.
const std = @import("std");
const inference = @import("inference");
const config = @import("config.zig");
const catalog = @import("catalog.zig");
const paths = @import("paths.zig");
const model = @import("model.zig");
const engine = @import("engine.zig");
const resume_mod = @import("agent/resume.zig");
const Allocator = std.mem.Allocator;

/// What a flag's value or a positional argument is, and so where its
/// candidates come from.
pub const Value = union(enum) {
    /// A switch: the flag takes no value.
    none,
    text,
    number,
    file,
    dir,
    choice: []const []const u8,
    /// A registry entry, a catalogue name, or a path.
    model,
    /// What `model pull`/`inspect` name: a catalogue or registry name, or
    /// an `owner/repo` typed out.
    repo,
    /// A saved session of the working directory.
    session,
    config_key,
    /// The value of the `config_key` before it.
    config_value,
    /// A comma-separated list of companion roles.
    roles,
};

pub const Flag = struct {
    name: []const u8,
    short: ?[]const u8 = null,
    value: Value = .none,
    /// The value may be left out (`--resume` alone is the newest session).
    optional: bool = false,
    repeat: bool = false,
    summary: []const u8,
};

/// A command's subcommand, or its only form (`name` empty).
pub const Action = struct {
    name: []const u8 = "",
    summary: []const u8 = "",
    flags: []const Flag = &.{},
    positionals: []const Value = &.{},

    fn flag(self: *const Action, word: []const u8) ?*const Flag {
        for (self.flags) |*f| {
            if (std.mem.eql(u8, f.name, word)) return f;
            if (f.short) |s| if (std.mem.eql(u8, s, word)) return f;
        }
        return null;
    }
};

pub const Command = struct {
    name: []const u8,
    summary: []const u8,
    /// The first action is the default when it is unnamed (`agent` runs,
    /// `agent ls` lists); a command whose actions are all named needs one.
    actions: []const Action,

    fn named(self: *const Command) bool {
        return self.actions[0].name.len > 0;
    }
};

pub const Shell = enum { fish, bash, zsh };

/// The most lines one answer prints: a shell shows the rest as noise.
pub const max_candidates = 200;
/// Sessions offered after `--resume`, newest first.
pub const max_sessions = 50;

// ----- the table -----

fn names(comptime E: type) []const []const u8 {
    const list = comptime blk: {
        var list: []const []const u8 = &.{};
        for (@typeInfo(E).@"enum".fields) |f| list = list ++ &[_][]const u8{f.name};
        break :blk list;
    };
    return list;
}

const json: Flag = .{ .name = "--json", .summary = "machine-readable output" };
const model_flag: Flag = .{ .name = "--model", .value = .model, .summary = "registry entry, catalogue name, or path" };
const prompt: Flag = .{ .name = "--prompt", .value = .text, .summary = "the prompt text" };
const prompt_file: Flag = .{ .name = "--prompt-file", .value = .file, .summary = "the prompt, read from a file" };
const prompt_tokens: Flag = .{ .name = "--prompt-tokens", .value = .text, .summary = "a JSON array of token ids" };
const raw: Flag = .{ .name = "--raw", .summary = "skip the chat template" };
const think: Flag = .{ .name = "--think", .value = .{ .choice = names(inference.profiles.Effort) }, .summary = "reasoning effort" };
const image_max_tokens: Flag = .{ .name = "--image-max-tokens", .value = .{ .choice = &.{"auto"} }, .summary = "most tokens one image becomes" };
const backend: Flag = .{ .name = "--backend", .value = .{ .choice = names(engine.Backend) }, .summary = "where the model runs" };
const ctx_size: Flag = .{ .name = "--ctx-size", .value = .number, .summary = "context window in tokens" };
const kv: Flag = .{ .name = "--kv", .value = .{ .choice = names(config.KvPrecision) }, .summary = "attention cache precision" };
const prompt_profile: Flag = .{ .name = "--prompt-profile", .value = .{ .choice = names(inference.profiles.Profile) }, .summary = "force a prompt profile" };

/// The engine and sampling options `generate`, `bench`, and `agent` share.
const engine_flags = [_]Flag{
    model_flag,
    backend,
    ctx_size,
    kv,
    .{ .name = "--max-tokens", .value = .number, .summary = "output budget" },
    .{ .name = "--speculative", .value = .{ .choice = &.{ "on", "off" } }, .summary = "verify drafts from the draft source" },
    .{ .name = "--draft-length", .value = .number, .summary = "drafts per verify batch" },
    prompt_profile,
    .{ .name = "--seed", .value = .number, .summary = "sampler seed" },
    .{ .name = "--temperature", .value = .number, .summary = "0 is greedy" },
    .{ .name = "--top-k", .value = .number, .summary = "sampling override" },
    .{ .name = "--top-p", .value = .number, .summary = "sampling override" },
    .{ .name = "--min-p", .value = .number, .summary = "sampling override" },
    .{ .name = "--presence-penalty", .value = .number, .summary = "sampling override" },
    .{ .name = "--repetition-penalty", .value = .number, .summary = "sampling override" },
};

pub const commands = [_]Command{
    .{ .name = "agent", .summary = "the interactive chat", .actions = &.{
        .{ .flags = &([_]Flag{
            .{ .name = "--prompt", .short = "-p", .value = .text, .summary = "run one turn without a terminal" },
            .{ .name = "--print", .summary = "one turn, the prompt from --prompt-file" },
            prompt_file,
            .{ .name = "--json", .summary = "print mode: one event per line" },
            .{ .name = "--session", .value = .file, .summary = "record a printed turn to this file" },
            .{ .name = "--resume", .value = .session, .optional = true, .summary = "replay a saved session first" },
            .{ .name = "--system-prompt", .value = .file, .summary = "replace the built system prompt" },
            think,
            .{ .name = "--thinking-budget", .value = .number, .summary = "most reasoning tokens per step" },
            image_max_tokens,
        } ++ engine_flags) },
        .{ .name = "ls", .summary = "this workspace's saved sessions", .flags = &.{json} },
    } },
    .{ .name = "generate", .summary = "one completion from a prompt", .actions = &.{.{ .flags = &([_]Flag{
        prompt,
        prompt_file,
        prompt_tokens,
        raw,
        .{ .name = "--image", .value = .file, .repeat = true, .summary = "attach an image" },
        image_max_tokens,
        think,
        .{ .name = "--logits", .value = .file, .summary = "write the final prompt logits" },
        .{ .name = "--trace-dir", .value = .dir, .summary = "write every layer's output" },
        json,
    } ++ engine_flags) }} },
    .{ .name = "bench", .summary = "repeated prefill and decode measurements", .actions = &.{.{ .flags = &([_]Flag{
        prompt,
        prompt_file,
        prompt_tokens,
        raw,
        .{ .name = "--repeat", .value = .number, .summary = "measured runs" },
        .{ .name = "--warmup", .value = .number, .summary = "unmeasured runs first" },
        .{ .name = "--profile", .summary = "per-kernel GPU time" },
        .{ .name = "--unfused-norms", .summary = "run the norm pairs unfused" },
        .{ .name = "--kernel-stats", .summary = "each Metal pipeline's limits, no model" },
        .{ .name = "--capture", .value = .file, .summary = "one decode step into a .gputrace" },
        .{ .name = "--prefix-cache", .value = .dir, .summary = "save and restore the prefilled prompt" },
        .{ .name = "--verify-rows", .value = .number, .summary = "time verify batches of R rows" },
        .{ .name = "--accept", .value = .number, .summary = "drafts each verify batch accepts" },
        json,
    } ++ engine_flags) }} },
    .{ .name = "tokenize", .summary = "the prompt's token ids, no model run", .actions = &.{.{ .flags = &.{
        prompt,
        prompt_file,
        raw,
        think,
        prompt_profile,
        model_flag,
        json,
    } }} },
    .{ .name = "eval", .summary = "perplexity of a text file", .actions = &.{.{ .flags = &.{
        .{ .name = "--file", .value = .file, .summary = "the text, read raw" },
        ctx_size,
        .{ .name = "--chunks", .value = .number, .summary = "windows from the start" },
        .{ .name = "--reference", .value = .file, .summary = "a pinned reference run" },
        model_flag,
        backend,
        kv,
        prompt_profile,
        json,
    } }} },
    .{ .name = "inspect", .summary = "an artifact's identity and encodings", .actions = &.{.{ .flags = &.{
        model_flag,
        .{ .name = "--tensor", .value = .text, .summary = "one safetensors tensor" },
        json,
    } }} },
    .{ .name = "validate", .summary = "whether a file binds to its adapter", .actions = &.{.{ .flags = &.{ model_flag, json } }} },
    .{ .name = "model", .summary = "pull, list, and judge Hub artifacts", .actions = &.{
        .{ .name = "ls", .summary = "the catalogue and every local file", .flags = &.{json} },
        .{ .name = "pull", .summary = "fetch a model, verified by digest", .positionals = &.{.repo}, .flags = &.{
            .{ .name = "--with", .value = .roles, .summary = "also fetch these companions" },
            .{ .name = "--all", .summary = "every companion the entry names" },
            .{ .name = "--file", .value = .text, .summary = "which file of the repository" },
            .{ .name = "--revision", .value = .text, .summary = "a branch, tag, or commit" },
            .{ .name = "--role", .value = .{ .choice = names(model.Role) }, .summary = "the file's role" },
            .{ .name = "--register", .value = .text, .summary = "record the pull as a registry entry" },
            .{ .name = "--profile", .value = .{ .choice = names(config.Profile) }, .summary = "with --register: force a profile" },
            .{ .name = "--force", .summary = "replace a file with other content" },
            json,
        } },
        .{ .name = "inspect", .summary = "judge a remote file before the download", .positionals = &.{.repo}, .flags = &.{
            .{ .name = "--file", .value = .text, .summary = "which file of the repository" },
            .{ .name = "--revision", .value = .text, .summary = "a branch, tag, or commit" },
            json,
        } },
    } },
    .{ .name = "decide", .summary = "typed questions about states, by Laya", .actions = &.{.{ .flags = &.{
        .{ .name = "--request", .value = .file, .summary = "a Jev request file, - for stdin" },
        .{ .name = "--questions", .value = .file, .summary = "the questions object alone" },
        .{ .name = "--state", .value = .text, .repeat = true, .summary = "a state as text" },
        .{ .name = "--state-file", .value = .file, .repeat = true, .summary = "a state read from a file" },
        .{ .name = "--choice", .value = .text, .repeat = true, .summary = "an inline choice question" },
        .{ .name = "--option", .value = .text, .repeat = true, .summary = "a choice option, key[=description]" },
        .{ .name = "--score", .value = .text, .repeat = true, .summary = "an inline score question" },
        .{ .name = "--level", .value = .text, .repeat = true, .summary = "a score level, 0 first" },
        .{ .name = "--noul", .value = .text, .repeat = true, .summary = "an inline yes/no question" },
        .{ .name = "--id", .value = .text, .repeat = true, .summary = "names the question before it" },
        .{ .name = "--model", .value = .dir, .summary = "the checkpoint directory" },
        .{ .name = "--truncate", .value = .{ .choice = &.{ "head", "tail" } }, .summary = "the end of a long state cut" },
        backend,
        .{ .name = "--uncalibrated", .summary = "no temperature" },
        .{ .name = "--explain", .summary = "the sequences, budgets, and logits" },
        json,
    } }} },
    .{ .name = "config", .summary = "write, show, or set a key of the config file", .actions = &.{
        .{ .name = "init", .summary = "write the file with the defaults", .flags = &.{
            .{ .name = "--discover", .summary = "also register the files found" },
            .{ .name = "--dry-run", .summary = "what --discover would register" },
            json,
        } },
        .{ .name = "show", .summary = "every key's value and its source", .flags = &.{json} },
        .{ .name = "set", .summary = "change one key", .positionals = &.{ .config_key, .config_value } },
    } },
    .{ .name = "completion", .summary = "the shell completion script", .actions = &.{.{ .positionals = &.{.{ .choice = names(Shell) }} }} },
};

const globals = [_]Flag{
    .{ .name = "--help", .short = "-h", .summary = "the overview, or a command's page" },
    .{ .name = "--version", .summary = "the version of this binary" },
};

pub fn find(name: []const u8) ?*const Command {
    for (&commands) |*c| if (std.mem.eql(u8, c.name, name)) return c;
    return null;
}

// ----- where the cursor is -----

/// What the last word completes to.
pub const Target = union(enum) {
    none,
    commands,
    /// A command's actions, and the default action's flags when it has one.
    actions: *const Command,
    /// The action's flags, less the ones in `used` (the words so far).
    flags: struct { action: *const Action, used: []const []const u8 },
    /// A value; `key` is the `config set` key a `config_value` belongs to.
    value: struct { kind: Value, key: []const u8 = "" },
};

/// `words` as the shell passed them: every word after `nuclis`, the last
/// one being completed.
pub fn resolve(words: []const []const u8) Target {
    if (words.len == 0) return .commands;
    const done = words[0 .. words.len - 1];
    const current = words[words.len - 1];
    if (done.len == 0) return .commands;
    const command = find(done[0]) orelse return .none;
    if (done.len == 1 and command.actions.len > 1) return .{ .actions = command };
    var action = &command.actions[0];
    var i: usize = 1;
    if (command.named()) {
        action = for (command.actions) |*a| {
            if (std.mem.eql(u8, a.name, done[1])) break a;
        } else return .none;
        i = 2;
    } else if (command.actions.len > 1) {
        for (command.actions[1..]) |*a| if (std.mem.eql(u8, a.name, done[1])) {
            action = a;
            i = 2;
        };
    }
    const start = i;
    var positional: usize = 0;
    var last_positional: []const u8 = "";
    while (i < done.len) : (i += 1) {
        const word = done[i];
        if (action.flag(word)) |f| {
            if (f.value == .none) continue;
            if (i + 1 == done.len) {
                // The word being completed is this flag's value, unless the
                // value is optional and a flag is being typed instead.
                if (!(f.optional and std.mem.startsWith(u8, current, "-"))) return .{ .value = .{ .kind = f.value } };
            } else if (!(f.optional and std.mem.startsWith(u8, done[i + 1], "-"))) i += 1;
        } else if (!std.mem.startsWith(u8, word, "-")) {
            positional += 1;
            last_positional = word;
        }
    }
    if (!std.mem.startsWith(u8, current, "-") and positional < action.positionals.len)
        return .{ .value = .{ .kind = action.positionals[positional], .key = last_positional } };
    return .{ .flags = .{ .action = action, .used = done[start..] } };
}

// ----- what it completes to -----

pub const Candidate = struct { text: []const u8, description: []const u8 = "" };

/// The user's state a value may need, read by `gather`; empty in a test
/// that does not name it.
pub const State = struct {
    /// Registry entries, then catalogue names.
    models: []const Candidate = &.{},
    /// The registry's names alone, for `models.<name>.` keys.
    registry: []const []const u8 = &.{},
    /// Newest first.
    sessions: []const Candidate = &.{},
};

pub const Answer = union(enum) {
    lines: []const Candidate,
    /// Hand the word to the shell's path completion.
    file,
    dir,
};

/// The candidates for `target` that start with `current`, at most
/// `max_candidates`. Slices borrow the table, `state`, and `arena`.
pub fn candidates(arena: Allocator, target: Target, current: []const u8, state: State) !Answer {
    var list: std.ArrayList(Candidate) = .empty;
    switch (target) {
        .none => {},
        .commands => {
            for (&commands) |*c| try list.append(arena, .{ .text = c.name, .description = c.summary });
            for (&globals) |*f| try list.append(arena, .{ .text = f.name, .description = f.summary });
        },
        .actions => |command| {
            for (command.actions) |*a| if (a.name.len > 0) try list.append(arena, .{ .text = a.name, .description = a.summary });
            if (!command.named()) try appendFlags(arena, &list, &command.actions[0], &.{});
        },
        .flags => |f| try appendFlags(arena, &list, f.action, f.used),
        .value => |v| switch (v.kind) {
            .none, .text, .number => {},
            .file => return .file,
            .dir => return .dir,
            .choice => |choices| for (choices) |c| try list.append(arena, .{ .text = c }),
            .model, .repo => {
                // A path is the shell's to complete; `owner/repo` has no
                // local list to offer.
                if (v.kind == .model and isPathLike(current)) return .file;
                try list.appendSlice(arena, state.models);
            },
            .session => try list.appendSlice(arena, state.sessions),
            .config_key => {
                for (config.global_keys) |k| try list.append(arena, .{ .text = k.path });
                if (std.mem.startsWith(u8, current, "models.")) {
                    for (state.registry) |name| for (config.entry_keys) |k| {
                        try list.append(arena, .{ .text = try std.fmt.allocPrint(arena, "models.{s}.{s}", .{ name, k.path }) });
                    };
                } else if (state.registry.len > 0) try list.append(arena, .{ .text = "models.", .description = "a registry entry's key" });
            },
            .config_value => {
                if (std.mem.eql(u8, v.key, "engine.model")) try list.appendSlice(arena, state.models);
                for (valueChoices(v.key)) |c| try list.append(arena, .{ .text = c });
            },
            .roles => {
                // Complete the item after the last comma, keeping the ones
                // already typed.
                const cut = if (std.mem.lastIndexOfScalar(u8, current, ',')) |c| c + 1 else 0;
                const typed = current[0..cut];
                for (names(model.Role)) |role| {
                    if (std.mem.eql(u8, role, "main") or std.mem.eql(u8, role, "support")) continue;
                    if (listed(typed, role)) continue;
                    try list.append(arena, .{ .text = try std.mem.concat(arena, u8, &.{ typed, role }) });
                }
            },
        },
    }
    var kept: std.ArrayList(Candidate) = .empty;
    for (list.items) |c| {
        if (!std.mem.startsWith(u8, c.text, current)) continue;
        if (kept.items.len == max_candidates) break;
        try kept.append(arena, c);
    }
    return .{ .lines = kept.items };
}

fn appendFlags(arena: Allocator, list: *std.ArrayList(Candidate), action: *const Action, used: []const []const u8) !void {
    for (action.flags) |*f| {
        if (!f.repeat and (contains(used, f.name) or (f.short != null and contains(used, f.short.?)))) continue;
        try list.append(arena, .{ .text = f.name, .description = f.summary });
    }
}

fn contains(words: []const []const u8, word: []const u8) bool {
    for (words) |w| if (std.mem.eql(u8, w, word)) return true;
    return false;
}

fn listed(typed: []const u8, role: []const u8) bool {
    var it = std.mem.splitScalar(u8, typed, ',');
    while (it.next()) |item| if (std.mem.eql(u8, item, role)) return true;
    return false;
}

fn isPathLike(word: []const u8) bool {
    return std.mem.indexOfScalar(u8, word, '/') != null or std.mem.startsWith(u8, word, ".") or std.mem.startsWith(u8, word, "~");
}

/// The spelled-out values of a `config set` key, global or `models.<n>.`.
fn valueChoices(key: []const u8) []const []const u8 {
    var path = key;
    var keys: []const config.KeyInfo = config.global_keys;
    if (std.mem.startsWith(u8, key, "models.")) {
        const rest = key["models.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return &.{};
        path = rest[dot + 1 ..];
        keys = config.entry_keys;
    }
    for (keys) |k| if (std.mem.eql(u8, k.path, path)) return k.choices;
    return &.{};
}

// ----- the command -----

/// `nuclis __complete <words…>`: prints the answer. Never fails: a broken
/// configuration or an unreadable directory completes less, silently.
pub fn run(gpa: Allocator, io: std.Io, environ: *const std.process.Environ.Map, words: []const []const u8, out: *std.Io.Writer) void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const current = if (words.len > 0) words[words.len - 1] else "";
    const target = resolve(words);
    const state = gather(arena, io, environ, target, current);
    const answer = candidates(arena, target, current, state) catch return;
    write(out, answer) catch {};
}

/// The lines as the shims read them: `text<TAB>description`, control
/// characters in a description made spaces.
pub fn write(out: *std.Io.Writer, answer: Answer) !void {
    switch (answer) {
        .file => try out.writeAll(":file\n"),
        .dir => try out.writeAll(":dir\n"),
        .lines => |lines| for (lines) |c| {
            try out.writeAll(c.text);
            try out.writeByte('\t');
            for (c.description) |b| try out.writeByte(if (b < 0x20 or b == 0x7f) ' ' else b);
            try out.writeByte('\n');
        },
    }
}

/// Reads only what `target` needs; each part that fails is left empty.
fn gather(arena: Allocator, io: std.Io, environ: *const std.process.Environ.Map, target: Target, current: []const u8) State {
    const kind = switch (target) {
        .value => |v| v,
        else => return .{},
    };
    const wants_models = switch (kind.kind) {
        .model, .repo => true,
        .config_value => std.mem.eql(u8, kind.key, "engine.model"),
        else => false,
    };
    const wants_registry = kind.kind == .config_key and std.mem.startsWith(u8, "models.", current[0..@min(current.len, "models.".len)]);
    if (!wants_models and !wants_registry and kind.kind != .session) return .{};
    const root = (paths.root(arena, environ.get("NUCLIS_HOME"), environ.get("HOME")) catch null) orelse return .{};
    var state: State = .{};
    if (kind.kind == .session) {
        state.sessions = sessions(arena, io, root) catch &.{};
        return state;
    }
    const registry = loadRegistry(arena, io, root);
    if (wants_registry) {
        var list: std.ArrayList([]const u8) = .empty;
        for (registry.entries) |named| list.append(arena, named.name) catch break;
        state.registry = list.items;
    }
    if (wants_models) state.models = models(arena, io, root, registry) catch &.{};
    return state;
}

fn loadRegistry(arena: Allocator, io: std.Io, root: []const u8) config.Models {
    const path = paths.configPath(arena, root) catch return .{};
    var diag: config.Diagnostic = .{};
    // The arena owns the loaded file; it is never deinitialized on its own.
    const loaded = config.load(arena, io, .cwd(), path, &diag) catch return .{};
    return loaded.config.models;
}

/// Registry entries (where each lives), then the catalogue's names not
/// shadowed by one (architecture, quantization, local status).
fn models(arena: Allocator, io: std.Io, root: []const u8, registry: config.Models) ![]const Candidate {
    var list: std.ArrayList(Candidate) = .empty;
    for (registry.entries) |named| {
        const e = named.entry;
        const where = if (e.path) |p| p else if (e.repo) |r| try std.fmt.allocPrint(arena, "{s}/{s}", .{ r, e.file orelse "" }) else "";
        try list.append(arena, .{ .text = named.name, .description = where });
    }
    const dir = try model.modelsDir(arena, root);
    for (&catalog.entries) |*e| {
        if (registry.find(e.name) != null) continue;
        const path = try catalog.localPath(arena, dir, e, e.file);
        const status = catalog.status(arena, io, path, e.sha256, e.size) catch .absent;
        try list.append(arena, .{ .text = e.name, .description = try std.fmt.allocPrint(arena, "{s} · {s} · {s}", .{ e.architecture, e.quantization, @tagName(status) }) });
    }
    return list.items;
}

fn sessions(arena: Allocator, io: std.Io, root: []const u8) ![]const Candidate {
    const cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", arena);
    const summaries = try resume_mod.list(arena, io, root, cwd);
    const list = try arena.alloc(Candidate, @min(summaries.len, max_sessions));
    for (list, summaries[0..list.len]) |*c, s| c.* = .{
        .text = s.id,
        .description = try std.fmt.allocPrint(arena, "{s} · {s}", .{ s.time, s.first_prompt }),
    };
    return list;
}

// ----- the shims -----

/// The script `nuclis completion <shell>` prints.
pub fn script(shell: Shell) []const u8 {
    return switch (shell) {
        .fish => fish_script,
        .bash => bash_script,
        .zsh => zsh_script,
    };
}

const fish_script =
    \\# nuclis completion for fish: every Tab asks the binary.
    \\function __nuclis_complete
    \\    set -l tokens (commandline -xpc 2>/dev/null; or commandline -opc)
    \\    set -l current (commandline -ct)
    \\    set -l lines (command nuclis __complete $tokens[2..-1] "$current" 2>/dev/null)
    \\    switch "$lines[1]"
    \\        case :file
    \\            __fish_complete_path "$current"
    \\        case :dir
    \\            __fish_complete_directories "$current"
    \\        case '*'
    \\            printf '%s\n' $lines
    \\    end
    \\end
    \\complete -c nuclis -f -k -a '(__nuclis_complete)'
    \\
;

const bash_script =
    \\# nuclis completion for bash (3.2 and later): every Tab asks the binary.
    \\_nuclis() {
    \\    local cur="${COMP_WORDS[COMP_CWORD]}" out line i
    \\    local -a args
    \\    # A loop, not a slice: bash 3.2 joins a quoted slice under another IFS.
    \\    for ((i = 1; i <= COMP_CWORD; i++)); do args[i-1]="${COMP_WORDS[i]}"; done
    \\    out=$(command nuclis __complete "${args[@]}" 2>/dev/null)
    \\    local IFS=$'\n'
    \\    COMPREPLY=()
    \\    case "$out" in
    \\        :file) COMPREPLY=($(compgen -f -- "$cur")); compopt -o filenames 2>/dev/null ;;
    \\        :dir) COMPREPLY=($(compgen -d -- "$cur")); compopt -o filenames 2>/dev/null ;;
    \\        *) for line in $out; do COMPREPLY+=("${line%%$'\t'*}"); done ;;
    \\    esac
    \\}
    \\complete -F _nuclis nuclis
    \\
;

const zsh_script =
    \\#compdef nuclis
    \\# nuclis completion for zsh: every Tab asks the binary.
    \\_nuclis() {
    \\    local -a lines described plain
    \\    local line
    \\    lines=("${(@f)$(command nuclis __complete "${(@)words[2,CURRENT]}" 2>/dev/null)}")
    \\    case "$lines[1]" in
    \\        :file) _files; return ;;
    \\        :dir) _files -/; return ;;
    \\    esac
    \\    for line in $lines; do
    \\        [[ -n $line ]] || continue
    \\        if [[ -n ${line#*$'\t'} ]]; then
    \\            described+=("${${line%%$'\t'*}//:/\\:}:${line#*$'\t'}")
    \\        else
    \\            plain+=("${line%%$'\t'*}")
    \\        fi
    \\    done
    \\    (( $#described )) && _describe -t values nuclis described
    \\    (( $#plain )) && compadd -a plain
    \\}
    \\if [[ "$funcstack[1]" = "_nuclis" ]]; then
    \\    _nuclis "$@"
    \\else
    \\    compdef _nuclis nuclis
    \\fi
    \\
;

// ----- tests -----

const testing = std.testing;
const cli = @import("cli.zig");
const help = @import("help.zig");

fn answerFor(arena: Allocator, words: []const []const u8, state: State) !Answer {
    return candidates(arena, resolve(words), words[words.len - 1], state);
}

fn texts(arena: Allocator, answer: Answer) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    for (answer.lines) |c| try list.append(arena, c.text);
    return list.items;
}

fn expectTexts(arena: Allocator, words: []const []const u8, state: State, expected: []const []const u8) !void {
    const got = try texts(arena, try answerFor(arena, words, state));
    if (got.len != expected.len) {
        std.debug.print("{s}: {d} candidates, expected {d}\n", .{ words[words.len - 1], got.len, expected.len });
        for (got) |g| std.debug.print("  {s}\n", .{g});
        return error.TestUnexpectedResult;
    }
    for (got, expected) |g, e| try testing.expectEqualStrings(e, g);
}

test "commands, actions, flags, and fixed values complete from the table" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try expectTexts(a, &.{"ag"}, .{}, &.{"agent"});
    try expectTexts(a, &.{"--v"}, .{}, &.{"--version"});
    try expectTexts(a, &.{ "config", "" }, .{}, &.{ "init", "show", "set" });
    try expectTexts(a, &.{ "model", "p" }, .{}, &.{"pull"});
    // An optional action comes with the default action's flags.
    try expectTexts(a, &.{ "agent", "l" }, .{}, &.{"ls"});
    try expectTexts(a, &.{ "agent", "--th" }, .{}, &.{ "--think", "--thinking-budget" });
    try expectTexts(a, &.{ "agent", "--think", "" }, .{}, &.{ "off", "low", "medium", "high", "xhigh" });
    try expectTexts(a, &.{ "agent", "--think", "low", "--backend", "m" }, .{}, &.{"metal"});
    // A flag given once is not offered again; a repeatable one is.
    try expectTexts(a, &.{ "agent", "--think", "low", "--thi" }, .{}, &.{"--thinking-budget"});
    try expectTexts(a, &.{ "generate", "--image", "a.png", "--ima" }, .{}, &.{ "--image", "--image-max-tokens" });
    try expectTexts(a, &.{ "agent", "-p", "hi", "--prompt" }, .{}, &.{ "--prompt-file", "--prompt-profile" });
    // Flags are per action.
    try expectTexts(a, &.{ "agent", "ls", "--" }, .{}, &.{"--json"});
    try expectTexts(a, &.{ "model", "inspect", "x", "--" }, .{}, &.{ "--file", "--revision", "--json" });
    try expectTexts(a, &.{ "completion", "" }, .{}, &.{ "fish", "bash", "zsh" });
    // Free text, numbers, and unknown commands complete to nothing.
    try expectTexts(a, &.{ "agent", "--prompt", "" }, .{}, &.{});
    try expectTexts(a, &.{ "agent", "--ctx-size", "" }, .{}, &.{});
    try expectTexts(a, &.{ "nope", "" }, .{}, &.{});
    try expectTexts(a, &.{ "model", "nope", "" }, .{}, &.{});
}

test "paths go to the shell, and the user's state fills models, sessions, and keys" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try testing.expect(try answerFor(a, &.{ "agent", "--prompt-file", "" }, .{}) == .file);
    try testing.expect(try answerFor(a, &.{ "generate", "--trace-dir", "o" }, .{}) == .dir);
    try testing.expect(try answerFor(a, &.{ "agent", "--model", "./m" }, .{}) == .file);
    const state: State = .{
        .models = &.{ .{ .text = "mine", .description = "o/r/f.gguf" }, .{ .text = "qwen3.8-27b" } },
        .registry = &.{"mine"},
        .sessions = &.{ .{ .text = "abc123" }, .{ .text = "def456" } },
    };
    try expectTexts(a, &.{ "agent", "--model", "" }, state, &.{ "mine", "qwen3.8-27b" });
    try expectTexts(a, &.{ "model", "pull", "q" }, state, &.{"qwen3.8-27b"});
    try expectTexts(a, &.{ "agent", "--resume", "" }, state, &.{ "abc123", "def456" });
    // `--resume` alone is complete: a dash starts a flag, not an id.
    try expectTexts(a, &.{ "agent", "--resume", "--thinking" }, state, &.{"--thinking-budget"});
    try expectTexts(a, &.{ "agent", "--resume", "--think", "o" }, state, &.{"off"});
    try expectTexts(a, &.{ "config", "set", "agent.th" }, state, &.{ "agent.think", "agent.theme", "agent.thinking_budget" });
    try expectTexts(a, &.{ "config", "set", "models.mine.pro" }, state, &.{"models.mine.profile"});
    try expectTexts(a, &.{ "config", "set", "agent.think", "h" }, state, &.{"high"});
    try expectTexts(a, &.{ "config", "set", "engine.model", "m" }, state, &.{"mine"});
    try expectTexts(a, &.{ "config", "set", "models.mine.profile", "gemma4_" }, state, &.{"gemma4_e"});
    try expectTexts(a, &.{ "model", "pull", "x", "--with", "mmproj," }, state, &.{ "mmproj,mtp", "mmproj,imatrix" });
}

test "the answer is one line per candidate, a description never breaks one" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    try write(&buffer.writer, .{ .lines = &.{ .{ .text = "a", .description = "one\ttwo\nthree" }, .{ .text = "b" } } });
    try testing.expectEqualStrings("a\tone two three\nb\t\n", buffer.written());
    buffer.clearRetainingCapacity();
    try write(&buffer.writer, .file);
    try testing.expectEqualStrings(":file\n", buffer.written());
}

/// A value `parseArgs` accepts for a flag or positional of this kind.
fn sample(value: Value) []const u8 {
    return switch (value) {
        .none => unreachable,
        .text, .file, .dir, .session, .model => "x",
        .number => "1",
        .choice => |c| c[0],
        .repo => "owner/repo",
        .config_key => "agent.think",
        .config_value => "low",
        .roles => "mmproj",
    };
}

test "every flag in the table is one the parser takes for that command" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for (&commands) |*command| for (command.actions) |*action| for (action.flags) |f| {
        var args: std.ArrayList([]const u8) = .empty;
        try args.append(a, command.name);
        if (action.name.len > 0) try args.append(a, action.name);
        for (action.positionals) |p| try args.append(a, sample(p));
        try args.append(a, f.name);
        if (f.value != .none) try args.append(a, sample(f.value));
        // What the command needs besides the flag under test.
        const sources = [_][]const u8{ "--prompt", "--prompt-file", "--prompt-tokens" };
        if (contains(&.{ "generate", "bench", "tokenize" }, command.name) and !contains(&sources, f.name) and !std.mem.eql(u8, f.name, "--kernel-stats")) try args.appendSlice(a, &.{ "--prompt", "x" });
        if (std.mem.eql(u8, command.name, "eval") and !std.mem.eql(u8, f.name, "--file")) try args.appendSlice(a, &.{ "--file", "x" });
        if (std.mem.eql(u8, command.name, "agent") and action.name.len == 0 and contains(&.{ "--json", "--session", "--print" }, f.name)) try args.appendSlice(a, &.{ "-p", "x" });
        if (std.mem.eql(u8, action.name, "init") and !std.mem.eql(u8, f.name, "--discover")) try args.append(a, "--discover");
        if (std.mem.eql(u8, f.name, "--profile") and std.mem.eql(u8, action.name, "pull")) try args.appendSlice(a, &.{ "--register", "x" });
        _ = cli.parseArgs(args.items) catch |err| {
            std.debug.print("{s}: {s}\n", .{ try std.mem.join(a, " ", args.items), @errorName(err) });
            return err;
        };
        if (f.short) |s| {
            for (args.items) |*arg| if (std.mem.eql(u8, arg.*, f.name)) {
                arg.* = s;
            };
            _ = try cli.parseArgs(args.items);
        }
    };
    _ = try cli.parseArgs(&.{ "config", "set", "agent.think", "low" });
    _ = try cli.parseArgs(&.{ "completion", "zsh" });
}

fn inTable(word: []const u8) bool {
    for (&commands) |*c| for (c.actions) |*act| if (act.flag(word) != null) return true;
    for (&globals) |*g| if (std.mem.eql(u8, g.name, word) or std.mem.eql(u8, g.short orelse "", word)) return true;
    return false;
}

test "every flag the parser spells is in the table" {
    // The parser's own source, up to its tests: a flag literal added there
    // and not here fails this.
    const source = @embedFile("cli.zig");
    const code = source[0 .. std.mem.indexOf(u8, source, "\ntest \"") orelse source.len];
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, code, i, "\"-")) |at| : (i = at + 2) {
        const end = std.mem.indexOfScalarPos(u8, code, at + 1, '"') orelse break;
        const literal = code[at + 1 .. end];
        if (literal.len < 2 or !std.ascii.isAlphabetic(literal[literal.len - 1])) continue;
        if (!inTable(literal)) {
            std.debug.print("the parser takes {s}; the completion table does not have it\n", .{literal});
            return error.TestUnexpectedResult;
        }
    }
}

test "every flag in the table is on its command's help page" {
    for (&commands) |*command| {
        const topic = std.meta.stringToEnum(help.Topic, command.name) orelse return error.TestUnexpectedResult;
        var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
        defer buffer.deinit();
        try help.write(&buffer.writer, .none, topic, "0.0.0-test");
        for (command.actions) |action| {
            if (action.name.len > 0) try testing.expect(std.mem.indexOf(u8, buffer.written(), action.name) != null);
            for (action.flags) |f| if (std.mem.indexOf(u8, buffer.written(), f.name) == null) {
                std.debug.print("the {s} page lacks {s}\n", .{ command.name, f.name });
                return error.TestUnexpectedResult;
            };
        }
    }
}

test "the scripts call the hidden command and handle both directives" {
    inline for (@typeInfo(Shell).@"enum".fields) |field| {
        const text = script(@enumFromInt(field.value));
        try testing.expect(std.mem.indexOf(u8, text, "nuclis __complete") != null);
        try testing.expect(std.mem.indexOf(u8, text, ":file") != null and std.mem.indexOf(u8, text, ":dir") != null);
        try testing.expect(std.mem.count(u8, text, "\n") <= 40);
    }
}
