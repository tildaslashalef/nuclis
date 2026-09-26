//! CLI parsing and command dispatch. The reusable library owns GGUF validation;
//! this module supplies file paths, an allocator, Io, and an output writer.
const std = @import("std");
const inference = @import("inference");
const paths = @import("paths.zig");
const inspection = @import("inspect.zig");
const validation = @import("validate.zig");
const generate = @import("generate.zig");
const bench = @import("bench.zig");
const discover = @import("discover.zig");
const tokenize = @import("tokenize.zig");
const eval = @import("eval.zig");
const engine = @import("engine.zig");
const config = @import("config.zig");
const agent = @import("agent/root.zig");
const model = @import("model.zig");
const style = @import("tui/style.zig");
const catalog = @import("catalog.zig");
const help_text = @import("help.zig");

// Fed from build.zig.zon through the build_options module (see build.zig);
// never edit a version string here.
pub const version = @import("build_options").version;
pub const Diagnostic = config.Diagnostic;
pub const Options = struct {
    command: enum { help, version, inspect, validate, generate, bench, tokenize, eval, agent, config, model },
    /// Which command's page `--help` asked for; null is the overview.
    help_topic: ?help_text.Topic = null,
    config_action: enum { init, show, set } = .show,
    /// `config init --discover [--dry-run]`: register the runnable files
    /// the catalogue does not name.
    discover: bool = false,
    dry_run: bool = false,
    /// `config set <key> <value>`.
    set_key: []const u8 = "",
    set_value: []const u8 = "",
    model_action: enum { pull, ls, inspect } = .ls,
    pull: model.PullOptions = .{},
    /// The command-line layer of the configuration: only what was stated,
    /// so `config.resolve` can put it above the file.
    flags: config.Flags = .{},
    generation: generate.Options = .{},
    benchmark: bench.Options = .{},
    evaluation: eval.Options = .{},
    /// `agent --print`: one turn without a terminal.
    print: agent.print_mode.Options = .{},
    printing: bool = false,
    /// `agent --resume [<id>]`: replay a saved session before the first turn;
    /// `resume.latest` when no id followed the flag.
    resume_id: ?[]const u8 = null,
    /// `agent ls` lists the workspace's sessions instead of running one.
    agent_action: enum { run, ls } = .run,
    /// `agent --system-prompt <path>`: the file's text replaces the built
    /// prompt sections (a tuning aid; the instructions file still follows).
    system_prompt_file: ?[]const u8 = null,
    model: ?[]const u8 = null,
    json: bool = false,
};

pub fn parseArgs(args: []const []const u8) !Options {
    if (args.len == 0) return .{ .command = .help };
    if (args.len == 1 and (std.mem.eql(u8, args[0], "--help") or std.mem.eql(u8, args[0], "-h")))
        return .{ .command = .help };
    if (args.len == 1 and std.mem.eql(u8, args[0], "--version"))
        return .{ .command = .version };
    const command: @FieldType(Options, "command") = if (std.mem.eql(u8, args[0], "inspect"))
        .inspect
    else if (std.mem.eql(u8, args[0], "validate"))
        .validate
    else if (std.mem.eql(u8, args[0], "generate"))
        .generate
    else if (std.mem.eql(u8, args[0], "bench"))
        .bench
    else if (std.mem.eql(u8, args[0], "tokenize"))
        .tokenize
    else if (std.mem.eql(u8, args[0], "eval"))
        .eval
    else if (std.mem.eql(u8, args[0], "agent"))
        .agent
    else if (std.mem.eql(u8, args[0], "config"))
        .config
    else if (std.mem.eql(u8, args[0], "model"))
        .model
    else
        return error.UnknownCommand;
    // `nuclis <command> --help` is that command's page; the bare `--help`
    // above is the overview.
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        return .{ .command = .help, .help_topic = std.meta.stringToEnum(help_text.Topic, args[0]) };
    }
    if (command == .model) return parseModelArgs(args[1..]);
    var options: Options = .{ .command = command };
    var i: usize = 1;
    if (command == .agent and args.len >= 2 and std.mem.eql(u8, args[1], "ls")) {
        options.agent_action = .ls;
        i = 2;
    }
    if (command == .config) {
        // The action is positional and required: `config init` | `config show`.
        if (args.len < 2) return error.MissingConfigAction;
        options.config_action = std.meta.stringToEnum(@FieldType(Options, "config_action"), args[1]) orelse return error.UnknownConfigAction;
        i = 2;
        if (options.config_action == .set) {
            // Two positionals, the key and its value; a value may look like
            // anything, so it is never mistaken for a flag.
            if (args.len < 4) return error.MissingSetArguments;
            options.set_key = args[2];
            options.set_value = args[3];
            i = 4;
        }
    }
    const generates = command == .generate or command == .bench or command == .agent;
    const samples = command == .generate or command == .agent;
    const prompts = command == .generate or command == .bench or command == .tokenize;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--json")) {
            if (options.json) return error.DuplicateOption;
            if (command == .config and options.config_action == .set) return error.UnknownOption;
            options.json = true;
        } else if (command == .config and options.config_action == .init and std.mem.eql(u8, args[i], "--discover")) {
            if (options.discover) return error.DuplicateOption;
            options.discover = true;
        } else if (command == .config and options.config_action == .init and std.mem.eql(u8, args[i], "--dry-run")) {
            if (options.dry_run) return error.DuplicateOption;
            options.dry_run = true;
        } else if (command != .config and std.mem.eql(u8, args[i], "--model")) {
            if (options.model != null) return error.DuplicateOption;
            i += 1;
            if (i == args.len or args[i].len == 0 or std.mem.startsWith(u8, args[i], "--"))
                return error.MissingModelPath;
            options.model = args[i];
        } else if (command == .agent and std.mem.eql(u8, args[i], "--print")) {
            if (options.printing) return error.DuplicateOption;
            options.printing = true;
        } else if (command == .agent and std.mem.eql(u8, args[i], "--system-prompt")) {
            if (options.system_prompt_file != null) return error.DuplicateOption;
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            options.system_prompt_file = args[i];
        } else if (command == .agent and std.mem.eql(u8, args[i], "--resume")) {
            if (options.resume_id != null) return error.DuplicateOption;
            // The id is optional: alone, the flag means the newest session.
            if (i + 1 == args.len or args[i + 1].len == 0 or std.mem.startsWith(u8, args[i + 1], "-")) {
                options.resume_id = agent.resume_mod.latest;
            } else {
                i += 1;
                options.resume_id = args[i];
            }
        } else if (command == .agent and (std.mem.eql(u8, args[i], "-p") or std.mem.eql(u8, args[i], "--prompt") or std.mem.eql(u8, args[i], "--prompt-file") or std.mem.eql(u8, args[i], "--session"))) {
            const flag = args[i];
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            const value = args[i];
            if (std.mem.eql(u8, flag, "--session")) {
                if (options.print.session != null) return error.DuplicateOption;
                options.print.session = value;
            } else if (std.mem.eql(u8, flag, "--prompt-file")) {
                if (options.print.prompt != null or options.print.prompt_file != null) return error.ConflictingPromptSources;
                options.print.prompt_file = value;
                options.printing = true;
            } else {
                // `-p <text>` is the short form of `--print --prompt <text>`.
                if (options.print.prompt != null or options.print.prompt_file != null) return error.ConflictingPromptSources;
                options.print.prompt = value;
                options.printing = true;
            }
        } else if (prompts and std.mem.eql(u8, args[i], "--raw")) {
            if (options.generation.raw or options.benchmark.raw) return error.DuplicateOption;
            options.generation.raw = true;
            options.benchmark.raw = true;
        } else if (command == .bench and std.mem.eql(u8, args[i], "--profile")) {
            if (options.benchmark.profile) return error.DuplicateOption;
            options.benchmark.profile = true;
        } else if (command == .bench and std.mem.eql(u8, args[i], "--unfused-norms")) {
            if (options.benchmark.unfused_norms) return error.DuplicateOption;
            options.benchmark.unfused_norms = true;
        } else if (command == .eval) {
            const flag = args[i];
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            const value = args[i];
            const e = &options.evaluation;
            const f = &options.flags;
            if (std.mem.eql(u8, flag, "--file")) {
                if (e.file != null) return error.DuplicateOption;
                e.file = value;
            } else if (std.mem.eql(u8, flag, "--reference")) {
                if (e.reference != null) return error.DuplicateOption;
                e.reference = value;
            } else if (std.mem.eql(u8, flag, "--ctx-size")) {
                // The window, not the configuration's context: an evaluation
                // is comparable from its command line alone.
                if (e.ctx != null) return error.DuplicateOption;
                e.ctx = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--chunks")) {
                if (e.chunks != null) return error.DuplicateOption;
                const n = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
                if (n == 0) return error.InvalidNumber;
                e.chunks = n;
            } else if (std.mem.eql(u8, flag, "--backend")) {
                if (f.backend != null) return error.DuplicateOption;
                f.backend = std.meta.stringToEnum(engine.Backend, value) orelse return error.UnsupportedBackend;
            } else if (std.mem.eql(u8, flag, "--kv")) {
                if (f.kv != null) return error.DuplicateOption;
                f.kv = std.meta.stringToEnum(config.KvPrecision, value) orelse return error.UnsupportedKvPrecision;
            } else if (std.mem.eql(u8, flag, "--prompt-profile")) {
                if (f.prompt_profile != null) return error.DuplicateOption;
                f.prompt_profile = std.meta.stringToEnum(inference.profiles.Profile, value) orelse return error.UnknownPromptProfile;
            } else return error.UnknownOption;
        } else if (generates or command == .tokenize) {
            const flag = args[i];
            i += 1;
            if (i == args.len) return error.MissingOptionValue;
            const value = args[i];
            const g = &options.generation;
            const b = &options.benchmark;
            const f = &options.flags;
            // Tokenize takes only the prompt and the rendering effort: no
            // backend, budget, or sampling can change what it reports.
            const tokenizes = command == .tokenize and !std.mem.eql(u8, flag, "--prompt") and !std.mem.eql(u8, flag, "--prompt-file") and !std.mem.eql(u8, flag, "--think") and !std.mem.eql(u8, flag, "--prompt-profile");
            if (tokenizes) return error.UnknownOption;
            if (std.mem.eql(u8, flag, "--backend")) {
                if (f.backend != null) return error.DuplicateOption;
                f.backend = std.meta.stringToEnum(engine.Backend, value) orelse return error.UnsupportedBackend;
            } else if (std.mem.eql(u8, flag, "--prompt")) {
                if (g.prompt != null) return error.DuplicateOption;
                g.prompt = value;
                b.prompt = value;
            } else if (std.mem.eql(u8, flag, "--prompt-file")) {
                if (g.prompt_file != null) return error.DuplicateOption;
                g.prompt_file = value;
                b.prompt_file = value;
            } else if (command == .bench and std.mem.eql(u8, flag, "--prompt-tokens")) {
                if (b.prompt_tokens != null) return error.DuplicateOption;
                b.prompt_tokens = value;
            } else if (command == .generate and std.mem.eql(u8, flag, "--prompt-tokens")) {
                if (g.prompt_tokens != null) return error.DuplicateOption;
                g.prompt_tokens = value;
            } else if (std.mem.eql(u8, flag, "--max-tokens")) {
                if (f.max_tokens != null) return error.DuplicateOption;
                f.max_tokens = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--ctx-size")) {
                if (f.ctx_size != null) return error.DuplicateOption;
                f.ctx_size = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--kv")) {
                if (f.kv != null) return error.DuplicateOption;
                f.kv = std.meta.stringToEnum(config.KvPrecision, value) orelse return error.UnsupportedKvPrecision;
            } else if (command == .bench and std.mem.eql(u8, flag, "--repeat")) {
                if (b.repeat != null) return error.DuplicateOption;
                b.repeat = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
            } else if (command == .bench and std.mem.eql(u8, flag, "--warmup")) {
                if (b.warmup != null) return error.DuplicateOption;
                b.warmup = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--prompt-profile")) {
                if (f.prompt_profile != null) return error.DuplicateOption;
                f.prompt_profile = std.meta.stringToEnum(inference.profiles.Profile, value) orelse return error.UnknownPromptProfile;
            } else if ((samples or command == .tokenize) and std.mem.eql(u8, flag, "--think")) {
                if (f.think != null) return error.DuplicateOption;
                f.think = std.meta.stringToEnum(inference.profiles.Effort, value) orelse return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--speculative")) {
                if (f.speculative != null) return error.DuplicateOption;
                f.speculative = parseOnOff(value) orelse return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--draft-length")) {
                if (f.draft_length != null) return error.DuplicateOption;
                const n = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
                if (n == 0 or n > config.max_draft_length) return error.InvalidNumber;
                f.draft_length = n;
            } else if (command == .agent and std.mem.eql(u8, flag, "--thinking-budget")) {
                if (f.thinking_budget != null) return error.DuplicateOption;
                const n = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
                if (n > config.max_output_tokens) return error.InvalidNumber;
                f.thinking_budget = n;
            } else if ((command == .generate or command == .agent) and std.mem.eql(u8, flag, "--image-max-tokens")) {
                if (f.image_max_tokens != null) return error.DuplicateOption;
                const cap = config.ImageMaxTokens.parse(value) orelse return error.InvalidNumber;
                if (cap.count()) |n| if (n == 0 or n > config.max_image_tokens) return error.InvalidNumber;
                f.image_max_tokens = cap;
            } else if (std.mem.eql(u8, flag, "--temperature")) {
                if (f.sampling.temperature != null) return error.DuplicateOption;
                f.sampling.temperature = std.fmt.parseFloat(f32, value) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--top-p")) {
                if (f.sampling.top_p != null) return error.DuplicateOption;
                f.sampling.top_p = std.fmt.parseFloat(f32, value) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--top-k")) {
                if (f.sampling.top_k != null) return error.DuplicateOption;
                f.sampling.top_k = std.fmt.parseInt(usize, value, 10) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--min-p")) {
                if (f.sampling.min_p != null) return error.DuplicateOption;
                f.sampling.min_p = std.fmt.parseFloat(f32, value) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--presence-penalty")) {
                if (f.sampling.presence_penalty != null) return error.DuplicateOption;
                f.sampling.presence_penalty = std.fmt.parseFloat(f32, value) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--repetition-penalty")) {
                if (f.sampling.repetition_penalty != null) return error.DuplicateOption;
                f.sampling.repetition_penalty = std.fmt.parseFloat(f32, value) catch return error.InvalidNumber;
            } else if (std.mem.eql(u8, flag, "--seed")) {
                if (g.seed != null) return error.DuplicateOption;
                g.seed = std.fmt.parseInt(u64, value, 10) catch return error.InvalidNumber;
                b.seed = g.seed;
            } else if (command == .generate and std.mem.eql(u8, flag, "--image")) {
                if (g.image_count >= g.images.len) return error.TooManyImages;
                g.images[g.image_count] = value;
                g.image_count += 1;
            } else if (command == .generate and std.mem.eql(u8, flag, "--logits")) {
                if (g.logits_path != null) return error.DuplicateOption;
                g.logits_path = value;
            } else if (command == .generate and std.mem.eql(u8, flag, "--trace-dir")) {
                if (g.trace_dir != null) return error.DuplicateOption;
                g.trace_dir = value;
            } else return error.UnknownOption;
        } else return error.UnknownOption;
    }
    if (command == .agent) {
        // A terminal surface has no JSON form, and a printed turn needs
        // something to print.
        if (options.json and !options.printing and options.agent_action != .ls) return error.UnknownOption;
        if (options.printing and options.print.prompt == null and options.print.prompt_file == null) return error.MissingPrompt;
        if (options.print.session != null and !options.printing) return error.UnknownOption;
        options.print.json = options.json;
        options.print.seed = options.generation.seed;
        options.print.resume_id = options.resume_id;
        options.print.system_prompt = options.system_prompt_file;
    }
    if (command == .config and options.config_action == .init and !options.discover and (options.json or options.dry_run)) return error.UnknownOption;
    if (command == .eval and options.evaluation.file == null) return error.MissingTextFile;
    const token_prompt = options.benchmark.prompt_tokens != null or options.generation.prompt_tokens != null;
    const sources = @as(u8, @intFromBool(options.generation.prompt != null)) + @intFromBool(options.generation.prompt_file != null) + @intFromBool(token_prompt);
    if (prompts and sources == 0) return error.MissingPrompt;
    // Token prompts are fed as given: --raw would suggest a rendering choice that does not exist.
    if (sources > 1 or (token_prompt and (options.benchmark.raw or options.generation.raw))) return error.ConflictingPromptSources;
    return options;
}

/// `on`/`off` as a boolean flag value (`--speculative on`), or null for
/// anything else so the caller reports `InvalidNumber`.
fn parseOnOff(value: []const u8) ?bool {
    if (std.mem.eql(u8, value, "on")) return true;
    if (std.mem.eql(u8, value, "off")) return false;
    return null;
}

/// `model pull <name|owner/repo> [flags]` | `model inspect <name|owner/repo>
/// [--file] [--revision] [--json]` | `model ls [--json]`: the action and the
/// repository are positional; the flags are the command's own, none of the
/// generation flags apply, and `--model` in particular does not (there is
/// no model to open).
fn parseModelArgs(args: []const []const u8) !Options {
    if (args.len == 0) return error.MissingModelAction;
    var options: Options = .{ .command = .model };
    options.model_action = std.meta.stringToEnum(@FieldType(Options, "model_action"), args[0]) orelse return error.UnknownModelAction;
    var i: usize = 1;
    if (options.model_action != .ls) {
        if (i == args.len or std.mem.startsWith(u8, args[i], "--")) return error.MissingRepository;
        options.pull.repo = args[i];
        i += 1;
    }
    const p = &options.pull;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--json")) {
            if (options.json) return error.DuplicateOption;
            options.json = true;
            continue;
        }
        if (options.model_action == .ls) return error.UnknownOption;
        const pulls = options.model_action == .pull;
        if (pulls and std.mem.eql(u8, flag, "--force")) {
            if (p.force) return error.DuplicateOption;
            p.force = true;
            continue;
        }
        if (pulls and std.mem.eql(u8, flag, "--all")) {
            if (p.all) return error.DuplicateOption;
            p.all = true;
            continue;
        }
        if (!pulls and !std.mem.eql(u8, flag, "--file") and !std.mem.eql(u8, flag, "--revision")) return error.UnknownOption;
        i += 1;
        if (i == args.len) return error.MissingOptionValue;
        const value = args[i];
        if (std.mem.eql(u8, flag, "--file")) {
            if (p.file != null) return error.DuplicateOption;
            p.file = value;
        } else if (std.mem.eql(u8, flag, "--revision")) {
            if (p.revision != null) return error.DuplicateOption;
            p.revision = value;
        } else if (pulls and std.mem.eql(u8, flag, "--role")) {
            if (p.role != null) return error.DuplicateOption;
            p.role = std.meta.stringToEnum(model.Role, value) orelse return error.UnknownRole;
        } else if (pulls and std.mem.eql(u8, flag, "--register")) {
            if (p.register != null) return error.DuplicateOption;
            if (value.len == 0 or std.mem.startsWith(u8, value, "--")) return error.MissingOptionValue;
            p.register = value;
        } else if (pulls and std.mem.eql(u8, flag, "--profile")) {
            if (p.profile != null) return error.DuplicateOption;
            p.profile = std.meta.stringToEnum(config.Profile, value) orelse return error.UnknownPromptProfile;
        } else if (pulls and std.mem.eql(u8, flag, "--with")) {
            if (p.with.count() != 0) return error.DuplicateOption;
            var roles = std.mem.splitScalar(u8, value, ',');
            while (roles.next()) |name| {
                const role = std.meta.stringToEnum(model.Role, name) orelse return error.UnknownRole;
                if (role == .main) return error.UnknownRole;
                p.with.insert(role);
            }
            if (p.with.count() == 0) return error.MissingOptionValue;
        } else return error.UnknownOption;
    }
    if (p.profile != null and p.register == null) return error.ConflictingOptions;
    return options;
}

/// `diag` receives the human-readable reason when a configuration error
/// unwinds (the key path and the rule it broke); `main` prints it beside
/// the error name. Nothing here logs, so the function is testable.
pub fn run(alloc: std.mem.Allocator, io: std.Io, environ: *const std.process.Environ.Map, options: Options, out: *std.Io.Writer, diag: *config.Diagnostic) !void {
    // Text reports are styled only on a terminal; `--json` never is.
    const sty: style.Style = if (options.json) .none else style.Style.detect(environ, io, std.Io.File.stdout());
    switch (options.command) {
        .help => return help_text.write(out, sty, options.help_topic, version),
        .version => return out.writeAll("nuclis " ++ version ++ "\n"),
        else => {},
    }
    // Every other command starts from the user root and the configuration
    // file beneath it (defaults when there is none); `--model` alone works
    // without a root.
    const root = try paths.root(alloc, environ.get("NUCLIS_HOME"), environ.get("HOME"));
    defer if (root) |dir| alloc.free(dir);
    const config_path: ?[]u8 = if (root) |dir| try paths.configPath(alloc, dir) else null;
    defer if (config_path) |p| alloc.free(p);
    if (options.command == .model) {
        // Needs the root (the models directory) and nothing from the
        // configuration file: a broken nuclis.json must not block a
        // download. The one exception is a pull by registry name, which
        // only the file can resolve.
        const dir = root orelse return error.MissingHome;
        switch (options.model_action) {
            .ls => {
                // The registry names what is listed; a file that fails to
                // load leaves the listing unannotated rather than blocked.
                var loaded: ?config.Loaded = config.load(alloc, io, .cwd(), config_path, diag) catch |err| blk: {
                    try out.print("{s}{s}: {s} ({s}); listing without registry names{s}\n", .{ sty.on(.warning), config_path.?, diag.message(), @errorName(err), sty.off() });
                    break :blk null;
                };
                defer if (loaded) |*l| l.deinit();
                return model.ls(alloc, io, dir, if (loaded) |l| l.config.models else .{}, options.json, out, sty);
            },
            .inspect => return model.inspect(alloc, io, environ, options.pull, options.json, out, sty, diag),
            .pull => {
                var pull_options = options.pull;
                pull_options.config_path = config_path;
                var loaded: ?config.Loaded = null;
                defer if (loaded) |*l| l.deinit();
                if (model.isRegistryName(options.pull.repo)) {
                    if (options.pull.register != null) {
                        diag.set("{s} is a registry entry already; --register names a new one for an owner/repo pull", .{options.pull.repo});
                        return error.ConflictingOptions;
                    }
                    loaded = try config.load(alloc, io, .cwd(), config_path, diag);
                    const entry = loaded.?.config.models.find(options.pull.repo) orelse {
                        diag.set("{s} is not a catalogue name, an entry of the models registry in {s}, or an owner/repo id", .{ options.pull.repo, config_path.? });
                        return error.UnknownModel;
                    };
                    pull_options = try model.fromRegistry(options.pull, entry, diag);
                }
                return model.pull(alloc, io, environ, dir, pull_options, options.json, out, sty, diag);
            },
        }
    }
    if (options.command == .agent and options.agent_action == .ls) {
        const dir = root orelse return error.MissingHome;
        const cwd = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", alloc);
        defer alloc.free(cwd);
        return agent.resume_mod.ls(alloc, io, dir, cwd, options.json, out, sty);
    }
    if (options.command == .config) {
        const file = config_path orelse return error.MissingHome;
        switch (options.config_action) {
            .init => {
                const created = try config.init(io, .cwd(), file);
                if (!options.discover) {
                    switch (created) {
                        .created => {
                            try out.print("{s}wrote{s} {s}{s}{s}\n", .{ sty.on(.success), sty.off(), sty.on(.code), file, sty.off() });
                            try renderStart(alloc, io, root.?, out, sty);
                        },
                        .exists => try out.print("{s}{s}{s} exists; not overwritten {s}(`nuclis config show` prints it; move it away to start over){s}\n", .{ sty.on(.code), file, sty.off(), sty.on(.dim), sty.off() }),
                    }
                    return;
                }
                if (!options.json and created == .created) try out.print("{s}wrote{s} {s}{s}{s}\n", .{ sty.on(.success), sty.off(), sty.on(.code), file, sty.off() });
                var arena_state = std.heap.ArenaAllocator.init(alloc);
                defer arena_state.deinit();
                const arena = arena_state.allocator();
                const current = try config.readText(alloc, io, .cwd(), file, diag);
                defer if (current) |c| alloc.free(c);
                var loaded = try config.fromText(alloc, current orelse return error.FileNotFound, file, diag);
                defer loaded.deinit();
                var report = try discover.discover(arena, alloc, io, root.?, loaded.config.models);
                report.config_file = file;
                report.dry_run = options.dry_run;
                if (!options.dry_run and report.registered.len > 0) {
                    const text = try config.registerDiscovered(alloc, current, file, try discover.toEntries(arena, report.registered), diag);
                    defer alloc.free(text);
                    try config.write(io, .cwd(), file, text);
                    report.written = true;
                }
                try report.render(out, options.json, sty);
            },
            .show => {
                var loaded = try config.load(alloc, io, .cwd(), file, diag);
                defer loaded.deinit();
                try config.show(&loaded, out, options.json, sty);
            },
            .set => {
                const current = try config.readText(alloc, io, .cwd(), file, diag);
                defer if (current) |c| alloc.free(c);
                var edit = try config.set(alloc, current, file, options.set_key, options.set_value, diag);
                defer edit.deinit(alloc);
                // The schema accepted the value; the model must also be
                // reachable, or the next command fails on it.
                if (std.mem.eql(u8, options.set_key, "engine.model")) {
                    var loaded = try config.fromText(alloc, edit.text, file, diag);
                    defer loaded.deinit();
                    const path = try paths.modelPath(alloc, null, loaded.config.engine.model, root, loaded.config.models);
                    defer alloc.free(path);
                    std.Io.Dir.cwd().access(io, path, .{}) catch {
                        diag.set("engine.model {s}: no file at {s} (a registry entry, a catalogue name, or a path; `nuclis model ls` shows what is present)", .{ options.set_value, path });
                        return error.UnknownModel;
                    };
                }
                try config.write(io, .cwd(), file, edit.text);
                try out.print("{s}set{s} {s}{s}{s} = {s}{s}{s} in {s}{s}{s}", .{ sty.on(.success), sty.off(), sty.on(.label), options.set_key, sty.off(), sty.on(.keyword), options.set_value, sty.off(), sty.on(.code), file, sty.off() });
                if (edit.previous) |p| try out.print(" {s}(was {s}){s}", .{ sty.on(.dim), p, sty.off() });
                try out.writeByte('\n');
            },
        }
        return;
    }
    var loaded = try config.load(alloc, io, .cwd(), config_path, diag);
    defer loaded.deinit();
    const path = try paths.modelPath(alloc, options.model, loaded.config.engine.model, root, loaded.config.models);
    defer alloc.free(path);
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            diag.set("no model file at {s} (engine.model or --model; `nuclis model ls` shows what is present, `nuclis model pull <name>` fetches it)", .{path});
            return error.ModelFileNotFound;
        },
        else => return err,
    };
    // The engine reports an architecture without an adapter as a typed
    // error; the reason (which ids the tree has) is added here, where the
    // diagnostic lives.
    (switch (options.command) {
        .help, .version, .config, .model => unreachable,
        .generate => generate.run(alloc, io, path, config.resolve(&loaded, options.model, options.flags, .generate), options.generation, options.json, out),
        .bench => bench.run(alloc, io, path, config.resolve(&loaded, options.model, options.flags, .bench), options.benchmark, options.json, out, sty),
        .tokenize => blk: {
            const resolved = config.resolve(&loaded, options.model, options.flags, .generate);
            break :blk tokenize.run(alloc, io, path, resolved.think, resolved.forced_profile, options.generation, options.json, out, sty);
        },
        .eval => eval.run(alloc, io, path, config.resolve(&loaded, options.model, options.flags, .bench), options.evaluation, options.json, out, sty, diag),
        .agent => if (options.printing)
            agent.print_mode.run(alloc, io, environ, path, config.resolve(&loaded, options.model, options.flags, .agent), options.print, out, diag)
        else
            agent.run(alloc, io, environ, path, root, config.resolve(&loaded, options.model, options.flags, .agent), options.generation.seed, options.resume_id, options.system_prompt_file, out),
        .inspect, .validate => blk: {
            var document = try inference.gguf.open(alloc, io, path, .{});
            defer document.deinit();
            if (options.command == .validate) {
                const adapter = try selectAdapter(&document, diag);
                const summary = try inference.models.registry.validate(adapter, alloc, &document);
                try validation.render(path, summary, out, options.json, sty);
            } else {
                var histogram: [32]inspection.EncodingCount = undefined;
                const result = try inspection.snapshot(document, path, &histogram);
                try result.render(out, options.json, sty);
            }
            break :blk {};
        },
    }) catch |err| switch (err) {
        error.MetalNotEnabled => {
            diag.set("this binary was built without the Metal backend: rebuild with `zig build -Dmetal=true` (or `make metal`), or run with `--backend cpu`", .{});
            return err;
        },
        error.UnknownArchitecture => {
            if (diag.len == 0) diag.set("the model's general.architecture has no adapter; known: {s} (`nuclis model inspect` names the file's)", .{known_architectures});
            return err;
        },
        error.SessionNotFound => {
            if (std.mem.eql(u8, options.resume_id orelse "", agent.resume_mod.latest))
                diag.set("no saved session for this workspace yet (`nuclis agent ls` lists them under {s})", .{root orelse "~/.nuclis"})
            else
                diag.set("no saved session {s} for this workspace (`nuclis agent ls` and /resume list the sessions under {s})", .{ options.resume_id orelse "", root orelse "~/.nuclis" });
            return err;
        },
        else => return err,
    };
}

/// The registry's ids as one comma-separated list, joined at compile time
/// for the diagnostics above.
const known_architectures = blk: {
    var list: []const u8 = "";
    for (inference.models.known, 0..) |name, i| list = list ++ (if (i > 0) ", " else "") ++ name;
    break :blk list;
};

/// The registry's adapter for an opened file, or `UnknownArchitecture`
/// with the file's id and the known ones in the diagnostic.
fn selectAdapter(document: *const inference.gguf.Document, diag: *Diagnostic) !inference.models.Adapter {
    const architecture = document.string("general.architecture") orelse "";
    return inference.models.select(architecture) catch |err| {
        diag.set("no adapter for architecture \"{s}\"; known: {s}", .{ architecture, known_architectures });
        return err;
    };
}

/// After `config init`: what the file says, the state of every catalogue
/// model under the root, and the command to run next (a fresh root has
/// nothing pulled yet).
fn renderStart(alloc: std.mem.Allocator, io: std.Io, root: []const u8, out: *std.Io.Writer, sty: style.Style) !void {
    const defaults: config.Config = .{};
    const off = sty.off();
    try out.print("  {s}engine.model{s} {s}{s}{s} (catalogue name), backend {s}, ctx_size {d}; think {s} (generate) / {s} (agent)\n", .{ sty.on(.label), off, sty.on(.keyword), defaults.engine.model, off, @tagName(defaults.engine.backend), defaults.engine.ctx_size, @tagName(defaults.generation.think), @tagName(defaults.agent.think) });
    try out.print("  {s}models{s} {s}(the catalogue, registered in the file; `nuclis model ls` shows them){s}\n", .{ sty.on(.header), off, sty.on(.dim), off });
    const models = try model.modelsDir(alloc, root);
    defer alloc.free(models);
    var missing: ?*const catalog.Entry = null;
    for (&catalog.entries) |*entry| {
        const path = try catalog.localPath(alloc, models, entry, entry.file);
        defer alloc.free(path);
        const status = try catalog.status(alloc, io, path, entry.sha256, entry.size);
        if (status != .present and missing == null) missing = entry;
        var name_buffer: [64]u8 = undefined;
        try out.print("    {s}{s}{s} {s}{s:<10}{s} {s}{d}{s} bytes\n", .{ sty.on(.keyword), catalog.padName(&name_buffer, entry.name), off, sty.on(model.statusStyle(status)), @tagName(status), off, sty.on(.number), entry.size, off });
    }
    if (missing) |entry|
        try out.print("{s}next:{s} {s}nuclis model pull {s}{s} fetches the model ({s}--all{s} adds its companions), then {s}nuclis agent{s}\n", .{ sty.on(.bold), off, sty.on(.code), entry.name, off, sty.on(.code), off, sty.on(.code), off })
    else
        try out.print("{s}next:{s} every catalogue model is present; {s}nuclis agent{s}\n", .{ sty.on(.bold), off, sty.on(.code), off });
}

test "parse inspection and reject ambiguous or incomplete flags" {
    const options = try parseArgs(&.{ "inspect", "--json", "--model", "a.gguf" });
    try std.testing.expect(options.json);
    try std.testing.expectEqualStrings("a.gguf", options.model.?);
    try std.testing.expectError(error.MissingModelPath, parseArgs(&.{ "inspect", "--model", "--json" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "inspect", "--json", "--json" }));
    try std.testing.expectError(error.MissingPrompt, parseArgs(&.{"generate"}));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "inspect", "--typo" }));
}

test "agent print mode takes a prompt, JSON lines, and a session file" {
    const short = try parseArgs(&.{ "agent", "-p", "hello" });
    try std.testing.expect(short.printing);
    try std.testing.expectEqualStrings("hello", short.print.prompt.?);
    try std.testing.expect(!short.print.json);
    const long = try parseArgs(&.{ "agent", "--print", "--prompt-file", "p.txt", "--json", "--session", "s.jsonl", "--think", "low" });
    try std.testing.expect(long.printing and long.json and long.print.json);
    try std.testing.expectEqualStrings("p.txt", long.print.prompt_file.?);
    try std.testing.expectEqualStrings("s.jsonl", long.print.session.?);
    try std.testing.expectEqual(.low, long.flags.think.?);
    // The seed is shared with the other generating commands.
    try std.testing.expectEqual(@as(u64, 7), (try parseArgs(&.{ "agent", "-p", "x", "--seed", "7" })).print.seed.?);
    // An interactive agent has no JSON form, and a printed turn needs a prompt.
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "agent", "--json" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "agent", "--session", "s.jsonl" }));
    try std.testing.expectError(error.MissingPrompt, parseArgs(&.{ "agent", "--print" }));
    try std.testing.expectError(error.ConflictingPromptSources, parseArgs(&.{ "agent", "-p", "a", "--prompt-file", "b" }));
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "agent", "-p" }));
    // The other commands do not gain the agent's flags.
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "generate", "--prompt", "a", "--session", "s" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "bench", "--prompt", "a", "-p", "x" }));
    // `--resume` works on the interactive surface and in print mode alike.
    const resumed = try parseArgs(&.{ "agent", "--resume", "abcdef" });
    try std.testing.expectEqualStrings("abcdef", resumed.resume_id.?);
    try std.testing.expect(!resumed.printing);
    const print_resume = try parseArgs(&.{ "agent", "-p", "hi", "--resume", "abcdef" });
    try std.testing.expectEqualStrings("abcdef", print_resume.print.resume_id.?);
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "agent", "--resume", "a", "--resume", "b" }));
    // Alone, `--resume` is the newest session; a following flag is not an id.
    try std.testing.expectEqualStrings(agent.resume_mod.latest, (try parseArgs(&.{ "agent", "--resume" })).resume_id.?);
    const latest_print = try parseArgs(&.{ "agent", "--resume", "-p", "hi" });
    try std.testing.expectEqualStrings(agent.resume_mod.latest, latest_print.print.resume_id.?);
    try std.testing.expectEqualStrings("hi", latest_print.print.prompt.?);
    // `agent ls` lists sessions; `--json` applies.
    const listing = try parseArgs(&.{ "agent", "ls", "--json" });
    try std.testing.expect(listing.agent_action == .ls);
    try std.testing.expect(listing.json);
    try std.testing.expect((try parseArgs(&.{"agent"})).agent_action == .run);
    // A flag after a bare `--resume` is parsed as itself: print mode still wants its prompt.
    try std.testing.expectError(error.MissingPrompt, parseArgs(&.{ "agent", "--resume", "--print" }));
    // `--system-prompt` reaches both modes; it needs a value and is given once.
    const tuned = try parseArgs(&.{ "agent", "-p", "hi", "--system-prompt", "p.txt" });
    try std.testing.expectEqualStrings("p.txt", tuned.print.system_prompt.?);
    try std.testing.expectEqualStrings("p.txt", (try parseArgs(&.{ "agent", "--system-prompt", "p.txt" })).system_prompt_file.?);
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "agent", "--system-prompt" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "agent", "--system-prompt", "a", "--system-prompt", "b" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "generate", "--system-prompt", "a" }));
}

test "agent parses sampling and effort flags without a prompt" {
    const options = try parseArgs(&.{ "agent", "--think", "low", "--backend", "cpu", "--max-tokens", "8", "--temperature", "0.7" });
    try std.testing.expectEqual(.agent, options.command);
    try std.testing.expectEqual(.low, options.flags.think.?);
    try std.testing.expectEqual(.cpu, options.flags.backend.?);
    try std.testing.expectEqual(@as(usize, 8), options.flags.max_tokens.?);
    try std.testing.expect(options.flags.ctx_size == null);
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "agent", "--think", "loud" }));
    // A forced prompt profile is a common option of every command that renders.
    try std.testing.expectEqual(.gemma4, (try parseArgs(&.{ "agent", "--prompt-profile", "gemma4" })).flags.prompt_profile.?);
    try std.testing.expectEqual(.gemma4, (try parseArgs(&.{ "tokenize", "--prompt", "a", "--prompt-profile", "gemma4" })).flags.prompt_profile.?);
    try std.testing.expectEqual(.qwen38, (try parseArgs(&.{ "bench", "--prompt", "a", "--prompt-profile", "qwen38" })).flags.prompt_profile.?);
    try std.testing.expectError(error.UnknownPromptProfile, parseArgs(&.{ "generate", "--prompt", "a", "--prompt-profile", "llama" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "generate", "--prompt", "a", "--prompt-profile", "gemma4", "--prompt-profile", "gemma4" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "agent", "--raw", "--think", "low" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "bench", "--prompt", "a", "--think", "low" }));
    // Speculative decoding: the switch and the draft length on every command
    // that generates; the host bound is `config.max_draft_length`.
    try std.testing.expect((try parseArgs(&.{ "generate", "--prompt", "a", "--speculative", "on" })).flags.speculative.?);
    try std.testing.expect(!(try parseArgs(&.{ "bench", "--prompt", "a", "--speculative", "off" })).flags.speculative.?);
    try std.testing.expectEqual(@as(usize, 2), (try parseArgs(&.{ "agent", "--speculative", "on", "--draft-length", "2" })).flags.draft_length.?);
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--speculative", "maybe" }));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--draft-length", "0" }));
    try std.testing.expectEqual(config.ImageMaxTokens{ .tokens = 512 }, (try parseArgs(&.{ "generate", "--prompt", "a", "--image-max-tokens", "512" })).flags.image_max_tokens.?);
    try std.testing.expectEqual(config.ImageMaxTokens.auto, (try parseArgs(&.{ "agent", "--image-max-tokens", "auto" })).flags.image_max_tokens.?);
    try std.testing.expectEqual(@as(?usize, 0), (try parseArgs(&.{ "agent", "--thinking-budget", "0" })).flags.thinking_budget);
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "agent", "--thinking-budget", "99999" }));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--image-max-tokens", "0" }));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--image-max-tokens", "4097" }));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--image-max-tokens", "most" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "bench", "--prompt", "a", "--image-max-tokens", "512" }));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--draft-length", "16" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "generate", "--prompt", "a", "--speculative", "on", "--speculative", "off" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "generate", "--prompt", "a", "--draft-length", "2", "--draft-length", "3" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "tokenize", "--prompt", "a", "--speculative", "on" }));
    const sampled = try parseArgs(&.{ "bench", "--prompt", "a", "--temperature", "0.7", "--top-k", "40", "--top-p", "0.95", "--seed", "3" });
    try std.testing.expectEqual(@as(f32, 0.7), sampled.flags.sampling.temperature.?);
    try std.testing.expectEqual(@as(usize, 40), sampled.flags.sampling.top_k.?);
    try std.testing.expectEqual(@as(f32, 0.95), sampled.flags.sampling.top_p.?);
    try std.testing.expectEqual(@as(u64, 3), sampled.benchmark.seed.?);
    try std.testing.expect(sampled.flags.sampling.min_p == null);
    try std.testing.expectEqual(.medium, (try parseArgs(&.{ "generate", "--prompt", "a", "--think", "medium" })).flags.think.?);
}

test "sampling flags are per-option overrides shared by generate, bench, and the agent" {
    // Unflagged: every override is null, so the profile (or bench's neutral
    // options) applies untouched.
    const plain = try parseArgs(&.{ "generate", "--prompt", "a" });
    try std.testing.expectEqual(config.Flags{}, plain.flags);
    try std.testing.expectEqual(inference.sampling.Options{ .temperature = 0.7, .top_p = 0.8, .top_k = 20, .presence_penalty = 1.5 }, inference.profiles.Profile.qwen38.samplingOptions(.off, plain.flags.sampling));
    const flagged = try parseArgs(&.{ "agent", "--min-p", "0.05", "--presence-penalty", "0.5", "--repetition-penalty", "1.1" });
    try std.testing.expectEqual(@as(f32, 0.05), flagged.flags.sampling.min_p.?);
    try std.testing.expectEqual(@as(f32, 0.5), flagged.flags.sampling.presence_penalty.?);
    try std.testing.expectEqual(@as(f32, 1.1), flagged.flags.sampling.repetition_penalty.?);
    try std.testing.expect(flagged.flags.sampling.temperature == null);
    const for_bench = try parseArgs(&.{ "bench", "--prompt", "a", "--min-p", "0.1", "--presence-penalty", "1.5", "--repetition-penalty", "1.2" });
    try std.testing.expectEqual(flagged.flags.sampling.min_p.? * 2, for_bench.flags.sampling.min_p.?);
    try std.testing.expectEqual(@as(f32, 1.5), for_bench.flags.sampling.presence_penalty.?);
    try std.testing.expectEqual(@as(f32, 1.2), for_bench.flags.sampling.repetition_penalty.?);
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "generate", "--prompt", "a", "--min-p", "0.1", "--min-p", "0.2" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "generate", "--prompt", "a", "--presence-penalty", "1", "--presence-penalty", "1" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "generate", "--prompt", "a", "--repetition-penalty", "1", "--repetition-penalty", "1" }));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--min-p", "lots" }));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--presence-penalty", "" }));
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "generate", "--prompt", "a", "--repetition-penalty" }));
    // Range validation of flags is the sampler's job (it runs before the model loads), not the parser's.
    try std.testing.expectEqual(@as(f32, 5), (try parseArgs(&.{ "generate", "--prompt", "a", "--min-p", "5" })).flags.sampling.min_p.?);
    try std.testing.expectError(error.InvalidSamplingOptions, inference.sampling.Sampler.init(0, inference.profiles.Profile.qwen38.samplingOptions(.off, .{ .min_p = 5 })));
}

test "config parses its positional action and rejects the model and generation flags" {
    try std.testing.expectEqual(.init, (try parseArgs(&.{ "config", "init" })).config_action);
    const discovering = try parseArgs(&.{ "config", "init", "--discover", "--dry-run", "--json" });
    try std.testing.expect(discovering.discover and discovering.dry_run and discovering.json);
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "config", "init", "--dry-run" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "config", "init", "--json" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "config", "show", "--discover" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "config", "init", "--discover", "--discover" }));
    const show = try parseArgs(&.{ "config", "show", "--json" });
    try std.testing.expectEqual(.config, show.command);
    try std.testing.expectEqual(.show, show.config_action);
    try std.testing.expect(show.json);
    try std.testing.expectError(error.MissingConfigAction, parseArgs(&.{"config"}));
    try std.testing.expectError(error.UnknownConfigAction, parseArgs(&.{ "config", "reset" }));
    const set = try parseArgs(&.{ "config", "set", "engine.model", "hauhau" });
    try std.testing.expectEqual(.set, set.config_action);
    try std.testing.expectEqualStrings("engine.model", set.set_key);
    try std.testing.expectEqualStrings("hauhau", set.set_value);
    try std.testing.expectError(error.MissingSetArguments, parseArgs(&.{ "config", "set", "engine.model" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "config", "set", "engine.model", "x", "--json" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "config", "init", "--json" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "config", "show", "--model", "m.gguf" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "config", "show", "--ctx-size", "16" }));
    try std.testing.expectEqual(.help, (try parseArgs(&.{ "config", "--help" })).command);
}

test "config init and show run against NUCLIS_HOME without touching the real root" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    var environ: std.process.Environ.Map = .init(alloc);
    defer environ.deinit();
    try environ.put("NUCLIS_HOME", root);
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var diag: config.Diagnostic = .{};
    // show before init: defaults, not found, and nothing written.
    try run(alloc, io, &environ, try parseArgs(&.{ "config", "show" }), &out.writer, &diag);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "not found; built-in defaults") != null);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "nuclis.json", .{}));
    out.clearRetainingCapacity();
    try run(alloc, io, &environ, try parseArgs(&.{ "config", "init" }), &out.writer, &diag);
    try std.testing.expect(std.mem.startsWith(u8, out.written(), "wrote "));
    // A fresh root: the catalogue model is absent and the next step is its pull.
    // The column width comes from the catalogue's widest name, so the
    // assertion is on the two parts rather than the spacing between them.
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "qwen3.8-27b ") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), " absent") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "next: nuclis model pull qwen3.8-27b fetches the model") != null);
    try tmp.dir.access(io, "nuclis.json", .{});
    out.clearRetainingCapacity();
    try run(alloc, io, &environ, try parseArgs(&.{ "config", "init" }), &out.writer, &diag);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "exists; not overwritten") != null);
    out.clearRetainingCapacity();
    try run(alloc, io, &environ, try parseArgs(&.{ "config", "show", "--json" }), &out.writer, &diag);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("found").?.bool);
    try std.testing.expectEqualStrings("file", parsed.value.object.get("sources").?.object.get("engine.model").?.string);
    // A broken file stops every model command with the key named, before any model is opened.
    try tmp.dir.writeFile(io, .{ .sub_path = "nuclis.json", .data = "{ \"schema_version\": 1, \"engine\": { \"ctx\": 1 } }" });
    try std.testing.expectError(error.UnknownConfigKey, run(alloc, io, &environ, try parseArgs(&.{ "generate", "--prompt", "hi" }), &out.writer, &diag));
    try std.testing.expectEqualStrings("unknown key engine.ctx", diag.message());
    try std.testing.expectError(error.UnknownConfigKey, run(alloc, io, &environ, try parseArgs(&.{ "config", "show" }), &out.writer, &diag));
    // No root at all: config needs one, model commands with --model do not need it to resolve the path.
    var empty: std.process.Environ.Map = .init(alloc);
    defer empty.deinit();
    try std.testing.expectError(error.MissingHome, run(alloc, io, &empty, try parseArgs(&.{ "config", "show" }), &out.writer, &diag));
    try std.testing.expectError(error.MissingHome, run(alloc, io, &empty, try parseArgs(&.{ "inspect", "--json" }), &out.writer, &diag));
    // A model that is not there fails before anything opens, naming the resolved path.
    try tmp.dir.writeFile(io, .{ .sub_path = "nuclis.json", .data = "{ \"schema_version\": 1 }" });
    try std.testing.expectError(error.ModelFileNotFound, run(alloc, io, &environ, try parseArgs(&.{ "inspect", "--json" }), &out.writer, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf") != null);
    try std.testing.expectError(error.ModelFileNotFound, run(alloc, io, &environ, try parseArgs(&.{ "inspect", "--model", "missing.gguf" }), &out.writer, &diag));
    try std.testing.expectEqualStrings("no model file at missing.gguf", diag.message()[0.."no model file at missing.gguf".len]);
}

test "model parses its action, the positional repository, and its own flags only" {
    const pull = try parseArgs(&.{ "model", "pull", "unsloth/Qwen3.8-27B-GGUF", "--file", "MTP/mtp.gguf", "--revision", "main", "--role", "mtp", "--force", "--json" });
    try std.testing.expectEqual(.model, pull.command);
    try std.testing.expectEqual(.pull, pull.model_action);
    try std.testing.expectEqualStrings("unsloth/Qwen3.8-27B-GGUF", pull.pull.repo);
    try std.testing.expectEqualStrings("MTP/mtp.gguf", pull.pull.file.?);
    try std.testing.expectEqualStrings("main", pull.pull.revision.?);
    try std.testing.expectEqual(model.Role.mtp, pull.pull.role.?);
    try std.testing.expect(pull.pull.force and pull.json);
    const plain = try parseArgs(&.{ "model", "pull", "a/b" });
    try std.testing.expect(plain.pull.file == null and plain.pull.revision == null and plain.pull.role == null and !plain.pull.force and !plain.json);
    try std.testing.expect(plain.pull.with.count() == 0 and !plain.pull.all);
    const with = try parseArgs(&.{ "model", "pull", "qwen3.8-27b", "--with", "mmproj,mtp" });
    try std.testing.expect(with.pull.with.contains(.mmproj) and with.pull.with.contains(.mtp) and !with.pull.with.contains(.imatrix));
    try std.testing.expect((try parseArgs(&.{ "model", "pull", "qwen3.8-27b", "--all" })).pull.all);
    const registered = try parseArgs(&.{ "model", "pull", "HauhauCS/Gemma4", "--file", "g.gguf", "--register", "hauhau", "--profile", "gemma4" });
    try std.testing.expectEqualStrings("hauhau", registered.pull.register.?);
    try std.testing.expectEqual(.gemma4, registered.pull.profile.?);
    try std.testing.expectError(error.ConflictingOptions, parseArgs(&.{ "model", "pull", "HauhauCS/Gemma4", "--file", "g.gguf", "--profile", "gemma4" }));
    try std.testing.expectError(error.UnknownPromptProfile, parseArgs(&.{ "model", "pull", "a/b", "--register", "x", "--profile", "llama" }));
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "model", "pull", "a/b", "--register", "--json" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "model", "inspect", "a/b", "--register", "x" }));
    try std.testing.expectError(error.UnknownRole, parseArgs(&.{ "model", "pull", "qwen3.8-27b", "--with", "main" }));
    try std.testing.expectError(error.UnknownRole, parseArgs(&.{ "model", "pull", "qwen3.8-27b", "--with", "mmproj,draft" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "model", "pull", "qwen3.8-27b", "--with", "mtp", "--with", "mmproj" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "model", "pull", "qwen3.8-27b", "--all", "--all" }));
    const ls = try parseArgs(&.{ "model", "ls", "--json" });
    try std.testing.expectEqual(.ls, ls.model_action);
    try std.testing.expect(ls.json);
    try std.testing.expectEqual(.help, (try parseArgs(&.{ "model", "--help" })).command);
    try std.testing.expectError(error.MissingModelAction, parseArgs(&.{"model"}));
    try std.testing.expectError(error.UnknownModelAction, parseArgs(&.{ "model", "rm" }));
    try std.testing.expectError(error.MissingRepository, parseArgs(&.{ "model", "pull" }));
    try std.testing.expectError(error.MissingRepository, parseArgs(&.{ "model", "pull", "--file", "x.gguf" }));
    try std.testing.expectError(error.UnknownRole, parseArgs(&.{ "model", "pull", "a/b", "--role", "draft" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "model", "pull", "a/b", "--file", "x", "--file", "y" }));
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "model", "pull", "a/b", "--revision" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "model", "pull", "a/b", "--model", "m.gguf" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "model", "ls", "--force" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "model", "ls", "--file", "x" }));
    // inspect: the repository, --file, --revision, --json; nothing of pull's.
    const inspect = try parseArgs(&.{ "model", "inspect", "a/b", "--file", "x.gguf", "--revision", "v1", "--json" });
    try std.testing.expectEqual(.inspect, inspect.model_action);
    try std.testing.expectEqualStrings("a/b", inspect.pull.repo);
    try std.testing.expectEqualStrings("x.gguf", inspect.pull.file.?);
    try std.testing.expectEqualStrings("v1", inspect.pull.revision.?);
    try std.testing.expect(inspect.json);
    try std.testing.expectEqualStrings("qwen3.8-27b", (try parseArgs(&.{ "model", "inspect", "qwen3.8-27b" })).pull.repo);
    try std.testing.expectError(error.MissingRepository, parseArgs(&.{ "model", "inspect" }));
    try std.testing.expectError(error.MissingRepository, parseArgs(&.{ "model", "inspect", "--file", "x.gguf" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "model", "inspect", "a/b", "--force" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "model", "inspect", "a/b", "--all" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "model", "inspect", "a/b", "--with", "mtp" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "model", "inspect", "a/b", "--role", "mtp" }));
}

test "model ls runs against NUCLIS_HOME and needs no configuration file" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    var environ: std.process.Environ.Map = .init(alloc);
    defer environ.deinit();
    try environ.put("NUCLIS_HOME", root);
    // A broken configuration must not block model commands.
    try tmp.dir.writeFile(io, .{ .sub_path = "nuclis.json", .data = "{ \"schema_version\": 1, \"engine\": { \"ctx\": 1 } }" });
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var diag: config.Diagnostic = .{};
    try run(alloc, io, &environ, try parseArgs(&.{ "model", "ls" }), &out.writer, &diag);
    // The column width comes from the catalogue's widest name, so the
    // assertion is on the two parts rather than the spacing between them.
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "qwen3.8-27b ") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), " absent") != null);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "models", .{}));
    var empty: std.process.Environ.Map = .init(alloc);
    defer empty.deinit();
    try std.testing.expectError(error.MissingHome, run(alloc, io, &empty, try parseArgs(&.{ "model", "ls" }), &out.writer, &diag));
    try std.testing.expectError(error.MissingRepository, run(alloc, io, &environ, .{ .command = .model, .model_action = .pull }, &out.writer, &diag));
    try std.testing.expectError(error.MissingRepository, run(alloc, io, &environ, .{ .command = .model, .model_action = .inspect }, &out.writer, &diag));
    // A pull by registry name is the one model command that reads the
    // file: the broken file above stops it with the key named ...
    try std.testing.expectError(error.UnknownConfigKey, run(alloc, io, &environ, try parseArgs(&.{ "model", "pull", "gemma" }), &out.writer, &diag));
    // ... a valid file without the entry names the three forms ...
    try tmp.dir.writeFile(io, .{ .sub_path = "nuclis.json", .data = "{ \"schema_version\": 1, \"models\": { \"local\": { \"path\": \"/scratch/x.gguf\" } } }" });
    try std.testing.expectError(error.UnknownModel, run(alloc, io, &environ, try parseArgs(&.{ "model", "pull", "gemma" }), &out.writer, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "gemma is not a catalogue name, an entry of the models registry in ") != null);
    // ... and an entry that names a path has nothing to pull.
    try std.testing.expectError(error.NotPullable, run(alloc, io, &environ, try parseArgs(&.{ "model", "pull", "local" }), &out.writer, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "local names a local path (/scratch/x.gguf)") != null);
    // A registry entry resolves `--model` and `engine.model` to its path before anything opens.
    try std.testing.expectError(error.ModelFileNotFound, run(alloc, io, &environ, try parseArgs(&.{ "inspect", "--model", "local" }), &out.writer, &diag));
    try std.testing.expect(std.mem.startsWith(u8, diag.message(), "no model file at /scratch/x.gguf"));
    try tmp.dir.writeFile(io, .{ .sub_path = "nuclis.json", .data = "{ \"schema_version\": 1, \"engine\": { \"model\": \"g\" }, \"models\": { \"g\": { \"repo\": \"a/b\", \"file\": \"g.gguf\" } } }" });
    try std.testing.expectError(error.ModelFileNotFound, run(alloc, io, &environ, try parseArgs(&.{ "inspect", "--json" }), &out.writer, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "/models/a/b/g.gguf") != null);
}

test {
    _ = paths;
    _ = config;
    _ = agent;
    _ = model;
    _ = @import("catalog.zig");
    _ = inspection;
    _ = validation;
    _ = engine;
    _ = bench;
    _ = eval;
    _ = discover;
    _ = @import("interrupt.zig");
    _ = style;
}

test "bench parses shared prompt flags and its own repetition flags" {
    const options = try parseArgs(&.{ "bench", "--prompt-file", "p.txt", "--repeat", "5", "--warmup", "0", "--backend", "metal", "--json" });
    try std.testing.expectEqual(.bench, options.command);
    try std.testing.expectEqualStrings("p.txt", options.benchmark.prompt_file.?);
    try std.testing.expectEqual(@as(usize, 5), options.benchmark.repeat.?);
    try std.testing.expectEqual(@as(usize, 0), options.benchmark.warmup.?);
    try std.testing.expectEqual(.metal, options.flags.backend.?);
    try std.testing.expectError(error.MissingPrompt, parseArgs(&.{"bench"}));
    try std.testing.expectError(error.ConflictingPromptSources, parseArgs(&.{ "bench", "--prompt", "a", "--prompt-file", "b" }));
    const tokens = try parseArgs(&.{ "bench", "--prompt-tokens", "ids.json", "--max-tokens", "128" });
    try std.testing.expectEqualStrings("ids.json", tokens.benchmark.prompt_tokens.?);
    try std.testing.expect(tokens.benchmark.prompt == null and tokens.benchmark.prompt_file == null);
    try std.testing.expectError(error.ConflictingPromptSources, parseArgs(&.{ "bench", "--prompt-tokens", "a", "--prompt-file", "b" }));
    try std.testing.expectError(error.ConflictingPromptSources, parseArgs(&.{ "bench", "--prompt-tokens", "a", "--raw" }));
    const generate_tokens = try parseArgs(&.{ "generate", "--prompt-tokens", "ids.json" });
    try std.testing.expectEqualStrings("ids.json", generate_tokens.generation.prompt_tokens.?);
    try std.testing.expectError(error.ConflictingPromptSources, parseArgs(&.{ "generate", "--prompt-tokens", "a", "--raw" }));
    try std.testing.expectError(error.ConflictingPromptSources, parseArgs(&.{ "generate", "--prompt-tokens", "a", "--prompt", "b" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "tokenize", "--prompt-tokens", "a" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "bench", "--prompt", "a", "--logits", "l.f32" }));
    try std.testing.expect((try parseArgs(&.{ "bench", "--prompt", "a", "--profile" })).benchmark.profile);
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "bench", "--prompt", "a", "--profile", "--profile" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "generate", "--profile", "--prompt", "a" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "generate", "--prompt", "a", "--repeat", "1" }));
}

test "validation uses the shared model flags and supports command help" {
    const options = try parseArgs(&.{ "validate", "--model", "model.gguf", "--json" });
    try std.testing.expectEqual(.validate, options.command);
    try std.testing.expect(options.json);
    try std.testing.expectEqualStrings("model.gguf", options.model.?);
    try std.testing.expectEqual(.help, (try parseArgs(&.{ "validate", "--help" })).command);
}

test "generation parses bounds and rejects duplicate or malformed options" {
    const options = try parseArgs(&.{ "generate", "--prompt", "Hi", "--max-tokens", "2", "--ctx-size", "16", "--raw" });
    try std.testing.expectEqual(@as(usize, 2), options.flags.max_tokens.?);
    try std.testing.expectEqual(@as(usize, 16), options.flags.ctx_size.?);
    try std.testing.expect(options.generation.raw);
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "generate", "--prompt", "a", "--prompt", "b" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "generate", "--prompt", "a", "--backend", "cpu", "--backend", "metal" }));
    try std.testing.expectEqual(.f32, (try parseArgs(&.{ "generate", "--prompt", "a", "--kv", "f32" })).flags.kv.?);
    try std.testing.expectEqual(.f16, (try parseArgs(&.{ "bench", "--prompt", "a", "--kv", "f16" })).flags.kv.?);
    try std.testing.expect(options.flags.kv == null);
    try std.testing.expectError(error.UnsupportedKvPrecision, parseArgs(&.{ "agent", "--kv", "f64" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "agent", "--kv", "f16", "--kv", "f32" }));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "generate", "--prompt", "a", "--max-tokens", "-1" }));
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "generate", "--prompt" }));
}

test "eval takes a text file, its window, and the engine's precision flags" {
    const options = try parseArgs(&.{ "eval", "--file", "wiki.test.raw", "--ctx-size", "512", "--chunks", "8", "--reference", "r.json", "--backend", "metal", "--kv", "f32", "--model", "m.gguf", "--json" });
    try std.testing.expectEqual(.eval, options.command);
    try std.testing.expectEqualStrings("wiki.test.raw", options.evaluation.file.?);
    try std.testing.expectEqual(@as(usize, 512), options.evaluation.ctx.?);
    try std.testing.expectEqual(@as(usize, 8), options.evaluation.chunks.?);
    try std.testing.expectEqualStrings("r.json", options.evaluation.reference.?);
    try std.testing.expectEqual(.f32, options.flags.kv.?);
    // The window is the command's own, never the configuration's context.
    try std.testing.expect(options.flags.ctx_size == null);
    try std.testing.expect(options.json);
    try std.testing.expectError(error.MissingTextFile, parseArgs(&.{"eval"}));
    try std.testing.expectError(error.InvalidNumber, parseArgs(&.{ "eval", "--file", "t", "--chunks", "0" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "eval", "--file", "t", "--file", "u" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "eval", "--file", "t", "--prompt", "a" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "eval", "--raw", "--file", "t" }));
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "eval", "--file" }));
}

test "tokenize takes the prompt, the rendering, and the effort only" {
    const options = try parseArgs(&.{ "tokenize", "--prompt-file", "p.txt", "--raw", "--think", "low", "--model", "m.gguf", "--json" });
    try std.testing.expectEqual(.tokenize, options.command);
    try std.testing.expectEqualStrings("p.txt", options.generation.prompt_file.?);
    try std.testing.expect(options.generation.raw and options.json);
    try std.testing.expectEqual(.low, options.flags.think.?);
    try std.testing.expectEqualStrings("m.gguf", options.model.?);
    try std.testing.expectError(error.MissingPrompt, parseArgs(&.{"tokenize"}));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "tokenize", "--prompt", "a", "--backend", "cpu" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "tokenize", "--prompt", "a", "--max-tokens", "4" }));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "tokenize", "--prompt", "a", "--temperature", "0" }));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "tokenize", "--prompt", "a", "--raw", "--raw" }));
}
