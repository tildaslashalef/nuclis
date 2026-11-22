//! `nuclis --help`, `nuclis help <command>`, and one page per command.
//!
//! Every page is self-contained: what the command does, its usage lines,
//! every flag the parser accepts for it with its default, examples, and the
//! few behaviours a user would otherwise be surprised by. Nothing here
//! points at the repository's documents; a page that needs them to be read
//! is not finished.
//!
//! The shape is the output kit's (docs/app/command-output.md): `Usage:`
//! first, the description, then titled sections whose rows align per
//! section and wrap to the terminal (80 columns in a pipe, which a test
//! holds). Pages are written through `Page`, which buffers a section so it
//! can align it; colour comes from `kit.Ui`, nothing in a pipe.
const std = @import("std");
const kit = @import("ui.zig");
const config_schema = @import("config.zig");
/// The context ceiling of this build (`-Dmax-context`).
const max_ctx = std.fmt.comptimePrint("{d}", .{config_schema.max_context});
/// What `--backend` defaults to: the CPU in a build without Metal.
const default_backend = if (@import("inference").metal.enabled) "default metal" else "default cpu";

/// The commands with a page of their own; `null` is the overview.
pub const Topic = enum { inspect, validate, generate, bench, tokenize, eval, agent, config, model, cache, decide, embed, transcribe, speak, voice, serve, completion };

pub fn write(out: *std.Io.Writer, ui: kit.Ui, topic: ?Topic, version: []const u8) !void {
    var storage: [96 * 1024]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&storage);
    var page: Page = .{ .out = out, .ui = ui, .arena = fba.allocator(), .width = @min(ui.width orelse 80, 100) };
    const p = &page;
    if (topic) |value| return switch (value) {
        .inspect => inspect(p),
        .validate => validate(p),
        .generate => generate(p),
        .bench => bench(p),
        .tokenize => tokenize(p),
        .eval => eval(p),
        .agent => agent(p),
        .config => config(p),
        .model => model(p),
        .cache => cache(p),
        .decide => decide(p),
        .embed => embed(p),
        .transcribe => transcribe(p),
        .speak => speak(p),
        .voice => voice(p),
        .serve => serve(p),
        .completion => completion(p),
    };
    return overview(p, version);
}

// ----- the page writer -----

/// One page being written. A section's items are kept until the next
/// heading so its rows share one name column; usage lines wait for the
/// first heading after them, so the page opens `Usage: …` and then says
/// what the command is.
const Page = struct {
    out: *std.Io.Writer,
    ui: kit.Ui,
    /// Fixed-buffer allocator; a page is a few kilobytes.
    arena: std.mem.Allocator,
    width: usize,
    description: []const u8 = "",
    in_usage: bool = false,
    /// The section being written: a flag section's values stay lowercase.
    flags_section: bool = false,
    usage: std.ArrayList([]const u8) = .empty,
    items: std.ArrayList(Item) = .empty,

    const Item = union(enum) {
        /// `literal`: the text is values (keys, ids), never made a sentence.
        row: struct { name: []const u8, text: std.ArrayList(u8), literal: bool = false },
        prose: std.ArrayList(u8),
        code: []const u8,
        comment: std.ArrayList(u8),
    };

    /// The widest name that keeps its text on the same line.
    const max_name = 28;

    fn heading(p: *Page, text: []const u8) !void {
        try p.flush();
        if (std.mem.eql(u8, text, "Usage:")) {
            p.in_usage = true;
            return;
        }
        try p.openUsage();
        const shown = if (std.mem.eql(u8, text, "Options:")) "Flags:" else if (std.mem.eql(u8, text, "Global options:")) "Global flags:" else text;
        p.flags_section = std.mem.endsWith(u8, shown, "lags:");
        try p.out.print("\n{s}{s}{s}\n", .{ p.ui.on(.title), shown, p.ui.off(.title) });
    }

    /// The usage lines and the description, once, before the first section.
    fn openUsage(p: *Page) !void {
        if (!p.in_usage and p.usage.items.len == 0 and p.description.len == 0) return;
        if (p.usage.items.len == 1) {
            try p.out.print("{s}Usage:{s} {s}\n", .{ p.ui.on(.title), p.ui.off(.title), p.usage.items[0] });
        } else if (p.usage.items.len > 1) {
            try p.out.print("{s}Usage:{s}\n", .{ p.ui.on(.title), p.ui.off(.title) });
            for (p.usage.items) |line| try p.out.print("  {s}\n", .{line});
        }
        if (p.description.len > 0) {
            try p.out.writeByte('\n');
            try p.wrap(try sentence(p.arena, p.description, false), 0, 0, 0);
        }
        p.in_usage = false;
        p.usage.clearRetainingCapacity();
        p.description = "";
    }

    fn row(p: *Page, name: []const u8, text: []const u8) !void {
        try p.rowOf(name, text, false);
    }

    fn rowOf(p: *Page, name: []const u8, text: []const u8, literal: bool) !void {
        var t: std.ArrayList(u8) = .empty;
        try t.appendSlice(p.arena, text);
        try p.items.append(p.arena, .{ .row = .{ .name = name, .text = t, .literal = literal } });
    }

    /// A continuation: of the row or paragraph before it, else the comment
    /// under an example's command line.
    fn more(p: *Page, text: []const u8) !void {
        if (p.items.items.len > 0) switch (p.items.items[p.items.items.len - 1]) {
            .row => |*r| return join(p.arena, &r.text, text),
            .prose => |*t| return join(p.arena, t, text),
            .comment => |*t| return join(p.arena, t, text),
            .code => {},
        };
        var t: std.ArrayList(u8) = .empty;
        try t.appendSlice(p.arena, text);
        try p.items.append(p.arena, .{ .comment = t });
    }

    fn code(p: *Page, line: []const u8) !void {
        if (p.in_usage) return p.usage.append(p.arena, line);
        try p.items.append(p.arena, .{ .code = line });
    }

    fn plain(p: *Page, text: []const u8) !void {
        if (p.items.items.len > 0) switch (p.items.items[p.items.items.len - 1]) {
            .prose => |*t| return join(p.arena, t, text),
            else => {},
        };
        var t: std.ArrayList(u8) = .empty;
        try t.appendSlice(p.arena, text);
        try p.items.append(p.arena, .{ .prose = t });
    }

    /// Writes the section held so far: rows aligned on its widest name (a
    /// long-only flag indented under the short ones), text wrapped.
    fn flush(p: *Page) !void {
        defer p.items.clearRetainingCapacity();
        var column: usize = 0;
        var shorts = false;
        for (p.items.items) |item| switch (item) {
            .row => |r| {
                if (r.name.len > 1 and r.name[0] == '-' and r.name[1] != '-') shorts = true;
            },
            else => {},
        };
        for (p.items.items) |item| switch (item) {
            .row => |r| {
                const w = r.name.len + @as(usize, if (shorts and isLongOnly(r.name)) 4 else 0);
                if (w <= max_name) column = @max(column, w);
            },
            else => {},
        };
        for (p.items.items) |item| switch (item) {
            .row => |r| {
                const pad: usize = if (shorts and isLongOnly(r.name)) 4 else 0;
                try p.out.splatByteAll(' ', 2 + pad);
                try p.out.writeAll(r.name);
                const text = if (r.literal) r.text.items else try sentence(p.arena, r.text.items, p.flags_section);
                if (text.len == 0) {
                    try p.out.writeByte('\n');
                    continue;
                }
                if (pad + r.name.len > column) {
                    try p.out.writeByte('\n');
                    try p.wrap(text, 2 + column + 2, 2 + column + 2, 2 + column + 2);
                } else {
                    try p.out.splatByteAll(' ', column - pad - r.name.len + 2);
                    try p.wrap(text, 0, 2 + column + 2, 2 + column + 2);
                }
            },
            .prose => |t| try p.wrap(t.items, 2, 2, 2),
            .code => |line| try p.out.print("  {s}\n", .{line}),
            .comment => |t| {
                try p.out.writeAll(p.ui.on(.muted));
                try p.wrap(t.items, 6, 6, 6);
                try p.out.writeAll(p.ui.off(.muted));
            },
        };
    }

    /// Greedy word wrap at the page width: `lead` spaces first, the first
    /// word at column `start`, every next line at `indent`.
    fn wrap(p: *Page, text: []const u8, lead: usize, start: usize, indent: usize) !void {
        try p.out.splatByteAll(' ', lead);
        var at = start;
        var fresh = true;
        var words = std.mem.tokenizeScalar(u8, text, ' ');
        while (words.next()) |word| {
            const w = kit.text.width(word);
            if (!fresh and at + 1 + w > p.width) {
                try p.out.writeByte('\n');
                try p.out.splatByteAll(' ', indent);
                at = indent;
                fresh = true;
            }
            if (!fresh) {
                try p.out.writeByte(' ');
                at += 1;
            }
            try p.out.writeAll(word);
            at += w;
            fresh = false;
        }
        try p.out.writeByte('\n');
    }

    /// Ends the page: the last section, then a blank line.
    fn end(p: *Page) !void {
        try p.flush();
        try p.openUsage();
        try p.out.writeByte('\n');
    }

    fn isLongOnly(name: []const u8) bool {
        return std.mem.startsWith(u8, name, "--");
    }

    fn join(arena: std.mem.Allocator, list: *std.ArrayList(u8), text: []const u8) !void {
        if (text.len == 0) return;
        if (list.items.len > 0) try list.append(arena, ' ');
        try list.appendSlice(arena, text);
    }
};

/// A summary as a sentence: its first letter capitalized, unless the first
/// word is code (a flag, a path, a key, `ls:`) or, in a flag section,
/// starts the list of the flag's values (`off, low, medium`).
fn sentence(arena: std.mem.Allocator, text: []const u8, values: bool) ![]const u8 {
    if (text.len == 0 or !std.ascii.isLower(text[0])) return text;
    const word_end = std.mem.indexOfAny(u8, text, " ,;") orelse text.len;
    if (values and word_end < text.len and text[word_end] == ',') return text;
    if (std.mem.indexOfAny(u8, text[0..word_end], ":./_=<>~`|0123456789") != null) return text;
    const out = try arena.dupe(u8, text);
    out[0] = std.ascii.toUpper(out[0]);
    return out;
}

// The calls every page is written with.

/// The page's command and what it is; printed after its usage lines.
fn title(p: *Page, command: []const u8, text: []const u8) !void {
    _ = command;
    p.description = text;
}

fn heading(p: *Page, text: []const u8) !void {
    try p.heading(text);
}

fn row(p: *Page, name: []const u8, text: []const u8) !void {
    try p.row(name, text);
}

fn more(p: *Page, text: []const u8) !void {
    try p.more(text);
}

fn code(p: *Page, line: []const u8) !void {
    try p.code(line);
}

/// An example's command line, its comment under it, muted.
fn example(p: *Page, line: []const u8, comment: []const u8) !void {
    try p.code(line);
    try p.more(comment);
}

fn plain(p: *Page, text: []const u8) !void {
    try p.plain(text);
}

/// One row per section of the configuration file, its keys read from the
/// schema itself so the page cannot miss one; nested groups print as
/// `group.{a,b}`. The registry (`models`) has its own row.
fn configKeys(p: *Page) !void {
    const info = @typeInfo(config_schema.Config).@"struct";
    inline for (info.field_names, info.field_types) |name, Section| {
        if (comptime @typeInfo(Section) != .@"struct" or Section == config_schema.Models) continue;
        try p.rowOf(name ++ ".*", comptime keyList(Section), true);
    }
}

fn keyList(comptime T: type) []const u8 {
    const info = @typeInfo(T).@"struct";
    comptime var text: []const u8 = "";
    inline for (info.field_names, info.field_types, 0..) |name, Field, i| {
        text = text ++ (if (i == 0) "" else ", ") ++ name;
        if (@typeInfo(Field) == .@"struct") text = text ++ ".{" ++ innerList(Field) ++ "}";
    }
    return text;
}

fn innerList(comptime T: type) []const u8 {
    comptime var text: []const u8 = "";
    inline for (@typeInfo(T).@"struct".field_names, 0..) |name, i| text = text ++ (if (i == 0) "" else ", ") ++ name;
    return text;
}

/// The options `generate`, `bench`, and `agent` share (the engine's).
fn engineOptions(p: *Page) !void {
    try row(p, "--model <name|path>", "registry entry, catalogue name, or path; default");
    try more(p, "engine.model of the config file (qwen3.8-27b)");
    try row(p, "--backend cpu|metal", default_backend);
    try row(p, "--ctx-size <n>", "context window in tokens, 1.." ++ max_ctx ++ "; default 65536");
    try row(p, "--kv f16|f32|q8_0", "attention cache precision on the GPU; default f16");
    try row(p, "--max-tokens <n>", "output budget, 1..16384; default 8192 (bench: 32)");
    try row(p, "--speculative on|off", "verify drafts; default the model's verdict, else off");
    try row(p, "--draft <variant>", "another drafter of an index model (qwen: dflash2)");
    try row(p, "--draft-length <n>", "drafts per verify batch, 1..15, capped by the");
    try more(p, "draft source; default the model entry's, else 4");
    try row(p, "--draft-max-depth <n>", "depth drafting stops at; 0 never; default measured");
    try row(p, "--prompt-profile <p>", "force qwen38, gemma4, gemma4_e, or muse_glimmer");
    try more(p, "on a finetune whose chat template is not pinned");
    try row(p, "--seed <n>", "sampler seed; default 0");
}

fn samplingOptions(p: *Page) !void {
    try row(p, "--temperature <t>", "0 is greedy; each flag overrides one option of the");
    try row(p, "--top-k <n>", "profile's defaults for the reasoning mode");
    try row(p, "--top-p <p>", "");
    try row(p, "--min-p <p>", "");
    try row(p, "--presence-penalty <x>", "");
    try row(p, "--repetition-penalty <x>", "");
}

// ----- the overview -----

fn overview(p: *Page, version: []const u8) !void {
    _ = version;
    try title(p, "nuclis", "A local inference engine for open-weight models on Apple Silicon");
    try heading(p, "Usage:");
    try code(p, "nuclis <command> [flags]");

    try heading(p, "Commands:");
    try row(p, "agent", "the interactive chat; -p <text> runs one turn");
    try row(p, "generate", "one completion from a prompt");
    try row(p, "bench", "repeated prefill and decode measurements");
    try row(p, "tokenize", "the prompt's token ids and byte offsets, no model run");
    try row(p, "eval", "perplexity of a text file, teacher-forced");
    try row(p, "inspect", "an artifact's identity, dimensions, tensor encodings");
    try row(p, "validate", "whether a file binds to its architecture's adapter");
    try row(p, "model", "pull, list, judge, and remove Hugging Face Hub artifacts");
    try row(p, "cache", "list or clear the agent's saved model states");
    try row(p, "decide", "typed questions about text or JSON, by Laya or clef-flash");
    try row(p, "embed", "text, images, audio into vectors, by EmbeddingGemma 2");
    try row(p, "speak", "speech from text, by Qwen3-TTS");
    try row(p, "voice", "saved voices for speak and the server");
    try row(p, "transcribe", "text from speech, by Nemotron ASR");
    try row(p, "serve", "the nuclis API over HTTP, models kept open");
    try row(p, "config", "write, show, or set a key of ~/.nuclis/nuclis.json");
    try row(p, "completion", "the shell completion script for fish, bash, or zsh");
    try row(p, "help", "show help for a command");

    try heading(p, "Global flags:");
    try row(p, "-h, --help", "this page, or a command's page after its name");
    try row(p, "--version", "the version of this binary");
    try row(p, "--json", "machine-readable output, where a command reports");

    try heading(p, "Examples:");
    try example(p, "nuclis model pull --catalogue", "every current model with its companions");
    try example(p, "nuclis config init", "the config file, from what is installed");
    try example(p, "nuclis agent", "chat with the configured model");

    try heading(p, "Files:");
    try row(p, "~/.nuclis/nuclis.json", "engine, generation, agent settings; the registry");
    try row(p, "~/.nuclis/models/", "artifacts as <owner>/<repo>/<file> plus sidecars");
    try row(p, "~/.nuclis/agent/", "sessions, exports, prompt history");
    try plain(p, "NUCLIS_HOME (an absolute path) moves the root; NO_COLOR drops colour.");
    try p.flush();
    try p.out.print("\nRun '{s}nuclis help <command>{s}' for its flags and examples.\n\n", .{ p.ui.on(.id), p.ui.off(.id) });
}

// ----- one page per command -----

fn agent(p: *Page) !void {
    try title(p, "nuclis agent", "the interactive chat, with tools for small coding tasks");
    try heading(p, "Usage:");
    try code(p, "nuclis agent [flags] [--resume [<id>]]");
    try code(p, "nuclis agent -p <text> [--json] [--session <path>]");
    try code(p, "nuclis agent --print --prompt-file <path> [--json] [--session <path>]");
    try code(p, "nuclis agent ls [--json]");
    try code(p, "nuclis agent rm <id>");
    try code(p, "nuclis agent export <id> [path]");

    try heading(p, "Options:");
    try row(p, "-p, --prompt <text>", "run one turn without a terminal and exit");
    try row(p, "--print", "the same, with the prompt from --prompt-file");
    try row(p, "--prompt-file <path>", "the prompt's text, for print mode");
    try row(p, "--json", "in print mode: one event per line instead of text");
    try row(p, "--session <path>", "record a printed turn to this session file");
    try row(p, "--resume [<id>]", "replay a saved session first; alone, pick (-p: latest)");
    try row(p, "--think <effort>", "off, low, medium, high, xhigh; default low");
    try row(p, "--thinking-budget <n>", "reasoning cap at low (x2 medium ... x4 xhigh, at");
    try more(p, "most max-tokens - 2048); 0 none; entry's, else 2048");
    try row(p, "--system-prompt <path>", "the file's text replaces the built prompt");
    try more(p, "sections (for tuning); AGENTS.md still follows");
    try row(p, "--image-max-tokens <n>", "most tokens a dropped image becomes: auto (the");
    try more(p, "projector's maximum) or a count; default auto");
    try engineOptions(p);
    try samplingOptions(p);

    try heading(p, "Keys:");
    try row(p, "Enter", "send; while a turn runs, steer it: the text reaches");
    try more(p, "the model before its next step");
    try row(p, "Alt-Enter", "while a turn runs, queue the text for after it");
    try row(p, "Shift-Enter, Ctrl-J", "newline");
    try row(p, "Up, Down", "move in the input; history at its first and last row");
    try row(p, "Tab", "complete a /command or a fuzzy @path, else fold");
    try row(p, "Ctrl-E", "expand a paste chip into editable text");
    try row(p, "Ctrl-T, Ctrl-W", "cycle the effort, the context window (as /model)");
    try row(p, "Ctrl-N", "drop the conversation (as /clear)");
    try row(p, "Esc, Ctrl-C, Ctrl-D", "cancel the turn (Esc never quits), or quit");

    try heading(p, "Commands:");
    try row(p, "/model [name]", "switch the model; pick its effort and context window");
    try row(p, "/resume [<id>]", "continue a saved session; Ctrl-D in the list deletes");
    try row(p, "/clear", "drop the conversation; the session stays resumable");
    try row(p, "/help", "keys and commands, inside the surface");

    try heading(p, "Examples:");
    try example(p, "nuclis agent --model gemma-4-e4b-qat --think medium", "chat with another model at more effort");
    try example(p, "nuclis agent -p \"Summarize README.md\" --json", "one scripted turn, events as JSON lines");
    try example(p, "nuclis agent ls", "the sessions saved for this directory");

    try heading(p, "Notes:");
    try plain(p, "Completed turns are written into the terminal's scrollback and survive");
    try plain(p, "exit; the live region at the bottom holds the turn being written, the");
    try plain(p, "editor, and a status bar. A paste of more than a few lines becomes one");
    try plain(p, "chip; the input limit is 128 KiB. Sessions live under ~/.nuclis/agent");
    try plain(p, "(sessions/, exports/, history.jsonl), created when first written; print");
    try plain(p, "mode writes nothing unless --session names a file.");
    try p.end();
}

fn generate(p: *Page) !void {
    try title(p, "nuclis generate", "one completion from a prompt");
    try heading(p, "Usage:");
    try code(p, "nuclis generate --prompt <text> [flags]");
    try code(p, "nuclis generate --prompt-file <path> [flags]");
    try code(p, "nuclis generate --prompt-tokens <json> [flags]");

    try heading(p, "Options:");
    try row(p, "--prompt <text>", "the user turn (one of the three sources, required)");
    try row(p, "--prompt-file <path>", "the same, read from a file");
    try row(p, "--prompt-tokens <json>", "a JSON array of token ids fed untokenized");
    try row(p, "--raw", "skip the chat template; send the text as written");
    try row(p, "--image <path>", "attach an image (repeatable, up to 8); needs a");
    try more(p, "projector (models.<name>.mmproj) for the model");
    try row(p, "--audio <path>", "attach a clip after the images (repeatable, up");
    try more(p, "to 8, at most 30 s each); Gemma 4 E4B's projector");
    try row(p, "--image-max-tokens <n>", "most tokens one image becomes: auto (the");
    try more(p, "projector's maximum) or a count; default auto");
    try row(p, "--think <effort>", "off, low, medium, high, xhigh; default off");
    try row(p, "--logits <path>", "write the final prompt logits, F32 little-endian");
    try row(p, "--trace-dir <dir>", "write every layer's output as F32 (slow)");
    try row(p, "--json", "the run's report (tokens, timings) instead of text");
    try engineOptions(p);
    try samplingOptions(p);

    try heading(p, "Examples:");
    try example(p, "nuclis generate --prompt \"Write a haiku about Zig\"", "the configured model, its profile's sampling");
    try example(p, "nuclis generate --prompt-file q.txt --think high --max-tokens 4096", "a long reasoned answer");
    try example(p, "nuclis generate --prompt \"Hello,\" --raw --temperature 0 --max-tokens 8", "greedy continuation of literal text");

    try heading(p, "Notes:");
    try plain(p, "The prompt is rendered as one user turn with the profile pinned to the");
    try plain(p, "file's own chat template, and sampled with that profile's defaults for the");
    try plain(p, "reasoning mode; each sampling flag overrides one option. Text streams as");
    try plain(p, "valid UTF-8 even when a token ends inside a character. Ctrl-C stops at the");
    try plain(p, "next layer and keeps what was written; a second Ctrl-C ends the process.");
    try p.end();
}

fn bench(p: *Page) !void {
    try title(p, "nuclis bench", "repeated prefill and decode measurements on one loaded model");
    try heading(p, "Usage:");
    try code(p, "nuclis bench --prompt <text> [flags]");
    try code(p, "nuclis bench --prompt-file <path> [flags]");
    try code(p, "nuclis bench --prompt-tokens <json> [flags]");

    try heading(p, "Options:");
    try row(p, "--prompt <text>", "the prompt (one of the three sources, required)");
    try row(p, "--prompt-file <path>", "the same, read from a file");
    try row(p, "--prompt-tokens <json>", "a JSON array of ids, so a reference's exact input");
    try more(p, "is measured rather than re-tokenized");
    try row(p, "--raw", "skip the chat template");
    try row(p, "--repeat <n>", "measured runs, 1..100; default 3");
    try row(p, "--warmup <n>", "unmeasured runs first, 0..100; default 1");
    try row(p, "--profile", "per-kernel GPU time (Metal); perturbs the rates");
    try row(p, "--unfused-norms", "run the norm pairs the fused kernels replace");
    try row(p, "--profile-prefill", "--profile with each run's prompt kept in it");
    try row(p, "--expert-counts", "count the experts each batch routes to (MoE)");
    try row(p, "--kernel-stats", "instead of benchmarking, print each Metal");
    try more(p, "pipeline's thread limit, SIMD width, and static");
    try more(p, "threadgroup memory (no model or prompt)");
    try row(p, "--capture <path>", "after the runs, one more decode step into a");
    try row(p, "", "new .gputrace for Xcode (MTL_CAPTURE_ENABLED=1)");
    try row(p, "--prefix-cache <dir>", "save the prefilled prompt there and restore it");
    try more(p, "on later runs with the same model, tokens, and");
    try more(p, "session layout (prefill rates are then omitted)");
    try row(p, "--verify-rows <R>", "instead of decoding, time --max-tokens verify");
    try more(p, "batches of R rows (a seed and R-1 fixed drafts)");
    try more(p, "from the prompt's depth, split into propose,");
    try more(p, "checkpoint, verify, recover, and commit; needs");
    try more(p, "--speculative on");
    try row(p, "--accept <a>", "drafts each verify batch accepts, 0..R-1 (0)");
    try row(p, "--draft-p-min <p>", "the drafter's proposal threshold, 0..1, for the");
    try more(p, "speculative runs (0.7; 0 proposes every position)");
    try row(p, "--json", "the report: every sample and the measured means");
    try engineOptions(p);
    try samplingOptions(p);

    try heading(p, "Examples:");
    try example(p, "nuclis bench --prompt-file prompt.txt --max-tokens 128 --json", "three measured runs after one warmup");
    try example(p, "nuclis bench --prompt-tokens p.json --ctx-size 32768 --repeat 5", "a pinned token array at full context");
    try example(p, "nuclis bench --prompt \"Hi\" --speculative on --draft-length 4", "each run measured with the switch off and on");
    try example(p, "nuclis bench --prompt-tokens p.json --speculative on --verify-rows 4", "a 4-row verify batch's cost at the prompt's depth");
    try example(p, "nuclis bench --kernel-stats", "which kernels' registers cap their threadgroups");

    try heading(p, "Notes:");
    try plain(p, "Greedy with 32 output tokens by default, never the file's sampling and");
    try plain(p, "never a profile, so a run measures the engine and not a sampler; the");
    try plain(p, "model, backend, and context come from the config file. Every sample and");
    try plain(p, "the measured means are reported; a rate that could not be measured is");
    try plain(p, "omitted, never estimated. With --speculative on, each run is done both");
    try plain(p, "ways on one loaded model and the report carries the decode speedup. A");
    try plain(p, "profiled run records one encoder per dispatch; its rates are not");
    try plain(p, "comparable with unprofiled ones.");
    try p.end();
}

fn eval(p: *Page) !void {
    try title(p, "nuclis eval", "teacher-forced perplexity of a text file");
    try heading(p, "Usage:");
    try code(p, "nuclis eval --file <path> [flags]");

    try heading(p, "Options:");
    try row(p, "--file <path>", "the text, read raw (no template), up to 16 MiB");
    try row(p, "--ctx-size <n>", "tokens per window, 2.." ++ max_ctx ++ "; default 512, or the");
    try more(p, "reference's");
    try row(p, "--chunks <n>", "windows from the start of the text; default every");
    try more(p, "whole window, or the reference's; fewer than the");
    try more(p, "reference's compare with its running value");
    try row(p, "--reference <json>", "a pinned run on the same file; fails beyond");
    try more(p, "0.5 % of its perplexity, or its bound");
    try row(p, "--write-reference <json>", "write this run as a reference (never over");
    try more(p, "an existing file): the result, the text's and");
    try more(p, "the model's digests, the revision, and the flags");
    try row(p, "--model <name|path>", "registry entry, catalogue name, or path; default");
    try more(p, "engine.model of the config file (qwen3.8-27b)");
    try row(p, "--backend cpu|metal", default_backend);
    try row(p, "--kv f16|f32|q8_0", "attention cache precision on the GPU; default f16");
    try row(p, "--prompt-profile <p>", "force qwen38, gemma4, gemma4_e, or muse_glimmer");
    try more(p, "(its BOS)");
    try row(p, "--json", "the report: every window's running perplexity");

    try heading(p, "Examples:");
    try example(p, "nuclis eval --file wiki.test.raw --chunks 8", "the first 4,096 tokens in windows of 512");
    try example(p, "nuclis eval --file wiki.test.raw --reference ref.json", "the reference's windows, compared with its result");

    try heading(p, "Notes:");
    try plain(p, "Each window starts from an empty cache with the model's BOS over its first");
    try plain(p, "token, and only the second half is scored, each position against the");
    try plain(p, "text's actual next token: the llama-perplexity method, so the numbers");
    try plain(p, "compare with its output. The perplexity is exp of the mean negative");
    try plain(p, "log-likelihood; its error is the standard error of that mean, scaled.");
    try plain(p, "A reference names the text's sha256; a different file is refused.");
    try p.end();
}

fn tokenize(p: *Page) !void {
    try title(p, "nuclis tokenize", "the prompt as the model receives it, without running it");
    try heading(p, "Usage:");
    try code(p, "nuclis tokenize --prompt <text> [flags]");
    try code(p, "nuclis tokenize --prompt-file <path> [flags]");

    try heading(p, "Options:");
    try row(p, "--prompt <text>", "the user turn (one of the two sources, required)");
    try row(p, "--prompt-file <path>", "the same, read from a file");
    try row(p, "--raw", "tokenize the text as written, without the template");
    try row(p, "--think <effort>", "render the turn at this effort; default off");
    try row(p, "--prompt-profile <p>", "force qwen38, gemma4, gemma4_e, or muse_glimmer");
    try row(p, "--model <name|path>", "whose vocabulary and template; default engine.model");
    try row(p, "--json", "ids, offsets, and the rendered text as JSON");

    try heading(p, "Examples:");
    try example(p, "nuclis tokenize --prompt \"Hello, world\"", "the rendered turn with one id per line");
    try example(p, "nuclis tokenize --prompt-file p.txt --raw --json", "exact ids and byte offsets for a script");

    try heading(p, "Notes:");
    try plain(p, "Renders exactly what generate and bench would feed (raw, or one user turn");
    try plain(p, "at the given effort) and prints each token's id with its byte offset in");
    try plain(p, "the rendered text. Only the artifact's header is read, so it is fast");
    try plain(p, "enough to use while writing a prompt.");
    try p.end();
}

fn decide(p: *Page) !void {
    try title(p, "nuclis decide", "typed questions about states, answered by a decision model");
    try heading(p, "Usage:");
    try code(p, "nuclis decide --request <file|-> [flags]");
    try code(p, "nuclis decide --questions <file> <states> [flags]");
    try code(p, "nuclis decide <inline questions> <states> [flags]");
    try code(p, "  <states>: --state <text> and --state-file <path>, repeatable");
    try code(p, "  <inline questions>: --choice <text> --option <key[=desc]>…,");
    try code(p, "  --score <text> --level <text>…, --noul <text>; each [--id <name>]");

    try heading(p, "Questions and states:");
    try row(p, "--request <file|->", "a Jev request: \"questions\" and \"state\" or");
    try more(p, "\"states\"; a state is text, JSON, or {\"file\": path}");
    try row(p, "--questions <file>", "the questions object (id -> definition) alone");
    try row(p, "--state <text>", "a state, repeatable (a question per state each)");
    try row(p, "--state-file <path>", "a state read from a file, repeatable");
    try row(p, "--image <path>", "an image read before every state, repeatable");
    try more(p, "(clef-flash, pulled --with mmproj; at most 8)");
    try row(p, "--choice <text>", "a choice question; its --option key[=description]s");
    try row(p, "--score <text>", "a score question; its --level texts, 0 first");
    try row(p, "--noul <text>", "a yes/no question: the probability it holds");
    try row(p, "--id <name>", "names the question before it; default its type");

    try heading(p, "Options:");
    try row(p, "--model <name|path>", "a decision entry or checkpoint directory;");
    try more(p, "default decide.model (laya, English);");
    try more(p, "laya-multilingual reads other languages;");
    try more(p, "clef-flash: a 9B model, every question in one pass");
    try row(p, "--truncate head|tail", "the end of a long state that is cut; default tail,");
    try more(p, "head for a JSON list (a conversation)");
    try row(p, "--backend cpu|metal", default_backend);
    try row(p, "--uncalibrated", "softmax of the raw logits, no temperature");
    try row(p, "--explain", "each sequence decoded, its budget, the logits");
    try row(p, "--json", "one Jev response per state, with timings");

    try heading(p, "Examples:");
    try example(p, "nuclis decide --noul \"Will they cancel?\" --state \"Refund me or I go.\"", "one probability");
    try example(p, "nuclis decide --noul \"A failure?\" --state-file a.log --state-file b.log", "the files ranked by it");
    try example(p, "nuclis decide --request ticket.json --json", "the Jev response, for a script");

    try heading(p, "Notes:");
    try plain(p, "Each question is one encoder pass over the question, its options, and the");
    try plain(p, "state (512 tokens in all, 1,024 for laya-multilingual; a longer state is");
    try plain(p, "cut and flagged). Several states rank by the first question: P(true),");
    try plain(p, "the expected score, or the first option's probability. A request's");
    try plain(p, "\"model\" field is ignored here; `nuclis serve` reads it. clef-flash");
    try plain(p, "reads a state and every question in one pass of up to 16,384 tokens, its");
    try plain(p, "answers uncalibrated (a softmax per question), its images from --image or");
    try plain(p, "a request's \"images\" (data URLs, base64, or {\"file\": path}).");
    try p.end();
}

fn embed(p: *Page) !void {
    try title(p, "nuclis embed", "text, images and audio into unit vectors, by an embedding model");
    try heading(p, "Usage:");
    try code(p, "nuclis embed <text>… [flags]");
    try code(p, "nuclis embed --image <file>… --audio <file>… [flags]");
    try code(p, "nuclis embed --input-file <file|-> [flags]");
    try code(p, "  each argument is one input: quote a sentence");

    try heading(p, "Inputs:");
    try row(p, "<text>", "an input, repeatable; after --, words that start");
    try more(p, "with - are text too");
    try row(p, "--image <file>", "an image as one input, repeatable (PNG, JPEG,");
    try more(p, "HEIC, WebP, TIFF, GIF, BMP); needs the projector");
    try more(p, "(nuclis model pull embeddinggemma-2 --with mmproj)");
    try row(p, "--audio <file>", "a clip as one input, repeatable (WAV, AIFF, MP3,");
    try more(p, "M4A, FLAC, CAF), at most 30 s; needs the projector");
    try row(p, "--input-file <file|->", "one input per line: a JSON string, or a list of");
    try more(p, "content parts ({\"type\": \"text\", \"text\": …},");
    try more(p, "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:…\"}},");
    try more(p, "{\"type\": \"input_audio\", \"input_audio\": {\"data\": …,");
    try more(p, "\"format\": \"wav\"}})");

    try heading(p, "Options:");
    try row(p, "--task <task>", "open each input with the model's prefix:");
    try more(p, "search_query, document, question_answering,");
    try more(p, "fact_checking, code_retrieval, classification,");
    try more(p, "clustering, similarity; default none (as given)");
    try row(p, "--title <text>", "with --task document: the document's title");
    try row(p, "--dimensions <n>", "768, 512, 256, or 128 (the leading values,");
    try more(p, "renormalized); default 768");
    try row(p, "--image-tokens <n>", "soft tokens per image: 70, 140, 280, 560, or");
    try more(p, "1120; default 280 (more is finer, and slower)");
    try row(p, "--truncate", "cut an input over 8,192 tokens instead of refusing");
    try more(p, "it, and a clip over 30 s to its first 30 s");
    try row(p, "--model <name|path>", "an embedding entry or GGUF file; default");
    try more(p, "embed.model (embeddinggemma-2)");
    try row(p, "--backend cpu|metal", default_backend);
    try row(p, "--json", "every vector with its tokens, space, and timings");

    try heading(p, "Examples:");
    try example(p, "nuclis embed --task search_query \"how do auroras form?\"", "a query vector");
    try example(p, "nuclis embed \"a cat\" \"a kitten\" \"a tax form\"", "three vectors and their cosines");
    try example(p, "nuclis embed --image cat.jpg \"a photo of a cat\"", "an image and a caption, crossed");
    try example(p, "nuclis embed --audio memo.m4a \"the shopping list\"", "a voice memo against a text");
    try example(p, "nuclis embed --task document --input-file docs.jsonl --json", "document vectors, for an index");

    try heading(p, "Notes:");
    try plain(p, "Text is embedded exactly as given unless --task names a use: queries and");
    try plain(p, "documents are embedded with different prefixes, and only vectors of one");
    try plain(p, "space compare (the file's digest and the width, printed as `space`).");
    try plain(p, "Text, images and audio share the space; the image budget changes the");
    try plain(p, "vector. Audio is resampled to 16 kHz mono (16 kHz mono WAV is read as is).");
    try plain(p, "Inputs are packed into batches on Metal; batching never changes a vector.");
    try p.end();
}

fn transcribe(p: *Page) !void {
    try title(p, "nuclis transcribe", "speech to text, by Nemotron 3.5 ASR streaming");
    try heading(p, "Usage:");
    try code(p, "nuclis transcribe <file> [flags]");

    try heading(p, "Options:");
    try row(p, "--stream", "feed the clip in 80 ms pieces, as a microphone");
    try more(p, "would, and print the transcript as it changes");
    try row(p, "--lookahead <n>", "rows of 80 ms each chunk waits for: 0, 3, 6,");
    try more(p, "or 13 (latency 80, 320, 560, 1120 ms); default 3");
    try row(p, "--language <code>", "the language to expect, a locale (en-US, de-DE,");
    try more(p, "fr, es-US, ...); default auto, which detects it.");
    try more(p, "Up to 4 as a list (ar,en-US,auto): each decodes,");
    try more(p, "the words take the one that writes them, and auto,");
    try more(p, "only last, is used when none of the others does");
    try row(p, "--phrase <words>", "a name or words to expect (Siobhan, Dun");
    try more(p, "Laoghaire); repeat it, at most 64 times");
    try row(p, "--model <name|dir>", "nemotron-asr-0.6b, a transcription registry");
    try more(p, "entry, or a checkpoint directory; default");
    try more(p, "transcribe.model (nemotron-asr-0.6b)");
    try row(p, "--backend cpu|metal", default_backend);
    try row(p, "--format <f>", "text (default); json: the text, language, tokens");
    try more(p, "with times, timings; srt or vtt: subtitles");
    try row(p, "--segments", "the segments: sentences, or speech between");
    try more(p, "pauses, with their times (and words, in json)");
    try row(p, "--alternatives", "each listed language's reading, in json");
    try row(p, "--json", "the same as --format json");

    try heading(p, "Examples:");
    try example(p, "nuclis transcribe memo.m4a", "the transcript and its language");
    try example(p, "nuclis transcribe talk.wav --stream --lookahead 0", "partials as fast as they come");
    try example(p, "nuclis transcribe clip.flac --language de-DE --json", "German, with token times");
    try example(p, "nuclis transcribe turn.wav --language ar,en-US,auto", "Arabic or English, else any language");
    try example(p, "nuclis transcribe lecture.m4a --format srt > lecture.srt", "subtitles");

    try heading(p, "Notes:");
    try plain(p, "The model writes punctuated, capitalized text. Under auto it may end a");
    try plain(p, "sentence with a language tag, which is reported as the language and");
    try plain(p, "never printed in the text. Audio is resampled to 16 kHz mono (a 16 kHz");
    try plain(p, "mono WAV, 16-bit or float, is read as is). A token's time is the start");
    try plain(p, "of its 80 ms row. A long recording is decoded and fed in 2.56 s pieces,");
    try plain(p, "as `nuclis serve` does, so memory does not grow with its length (files");
    try plain(p, "up to 256 MB). Pull the model with `nuclis model pull");
    try plain(p, "nemotron-asr-0.6b`.");
    try p.end();
}

fn speak(p: *Page) !void {
    try title(p, "nuclis speak", "text into speech, by Qwen3-TTS");
    try heading(p, "Usage:");
    try code(p, "nuclis speak <text>… -o <file.wav> [flags]");
    try code(p, "  the words are joined by spaces: quote a sentence");

    try heading(p, "Options:");
    try row(p, "-o, --output <file>", "the WAV to write: 24 kHz mono 16-bit");
    try row(p, "--voice <name>", "on CustomVoice one of its speakers, Qwen's nine:");
    try more(p, "aiden, dylan, eric, ono_anna, ryan, serena,");
    try more(p, "sohee, uncle_fu, vivian; on a Base a saved voice");
    try more(p, "(nuclis voice); default speech.voice (sohee)");
    try row(p, "--language <name>", "english, chinese, german, spanish, french,");
    try more(p, "italian, portuguese, russian, japanese, korean,");
    try more(p, "or auto; default auto");
    try row(p, "--instruct <text>", "how to say it (\"Speak in a warm, cheerful");
    try more(p, "tone.\"), on the 1.7B CustomVoice; on VoiceDesign");
    try more(p, "the voice itself (\"a calm elderly man, deep\");");
    try more(p, "on voicetut-tts attribute tags (\"male, whisper\")");
    try row(p, "--model <name|dir>", "qwen3-tts-1.7b-customvoice, voicetut-tts,");
    try more(p, "drama-3, qwen3-tts-*-base/-voicedesign, a speech");
    try more(p, "registry entry,");
    try more(p, "or a checkpoint directory; default speech.model");
    try more(p, "(qwen3-tts-1.7b-customvoice)");
    try row(p, "--seed <n>", "the sampler's seed; default 0");
    try row(p, "--speed <rate>", "speaking rate multiplier (> 0, default 1.0;");
    try more(p, "voicetut-tts only)");
    try row(p, "--backend cpu|metal", default_backend);
    try row(p, "--stream", "write each chunk as it is decoded; report the");
    try more(p, "time from the text to the first audio");
    try row(p, "--chunk <n>", "frames of 80 ms per chunk, 1 to 32; default 4");
    try row(p, "--first-chunk <n>", "frames in the first chunk; default 1");
    try row(p, "--split", "time the talker and the predictor apart (on");
    try more(p, "Metal, one more command buffer a frame)");
    try row(p, "--host-sampling", "on Metal, sample each code group on the CPU");
    try more(p, "between command buffers: the same audio, slower");
    try row(p, "--profile", "time every Metal dispatch; print the kernels");

    try heading(p, "Examples:");
    try example(p, "nuclis speak \"Hello there.\" -o hello.wav", "the default voice");
    try example(p, "nuclis speak --voice vivian --language chinese \"你好。\" -o hi.wav", "another speaker");
    try example(p, "nuclis speak --instruct \"Whisper.\" \"Good night.\" -o night.wav", "a mood, on the 1.7B");
    try example(p, "nuclis speak --stream \"One moment, please.\" -o wait.wav", "chunk by chunk, timed");
    try example(p, "nuclis speak --model qwen3-tts-1.7b-base --voice anna \"Hi.\" -o hi.wav", "a saved voice");
    try example(p, "nuclis speak --model voicetut-tts \"أهلاً بيك في القاهرة.\" -o cairo.wav", "Egyptian Arabic");

    try heading(p, "Notes:");
    try plain(p, "Speech is sampled: the same text, voice, and seed give the same audio,");
    try plain(p, "another seed says it differently. The whole text is read before the");
    try plain(p, "first frame, so give it a sentence or two at a time. The CPU and Metal");
    try plain(p, "agree to rounding, so a seed gives the same speech on both only nearly.");
    try plain(p, "A saved voice is encoded the first time a Base speaks in it (a second");
    try plain(p, "or less on the CPU), and kept. On qwen3-tts-1.7b-voicedesign,");
    try plain(p, "--instruct describes the voice itself (\"a calm elderly man with a deep");
    try plain(p, "voice\"), sampled anew each time; for one that stays the same, save a");
    try plain(p, "clip of it with `nuclis voice add` and speak in it on a Base.");
    try p.end();
}

fn voice(p: *Page) !void {
    try title(p, "nuclis voice", "save a clip as a voice for cloning speech models");
    try heading(p, "Usage:");
    try code(p, "nuclis voice add <name> <audio> [--text <transcript>] [--json]");
    try code(p, "nuclis voice list [--json]");
    try code(p, "nuclis voice rm <name> [--json]");

    try heading(p, "Actions:");
    try row(p, "add", "save <audio> (WAV, AIFF, MP3, M4A, FLAC, CAF; 1");
    try more(p, "to 30 s) as <name>: a-z, 0-9, _ and -, up to 64,");
    try more(p, "not one of Qwen's speakers");
    try row(p, "list", "every saved voice, its length and transcript");
    try row(p, "rm", "delete a voice and all that was made of it");

    try heading(p, "Options:");
    try row(p, "--text <transcript>", "what the clip says, up to 2 KiB: the closest");
    try more(p, "likeness; without it, the speaker's timbre alone");
    try more(p, "(Qwen3-TTS Base only: Drama 3 and VoiceTut-TTS need it)");
    try row(p, "--json", "the voice, the list, or the removal as JSON");

    try heading(p, "Examples:");
    try example(p, "nuclis voice add anna anna.m4a --text \"Hello, it's Anna.\"", "clip and words");
    try example(p, "nuclis speak --model qwen3-tts-1.7b-base --voice anna \"Hi.\" -o hi.wav", "use it");
    try example(p, "nuclis speak --model drama-3 --voice anna \"[warmly] Hi.\" -o hi.wav", "or with Drama 3");
    try example(p, "nuclis voice rm anna", "forget it");

    try heading(p, "Notes:");
    try plain(p, "A voice is kept under ~/.nuclis/voices/<name>: the clip at 24 and 44.1 kHz");
    try plain(p, "and what each model made of it, made on its first use with that model. A");
    try plain(p, "clean clip of a few sentences, with its exact words, clones best. Clone");
    try plain(p, "only a voice you have the right to use.");
    try p.end();
}

fn serve(p: *Page) !void {
    try title(p, "nuclis serve", "the nuclis API: decision, language, embedding, speech models");
    try heading(p, "Usage:");
    try code(p, "nuclis serve [--host <ip>] [--port <n>] [--model <name>]… [--pin <name>]…");
    try code(p, "             [--memory <GiB>] [--backend cpu|metal] [--timeout <s>]");
    try code(p, "             [--web-url <url> | --no-web]");

    try heading(p, "Options:");
    try row(p, "--host <ip>", "the address to listen on; default serve.host,");
    try more(p, "127.0.0.1 (any other prints a warning: there is no");
    try more(p, "authentication)");
    try row(p, "--port <n>", "default serve.port, 9000");
    try row(p, "--model <name|path>", "open a model at start, any kind: at most 2");
    try more(p, "decision, 1 chat, 1 embedding, 1 speech, 1");
    try more(p, "transcription (what stays open); without it");
    try more(p, "nothing opens until its first request");
    try row(p, "--pin <name>", "open it at start, and never close it to make");
    try more(p, "room; repeated; default serve.pin");
    try row(p, "--memory <GiB>", "what every open model may hold together; default");
    try more(p, "serve.memory_bytes, else physical memory less a");
    try more(p, "third of it, at most 16 GiB");
    try row(p, "--backend cpu|metal", default_backend);
    try row(p, "--quiet", "no line per request (serve.log false does the same)");
    try row(p, "--timeout <s>", "how long a request may wait for the GPU before");
    try more(p, "529 timeout; default serve.timeout, 300");
    try row(p, "--web-url <url>", "the hosted web tools on, against this search");
    try more(p, "service (https, http only on loopback); the");
    try more(p, "config file is not read for them nor written");
    try row(p, "--no-web", "the hosted web tools off, whatever serve.web says");

    try heading(p, "Routes:");
    try row(p, "POST /v1/decisions", "a `decide --request` body (\"model\" picks the");
    try more(p, "model); the `decide --json` response; ?explain=1");
    try row(p, "POST /v1/systemone", "TypeSafe's Jev call: one state, Jev's answers;");
    try more(p, "\"jev-latest\" (any jev- id) is decide.model");
    try row(p, "POST /v1/chat/completions", "OpenAI's Chat Completions for the language");
    try more(p, "models; a client of an OpenAI-compatible server");
    try more(p, "connects by changing its base URL");
    try row(p, "POST /v1/embeddings", "OpenAI's embeddings for the embedding models;");
    try more(p, "\"task\", \"title\", \"truncate\", \"image_tokens\" as in");
    try more(p, "nuclis embed; image_url parts as base64 data URLs,");
    try more(p, "input_audio parts as base64 with a format");
    try row(p, "POST /v1/audio/speech", "OpenAI's speech: \"input\" spoken by \"voice\",");
    try more(p, "pcm or wav, streamed as raw audio or as events");
    try row(p, "GET /v1/models", "every model of every kind, OpenAI's list");
    try more(p, "shape");
    try row(p, "GET /v1/health", "version, backend, open models, queue depth");

    try heading(p, "Trust:");
    try plain(p, "There is no authentication: any client that can connect may call every");
    try plain(p, "route and name any registry or catalogue model. The default host is");
    try plain(p, "loopback; beyond it, put an authenticating reverse proxy in front.");

    try heading(p, "Examples:");
    try example(p, "nuclis serve", "listen on 127.0.0.1:9000, nothing open yet");
    try example(p, "curl -s localhost:9000/v1/decisions -d @ticket.json", "one decision request");
    try example(p, "nuclis serve --model laya --model qwen3.8-27b", "a decision and a language model open");

    try heading(p, "Notes:");
    try plain(p, "Requests run on the GPU one at a time in arrival order, decision and");
    try plain(p, "embedding passes between a generation's steps; a request waits at most");
    try plain(p, "300 s to start (then 529 timeout), at most 64 wait (then 529 busy).");
    try plain(p, "Opening a model closes the least recently used others, of any kind, until");
    try plain(p, "it fits the memory budget. Bodies up to 4 MiB (32 MiB for chat and");
    try plain(p, "embeddings); states are text or JSON, never {\"file\": path}; a chat or");
    try plain(p, "embedding request names a model, never a path.");
    try plain(p, "Errors are {\"error\": {\"code\", \"message\", \"type\", \"param\"}} with a");
    try plain(p, "fitting HTTP status.");
    try plain(p, "Ctrl-C stops accepting, finishes queued decisions and embeddings (up to");
    try plain(p, "10 s), and exits; a second Ctrl-C ends the process at once.");
    try p.end();
}
fn inspect(p: *Page) !void {
    try title(p, "nuclis inspect", "what an artifact is");
    try heading(p, "Usage:");
    try code(p, "nuclis inspect [--model <name|path>] [--metadata] [--tensors]");
    try code(p, "               [--tensor <name>] [--json]");

    try heading(p, "Options:");
    try row(p, "--model <name|path>", "the file to read; default engine.model; a");
    try more(p, "safetensors file, index, or directory too");
    try row(p, "--metadata", "GGUF: every metadata key, its type, and its value");
    try more(p, "(long strings cut, vocabulary arrays counted)");
    try row(p, "--tensors", "GGUF: every tensor's encoding, shape, size, and");
    try more(p, "offset in the file");
    try row(p, "--tensor <name>", "one tensor's encoding or dtype, shape, and first");
    try more(p, "values (decoded from any encoding nuclis reads)");
    try row(p, "--json", "the same facts as JSON");

    try heading(p, "Examples:");
    try example(p, "nuclis inspect", "the configured model");
    try example(p, "nuclis inspect --model ~/Downloads/some-model.gguf --json", "any GGUF file, for a script");
    try example(p, "nuclis inspect --model clef-flash --tensors", "every tensor of the file");

    try heading(p, "Notes:");
    try plain(p, "Reads the GGUF directory only: identity, architecture, dimensions, the");
    try plain(p, "tensor encoding histogram, and the memory a context would need. Ranges");
    try plain(p, "and offsets are validated; weight values are not read (--tensor reads one");
    try plain(p, "row). Little-endian GGUF");
    try plain(p, "v3 only; an unknown tensor layout is an error rather than a guess.");
    try plain(p, "A safetensors checkpoint gets its header, shards, and dtype histogram,");
    try plain(p, "with the byte buffer's tiling checked. No text-generation family runs");
    try plain(p, "one; `decide`, `speak`, and `transcribe` do.");
    try p.end();
}

fn validate(p: *Page) !void {
    try title(p, "nuclis validate", "will this file run");
    try heading(p, "Usage:");
    try code(p, "nuclis validate [--model <name|path>] [--json]");

    try heading(p, "Options:");
    try row(p, "--model <name|path>", "the file to check; default engine.model");
    try row(p, "--json", "the verdict as JSON");

    try heading(p, "Examples:");
    try example(p, "nuclis validate --model gemma-4-e4b-qat", "bind an index model's weights");

    try heading(p, "Notes:");
    try plain(p, "Checks the file against its architecture's adapter (qwen35, gemma4,");
    try plain(p, "muse-glimmer) and binds the text-layer weights, so a missing tensor or an");
    try plain(p, "unsupported encoding is named here instead of at the first token. It does");
    try plain(p, "not verify tokenizer behaviour, weight values, or numerical execution.");
    try p.end();
}

fn model(p: *Page) !void {
    try title(p, "nuclis model", "artifacts from the Hugging Face Hub");
    try heading(p, "Usage:");
    try code(p, "nuclis model ls [-l] [--available [--all]] [--json]");
    try code(p, "nuclis model pull <name> [--with mmproj,draft[=<v>] | --all [--variants]]");
    try code(p, "                  [--force] [--json]");
    try code(p, "nuclis model pull --catalogue [--force] [--json]");
    try code(p, "nuclis model pull <owner/repo> [--file <name>] [--revision <rev>] [--role <r>]");
    try code(p, "                  [--register <name> [--profile <p>]] [--force] [--json]");
    try code(p, "nuclis model inspect <name> [--metadata] [--tensors] [--tensor <name>]");
    try code(p, "                     [--json]");
    try code(p, "nuclis model inspect <owner/repo> --file <name> [--revision <rev>]");
    try code(p, "                     [--metadata] [--tensors] [--tensor <name>] [--json]");
    try code(p, "nuclis model adopt <path> [--force] [--json]");
    try code(p, "nuclis model rm <name> [--with mmproj,draft | --all] [--json]");

    try heading(p, "Commands:");
    try row(p, "ls", "what is installed under ~/.nuclis/models, one line");
    try more(p, "a model, by kind; a file outside the index is a row");
    try more(p, "of its own; --available lists the index instead");
    try row(p, "pull", "fetch an index model, a registry entry, or any");
    try more(p, "artifact of a repository, verified by digest and");
    try more(p, "published atomically, with a safetensors set's files");
    try row(p, "inspect", "read a remote file's header (at most 64 MiB, no");
    try more(p, "weights) and say whether it would run: supported,");
    try more(p, "runnable, or not");
    try row(p, "adopt", "hash a file placed by hand, match it to the index's");
    try more(p, "pinned file of that size and digest, and write its");
    try more(p, "sidecar; an unmatched file keeps its local digest");
    try row(p, "rm", "delete an index or registry entry's files and");
    try more(p, "sidecars; a file another present entry uses stays,");
    try more(p, "and the registry entry is not edited");

    try heading(p, "Options:");
    try row(p, "-l, --long", "ls: each file's commit and digest (12 characters),");
    try more(p, "its path, and the defaults the model runs with");
    try row(p, "--available", "ls: the index, pulled or not; --all adds legacy");
    try row(p, "--with mmproj,draft", "also fetch (rm: also delete) those companions;");
    try more(p, "draft=<variant> picks another drafter (qwen3.8-27b:");
    try more(p, "mtp, dflash2); mtp is the old name of draft");
    try row(p, "--all", "every companion the entry names, one per role;");
    try more(p, "with --variants, every draft variant");
    try row(p, "--catalogue", "pull: every current entry, with --all (about 42 GB)");
    try row(p, "--file <name>", "which weight file (.gguf, .safetensors); required");
    try more(p, "when several; its support files come with it");
    try row(p, "--revision <rev>", "a branch, tag, or commit; default main");
    try row(p, "--role <r>", "main, mmproj, draft, or imatrix; default by content");
    try row(p, "--metadata", "inspect: every metadata key, its type, and its value");
    try row(p, "--tensors", "inspect: every tensor's encoding, shape, and offset");
    try row(p, "--tensor <name>", "inspect: one tensor and its first values (reads that");
    try more(p, "tensor's first row, no other weights)");
    try row(p, "--register <name>", "once verified, write the pull as a registry entry");
    try more(p, "(repo, file, commit); a companion fills the same entry");
    try row(p, "--profile <p>", "with --register: force qwen38, gemma4,");
    try more(p, "gemma4_e, or muse_glimmer");
    try row(p, "--force", "download every file again, replacing each once the");
    try more(p, "copy verifies; adopt: replace a sidecar already there");
    try row(p, "--json", "the listing, pull, verdict, or removal as JSON");
    try more(p, "(a pull from several repositories: an array)");

    try heading(p, "Examples:");
    try example(p, "nuclis model pull qwen3.8-27b --all", "the default model, its projector and drafter");
    try example(p, "nuclis model pull --catalogue", "every current model, then nuclis config init");
    try code(p, "nuclis model pull unsloth/gemma-4-12b-it-GGUF \\");
    try example(p, "    --file gemma-4-12b-it-UD-Q4_K_XL.gguf --register gemma-fast", "any file of a repository, under a name");
    try code(p, "nuclis model pull convaiinnovations/laya --file model.safetensors");
    try more(p, "a safetensors set with its config and tokenizer");
    try example(p, "nuclis model inspect unsloth/Muse-Glimmer-30B-GGUF --file dflash-kquant.gguf", "will it run, before the download");
    try example(p, "nuclis model adopt unsloth/Qwen3.8-27B-GGUF/MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf", "a copy placed by hand");
    try example(p, "nuclis model rm qwen3.8-27b --all", "the model, its projector and drafters");

    try heading(p, "Notes:");
    try plain(p, "Downloads go over Xet, verify the digest, publish atomically, and leave a");
    try plain(p, "<file>.nuclis.json sidecar beside the file; a second pull of the same file");
    try plain(p, "downloads nothing. Ctrl-C stops a pull and removes the partial file; SIGTERM");
    try plain(p, "stops it and keeps <file>.partial, which the next pull continues.");
    try plain(p, "HF_TOKEN is read for gated repositories only. A model a running serve has");
    try plain(p, "open keeps working after rm; its space is freed when the server closes it.");
    try p.end();
}

fn config(p: *Page) !void {
    try title(p, "nuclis config", "the file every command reads (~/.nuclis/nuclis.json)");
    try heading(p, "Usage:");
    try code(p, "nuclis config init [--dry-run] [--json]");
    try code(p, "nuclis config sync [--prune] [--dry-run] [--json]");
    try code(p, "nuclis config show [--json]");
    try code(p, "nuclis config set <key> <value>");

    try heading(p, "Commands:");
    try row(p, "init", "write the file from what is installed: the defaults,");
    try more(p, "an entry for each installed model that needs one, and");
    try more(p, "engine.model the default (qwen3.8-27b) or, when it");
    try more(p, "is not installed, the first installed chat model; an");
    try more(p, "existing file is kept");
    try row(p, "sync", "add entries for installed models that have none, set");
    try more(p, "index entries' measured defaults to the index's, add");
    try more(p, "keys a newer nuclis introduced, and report the");
    try more(p, "entries whose file is gone");
    try row(p, "show", "every key's effective value and where it came from:");
    try more(p, "default, profile, file, model entry, or flag");
    try row(p, "set", "change one key by its dotted name; the file is");
    try more(p, "validated before it is written, and left as it was");
    try more(p, "on a refused value");

    try heading(p, "Options:");
    try row(p, "--prune", "sync: also remove the entries whose file is gone");
    try row(p, "--dry-run", "print what init or sync would write; write nothing");
    try row(p, "--json", "show, or the init or sync report, as JSON");
    try row(p, "--discover", "init: the old spelling of what init now does");

    try heading(p, "Keys:");
    try configKeys(p);
    try row(p, "models.<name>.*", "path, or repo + file + revision; mmproj, draft (a");
    try more(p, "file; mtp is its old name), profile, ctx_size,");
    try more(p, "generation.*, agent.* for that model only, and");
    try more(p, "generation.draft: an index model's draft variant");
    try more(p, "(qwen3.8-27b: mtp, dflash2). Entries are created by");
    try more(p, "init, sync, and model pull --register, never by set.");
    try plain(p, "A null sampling key takes the profile's value. agent.instructions is");
    try plain(p, "auto (AGENTS.md then CLAUDE.md), off, or a path.");

    try heading(p, "Examples:");
    try example(p, "nuclis config sync --dry-run", "what a sync would change");
    try example(p, "nuclis config set engine.model qwen3.8-27b", "the default model");
    try example(p, "nuclis config set models.qwen3.8-27b.generation.draft dflash2", "another drafter for one model");
    try example(p, "nuclis config set generation.sampling.temperature 0.7", "a sampling override");
    try example(p, "nuclis config set models.mine.profile gemma4", "force a profile on one entry");
    try example(p, "nuclis config set generation.think null", "clear an override");

    try heading(p, "Notes:");
    try plain(p, "Precedence: defaults < the model's profile < the file < the model's");
    try plain(p, "entry < flags. NUCLIS_HOME (an absolute path) moves ~/.nuclis. An unknown");
    try plain(p, "key or an out-of-range value is an error naming the key.");
    try p.end();
}

fn cache(p: *Page) !void {
    try title(p, "nuclis cache", "the agent's saved model states");
    try heading(p, "Usage:");
    try code(p, "nuclis cache [ls] [--json]");
    try code(p, "nuclis cache clear");

    try heading(p, "Options:");
    try row(p, "ls", "each state's size and last use, newest first, and");
    try more(p, "the total against cache.disk_bytes (the default)");
    try row(p, "clear", "delete every saved state");
    try row(p, "--json", "ls as one JSON object");

    try heading(p, "Examples:");
    try example(p, "nuclis cache", "what the cache holds and how much of its budget");
    try example(p, "nuclis cache clear", "free the space; the next start prefills again");

    try heading(p, "Notes:");
    try plain(p, "The agent saves the model's state after the system prompt and tools,");
    try plain(p, "and after each answered turn of a recorded session, under");
    try plain(p, "~/.nuclis/cache/prefix. A start or a resume restores the state instead");
    try plain(p, "of prefilling those tokens again. A state is bound to the model files,");
    try plain(p, "the backend, the context size, and the nuclis build; any other combination");
    try plain(p, "is a miss, never a wrong restore. Least recently used states go first");
    try plain(p, "once cache.disk_bytes (8 GiB) is reached; 0 turns the directory off.");
    try p.end();
}

fn completion(p: *Page) !void {
    try title(p, "nuclis completion", "Tab completion for fish, bash, and zsh");
    try heading(p, "Usage:");
    try code(p, "nuclis completion fish|bash|zsh");

    try heading(p, "Options:");
    try row(p, "fish", "the script for fish");
    try row(p, "bash", "for bash 3.2 and later");
    try row(p, "zsh", "for zsh, after compinit");

    try heading(p, "Examples:");
    try example(p, "nuclis completion fish > ~/.config/fish/completions/nuclis.fish", "fish loads it on the first Tab");
    try example(p, "echo 'source <(nuclis completion bash)' >> ~/.bashrc", "bash, every new shell");
    try example(p, "echo 'source <(nuclis completion zsh)' >> ~/.zshrc", "zsh, after the line that runs compinit");

    try heading(p, "Notes:");
    try plain(p, "The script is a thin shim: each Tab runs nuclis with the words typed so");
    try plain(p, "far and shows its answer, so new flags, registered models, config keys,");
    try plain(p, "and this directory's saved sessions complete without writing the script");
    try plain(p, "again. It calls the nuclis on PATH.");
    try p.end();
}

// ----- tests -----

const testing = std.testing;

fn rendered(alloc: std.mem.Allocator, topic: ?Topic, ui: kit.Ui) ![]u8 {
    var buffer: std.Io.Writer.Allocating = .init(alloc);
    errdefer buffer.deinit();
    try write(&buffer.writer, ui, topic, "0.0.0-test");
    return buffer.toOwnedSlice();
}

test "the overview names every command and stays one screen" {
    const text = try rendered(testing.allocator, null, .plain);
    defer testing.allocator.free(text);
    inline for (@typeInfo(Topic).@"enum".field_names) |name| {
        try testing.expect(std.mem.indexOf(u8, text, "  " ++ name ++ " ") != null);
    }
    try testing.expect(std.mem.startsWith(u8, text, "Usage: nuclis <command> [flags]\n\nA local inference engine"));
    try testing.expect(std.mem.indexOf(u8, text, "Run 'nuclis help <command>' for its flags and examples.") != null);
    try testing.expect(std.mem.count(u8, text, "\n") <= 50);
    for (text) |byte| try testing.expect(byte >= 0x20 or byte == '\n');
}

test "every page is self-contained, has the standard sections, and fits 80 columns" {
    inline for (@typeInfo(Topic).@"enum".field_names) |name| {
        const topic = @field(Topic, name);
        const text = try rendered(testing.allocator, topic, .plain);
        defer testing.allocator.free(text);
        try testing.expect(std.mem.startsWith(u8, text, "Usage:"));
        try testing.expect(std.mem.indexOf(u8, text, "nuclis " ++ name) != null);
        inline for (.{ "Flags:", "Examples:", "Notes:" }) |section| {
            if (std.mem.indexOf(u8, text, "\n" ++ section ++ "\n") == null) {
                std.debug.print("the {s} page lacks {s}\n", .{ name, section });
                return error.TestUnexpectedResult;
            }
        }
        // No pointer at the repository's documents: the page is the help.
        try testing.expect(std.mem.indexOf(u8, text, "docs/") == null);
        try testing.expect(std.mem.indexOf(u8, text, "see ") == null);
        var lines = std.mem.splitScalar(u8, text, '\n');
        var count: usize = 0;
        while (lines.next()) |line| : (count += 1) {
            if ((std.unicode.utf8CountCodepoints(line) catch line.len) > 80) {
                std.debug.print("the {s} page has a line past 80 columns: {s}\n", .{ name, line });
                return error.TestUnexpectedResult;
            }
        }
        try testing.expect(count > 8 and count <= 90);
        try testing.expect(std.mem.endsWith(u8, text, "\n\n"));
    }
}

test "colour is on a terminal only, and never moves a column" {
    const lit: kit.Ui = .{ .paint = .{ .theme = .{ .kind = .truecolor }, .enabled = true } };
    inline for (.{ null, Topic.agent, Topic.model }) |topic| {
        const colored = try rendered(testing.allocator, topic, lit);
        defer testing.allocator.free(colored);
        const piped = try rendered(testing.allocator, topic, .plain);
        defer testing.allocator.free(piped);
        try testing.expect(std.mem.indexOfScalar(u8, colored, 0x1b) != null);
        try testing.expect(std.mem.indexOfScalar(u8, piped, 0x1b) == null);
        // With the escapes taken out, the bytes are the pipe's.
        var stripped: std.ArrayList(u8) = .empty;
        defer stripped.deinit(testing.allocator);
        var i: usize = 0;
        while (i < colored.len) : (i += 1) {
            if (colored[i] == 0x1b) {
                while (i < colored.len and colored[i] != 'm') i += 1;
                continue;
            }
            try stripped.append(testing.allocator, colored[i]);
        }
        try testing.expectEqualStrings(piped, stripped.items);
    }
}

test "rows align per section; a long-only flag sits under the short ones" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var storage: [4096]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&storage);
    var p: Page = .{ .out = &buffer.writer, .ui = .plain, .arena = fba.allocator(), .width = 80 };
    try p.heading("Options:");
    try p.row("-l, --long", "each file's digest");
    try p.row("--json", "the listing as JSON");
    try p.row("--seed <n>", "sampler seed; default 0");
    try p.more("(a second line)");
    try p.end();
    try testing.expectEqualStrings(
        \\
        \\Flags:
        \\  -l, --long      Each file's digest
        \\      --json      The listing as JSON
        \\      --seed <n>  Sampler seed; default 0 (a second line)
        \\
        \\
    , buffer.written());
}

test "a summary becomes a sentence unless it starts with code or a list of values" {
    var storage: [256]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&storage);
    const a = fba.allocator();
    try testing.expectEqualStrings("off, low, medium", try sentence(a, "off, low, medium", true));
    try testing.expectEqualStrings("Pull, list, and remove", try sentence(a, "pull, list, and remove", false));
    try testing.expectEqualStrings("Registry entry or path", try sentence(a, "registry entry or path", true));
    try testing.expectEqualStrings("ls: each file", try sentence(a, "ls: each file", false));
    try testing.expectEqualStrings("models.<name>.mmproj", try sentence(a, "models.<name>.mmproj", false));
    try testing.expectEqualStrings("", try sentence(a, "", false));
}
