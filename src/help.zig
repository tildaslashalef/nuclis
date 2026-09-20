//! `nuclis --help`, and one page per command.
//!
//! Help is the first thing anyone reads, and for a while it was the *only*
//! thing: every rule the CLI has was appended to one string until it became
//! ninety lines of prose nobody could scan (reported 2026-09-12). It is now
//! a table of commands, a table of shared options, and a page per command
//! reached with `nuclis <command> --help` — each one short enough to read at
//! a glance and pointing at the document that explains the rest.
//!
//! Styling goes through the same palette as every other report
//! (`tui/style.zig`), which means colour on a terminal and exactly nothing in
//! a pipe, so `nuclis --help | diff` stays stable.
//!
//! What belongs here: what a command does, its flags, its defaults, and the
//! few behaviours a user would otherwise be surprised by. What does not: the
//! reasoning, the tolerances, and the measurements — those live in
//! `docs/spec.md`, `docs/development.md`, and `docs/reference/`, and a help
//! page that duplicates them is a help page that will contradict them.
const std = @import("std");
const style = @import("tui/style.zig");

/// The commands with a page of their own; `null` is the overview.
pub const Topic = enum { inspect, validate, generate, bench, tokenize, agent, config, model };

pub fn write(out: *std.Io.Writer, sty: style.Style, topic: ?Topic, version: []const u8) !void {
    if (topic) |value| return switch (value) {
        .inspect => inspect(out, sty),
        .validate => validate(out, sty),
        .generate => generate(out, sty),
        .bench => bench(out, sty),
        .tokenize => tokenize(out, sty),
        .agent => agent(out, sty),
        .config => config(out, sty),
        .model => model(out, sty),
    };
    return overview(out, sty, version);
}

// ----- the pieces every page is built from -----

/// Width of the name column in a `row`: two spaces of margin, the name, two
/// spaces, then the description, so every page has one text column.
const name_column = 20;
/// The indent that lines a continuation up under a description.
const continuation = " " ** (2 + name_column + 2);

/// A section heading, with a blank line before it unless it opens the page.
fn heading(out: *std.Io.Writer, sty: style.Style, text: []const u8) !void {
    try out.print("\n{s}{s}{s}\n", .{ sty.on(.header), text, sty.off() });
}

/// One `name  description` row, the name in the keyword colour and padded so
/// the descriptions line up. Padding happens before the escapes, so colour
/// never changes the column (the rule every report here follows).
fn row(out: *std.Io.Writer, sty: style.Style, name: []const u8, text: []const u8) !void {
    var padded: [name_column]u8 = undefined;
    const width = @min(name.len, padded.len);
    @memcpy(padded[0..width], name[0..width]);
    @memset(padded[width..], ' ');
    try out.print("  {s}{s}{s}  {s}\n", .{ sty.on(.keyword), padded[0..@max(width, name_column)], sty.off(), text });
}

/// A usage line: the command in code style, the rest plain.
fn usage(out: *std.Io.Writer, sty: style.Style, line: []const u8) !void {
    try out.print("  {s}{s}{s}\n", .{ sty.on(.code), line, sty.off() });
}

/// A note under a page: dim, wrapped by hand at a width that fits 80 columns.
fn note(out: *std.Io.Writer, sty: style.Style, text: []const u8) !void {
    try out.print("  {s}{s}{s}\n", .{ sty.on(.dim), text, sty.off() });
}

fn plain(out: *std.Io.Writer, text: []const u8) !void {
    try out.print("  {s}\n", .{text});
}

/// Where the long form of each subject lives, printed at the end of a page.
fn seeAlso(out: *std.Io.Writer, sty: style.Style, text: []const u8) !void {
    try out.print("\n  {s}see{s} {s}\n", .{ sty.on(.label), sty.off(), text });
}

// ----- the overview -----

fn overview(out: *std.Io.Writer, sty: style.Style, version: []const u8) !void {
    try out.print("{s}nuclis{s} {s}{s}{s} — a local inference engine for GGUF models on Apple Silicon\n", .{
        sty.on(.header), sty.off(), sty.on(.number), version, sty.off(),
    });
    try heading(out, sty, "usage");
    try usage(out, sty, "nuclis <command> [options]");
    try usage(out, sty, "nuclis <command> --help");

    try heading(out, sty, "commands");
    try row(out, sty, "agent", "the interactive surface; -p <text> runs one turn without a TTY");
    try row(out, sty, "generate", "one completion from a prompt");
    try row(out, sty, "bench", "repeated prefill and decode measurements");
    try row(out, sty, "tokenize", "the prompt's token ids and their byte offsets");
    try row(out, sty, "inspect", "an artifact's identity, dimensions, and tensor encodings");
    try row(out, sty, "validate", "check a file against its architecture's adapter");
    try row(out, sty, "model", "pull, list, and judge artifacts from the Hugging Face Hub");
    try row(out, sty, "config", "write, show, or set a key of ~/.nuclis/nuclis.json");

    try heading(out, sty, "common options");
    try row(out, sty, "--model <name|path>", "a catalogue name, a registry entry, or a path");
    try row(out, sty, "--backend cpu|metal", "default metal (needs a Metal build)");
    try row(out, sty, "--ctx-size <n>", "context window in tokens, default 16384");
    try row(out, sty, "--max-tokens <n>", "output budget, default 2048");
    try row(out, sty, "--kv f16|f32", "attention cache precision on the GPU, default f16");
    try row(out, sty, "--think <effort>", "off, low, medium, high, xhigh — what the profile supports");
    try row(out, sty, "--speculative on|off", "verify drafts from the model's draft source, default off");
    try row(out, sty, "--draft-length <n>", "drafts per step, 1..7, default 4");
    try row(out, sty, "--prompt-profile <p>", "qwen38, gemma4, or muse_glimmer: force the prompt profile on a file whose");
    try out.print("{s}chat template is not the pinned one (a finetune); default: by digest\n", .{continuation});
    try row(out, sty, "--seed <n>", "sampler seed, default 0");
    try row(out, sty, "--json", "machine-readable output instead of a text report");
    try row(out, sty, "sampling", "--temperature --top-k --top-p --min-p");
    try out.print("{s}--presence-penalty --repetition-penalty\n", .{continuation});

    try heading(out, sty, "configuration");
    try plain(out, "~/.nuclis/nuclis.json; NUCLIS_HOME (absolute) moves the root.");
    try plain(out, "Precedence: defaults < the model's profile < file < registry entry < flags.");
    try note(out, sty, "`nuclis config show` prints every key's effective value and its source.");

    try seeAlso(out, sty, "docs/spec.md (scope), docs/development.md (environment, gates),");
    try out.print("      docs/architecture.md (how a token flows through the engine)\n", .{});
    try out.writeByte('\n');
}

// ----- one page per command -----

fn agent(out: *std.Io.Writer, sty: style.Style) !void {
    try heading(out, sty, "nuclis agent — the interactive surface");
    try usage(out, sty, "nuclis agent [<common options>] [--resume [<id>]]");
    try usage(out, sty, "nuclis agent -p <text> [--json] [--session <path>] [--resume [<id>]]");
    try usage(out, sty, "nuclis agent --print --prompt-file <path> [--json] [--session <path>]");
    try usage(out, sty, "nuclis agent ls [--json]");

    try heading(out, sty, "keys");
    try row(out, sty, "Enter", "send; while a turn runs, queue it for the next one");
    try row(out, sty, "Shift-Enter, Ctrl-J", "newline");
    try row(out, sty, "Up, Down", "move in the input; history at its first and last row");
    try row(out, sty, "Tab", "complete a /command or an @path, else fold thinking");
    try row(out, sty, "Ctrl-E", "expand a paste chip into editable text");
    try row(out, sty, "Ctrl-T, Ctrl-W", "cycle reasoning effort, context window");
    try row(out, sty, "Ctrl-N", "new session");
    try row(out, sty, "Ctrl-C, Ctrl-D", "cancel a turn, or quit");

    try heading(out, sty, "commands");
    try row(out, sty, "/new", "start a new session");
    try row(out, sty, "/ctx <n>", "context window in tokens");
    try row(out, sty, "/think <effort>", "off, low, medium, high, xhigh");
    try row(out, sty, "/save [path]", "export this session as markdown");
    try row(out, sty, "/help", "keys and commands, inside the surface");

    try heading(out, sty, "behaviour");
    try plain(out, "Completed turns are written into the terminal's own scrollback block by");
    try plain(out, "block and survive exit; a live region at the bottom holds what is still");
    try plain(out, "being written, the editor, and a status bar. A paste of more than a few");
    try plain(out, "lines becomes one chip; the input limit is 128 KiB.");
    try plain(out, "Print mode needs no terminal: text streams the answer, --json writes one");
    try plain(out, "event per line, and no session file is written unless --session names one.");

    try heading(out, sty, "files");
    try row(out, sty, "sessions/", "one append-only JSONL file per conversation");
    try row(out, sty, "exports/", "what /save writes");
    try row(out, sty, "history.jsonl", "submitted prompts, recalled with Up");
    try note(out, sty, "all under ~/.nuclis/agent, created when first written");

    try seeAlso(out, sty, "docs/agent-spec.md");
    try out.writeByte('\n');
}

fn generate(out: *std.Io.Writer, sty: style.Style) !void {
    try heading(out, sty, "nuclis generate — one completion");
    try usage(out, sty, "nuclis generate (--prompt <text> | --prompt-file <path> | --prompt-tokens <json>)");
    try usage(out, sty, "                [<common options>] [--raw] [--logits <path>] [--trace-dir <dir>]");

    try heading(out, sty, "options");
    try row(out, sty, "--prompt-tokens <json>", "a JSON array of ids, fed untokenized");
    try row(out, sty, "--raw", "skip chat rendering; send the prompt as written");
    try row(out, sty, "--logits <path>", "final prompt logits, F32 little-endian");
    try row(out, sty, "--trace-dir <dir>", "every layer as F32 LE (slow: commits per layer)");

    try heading(out, sty, "behaviour");
    try plain(out, "Renders one user turn with the profile pinned to the file's own chat");
    try plain(out, "template, and samples with that profile's defaults for the reasoning mode;");
    try plain(out, "each sampling flag overrides one option, and --temperature 0 is greedy.");
    try plain(out, "Text streams as valid UTF-8 even when a token ends inside a character.");
    try plain(out, "Ctrl-C stops at the next layer boundary and keeps what was written; a");
    try plain(out, "second Ctrl-C ends the process.");

    try seeAlso(out, sty, "docs/spec.md § Command-line interface");
    try out.writeByte('\n');
}

fn bench(out: *std.Io.Writer, sty: style.Style) !void {
    try heading(out, sty, "nuclis bench — prefill and decode measurements");
    try usage(out, sty, "nuclis bench (--prompt <text> | --prompt-file <path> | --prompt-tokens <json>)");
    try usage(out, sty, "             [<common options>] [--repeat <n>] [--warmup <n>] [--profile]");

    try heading(out, sty, "options");
    try row(out, sty, "--repeat <n>", "measured runs, default 3");
    try row(out, sty, "--warmup <n>", "unmeasured runs first, default 1");
    try row(out, sty, "--profile", "per-kernel GPU time (Metal); perturbs the rates");

    try heading(out, sty, "behaviour");
    try plain(out, "Greedy by default — never the file's sampling, never a profile — with 32");
    try plain(out, "output tokens, so a run measures the engine and not a sampler. Every");
    try plain(out, "sample and the measured means are reported, and a rate that could not be");
    try plain(out, "measured is omitted rather than estimated. A profiled run records one");
    try plain(out, "encoder per dispatch, so its rates are not comparable with unprofiled ones.");

    try seeAlso(out, sty, "docs/reference/bench.md");
    try out.writeByte('\n');
}

fn tokenize(out: *std.Io.Writer, sty: style.Style) !void {
    try heading(out, sty, "nuclis tokenize — the prompt as the model receives it");
    try usage(out, sty, "nuclis tokenize (--prompt <text> | --prompt-file <path>) [--raw] [--think <effort>]");
    try usage(out, sty, "                [--prompt-profile <p>] [--model <name|path>] [--json]");
    try heading(out, sty, "behaviour");
    try plain(out, "Renders the prompt exactly as generate and bench would — raw, or one user");
    try plain(out, "turn at the given effort — and prints its token ids with the byte offset of");
    try plain(out, "each token in the rendered text. Reads the artifact's header only; no model");
    try plain(out, "runs, so it is fast enough to use while writing a prompt.");
    try out.writeByte('\n');
}

fn inspect(out: *std.Io.Writer, sty: style.Style) !void {
    try heading(out, sty, "nuclis inspect — what an artifact is");
    try usage(out, sty, "nuclis inspect [--model <name|path>] [--json]");
    try heading(out, sty, "behaviour");
    try plain(out, "Reads the GGUF directory: identity, architecture, dimensions, the tensor");
    try plain(out, "encoding histogram, and the memory a requested context would need. Ranges");
    try plain(out, "and offsets are validated; weight values are not read. Little-endian GGUF");
    try plain(out, "v3 only, and an unknown tensor layout is an error rather than a guess.");
    try seeAlso(out, sty, "docs/reference/gguf-inspection.md");
    try out.writeByte('\n');
}

fn validate(out: *std.Io.Writer, sty: style.Style) !void {
    try heading(out, sty, "nuclis validate — will this file run");
    try usage(out, sty, "nuclis validate [--model <name|path>] [--json]");
    try heading(out, sty, "behaviour");
    try plain(out, "Checks the file against its architecture's adapter (qwen35, gemma4) and");
    try plain(out, "binds the text-layer weights, so a missing tensor or an unsupported");
    try plain(out, "encoding is named here instead of at the first token. It does not verify");
    try plain(out, "tokenizer behaviour, weight values, or numerical execution.");
    try out.writeByte('\n');
}

fn model(out: *std.Io.Writer, sty: style.Style) !void {
    try heading(out, sty, "nuclis model — artifacts from the Hugging Face Hub");
    try usage(out, sty, "nuclis model ls [--json]");
    try usage(out, sty, "nuclis model pull <name> [--with mmproj,mtp | --all] [--force] [--json]");
    try usage(out, sty, "nuclis model pull <owner/repo> [--file <name>] [--revision <rev>] [--role <role>]");
    try usage(out, sty, "                  [--register <name> [--profile <p>]]");
    try usage(out, sty, "nuclis model inspect (<name> | <owner/repo> --file <name>) [--revision <rev>]");

    try heading(out, sty, "options");
    try row(out, sty, "--with mmproj,mtp", "also fetch those companions");
    try row(out, sty, "--all", "every companion the entry names");
    try row(out, sty, "--force", "replace a file whose sidecar records other content");
    try row(out, sty, "--role <role>", "main, mmproj, mtp, imatrix (with owner/repo)");
    try row(out, sty, "--register <name>", "once verified, write the pull as a registry entry of");
    try out.print("{s}nuclis.json (repo, file, commit); a companion fills an entry\n", .{continuation});
    try row(out, sty, "--profile <p>", "qwen38 or gemma4, forced on the registered entry's file");

    try heading(out, sty, "behaviour");
    try plain(out, "A catalogue name pins repository, file, commit, and SHA-256; `model ls`");
    try plain(out, "lists them with their local status and names the registry entry of every");
    try plain(out, "file one locates. Downloads go over Xet, verify the");
    try plain(out, "digest, publish atomically, and leave a <file>.nuclis.json sidecar beside");
    try plain(out, "the file. `model inspect` reads only the remote head (at most 64 MiB, no");
    try plain(out, "weights) and ends with a verdict: supported, runnable, or not runnable.");
    try note(out, sty, "HF_TOKEN is optional and only needed for gated repositories.");

    try seeAlso(out, sty, "docs/development.md § Model download");
    try out.writeByte('\n');
}

fn config(out: *std.Io.Writer, sty: style.Style) !void {
    try heading(out, sty, "nuclis config — the file every command reads");
    try usage(out, sty, "nuclis config init");
    try usage(out, sty, "nuclis config show [--json]");
    try usage(out, sty, "nuclis config set <key> <value>");

    try heading(out, sty, "behaviour");
    try plain(out, "`init` writes ~/.nuclis/nuclis.json with the defaults and every catalogue");
    try plain(out, "model registered, then says what to pull next; it never overwrites an");
    try plain(out, "existing file. `show` prints the effective value of every key with the");
    try plain(out, "layer it came from — default, profile, file, model entry, or flag.");
    try plain(out, "`set` changes one key by its dotted name (engine.model hauhau,");
    try plain(out, "generation.sampling.temperature 0.7, models.<name>.profile gemma4; null");
    try plain(out, "clears an override) and validates the file before writing it; a missing");
    try plain(out, "file is created as `init` would. Entries are created by `model pull");
    try plain(out, "--register <name>`, never by `set`. An unknown key or an out-of-range");
    try plain(out, "value is an error naming the key, and the file is left as it was.");

    try heading(out, sty, "sections");
    try row(out, sty, "engine", "model, backend, ctx_size, kv_precision");
    try row(out, sty, "generation", "max_tokens, think, speculative, draft_length, sampling overrides");
    try row(out, sty, "agent", "think, fold_thinking, theme");
    try row(out, sty, "models", "named entries: path, or repo + file + revision");

    try seeAlso(out, sty, "docs/development.md § Configuration");
    try out.writeByte('\n');
}

// ----- tests -----

const testing = std.testing;

fn rendered(alloc: std.mem.Allocator, topic: ?Topic, sty: style.Style) ![]u8 {
    var buffer: std.Io.Writer.Allocating = .init(alloc);
    errdefer buffer.deinit();
    try write(&buffer.writer, sty, topic, "0.0.0-test");
    return buffer.toOwnedSlice();
}

test "the overview names every command and stays short enough to read" {
    const text = try rendered(testing.allocator, null, .none);
    defer testing.allocator.free(text);
    inline for (@typeInfo(Topic).@"enum".fields) |field| {
        try testing.expect(std.mem.indexOf(u8, text, field.name) != null);
    }
    try testing.expect(std.mem.indexOf(u8, text, "0.0.0-test") != null);
    try testing.expect(std.mem.indexOf(u8, text, "nuclis <command> --help") != null);
    // The wall of prose this replaced was ninety lines; the overview is a
    // page, and a test is the only thing that keeps it one.
    try testing.expect(std.mem.count(u8, text, "\n") <= 45);
    for (text) |byte| try testing.expect(byte >= 0x20 or byte == '\n');
}

test "every command has a page, and every page fits a screen" {
    inline for (@typeInfo(Topic).@"enum".fields) |field| {
        const topic: Topic = @enumFromInt(field.value);
        const text = try rendered(testing.allocator, topic, .none);
        defer testing.allocator.free(text);
        // The page names its own command and says how to invoke it.
        try testing.expect(std.mem.indexOf(u8, text, "nuclis " ++ field.name) != null);
        const lines = std.mem.count(u8, text, "\n");
        try testing.expect(lines > 4 and lines <= 45);
    }
}

test "styling is on the palette and absent from a pipe" {
    const lit: style.Style = .{ .theme = .{ .kind = .truecolor }, .enabled = true };
    const colored = try rendered(testing.allocator, .agent, lit);
    defer testing.allocator.free(colored);
    const piped = try rendered(testing.allocator, .agent, .none);
    defer testing.allocator.free(piped);
    try testing.expect(std.mem.indexOfScalar(u8, colored, 0x1b) != null);
    // A pipe gets exactly the same text with no escapes at all.
    try testing.expect(std.mem.indexOfScalar(u8, piped, 0x1b) == null);
    try testing.expect(colored.len > piped.len);
}

test "a row's description starts at the same column whatever the palette" {
    // Padding happens before the escapes, so colour cannot move a column.
    var plain_buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer plain_buffer.deinit();
    var lit_buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer lit_buffer.deinit();
    const lit: style.Style = .{ .theme = .{ .kind = .truecolor }, .enabled = true };
    try row(&plain_buffer.writer, .none, "--seed <n>", "sampler seed");
    try row(&lit_buffer.writer, lit, "--seed <n>", "sampler seed");
    const at = std.mem.indexOf(u8, plain_buffer.written(), "sampler").?;
    const escapes = std.mem.indexOf(u8, lit_buffer.written(), "sampler").? -
        (lit.on(.keyword).len + lit.off().len);
    try testing.expectEqual(at, escapes);
}
