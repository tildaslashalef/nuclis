//! `nuclis --help`, and one page per command.
//!
//! Every page is self-contained: what the command does, its usage lines,
//! every option the parser accepts for it with its default, examples, and
//! the few behaviours a user would otherwise be surprised by. Nothing here
//! points at the repository's documents; a page that needs them to be read
//! is not finished.
//!
//! Styling goes through the same palette as every other report
//! (`tui/style.zig`): colour on a terminal, nothing in a pipe, so
//! `nuclis --help | diff` stays stable. Widths are fixed so every line fits
//! 80 columns (a test holds that).
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

/// The name column of a `row`: two spaces of margin, the name padded to this
/// width, two spaces, then the description; one text column per page.
const name_column = 22;
const continuation = " " ** (2 + name_column + 2);

/// The page's title line: the command in the header colour, a dash, what it is.
fn title(out: *std.Io.Writer, sty: style.Style, command: []const u8, text: []const u8) !void {
    try out.print("{s}{s}{s} — {s}\n", .{ sty.on(.header), command, sty.off(), text });
}

/// A section heading (`Usage:`, `Options:`), a blank line before it.
fn heading(out: *std.Io.Writer, sty: style.Style, text: []const u8) !void {
    try out.print("\n{s}{s}{s}\n", .{ sty.on(.bold), text, sty.off() });
}

/// One `name  description` row, the name in the keyword colour and padded
/// before the escapes so colour never moves the column.
fn row(out: *std.Io.Writer, sty: style.Style, name: []const u8, text: []const u8) !void {
    var padded: [name_column]u8 = undefined;
    const width = @min(name.len, padded.len);
    @memcpy(padded[0..width], name[0..width]);
    @memset(padded[width..], ' ');
    try out.print("  {s}{s}{s}  {s}\n", .{ sty.on(.keyword), padded[0..@max(width, name_column)], sty.off(), text });
}

/// A description's next line, aligned under the first.
fn more(out: *std.Io.Writer, text: []const u8) !void {
    try out.print("{s}{s}\n", .{ continuation, text });
}

/// A usage or example line, in code style.
fn code(out: *std.Io.Writer, sty: style.Style, line: []const u8) !void {
    try out.print("  {s}{s}{s}\n", .{ sty.on(.code), line, sty.off() });
}

/// An example with a dim comment after it.
fn example(out: *std.Io.Writer, sty: style.Style, line: []const u8, comment: []const u8) !void {
    try out.print("  {s}{s}{s}\n{s}{s}{s}{s}\n", .{ sty.on(.code), line, sty.off(), continuation, sty.on(.dim), comment, sty.off() });
}

fn plain(out: *std.Io.Writer, text: []const u8) !void {
    try out.print("  {s}\n", .{text});
}

/// The options `generate`, `bench`, and `agent` share (the engine's).
fn engineOptions(out: *std.Io.Writer, sty: style.Style) !void {
    try row(out, sty, "--model <name|path>", "registry entry, catalogue name, or path; default");
    try more(out, "engine.model of the config file (qwen3.8-27b)");
    try row(out, sty, "--backend cpu|metal", "default metal");
    try row(out, sty, "--ctx-size <n>", "context window in tokens, 1..32768; default 16384");
    try row(out, sty, "--kv f16|f32", "attention cache precision on the GPU; default f16");
    try row(out, sty, "--max-tokens <n>", "output budget, 1..16384; default 4096 (bench: 32)");
    try row(out, sty, "--speculative on|off", "verify drafts from the model's draft source; the");
    try more(out, "default is the model entry's verdict, else off");
    try row(out, sty, "--draft-length <n>", "drafts per verify batch, 1..7; default 4");
    try row(out, sty, "--prompt-profile <p>", "force qwen38, gemma4, or muse_glimmer on a file");
    try more(out, "whose chat template is not a pinned one (a finetune)");
    try row(out, sty, "--seed <n>", "sampler seed; default 0");
}

fn samplingOptions(out: *std.Io.Writer, sty: style.Style) !void {
    try row(out, sty, "--temperature <t>", "0 is greedy; each flag overrides one option of the");
    try row(out, sty, "--top-k <n>", "profile's defaults for the reasoning mode");
    try row(out, sty, "--top-p <p>", "");
    try row(out, sty, "--min-p <p>", "");
    try row(out, sty, "--presence-penalty <x>", "");
    try row(out, sty, "--repetition-penalty <x>", "");
}

// ----- the overview -----

fn overview(out: *std.Io.Writer, sty: style.Style, version: []const u8) !void {
    try out.print("{s}nuclis{s} {s}{s}{s} — a local inference engine for GGUF models on Apple Silicon\n", .{
        sty.on(.header), sty.off(), sty.on(.number), version, sty.off(),
    });
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis <command> [options]");
    try code(out, sty, "nuclis <command> --help");
    try code(out, sty, "nuclis --version");

    try heading(out, sty, "Commands:");
    try row(out, sty, "agent", "the interactive chat; -p <text> runs one turn");
    try row(out, sty, "generate", "one completion from a prompt");
    try row(out, sty, "bench", "repeated prefill and decode measurements");
    try row(out, sty, "tokenize", "the prompt's token ids and byte offsets, no model run");
    try row(out, sty, "inspect", "an artifact's identity, dimensions, tensor encodings");
    try row(out, sty, "validate", "whether a file binds to its architecture's adapter");
    try row(out, sty, "model", "pull, list, and judge Hugging Face Hub artifacts");
    try row(out, sty, "config", "write, show, or set a key of ~/.nuclis/nuclis.json");

    try heading(out, sty, "Global options:");
    try row(out, sty, "--help, -h", "this page, or a command's page after its name");
    try row(out, sty, "--version", "the version of this binary");
    try row(out, sty, "--json", "machine-readable output, where a command reports");

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis config init", "the config file with every catalogue model registered");
    try example(out, sty, "nuclis model pull qwen3.8-27b --all", "fetch the default model and its companions");
    try example(out, sty, "nuclis agent", "chat with the configured model");
    try example(out, sty, "nuclis generate --model gemma-4-12b-qat --prompt \"Explain RoPE\"", "one answer, another model");

    try heading(out, sty, "Files:");
    try row(out, sty, "~/.nuclis/nuclis.json", "engine, generation, agent settings; the registry");
    try row(out, sty, "~/.nuclis/models/", "artifacts as <owner>/<repo>/<file> plus sidecars");
    try row(out, sty, "~/.nuclis/agent/", "sessions, exports, prompt history");
    try plain(out, "NUCLIS_HOME (an absolute path) moves the root; NO_COLOR drops colour.");
    try out.writeByte('\n');
}

// ----- one page per command -----

fn agent(out: *std.Io.Writer, sty: style.Style) !void {
    try title(out, sty, "nuclis agent", "the interactive chat, with tools for small coding tasks");
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis agent [options] [--resume [<id>]]");
    try code(out, sty, "nuclis agent -p <text> [--json] [--session <path>]");
    try code(out, sty, "nuclis agent --print --prompt-file <path> [--json] [--session <path>]");
    try code(out, sty, "nuclis agent ls [--json]");

    try heading(out, sty, "Options:");
    try row(out, sty, "-p, --prompt <text>", "run one turn without a terminal and exit");
    try row(out, sty, "--print", "the same, with the prompt from --prompt-file");
    try row(out, sty, "--prompt-file <path>", "the prompt's text, for print mode");
    try row(out, sty, "--json", "in print mode: one event per line instead of text");
    try row(out, sty, "--session <path>", "record a printed turn to this session file");
    try row(out, sty, "--resume [<id>]", "replay a saved session first; the latest without <id>");
    try row(out, sty, "--think <effort>", "off, low, medium, high, xhigh; default low");
    try engineOptions(out, sty);
    try samplingOptions(out, sty);

    try heading(out, sty, "Keys:");
    try row(out, sty, "Enter", "send; while a turn runs, queue it for the next one");
    try row(out, sty, "Shift-Enter, Ctrl-J", "newline");
    try row(out, sty, "Up, Down", "move in the input; history at its first and last row");
    try row(out, sty, "Tab", "complete a /command or an @path, else fold thinking");
    try row(out, sty, "Ctrl-E", "expand a paste chip into editable text");
    try row(out, sty, "Ctrl-T, Ctrl-W", "cycle the reasoning effort, the context window");
    try row(out, sty, "Ctrl-N", "new session");
    try row(out, sty, "Ctrl-C, Ctrl-D", "cancel the turn, or quit");

    try heading(out, sty, "Commands:");
    try row(out, sty, "/new, /resume [<id>]", "start a new session, or replay a saved one");
    try row(out, sty, "/ctx <n>, /think <e>", "the context window, the reasoning effort");
    try row(out, sty, "/save [path]", "export this session as markdown");
    try row(out, sty, "/help", "keys and commands, inside the surface");

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis agent --model gemma-4-12b-qat --think medium", "chat with another model at more effort");
    try example(out, sty, "nuclis agent -p \"Summarize README.md\" --json", "one scripted turn, events as JSON lines");
    try example(out, sty, "nuclis agent ls", "the sessions saved for this directory");

    try heading(out, sty, "Notes:");
    try plain(out, "Completed turns are written into the terminal's scrollback and survive");
    try plain(out, "exit; the live region at the bottom holds the turn being written, the");
    try plain(out, "editor, and a status bar. A paste of more than a few lines becomes one");
    try plain(out, "chip; the input limit is 128 KiB. Sessions live under ~/.nuclis/agent");
    try plain(out, "(sessions/, exports/, history.jsonl), created when first written; print");
    try plain(out, "mode writes nothing unless --session names a file.");
    try out.writeByte('\n');
}

fn generate(out: *std.Io.Writer, sty: style.Style) !void {
    try title(out, sty, "nuclis generate", "one completion from a prompt");
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis generate --prompt <text> [options]");
    try code(out, sty, "nuclis generate --prompt-file <path> [options]");
    try code(out, sty, "nuclis generate --prompt-tokens <json> [options]");

    try heading(out, sty, "Options:");
    try row(out, sty, "--prompt <text>", "the user turn (one of the three sources, required)");
    try row(out, sty, "--prompt-file <path>", "the same, read from a file");
    try row(out, sty, "--prompt-tokens <json>", "a JSON array of token ids fed untokenized");
    try row(out, sty, "--raw", "skip the chat template; send the text as written");
    try row(out, sty, "--think <effort>", "off, low, medium, high, xhigh; default off");
    try row(out, sty, "--logits <path>", "write the final prompt logits, F32 little-endian");
    try row(out, sty, "--trace-dir <dir>", "write every layer's output as F32 (slow)");
    try row(out, sty, "--json", "the run's report (tokens, timings) instead of text");
    try engineOptions(out, sty);
    try samplingOptions(out, sty);

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis generate --prompt \"Write a haiku about Zig\"", "the configured model, its profile's sampling");
    try example(out, sty, "nuclis generate --prompt-file q.txt --think high --max-tokens 4096", "a long reasoned answer");
    try example(out, sty, "nuclis generate --prompt \"Hello,\" --raw --temperature 0 --max-tokens 8", "greedy continuation of literal text");

    try heading(out, sty, "Notes:");
    try plain(out, "The prompt is rendered as one user turn with the profile pinned to the");
    try plain(out, "file's own chat template, and sampled with that profile's defaults for the");
    try plain(out, "reasoning mode; each sampling flag overrides one option. Text streams as");
    try plain(out, "valid UTF-8 even when a token ends inside a character. Ctrl-C stops at the");
    try plain(out, "next layer and keeps what was written; a second Ctrl-C ends the process.");
    try out.writeByte('\n');
}

fn bench(out: *std.Io.Writer, sty: style.Style) !void {
    try title(out, sty, "nuclis bench", "repeated prefill and decode measurements on one loaded model");
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis bench --prompt <text> [options]");
    try code(out, sty, "nuclis bench --prompt-file <path> [options]");
    try code(out, sty, "nuclis bench --prompt-tokens <json> [options]");

    try heading(out, sty, "Options:");
    try row(out, sty, "--prompt <text>", "the prompt (one of the three sources, required)");
    try row(out, sty, "--prompt-file <path>", "the same, read from a file");
    try row(out, sty, "--prompt-tokens <json>", "a JSON array of ids, so a reference's exact input");
    try more(out, "is measured rather than re-tokenized");
    try row(out, sty, "--raw", "skip the chat template");
    try row(out, sty, "--repeat <n>", "measured runs, 1..100; default 3");
    try row(out, sty, "--warmup <n>", "unmeasured runs first, 0..100; default 1");
    try row(out, sty, "--profile", "per-kernel GPU time (Metal); perturbs the rates");
    try row(out, sty, "--unfused-norms", "run the norm pairs the fused kernels replace");
    try row(out, sty, "--json", "the report: every sample and the measured means");
    try engineOptions(out, sty);
    try samplingOptions(out, sty);

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis bench --prompt-file prompt.txt --max-tokens 128 --json", "three measured runs after one warmup");
    try example(out, sty, "nuclis bench --prompt-tokens p.json --ctx-size 32768 --repeat 5", "a pinned token array at full context");
    try example(out, sty, "nuclis bench --prompt \"Hi\" --speculative on --draft-length 4", "each run measured with the switch off and on");

    try heading(out, sty, "Notes:");
    try plain(out, "Greedy with 32 output tokens by default, never the file's sampling and");
    try plain(out, "never a profile, so a run measures the engine and not a sampler; the");
    try plain(out, "model, backend, and context come from the config file. Every sample and");
    try plain(out, "the measured means are reported; a rate that could not be measured is");
    try plain(out, "omitted, never estimated. With --speculative on, each run is done both");
    try plain(out, "ways on one loaded model and the report carries the decode speedup. A");
    try plain(out, "profiled run records one encoder per dispatch; its rates are not");
    try plain(out, "comparable with unprofiled ones.");
    try out.writeByte('\n');
}

fn tokenize(out: *std.Io.Writer, sty: style.Style) !void {
    try title(out, sty, "nuclis tokenize", "the prompt as the model receives it, without running it");
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis tokenize --prompt <text> [options]");
    try code(out, sty, "nuclis tokenize --prompt-file <path> [options]");

    try heading(out, sty, "Options:");
    try row(out, sty, "--prompt <text>", "the user turn (one of the two sources, required)");
    try row(out, sty, "--prompt-file <path>", "the same, read from a file");
    try row(out, sty, "--raw", "tokenize the text as written, without the template");
    try row(out, sty, "--think <effort>", "render the turn at this effort; default off");
    try row(out, sty, "--prompt-profile <p>", "force qwen38, gemma4, or muse_glimmer");
    try row(out, sty, "--model <name|path>", "whose vocabulary and template; default engine.model");
    try row(out, sty, "--json", "ids, offsets, and the rendered text as JSON");

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis tokenize --prompt \"Hello, world\"", "the rendered turn with one id per line");
    try example(out, sty, "nuclis tokenize --prompt-file p.txt --raw --json", "exact ids and byte offsets for a script");

    try heading(out, sty, "Notes:");
    try plain(out, "Renders exactly what generate and bench would feed (raw, or one user turn");
    try plain(out, "at the given effort) and prints each token's id with its byte offset in");
    try plain(out, "the rendered text. Only the artifact's header is read, so it is fast");
    try plain(out, "enough to use while writing a prompt.");
    try out.writeByte('\n');
}

fn inspect(out: *std.Io.Writer, sty: style.Style) !void {
    try title(out, sty, "nuclis inspect", "what an artifact is");
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis inspect [--model <name|path>] [--json]");

    try heading(out, sty, "Options:");
    try row(out, sty, "--model <name|path>", "the file to read; default engine.model");
    try row(out, sty, "--json", "the same facts as JSON");

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis inspect", "the configured model");
    try example(out, sty, "nuclis inspect --model ~/Downloads/some-model.gguf --json", "any GGUF file, for a script");

    try heading(out, sty, "Notes:");
    try plain(out, "Reads the GGUF directory only: identity, architecture, dimensions, the");
    try plain(out, "tensor encoding histogram, and the memory a context would need. Ranges");
    try plain(out, "and offsets are validated; weight values are not read. Little-endian GGUF");
    try plain(out, "v3 only; an unknown tensor layout is an error rather than a guess.");
    try out.writeByte('\n');
}

fn validate(out: *std.Io.Writer, sty: style.Style) !void {
    try title(out, sty, "nuclis validate", "will this file run");
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis validate [--model <name|path>] [--json]");

    try heading(out, sty, "Options:");
    try row(out, sty, "--model <name|path>", "the file to check; default engine.model");
    try row(out, sty, "--json", "the verdict as JSON");

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis validate --model gemma-4-12b-qat", "bind a catalogue model's weights");

    try heading(out, sty, "Notes:");
    try plain(out, "Checks the file against its architecture's adapter (qwen35, gemma4,");
    try plain(out, "muse-glimmer) and binds the text-layer weights, so a missing tensor or an");
    try plain(out, "unsupported encoding is named here instead of at the first token. It does");
    try plain(out, "not verify tokenizer behaviour, weight values, or numerical execution.");
    try out.writeByte('\n');
}

fn model(out: *std.Io.Writer, sty: style.Style) !void {
    try title(out, sty, "nuclis model", "artifacts from the Hugging Face Hub");
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis model ls [--json]");
    try code(out, sty, "nuclis model pull <name> [--with mmproj,mtp | --all] [--force] [--json]");
    try code(out, sty, "nuclis model pull <owner/repo> [--file <name>] [--revision <rev>] [--role <r>]");
    try code(out, sty, "                  [--register <name> [--profile <p>]] [--force] [--json]");
    try code(out, sty, "nuclis model inspect <name> [--json]");
    try code(out, sty, "nuclis model inspect <owner/repo> --file <name> [--revision <rev>] [--json]");

    try heading(out, sty, "Commands:");
    try row(out, sty, "ls", "the catalogue with each entry's local status, then");
    try more(out, "every other GGUF file under ~/.nuclis/models");
    try row(out, sty, "pull", "fetch a catalogue entry, a registry entry, or any");
    try more(out, "file of a repository, verified by SHA-256 and");
    try more(out, "published atomically");
    try row(out, sty, "inspect", "read a remote file's header (at most 64 MiB, no");
    try more(out, "weights) and say whether it would run: supported,");
    try more(out, "runnable, or not");

    try heading(out, sty, "Options:");
    try row(out, sty, "--with mmproj,mtp", "also fetch those companions of the entry");
    try row(out, sty, "--all", "every companion the entry names");
    try row(out, sty, "--file <name>", "which file of the repository (required when several)");
    try row(out, sty, "--revision <rev>", "a branch, tag, or commit; default main");
    try row(out, sty, "--role <r>", "main, mmproj, mtp, or imatrix; default by content");
    try row(out, sty, "--register <name>", "once verified, write the pull as a registry entry");
    try more(out, "(repo, file, commit); a companion fills the same entry");
    try row(out, sty, "--profile <p>", "with --register: force qwen38, gemma4, or muse_glimmer");
    try row(out, sty, "--force", "replace a file whose sidecar records other content");
    try row(out, sty, "--json", "the listing, transfer report, or verdict as JSON");

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis model pull qwen3.8-27b --all", "the default model with its projector and draft head");
    try code(out, sty, "nuclis model pull unsloth/gemma-4-12b-it-GGUF \\");
    try example(out, sty, "    --file gemma-4-12b-it-UD-Q4_K_XL.gguf --register gemma-fast", "any file of a repository, under a name");
    try example(out, sty, "nuclis model inspect unsloth/Muse-Glimmer-30B-GGUF --file dflash-kquant.gguf", "will it run, before the download");

    try heading(out, sty, "Notes:");
    try plain(out, "Downloads go over Xet, verify the digest, publish atomically, and leave a");
    try plain(out, "<file>.nuclis.json sidecar beside the file; a second pull of the same file");
    try plain(out, "downloads nothing. HF_TOKEN is read for gated repositories only.");
    try out.writeByte('\n');
}

fn config(out: *std.Io.Writer, sty: style.Style) !void {
    try title(out, sty, "nuclis config", "the file every command reads (~/.nuclis/nuclis.json)");
    try heading(out, sty, "Usage:");
    try code(out, sty, "nuclis config init [--discover [--dry-run] [--json]]");
    try code(out, sty, "nuclis config show [--json]");
    try code(out, sty, "nuclis config set <key> <value>");

    try heading(out, sty, "Commands:");
    try row(out, sty, "init", "write the file with the defaults and every catalogue");
    try more(out, "model registered, then say what to pull next; an");
    try more(out, "existing file is kept");
    try row(out, sty, "show", "every key's effective value and where it came from:");
    try more(out, "default, profile, file, model entry, or flag");
    try row(out, sty, "set", "change one key by its dotted name; the file is");
    try more(out, "validated before it is written, and left as it was");
    try more(out, "on a refused value");

    try heading(out, sty, "Options:");
    try row(out, sty, "--discover", "register the runnable GGUF files under");
    try more(out, "~/.nuclis/models that neither the catalogue nor the");
    try more(out, "file names (a finetune pulled by repository): the");
    try more(out, "adapter is read from the header, the profile is");
    try more(out, "forced when the chat template is not a pinned one,");
    try more(out, "and companions beside the file fill mmproj and mtp");
    try row(out, sty, "--dry-run", "print what --discover would register; write nothing");
    try row(out, sty, "--json", "show, or the discovery report, as JSON");

    try heading(out, sty, "Keys:");
    try row(out, sty, "engine.*", "model, backend, ctx_size, kv_precision");
    try row(out, sty, "generation.*", "max_tokens, think, speculative, draft_length,");
    try more(out, "sampling.{temperature,top_k,top_p,min_p,");
    try more(out, "presence_penalty,repetition_penalty}; null takes");
    try more(out, "the profile's value");
    try row(out, sty, "agent.*", "think, fold_thinking, theme");
    try row(out, sty, "models.<name>.*", "path, or repo + file + revision; mmproj, mtp, profile,");
    try more(out, "ctx_size, generation.*, agent.* for that model only.");
    try more(out, "Entries are created by init --discover and model pull");
    try more(out, "--register, never by set.");

    try heading(out, sty, "Examples:");
    try example(out, sty, "nuclis config init --discover --dry-run", "what a discovery would add");
    try example(out, sty, "nuclis config set engine.model gemma-4-12b-qat", "the default model");
    try example(out, sty, "nuclis config set generation.sampling.temperature 0.7", "a sampling override");
    try example(out, sty, "nuclis config set models.mine.profile gemma4", "force a profile on one entry");
    try example(out, sty, "nuclis config set generation.think null", "clear an override");

    try heading(out, sty, "Notes:");
    try plain(out, "Precedence: defaults < the model's profile < the file < the model's");
    try plain(out, "entry < flags. NUCLIS_HOME (an absolute path) moves ~/.nuclis. An unknown");
    try plain(out, "key or an out-of-range value is an error naming the key.");
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

test "the overview names every command and stays one screen" {
    const text = try rendered(testing.allocator, null, .none);
    defer testing.allocator.free(text);
    inline for (@typeInfo(Topic).@"enum".fields) |field| {
        try testing.expect(std.mem.indexOf(u8, text, field.name) != null);
    }
    try testing.expect(std.mem.indexOf(u8, text, "0.0.0-test") != null);
    try testing.expect(std.mem.indexOf(u8, text, "nuclis <command> --help") != null);
    try testing.expect(std.mem.count(u8, text, "\n") <= 45);
    for (text) |byte| try testing.expect(byte >= 0x20 or byte == '\n');
}

test "every page is self-contained, has the standard sections, and fits 80 columns" {
    inline for (@typeInfo(Topic).@"enum".fields) |field| {
        const topic: Topic = @enumFromInt(field.value);
        const text = try rendered(testing.allocator, topic, .none);
        defer testing.allocator.free(text);
        try testing.expect(std.mem.indexOf(u8, text, "nuclis " ++ field.name) != null);
        for ([_][]const u8{ "Usage:", "Options:", "Examples:", "Notes:" }) |section| {
            try testing.expect(std.mem.indexOf(u8, text, section) != null);
        }
        // No pointer at the repository's documents: the page is the help.
        try testing.expect(std.mem.indexOf(u8, text, "docs/") == null);
        try testing.expect(std.mem.indexOf(u8, text, "see ") == null);
        var lines = std.mem.splitScalar(u8, text, '\n');
        var count: usize = 0;
        while (lines.next()) |line| : (count += 1) {
            try testing.expect(std.unicode.utf8CountCodepoints(line) catch line.len <= 80);
        }
        try testing.expect(count > 8 and count <= 80);
    }
}

test "every flag the parser accepts for a command is on its page" {
    const Flags = struct { topic: Topic, flags: []const []const u8 };
    const table = [_]Flags{
        .{ .topic = .generate, .flags = &.{ "--prompt", "--prompt-file", "--prompt-tokens", "--raw", "--logits", "--trace-dir", "--think", "--model", "--backend", "--ctx-size", "--max-tokens", "--kv", "--speculative", "--draft-length", "--prompt-profile", "--seed", "--json", "--temperature", "--top-k", "--top-p", "--min-p", "--presence-penalty", "--repetition-penalty" } },
        .{ .topic = .bench, .flags = &.{ "--prompt", "--prompt-file", "--prompt-tokens", "--raw", "--repeat", "--warmup", "--profile", "--unfused-norms", "--model", "--backend", "--ctx-size", "--max-tokens", "--kv", "--speculative", "--draft-length", "--prompt-profile", "--seed", "--json", "--temperature" } },
        .{ .topic = .agent, .flags = &.{ "-p", "--prompt", "--print", "--prompt-file", "--json", "--session", "--resume", "--think", "--model", "--backend", "--ctx-size", "--max-tokens", "--kv", "--speculative", "--draft-length", "--prompt-profile", "--seed", "--temperature" } },
        .{ .topic = .tokenize, .flags = &.{ "--prompt", "--prompt-file", "--raw", "--think", "--prompt-profile", "--model", "--json" } },
        .{ .topic = .inspect, .flags = &.{ "--model", "--json" } },
        .{ .topic = .validate, .flags = &.{ "--model", "--json" } },
        .{ .topic = .model, .flags = &.{ "--with", "--all", "--file", "--revision", "--role", "--register", "--profile", "--force", "--json" } },
        .{ .topic = .config, .flags = &.{ "--discover", "--dry-run", "--json" } },
    };
    for (table) |entry| {
        const text = try rendered(testing.allocator, entry.topic, .none);
        defer testing.allocator.free(text);
        for (entry.flags) |flag| {
            if (std.mem.indexOf(u8, text, flag) == null) {
                std.debug.print("{s} page lacks {s}\n", .{ @tagName(entry.topic), flag });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "styling is on the palette and absent from a pipe" {
    const lit: style.Style = .{ .theme = .{ .kind = .truecolor }, .enabled = true };
    const colored = try rendered(testing.allocator, .agent, lit);
    defer testing.allocator.free(colored);
    const piped = try rendered(testing.allocator, .agent, .none);
    defer testing.allocator.free(piped);
    try testing.expect(std.mem.indexOfScalar(u8, colored, 0x1b) != null);
    try testing.expect(std.mem.indexOfScalar(u8, piped, 0x1b) == null);
    try testing.expect(colored.len > piped.len);
}

test "a row's description starts at the same column whatever the palette" {
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
