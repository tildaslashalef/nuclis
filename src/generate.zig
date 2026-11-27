//! Bounded native generation. Model semantics stay in the inference adapter;
//! this module owns CLI presentation, prompt construction, and output files.
//! A rendered turn's tokens go through the profile's stream decoder, as the
//! agent's and the server's do, so the reasoning is kept apart from the
//! answer (`Split`); a raw prompt or a token array is text as decoded.
const std = @import("std");
const inference = @import("inference");
const engine = @import("engine.zig");
const expert_counts = @import("expert_counts.zig");
const config = @import("config.zig");
const interrupt = @import("interrupt.zig");
const kit = @import("ui.zig");
const tui_style = @import("tui/style.zig");

/// What only this command reads from the command line. The backend,
/// context, output budget, reasoning effort, and sampling overrides arrive
/// resolved (`config.Resolved`: defaults < file < flags).
pub const Options = struct {
    prompt: ?[]const u8 = null,
    prompt_file: ?[]const u8 = null,
    /// A JSON array of token ids fed untokenized (a reference run's exact
    /// input; also how a model without an encoder path is traced).
    prompt_tokens: ?[]const u8 = null,
    raw: bool = false,
    seed: ?u64 = null,
    logits_path: ?[]const u8 = null,
    trace_dir: ?[]const u8 = null,
    /// Image files attached to the prompt (`--image`, repeatable), fed to the
    /// model's projector; at most the vision contract's per-turn bound.
    images: [8][]const u8 = undefined,
    image_count: usize = 0,
    /// Audio files attached after the images (`--audio`, repeatable), fed to
    /// the companion's audio encoder (Gemma 4 E4B).
    clips: [8][]const u8 = undefined,
    clip_count: usize = 0,
};

/// Layer observer: enforces the wall-clock bound and propagates Io and Ctrl-C
/// cancellation at every layer boundary (`check`, no activations needed),
/// writes per-layer traces only when a directory is given (`layer`, which
/// makes the GPU plan synchronize after every layer), and forwards turn
/// progress to a caller that asked for it (`progress`).
pub const Trace = struct {
    io: std.Io,
    directory: ?[]const u8,
    position: usize = 0,
    started: std.Io.Timestamp,
    /// Where turn progress goes. The agent's status bar installs one;
    /// `generate` and `bench` leave it null, so a measured run never pays
    /// for display work.
    progress: ?Sink = null,
    /// Where a mixture of experts' routing is tallied (`bench --expert-counts`).
    experts: ?*expert_counts.Counts = null,

    /// A progress consumer: the callback and the context it belongs to. A
    /// second indirection because the observer's context is this `Trace`.
    pub const Sink = struct {
        context: *anyopaque,
        call: *const fn (*anyopaque, inference.observer.Progress) anyerror!void,
    };
    pub fn check(context: *anyopaque) !void {
        const self: *Trace = @ptrCast(@alignCast(context));
        try self.io.checkCancel();
        try interrupt.check();
        if (self.started.durationTo(std.Io.Clock.awake.now(self.io)).toSeconds() > 3600) return error.TimeLimitExceeded;
    }
    pub fn layer(context: *anyopaque, index: usize, values: []const f32) !void {
        const self: *Trace = @ptrCast(@alignCast(context));
        var path: [4096]u8 = undefined;
        const name = try std.fmt.bufPrint(&path, "{s}/token-{d}-layer-{d}.f32", .{ self.directory.?, self.position, index });
        try writeFloats(self.io, name, values);
    }
    fn forward(context: *anyopaque, value: inference.observer.Progress) !void {
        const self: *Trace = @ptrCast(@alignCast(context));
        const sink = self.progress.?;
        try sink.call(sink.context, value);
    }
    fn routing(context: *anyopaque, value: inference.observer.Routing) !void {
        const self: *Trace = @ptrCast(@alignCast(context));
        try self.experts.?.add(value);
    }
    pub fn observer(self: *Trace) inference.engine.Observer {
        return .{
            .context = self,
            .check = check,
            .layer = if (self.directory != null) layer else null,
            .progress = if (self.progress != null) forward else null,
            .experts = if (self.experts != null) routing else null,
        };
    }
};

fn writeFloats(io: std.Io, path: []const u8, values: []const f32) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    for (values) |value| try writer.interface.writeInt(u32, @bitCast(value), .little);
    try writer.interface.flush();
}

pub const Timing = inference.engine.Timing;
pub const Outcome = inference.engine.Outcome;
pub const Hooks = inference.engine.Hooks;

/// Runs `inference.engine.runLoop` with the executable's `Trace` as the
/// layer observer (Ctrl-C, time limit, optional per-layer files) and the
/// caller's hooks forwarded. Kept so `bench` and the agent share one call.
pub fn runLoop(
    eng: *engine.Engine,
    tokens: []const u32,
    limit: usize,
    sampler: *inference.sampling.Sampler,
    history: ?*inference.sampling.History,
    settings: inference.engine.Speculative,
    images: ?inference.engine.ImagePrefill,
    logits: []f32,
    candidates: []inference.sampling.Candidate,
    generated: []u32,
    trace: *Trace,
    hooks: ?Hooks,
) !Outcome {
    var bridge: Bridge = .{ .trace = trace, .hooks = hooks };
    return inference.engine.runLoop(eng, tokens, limit, sampler, history, settings, images, logits, candidates, generated, trace.observer(), .{
        .context = &bridge,
        .prefill = if (hooks != null and hooks.?.prefill != null) Bridge.prefill else null,
        .token = if (hooks != null and hooks.?.token != null) Bridge.token else null,
        .step = if (hooks != null and hooks.?.step != null) Bridge.step else null,
        .before_step = Bridge.beforeStep,
    });
}

/// Joins the trace's position bookkeeping with the caller's hooks behind one
/// context pointer.
const Bridge = struct {
    trace: *Trace,
    hooks: ?Hooks,
    fn beforeStep(context: *anyopaque, position: usize) !void {
        const self: *Bridge = @ptrCast(@alignCast(context));
        self.trace.position = position;
    }
    fn prefill(context: *anyopaque, logits: []const f32) !void {
        const self: *Bridge = @ptrCast(@alignCast(context));
        try self.hooks.?.prefill.?(self.hooks.?.context, logits);
    }
    fn token(context: *anyopaque, id: u32) !void {
        const self: *Bridge = @ptrCast(@alignCast(context));
        try self.hooks.?.token.?(self.hooks.?.context, id);
    }
    fn step(context: *anyopaque, position: usize) !void {
        const self: *Bridge = @ptrCast(@alignCast(context));
        try self.hooks.?.step.?(self.hooks.?.context, position);
    }
};

const Presenter = struct {
    eng: *engine.Engine,
    writer: *std.Io.Writer,
    stream: inference.text_stream.Stream = .{},
    enabled: bool,
    logits_path: ?[]const u8,
    /// Set for a rendered turn: the profile's decoder and what it split.
    turn: ?Turn = null,
    fn prefill(context: *anyopaque, logits: []const f32) !void {
        const self: *Presenter = @ptrCast(@alignCast(context));
        if (self.logits_path) |path| try writeFloats(self.eng.io, path, logits);
    }
    fn token(context: *anyopaque, id: u32) !void {
        const self: *Presenter = @ptrCast(@alignCast(context));
        if (self.eng.isStop(id)) return;
        if (self.turn) |*turn| {
            try turn.feed(self.eng.alloc, self.eng.vocab, id);
            if (self.enabled) try self.writer.flush();
            return;
        }
        if (!self.enabled) return;
        const piece = try inference.bpe.decode(self.eng.alloc, self.eng.vocab, &.{id}, false, .{});
        defer self.eng.alloc.free(piece);
        try self.stream.write(piece, self.writer);
        try self.writer.flush();
    }
};

/// A rendered turn: the profile's decoder, fed every generated token but
/// the stop tokens, and the `Split` it sends its events to.
const Turn = struct {
    decoder: inference.profiles.stream.Decoder,
    split: Split,

    fn deinit(self: *Turn) void {
        self.decoder.deinit();
        self.split.deinit();
    }

    /// One token, decoded with its control text as the decoder expects.
    fn feed(self: *Turn, alloc: std.mem.Allocator, vocab: *const inference.vocabulary.Vocabulary, id: u32) !void {
        const piece = try inference.bpe.decode(alloc, vocab, &.{id}, true, .{});
        defer alloc.free(piece);
        try self.decoder.feed(id, piece, &self.split);
    }

    /// Settles an open call and flushes held bytes at the completion's stop.
    fn end(self: *Turn, outcome: Outcome) !void {
        try self.decoder.end(outcome, &self.split);
    }
};

/// A rendered turn's events as `generate` presents them: the reasoning and
/// the answer collected apart for the report and, when `out` is set,
/// streamed there, the reasoning under a label in the chat surface's
/// thinking style (unstyled off a terminal: `paint` is then `.none`). A
/// call is neither: `generate` offers no tools and runs none, so a call
/// (`tool_call`) or a dropped one (`tool_cut`) is only counted, for `run`
/// to report; its tokens stay in the report's `tokens`.
pub const Split = struct {
    alloc: std.mem.Allocator,
    out: ?*std.Io.Writer = null,
    paint: tui_style.Style = .none,
    reasoning: std.ArrayList(u8) = .empty,
    answer: std.ArrayList(u8) = .empty,
    calls: usize = 0,
    cuts: usize = 0,
    /// What `out` showed last, and whether its last byte ended a line.
    shown: enum { nothing, reasoning, answer } = .nothing,
    line_ended: bool = true,

    pub fn deinit(self: *Split) void {
        self.reasoning.deinit(self.alloc);
        self.answer.deinit(self.alloc);
        self.* = undefined;
    }

    pub fn send(self: *Split, event: inference.events.Event) !void {
        switch (event) {
            .thinking => |bytes| {
                try self.reasoning.appendSlice(self.alloc, bytes);
                const out = self.out orelse return;
                if (self.shown != .reasoning) {
                    if (self.shown == .answer) try self.blankLine(out);
                    const label = self.paint.theme.glyphs().fold_open;
                    try out.print("{s}{s} Thinking{s}\n", .{ self.paint.on(.thinking_header), label, self.paint.off() });
                    self.shown = .reasoning;
                }
                // Each piece is its own span, so a stop at any point leaves
                // the terminal unstyled.
                try out.print("{s}{s}{s}", .{ self.paint.on(.thinking), bytes, self.paint.off() });
                self.noteEnd(bytes);
            },
            .answer => |bytes| {
                try self.answer.appendSlice(self.alloc, bytes);
                const out = self.out orelse return;
                if (self.shown == .reasoning) try self.blankLine(out);
                self.shown = .answer;
                try out.writeAll(bytes);
                self.noteEnd(bytes);
            },
            .tool_call => self.calls += 1,
            .tool_cut => self.cuts += 1,
            .tool_progress, .stop => {},
        }
    }

    fn noteEnd(self: *Split, bytes: []const u8) void {
        if (bytes.len != 0) self.line_ended = bytes[bytes.len - 1] == '\n';
    }

    /// Ends the line being written, then leaves one empty line.
    fn blankLine(self: *Split, out: *std.Io.Writer) !void {
        if (!self.line_ended) try out.writeByte('\n');
        try out.writeByte('\n');
        self.line_ended = true;
    }
};

/// A raw completion's text: every id decoded at once, control tokens
/// omitted, repaired to valid UTF-8.
fn rawText(alloc: std.mem.Allocator, vocab: *const inference.vocabulary.Vocabulary, ids: []const u32, out: *std.Io.Writer) !void {
    const decoded = try inference.bpe.decode(alloc, vocab, ids, false, .{});
    defer alloc.free(decoded);
    var stream: inference.text_stream.Stream = .{};
    try stream.write(decoded, out);
    try stream.finish(out);
}

/// Encodes inputs `0..count` in order through `encoder.encode(idx)` into a
/// slice the caller owns, features included. On an error it frees what it
/// encoded, so a caller only ever holds a complete result or nothing.
fn encodeAll(alloc: std.mem.Allocator, count: usize, encoder: anytype) ![]inference.engine.PreparedImage {
    const out = try alloc.alloc(inference.engine.PreparedImage, count);
    var loaded: usize = 0;
    errdefer {
        for (out[0..loaded]) |p| alloc.free(p.features);
        alloc.free(out);
    }
    while (loaded < count) : (loaded += 1) out[loaded] = try encoder.encode(loaded);
    return out;
}

const ImageEncoder = struct {
    eng: *engine.Engine,
    paths: []const []const u8,
    fn encode(self: *ImageEncoder, idx: usize) !inference.engine.PreparedImage {
        const alloc = self.eng.alloc;
        const bytes = engine.readImage(alloc, self.eng.io, self.paths[idx]) catch |err| {
            std.log.err("could not read image {d} ({s}): {s}", .{ idx + 1, self.paths[idx], @errorName(err) });
            return err;
        };
        defer alloc.free(bytes);
        return self.eng.encodeImage(bytes);
    }
};

const ClipEncoder = struct {
    eng: *engine.Engine,
    paths: []const []const u8,
    fn encode(self: *ClipEncoder, idx: usize) !inference.engine.PreparedImage {
        const alloc = self.eng.alloc;
        const path = self.paths[idx];
        const bytes = std.Io.Dir.cwd().readFileAlloc(self.eng.io, path, alloc, .limited(inference.audio.audio.max_bytes)) catch |err| {
            std.log.err("could not read audio {d} ({s}): {s}", .{ idx + 1, path, @errorName(err) });
            return err;
        };
        defer alloc.free(bytes);
        const clip = self.eng.encodeAudio(bytes) catch |err| {
            std.log.err("could not hear audio {d} ({s}): {s}", .{ idx + 1, path, @errorName(err) });
            return err;
        };
        return .{ .width_tokens = clip.tokens, .height_tokens = 1, .features = clip.features };
    }
};

/// `ui` styles the streamed text (the reasoning's label and span); `--json`
/// passes `.plain`.
pub fn run(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, settings: config.Resolved, options: Options, json: bool, writer: *std.Io.Writer, ui: kit.Ui) !void {
    const user: ?[]u8 = if (options.prompt_tokens == null) try engine.readPrompt(alloc, io, options.prompt, options.prompt_file) else null;
    defer if (user) |u| alloc.free(u);
    const limit = settings.max_tokens;
    const capacity = settings.ctx_size;
    // The file was range-checked when loaded; flags are checked here.
    if (limit == 0 or limit > config.max_output_tokens or capacity == 0 or capacity > config.max_context) return error.InvalidGenerationBudget;
    // The reasoning mode selects the official sampling profile; the file and
    // the flags override it per option (`samplingDefaults`), so an
    // unconfigured run samples like the reference configuration and
    // `--temperature 0` restores greedy decoding. Validation happens here,
    // before the model loads.
    var sampler = try inference.sampling.Sampler.init(options.seed orelse 0, settings.samplingOptions());
    // The switch decides the load: a drafter is loaded only when speculation
    // is on, and a missing source then reports `DraftSourceMissing` rather
    // than running silently. The entry's preference is its own embedded block
    // (Qwen) or the companion it names (Gemma 4, Muse Glimmer).
    const draft_path = try engine.draftPath(alloc, model_path, try settings.draftFile());
    defer if (draft_path) |path| alloc.free(path);
    const draft: inference.engine.DraftRequest = if (settings.speculative) .{ .preferred = draft_path } else .none;
    var eng = try engine.Engine.open(alloc, io, model_path, settings.backend, capacity, settings.kv_precision, settings.forced_profile, draft);
    defer eng.deinit();
    // The configuration resolved the profile from the catalogue name (a
    // bare path takes the first profile); the file's own template decides.
    if (eng.profile) |profile| if (profile != settings.profile) try sampler.setOptions(profile.samplingOptions(settings.think, settings.sampling));
    interrupt.install();
    // Images and clips attach to the prompt through the model's companion
    // encoders: each is decoded and encoded to feature rows, the profile
    // renders its markers, and the engine pairs the placeholder runs with
    // the rows.
    var image_encode: ?std.Io.Duration = null;
    var audio_encode: ?std.Io.Duration = null;
    var prepared: []inference.engine.PreparedImage = &.{};
    var heard: []inference.engine.PreparedImage = &.{};
    var media_prefill: ?inference.engine.MediaPrefill = null;
    defer {
        for (prepared) |p| alloc.free(p.features);
        alloc.free(prepared);
        for (heard) |p| alloc.free(p.features);
        alloc.free(heard);
        if (media_prefill) |*m| m.deinit(alloc);
    }
    if (options.image_count > 0 or options.clip_count > 0) {
        if (options.raw or options.prompt_tokens != null) return error.ImagesNeedChatPrompt;
    }
    if (options.image_count > 0) {
        const projector = try engine.visionPath(alloc, model_path, if (settings.entry) |entry| entry.mmproj else null) orelse return error.NoProjector;
        defer alloc.free(projector);
        try eng.loadVision(projector, settings.image_max_tokens.count());
        const encode_started = std.Io.Clock.awake.now(io);
        defer image_encode = encode_started.durationTo(std.Io.Clock.awake.now(io));
        var images: ImageEncoder = .{ .eng = &eng, .paths = options.images[0..options.image_count] };
        prepared = try encodeAll(alloc, options.image_count, &images);
    }
    if (options.clip_count > 0) {
        const projector = try engine.visionPath(alloc, model_path, if (settings.entry) |entry| entry.mmproj else null) orelse return error.NoProjector;
        defer alloc.free(projector);
        try eng.loadAudio(projector);
        const encode_started = std.Io.Clock.awake.now(io);
        defer audio_encode = encode_started.durationTo(std.Io.Clock.awake.now(io));
        var clips: ClipEncoder = .{ .eng = &eng, .paths = options.clips[0..options.clip_count] };
        heard = try encodeAll(alloc, options.clip_count, &clips);
    }
    const tokens = if (options.prompt_tokens) |path| try engine.readPromptTokens(alloc, io, path, eng.vocab.tokens.len) else if (prepared.len + heard.len > 0) blk: {
        const refs = try alloc.alloc(inference.profiles.ImageRef, prepared.len);
        defer alloc.free(refs);
        for (refs, prepared) |*r, p| r.* = .{ .width_tokens = p.width_tokens, .height_tokens = p.height_tokens };
        const clips = try alloc.alloc(inference.profiles.AudioRef, heard.len);
        defer alloc.free(clips);
        for (clips, heard) |*r, p| r.* = .{ .tokens = p.width_tokens };
        const prompt = try eng.render(&.{.{ .role = .user, .content = user.?, .images = refs, .audio = clips }}, &.{}, settings.think);
        defer alloc.free(prompt);
        const ids = try eng.encode(prompt);
        errdefer alloc.free(ids);
        media_prefill = try eng.mediaSpans(alloc, ids, prepared, heard);
        break :blk ids;
    } else blk: {
        const prompt = try eng.prompt(user.?, options.raw, settings.think);
        defer alloc.free(prompt);
        break :blk try eng.encode(prompt);
    };
    defer alloc.free(tokens);
    if (tokens.len > capacity or limit > capacity - tokens.len) return error.ContextFull;
    const logits = try alloc.alloc(f32, eng.vocab.tokens.len);
    defer alloc.free(logits);
    const candidates = try alloc.alloc(inference.sampling.Candidate, if (sampler.options.temperature == 0) 0 else eng.vocab.tokens.len);
    defer alloc.free(candidates);
    const generated = try alloc.alloc(u32, limit);
    defer alloc.free(generated);
    var history = try inference.sampling.History.init(alloc, eng.vocab.tokens.len);
    defer history.deinit();
    var trace: Trace = .{ .io = io, .directory = options.trace_dir, .started = std.Io.Clock.awake.now(io) };
    // Final prompt logits are written before decoding so a later failure
    // still leaves the comparison artifact behind.
    var presenter: Presenter = .{ .eng = &eng, .writer = writer, .enabled = !json, .logits_path = options.logits_path };
    // A rendered turn speaks the profile's grammar; literal text and a token
    // array are the caller's, shown as decoded.
    const rendered = !options.raw and options.prompt_tokens == null;
    if (rendered) if (eng.profile) |profile| {
        presenter.turn = .{
            .decoder = try profile.decoder(alloc, eng.vocab, settings.think, tokens),
            .split = .{ .alloc = alloc, .out = if (json) null else writer, .paint = ui.paint },
        };
    };
    defer if (presenter.turn) |*turn| turn.deinit();
    const outcome = try runLoop(&eng, tokens, limit, &sampler, &history, .{ .enabled = settings.speculative, .draft_length = settings.draft_length, .stop = settings.draft_stop, .max_position = settings.draft_max_depth }, if (media_prefill) |m| m.value() else null, logits, candidates, generated, &trace, .{ .context = &presenter, .prefill = if (options.logits_path != null) Presenter.prefill else null, .token = Presenter.token });
    const count = outcome.timing.generated_tokens;
    if (presenter.turn) |*turn| {
        try turn.end(outcome);
        const split = &turn.split;
        if (split.calls + split.cuts != 0) std.log.warn("the model wrote {d} tool call(s); generate offers no tools and runs none (the tokens are in --json's tokens)", .{split.calls + split.cuts});
    }
    var presentation: std.Io.Writer.Allocating = .init(alloc);
    defer presentation.deinit();
    if (presenter.turn == null) try rawText(alloc, eng.vocab, generated[0..count], &presentation.writer);
    const text = if (presenter.turn) |*turn| turn.split.answer.items else presentation.written();
    const reasoning = if (presenter.turn) |*turn| turn.split.reasoning.items else "";
    if (json) {
        try std.json.Stringify.value(.{
            .schema_version = @as(u32, 3),
            .backend = @tagName(settings.backend),
            .kv_precision = @tagName(eng.kv_precision),
            .session_bytes = eng.model.session().bytes(),
            .sampling = sampler.options,
            .speculative = settings.speculative,
            .draft_length = settings.draft_length,
            .accepted_drafts = outcome.timing.accepted_drafts,
            .proposed_drafts = outcome.timing.proposed_drafts,
            .seed = options.seed orelse 0,
            .prompt_tokens = tokens.len,
            .tokens = generated[0..count],
            .text = text,
            .reasoning = reasoning,
            .stop_reason = @tagName(outcome.stop),
            .stopped_eos = outcome.stop == .eos,
            .load_milliseconds = engine.milliseconds(eng.load),
            // Decoding and the projector over every image, before the prefill.
            .image_milliseconds = if (image_encode) |d| @as(?f64, engine.milliseconds(d)) else null,
            // Decoding, the log-mel frames, and the encoder over every clip.
            .audio_milliseconds = if (audio_encode) |d| @as(?f64, engine.milliseconds(d)) else null,
            .prefill_milliseconds = engine.milliseconds(outcome.timing.prefill),
            .first_token_milliseconds = engine.milliseconds(outcome.timing.first_token),
            .decode_milliseconds = engine.milliseconds(outcome.timing.decode),
            .gpu_busy_milliseconds = if (outcome.timing.gpu_seconds) |s| s * 1000 else null,
            .gpu_topk_fallbacks = outcome.timing.topk_fallbacks,
            .elapsed_milliseconds = trace.started.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds(),
        }, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, writer);
        try writer.writeByte('\n');
    } else {
        if (presenter.turn == null) try presenter.stream.finish(writer);
        try writer.writeByte('\n');
        if (outcome.stop == .cancelled) std.log.info("generation cancelled after {d} tokens", .{count});
    }
}

test "the trace observer carries progress only when a caller asks for it" {
    const io = std.testing.io;
    // What `generate` and `bench` build: cancellation and the time limit,
    // no per-layer files, and no progress — a measured run pays for nothing
    // it does not use.
    var plain: Trace = .{ .io = io, .directory = null, .started = std.Io.Clock.awake.now(io) };
    const bare = plain.observer();
    try std.testing.expect(bare.check != null);
    try std.testing.expect(bare.layer == null);
    try std.testing.expect(bare.progress == null);

    // What the agent builds: the same observer with its status bar attached.
    const Counter = struct {
        seen: usize = 0,
        last: inference.observer.Progress = .{ .phase = .prefill, .position = 0, .target = 0 },
        fn call(context: *anyopaque, value: inference.observer.Progress) !void {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.seen += 1;
            self.last = value;
        }
    };
    var counter: Counter = .{};
    var watched: Trace = .{
        .io = io,
        .directory = null,
        .started = std.Io.Clock.awake.now(io),
        .progress = .{ .context = &counter, .call = Counter.call },
    };
    const observer = watched.observer();
    try std.testing.expect(observer.progress != null);
    // The forwarder unwraps the trace's context, not the sink's.
    try observer.progress.?(observer.context, .{ .phase = .prefill, .position = 768, .target = 1226 });
    try observer.progress.?(observer.context, .{ .phase = .decode, .position = 3, .target = 64 });
    try std.testing.expectEqual(@as(usize, 2), counter.seen);
    try std.testing.expectEqual(inference.observer.Phase.decode, counter.last.phase);
    try std.testing.expectEqual(@as(usize, 3), counter.last.position);
}

test "a failing input frees what was encoded once and leaves the caller nothing" {
    const Failing = struct {
        alloc: std.mem.Allocator,
        fail_at: usize,
        fn encode(self: *@This(), idx: usize) !inference.engine.PreparedImage {
            if (idx == self.fail_at) return error.FileNotFound;
            return .{ .width_tokens = 1, .height_tokens = 1, .features = try self.alloc.alloc(f32, 4) };
        }
    };
    const alloc = std.testing.allocator;
    var second: Failing = .{ .alloc = alloc, .fail_at = 1 };
    try std.testing.expectError(error.FileNotFound, encodeAll(alloc, 3, &second));
    var first: Failing = .{ .alloc = alloc, .fail_at = 0 };
    try std.testing.expectError(error.FileNotFound, encodeAll(alloc, 2, &first));
    var none: Failing = .{ .alloc = alloc, .fail_at = 9 };
    const all = try encodeAll(alloc, 2, &none);
    defer {
        for (all) |p| alloc.free(p.features);
        alloc.free(all);
    }
    try std.testing.expectEqual(@as(usize, 2), all.len);
}

/// Control tokens of both test grammars (Inkling 2's channels, Qwen's
/// brackets) and a few words, so `Profile.decoder` resolves the markers as
/// on the files. Words are byte-level text (`Ġ` a space).
fn testVocabulary(alloc: std.mem.Allocator) !inference.vocabulary.Vocabulary {
    const controls = [_][]const u8{ "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|content_invoke_tool_json|>", "<|end_message|>", "<think>", "</think>", "<tool_call>", "</tool_call>" };
    const words = [_][]const u8{ "plan", "Hello", "Ġthere", "{\"name\":\"read\",\"args\":{}}" };
    var arena: std.heap.ArenaAllocator = .init(alloc);
    errdefer arena.deinit();
    const a = arena.allocator();
    var tokens: std.ArrayList(inference.vocabulary.Token) = .empty;
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    for (controls) |text| {
        try ids.put(a, text, @intCast(tokens.items.len));
        try tokens.append(a, .{ .text = text, .kind = .control });
    }
    for (words) |text| {
        try ids.put(a, text, @intCast(tokens.items.len));
        try tokens.append(a, .{ .text = text, .kind = .normal });
    }
    return .{ .storage = arena, .tokens = tokens.items, .pre = "test", .bos = null, .eos = null, .padding = null, .token_ids = ids, .merge_ranks = .empty };
}

/// Feeds `texts`, each one token of `vocab`, through `turn` and ends it at EOS.
fn feedAll(turn: *Turn, vocab: *const inference.vocabulary.Vocabulary, texts: []const []const u8) !void {
    for (texts) |text| try turn.feed(std.testing.allocator, vocab, vocab.tokenId(text).?);
    try turn.end(.{ .stop = .eos, .timing = .{ .prompt_tokens = 1, .generated_tokens = texts.len } });
}

test "a channel-grammar turn keeps the reasoning apart from the answer, even at effort off" {
    const alloc = std.testing.allocator;
    var vocab = try testVocabulary(alloc);
    defer vocab.deinit();
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    // What Inkling 2 writes at `Thinking effort level: 0`: a reasoning
    // segment first, then the answer.
    var turn: Turn = .{
        .decoder = try inference.profiles.Profile.inkling2.decoder(alloc, &vocab, .off, &.{}),
        .split = .{ .alloc = alloc, .out = &out.writer },
    };
    defer turn.deinit();
    try feedAll(&turn, &vocab, &.{ "<|message_model|>", "<|content_thinking|>", "plan", "<|end_message|>", "<|message_model|>", "<|content_text|>", "Hello", "Ġthere", "<|end_message|>" });
    try std.testing.expectEqualStrings("plan", turn.split.reasoning.items);
    try std.testing.expectEqualStrings("Hello there", turn.split.answer.items);
    // Off a terminal: the label alone sets the reasoning apart.
    try std.testing.expectEqualStrings("▾ Thinking\nplan\n\nHello there", out.written());
}

test "a bracket-grammar turn paints the reasoning on a terminal and leaves the answer plain" {
    const alloc = std.testing.allocator;
    var vocab = try testVocabulary(alloc);
    defer vocab.deinit();
    const paint: tui_style.Style = .{ .theme = .{ .kind = .truecolor }, .enabled = true };
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    // Qwen's thinking prompt ends inside `<think>`.
    var turn: Turn = .{
        .decoder = try inference.profiles.Profile.qwen38.decoder(alloc, &vocab, .medium, &.{vocab.tokenId("<think>").?}),
        .split = .{ .alloc = alloc, .out = &out.writer, .paint = paint },
    };
    defer turn.deinit();
    try feedAll(&turn, &vocab, &.{ "plan", "</think>", "Hello" });
    try std.testing.expectEqualStrings("plan", turn.split.reasoning.items);
    try std.testing.expectEqualStrings("Hello", turn.split.answer.items);
    const thinking = paint.on(.thinking);
    try std.testing.expect(thinking.len != 0);
    const expected = try std.mem.concat(alloc, u8, &.{ paint.on(.thinking_header), "▾ Thinking", paint.off(), "\n", thinking, "plan", paint.off(), "\n\nHello" });
    defer alloc.free(expected);
    try std.testing.expectEqualStrings(expected, out.written());

    // Thinking off: the prompt closed the reasoning, so all is answer and
    // nothing is labelled.
    var plain: std.Io.Writer.Allocating = .init(alloc);
    defer plain.deinit();
    var off: Turn = .{
        .decoder = try inference.profiles.Profile.qwen38.decoder(alloc, &vocab, .off, &.{ vocab.tokenId("<think>").?, vocab.tokenId("</think>").? }),
        .split = .{ .alloc = alloc, .out = &plain.writer, .paint = paint },
    };
    defer off.deinit();
    try feedAll(&off, &vocab, &.{ "Hello", "Ġthere" });
    try std.testing.expectEqualStrings("", off.split.reasoning.items);
    try std.testing.expectEqualStrings("Hello there", plain.written());
}

test "a call in a turn that offers no tools is neither answer nor reasoning, only counted" {
    const alloc = std.testing.allocator;
    var vocab = try testVocabulary(alloc);
    defer vocab.deinit();
    var turn: Turn = .{
        .decoder = try inference.profiles.Profile.inkling2.decoder(alloc, &vocab, .off, &.{}),
        .split = .{ .alloc = alloc },
    };
    defer turn.deinit();
    try feedAll(&turn, &vocab, &.{ "<|message_model|>", "<|content_invoke_tool_json|>", "{\"name\":\"read\",\"args\":{}}", "<|end_message|>" });
    try std.testing.expectEqual(@as(usize, 1), turn.split.calls);
    try std.testing.expectEqualStrings("", turn.split.answer.items);
    try std.testing.expectEqualStrings("", turn.split.reasoning.items);
}

test "a raw completion is text as decoded: no grammar, control tokens omitted" {
    const alloc = std.testing.allocator;
    var vocab = try testVocabulary(alloc);
    defer vocab.deinit();
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var ids: [4]u32 = undefined;
    for (&ids, [_][]const u8{ "<think>", "plan", "</think>", "Hello" }) |*id, text| id.* = vocab.tokenId(text).?;
    try rawText(alloc, &vocab, &ids, &out.writer);
    try std.testing.expectEqualStrings("planHello", out.written());
}
