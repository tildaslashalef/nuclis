//! Bounded native generation. Model semantics stay in the inference adapter;
//! this module owns CLI presentation, prompt construction, and output files.
const std = @import("std");
const inference = @import("inference");
const engine = @import("engine.zig");
const config = @import("config.zig");
const interrupt = @import("interrupt.zig");

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
    pub fn observer(self: *Trace) inference.engine.Observer {
        return .{
            .context = self,
            .check = check,
            .layer = if (self.directory != null) layer else null,
            .progress = if (self.progress != null) forward else null,
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
    fn prefill(context: *anyopaque, logits: []const f32) !void {
        const self: *Presenter = @ptrCast(@alignCast(context));
        if (self.logits_path) |path| try writeFloats(self.eng.io, path, logits);
    }
    fn token(context: *anyopaque, id: u32) !void {
        const self: *Presenter = @ptrCast(@alignCast(context));
        if (!self.enabled or self.eng.isStop(id)) return;
        const piece = try inference.bpe.decode(self.eng.alloc, &self.eng.vocab, &.{id}, false, .{});
        defer self.eng.alloc.free(piece);
        try self.stream.write(piece, self.writer);
        try self.writer.flush();
    }
};

pub fn run(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, settings: config.Resolved, options: Options, json: bool, writer: *std.Io.Writer) !void {
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
    const draft_path = try engine.draftPath(alloc, model_path, if (settings.entry) |entry| entry.mtp else null);
    defer if (draft_path) |path| alloc.free(path);
    const draft: inference.engine.DraftRequest = if (settings.speculative) .{ .preferred = draft_path } else .none;
    var eng = try engine.Engine.open(alloc, io, model_path, settings.backend, capacity, settings.kv_precision, settings.forced_profile, draft);
    defer eng.deinit();
    // The configuration resolved the profile from the catalogue name (a
    // bare path takes the first profile); the file's own template decides.
    if (eng.profile) |profile| if (profile != settings.profile) try sampler.setOptions(profile.samplingOptions(settings.think, settings.sampling));
    interrupt.install();
    // Images attach to the prompt through the model's projector: each is
    // decoded and encoded to feature rows, the profile renders its markers,
    // and the engine locates the placeholder runs to pair them with the rows.
    var prepared: []inference.engine.PreparedImage = &.{};
    var image_prefill: ?inference.engine.ImagePrefill = null;
    var image_features: []f32 = &.{};
    var image_spans: []inference.vision.Span = &.{};
    defer {
        for (prepared) |p| alloc.free(p.features);
        alloc.free(prepared);
        alloc.free(image_features);
        alloc.free(image_spans);
    }
    if (options.image_count > 0) {
        if (options.raw or options.prompt_tokens != null) return error.ImagesNeedChatPrompt;
        const projector = try engine.visionPath(alloc, model_path, if (settings.entry) |entry| entry.mmproj else null) orelse return error.NoProjector;
        defer alloc.free(projector);
        try eng.loadVision(projector);
        prepared = try alloc.alloc(inference.engine.PreparedImage, options.image_count);
        var loaded: usize = 0;
        errdefer for (prepared[0..loaded]) |p| alloc.free(p.features);
        var total_rows: usize = 0;
        for (options.images[0..options.image_count], 0..) |path, idx| {
            const bytes = engine.readImage(alloc, io, path) catch |err| {
                std.log.err("could not read image {d} ({s}): {s}", .{ idx + 1, path, @errorName(err) });
                return err;
            };
            defer alloc.free(bytes);
            prepared[idx] = try eng.encodeImage(bytes);
            loaded += 1;
            total_rows += prepared[idx].tokens();
        }
    }
    const tokens = if (options.prompt_tokens) |path| try engine.readPromptTokens(alloc, io, path, eng.vocab.tokens.len) else if (options.image_count > 0) blk: {
        const refs = try alloc.alloc(inference.profiles.ImageRef, prepared.len);
        defer alloc.free(refs);
        for (refs, prepared) |*r, p| r.* = .{ .width_tokens = p.width_tokens, .height_tokens = p.height_tokens };
        const prompt = try eng.render(&.{.{ .role = .user, .content = user.?, .images = refs }}, &.{}, settings.think);
        defer alloc.free(prompt);
        const ids = try eng.encode(prompt);
        errdefer alloc.free(ids);
        image_spans = try eng.locateImageSpans(ids, prepared);
        var rows: usize = 0;
        for (prepared) |p| rows += p.tokens();
        image_features = try alloc.alloc(f32, rows * inference.vision.qwen3vl.output_width);
        var off: usize = 0;
        for (prepared) |p| {
            @memcpy(image_features[off..][0..p.features.len], p.features);
            off += p.features.len;
        }
        image_prefill = .{ .spans = image_spans, .features = image_features };
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
    const outcome = try runLoop(&eng, tokens, limit, &sampler, &history, .{ .enabled = settings.speculative, .draft_length = settings.draft_length }, image_prefill, logits, candidates, generated, &trace, .{ .context = &presenter, .prefill = if (options.logits_path != null) Presenter.prefill else null, .token = Presenter.token });
    const count = outcome.timing.generated_tokens;
    const decoded = try inference.bpe.decode(alloc, &eng.vocab, generated[0..count], false, .{});
    defer alloc.free(decoded);
    var presentation: std.Io.Writer.Allocating = .init(alloc);
    defer presentation.deinit();
    var json_stream: inference.text_stream.Stream = .{};
    try json_stream.write(decoded, &presentation.writer);
    try json_stream.finish(&presentation.writer);
    const text = presentation.written();
    if (json) {
        try std.json.Stringify.value(.{
            .schema_version = @as(u32, 2),
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
            .stop_reason = @tagName(outcome.stop),
            .stopped_eos = outcome.stop == .eos,
            .load_milliseconds = engine.milliseconds(eng.load),
            .prefill_milliseconds = engine.milliseconds(outcome.timing.prefill),
            .first_token_milliseconds = engine.milliseconds(outcome.timing.first_token),
            .decode_milliseconds = engine.milliseconds(outcome.timing.decode),
            .gpu_busy_milliseconds = if (outcome.timing.gpu_seconds) |s| s * 1000 else null,
            .gpu_topk_fallbacks = outcome.timing.topk_fallbacks,
            .elapsed_milliseconds = trace.started.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds(),
        }, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, writer);
        try writer.writeByte('\n');
    } else {
        try presenter.stream.finish(writer);
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
