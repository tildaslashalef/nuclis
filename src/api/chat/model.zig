//! The language model `nuclis serve` keeps open: one at a time, beside the
//! decision pool, opened, run, and closed only on the GPU worker. It holds
//! the engine and its buffers (`completer.Open`), the `Completer` that
//! continues or restores a conversation, and the sampler, set up as the
//! agent sets them up for the same name, so a served conversation runs as
//! `nuclis chat` would run it. A request naming another model swaps it.
const std = @import("std");
const inference = @import("inference");
const config = @import("../../config.zig");
const catalog = @import("../../catalog.zig");
const paths = @import("../../paths.zig");
const engine = @import("../../engine.zig");
const completer_mod = @import("../../completer.zig");
const wire = @import("wire.zig");

const Allocator = std.mem.Allocator;
const Profile = inference.profiles;

pub const Language = struct {
    gpa: Allocator,
    /// Borrowed for the server's lifetime: the registry and the settings a
    /// name resolves to.
    loaded: *const config.Loaded,
    root: ?[]const u8,
    /// Set while a model is open; `completer` and `sampler` are valid then.
    open: ?completer_mod.Open = null,
    completer: completer_mod.Completer = undefined,
    sampler: inference.sampling.Sampler = undefined,
    /// The name the open model was asked for (owned), its settings, and the
    /// output budget a request that states none gets.
    name: ?[]u8 = null,
    settings: config.Resolved = undefined,
    default_tokens: usize = 0,
    /// Guards `name` for readers on connection tasks (`isOpen`).
    mutex: std.Io.Mutex = .init,

    /// What a request asked for that cannot be served, beside engine errors.
    pub const Error = error{ ModelNotFound, NotALanguageModel, ImagesUnsupported, ContextFull, InvalidSamplingOptions };

    /// The model a request names (or `engine.model`), opened on this worker
    /// and the one before it closed when it is another. Only registry and
    /// catalogue names: the server opens no path a client names.
    pub fn ensure(self: *Language, io: std.Io, requested: ?[]const u8) !void {
        const name = requested orelse self.loaded.config.engine.model;
        if (self.name) |current| if (std.mem.eql(u8, current, name)) return;
        const registry = self.loaded.config.models;
        if (registry.find(name)) |entry| {
            if (entry.kind == .decision) return error.NotALanguageModel;
        } else if (catalog.find(name) == null) {
            return if (catalog.findDecision(name) != null) error.NotALanguageModel else error.ModelNotFound;
        }
        const path = try paths.modelPath(self.gpa, name, "", self.root, registry);
        defer self.gpa.free(path);
        std.Io.Dir.cwd().access(io, path, .{}) catch return error.ModelNotFound;
        self.close(io);
        var settings = config.resolve(self.loaded, name, .{}, .agent);
        const default_tokens = settings.max_tokens;
        // Sized for the largest request; each request slices its own budget.
        settings.max_tokens = wire.max_output_tokens;
        self.open = try completer_mod.Open.init(self.gpa, io, path, settings);
        const open = &self.open.?;
        errdefer self.close(io);
        const profile = open.profile();
        const effort = profile.nearestEffort(settings.think);
        self.sampler = try inference.sampling.Sampler.init(0, profile.samplingOptions(effort, settings.sampling));
        self.completer = .{
            .alloc = self.gpa,
            .eng = &open.eng,
            .effort = effort,
            .sampler = &self.sampler,
            .history = &open.history,
            .buffers = .{ .logits = open.logits, .candidates = open.candidates, .generated = open.generated, .effort = effort },
            .thinking_budget = settings.thinking_budget,
            .speculative = .{ .enabled = settings.speculative, .draft_length = settings.draft_length },
            .cache = .{ .alloc = self.gpa, .budget = settings.cache.memory_bytes },
            .clamp_budget = true,
        };
        self.settings = settings;
        self.default_tokens = default_tokens;
        const owned = try self.gpa.dupe(u8, name);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.name = owned;
    }

    /// Closes the open model, if any; on the worker, or after it stopped.
    pub fn close(self: *Language, io: std.Io) void {
        if (self.open) |*open| {
            self.completer.deinit();
            open.deinit(self.gpa);
            self.open = null;
        }
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.name) |n| self.gpa.free(n);
        self.name = null;
    }

    /// The open model's name, copied into `arena`; any task.
    pub fn openName(self: *Language, io: std.Io, arena: Allocator) Allocator.Error!?[]const u8 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return if (self.name) |n| try arena.dupe(u8, n) else null;
    }

    /// Whether `name` is the open model; any task.
    pub fn isOpen(self: *Language, io: std.Io, name: []const u8) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return if (self.name) |n| std.mem.eql(u8, n, name) else false;
    }

    pub const Ran = struct {
        reply: completer_mod.Reply,
        effort: Profile.Effort,
    };

    /// Completes `request` on the open model, its events into `collected`.
    /// Everything the request needs lives in `arena`.
    pub fn run(self: *Language, io: std.Io, arena: Allocator, request: wire.Request, collected: *wire.Collector) !Ran {
        const open = &self.open.?;
        const profile = open.profile();
        const effort = profile.nearestEffort(request.effort orelse self.settings.think);
        self.completer.effort = effort;
        self.sampler.setOptions(profile.samplingOptions(effort, self.settings.sampling.merge(request.sampling))) catch
            return error.InvalidSamplingOptions;
        if (request.seed) |seed| self.sampler.rng = .init(seed);
        const limit = @min(request.max_tokens orelse self.default_tokens, open.generated.len);
        self.completer.buffers.generated = open.generated[0..limit];
        self.completer.buffers.effort = effort;

        const m = self.completer.model();
        const images = try arena.alloc(completer_mod.Image, request.images.len);
        if (images.len > 0) {
            try self.loadVision(io);
            for (request.images, images) |bytes, *image| image.* = .{
                .path = try arena.dupe(u8, "request"),
                .bytes = bytes.len,
                .prepared = try m.encode_image(m.context, arena, bytes),
            };
        }
        var next: usize = 0;
        for (request.messages, request.image_counts) |*message, count| {
            if (count == 0) continue;
            const refs = try arena.alloc(Profile.ImageRef, count);
            for (images[next..][0..count], refs) |image, *ref| ref.* = image.ref();
            message.images = refs;
            next += count;
        }

        var sink: completer_mod.Sink = .{ .context = collected, .call = collect };
        const reply = try m.run(m.context, request.messages, request.tools, images, &sink);
        // A turn that ends in an answer is a boundary the next request's
        // render starts with; a turn of calls continues the live session.
        if (reply.outcome.stop == .eos and collected.calls.items.len == 0) _ = m.checkpoint(m.context);
        return .{ .reply = reply, .effort = effort };
    }

    /// The tokens the open model's window holds.
    pub fn capacity(self: *const Language) usize {
        return self.open.?.eng.model.session().capacity;
    }

    fn loadVision(self: *Language, io: std.Io) !void {
        const open = &self.open.?;
        if (open.eng.vision != null) return;
        const name = self.name.?;
        const mmproj = if (self.settings.entry) |entry| entry.mmproj else if (catalog.find(name)) |entry| if (entry.companion(.mmproj)) |c| c.file else null else null;
        const path = (try engine.visionPath(self.gpa, open.model_path, mmproj)) orelse return error.ImagesUnsupported;
        defer self.gpa.free(path);
        std.Io.Dir.cwd().access(io, path, .{}) catch return error.ImagesUnsupported;
        try open.eng.loadVision(path, self.settings.image_max_tokens.count());
    }

    fn collect(context: *anyopaque, event: inference.events.Event) anyerror!void {
        const collected: *wire.Collector = @ptrCast(@alignCast(context));
        try collected.send(event);
    }
};
