//! The transcription models `nuclis serve` keeps: one open at a time
//! (`slot.zig`), counted in the memory budget, listed by `GET /v1/models`,
//! opened at start by `--model` or `--pin`. A route that transcribes
//! acquires it on the GPU worker. docs/guide/api.md § Memory.
const std = @import("std");
const inference = @import("inference");
const config = @import("../../config.zig");
const audio_catalog = @import("../../audio/catalog.zig");
const gpu = @import("../gpu.zig");
const models = @import("../models.zig");
const router_mod = @import("../router.zig");
const slot_mod = @import("../slot.zig");
const speech = @import("../speech/service.zig");
const live_route = @import("stream.zig");
const file = @import("file.zig");

const asr = inference.audio.nemotron_asr;

pub const Slot = slot_mod.Slot(asr.Model, .transcription);

pub const Service = struct {
    executor: *gpu.Executor,
    slot: Slot,
    root: ?[]const u8,
    registry: config.Models,
    /// `transcribe.model`: what a request that names none gets.
    default_model: []const u8,
    live: live_route.Live = .{},
    /// `POST /v1/audio/transcriptions`: whole recordings.
    files: file.Files = .{},

    /// `root` and `registry` are borrowed for the service's lifetime.
    pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, metal: bool, root: ?[]const u8, registry: config.Models, default_model: []const u8) Service {
        return .{ .executor = executor, .slot = .init(gpa, openModel, closeModel, .{ .metal = metal }), .root = root, .registry = registry, .default_model = default_model };
    }

    /// After the executor's worker stopped.
    pub fn deinit(self: *Service, io: std.Io) void {
        self.slot.deinit(io);
    }

    pub fn register(self: *Service, gpa: std.mem.Allocator, router: *router_mod.Router, listing: *models.Models) !void {
        try live_route.register(self, gpa, router);
        try self.files.register(gpa, router, self);
        try listing.add(gpa, .{ .context = self, .list = list });
    }

    /// Opens `name` now (a `--model` or `--pin` of `nuclis serve`) through the GPU.
    pub fn preload(self: *Service, arena: std.mem.Allocator, io: std.Io, name: []const u8, diag: *config.Diagnostic) !void {
        const located = try audio_catalog.locate(arena, io, self.root, self.registry, name, .transcription, false, diag);
        var opening: Opening = .{ .slot = &self.slot, .directory = located.directory, .name = name, .bytes = located.bytes };
        try self.executor.submit(io, &opening.item);
        opening.item.done.waitUncancelable(io);
        if (opening.failed) |err| {
            diag.set("the transcription model {s} did not open: {s}", .{ name, @errorName(err) });
            return err;
        }
    }

    fn list(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, out: *std.ArrayList(models.Entry)) anyerror!void {
        const self: *Service = @ptrCast(@alignCast(context));
        const first = out.items.len;
        try speech.listAudio(arena, io, self.root, self.registry, .transcription, self.default_model, null, &self.slot, out);
        for (out.items[first..]) |*e| {
            const details = &e.details.object;
            if (details.getPtr("limits")) |l| try l.object.put(arena, "max_file_bytes", .{ .integer = @intCast(self.files.max_bytes) });
            var diag: config.Diagnostic = .{};
            const resolved = audio_catalog.resolve(arena, self.root, self.registry, e.id, .transcription, false, &diag) catch continue;
            const languages = (try audio_catalog.transcriptionLanguages(arena, io, resolved.directory)) orelse &.{};
            try details.put(arena, "languages", .{ .array = try models.strings(arena, languages) });
            try details.put(arena, "options", .{ .object = try requestOptions(arena, languages) });
        }
    }
};

const Opening = struct {
    item: gpu.Item = .{ .run = run },
    slot: *Slot,
    directory: []const u8,
    name: []const u8,
    bytes: u64,
    failed: ?anyerror = null,

    fn run(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Opening = @alignCast(@fieldParentPtr("item", item));
        _ = self.slot.acquire(io, self.directory, self.name, self.bytes) catch |err| {
            self.failed = err;
        };
        return .done;
    }
};

/// The checkpoint, with its Metal plan when the server runs on Metal.
fn openModel(gpa: std.mem.Allocator, io: std.Io, directory: []const u8, options: Slot.Options) anyerror!*asr.Model {
    const model = try asr.Model.open(gpa, io, directory);
    errdefer model.close();
    if (options.metal) try model.openMetal();
    return model;
}

fn closeModel(model: *asr.Model, io: std.Io) void {
    _ = io;
    model.close();
}

test {
    _ = live_route;
}

/// The request fields a transcription model takes (the file route's; the
/// live route takes `language` and `lookahead` as query parameters).
fn requestOptions(arena: std.mem.Allocator, languages: []const []const u8) !std.json.ObjectMap {
    const Option = models.Option;
    const defaults: file.Request = .{ .file = "" };
    var o: std.json.ObjectMap = .empty;
    var language_values = try models.strings(arena, &.{"auto"});
    for (languages) |l| if (!std.mem.eql(u8, l, "auto")) try language_values.append(.{ .string = l });
    try o.put(arena, "language", try (Option{ .type = .@"enum", .values = language_values, .default = .{ .string = "auto" } }).value(arena));
    var formats: std.ArrayList([]const u8) = .empty;
    for (std.enums.values(file.Format)) |f| try formats.append(arena, @tagName(f));
    try o.put(arena, "response_format", try (Option{ .type = .@"enum", .values = try models.strings(arena, formats.items), .default = .{ .string = @tagName(defaults.format) } }).value(arena));
    try o.put(arena, "timestamp_granularities", try (Option{ .type = .enum_list, .values = try models.strings(arena, &.{ "segment", "word" }), .default = .{ .array = try models.strings(arena, &.{"segment"}) } }).value(arena));
    try o.put(arena, "stream", try (Option{ .type = .boolean, .default = .{ .bool = defaults.stream } }).value(arena));
    try o.put(arena, "lookahead", try (Option{ .type = .@"enum", .values = try models.integers(arena, inference.audio.nemotron_encoder.lookaheads), .default = .{ .integer = @intCast(defaults.lookahead) } }).value(arena));
    try o.put(arena, "phrases", try (Option{ .type = .string_list, .max_items = inference.audio.boost.max_phrases, .max_bytes = inference.audio.boost.max_phrase_bytes, .default = .{ .array = .init(arena) } }).value(arena));
    return o;
}
