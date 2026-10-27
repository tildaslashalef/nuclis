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

const asr = inference.audio.nemotron_asr;

pub const Slot = slot_mod.Slot(asr.Model, .transcription);

pub const Service = struct {
    executor: *gpu.Executor,
    slot: Slot,
    root: ?[]const u8,
    registry: config.Models,
    /// `transcribe.model`: what a request that names none gets.
    default_model: []const u8,

    /// `root` and `registry` are borrowed for the service's lifetime.
    pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, metal: bool, root: ?[]const u8, registry: config.Models, default_model: []const u8) Service {
        return .{ .executor = executor, .slot = .init(gpa, openModel, closeModel, .{ .metal = metal }), .root = root, .registry = registry, .default_model = default_model };
    }

    /// After the executor's worker stopped.
    pub fn deinit(self: *Service, io: std.Io) void {
        self.slot.deinit(io);
    }

    pub fn register(self: *Service, gpa: std.mem.Allocator, router: *router_mod.Router, listing: *models.Models) !void {
        _ = router;
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
        try speech.listAudio(arena, io, self.root, self.registry, .transcription, self.default_model, &self.slot, out);
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
