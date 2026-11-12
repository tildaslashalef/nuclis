const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.addModule("inference", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const metal = b.option(bool, "metal", "Build the Metal backend (the root build sets it)") orelse false;
    configure(b, mod, metal, .{});
    const tests = b.addTest(.{ .root_module = mod });
    const test_step = b.step("test", "Test inference library without GPU or model weights");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // The same sources with the adapters' pinned shapes swapped for tiny ones
    // (docs/models/README.md § Tiny models). Only tiny-check imports this
    // module: nothing the application links can see those shapes.
    const tiny_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    configure(b, tiny_mod, metal, .{ .tiny_models = true });
    const tiny_root = b.createModule(.{
        .root_source_file = b.path("tiny-check.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "inference", .module = tiny_mod }},
    });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tiny_root })).step);
    const tiny_check = b.addExecutable(.{ .name = "tiny-check", .root_module = tiny_root });
    b.installArtifact(tiny_check);

    // The same sources with the benchmark-only Metal kernels (split-K, the
    // fragment layout pin) compiled in. Only metal-check imports it, so the
    // application's library neither embeds nor compiles them.
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    configure(b, bench_mod, metal, .{ .bench_kernels = true });

    // The check executables. The root declares every step that runs them
    // (AGENTS.md § Repository architecture); this file only builds them.
    const checks = [_][]const u8{
        "vocabulary-check",     "laya-check",       "clef-check",
        "embeddinggemma-check", "qwen3-tts-check",  "drama3-check",
        "nemotron-asr-check",   "generation-check", "cross-encoding-check",
        "metal-check",
    };
    for (checks) |name| {
        b.installArtifact(b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("{s}.zig", .{name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "inference", .module = if (std.mem.eql(u8, name, "metal-check")) bench_mod else mod }},
            }),
        }));
    }
}

/// What one instance of the module swaps in: the adapters' tiny shapes, or
/// the benchmark-only Metal kernels.
const Variant = struct { tiny_models: bool = false, bench_kernels: bool = false };

/// The build options and platform bridges of one instance of the module.
fn configure(b: *std.Build, mod: *std.Build.Module, metal: bool, variant: Variant) void {
    const options = b.addOptions();
    options.addOption(bool, "metal", metal);
    options.addOption(bool, "tiny_models", variant.tiny_models);
    options.addOption(bool, "bench_kernels", variant.bench_kernels);
    mod.addOptions("build_options", options);
    if (metal) {
        mod.addCSourceFile(.{ .file = b.path("src/backends/metal/bridge.m"), .flags = &.{"-fno-objc-arc"} });
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Metal", .{});
        // The image decoder is the other macOS bridge (vision/image.zig).
        mod.addCSourceFile(.{ .file = b.path("src/vision/image_bridge.m"), .flags = &.{"-fno-objc-arc"} });
        mod.linkFramework("CoreGraphics", .{});
        mod.linkFramework("ImageIO", .{});
        // And the audio decoder (audio/audio.zig).
        mod.addCSourceFile(.{ .file = b.path("src/audio/audio_bridge.m"), .flags = &.{"-fno-objc-arc"} });
        mod.linkFramework("AudioToolbox", .{});
        mod.link_libc = true;
    }
}
