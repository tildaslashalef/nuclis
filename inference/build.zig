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
    configure(b, mod, metal, false);
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
    configure(b, tiny_mod, metal, true);
    const tiny_root = b.createModule(.{
        .root_source_file = b.path("tiny-check.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "inference", .module = tiny_mod }},
    });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tiny_root })).step);
    const tiny_check = b.addExecutable(.{ .name = "tiny-check", .root_module = tiny_root });
    b.installArtifact(tiny_check);

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
                .imports = &.{.{ .name = "inference", .module = mod }},
            }),
        }));
    }
}

/// The build options and platform bridges of one instance of the module.
fn configure(b: *std.Build, mod: *std.Build.Module, metal: bool, tiny_models: bool) void {
    const options = b.addOptions();
    options.addOption(bool, "metal", metal);
    options.addOption(bool, "tiny_models", tiny_models);
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
