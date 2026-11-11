const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.addModule("huggingface", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tests = b.addTest(.{ .root_module = mod });
    b.step("test", "Offline library tests").dependOn(&b.addRunArtifact(tests).step);
    const exe = b.addExecutable(.{
        .name = "hf-downloader",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "huggingface", .module = mod }},
        }),
    });
    // The root installs it on request (`make hf-downloader`) and declares
    // every step; this file only builds the module, its tests, and the binary.
    b.installArtifact(exe);
}
