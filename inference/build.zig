const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.addModule("inference", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const metal = b.option(bool, "metal", "Enable macOS Metal projection backend") orelse false;
    const options = b.addOptions();
    options.addOption(bool, "metal", metal);
    mod.addOptions("build_options", options);
    if (metal) {
        mod.addCSourceFile(.{ .file = b.path("src/backends/metal/bridge.m"), .flags = &.{"-fno-objc-arc"} });
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Metal", .{});
        mod.link_libc = true;
    }
    const tests = b.addTest(.{ .root_module = mod });
    b.step("test", "Test inference library without GPU or model weights").dependOn(&b.addRunArtifact(tests).step);

    const check = b.addExecutable(.{
        .name = "vocabulary-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("vocabulary-check.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "inference", .module = mod }},
        }),
    });
    b.installArtifact(check);
    const run_check = b.addRunArtifact(check);
    if (b.args) |args| run_check.addArgs(args);
    b.step("test-vocabulary", "Check the pinned vocabulary and tokenizer (-- MODEL_PATH)").dependOn(&run_check.step);
    const generation_check = b.addExecutable(.{
        .name = "generation-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("generation-check.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "inference", .module = mod }},
        }),
    });
    b.installArtifact(generation_check);
    const run_generation = b.addRunArtifact(generation_check);
    if (b.args) |args| run_generation.addArgs(args);
    b.step("test-generation", "Check CPU full-model session isolation/reset (-- MODEL_PATH)").dependOn(&run_generation.step);
    const metal_check = b.addExecutable(.{
        .name = "metal-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("metal-check.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "inference", .module = mod }},
        }),
    });
    b.installArtifact(metal_check);
    b.step("test-metal", "Explicit Metal fixtures and lifecycle checks (-Dmetal=true)").dependOn(&b.addRunArtifact(metal_check).step);
    const matvec_bench = b.addRunArtifact(metal_check);
    matvec_bench.addArg("--matvec-bench");
    b.step("bench-kernels", "Achieved weight bandwidth of the matvec kernels (-Dmetal=true)").dependOn(&matvec_bench.step);
    const matmul_bench = b.addRunArtifact(metal_check);
    matmul_bench.addArg("--matmul-bench");
    b.step("bench-matmul", "Throughput of the batched prefill matmul on model shapes (-Dmetal=true)").dependOn(&matmul_bench.step);
    const experts_bench = b.addRunArtifact(metal_check);
    experts_bench.addArg("--experts-bench");
    b.step("bench-experts", "Bandwidth of the gathered expert kernels on the 26B-A4B shape (-Dmetal=true)").dependOn(&experts_bench.step);
    const attention_bench = b.addRunArtifact(metal_check);
    attention_bench.addArg("--attention-bench");
    b.step("bench-attention", "Prefill chunk attention, row-split vs register-reuse at 4K-32K visible (-Dmetal=true)").dependOn(&attention_bench.step);
}
