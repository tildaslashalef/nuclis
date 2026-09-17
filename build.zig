const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // Metal is the reason this engine exists on Apple Silicon, so it is on
    // by default wherever it can be built and off everywhere else; a CPU-only
    // build is `-Dmetal=false`. Before this, a plain `zig build` produced a
    // binary that failed at run time against the default configuration's
    // `backend: metal` (reported 2026-09-12).
    const metal_default = target.result.os.tag == .macos and target.result.cpu.arch == .aarch64;
    const metal = b.option(bool, "metal", "Build the Metal backend (default: on for macOS aarch64)") orelse metal_default;

    // Single source of truth for the version: parsed from this manifest so the
    // CLI's --version and the package version cannot drift (development.md §
    // Versioning). A manifest that cannot be read reports "unknown" rather
    // than a version this build might not be.
    const version = blk: {
        const zon = std.Io.Dir.cwd().readFileAlloc(b.graph.io, "build.zig.zon", b.allocator, .limited(1 << 16)) catch
            break :blk "unknown";
        const key = ".version = \"";
        const start = std.mem.indexOf(u8, zon, key) orelse break :blk "unknown";
        const rest = zon[start + key.len ..];
        break :blk rest[0 .. std.mem.indexOfScalar(u8, rest, '"') orelse break :blk "unknown"];
    };
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);

    const inference = b.dependency("inference", .{ .target = target, .optimize = optimize, .metal = metal });
    // The Hub downloader is imported by the executable only (`nuclis model`);
    // the inference library never depends on it (AGENTS.md § Repository
    // architecture).
    const huggingface = b.dependency("huggingface", .{ .target = target, .optimize = optimize });
    const exe = b.addExecutable(.{
        .name = "nuclis",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "inference", .module = inference.module("inference") },
                .{ .name = "huggingface", .module = huggingface.module("huggingface") },
                .{ .name = "build_options", .module = build_options.createModule() },
            },
        }),
    });
    b.installArtifact(exe);

    const tests = b.addTest(.{ .root_module = exe.root_module });
    const test_step = b.step("test", "Run all CPU-only unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
    test_step.dependOn(&inference.builder.top_level_steps.get("test").?.step);
    const hf_tests = &huggingface.builder.top_level_steps.get("test").?.step;
    test_step.dependOn(hf_tests);
    b.step("test-hf", "Offline tests of the huggingface package only").dependOn(hf_tests);
    // The package's standalone binary, installed on request only (`make hf-downloader`).
    b.step("hf-downloader", "Install the standalone Hugging Face downloader (zig-out/bin/hf-downloader)").dependOn(&b.addInstallArtifact(huggingface.artifact("hf-downloader"), .{}).step);

    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run nuclis").dependOn(&run.step);

    const vocabulary_check = b.addRunArtifact(inference.artifact("vocabulary-check"));
    if (b.args) |args| vocabulary_check.addArgs(args);
    b.step("test-vocabulary", "Check the pinned vocabulary and tokenizer (-- MODEL_PATH)").dependOn(&vocabulary_check.step);
    const generation_check = b.addRunArtifact(inference.artifact("generation-check"));
    if (b.args) |args| generation_check.addArgs(args);
    b.step("test-generation", "Check CPU full-model session isolation/reset (-- MODEL_PATH)").dependOn(&generation_check.step);
    b.step("test-metal", "Explicit Metal fixture checks (-Dmetal=true)").dependOn(&b.addRunArtifact(inference.artifact("metal-check")).step);
    const matvec_bench = b.addRunArtifact(inference.artifact("metal-check"));
    matvec_bench.addArg("--matvec-bench");
    if (b.args) |args| matvec_bench.addArgs(args);
    b.step("bench-kernels", "Achieved weight bandwidth of the matvec kernels (-Dmetal=true)").dependOn(&matvec_bench.step);
    const matmul_bench = b.addRunArtifact(inference.artifact("metal-check"));
    matmul_bench.addArg("--matmul-bench");
    if (b.args) |args| matmul_bench.addArgs(args);
    b.step("bench-matmul", "Throughput of the batched prefill matmul on model shapes (-Dmetal=true)").dependOn(&matmul_bench.step);
    const experts_bench = b.addRunArtifact(inference.artifact("metal-check"));
    experts_bench.addArg("--experts-bench");
    if (b.args) |args| experts_bench.addArgs(args);
    b.step("bench-experts", "Bandwidth of the gathered expert kernels on the 26B-A4B shape (-Dmetal=true)").dependOn(&experts_bench.step);
}
