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
    // than a version this build might not be. Read from the build root, not
    // the working directory (`zig build` may run in a subdirectory); the
    // configure cache already tracks every package's manifest.
    const manifest_version = blk: {
        const zon_path = b.pathJoin(&.{ b.root.sub_path, "build.zig.zon" });
        const zon = b.root.root_dir.handle.readFileAlloc(b.graph.io, zon_path, b.allocator, .limited(1 << 16)) catch
            break :blk "unknown";
        const key = ".version = \"";
        const start = std.mem.indexOf(u8, zon, key) orelse break :blk "unknown";
        const rest = zon[start + key.len ..];
        break :blk rest[0 .. std.mem.indexOfScalar(u8, rest, '"') orelse break :blk "unknown"];
    };
    // The commit the tree was built from, and whether the work tree differed
    // from it: the agent's prefix cache keys saved model states by build, so
    // a state is restored only into the build that computed it. Git state is
    // not a file the configure cache can track, so configure re-runs on every
    // build; otherwise a cached revision outlives the commit it names.
    b.graph.poisonCache();
    const root_path = b.root.root_dir.path orelse ".";
    const revision = blk: {
        var code: u8 = 0;
        const out = b.runAllowFail(&.{ "git", "-C", root_path, "rev-parse", "--short=12", "HEAD" }, &code, .ignore) catch break :blk "unknown";
        break :blk std.mem.trim(u8, out, " \n");
    };
    const dirty = blk: {
        var code: u8 = 0;
        const out = b.runAllowFail(&.{ "git", "-C", root_path, "status", "--porcelain" }, &code, .ignore) catch break :blk true;
        break :blk out.len != 0;
    };
    const version = reportedVersion(b, root_path, manifest_version, revision, dirty);
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption([]const u8, "revision", revision);
    build_options.addOption(bool, "dirty", dirty);

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
            // `src/` calls `std.c`; Linux links no libc unless asked.
            .link_libc = true,
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
    run.addPassthruArgs();
    b.step("run", "Run nuclis").dependOn(&run.step);

    const vocabulary_check = b.addRunArtifact(inference.artifact("vocabulary-check"));
    vocabulary_check.addPassthruArgs();
    b.step("test-vocabulary", "Check the pinned vocabulary and tokenizer (-- MODEL_PATH)").dependOn(&vocabulary_check.step);
    const laya_check = b.addRunArtifact(inference.artifact("laya-check"));
    laya_check.addPassthruArgs();
    b.step("test-laya", "Check Laya's CPU forward against the oracle's fixtures (-- LAYA_DIR)").dependOn(&laya_check.step);
    const clef_check = b.addRunArtifact(inference.artifact("clef-check"));
    clef_check.addPassthruArgs();
    b.step("test-clef", "Check clef-flash against the oracle's fixtures (-- sequences|head CLEF_DIR ...)").dependOn(&clef_check.step);
    const embeddinggemma_check = b.addRunArtifact(inference.artifact("embeddinggemma-check"));
    embeddinggemma_check.addPassthruArgs();
    b.step("test-embeddinggemma", "Check EmbeddingGemma 2 against the oracles, or time it (-- MODEL traces|vectors|google|bench [--backend cpu|metal])").dependOn(&embeddinggemma_check.step);
    const qwen3_tts_check = b.addRunArtifact(inference.artifact("qwen3-tts-check"));
    qwen3_tts_check.addPassthruArgs();
    b.step("test-qwen3-tts", "Check Qwen3-TTS against the authors' trace (-- MODEL_DIR codec|talker)").dependOn(&qwen3_tts_check.step);
    const drama3_check = b.addRunArtifact(inference.artifact("drama3-check"));
    drama3_check.addPassthruArgs();
    b.step("test-drama3", "Check Drama 3's backbone and tower on the CPU (-- MODEL.gguf smoke|trace)").dependOn(&drama3_check.step);
    const nemotron_asr_check = b.addRunArtifact(inference.artifact("nemotron-asr-check"));
    nemotron_asr_check.addPassthruArgs();
    b.step("test-nemotron-asr", "Check Nemotron ASR's front end and encoder against the pinned trace (-- MODEL_DIR [FIXTURES])").dependOn(&nemotron_asr_check.step);
    const generation_check = b.addRunArtifact(inference.artifact("generation-check"));
    generation_check.addPassthruArgs();
    b.step("test-generation", "Check CPU full-model session isolation/reset (-- MODEL_PATH)").dependOn(&generation_check.step);
    const cross_encoding_check = b.addRunArtifact(inference.artifact("cross-encoding-check"));
    cross_encoding_check.addPassthruArgs();
    b.step("test-cross-encoding", "Compare one model's matrices across two files' encodings (-- NEW VALIDATED [--rows n] [--min-cosine x])").dependOn(&cross_encoding_check.step);
    // Compiles the check tools without running them, so the gate runner
    // can time the build apart from the checks.
    const check_tools = b.step("check-tools", "Install the vocabulary, Laya, clef, EmbeddingGemma, Qwen3-TTS, Drama 3, Nemotron ASR, generation, and cross-encoding check tools");
    for ([_][]const u8{ "vocabulary-check", "laya-check", "clef-check", "embeddinggemma-check", "qwen3-tts-check", "drama3-check", "nemotron-asr-check", "generation-check", "cross-encoding-check" }) |name|
        check_tools.dependOn(&b.addInstallArtifact(inference.artifact(name), .{}).step);
    const metal_check = b.addRunArtifact(inference.artifact("metal-check"));
    metal_check.addPassthruArgs();
    b.step("test-metal", "Explicit Metal fixture checks (-Dmetal=true; -- --wide-rows ID: one encoding's matvec rows)").dependOn(&metal_check.step);
    const matvec_bench = b.addRunArtifact(inference.artifact("metal-check"));
    matvec_bench.addArg("--matvec-bench");
    matvec_bench.addPassthruArgs();
    b.step("bench-kernels", "Achieved weight bandwidth of the matvec kernels (-Dmetal=true)").dependOn(&matvec_bench.step);
    const matvec_split_bench = b.addRunArtifact(inference.artifact("metal-check"));
    matvec_split_bench.addArg("--matvec-split");
    matvec_split_bench.addPassthruArgs();
    b.step("bench-matvec-split", "Split-K matvec bandwidth on the row-poor shapes at 1/2/4/8 splits (-Dmetal=true)").dependOn(&matvec_split_bench.step);
    const matmul_bench = b.addRunArtifact(inference.artifact("metal-check"));
    matmul_bench.addArg("--matmul-bench");
    matmul_bench.addPassthruArgs();
    b.step("bench-matmul", "Throughput of the batched prefill matmul on model shapes (-Dmetal=true)").dependOn(&matmul_bench.step);
    const matvec_rows_bench = b.addRunArtifact(inference.artifact("metal-check"));
    matvec_rows_bench.addArg("--matvec-rows-bench");
    matvec_rows_bench.addPassthruArgs();
    b.step("bench-matvec-rows", "Multi-row matvec vs the 16x8 tile at 1-8 rows (-Dmetal=true)").dependOn(&matvec_rows_bench.step);
    const hadamard_bench = b.addRunArtifact(inference.artifact("metal-check"));
    hadamard_bench.addArg("--hadamard-bench");
    b.step("bench-hadamard", "GPU time of one token's Hadamard transforms on the Bonsai schedule (-Dmetal=true)").dependOn(&hadamard_bench.step);
    const experts_bench = b.addRunArtifact(inference.artifact("metal-check"));
    experts_bench.addArg("--experts-bench");
    experts_bench.addPassthruArgs();
    b.step("bench-experts", "Bandwidth of the gathered expert kernels on the 26B-A4B shape (-Dmetal=true)").dependOn(&experts_bench.step);
    const attention_bench = b.addRunArtifact(inference.artifact("metal-check"));
    attention_bench.addArg("--attention-bench");
    b.step("bench-attention", "Prefill chunk attention, row-split vs register-reuse at 4K-32K visible (-Dmetal=true)").dependOn(&attention_bench.step);
}

/// The version `--version` reports: the manifest's on a clean checkout of its
/// release tag (or with no git at all, as in a release's source archive);
/// otherwise the next minor as a prerelease with the commit in the build
/// metadata, `0.11.0-dev+48e8e1e[.dirty]`, so a binary built from `main`
/// never claims the last release.
fn reportedVersion(b: *std.Build, root_path: []const u8, manifest: []const u8, revision: []const u8, dirty: bool) []const u8 {
    if (std.mem.eql(u8, manifest, "unknown") or std.mem.eql(u8, revision, "unknown")) return manifest;
    if (!dirty) {
        var code: u8 = 0;
        const tag = b.fmt("v{s}", .{manifest});
        const out = b.runAllowFail(&.{ "git", "-C", root_path, "tag", "--list", "--points-at", "HEAD", tag }, &code, .ignore) catch "";
        if (std.mem.trim(u8, out, " \n").len != 0) return manifest;
    }
    // A prerelease manifest (0.11.0-rc.1) is already heading to its core
    // version; a release one (0.10.0) is followed by the next minor.
    const next = std.SemanticVersion.parse(manifest) catch return b.fmt("{s}-dev+{s}", .{ manifest, revision[0..@min(revision.len, 7)] });
    const minor = if (next.pre == null) next.minor + 1 else next.minor;
    const patch = if (next.pre == null) 0 else next.patch;
    return b.fmt("{d}.{d}.{d}-dev+{s}{s}", .{ next.major, minor, patch, revision[0..@min(revision.len, 7)], if (dirty) ".dirty" else "" });
}
