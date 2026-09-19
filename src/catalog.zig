//! The built-in catalogue of supported artifacts.
//!
//! "Supported" has one source: this table. A file under `models/` is
//! *runnable* when its architecture id has an adapter, *supported* only when
//! it is here with a pinned commit and SHA-256; `generate` runs either, and
//! `model ls` and the docs speak only of the catalogue. Entries are named
//! after the model, not the quantization (`qwen3.8-27b`), so a later
//! quantization change edits the entry rather than renaming it in every
//! configuration file. Companions (vision projector, MTP draft head) carry
//! the unit that will consume them; nothing loads them today, but a pulled
//! model directory is complete and verified before those units exist.
//!
//! Facts here are copied from the Hub as `nuclis model pull` resolved them
//! (docs/reference/artifacts.md § Pinned commits and digests); the digest
//! of the main file is also the spec's. Status derivation reads sidecars
//! only, never hashes a file, so a listing is instant.
const std = @import("std");
const inference = @import("inference");
const model = @import("model.zig");
const Allocator = std.mem.Allocator;

pub const Role = model.Role;

/// Which prompt profile (chat template, sampling defaults) an entry's
/// checkpoint uses: the profile registry's enum. The profile is the
/// checkpoint's, not the user's: `config show` names it and fills unset
/// sampling keys from it.
pub const Profile = inference.profiles.Profile;

pub const Companion = struct {
    role: Role,
    file: []const u8,
    size: u64,
    sha256: []const u8,
    /// The unit that will consume it (roadmap ids); informational until then.
    loaded_by: []const u8,
};

pub const Entry = struct {
    name: []const u8,
    repo: []const u8,
    file: []const u8,
    /// The 40-character commit every digest below was read at.
    revision: []const u8,
    sha256: []const u8,
    size: u64,
    quantization: []const u8,
    /// `general.architecture` of the main file.
    architecture: []const u8,
    /// The checkpoint's prompt profile, forced at open (which lets an entry
    /// pin a protocol onto a file whose own template digest is not); `null`
    /// while its profile unit is pending, when configuration falls back as
    /// it does for a bare path.
    profile: ?Profile,
    companions: []const Companion,

    pub fn companion(self: *const Entry, role: Role) ?*const Companion {
        for (self.companions) |*c| if (c.role == role) return c;
        return null;
    }
};

pub const entries = [_]Entry{
    .{
        .name = "qwen3.8-27b",
        .repo = "unsloth/Qwen3.8-27B-GGUF",
        .file = "Qwen3.8-27B-UD-Q4_K_M.gguf",
        .revision = "4ca720788d1e01f1bff70c033e0d0028fd02e502",
        .sha256 = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
        .size = 16_464_440_224,
        .quantization = "UD-Q4_K_M",
        .architecture = "qwen35",
        .profile = .qwen38,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-BF16.gguf", .size = 931_146_432, .sha256 = "83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53", .loaded_by = "the vision unit" },
            .{ .role = .mtp, .file = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", .size = 1_369_590_656, .sha256 = "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e", .loaded_by = "the MTP unit" },
        },
    },
    // Gemma 4 12B in two quantizations, as two entries. The plain name is
    // the ordinary K-quant release the adapter was brought up on,
    // and `-qat` is Google's quantization-aware-trained checkpoint,
    // whose every weight matrix is Q4_0 — the encoding it was *trained* for,
    // which is why it decodes faster at the same size. Naming them this way
    // (decided 2026-09-12) keeps the plain name on the plain release and
    // makes the special one ask for itself; both are pinned, both are
    // supported, and each has its own reference traces and acceptance record.
    .{
        .name = "gemma-4-12b",
        .repo = "unsloth/gemma-4-12b-it-GGUF",
        .file = "gemma-4-12b-it-UD-Q4_K_XL.gguf",
        .revision = "fc034cfff751157913579611efad8462ac1be606",
        .sha256 = "90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e",
        .size = 7_366_423_360,
        .quantization = "UD-Q4_K_XL",
        .architecture = "gemma4",
        .profile = .gemma4,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-BF16.gguf", .size = 175_115_840, .sha256 = "2e269f906eb15169ee9ce880ea649bd6d42d4964c21f8ede10d0d0efc738bcbb", .loaded_by = "the vision unit" },
            .{ .role = .mtp, .file = "mtp-gemma-4-12b-it.gguf", .size = 465_109_248, .sha256 = "145db9094bc0f85f1701e255a2ed216dcc9800fc8bc8631ad00905b456bd451b", .loaded_by = "the MTP unit" },
        },
    },
    .{
        .name = "gemma-4-12b-qat",
        .repo = "unsloth/gemma-4-12B-it-qat-GGUF",
        .file = "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        .revision = "980b060c40a8539ac159e0501a3e0f66a6365af3",
        .sha256 = "90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370",
        .size = 6_716_356_800,
        // The Hub name says UD-Q4_K_XL; the file's own header says
        // "smart Q4_0, QAT-lossless" and every matrix is Q4_0.
        .quantization = "Q4_0 (QAT)",
        .architecture = "gemma4",
        .profile = .gemma4,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-BF16.gguf", .size = 175_115_840, .sha256 = "dcb8103adad042b1bf99df767aaf34eb37c5a73a4a2f0417e4d7ba557e91664f", .loaded_by = "the vision unit" },
            .{ .role = .mtp, .file = "mtp-gemma-4-12B-it.gguf", .size = 253_708_800, .sha256 = "fcb35dea42c71333db904cee11baac525c9ef872818ee3753f6cb156f3c6f4f6", .loaded_by = "the MTP unit" },
        },
    },
    // The mixture-of-experts sibling, QAT file only: the K-quant release
    // stores expert down-projections as Q5_1, which no kernel executes.
    // Pinned 2026-09-17 by the pull; runnable once the adapter binds the
    // expert tensors (MODL-09), supported with its acceptance record (MODL-10).
    .{
        .name = "gemma-4-26b-a4b",
        .repo = "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        .file = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        .revision = "7b92b5b28818151e8669af2e45e88d6086f490dd",
        .sha256 = "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891",
        .size = 14_249_047_104,
        .quantization = "Q4_0 (QAT)",
        .architecture = "gemma4",
        .profile = .gemma4,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-BF16.gguf", .size = 1_194_828_256, .sha256 = "7b06953ccdbe8cf363f47841a7afaacd2b1c2ff9a8d6b426fdec7521a6878744", .loaded_by = "the vision unit" },
            .{ .role = .mtp, .file = "MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf", .size = 251_939_328, .sha256 = "7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a", .loaded_by = "the MTP unit" },
        },
    },
    // Meta's dense agentic model. Its draft companion is a DFlash drafter,
    // not an MTP head; it takes the `mtp` role because that role means
    // "the draft source the speculative-decoding unit loads", whatever
    // its mechanism (decided 2026-09-17). The adapter binds it since
    // MODL-11; the profile arrives with MODL-13.
    .{
        .name = "muse-glimmer-30b",
        .repo = "unsloth/Muse-Glimmer-30B-GGUF",
        .file = "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        .revision = "faa5b025c584459c13febfa5c59883516710ae39",
        .sha256 = "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
        .size = 15_878_222_368,
        .quantization = "UD-Q4_K_XL",
        .architecture = "muse-glimmer",
        .profile = null,
        .companions = &.{
            .{ .role = .mmproj, .file = "mmproj-kquant.gguf", .size = 1_400_328_928, .sha256 = "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6", .loaded_by = "the vision unit" },
            .{ .role = .mtp, .file = "dflash-kquant.gguf", .size = 1_631_205_312, .sha256 = "27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc", .loaded_by = "the speculative-decoding unit (a DFlash drafter)" },
        },
    },
    // Prism ML's ternary re-encoding of Qwen3.8-27B: the same architecture
    // as `qwen3.8-27b`, its weights ternary at group 128 in a Hadamard-rotated
    // basis (docs/reference/bonsai.md). The 2-bit-slot PQ2_0 packing was the
    // bring-up file; the entry moved to the denser PTQ1_0 packing of the same
    // weights on 2026-09-18 once measured not slower on the whole token
    // (13.05 against 12.87 tok/s) at 1.26 GB less, matching the same traces
    // (bench.md § Bonsai 2 27B acceptance record). The profile is the pinned Qwen3.8 one
    // (decided 2026-09-18): the file's upstream template renders every
    // conversation it accepts byte-identically, and refuses a second system
    // message the pinned one merges (docs/reference/bonsai.md).
    .{
        .name = "bonsai-2-27b",
        .repo = "prism-ml/Ternary-Bonsai-2-27B-gguf",
        .file = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        .revision = "6ed5e12bf84b7a63069882c91dd9e9218647d17b",
        .sha256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
        .size = 5_946_648_928,
        .quantization = "PTQ1_0 (ternary g128)",
        .architecture = "qwen35",
        .profile = .qwen38,
        .companions = &.{
            .{ .role = .mmproj, .file = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf", .size = 629_246_976, .sha256 = "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903", .loaded_by = "the vision unit" },
        },
    },
};

/// The widest entry name, so every listing lines its status column up
/// however the catalogue grows. Computed here rather than guessed at each
/// call site, which is what let `gemma-4-12b-qat` overflow a hardcoded 12.
pub const name_width = blk: {
    var widest: usize = 0;
    for (&entries) |entry| widest = @max(widest, entry.name.len);
    break :blk widest + 1;
};

/// Pads `text` to `name_width` in `buffer` (which must be at least that
/// wide), so padding happens before any escape sequence and colour can never
/// move a column.
pub fn padName(buffer: []u8, text: []const u8) []const u8 {
    const written = @min(text.len, buffer.len);
    @memcpy(buffer[0..written], text[0..written]);
    const total = @min(@max(written, name_width), buffer.len);
    @memset(buffer[written..total], ' ');
    return buffer[0..total];
}

/// Exact name match; names never contain `/` or end in `.gguf`, so a path
/// can never be mistaken for one.
pub fn find(name: []const u8) ?*const Entry {
    for (&entries) |*e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

/// A file the catalogue pins, main or companion, by repository and Hub
/// name; `model inspect` compares the Hub's digest against `sha256`.
pub const FileMatch = struct {
    entry: *const Entry,
    role: Role,
    sha256: []const u8,
    /// The companion's consuming unit; null for the main file.
    loaded_by: ?[]const u8,
};

pub fn findFile(repo: []const u8, file: []const u8) ?FileMatch {
    for (&entries) |*e| {
        if (!std.mem.eql(u8, e.repo, repo)) continue;
        if (std.mem.eql(u8, e.file, file)) return .{ .entry = e, .role = .main, .sha256 = e.sha256, .loaded_by = null };
        for (e.companions) |*c| if (std.mem.eql(u8, c.file, file)) return .{ .entry = e, .role = c.role, .sha256 = c.sha256, .loaded_by = c.loaded_by };
    }
    return null;
}

/// `<models>/<repo>/<file>`, the layout `model pull` writes.
pub fn localPath(alloc: Allocator, models_dir: []const u8, entry: *const Entry, file: []const u8) ![]u8 {
    return std.fs.path.join(alloc, &.{ models_dir, entry.repo, file });
}

/// From the sidecar alone: `present` when it records the pinned digest and
/// size, `mismatch` when it records something else, `unverified` when the
/// file is there without one (copied in by hand; a `pull` verifies it),
/// `absent` otherwise.
pub const Status = enum { present, absent, mismatch, unverified };

pub fn status(alloc: Allocator, io: std.Io, path: []const u8, sha256: []const u8, size: u64) !Status {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return .absent,
        else => return err,
    };
    // The sidecar's strings live for this call only.
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const sidecar = model.readSidecar(arena, io, .cwd(), try model.sidecarPath(arena, path)) catch |err| switch (err) {
        error.InvalidSidecar => return .mismatch,
        else => return err,
    } orelse return .unverified;
    return if (std.mem.eql(u8, sidecar.sha256, sha256) and sidecar.size == size) .present else .mismatch;
}

test "the table is well formed: unique names, 40-character commits, 64-character digests" {
    for (&entries, 0..) |e, i| {
        try std.testing.expectEqual(@as(usize, 40), e.revision.len);
        try std.testing.expectEqual(@as(usize, 64), e.sha256.len);
        try std.testing.expect(std.mem.indexOfScalar(u8, e.name, '/') == null and !std.mem.endsWith(u8, e.name, ".gguf"));
        for (e.companions) |c| {
            try std.testing.expectEqual(@as(usize, 64), c.sha256.len);
            try std.testing.expect(c.role != .main);
        }
        for (entries[i + 1 ..]) |other| try std.testing.expect(!std.mem.eql(u8, e.name, other.name));
    }
    try std.testing.expectEqualStrings("unsloth/Qwen3.8-27B-GGUF", find("qwen3.8-27b").?.repo);
    try std.testing.expectEqual(@as(?Profile, .gemma4), find("gemma-4-12b").?.profile);
    try std.testing.expectEqualStrings("gemma4", find("gemma-4-12b").?.architecture);
    try std.testing.expectEqual(@as(?Profile, .gemma4), find("gemma-4-26b-a4b").?.profile);
    try std.testing.expectEqual(@as(?Profile, null), find("muse-glimmer-30b").?.profile);
    try std.testing.expectEqual(@as(?Profile, .qwen38), find("bonsai-2-27b").?.profile);
    try std.testing.expectEqualStrings("qwen35", find("bonsai-2-27b").?.architecture);
    try std.testing.expect(find("bonsai-2-27b").?.companion(.mtp) == null);
    try std.testing.expectEqualStrings("muse-glimmer", find("muse-glimmer-30b").?.architecture);
    try std.testing.expectEqual(Role.mtp, findFile("unsloth/Muse-Glimmer-30B-GGUF", "dflash-kquant.gguf").?.role);
    try std.testing.expectEqual(Role.mtp, findFile("unsloth/gemma-4-12B-it-qat-GGUF", "mtp-gemma-4-12B-it.gguf").?.role);
    try std.testing.expect(find("qwen") == null);
    try std.testing.expect(find("qwen/Qwen3.8-27B-UD-Q4_K_M.gguf") == null);
    try std.testing.expectEqualStrings("the MTP unit", find("qwen3.8-27b").?.companion(.mtp).?.loaded_by);
    try std.testing.expect(find("qwen3.8-27b").?.companion(.imatrix) == null);
    try std.testing.expectEqual(Role.main, findFile("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf").?.role);
    try std.testing.expectEqualStrings("the vision unit", findFile("unsloth/Qwen3.8-27B-GGUF", "mmproj-BF16.gguf").?.loaded_by.?);
    try std.testing.expect(findFile("unsloth/Qwen3.8-27B-GGUF", "other.gguf") == null);
    try std.testing.expect(findFile("other/repo", "Qwen3.8-27B-UD-Q4_K_M.gguf") == null);
}

test "status derives from the sidecar without hashing" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    const entry = find("qwen3.8-27b").?;
    const path = try localPath(alloc, root, entry, entry.file);
    defer alloc.free(path);
    try std.testing.expectEqual(Status.absent, try status(alloc, io, path, entry.sha256, entry.size));
    try tmp.dir.createDirPath(io, "unsloth/Qwen3.8-27B-GGUF");
    try tmp.dir.writeFile(io, .{ .sub_path = "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf", .data = "GGUF" });
    try std.testing.expectEqual(Status.unverified, try status(alloc, io, path, entry.sha256, entry.size));
    var sidecar: model.Sidecar = .{ .repo = entry.repo, .file = entry.file, .revision = entry.revision, .sha256 = entry.sha256, .size = entry.size, .role = .main, .downloaded_at = "2026-09-11T00:00:00Z", .nuclis_version = model.version };
    const sidecar_path = try model.sidecarPath(alloc, path);
    defer alloc.free(sidecar_path);
    try model.writeSidecar(alloc, io, .cwd(), sidecar_path, sidecar);
    try std.testing.expectEqual(Status.present, try status(alloc, io, path, entry.sha256, entry.size));
    sidecar.size = 1;
    try model.writeSidecar(alloc, io, .cwd(), sidecar_path, sidecar);
    try std.testing.expectEqual(Status.mismatch, try status(alloc, io, path, entry.sha256, entry.size));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = sidecar_path, .data = "{" });
    try std.testing.expectEqual(Status.mismatch, try status(alloc, io, path, entry.sha256, entry.size));
}
