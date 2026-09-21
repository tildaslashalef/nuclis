//! Engine configuration: `<root>/nuclis.json`.
//!
//! One typed, sectioned file holds what would otherwise be repeated on every
//! command line: the model, backend, context, output budget, reasoning
//! effort, sampling overrides, the agent's thinking fold, and a `models`
//! registry of named entries that locate a model and carry their own
//! overrides. Five layers combine, in order: the built-in defaults
//! (`Config{}`), the model's sampling profile, the file's global sections,
//! the registry entry the model names, and the command-line `Flags`.
//! `resolve` applies them and records which layer produced each value.
//! `NUCLIS_HOME` only moves the root; there are no per-key environment
//! overrides and no per-project files.
//!
//! Sampling overrides default to `null` — "the official profile of the
//! reasoning mode" (`Profile.samplingDefaults`) — so the file never freezes a
//! model's recommended settings into user state; `config show` prints the
//! effective values and names `profile` as their source.
//!
//! `load` returns a `Loaded` that owns an arena for the file's strings;
//! `Resolved` borrows from it and must not outlive it. Every function is pure
//! over an injected `std.Io.Dir` and path, so tests run in temporary roots.
//! `Config` is the schema, walked at compile time.
const std = @import("std");
const inference = @import("inference");
const catalog = @import("catalog.zig");
const style = @import("tui/style.zig");
const tui_theme = @import("tui/theme.zig");
const Allocator = std.mem.Allocator;

pub const ThemeName = tui_theme.Name;
pub const Backend = inference.engine.Backend;
pub const KvPrecision = inference.engine.KvPrecision;
pub const Effort = inference.profiles.Effort;
pub const Overrides = inference.sampling.Overrides;
pub const Profile = catalog.Profile;

/// The only schema this build reads and writes. Adding a key with a default
/// does not bump it (a missing key takes the default); renaming or
/// re-typing one does. The `models` registry was added to schema 1:
/// a file without it loads with an empty registry.
pub const schema_version: u32 = 1;
/// Upper bound on the file; a configuration is a few hundred bytes.
pub const max_file_bytes = 64 * 1024;
/// Range rules shared with the command-line flags (`generate`, `bench`, and
/// the chat check their resolved values against the same bounds).
pub const max_context = 32768;
pub const max_output_tokens = 4096;
/// The largest draft block: KERN-11's 8-row token tile less the seed row.
/// One host constant, the engine's.
pub const max_draft_length = inference.engine.max_draft_length;
/// `bench` never takes its output budget from the file: the measurement
/// workload is a command-line matter so runs stay comparable.
pub const bench_max_tokens = 32;
/// Registry entry names are short handles (`gemma`, `qwen-q3`); the bound
/// also sizes the dotted-path buffers of the diagnostics.
pub const max_entry_name_bytes = 64;
const max_path_bytes = 160;

/// The file's shape and the built-in defaults. Section structs nest; every
/// non-struct field is a key, except `models`, a map handled by itself.
pub const Config = struct {
    schema_version: u32 = schema_version,
    engine: Engine = .{},
    generation: Generation = .{},
    agent: Agent = .{},
    models: Models = .{},

    pub const Engine = struct {
        /// A registry entry name, a catalogue name (`src/catalog.zig`), or a
        /// path relative to `<root>/models` unless absolute, in that order
        /// (`paths.modelPath`).
        model: []const u8 = "qwen3.8-27b",
        backend: Backend = if (inference.metal.enabled) .metal else .cpu,
        /// 16K: room for an agent turn's system prompt, tools, and a few
        /// results on top of the prompt (8K ran out in ordinary use).
        ctx_size: usize = 16384,
        /// Attention cache precision on the GPU; the CPU reference
        /// always keeps F32 and reports it.
        kv_precision: KvPrecision = .f16,
    };
    pub const Generation = struct {
        max_tokens: usize = 2048,
        think: Effort = .off,
        /// Speculative decoding: propose drafts with the model's draft source
        /// and verify them in batches (docs/spec.md § Speculative decoding).
        speculative: bool = false,
        /// Drafts proposed per step, 1..`max_draft_length`.
        draft_length: usize = 4,
        /// Per-option overrides of the reasoning mode's profile; `null`
        /// keeps the profile's value.
        sampling: Overrides = .{},
    };
    pub const Agent = struct {
        think: Effort = .low,
        /// Start with thinking folded (Tab unfolds).
        fold_thinking: bool = true,
        /// Named palette of the terminal surface (`tui/theme.zig`). A theme
        /// changes colour, never layout, so it is a display key with no
        /// per-model meaning (registry entries do not carry one).
        theme: ThemeName = tui_theme.default_name,
    };
};

/// One registry entry: where a model lives and what it overrides for that
/// model only. `path` (relative to `<root>/models` unless absolute) or
/// `repo` + `file` (the layout `model pull` writes, `<root>/models/<repo>/<file>`)
/// locate it; `revision` pins the commit `model pull <name>` fetches;
/// `mmproj` and `mtp` name companion files in the same directory for the
/// consumers and for `model pull <name> --with`.
/// Every other key is `null` by default, meaning "the file's global value".
pub const ModelEntry = struct {
    path: ?[]const u8 = null,
    repo: ?[]const u8 = null,
    file: ?[]const u8 = null,
    revision: ?[]const u8 = null,
    mmproj: ?[]const u8 = null,
    mtp: ?[]const u8 = null,
    /// Forces the prompt profile on a file whose template is not pinned (a
    /// finetune converted with another template revision); null selects by
    /// the file's template digest.
    profile: ?Profile = null,
    ctx_size: ?usize = null,
    generation: Generation = .{},
    agent: Agent = .{},

    pub const Generation = struct {
        max_tokens: ?usize = null,
        think: ?Effort = null,
        speculative: ?bool = null,
        draft_length: ?usize = null,
        sampling: Overrides = .{},
    };
    pub const Agent = struct {
        think: ?Effort = null,
        fold_thinking: ?bool = null,
    };
};

pub const NamedModel = struct { name: []const u8, entry: ModelEntry };

/// The registry: an object keyed by entry name in the file, a slice here
/// (a handful of entries; the file's order is kept for `show`). Names never
/// contain `/` or end in `.gguf`, so a path is never taken for one.
pub const Models = struct {
    entries: []const NamedModel = &.{},

    pub fn find(self: Models, name: []const u8) ?*const ModelEntry {
        for (self.entries) |*named| if (std.mem.eql(u8, named.name, name)) return &named.entry;
        return null;
    }

    /// `std.json.Stringify` calls this instead of walking the struct, so
    /// the file shows a map, not a list of `{name, entry}` pairs.
    pub fn jsonStringify(self: Models, jws: *std.json.Stringify) !void {
        try jws.beginObject();
        for (self.entries) |named| {
            try jws.objectField(named.name);
            try jws.write(named.entry);
        }
        try jws.endObject();
    }
};

/// Which layer produced a value, lowest first.
pub const Source = enum { default, profile, file, model, flag };

fn isSection(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and T != Models;
}

/// Number of keys under `T` (a key is any field that is not a section;
/// the registry has none: its keys are not part of the schema).
fn leafCount(comptime T: type) usize {
    if (T == Models) return 0;
    if (!isSection(T)) return 1;
    var n: usize = 0;
    for (@typeInfo(T).@"struct".fields) |f| n += leafCount(f.type);
    return n;
}

/// Position of the key named by a dotted `path` in a pre-order walk of `T`.
/// Always called as `comptime leafIndex(...)`; an unknown path does not
/// compile.
fn leafIndex(comptime T: type, comptime path: []const u8) usize {
    // The walk over every field's name comparison grows with the key count;
    // the default 1,000-branch quota ran out at the fourteenth key.
    @setEvalBranchQuota(4000);
    const dot = std.mem.indexOfScalar(u8, path, '.');
    const head = if (dot) |d| path[0..d] else path;
    var offset: usize = 0;
    inline for (@typeInfo(T).@"struct".fields) |f| {
        if (comptime std.mem.eql(u8, f.name, head)) {
            if (f.type == Models) @compileError("the registry has no key slots: " ++ path);
            if (dot) |d| return offset + leafIndex(f.type, path[d + 1 ..]);
            if (isSection(f.type)) @compileError("config path names a section, not a key: " ++ path);
            return offset;
        }
        offset += comptime leafCount(f.type);
    }
    @compileError("no such config key: " ++ path);
}

pub const leaf_count = leafCount(Config);
/// One `Source` per key, in the walk order of `Config`.
pub const Origin = [leaf_count]Source;

/// Carries the human-readable reason for a typed error (the key path, the
/// expected type or range) since Zig errors carry no payload.
pub const Diagnostic = struct {
    buffer: [256]u8 = undefined,
    len: usize = 0,
    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buffer[0..self.len];
    }
    pub fn set(self: *Diagnostic, comptime fmt: []const u8, args: anytype) void {
        self.len = if (std.fmt.bufPrint(&self.buffer, fmt, args)) |s| s.len else |_| self.buffer.len;
    }
};

pub const Loaded = struct {
    arena: std.heap.ArenaAllocator,
    config: Config = .{},
    origin: Origin = @splat(.default),
    /// Where the file was looked for, as given to `load`; empty when there
    /// was no user root.
    path: []const u8 = "",
    /// Whether a file was read (false: defaults only).
    found: bool = false,

    pub fn deinit(self: *Loaded) void {
        self.arena.deinit();
    }
    pub fn source(self: *const Loaded, comptime path: []const u8) Source {
        return self.origin[comptime leafIndex(Config, path)];
    }
};

/// Reads `path` relative to `dir` (`Dir.cwd()` with an absolute path in
/// production). A missing file yields the defaults with `found == false`
/// and never writes; every other failure is an error, and the validation
/// errors describe the offending key in `diag`. `path == null` means there
/// is no user root: defaults only.
pub fn load(alloc: Allocator, io: std.Io, dir: std.Io.Dir, path: ?[]const u8, diag: *Diagnostic) !Loaded {
    const file_path = path orelse return .{ .arena = .init(alloc) };
    const text = dir.readFileAlloc(io, file_path, alloc, .limited(max_file_bytes + 1)) catch |err| switch (err) {
        error.FileNotFound => {
            var loaded: Loaded = .{ .arena = .init(alloc) };
            errdefer loaded.deinit();
            loaded.path = try loaded.arena.allocator().dupe(u8, file_path);
            return loaded;
        },
        error.StreamTooLong => {
            diag.set("{s}: larger than {d} bytes", .{ file_path, max_file_bytes });
            return error.ConfigTooLarge;
        },
        else => return err,
    };
    defer alloc.free(text);
    return fromText(alloc, text, file_path, diag);
}

/// Parses and validates one document. Split from `load` so the error paths
/// are tested on literal text.
pub fn fromText(alloc: Allocator, text: []const u8, path: []const u8, diag: *Diagnostic) !Loaded {
    var loaded: Loaded = .{ .arena = .init(alloc), .found = true };
    errdefer loaded.deinit();
    const arena = loaded.arena.allocator();
    loaded.path = try arena.dupe(u8, path);
    // The dynamic tree is walked once and dropped; strings the config keeps
    // are copied into the arena on the way.
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, text, .{}) catch |err| {
        diag.set("{s}: not valid JSON ({s})", .{ path, @errorName(err) });
        return error.InvalidConfigJson;
    };
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |object| object,
        else => {
            diag.set("{s}: the document must be a JSON object", .{path});
            return error.InvalidConfigValue;
        },
    };
    // The version is checked before the keys: a file from a newer schema
    // should say so, not fail on the first key this build does not know.
    const version = root.get("schema_version") orelse {
        diag.set("{s}: schema_version is required (this build writes {d})", .{ path, schema_version });
        return error.UnsupportedConfigVersion;
    };
    switch (version) {
        .integer => |n| if (n != schema_version) {
            diag.set("{s}: schema_version {d} is not supported by this build (it reads {d}); move the file aside, run `nuclis config init`, and copy your settings back", .{ path, n, schema_version });
            return error.UnsupportedConfigVersion;
        },
        else => {
            diag.set("{s}: schema_version must be the integer {d}", .{ path, schema_version });
            return error.UnsupportedConfigVersion;
        },
    }
    try applySection(Config, &loaded.config, root, "", &loaded.origin, 0, arena, diag);
    try validate(&loaded.config, diag);
    return loaded;
}

fn hasField(comptime T: type, name: []const u8) bool {
    inline for (@typeInfo(T).@"struct".fields) |f| if (std.mem.eql(u8, f.name, name)) return true;
    return false;
}

/// Copies the keys present in `object` onto `target`, recursing into
/// sections. Unknown keys are rejected first so a typo is reported even
/// when the rest of the section is valid. `prefix` is the dotted path of
/// `object` with its trailing dot ("" at the root, "models.<name>." inside
/// a registry entry, hence a run-time string); `origin` records the file
/// layer for the keys of `Config` from slot `base` on, and `base == null`
/// (registry entries, which have no origin slots) prunes that write at
/// compile time.
fn applySection(comptime T: type, target: *T, object: std.json.ObjectMap, prefix: []const u8, origin: ?*Origin, comptime base: ?usize, arena: Allocator, diag: *Diagnostic) !void {
    var it = object.iterator();
    while (it.next()) |entry| {
        if (!hasField(T, entry.key_ptr.*)) {
            // The section was `chat` until 2026-09-11; same keys, new name.
            if (hasField(T, "agent") and std.mem.eql(u8, entry.key_ptr.*, "chat"))
                diag.set("unknown key {s}chat: the section is now agent (rename it; its keys are unchanged)", .{prefix})
            else
                diag.set("unknown key {s}{s}", .{ prefix, entry.key_ptr.* });
            return error.UnknownConfigKey;
        }
    }
    // One buffer per nesting level: the unrolled loop below uses it for one
    // field at a time, and a recursive call brings its own frame.
    var path_buffer: [max_path_bytes]u8 = undefined;
    var prefix_buffer: [max_path_bytes]u8 = undefined;
    comptime var offset: usize = 0;
    inline for (@typeInfo(T).@"struct".fields) |field| {
        if (object.get(field.name)) |value| {
            const path = std.fmt.bufPrint(&path_buffer, "{s}{s}", .{ prefix, field.name }) catch unreachable;
            if (field.type == Models) {
                @field(target, field.name) = switch (value) {
                    .object => |map| try parseModels(map, arena, diag),
                    else => {
                        diag.set("{s} must be an object", .{path});
                        return error.InvalidConfigValue;
                    },
                };
            } else if (comptime isSection(field.type)) {
                switch (value) {
                    .object => |section| {
                        const section_prefix = std.fmt.bufPrint(&prefix_buffer, "{s}.", .{path}) catch unreachable;
                        try applySection(field.type, &@field(target, field.name), section, section_prefix, origin, if (base) |b| b + offset else null, arena, diag);
                    },
                    else => {
                        diag.set("{s} must be an object", .{path});
                        return error.InvalidConfigValue;
                    },
                }
            } else {
                @field(target, field.name) = try parseLeaf(field.type, value, path, arena, diag);
                if (base) |b| origin.?[b + offset] = .file;
            }
        }
        offset += comptime leafCount(field.type);
    }
}

fn printable(text: []const u8) bool {
    for (text) |byte| if (byte < 0x21 or byte == 0x7f) return false;
    return true;
}

/// The registry object: every key is an entry name, every value an entry
/// parsed like a section under `models.<name>.`.
fn parseModels(map: std.json.ObjectMap, arena: Allocator, diag: *Diagnostic) !Models {
    const entries = try arena.alloc(NamedModel, map.count());
    var it = map.iterator();
    var index: usize = 0;
    while (it.next()) |kv| : (index += 1) {
        const name = kv.key_ptr.*;
        if (name.len == 0 or name.len > max_entry_name_bytes or !printable(name) or std.mem.indexOfScalar(u8, name, '/') != null or std.mem.endsWith(u8, name, ".gguf")) {
            diag.set("models: entry name \"{s}\" must be 1..{d} printable characters, contain no `/`, and not end in .gguf", .{ name, max_entry_name_bytes });
            return error.InvalidConfigValue;
        }
        var prefix_buffer: [max_path_bytes]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buffer, "models.{s}.", .{name}) catch unreachable;
        const object = switch (kv.value_ptr.*) {
            .object => |object| object,
            else => {
                diag.set("models.{s} must be an object", .{name});
                return error.InvalidConfigValue;
            },
        };
        var entry: ModelEntry = .{};
        try applySection(ModelEntry, &entry, object, prefix, null, null, arena, diag);
        entries[index] = .{ .name = try arena.dupe(u8, name), .entry = entry };
    }
    return .{ .entries = entries };
}

/// What a key of type `T` accepts, for error messages.
fn describe(comptime T: type) []const u8 {
    return switch (@typeInfo(T)) {
        .bool => "true or false",
        .int => "a non-negative integer",
        .float => "a number",
        .pointer => "a string",
        .optional => |o| "null or " ++ describe(o.child),
        .@"enum" => |e| blk: {
            comptime var names: []const u8 = "one of ";
            inline for (e.fields, 0..) |f, i| names = names ++ (if (i > 0) "|" else "") ++ f.name;
            break :blk names;
        },
        else => @compileError("unsupported config key type " ++ @typeName(T)),
    };
}

fn parseLeaf(comptime T: type, value: std.json.Value, path: []const u8, arena: Allocator, diag: *Diagnostic) !T {
    return parseValue(T, T, value, path, arena, diag);
}

/// `Described` is the key's whole type for the mismatch message ("null or
/// a number" for an optional key), `T` the part being parsed.
fn parseValue(comptime T: type, comptime Described: type, value: std.json.Value, path: []const u8, arena: Allocator, diag: *Diagnostic) !T {
    switch (@typeInfo(T)) {
        .optional => |o| return if (value == .null) null else try parseValue(o.child, Described, value, path, arena, diag),
        .bool => if (value == .bool) return value.bool,
        .int => if (value == .integer) {
            if (std.math.cast(T, value.integer)) |n| return n;
        },
        .float => switch (value) {
            .integer => |n| return @floatFromInt(n),
            .float => |f| return @floatCast(f),
            else => {},
        },
        .@"enum" => if (value == .string) {
            if (std.meta.stringToEnum(T, value.string)) |tag| return tag;
        },
        .pointer => if (value == .string) {
            if (value.string.len > 0) return try arena.dupe(u8, value.string);
            diag.set("{s} must not be empty", .{path});
            return error.InvalidConfigValue;
        },
        else => @compileError("unsupported config key type " ++ @typeName(T)),
    }
    diag.set("{s} must be {s}", .{ path, comptime describe(Described) });
    return error.InvalidConfigValue;
}

fn validateRange(value: usize, path: []const u8, max: usize, diag: *Diagnostic) !void {
    if (value == 0 or value > max) {
        diag.set("{s} must be 1..{d} (found {d})", .{ path, max, value });
        return error.InvalidConfigValue;
    }
}

/// Each sampling option is checked alone through `Sampler.init` so the
/// message names the key and the rule stays in the sampler.
fn validateSampling(overrides: Overrides, path: []const u8, diag: *Diagnostic) !void {
    inline for (@typeInfo(Overrides).@"struct".fields) |field| {
        if (@field(overrides, field.name)) |value| {
            var one: Overrides = .{};
            @field(one, field.name) = value;
            _ = inference.sampling.Sampler.init(0, (inference.sampling.Options{}).override(one)) catch {
                diag.set("{s}.{s} is out of range (found {d})", .{ path, field.name, value });
                return error.InvalidConfigValue;
            };
        }
    }
}

/// Range rules, the same the commands apply to flags, for the global
/// sections and every registry entry; an entry must locate its model one
/// way (`path`, or `repo` and `file`).
fn validate(cfg: *const Config, diag: *Diagnostic) !void {
    try validateRange(cfg.engine.ctx_size, "engine.ctx_size", max_context, diag);
    try validateRange(cfg.generation.max_tokens, "generation.max_tokens", max_output_tokens, diag);
    try validateRange(cfg.generation.draft_length, "generation.draft_length", max_draft_length, diag);
    try validateSampling(cfg.generation.sampling, "generation.sampling", diag);
    for (cfg.models.entries) |named| {
        const e = named.entry;
        var buffer: [max_path_bytes]u8 = undefined;
        const prefix = std.fmt.bufPrint(&buffer, "models.{s}", .{named.name}) catch unreachable;
        if (e.path != null and (e.repo != null or e.file != null or e.revision != null)) {
            diag.set("{s}: path excludes repo, file, and revision", .{prefix});
            return error.InvalidConfigValue;
        }
        // An entry named as a catalogue entry shadows it (`paths.modelPath`
        // tries the registry first), so it may only restate the catalogue's
        // location; another file wants another name.
        if (catalog.find(named.name)) |c| {
            const same = e.path == null and e.repo != null and e.file != null and std.mem.eql(u8, e.repo.?, c.repo) and std.mem.eql(u8, e.file.?, c.file);
            if (!same) {
                diag.set("{s}: {s} is a catalogue name pinned to {s}/{s}; an entry under it may not locate another file (choose another name)", .{ prefix, named.name, c.repo, c.file });
                return error.InvalidConfigValue;
            }
        }
        if (e.path == null and (e.repo == null or e.file == null)) {
            diag.set("{s}: needs path, or repo and file", .{prefix});
            return error.InvalidConfigValue;
        }
        if (e.repo) |repo| if (std.mem.count(u8, repo, "/") != 1 or repo[0] == '/' or repo[repo.len - 1] == '/') {
            diag.set("{s}.repo must be owner/repo (found {s})", .{ prefix, repo });
            return error.InvalidConfigValue;
        };
        var key_buffer: [max_path_bytes]u8 = undefined;
        if (e.ctx_size) |n| try validateRange(n, std.fmt.bufPrint(&key_buffer, "{s}.ctx_size", .{prefix}) catch unreachable, max_context, diag);
        if (e.generation.max_tokens) |n| try validateRange(n, std.fmt.bufPrint(&key_buffer, "{s}.generation.max_tokens", .{prefix}) catch unreachable, max_output_tokens, diag);
        if (e.generation.draft_length) |n| try validateRange(n, std.fmt.bufPrint(&key_buffer, "{s}.generation.draft_length", .{prefix}) catch unreachable, max_draft_length, diag);
        try validateSampling(e.generation.sampling, std.fmt.bufPrint(&key_buffer, "{s}.generation.sampling", .{prefix}) catch unreachable, diag);
    }
}

/// The command-line layer: only what the user stated. `--model` is passed
/// to `resolve` beside these because it selects the registry entry as well
/// as naming the file (`paths.modelPath` turns the value into a path).
pub const Flags = struct {
    backend: ?Backend = null,
    ctx_size: ?usize = null,
    kv: ?KvPrecision = null,
    max_tokens: ?usize = null,
    think: ?Effort = null,
    speculative: ?bool = null,
    draft_length: ?usize = null,
    /// `--prompt-profile`: see `ModelEntry.profile`.
    prompt_profile: ?Profile = null,
    sampling: Overrides = .{},
};

pub const Command = enum { generate, bench, agent };

/// What a command runs with after precedence. Borrows `config_file`,
/// `model`, and `entry` from the `Loaded` it came from.
pub const Resolved = struct {
    /// `--model` or `engine.model`: a registry name, a catalogue name, or a
    /// path (`paths.modelPath` decides which).
    model: []const u8,
    /// The registry entry `model` names, if any.
    entry: ?*const ModelEntry,
    /// The sampling profile the model runs with (see `pinnedProfile`), or the
    /// forced one.
    profile: Profile,
    /// A profile the flag or the registry entry forces on the file at open,
    /// whatever its template digest; null selects by digest.
    forced_profile: ?Profile,
    backend: Backend,
    ctx_size: usize,
    kv_precision: KvPrecision,
    max_tokens: usize,
    think: Effort,
    /// Speculative decoding on this run, and the drafts proposed per step.
    speculative: bool,
    draft_length: usize,
    /// Overrides over the reasoning mode's profile (`generate`, the agent);
    /// for `bench`, the flags alone over neutral greedy options.
    sampling: Overrides,
    fold_thinking: bool,
    /// The agent's palette; the other commands ignore it.
    theme: ThemeName,
    /// The file the values came from; null when only defaults and flags applied.
    config_file: ?[]const u8,
    origin: Origin,
    command: Command,

    pub fn source(self: *const Resolved, comptime path: []const u8) Source {
        return self.origin[comptime leafIndex(Config, path)];
    }

    /// The reasoning mode's profile with the overrides applied: the
    /// options a sampler runs with (`bench` uses neutral options instead).
    pub fn samplingOptions(self: *const Resolved) inference.sampling.Options {
        return self.profile.samplingDefaults(self.think).override(self.sampling);
    }
};

/// The sampling profile belongs to the checkpoint. The catalogue records it
/// per entry so `config show` can name it without opening the file: the
/// profile a catalogue entry pins onto its file, or null for a registry
/// entry or a bare path. The pin is
/// forced at open, so a catalogue file whose own template is not pinned
/// (Bonsai renders the Qwen3.8 protocol) still renders; a file whose digest
/// is pinned detects the same profile anyway, so nothing is reported forced.
fn pinnedProfile(model: []const u8) ?Profile {
    return if (catalog.find(model)) |entry| entry.profile else null;
}

/// Applies defaults < profile < file < entry < flags for one command. The
/// entry is the registry's for `model` (`--model`, else `engine.model`).
/// `bench` reads only the `engine` section and the entry's `ctx_size`: its
/// budget and sampling are command-line only.
pub fn resolve(loaded: *const Loaded, model: ?[]const u8, flags: Flags, command: Command) Resolved {
    const cfg = loaded.config;
    const name = model orelse cfg.engine.model;
    const entry = cfg.models.find(name);
    const e: ModelEntry = if (entry) |p| p.* else .{};
    const bench = command == .bench;
    var r: Resolved = .{
        .model = name,
        .entry = entry,
        .profile = flags.prompt_profile orelse e.profile orelse pinnedProfile(name) orelse .qwen38,
        .forced_profile = flags.prompt_profile orelse e.profile orelse pinnedProfile(name),
        .backend = flags.backend orelse cfg.engine.backend,
        .ctx_size = flags.ctx_size orelse e.ctx_size orelse cfg.engine.ctx_size,
        .kv_precision = flags.kv orelse cfg.engine.kv_precision,
        .max_tokens = flags.max_tokens orelse if (bench) bench_max_tokens else e.generation.max_tokens orelse cfg.generation.max_tokens,
        .think = flags.think orelse switch (command) {
            .generate => e.generation.think orelse cfg.generation.think,
            .agent => e.agent.think orelse cfg.agent.think,
            .bench => .off,
        },
        .speculative = flags.speculative orelse e.generation.speculative orelse cfg.generation.speculative,
        .draft_length = flags.draft_length orelse e.generation.draft_length orelse cfg.generation.draft_length,
        .sampling = if (bench) flags.sampling else cfg.generation.sampling.merge(e.generation.sampling).merge(flags.sampling),
        .fold_thinking = e.agent.fold_thinking orelse cfg.agent.fold_thinking,
        .theme = cfg.agent.theme,
        .config_file = if (loaded.found) loaded.path else null,
        .origin = loaded.origin,
        .command = command,
    };
    const o = &r.origin;
    if (model != null) o[comptime leafIndex(Config, "engine.model")] = .flag;
    if (flags.backend != null) o[comptime leafIndex(Config, "engine.backend")] = .flag;
    if (flags.ctx_size != null) o[comptime leafIndex(Config, "engine.ctx_size")] = .flag else if (e.ctx_size != null) o[comptime leafIndex(Config, "engine.ctx_size")] = .model;
    if (flags.kv != null) o[comptime leafIndex(Config, "engine.kv_precision")] = .flag;
    const budget = comptime leafIndex(Config, "generation.max_tokens");
    if (flags.max_tokens != null) o[budget] = .flag else if (bench) o[budget] = .default else if (e.generation.max_tokens != null) o[budget] = .model;
    // The entry's efforts are not command-specific, the flag is.
    if (e.generation.think != null) o[comptime leafIndex(Config, "generation.think")] = .model;
    if (e.agent.think != null) o[comptime leafIndex(Config, "agent.think")] = .model;
    if (flags.think != null) o[if (command == .agent) comptime leafIndex(Config, "agent.think") else comptime leafIndex(Config, "generation.think")] = .flag;
    if (e.agent.fold_thinking != null) o[comptime leafIndex(Config, "agent.fold_thinking")] = .model;
    if (e.generation.speculative != null) o[comptime leafIndex(Config, "generation.speculative")] = .model;
    if (flags.speculative != null) o[comptime leafIndex(Config, "generation.speculative")] = .flag;
    if (e.generation.draft_length != null) o[comptime leafIndex(Config, "generation.draft_length")] = .model;
    if (flags.draft_length != null) o[comptime leafIndex(Config, "generation.draft_length")] = .flag;
    inline for (@typeInfo(Overrides).@"struct".fields) |field| {
        const index = comptime leafIndex(Config, "generation.sampling." ++ field.name);
        if (@field(flags.sampling, field.name) != null) o[index] = .flag else if (bench) o[index] = .default else if (@field(e.generation.sampling, field.name) != null) o[index] = .model else if (@field(cfg.generation.sampling, field.name) == null) o[index] = .profile;
    }
    return r;
}

pub const InitResult = enum { created, exists };

/// What `config init` writes: the defaults plus every catalogue model as
/// a registry entry, so the file shows the entry shape with the
/// catalogue's facts rather than an empty map (the entries are optional:
/// a catalogue name resolves without one).
pub fn initial() Config {
    return .{ .models = .{ .entries = &initial_models } };
}

const initial_models = blk: {
    var list: [catalog.entries.len]NamedModel = undefined;
    for (&catalog.entries, 0..) |*entry, i| list[i] = .{ .name = entry.name, .entry = registryEntry(entry) };
    break :blk list;
};

/// A catalogue entry as a registry entry: repository, file, pinned commit,
/// companion file names, and the entry's speculative verdict (the measured
/// default of ENGN-17); every other override left to the global sections.
pub fn registryEntry(entry: *const catalog.Entry) ModelEntry {
    return .{
        .repo = entry.repo,
        .file = entry.file,
        .revision = entry.revision,
        .mmproj = if (entry.companion(.mmproj)) |c| c.file else null,
        .mtp = if (entry.companion(.mtp)) |c| c.file else null,
        .generation = .{ .speculative = entry.speculative, .draft_length = entry.draft_length },
    };
}

/// Writes `initial()` to `path` (creating its directories) unless a file
/// is already there, which is left untouched.
pub fn init(io: std.Io, dir: std.Io.Dir, path: []const u8) !InitResult {
    if (std.fs.path.dirname(path)) |parent| try dir.createDirPath(io, parent);
    const file = dir.createFile(io, path, .{ .exclusive = true }) catch |err| switch (err) {
        error.PathAlreadyExists => return .exists,
        else => return err,
    };
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writeInitial(&writer.interface);
    try writer.interface.flush();
    return .created;
}

pub fn writeInitial(out: *std.Io.Writer) !void {
    try std.json.Stringify.value(initial(), .{ .whitespace = .indent_2 }, out);
    try out.writeByte('\n');
}

// ----- editing the file: `config set`, `model pull --register` -----

/// Whether the dotted `path` names a key of `T` (a leaf, never a section or
/// the registry), resolved at run time against the compile-time schema.
fn isKey(comptime T: type, path: []const u8) bool {
    const dot = std.mem.indexOfScalar(u8, path, '.');
    const head = if (dot) |d| path[0..d] else path;
    inline for (@typeInfo(T).@"struct".fields) |field| {
        if (std.mem.eql(u8, field.name, head)) {
            if (field.type == Models) return false;
            if (comptime isSection(field.type)) return if (dot) |d| isKey(field.type, path[d + 1 ..]) else false;
            return dot == null;
        }
    }
    return false;
}

/// The file's JSON tree, edited in place and written back as it was found
/// (stated keys only, the file's own order) so an edit changes one thing.
/// A missing file starts from what `init` writes. Everything lives in the
/// arena; `text` is the result after `finish` validated it.
const Document = struct {
    arena: std.heap.ArenaAllocator,
    root: std.json.Value,

    fn open(gpa: Allocator, current: ?[]const u8, path: []const u8, diag: *Diagnostic) !Document {
        var doc: Document = .{ .arena = .init(gpa), .root = .null };
        errdefer doc.arena.deinit();
        const arena = doc.arena.allocator();
        const text = current orelse blk: {
            var w: std.Io.Writer.Allocating = .init(arena);
            try writeInitial(&w.writer);
            break :blk w.written();
        };
        doc.root = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch |err| {
            diag.set("{s}: not valid JSON ({s})", .{ path, @errorName(err) });
            return error.InvalidConfigJson;
        };
        if (doc.root != .object) {
            diag.set("{s}: the document must be a JSON object", .{path});
            return error.InvalidConfigValue;
        }
        return doc;
    }

    fn deinit(self: *Document) void {
        self.arena.deinit();
    }

    /// The object at `keys`, creating the missing levels (the caller has
    /// checked they are schema sections). A level that is not an object
    /// is reported by name.
    fn objectAt(self: *Document, keys: []const []const u8, diag: *Diagnostic) !*std.json.ObjectMap {
        const arena = self.arena.allocator();
        var object = &self.root.object;
        for (keys, 0..) |key, i| {
            if (object.getPtr(key)) |child| {
                if (child.* != .object) {
                    diag.set("{s} must be an object", .{try joinKeys(arena, keys[0 .. i + 1])});
                    return error.InvalidConfigValue;
                }
                object = &child.object;
            } else {
                try object.put(arena, try arena.dupe(u8, key), .{ .object = .empty });
                object = &object.getPtr(key).?.object;
            }
        }
        return object;
    }

    /// The value text as JSON when it parses as JSON (`0.7`, `null`,
    /// `true`, `"quoted"`), else as a string: what a shell argument means.
    fn literal(self: *Document, text: []const u8) !std.json.Value {
        const arena = self.arena.allocator();
        return std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch .{ .string = try arena.dupe(u8, text) };
    }

    /// Serializes and validates through the same loader every command
    /// runs, so the text returned is one `load` accepts. Owned by `gpa`.
    fn finish(self: *Document, gpa: Allocator, path: []const u8, diag: *Diagnostic) ![]u8 {
        const text = try std.json.Stringify.valueAlloc(gpa, self.root, .{ .whitespace = .indent_2 });
        errdefer gpa.free(text);
        var loaded = try fromText(gpa, text, path, diag);
        loaded.deinit();
        const with_newline = try gpa.realloc(text, text.len + 1);
        with_newline[text.len] = '\n';
        return with_newline;
    }
};

fn joinKeys(arena: Allocator, keys: []const []const u8) ![]u8 {
    return std.mem.join(arena, ".", keys);
}

/// The result of an edit: the new document and, for the report, the
/// previous value of the key (`null` when the file did not state it).
pub const Edit = struct {
    text: []u8,
    previous: ?[]u8,

    pub fn deinit(self: *Edit, gpa: Allocator) void {
        gpa.free(self.text);
        if (self.previous) |p| gpa.free(p);
    }
};

/// `config set <key> <value>` over the file's current text (`null`: no
/// file yet). The key is a global key (`engine.model`) or a registry key
/// (`models.<name>.profile`) of an entry that exists; `schema_version`
/// and `models` itself are not settings. A value the schema refuses
/// leaves nothing changed: the caller writes `Edit.text` only on success.
pub fn set(gpa: Allocator, current: ?[]const u8, path: []const u8, key: []const u8, value: []const u8, diag: *Diagnostic) !Edit {
    var doc = try Document.open(gpa, current, path, diag);
    defer doc.deinit();
    const arena = doc.arena.allocator();
    var parts: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, key, '.');
    while (it.next()) |part| try parts.append(arena, part);
    const keys = parts.items;
    if (std.mem.eql(u8, key, "schema_version")) {
        diag.set("schema_version is the file's format, not a setting", .{});
        return error.UnknownConfigKey;
    }
    const entry_key = keys.len >= 1 and std.mem.eql(u8, keys[0], "models");
    if (entry_key) {
        if (keys.len < 3) {
            diag.set("{s}: a registry key is models.<name>.<key> (`nuclis model pull --register <name>` creates an entry)", .{key});
            return error.UnknownConfigKey;
        }
        const rest = key[keys[0].len + keys[1].len + 2 ..];
        if (!isKey(ModelEntry, rest)) {
            diag.set("unknown key {s}", .{key});
            return error.UnknownConfigKey;
        }
        const models = try doc.objectAt(&.{"models"}, diag);
        if (models.get(keys[1]) == null) {
            diag.set("no registry entry named {s} (`nuclis model pull <owner/repo> --file <name> --register {s}` creates one)", .{ keys[1], keys[1] });
            return error.NoSuchEntry;
        }
    } else if (!isKey(Config, key)) {
        diag.set("unknown key {s}", .{key});
        return error.UnknownConfigKey;
    }
    const parent = try doc.objectAt(keys[0 .. keys.len - 1], diag);
    const leaf = keys[keys.len - 1];
    const previous: ?[]u8 = if (parent.get(leaf)) |old| try std.json.Stringify.valueAlloc(gpa, old, .{}) else null;
    errdefer if (previous) |p| gpa.free(p);
    try parent.put(arena, try arena.dupe(u8, leaf), try doc.literal(value));
    return .{ .text = try doc.finish(gpa, path, diag), .previous = previous };
}

/// What `model pull --register` records: the pulled repository at its
/// resolved commit, the main file or a companion, and an optional forced
/// profile.
pub const Registration = struct {
    repo: []const u8,
    revision: []const u8,
    /// The main file's Hub name, or null when only companions were pulled.
    file: ?[]const u8,
    /// Companion Hub names, filling the entry's keys of the same name.
    mmproj: ?[]const u8 = null,
    mtp: ?[]const u8 = null,
    profile: ?Profile = null,
};

/// Writes `name` into the registry of the file's current text (`null`: no
/// file yet): a new entry, or the same repository's entry gaining a
/// companion or a profile. A name that locates other content (another
/// repository, another main file, or a `path`) is refused.
pub fn register(gpa: Allocator, current: ?[]const u8, path: []const u8, name: []const u8, reg: Registration, diag: *Diagnostic) ![]u8 {
    try registrable(name, reg.repo, reg.file, diag);
    var doc = try Document.open(gpa, current, path, diag);
    defer doc.deinit();
    const arena = doc.arena.allocator();
    const entry = try doc.objectAt(&.{ "models", name }, diag);
    if (entry.get("path")) |p| if (p != .null) {
        diag.set("{s} names a local path; choose another name", .{name});
        return error.RegistryConflict;
    };
    if (entry.get("repo")) |r| if (r != .string or !std.mem.eql(u8, r.string, reg.repo)) {
        diag.set("{s} already names another repository; choose another name", .{name});
        return error.RegistryConflict;
    };
    if (reg.file) |file| if (entry.get("file")) |f| if (f != .string or !std.mem.eql(u8, f.string, file)) {
        diag.set("{s} already names another file of {s}; choose another name", .{ name, reg.repo });
        return error.RegistryConflict;
    };
    try entry.put(arena, "repo", .{ .string = try arena.dupe(u8, reg.repo) });
    if (reg.file) |file| try entry.put(arena, "file", .{ .string = try arena.dupe(u8, file) });
    try entry.put(arena, "revision", .{ .string = try arena.dupe(u8, reg.revision) });
    if (reg.mmproj) |file| try entry.put(arena, "mmproj", .{ .string = try arena.dupe(u8, file) });
    if (reg.mtp) |file| try entry.put(arena, "mtp", .{ .string = try arena.dupe(u8, file) });
    if (reg.profile) |profile| try entry.put(arena, "profile", .{ .string = @tagName(profile) });
    return doc.finish(gpa, path, diag);
}

/// Whether `name` may register `repo`/`file`: a catalogue name resolves
/// through the registry first, so an entry under it may only say what the
/// catalogue says. Checked before a transfer as well, so a wrong name
/// costs no download.
pub fn registrable(name: []const u8, repo: []const u8, file: ?[]const u8, diag: *Diagnostic) !void {
    if (catalog.find(name)) |c| if (!std.mem.eql(u8, c.repo, repo) or (file != null and !std.mem.eql(u8, c.file, file.?))) {
        diag.set("{s} is a catalogue name pinned to {s}/{s}; choose another name", .{ name, c.repo, c.file });
        return error.RegistryConflict;
    };
}

/// Writes an edited document over the file, creating its directories.
pub fn write(io: std.Io, dir: std.Io.Dir, path: []const u8, text: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| try dir.createDirPath(io, parent);
    try dir.writeFile(io, .{ .sub_path = path, .data = text });
}

/// The file's text for an edit, or null when there is none yet.
pub fn readText(gpa: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8, diag: *Diagnostic) !?[]u8 {
    return dir.readFileAlloc(io, path, gpa, .limited(max_file_bytes + 1)) catch |err| switch (err) {
        error.FileNotFound => null,
        error.StreamTooLong => {
            diag.set("{s}: larger than {d} bytes", .{ path, max_file_bytes });
            return error.ConfigTooLarge;
        },
        else => return err,
    };
}

/// Visits every key of `T` in walk order with its dotted path and value,
/// skipping the registry. `visitor` is a pointer to a struct with
/// `fn leaf(self, comptime path, value) !void`; the path is comptime so a
/// visitor can index `Origin` through `leafIndex`.
fn walk(comptime T: type, value: T, comptime prefix: []const u8, visitor: anytype) !void {
    inline for (@typeInfo(T).@"struct".fields) |field| {
        const path = prefix ++ field.name;
        if (field.type == Models) {
            // No key slots; printed separately by name.
        } else if (comptime isSection(field.type)) {
            try walk(field.type, @field(value, field.name), path ++ ".", visitor);
        } else {
            try visitor.leaf(path, @field(value, field.name));
        }
    }
}

fn formatLeaf(buf: []u8, value: anytype) []const u8 {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .optional => if (value) |v| formatLeaf(buf, v) else "null",
        .bool => if (value) "true" else "false",
        .int, .float => std.fmt.bufPrint(buf, "{d}", .{value}) catch "?",
        .@"enum" => @tagName(value),
        .pointer => value,
        else => @compileError("unsupported config key type " ++ @typeName(T)),
    };
}

/// The effective view `config show` prints: what `generate` (and, for the
/// `agent` keys, the agent) would run with from the file alone, no flags.
/// Its sections mirror `Config`'s so every leaf has an `Origin` slot.
pub const Effective = struct {
    /// How `engine.model` resolves: a registry entry, a catalogue name, or
    /// a path.
    model_kind: enum { registry, catalogue, path },
    profile: Profile,
    engine: struct { model: []const u8, backend: Backend, ctx_size: usize, kv_precision: KvPrecision },
    generation: struct { max_tokens: usize, think: Effort, speculative: bool, draft_length: usize, sampling: inference.sampling.Options },
    agent: struct { think: Effort, fold_thinking: bool, theme: ThemeName },
    origin: Origin,

    pub fn from(loaded: *const Loaded) Effective {
        const gen = resolve(loaded, null, .{}, .generate);
        const agent = resolve(loaded, null, .{}, .agent);
        var origin = gen.origin;
        const agent_think = comptime leafIndex(Config, "agent.think");
        origin[agent_think] = agent.origin[agent_think];
        return .{
            .model_kind = if (gen.entry != null) .registry else if (catalog.find(gen.model) != null) .catalogue else .path,
            .profile = gen.profile,
            .engine = .{ .model = gen.model, .backend = gen.backend, .ctx_size = gen.ctx_size, .kv_precision = gen.kv_precision },
            .generation = .{ .max_tokens = gen.max_tokens, .think = gen.think, .speculative = gen.speculative, .draft_length = gen.draft_length, .sampling = gen.samplingOptions() },
            .agent = .{ .think = agent.think, .fold_thinking = agent.fold_thinking, .theme = agent.theme },
            .origin = origin,
        };
    }
};

/// The color of a value is the layer it came from.
fn sourceStyle(source: Source) style.Kind {
    return switch (source) {
        .default => .dim,
        .profile => .number,
        .file => .success,
        .model => .warning,
        .flag => .user,
    };
}

/// Prints the effective value of every key with its source, then each
/// registry entry's stated keys. The JSON form carries the file as loaded
/// (`config`, with the registry as a map), the effective view, and a flat
/// `sources` map keyed by dotted path.
pub fn show(loaded: *const Loaded, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
    const view = Effective.from(loaded);
    if (json) {
        var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
        try s.beginObject();
        try s.objectField("path");
        try s.write(if (loaded.path.len == 0) null else loaded.path);
        try s.objectField("found");
        try s.write(loaded.found);
        try s.objectField("config");
        try s.write(loaded.config);
        try s.objectField("effective");
        try s.beginObject();
        try s.objectField("model_kind");
        try s.write(view.model_kind);
        try s.objectField("profile");
        try s.write(view.profile);
        try s.objectField("engine");
        try s.write(view.engine);
        try s.objectField("generation");
        try s.write(view.generation);
        try s.objectField("agent");
        try s.write(view.agent);
        try s.endObject();
        try s.objectField("sources");
        try s.beginObject();
        var visitor: struct {
            s: *std.json.Stringify,
            origin: *const Origin,
            fn leaf(self: *@This(), comptime path: []const u8, _: anytype) !void {
                try self.s.objectField(path);
                try self.s.write(self.origin[comptime leafIndex(Config, path)]);
            }
        } = .{ .s = &s, .origin = &view.origin };
        try walk(Config, loaded.config, "", &visitor);
        try s.endObject();
        try s.endObject();
        try out.writeByte('\n');
        return;
    }
    if (loaded.path.len == 0)
        try out.print("{s}config:{s} no user root (HOME and NUCLIS_HOME unset); built-in defaults\n", .{ sty.on(.label), sty.off() })
    else
        try out.print("{s}config:{s} {s}{s}{s} {s}({s}){s}\n", .{ sty.on(.label), sty.off(), sty.on(.code), loaded.path, sty.off(), sty.on(.dim), if (loaded.found) "file" else "not found; built-in defaults", sty.off() });
    try out.print("{s}model:{s} {s}{s}{s} ({s}; {s}{s}{s} profile). {s}Values are effective: a null sampling key in the file takes the profile's value for generation.think = {s} (source \"profile\").{s}\n", .{
        sty.on(.label),
        sty.off(),
        sty.on(.keyword),
        view.engine.model,
        sty.off(),
        switch (view.model_kind) {
            .registry => "registry entry",
            .catalogue => "catalogue name",
            .path => "path",
        },
        sty.on(.number),
        @tagName(view.profile),
        sty.off(),
        sty.on(.dim),
        @tagName(view.generation.think),
        sty.off(),
    });
    const Rows = struct {
        out: *std.Io.Writer,
        origin: *const Origin,
        sty: style.Style,
        fn leaf(self: *@This(), comptime path: []const u8, value: anytype) !void {
            var buf: [64]u8 = undefined;
            const source = self.origin[comptime leafIndex(Config, path)];
            const paint = self.sty.on(sourceStyle(source));
            try self.out.print("{s: <36} {s}{s: <40}{s} {s}{s}{s}\n", .{ path, paint, formatLeaf(&buf, value), self.sty.off(), paint, @tagName(source), self.sty.off() });
        }
    };
    var rows: Rows = .{ .out = out, .origin = &view.origin, .sty = sty };
    try rows.leaf("schema_version", loaded.config.schema_version);
    try walk(@TypeOf(view.engine), view.engine, "engine.", &rows);
    try walk(@TypeOf(view.generation), view.generation, "generation.", &rows);
    try walk(@TypeOf(view.agent), view.agent, "agent.", &rows);
    for (loaded.config.models.entries) |named| {
        var entry_rows: struct {
            out: *std.Io.Writer,
            name: []const u8,
            sty: style.Style,
            fn leaf(self: *@This(), comptime path: []const u8, value: anytype) !void {
                // Only what the entry states; a null inherits the global value.
                if (@typeInfo(@TypeOf(value)) == .optional and value == null) return;
                var buf: [64]u8 = undefined;
                var key: [max_path_bytes]u8 = undefined;
                const full = std.fmt.bufPrint(&key, "models.{s}.{s}", .{ self.name, path }) catch unreachable;
                const paint = self.sty.on(sourceStyle(.file));
                try self.out.print("{s: <36} {s}{s: <40}{s} {s}{s}{s}\n", .{ full, paint, formatLeaf(&buf, value), self.sty.off(), paint, "file", self.sty.off() });
            }
        } = .{ .out = out, .name = named.name, .sty = sty };
        try walk(ModelEntry, named.entry, "", &entry_rows);
    }
}

test "defaults round-trip through init and load, and a missing file never writes" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var diag: Diagnostic = .{};
    var missing = try load(alloc, io, tmp.dir, "sub/nuclis.json", &diag);
    defer missing.deinit();
    try std.testing.expect(!missing.found);
    try std.testing.expectEqualStrings("sub/nuclis.json", missing.path);
    try std.testing.expectEqualDeep(Config{}, missing.config);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "sub", .{}));
    try std.testing.expectEqual(.created, try init(io, tmp.dir, "sub/nuclis.json"));
    try std.testing.expectEqual(.exists, try init(io, tmp.dir, "sub/nuclis.json"));
    var loaded = try load(alloc, io, tmp.dir, "sub/nuclis.json", &diag);
    defer loaded.deinit();
    try std.testing.expect(loaded.found);
    // The written file is the defaults plus the default model as an entry.
    try std.testing.expectEqualDeep(initial(), loaded.config);
    try std.testing.expectEqualStrings("unsloth/Qwen3.8-27B-GGUF", loaded.config.models.find("qwen3.8-27b").?.repo.?);
    try std.testing.expectEqualStrings("mmproj-BF16.gguf", loaded.config.models.find("qwen3.8-27b").?.mmproj.?);
    try std.testing.expect(loaded.config.models.find("qwen3.8-27b").?.ctx_size == null);
    // Every key was present in the written file, so every source is `file`.
    for (loaded.origin) |source| try std.testing.expectEqual(.file, source);
    var no_root = try load(alloc, io, tmp.dir, null, &diag);
    defer no_root.deinit();
    try std.testing.expect(!no_root.found);
    try std.testing.expectEqual(@as(usize, 0), no_root.path.len);
}

test "file values override defaults per key and the source is recorded" {
    const alloc = std.testing.allocator;
    var diag: Diagnostic = .{};
    var loaded = try fromText(alloc,
        \\{ "schema_version": 1,
        \\  "engine": { "model": "/abs/other.gguf", "ctx_size": 4096 },
        \\  "generation": { "sampling": { "temperature": 0, "top_k": 40 } },
        \\  "agent": { "fold_thinking": false, "theme": "gruvbox-dark" } }
    , "t.json", &diag);
    defer loaded.deinit();
    try std.testing.expectEqualStrings("/abs/other.gguf", loaded.config.engine.model);
    try std.testing.expectEqual(@as(usize, 4096), loaded.config.engine.ctx_size);
    try std.testing.expectEqual((Config.Engine{}).backend, loaded.config.engine.backend);
    try std.testing.expectEqual(@as(f32, 0), loaded.config.generation.sampling.temperature.?);
    try std.testing.expectEqual(@as(usize, 40), loaded.config.generation.sampling.top_k.?);
    try std.testing.expect(loaded.config.generation.sampling.top_p == null);
    try std.testing.expectEqual(@as(usize, 2048), loaded.config.generation.max_tokens);
    try std.testing.expect(!loaded.config.agent.fold_thinking);
    try std.testing.expectEqual(@as(usize, 0), loaded.config.models.entries.len);
    try std.testing.expectEqual(.file, loaded.source("engine.model"));
    try std.testing.expectEqual(.default, loaded.source("engine.backend"));
    try std.testing.expectEqual(.file, loaded.source("generation.sampling.top_k"));
    try std.testing.expectEqual(.default, loaded.source("generation.sampling.top_p"));
    try std.testing.expectEqual(.default, loaded.source("agent.think"));
    try std.testing.expectEqual(.file, loaded.source("agent.fold_thinking"));
    // The palette is a global display key: stated in the file, it is the
    // agent's, and no registry entry can override it.
    try std.testing.expectEqual(ThemeName.@"gruvbox-dark", loaded.config.agent.theme);
    try std.testing.expectEqual(.file, loaded.source("agent.theme"));
    try std.testing.expect(!@hasField(ModelEntry.Agent, "theme"));
}

test "registry entries parse by name with their overrides and companions" {
    const alloc = std.testing.allocator;
    var diag: Diagnostic = .{};
    var loaded = try fromText(alloc,
        \\{ "schema_version": 1,
        \\  "models": {
        \\    "gemma": { "repo": "unsloth/gemma-4-12b-it-GGUF", "file": "gemma-4-12b-it-UD-Q4_K_XL.gguf",
        \\               "revision": "fc034cfff751157913579611efad8462ac1be606", "mmproj": "mmproj-F16.gguf",
        \\               "ctx_size": 4096, "generation": { "think": "off", "sampling": { "top_k": 64 } },
        \\               "agent": { "fold_thinking": false } },
        \\    "local": { "path": "/scratch/x.gguf", "profile": "gemma4" } } }
    , "t.json", &diag);
    defer loaded.deinit();
    try std.testing.expectEqual(@as(usize, 2), loaded.config.models.entries.len);
    const gemma = loaded.config.models.find("gemma").?;
    try std.testing.expectEqualStrings("unsloth/gemma-4-12b-it-GGUF", gemma.repo.?);
    try std.testing.expectEqualStrings("gemma-4-12b-it-UD-Q4_K_XL.gguf", gemma.file.?);
    try std.testing.expectEqualStrings("fc034cfff751157913579611efad8462ac1be606", gemma.revision.?);
    try std.testing.expectEqualStrings("mmproj-F16.gguf", gemma.mmproj.?);
    try std.testing.expect(gemma.mtp == null and gemma.path == null);
    try std.testing.expectEqual(@as(usize, 4096), gemma.ctx_size.?);
    try std.testing.expectEqual(.off, gemma.generation.think.?);
    try std.testing.expectEqual(@as(usize, 64), gemma.generation.sampling.top_k.?);
    try std.testing.expect(gemma.generation.sampling.temperature == null and gemma.generation.max_tokens == null);
    try std.testing.expect(!gemma.agent.fold_thinking.? and gemma.agent.think == null);
    try std.testing.expectEqualStrings("/scratch/x.gguf", loaded.config.models.find("local").?.path.?);
    try std.testing.expectEqual(.gemma4, loaded.config.models.find("local").?.profile.?);
    try std.testing.expect(gemma.profile == null);
    // The entry's profile is forced at open and names the sampling defaults;
    // the flag wins over it.
    const local = resolve(&loaded, "local", .{}, .generate);
    try std.testing.expectEqual(.gemma4, local.forced_profile.?);
    try std.testing.expectEqual(.gemma4, local.profile);
    const flagged = resolve(&loaded, "local", .{ .prompt_profile = .qwen38 }, .agent);
    try std.testing.expectEqual(.qwen38, flagged.forced_profile.?);
    try std.testing.expectEqual(.qwen38, flagged.profile);
    try std.testing.expect(resolve(&loaded, "gemma", .{}, .generate).forced_profile == null);
    try std.testing.expect(loaded.config.models.find("qwen3.8-27b") == null);
    // The registry has no origin slots; the global keys are untouched
    // (the version is a key of the file like any other).
    try std.testing.expectEqual(.file, loaded.source("schema_version"));
    for (loaded.origin[1..]) |source| try std.testing.expectEqual(.default, source);
}

test "unknown keys, wrong types, bad ranges, and wrong versions name the key" {
    const alloc = std.testing.allocator;
    const cases = [_]struct { text: []const u8, err: anyerror, needle: []const u8 }{
        .{ .text = "{ \"schema_version\": 1, \"engine\": { \"modle\": \"x\" } }", .err = error.UnknownConfigKey, .needle = "unknown key engine.modle" },
        .{ .text = "{ \"schema_version\": 1, \"generation\": { \"sampling\": { \"top_q\": 1 } } }", .err = error.UnknownConfigKey, .needle = "generation.sampling.top_q" },
        .{ .text = "{ \"schema_version\": 1, \"tui\": {} }", .err = error.UnknownConfigKey, .needle = "unknown key tui" },
        .{ .text = "{ \"schema_version\": 1, \"engine\": { \"backend\": \"cuda\" } }", .err = error.InvalidConfigValue, .needle = "engine.backend must be one of cpu|metal" },
        .{ .text = "{ \"schema_version\": 1, \"engine\": { \"kv_precision\": \"f64\" } }", .err = error.InvalidConfigValue, .needle = "engine.kv_precision must be one of f32|f16" },
        .{ .text = "{ \"schema_version\": 1, \"engine\": { \"ctx_size\": -1 } }", .err = error.InvalidConfigValue, .needle = "engine.ctx_size must be a non-negative integer" },
        .{ .text = "{ \"schema_version\": 1, \"engine\": { \"ctx_size\": 65536 } }", .err = error.InvalidConfigValue, .needle = "engine.ctx_size must be 1..32768" },
        .{ .text = "{ \"schema_version\": 1, \"engine\": { \"model\": \"\" } }", .err = error.InvalidConfigValue, .needle = "engine.model must not be empty" },
        .{ .text = "{ \"schema_version\": 1, \"engine\": \"metal\" }", .err = error.InvalidConfigValue, .needle = "engine must be an object" },
        .{ .text = "{ \"schema_version\": 1, \"generation\": { \"max_tokens\": 0 } }", .err = error.InvalidConfigValue, .needle = "generation.max_tokens must be 1..4096" },
        .{ .text = "{ \"schema_version\": 1, \"generation\": { \"think\": \"loud\" } }", .err = error.InvalidConfigValue, .needle = "generation.think must be one of off|low|medium|high|xhigh" },
        .{ .text = "{ \"schema_version\": 1, \"generation\": { \"sampling\": { \"top_p\": 0 } } }", .err = error.InvalidConfigValue, .needle = "generation.sampling.top_p is out of range" },
        .{ .text = "{ \"schema_version\": 1, \"generation\": { \"sampling\": { \"temperature\": \"hot\" } } }", .err = error.InvalidConfigValue, .needle = "generation.sampling.temperature must be null or a number" },
        .{ .text = "{ \"schema_version\": 1, \"agent\": { \"fold_thinking\": 1 } }", .err = error.InvalidConfigValue, .needle = "agent.fold_thinking must be true or false" },
        .{ .text = "{ \"schema_version\": 1, \"chat\": { \"think\": \"low\" } }", .err = error.UnknownConfigKey, .needle = "unknown key chat: the section is now agent" },
        .{ .text = "{ \"schema_version\": 1, \"agent\": { \"theme\": \"solarized\" } }", .err = error.InvalidConfigValue, .needle = "agent.theme must be one of gruvbox-dark" },
        // The registry: names, shape, unknown companion keys, ranges.
        .{ .text = "{ \"schema_version\": 1, \"models\": [] }", .err = error.InvalidConfigValue, .needle = "models must be an object" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": \"x.gguf\" } }", .err = error.InvalidConfigValue, .needle = "models.g must be an object" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"a/b\": { \"path\": \"x\" } } }", .err = error.InvalidConfigValue, .needle = "entry name \"a/b\"" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"x.gguf\": { \"path\": \"x\" } } }", .err = error.InvalidConfigValue, .needle = "entry name \"x.gguf\"" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"\": { \"path\": \"x\" } } }", .err = error.InvalidConfigValue, .needle = "entry name \"\"" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"x\", \"imatrix\": \"i.gguf\" } } }", .err = error.UnknownConfigKey, .needle = "unknown key models.g.imatrix" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"x\", \"generation\": { \"sampling\": { \"top_q\": 1 } } } } }", .err = error.UnknownConfigKey, .needle = "unknown key models.g.generation.sampling.top_q" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"x\", \"ctx_size\": \"big\" } } }", .err = error.InvalidConfigValue, .needle = "models.g.ctx_size must be null or a non-negative integer" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"x\", \"ctx_size\": 0 } } }", .err = error.InvalidConfigValue, .needle = "models.g.ctx_size must be 1..32768" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"x\", \"generation\": { \"max_tokens\": 5000 } } } }", .err = error.InvalidConfigValue, .needle = "models.g.generation.max_tokens must be 1..4096" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"x\", \"generation\": { \"sampling\": { \"top_p\": 2 } } } } }", .err = error.InvalidConfigValue, .needle = "models.g.generation.sampling.top_p is out of range" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"x\", \"agent\": { \"think\": \"loud\" } } } }", .err = error.InvalidConfigValue, .needle = "models.g.agent.think must be null or one of off|low|medium|high|xhigh" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": {} } }", .err = error.InvalidConfigValue, .needle = "models.g: needs path, or repo and file" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"repo\": \"a/b\" } } }", .err = error.InvalidConfigValue, .needle = "models.g: needs path, or repo and file" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"x\", \"repo\": \"a/b\" } } }", .err = error.InvalidConfigValue, .needle = "models.g: path excludes repo, file, and revision" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"repo\": \"nope\", \"file\": \"x.gguf\" } } }", .err = error.InvalidConfigValue, .needle = "models.g.repo must be owner/repo" },
        .{ .text = "{ \"schema_version\": 1, \"models\": { \"g\": { \"path\": \"\" } } }", .err = error.InvalidConfigValue, .needle = "models.g.path must not be empty" },
        .{ .text = "{ \"engine\": {} }", .err = error.UnsupportedConfigVersion, .needle = "schema_version is required" },
        .{ .text = "{ \"schema_version\": 2 }", .err = error.UnsupportedConfigVersion, .needle = "schema_version 2 is not supported" },
        .{ .text = "{ \"schema_version\": \"1\" }", .err = error.UnsupportedConfigVersion, .needle = "must be the integer 1" },
        .{ .text = "[1]", .err = error.InvalidConfigValue, .needle = "must be a JSON object" },
        .{ .text = "{ nope", .err = error.InvalidConfigJson, .needle = "not valid JSON" },
    };
    for (cases) |case| {
        var diag: Diagnostic = .{};
        try std.testing.expectError(case.err, fromText(alloc, case.text, "t.json", &diag));
        if (std.mem.indexOf(u8, diag.message(), case.needle) == null) {
            std.debug.print("expected \"{s}\" in \"{s}\"\n", .{ case.needle, diag.message() });
            return error.TestUnexpectedResult;
        }
    }
    var diag: Diagnostic = .{};
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const big = try alloc.alloc(u8, max_file_bytes + 1);
    defer alloc.free(big);
    @memset(big, ' ');
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "big.json", .data = big });
    try std.testing.expectError(error.ConfigTooLarge, load(alloc, std.testing.io, tmp.dir, "big.json", &diag));
}

test "resolve applies defaults < file < flags per command and records the source" {
    const alloc = std.testing.allocator;
    var diag: Diagnostic = .{};
    var loaded = try fromText(alloc,
        \\{ "schema_version": 1, "engine": { "backend": "cpu", "ctx_size": 4096 },
        \\  "generation": { "max_tokens": 64, "think": "medium", "sampling": { "temperature": 0.5, "top_k": 40 } },
        \\  "agent": { "think": "xhigh", "fold_thinking": false } }
    , "t.json", &diag);
    defer loaded.deinit();
    const gen = resolve(&loaded, null, .{ .ctx_size = 1024, .sampling = .{ .top_k = 10 } }, .generate);
    try std.testing.expectEqualStrings("qwen3.8-27b", gen.model);
    try std.testing.expect(gen.entry == null);
    try std.testing.expectEqual(.default, gen.source("engine.model"));
    try std.testing.expectEqual(.cpu, gen.backend);
    try std.testing.expectEqual(.file, gen.source("engine.backend"));
    try std.testing.expectEqual(@as(usize, 1024), gen.ctx_size);
    try std.testing.expectEqual(.flag, gen.source("engine.ctx_size"));
    try std.testing.expectEqual(@as(usize, 64), gen.max_tokens);
    try std.testing.expectEqual(.medium, gen.think);
    try std.testing.expectEqual(Overrides{ .temperature = 0.5, .top_k = 10 }, gen.sampling);
    try std.testing.expectEqual(.file, gen.source("generation.sampling.temperature"));
    try std.testing.expectEqual(.flag, gen.source("generation.sampling.top_k"));
    // An unset override takes the profile's value, and says so.
    try std.testing.expectEqual(.profile, gen.source("generation.sampling.min_p"));
    try std.testing.expectEqual(.profile, gen.source("generation.sampling.top_p"));
    try std.testing.expectEqual(inference.sampling.Options{ .temperature = 0.5, .top_k = 10, .top_p = 0.95 }, gen.samplingOptions());
    try std.testing.expectEqualStrings("t.json", gen.config_file.?);
    const agent = resolve(&loaded, "/abs/m.gguf", .{ .think = .low }, .agent);
    try std.testing.expectEqualStrings("/abs/m.gguf", agent.model);
    try std.testing.expectEqual(.flag, agent.source("engine.model"));
    try std.testing.expectEqual(.low, agent.think);
    try std.testing.expectEqual(.flag, agent.source("agent.think"));
    try std.testing.expectEqual(.file, agent.source("generation.think"));
    try std.testing.expect(!agent.fold_thinking);
    try std.testing.expectEqual(@as(usize, 64), agent.max_tokens);
    // bench: engine from the file; budget and sampling from flags only.
    const bench = resolve(&loaded, null, .{}, .bench);
    try std.testing.expectEqual(.cpu, bench.backend);
    try std.testing.expectEqual(@as(usize, 4096), bench.ctx_size);
    try std.testing.expectEqual(@as(usize, bench_max_tokens), bench.max_tokens);
    try std.testing.expectEqual(.default, bench.source("generation.max_tokens"));
    try std.testing.expectEqual(Overrides{}, bench.sampling);
    try std.testing.expectEqual(.default, bench.source("generation.sampling.temperature"));
    try std.testing.expectEqual(.default, bench.source("generation.sampling.min_p"));
    try std.testing.expectEqual(.off, bench.think);
    const flagged = resolve(&loaded, null, .{ .backend = .metal, .max_tokens = 8, .sampling = .{ .min_p = 0.1 } }, .bench);
    try std.testing.expectEqual(.metal, flagged.backend);
    try std.testing.expectEqual(@as(usize, 8), flagged.max_tokens);
    try std.testing.expectEqual(.flag, flagged.source("generation.max_tokens"));
    try std.testing.expectEqual(Overrides{ .min_p = 0.1 }, flagged.sampling);
    // No file: defaults unless flagged, no file to record, and the
    // profile as the source of every sampling option.
    var none: Loaded = .{ .arena = .init(alloc) };
    defer none.deinit();
    const plain = resolve(&none, null, .{}, .generate);
    try std.testing.expectEqual((Config.Engine{}).backend, plain.backend);
    try std.testing.expectEqual(@as(usize, 16384), plain.ctx_size);
    try std.testing.expectEqual(@as(usize, 2048), plain.max_tokens);
    try std.testing.expectEqual(.off, plain.think);
    try std.testing.expectEqual(.qwen38, plain.profile);
    try std.testing.expectEqual(inference.profiles.Profile.qwen38.samplingDefaults(.off), plain.samplingOptions());
    try std.testing.expectEqual(.low, resolve(&none, null, .{}, .agent).think);
    try std.testing.expect(plain.config_file == null);
    try std.testing.expectEqual(.profile, plain.source("generation.sampling.temperature"));
    try std.testing.expectEqual(.default, plain.source("engine.ctx_size"));
}

test "a registry entry layers between the file's globals and the flags, for its model only" {
    const alloc = std.testing.allocator;
    var diag: Diagnostic = .{};
    var loaded = try fromText(alloc,
        \\{ "schema_version": 1, "engine": { "model": "gemma", "ctx_size": 8192 },
        \\  "generation": { "max_tokens": 64, "sampling": { "temperature": 0.5, "top_k": 40 } },
        \\  "agent": { "think": "xhigh" },
        \\  "models": { "gemma": { "repo": "unsloth/gemma-4-12b-it-GGUF", "file": "g.gguf", "ctx_size": 4096,
        \\                         "generation": { "max_tokens": 32, "think": "low", "sampling": { "top_k": 64, "min_p": 0.05 } },
        \\                         "agent": { "think": "off", "fold_thinking": false } } } }
    , "t.json", &diag);
    defer loaded.deinit();
    const gen = resolve(&loaded, null, .{ .sampling = .{ .min_p = 0.1 } }, .generate);
    try std.testing.expectEqualStrings("gemma", gen.model);
    try std.testing.expectEqualStrings("g.gguf", gen.entry.?.file.?);
    try std.testing.expectEqual(@as(usize, 4096), gen.ctx_size);
    try std.testing.expectEqual(.model, gen.source("engine.ctx_size"));
    try std.testing.expectEqual(@as(usize, 32), gen.max_tokens);
    try std.testing.expectEqual(.model, gen.source("generation.max_tokens"));
    try std.testing.expectEqual(.low, gen.think);
    try std.testing.expectEqual(.model, gen.source("generation.think"));
    try std.testing.expectEqual(Overrides{ .temperature = 0.5, .top_k = 64, .min_p = 0.1 }, gen.sampling);
    try std.testing.expectEqual(.file, gen.source("generation.sampling.temperature"));
    try std.testing.expectEqual(.model, gen.source("generation.sampling.top_k"));
    try std.testing.expectEqual(.flag, gen.source("generation.sampling.min_p"));
    try std.testing.expectEqual(.profile, gen.source("generation.sampling.top_p"));
    try std.testing.expect(!gen.fold_thinking);
    try std.testing.expectEqual(.model, gen.source("agent.fold_thinking"));
    const agent = resolve(&loaded, null, .{}, .agent);
    try std.testing.expectEqual(.off, agent.think);
    try std.testing.expectEqual(.model, agent.source("agent.think"));
    // Another model: the entry does not apply, the globals do.
    const other = resolve(&loaded, "qwen3.8-27b", .{}, .generate);
    try std.testing.expect(other.entry == null);
    try std.testing.expectEqual(@as(usize, 8192), other.ctx_size);
    try std.testing.expectEqual(.file, other.source("engine.ctx_size"));
    try std.testing.expectEqual(@as(usize, 64), other.max_tokens);
    try std.testing.expectEqual(.off, other.think);
    try std.testing.expectEqual(Overrides{ .temperature = 0.5, .top_k = 40 }, other.sampling);
    try std.testing.expect(other.fold_thinking);
    try std.testing.expectEqual(.xhigh, resolve(&loaded, "qwen3.8-27b", .{}, .agent).think);
    // bench takes the entry's context, nothing else from it.
    const bench = resolve(&loaded, null, .{}, .bench);
    try std.testing.expectEqual(@as(usize, 4096), bench.ctx_size);
    try std.testing.expectEqual(@as(usize, bench_max_tokens), bench.max_tokens);
    try std.testing.expectEqual(Overrides{}, bench.sampling);
    try std.testing.expectEqual(.off, bench.think);
}

/// Finds the `show` row for `key` and checks its value and source columns.
fn expectRow(text: []const u8, key: []const u8, value: []const u8, source: []const u8) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, key) or line.len <= key.len or line[key.len] != ' ') continue;
        var columns = std.mem.tokenizeScalar(u8, line[key.len..], ' ');
        try std.testing.expectEqualStrings(value, columns.next() orelse "");
        try std.testing.expectEqualStrings(source, columns.next() orelse "");
        try std.testing.expect(columns.next() == null);
        return;
    }
    std.debug.print("no row for {s} in:\n{s}", .{ key, text });
    return error.TestUnexpectedResult;
}

test "set edits one key of the file's own text, validates it, and reports the previous value" {
    const alloc = std.testing.allocator;
    var diag: Diagnostic = .{};
    // No file yet: the edit starts from what `init` writes.
    var first = try set(alloc, null, "t.json", "engine.model", "gemma-4-12b", &diag);
    defer first.deinit(alloc);
    try std.testing.expect(first.previous != null);
    try std.testing.expectEqualStrings("\"qwen3.8-27b\"", first.previous.?);
    try std.testing.expect(std.mem.indexOf(u8, first.text, "\"model\": \"gemma-4-12b\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, first.text, "\"qwen3.8-27b\": {") != null);
    // A value is JSON when it parses as JSON, a string otherwise; a stated
    // key is replaced in place and an unstated section is created.
    const text =
        \\{ "schema_version": 1, "engine": { "ctx_size": 4096 },
        \\  "models": { "local": { "path": "/scratch/x.gguf" } } }
    ;
    var num = try set(alloc, text, "t.json", "generation.sampling.temperature", "0.7", &diag);
    defer num.deinit(alloc);
    try std.testing.expect(num.previous == null);
    try std.testing.expect(std.mem.indexOf(u8, num.text, "\"temperature\": 0.7") != null);
    try std.testing.expect(std.mem.indexOf(u8, num.text, "\"ctx_size\": 4096") != null);
    var cleared = try set(alloc, num.text, "t.json", "generation.sampling.temperature", "null", &diag);
    defer cleared.deinit(alloc);
    try std.testing.expectEqualStrings("0.7", cleared.previous.?);
    try std.testing.expect(std.mem.indexOf(u8, cleared.text, "\"temperature\": null") != null);
    var entry = try set(alloc, text, "t.json", "models.local.profile", "gemma4", &diag);
    defer entry.deinit(alloc);
    try std.testing.expect(std.mem.indexOf(u8, entry.text, "\"profile\": \"gemma4\"") != null);
    var think = try set(alloc, text, "t.json", "models.local.agent.think", "medium", &diag);
    defer think.deinit(alloc);
    try std.testing.expect(std.mem.indexOf(u8, think.text, "\"think\": \"medium\"") != null);
    // Refusals name the key; a rejected value produces no text at all.
    try std.testing.expectError(error.UnknownConfigKey, set(alloc, text, "t.json", "engine.speed", "1", &diag));
    try std.testing.expectError(error.UnknownConfigKey, set(alloc, text, "t.json", "engine", "1", &diag));
    try std.testing.expectError(error.UnknownConfigKey, set(alloc, text, "t.json", "schema_version", "2", &diag));
    try std.testing.expectError(error.UnknownConfigKey, set(alloc, text, "t.json", "models", "{}", &diag));
    try std.testing.expectError(error.UnknownConfigKey, set(alloc, text, "t.json", "models.local.speed", "1", &diag));
    try std.testing.expectError(error.NoSuchEntry, set(alloc, text, "t.json", "models.other.profile", "gemma4", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "--register other") != null);
    try std.testing.expectError(error.InvalidConfigValue, set(alloc, text, "t.json", "engine.ctx_size", "99999", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "engine.ctx_size") != null);
    try std.testing.expectError(error.InvalidConfigValue, set(alloc, text, "t.json", "engine.model", "null", &diag));
    try std.testing.expectError(error.InvalidConfigValue, set(alloc, text, "t.json", "agent.theme", "neon", &diag));
    try std.testing.expectError(error.InvalidConfigValue, set(alloc, text, "t.json", "engine.model", "123", &diag));
}

test "register writes a new entry, fills a companion or a profile, and refuses another content" {
    const alloc = std.testing.allocator;
    var diag: Diagnostic = .{};
    const text =
        \\{ "schema_version": 1, "models": { "local": { "path": "/scratch/x.gguf" },
        \\  "q": { "repo": "a/b", "file": "q.gguf", "revision": "0000000000000000000000000000000000000000" } } }
    ;
    const fresh = try register(alloc, text, "t.json", "hauhau", .{ .repo = "HauhauCS/Gemma4", .revision = "ae8045ac2bd216293ca49a3065da2c942dde4b68", .file = "g.gguf", .profile = .gemma4 }, &diag);
    defer alloc.free(fresh);
    var loaded = try fromText(alloc, fresh, "t.json", &diag);
    defer loaded.deinit();
    const entry = loaded.config.models.find("hauhau").?;
    try std.testing.expectEqualStrings("HauhauCS/Gemma4", entry.repo.?);
    try std.testing.expectEqualStrings("g.gguf", entry.file.?);
    try std.testing.expectEqualStrings("ae8045ac2bd216293ca49a3065da2c942dde4b68", entry.revision.?);
    try std.testing.expectEqual(.gemma4, entry.profile.?);
    try std.testing.expect(entry.mmproj == null and entry.ctx_size == null);
    try std.testing.expect(loaded.config.models.find("local") != null and loaded.config.models.find("q") != null);
    // A companion of the same repository fills the entry; the file stays.
    const with_mmproj = try register(alloc, fresh, "t.json", "hauhau", .{ .repo = "HauhauCS/Gemma4", .revision = "ae8045ac2bd216293ca49a3065da2c942dde4b68", .file = null, .mmproj = "mmproj.gguf" }, &diag);
    defer alloc.free(with_mmproj);
    var again = try fromText(alloc, with_mmproj, "t.json", &diag);
    defer again.deinit();
    try std.testing.expectEqualStrings("mmproj.gguf", again.config.models.find("hauhau").?.mmproj.?);
    try std.testing.expectEqualStrings("g.gguf", again.config.models.find("hauhau").?.file.?);
    // No file at all starts from the initial document.
    const first = try register(alloc, null, "t.json", "x", .{ .repo = "a/b", .revision = "0000000000000000000000000000000000000000", .file = "f.gguf" }, &diag);
    defer alloc.free(first);
    try std.testing.expect(std.mem.indexOf(u8, first, "\"qwen3.8-27b\": {") != null);
    // Other content under the name is a conflict, never an overwrite.
    try std.testing.expectError(error.RegistryConflict, register(alloc, text, "t.json", "local", .{ .repo = "a/b", .revision = "0000000000000000000000000000000000000000", .file = "f.gguf" }, &diag));
    try std.testing.expectError(error.RegistryConflict, register(alloc, text, "t.json", "q", .{ .repo = "c/d", .revision = "0000000000000000000000000000000000000000", .file = "q.gguf" }, &diag));
    try std.testing.expectError(error.RegistryConflict, register(alloc, text, "t.json", "q", .{ .repo = "a/b", .revision = "0000000000000000000000000000000000000000", .file = "other.gguf" }, &diag));
    // The result always passes the loader: a name the schema refuses fails here.
    try std.testing.expectError(error.InvalidConfigValue, register(alloc, text, "t.json", "bad/name", .{ .repo = "a/b", .revision = "0000000000000000000000000000000000000000", .file = "f.gguf" }, &diag));
    // A catalogue name may only be registered with the catalogue's own file.
    const qwen = catalog.find("qwen3.8-27b").?;
    try std.testing.expectError(error.RegistryConflict, register(alloc, text, "t.json", "qwen3.8-27b", .{ .repo = "a/b", .revision = qwen.revision, .file = "f.gguf" }, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "catalogue name") != null);
    const own = try register(alloc, text, "t.json", "qwen3.8-27b", .{ .repo = qwen.repo, .revision = qwen.revision, .file = qwen.file }, &diag);
    defer alloc.free(own);
    try std.testing.expect(std.mem.indexOf(u8, own, "\"qwen3.8-27b\": {") != null);
}

test "an entry named as a catalogue entry may not locate another file" {
    const alloc = std.testing.allocator;
    var diag: Diagnostic = .{};
    try std.testing.expectError(error.InvalidConfigValue, fromText(alloc,
        \\{ "schema_version": 1, "models": { "qwen3.8-27b": { "repo": "a/b", "file": "f.gguf" } } }
    , "t.json", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "models.qwen3.8-27b: qwen3.8-27b is a catalogue name") != null);
    try std.testing.expectError(error.InvalidConfigValue, fromText(alloc,
        \\{ "schema_version": 1, "models": { "qwen3.8-27b": { "path": "/x.gguf" } } }
    , "t.json", &diag));
    // The catalogue's own location, as `init` writes it, is fine.
    var ok = try fromText(alloc,
        \\{ "schema_version": 1, "models": { "qwen3.8-27b": { "repo": "unsloth/Qwen3.8-27B-GGUF", "file": "Qwen3.8-27B-UD-Q4_K_M.gguf", "ctx_size": 4096 } } }
    , "t.json", &diag);
    ok.deinit();
}

test "show prints the effective value of every key with its source in both forms" {
    const alloc = std.testing.allocator;
    var diag: Diagnostic = .{};
    var loaded = try fromText(alloc,
        \\{ "schema_version": 1, "engine": { "ctx_size": 4096 }, "generation": { "sampling": { "top_k": 40 } },
        \\  "models": { "mine": { "path": "custom/m.gguf", "ctx_size": 2048, "generation": { "sampling": { "top_p": 0.5 } } } } }
    , "/r/nuclis.json", &diag);
    defer loaded.deinit();
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try show(&loaded, &out.writer, false, .none);
    try std.testing.expect(std.mem.startsWith(u8, out.written(), "config: /r/nuclis.json (file)\nmodel: qwen3.8-27b (catalogue name; qwen38 profile)."));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "a null sampling key in the file takes the profile's value for generation.think = off") != null);
    try expectRow(out.written(), "schema_version", "1", "file");
    try expectRow(out.written(), "engine.model", "qwen3.8-27b", "default");
    try expectRow(out.written(), "engine.ctx_size", "4096", "file");
    // The profile's instruct values fill the unset sampling keys.
    try expectRow(out.written(), "generation.sampling.temperature", "0.7", "profile");
    try expectRow(out.written(), "generation.sampling.top_k", "40", "file");
    try expectRow(out.written(), "generation.sampling.presence_penalty", "1.5", "profile");
    try expectRow(out.written(), "agent.think", "low", "default");
    try expectRow(out.written(), "agent.fold_thinking", "true", "default");
    // The entry's stated keys, and only those.
    try expectRow(out.written(), "models.mine.path", "custom/m.gguf", "file");
    try expectRow(out.written(), "models.mine.ctx_size", "2048", "file");
    try expectRow(out.written(), "models.mine.generation.sampling.top_p", "0.5", "file");
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "models.mine.repo") == null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "models.mine.generation.sampling.top_k") == null);
    out.clearRetainingCapacity();
    try show(&loaded, &out.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, out.written(), .{});
    defer parsed.deinit();
    const object = parsed.value.object;
    try std.testing.expectEqualStrings("/r/nuclis.json", object.get("path").?.string);
    try std.testing.expect(object.get("found").?.bool);
    try std.testing.expectEqual(@as(i64, 4096), object.get("config").?.object.get("engine").?.object.get("ctx_size").?.integer);
    try std.testing.expect(object.get("config").?.object.get("generation").?.object.get("sampling").?.object.get("top_p").? == .null);
    try std.testing.expectEqualStrings("custom/m.gguf", object.get("config").?.object.get("models").?.object.get("mine").?.object.get("path").?.string);
    const effective = object.get("effective").?.object;
    try std.testing.expectEqualStrings("catalogue", effective.get("model_kind").?.string);
    try std.testing.expectEqualStrings("qwen38", effective.get("profile").?.string);
    try std.testing.expectEqual(@as(i64, 40), effective.get("generation").?.object.get("sampling").?.object.get("top_k").?.integer);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), effective.get("generation").?.object.get("sampling").?.object.get("top_p").?.float, 1e-6);
    try std.testing.expectEqualStrings("file", object.get("sources").?.object.get("engine.ctx_size").?.string);
    try std.testing.expectEqualStrings("default", object.get("sources").?.object.get("generation.think").?.string);
    try std.testing.expectEqualStrings("profile", object.get("sources").?.object.get("generation.sampling.top_p").?.string);
    try std.testing.expectEqual(@as(usize, leaf_count), object.get("sources").?.object.count());
    // The `config` object of `show --json` is itself a loadable file.
    var written: std.Io.Writer.Allocating = .init(alloc);
    defer written.deinit();
    try std.json.Stringify.value(object.get("config").?, .{}, &written.writer);
    var again = try fromText(alloc, written.written(), "again.json", &diag);
    defer again.deinit();
    try std.testing.expectEqual(loaded.config.engine.ctx_size, again.config.engine.ctx_size);
    try std.testing.expectEqualStrings(loaded.config.engine.model, again.config.engine.model);
    try std.testing.expectEqual(@as(usize, 2048), again.config.models.find("mine").?.ctx_size.?);
    // With the entry active, the view says so and its keys read `model`.
    out.clearRetainingCapacity();
    var active = try fromText(alloc,
        \\{ "schema_version": 1, "engine": { "model": "mine" }, "models": { "mine": { "path": "custom/m.gguf", "ctx_size": 2048 } } }
    , "/r/nuclis.json", &diag);
    defer active.deinit();
    try show(&active, &out.writer, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "model: mine (registry entry; qwen38 profile)") != null);
    try expectRow(out.written(), "engine.model", "mine", "file");
    try expectRow(out.written(), "engine.ctx_size", "2048", "model");
    // Styled: the same rows, values wrapped in the layer's color (orange
    // for `model`), the padding inside the span.
    out.clearRetainingCapacity();
    try show(&active, &out.writer, false, .{ .theme = .{ .kind = .truecolor }, .enabled = true });
    try std.testing.expect(std.mem.indexOf(u8, out.written(), " \x1b[38;2;254;128;25m2048") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\x1b[0m \x1b[38;2;254;128;25mmodel\x1b[0m\n") != null);
}

test "the written initial file parses back exactly and shows the entry shape" {
    const alloc = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try writeInitial(&out.writer);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"schema_version\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"temperature\": null") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"qwen3.8-27b\": {") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"revision\": \"4ca720788d1e01f1bff70c033e0d0028fd02e502\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"mtp\": \"MTP/mtp-Qwen3.8-27B-Q4_0.gguf\"") != null);
    var diag: Diagnostic = .{};
    var loaded = try fromText(alloc, out.written(), "d.json", &diag);
    defer loaded.deinit();
    try std.testing.expectEqualDeep(initial(), loaded.config);
    // The entry resolves like the catalogue name it mirrors: same globals, same profile.
    const gen = resolve(&loaded, null, .{}, .generate);
    try std.testing.expect(gen.entry != null);
    try std.testing.expectEqual(.qwen38, gen.profile);
    try std.testing.expectEqual(@as(usize, 16384), gen.ctx_size);
    // The entry carries the measured speculative verdict (ENGN-17): Qwen's
    // record does not pay for verification, Muse's does.
    try std.testing.expectEqual(false, gen.entry.?.generation.speculative.?);
    try std.testing.expectEqual(@as(usize, 4), gen.entry.?.generation.draft_length.?);
    const muse = resolve(&loaded, "muse-glimmer-30b", .{}, .generate);
    try std.testing.expectEqual(true, muse.entry.?.generation.speculative.?);
    try std.testing.expect(muse.speculative);
    try std.testing.expect(!gen.speculative);
}
