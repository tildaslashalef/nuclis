//! The generic adapter registry: a table of families in, an enum of
//! adapters and the dispatch over it out. `models/root.zig` instantiates it
//! over the real table; the tests here instantiate it over stubs, which is
//! why it is a function of the table rather than a fixed namespace.
//!
//! A family is a namespace (a Zig `type` used as a struct of declarations)
//! with:
//!
//! - `architecture`: the `general.architecture` id it binds;
//! - `executableEncoding(id) bool`: the storage layouts its kernels execute;
//! - `Binding` and `bind(alloc, doc) BindError!Binding`: metadata validation
//!   and named tensor binding; `Binding.summary` is a `Summary`;
//! - `Runtime` (the CPU reference) and `Plan` (the Metal executor), both
//!   with the step/prefill/reset/session contract the engine's `Executor`
//!   calls.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");

/// What `bind` may report. Shared so `validate` has one typed error set
/// whatever the family; each adapter narrows what it actually returns.
pub const BindError = std.mem.Allocator.Error || error{
    UnsupportedArchitecture,
    MissingMetadata,
    InvalidMetadata,
    UnsupportedConfiguration,
    MissingTensor,
    UnexpectedTensor,
    DuplicateTensor,
    InvalidTensorShape,
    UnsupportedTensorEncoding,
    Overflow,
};

/// One kind of decoder layer and how many the architecture has, named by
/// the adapter (`full_attention`, `delta_net`, ...): `nuclis validate`
/// prints the composition without knowing the families.
pub const LayerKind = struct { kind: []const u8, count: u32 };

/// What a successful binding says about the artifact, rendered by
/// `nuclis validate`. Adapter-neutral: the composition is a list of
/// `LayerKind`, not fields named after one architecture's layers.
pub const Summary = struct {
    /// The adapter's name for the exact configuration it binds.
    profile: []const u8,
    validation: []const u8 = "structure_only",
    decoder_layers: u32,
    layer_kinds: []const LayerKind,
    /// Layers bound only to be excluded from the text schedule (a draft
    /// head embedded in the main file); zero when the file has none.
    auxiliary_prediction_layers: u32 = 0,
    text_tensors: u32,
    auxiliary_tensors: u32,
    text_tensor_bytes: u64,
    auxiliary_tensor_bytes: u64,
    /// A successful binding is exactly the schedule the runtimes execute;
    /// this does not claim numerical or coding-quality acceptance.
    inference_available: bool = true,
};

/// Compile-time check that a family exposes the registry's surface; the
/// error names the missing declaration rather than failing at a use site.
fn checkFamily(comptime Family: type) void {
    inline for (.{ "architecture", "executableEncoding", "Binding", "bind", "Runtime", "Plan" }) |name| {
        if (!@hasDecl(Family, name)) @compileError("adapter family " ++ @typeName(Family) ++ " lacks `" ++ name ++ "`");
    }
}

/// The registry over a table of families. `families` must be comptime-known
/// and non-empty; every family's `architecture` becomes a tag of `Adapter`,
/// so the ids must be distinct Zig identifiers.
pub fn Registry(comptime families: []const type) type {
    comptime {
        if (families.len == 0) @compileError("the adapter registry needs at least one family");
        for (families) |Family| checkFamily(Family);
    }
    return struct {
        const Self = @This();

        /// The adapter for a `general.architecture` id: what makes a GGUF
        /// *runnable* as opposed to merely parseable. Built from the table
        /// with `@Enum`, so a tag exists exactly when a family does.
        pub const Adapter = blk: {
            var values: [families.len]u8 = undefined;
            for (0..families.len) |i| values[i] = i;
            break :blk @Enum(u8, .exhaustive, known, &values);
        };

        /// The architecture ids in table order, for messages that name the
        /// known ones.
        pub const known: []const []const u8 = blk: {
            var names: [families.len][]const u8 = undefined;
            for (families, 0..) |Family, i| names[i] = Family.architecture;
            const final = names;
            break :blk &final;
        };

        /// The family behind a tag, for `inline` dispatch.
        pub fn family(comptime adapter: Adapter) type {
            return families[@intFromEnum(adapter)];
        }

        pub fn adapterFor(architecture: []const u8) ?Adapter {
            inline for (families, 0..) |Family, i| {
                if (std.mem.eql(u8, architecture, Family.architecture)) return @enumFromInt(i);
            }
            return null;
        }

        /// `adapterFor` as a typed failure for loaders: the caller names
        /// `known` in its diagnostic.
        pub fn select(architecture: []const u8) error{UnknownArchitecture}!Adapter {
            return adapterFor(architecture) orelse error.UnknownArchitecture;
        }

        pub fn executableEncoding(adapter: Adapter, encoding_id: u32) bool {
            return switch (adapter) {
                inline else => |a| family(a).executableEncoding(encoding_id),
            };
        }

        /// Binds the directory with the family and returns what the binding
        /// says about it. No weights are read; the binding itself is
        /// discarded, so callers that will execute bind again through the
        /// engine.
        pub fn validate(adapter: Adapter, alloc: std.mem.Allocator, doc: *const gguf.Document) BindError!Summary {
            return switch (adapter) {
                inline else => |a| (try family(a).bind(alloc, doc)).summary,
            };
        }
    };
}

// A registry over stub families: the table is the only thing that decides
// the tags, the lookup, the known list, and the dispatch.
const StubBinding = struct { summary: Summary };
const StubRuntime = struct {};
const StubPlan = struct {};
const stub_kinds = [_]LayerKind{.{ .kind = "dense_attention", .count = 2 }};
const StubAlpha = struct {
    pub const architecture = "alpha";
    pub const Binding = StubBinding;
    pub const Runtime = StubRuntime;
    pub const Plan = StubPlan;
    pub fn executableEncoding(id: u32) bool {
        return id == 0;
    }
    pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) BindError!Binding {
        _ = alloc;
        if (doc.tensors.len != 0) return error.UnexpectedTensor;
        return .{ .summary = .{ .profile = "alpha_test", .decoder_layers = 2, .layer_kinds = &stub_kinds, .text_tensors = 0, .auxiliary_tensors = 0, .text_tensor_bytes = 0, .auxiliary_tensor_bytes = 0 } };
    }
};
const StubBeta = struct {
    pub const architecture = "beta";
    pub const Binding = StubBinding;
    pub const Runtime = StubRuntime;
    pub const Plan = StubPlan;
    pub fn executableEncoding(id: u32) bool {
        return id == 8;
    }
    pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) BindError!Binding {
        _ = alloc;
        _ = doc;
        return error.MissingTensor;
    }
};

test "a registry over stub families derives its tags and dispatch from the table" {
    const Stub = Registry(&.{ StubAlpha, StubBeta });
    try std.testing.expectEqual(2, @typeInfo(Stub.Adapter).@"enum".fields.len);
    try std.testing.expectEqualStrings("alpha", @tagName(Stub.adapterFor("alpha").?));
    try std.testing.expectEqualStrings("beta", @tagName(Stub.adapterFor("beta").?));
    try std.testing.expect(Stub.adapterFor("qwen35") == null);
    try std.testing.expectError(error.UnknownArchitecture, Stub.select("qwen35"));
    try std.testing.expectEqualStrings("alpha", Stub.known[0]);
    try std.testing.expectEqualStrings("beta", Stub.known[1]);
    const alpha = Stub.adapterFor("alpha").?;
    const beta = Stub.adapterFor("beta").?;
    try std.testing.expect(Stub.executableEncoding(alpha, 0) and !Stub.executableEncoding(alpha, 8));
    try std.testing.expect(Stub.executableEncoding(beta, 8) and !Stub.executableEncoding(beta, 0));
    // `family` takes a comptime tag: the lookup result is runtime, the
    // literal is not.
    try std.testing.expectEqual(StubRuntime, Stub.family(.alpha).Runtime);
    try std.testing.expectEqual(StubPlan, Stub.family(.beta).Plan);
    var doc: gguf.Document = .{ .storage = .init(std.testing.allocator), .version = 3, .file_bytes = 0, .directory_bytes = 0, .data_offset = 0, .alignment = 32, .metadata = &.{}, .tensors = &.{} };
    defer doc.deinit();
    const summary = try Stub.validate(alpha, std.testing.allocator, &doc);
    try std.testing.expectEqualStrings("alpha_test", summary.profile);
    try std.testing.expectEqual(@as(u32, 2), summary.layer_kinds[0].count);
    try std.testing.expectError(error.MissingTensor, Stub.validate(beta, std.testing.allocator, &doc));
}
