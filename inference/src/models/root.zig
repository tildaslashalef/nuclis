//! Architecture adapters own metadata interpretation and named weight
//! bindings. This root is the adapter registry's table: the explicit
//! list the engine selects from by `general.architecture`. Container code
//! stays unaware of the adapters' equations, layer schedules, and prompt
//! conventions; it only knows that every family in the table exposes the
//! names `registry.zig` documents.
//!
//! Adding an architecture is one line in `table` plus the adapter's own
//! files: the `Adapter` enum and the engine's executor union are both
//! derived from the table, so nothing else in the tree lists the families.
const std = @import("std");
pub const registry_module = @import("registry.zig");

pub const qwen35 = @import("qwen35.zig");
pub const qwen35_runtime = @import("qwen35_runtime.zig");
pub const qwen35_metal = @import("qwen35_metal.zig");
pub const gemma4 = @import("gemma4.zig");
pub const gemma4_runtime = @import("gemma4_runtime.zig");
pub const gemma4_metal = @import("gemma4_metal.zig");
pub const inventory = @import("inventory.zig");

/// The registered families, in lookup order. Each registers itself through
/// its `family` declaration.
pub const table = [_]type{ qwen35.family, gemma4.family };

pub const Registry = registry_module.Registry;
pub const BindError = registry_module.BindError;
pub const LayerKind = registry_module.LayerKind;
pub const Summary = registry_module.Summary;

pub const registry = Registry(&table);
/// The adapter for a `general.architecture` id: what makes a GGUF
/// *runnable* as opposed to merely parseable.
pub const Adapter = registry.Adapter;
/// The architecture ids the tree has, for diagnostics naming them.
pub const known = registry.known;
pub const adapterFor = registry.adapterFor;
pub const select = registry.select;

test "adapters are looked up by architecture id" {
    try std.testing.expectEqual(Adapter.qwen35, adapterFor("qwen35").?);
    try std.testing.expect(adapterFor("clip") == null);
    try std.testing.expect(adapterFor("") == null);
    try std.testing.expectEqual(Adapter.gemma4, try select("gemma4"));
    try std.testing.expectError(error.UnknownArchitecture, select("gemma3"));
    try std.testing.expect(registry.executableEncoding(.qwen35, 12) and !registry.executableEncoding(.qwen35, 2));
    try std.testing.expect(registry.executableEncoding(.gemma4, 2) and !registry.executableEncoding(.gemma4, 30));
    try std.testing.expectEqual(2, known.len);
    try std.testing.expectEqualStrings("qwen35", known[0]);
    try std.testing.expectEqualStrings("gemma4", known[1]);
}

test "validate through the registry is the adapter's binding summary" {
    var doc = try qwen35.inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const summary = try registry.validate(.qwen35, std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 64), summary.decoder_layers);
    try std.testing.expectEqual(@as(u32, 851), summary.text_tensors);
    try std.testing.expectEqualStrings("delta_net", summary.layer_kinds[1].kind);
}

test "validate through the registry reaches the second family" {
    var doc = try gemma4.inventoryDocument(std.testing.allocator);
    defer doc.deinit();
    const summary = try registry.validate(.gemma4, std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 48), summary.decoder_layers);
    try std.testing.expectEqualStrings("global_attention", summary.layer_kinds[1].kind);
    try std.testing.expectError(error.UnsupportedArchitecture, registry.validate(.qwen35, std.testing.allocator, &doc));
}

test {
    _ = registry_module;
    _ = inventory;
    _ = qwen35;
    _ = qwen35_runtime;
    _ = gemma4;
    _ = gemma4_runtime;
    _ = gemma4_metal;
}
