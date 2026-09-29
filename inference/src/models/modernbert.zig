//! The ModernBERT encoder family: its `config.json` and its tensors in a
//! safetensors checkpoint. Pure: the caller reads the config (bounded) and
//! opens the checkpoint; `bind` returns borrowed tensor refs, validated by
//! name, shape, and dtype. The forward is modernbert_runtime.zig; the shapes
//! and the forward's contract are in docs/reference/laya.md.
const std = @import("std");
const safetensors = @import("../formats/safetensors.zig");

pub const Error = std.mem.Allocator.Error || error{
    InvalidConfig,
    UnsupportedConfig,
    MissingTensor,
    TensorShapeMismatch,
    UnsupportedDtype,
};

pub const Config = struct {
    layers: usize,
    hidden: usize,
    heads: usize,
    intermediate: usize,
    vocabulary: usize,
    norm_eps: f32,
    /// Half the local window: a sliding layer's query i sees keys |i − j| ≤ window.
    window: usize,
    global_theta: f32,
    local_theta: f32,
    /// Bit i set: layer i attends globally, else within `window`.
    global: std.StaticBitSet(max_layers),

    pub const max_layers = 256;

    pub fn headDim(self: Config) usize {
        return self.hidden / self.heads;
    }
};

const RopeTheta = struct { rope_theta: f64 };
const Json = struct {
    model_type: []const u8,
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    norm_eps: f64 = 1e-5,
    local_attention: usize,
    hidden_activation: []const u8 = "gelu",
    attention_bias: bool = false,
    mlp_bias: bool = false,
    norm_bias: bool = false,
    layer_types: ?[]const []const u8 = null,
    global_attn_every_n_layers: ?usize = null,
    rope_parameters: ?struct { full_attention: RopeTheta, sliding_attention: RopeTheta } = null,
    // The Transformers 4.x spellings of the two thetas.
    global_rope_theta: ?f64 = null,
    local_rope_theta: ?f64 = null,
};

/// Parses `config.json`. Biases, an activation other than exact GELU, odd
/// head widths, and unknown layer types are rejected as unsupported.
pub fn parseConfig(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
    const parsed = std.json.parseFromSlice(Json, gpa, bytes, .{ .ignore_unknown_fields = true }) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidConfig,
    };
    defer parsed.deinit();
    const j = parsed.value;
    if (!std.mem.eql(u8, j.model_type, "modernbert")) return error.UnsupportedConfig;
    if (j.attention_bias or j.mlp_bias or j.norm_bias or !std.mem.eql(u8, j.hidden_activation, "gelu"))
        return error.UnsupportedConfig;
    if (j.num_hidden_layers == 0 or j.num_hidden_layers > Config.max_layers or j.hidden_size == 0 or
        j.hidden_size > 65536 or j.num_attention_heads == 0 or j.hidden_size % j.num_attention_heads != 0 or
        j.intermediate_size == 0 or j.intermediate_size > 1 << 20 or j.vocab_size == 0 or j.vocab_size > 1 << 24 or
        j.local_attention < 2 or !(j.norm_eps > 0 and j.norm_eps < 1)) return error.InvalidConfig;
    // RoPE pairs dimensions and the CPU kernels read eight lanes at a time.
    if ((j.hidden_size / j.num_attention_heads) % 8 != 0) return error.UnsupportedConfig;
    var config: Config = .{
        .layers = j.num_hidden_layers,
        .hidden = j.hidden_size,
        .heads = j.num_attention_heads,
        .intermediate = j.intermediate_size,
        .vocabulary = j.vocab_size,
        .norm_eps = @floatCast(j.norm_eps),
        .window = j.local_attention / 2,
        .global_theta = undefined,
        .local_theta = undefined,
        .global = .initEmpty(),
    };
    if (j.layer_types) |types| {
        if (types.len != config.layers) return error.InvalidConfig;
        for (types, 0..) |t, i| {
            if (std.mem.eql(u8, t, "full_attention")) config.global.set(i) else if (!std.mem.eql(u8, t, "sliding_attention")) return error.UnsupportedConfig;
        }
    } else {
        const every = j.global_attn_every_n_layers orelse return error.InvalidConfig;
        if (every == 0) return error.InvalidConfig;
        for (0..config.layers) |i| if (i % every == 0) config.global.set(i);
    }
    const global: f64, const local: f64 = if (j.rope_parameters) |r|
        .{ r.full_attention.rope_theta, r.sliding_attention.rope_theta }
    else
        .{ j.global_rope_theta orelse return error.InvalidConfig, j.local_rope_theta orelse return error.InvalidConfig };
    if (!(global >= 1 and global < 1e12 and local >= 1 and local < 1e12)) return error.InvalidConfig;
    config.global_theta = @floatCast(global);
    config.local_theta = @floatCast(local);
    return config;
}

pub const Layer = struct {
    /// Absent on layer 0, whose attention input is not normalized.
    attn_norm: ?safetensors.Ref,
    /// [3·hidden][hidden]: q, k, v rows, each [heads][head_dim].
    qkv: safetensors.Ref,
    out: safetensors.Ref,
    mlp_norm: safetensors.Ref,
    /// [2·intermediate][hidden]: the GELU input rows, then the gate rows.
    wi: safetensors.Ref,
    wo: safetensors.Ref,
};

pub const Weights = struct {
    embeddings: safetensors.Ref,
    embed_norm: safetensors.Ref,
    layers: []const Layer,
    final_norm: safetensors.Ref,
    /// How many checkpoint tensors these refs name.
    count: usize,
};

/// Binds every encoder tensor under `prefix` (`"encoder."` in Laya). `arena`
/// owns the layer array; refs borrow the checkpoint, which must outlive them.
pub fn bind(arena: std.mem.Allocator, checkpoint: *const safetensors.Checkpoint, config: Config, prefix: []const u8) Error!Weights {
    const h = config.hidden;
    var binder: Binder = .{ .checkpoint = checkpoint, .prefix = prefix };
    const layers = try arena.alloc(Layer, config.layers);
    for (layers, 0..) |*layer, i| {
        var name_buffer: [64]u8 = undefined;
        const at = std.fmt.bufPrint(&name_buffer, "layers.{d}.", .{i}) catch unreachable;
        layer.* = .{
            .attn_norm = if (i == 0) null else try binder.get(at, "attn_norm.weight", &.{h}),
            .qkv = try binder.get(at, "attn.Wqkv.weight", &.{ 3 * h, h }),
            .out = try binder.get(at, "attn.Wo.weight", &.{ h, h }),
            .mlp_norm = try binder.get(at, "mlp_norm.weight", &.{h}),
            .wi = try binder.get(at, "mlp.Wi.weight", &.{ 2 * config.intermediate, h }),
            .wo = try binder.get(at, "mlp.Wo.weight", &.{ h, config.intermediate }),
        };
    }
    return .{
        .embeddings = try binder.get("", "embeddings.tok_embeddings.weight", &.{ config.vocabulary, h }),
        .embed_norm = try binder.get("", "embeddings.norm.weight", &.{h}),
        .layers = layers,
        .final_norm = try binder.get("", "final_norm.weight", &.{h}),
        .count = binder.count,
    };
}

/// Looks tensors up by `prefix ++ at ++ name` and checks their shape and dtype.
pub const Binder = struct {
    checkpoint: *const safetensors.Checkpoint,
    prefix: []const u8,
    count: usize = 0,

    pub fn get(self: *Binder, at: []const u8, name: []const u8, shape: []const u64) Error!safetensors.Ref {
        var buffer: [256]u8 = undefined;
        const full = std.fmt.bufPrint(&buffer, "{s}{s}{s}", .{ self.prefix, at, name }) catch return error.MissingTensor;
        const ref = self.checkpoint.get(full) orelse return error.MissingTensor;
        if (!std.mem.eql(u64, ref.tensor.shape, shape)) return error.TensorShapeMismatch;
        switch (ref.tensor.dtype) {
            .f16, .bf16, .f32 => {},
            else => return error.UnsupportedDtype,
        }
        self.count += 1;
        return ref;
    }
};

const laya_config =
    \\{"model_type":"modernbert","hidden_size":1024,"num_attention_heads":16,"num_hidden_layers":4,
    \\ "intermediate_size":2624,"vocab_size":50368,"norm_eps":1e-05,"local_attention":128,
    \\ "hidden_activation":"gelu","attention_bias":false,"mlp_bias":false,"norm_bias":false,
    \\ "layer_types":["full_attention","sliding_attention","sliding_attention","full_attention"],
    \\ "rope_parameters":{"full_attention":{"rope_theta":160000.0,"rope_type":"default"},
    \\  "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}
;

test "config: layer schedule, thetas, both spellings, and rejections" {
    const gpa = std.testing.allocator;
    const c = try parseConfig(gpa, laya_config);
    try std.testing.expectEqual(@as(usize, 64), c.headDim());
    try std.testing.expectEqual(@as(usize, 64), c.window);
    try std.testing.expect(c.global.isSet(0) and !c.global.isSet(1) and !c.global.isSet(2) and c.global.isSet(3));
    try std.testing.expectEqual(@as(f32, 160000), c.global_theta);
    try std.testing.expectEqual(@as(f32, 10000), c.local_theta);
    const legacy =
        \\{"model_type":"modernbert","hidden_size":64,"num_attention_heads":1,"num_hidden_layers":4,
        \\ "intermediate_size":8,"vocab_size":10,"local_attention":8,"global_attn_every_n_layers":2,
        \\ "global_rope_theta":160000,"local_rope_theta":10000}
    ;
    const l = try parseConfig(gpa, legacy);
    try std.testing.expect(l.global.isSet(0) and !l.global.isSet(1) and l.global.isSet(2));
    try std.testing.expectEqual(@as(f32, 1e-5), l.norm_eps);
    const cases = .{
        .{ "\"model_type\":\"modernbert\"", "\"model_type\":\"bert\"", error.UnsupportedConfig },
        .{ "\"mlp_bias\":false", "\"mlp_bias\":true", error.UnsupportedConfig },
        .{ "\"hidden_activation\":\"gelu\"", "\"hidden_activation\":\"gelu_new\"", error.UnsupportedConfig },
        .{ "\"sliding_attention\",\"full", "\"chunked_attention\",\"full", error.UnsupportedConfig },
        .{ "\"num_attention_heads\":16", "\"num_attention_heads\":12", error.InvalidConfig },
        .{ "\"num_attention_heads\":16", "\"num_attention_heads\":256", error.UnsupportedConfig },
        .{ "\"num_hidden_layers\":4", "\"num_hidden_layers\":5", error.InvalidConfig },
        .{ "\"rope_parameters\"", "\"rope_parameterz\"", error.InvalidConfig },
        .{ "{\"model_type\"", "[{\"model_type\"", error.InvalidConfig },
    };
    inline for (cases) |case| {
        const bytes = try std.mem.replaceOwned(u8, gpa, laya_config, case[0], case[1]);
        defer gpa.free(bytes);
        try std.testing.expectError(case[2], parseConfig(gpa, bytes));
    }
}
