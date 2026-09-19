//! Reusable inference modules. Container inspection is the first implemented layer.
//! No CLI, user-directory policy, or architecture-specific behavior belongs here.
pub const gguf = @import("formats/gguf.zig");
pub const encoding = @import("tensor/encoding.zig");
pub const quant = @import("quant/decode.zig");
pub const metal = @import("backends/metal/root.zig");
pub const cpu = @import("backends/cpu/root.zig");
pub const profiles = @import("profiles/root.zig");
pub const vocabulary = @import("tokenizer/vocabulary.zig");
pub const bpe = @import("tokenizer/bpe.zig");
pub const text_stream = @import("tokenizer/stream.zig");
pub const tokenizer = @import("tokenizer/encode.zig");
pub const models = @import("models/root.zig");
pub const weights = @import("runtime/weights.zig");
pub const sampling = @import("sampling/root.zig");
pub const session = @import("runtime/session.zig");
pub const draft = @import("runtime/draft.zig");
pub const observer = @import("runtime/observer.zig");
pub const events = @import("events.zig");
pub const engine = @import("engine.zig");

test {
    _ = gguf;
    _ = encoding;
    _ = quant;
    _ = cpu;
    _ = profiles;
    _ = vocabulary;
    _ = bpe;
    _ = tokenizer;
    _ = text_stream;
    _ = models;
    _ = weights;
    _ = session;
    _ = sampling;
    _ = engine;
    _ = events;
}
