//! Vision through the companion projectors: an image becomes feature rows at
//! the language model's width, substituted for an image span's embedding
//! rows at prefill. The contract, each family's projector facts, and the
//! measurements live in docs/reference/vision.md.
/// One image span in a prompt: the placeholder-token run `[start, start +
/// count)` whose embedding rows are a projector's feature rows, with the
/// merged grid that sets its multi-axis RoPE positions.
pub const Span = struct { start: usize, count: usize, width_tokens: u32, height_tokens: u32 };

pub const image = @import("image.zig");
pub const preprocess = @import("preprocess.zig");
pub const qwen3vl = @import("qwen3vl.zig");
pub const qwen3vl_metal = @import("qwen3vl_metal.zig");

test {
    _ = image;
    _ = preprocess;
    _ = qwen3vl;
}
