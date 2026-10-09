//! Audio through the companion encoders: a clip becomes 16 kHz samples,
//! log-mel frames, and feature rows at the language model's width, which
//! replace an audio span's embedding rows. docs/engine/audio.md.
pub const audio = @import("audio.zig");
pub const mel = @import("mel.zig");
pub const gemma4a = @import("gemma4a.zig");

test {
    _ = audio;
    _ = mel;
    _ = gemma4a;
}
