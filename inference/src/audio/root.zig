//! Audio through the companion encoders: a clip becomes 16 kHz samples,
//! log-mel frames, and feature rows at the language model's width, which
//! replace an audio span's embedding rows. docs/engine/audio.md. Speech
//! out: Qwen3-TTS's codec decoder turns codes into 24 kHz samples
//! (docs/models/qwen3-tts.md).
pub const audio = @import("audio.zig");
pub const mel = @import("mel.zig");
pub const gemma4a = @import("gemma4a.zig");
/// Qwen3-TTS's codec decoder: speech out, codes to samples.
pub const qwen3_codec = @import("qwen3_codec.zig");
/// Nemotron 3.5 ASR's cache-aware encoder: speech in, mel rows to encoder rows.
pub const nemotron_encoder = @import("nemotron_encoder.zig");

test {
    _ = audio;
    _ = mel;
    _ = gemma4a;
    _ = qwen3_codec;
    _ = nemotron_encoder;
}
