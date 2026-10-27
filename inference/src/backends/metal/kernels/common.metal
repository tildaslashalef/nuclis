// nuclis compute kernels. Convention: data buffers bind at indices 0..6 in the
// order listed by each kernel; a small parameter struct binds at index 7.
// All arithmetic is F32; reductions use SIMD-group sums and, where a row is
// wider than one SIMD group serves well, a 256-thread threadgroup.

inline float nu_silu(float x) { return x / (1.0f + exp(-x)); }
inline float nu_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float nu_softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-abs(x))); }
// tanh saturating exactly: Metal's tanh goes through exp and returns NaN
// past about ±44 (exp overflow), where the true value is ±1 to F32
// precision already at ±20 (1 − 2e-17 rounds to 1). Clamping there changes
// no finite result and keeps large gate values and logits finite.
inline float nu_tanh(float x) { return tanh(clamp(x, -20.0f, 20.0f)); }
// GELU in the tanh approximation (`cpu.gelu`, the form GGUF checkpoints
// declare as gelu_pytorch_tanh; Gemma 4's FFN gate).
inline float nu_gelu(float x) { return 0.5f * x * (1.0f + nu_tanh(0.7978845608028654f * x * (1.0f + 0.044715f * x * x))); }

