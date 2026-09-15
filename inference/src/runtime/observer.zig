//! Layer-boundary callbacks shared by every adapter's executors. The
//! engine, the trace writer, and the agent install one `Observer`; the CPU
//! runtime and the Metal plan of any architecture call it the same way, so
//! it lives beside the session rather than inside one adapter.
//!
//! Every callback is optional and may return an error (cancellation, time
//! limit) that fails the step and poisons the session.

/// Which half of a turn is running. Prefill consumes the prompt — on the GPU
/// in chunks of many tokens per command buffer — and decode produces one
/// token per step.
pub const Phase = enum { prefill, decode };

/// How far a turn has got. `position` counts tokens consumed (prefill) or
/// produced (decode); `target` is the prompt length or the output budget.
/// Both are needed because a chunked prefill advances by hundreds of tokens
/// at a time, so a caller cannot infer either from the number of calls.
pub const Progress = struct {
    phase: Phase,
    position: usize,
    target: usize,
};

pub const Observer = struct {
    context: *anyopaque,
    /// Called at every layer boundary without activations. The GPU plan calls
    /// it while recording, so it costs no synchronization; use it for
    /// cancellation and time limits.
    check: ?*const fn (*anyopaque) anyerror!void = null,
    /// Called after each complete layer with its output, borrowed for the
    /// callback only. Needs the values computed, so the GPU plan commits a
    /// command buffer per layer while this is set: traces only, never production.
    layer: ?*const fn (*anyopaque, usize, []const f32) anyerror!void = null,
    /// Called when a turn advances: after every prefill chunk (GPU) or prompt
    /// token (CPU), and after every generated token. It is the only beat an
    /// interactive caller can count on — `check` says nothing about how far
    /// along the turn is, and the loop's per-token `step` hook fires once for
    /// a whole chunked prefill. `generate` and `bench` do not register it, so
    /// a measured run carries no display work.
    progress: ?*const fn (*anyopaque, Progress) anyerror!void = null,
};
