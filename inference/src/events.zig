//! Semantic completion output. Every string borrows memory only until the
//! sink's `send` returns; a consumer retaining history must copy it. Profiles
//! own wire syntax, while consumers own tool execution and presentation.
const engine = @import("engine.zig");

/// A completion's tool call and a history assistant call are the same
/// host-correlated shape, so the profile registry owns the one definition
/// (`profiles.ToolCall`): the `id` is host correlation data a native format
/// need not serialize, `name` selects the tool, and `arguments` is a complete
/// JSON object the profile normalizes from native syntax. The agent still
/// validates its fields against the registered tool.
pub const ToolCall = @import("profiles/root.zig").ToolCall;

pub const Event = union(enum) {
    thinking: []const u8,
    answer: []const u8,
    /// Bytes of a call body collected so far, sent as it grows: a long call
    /// streams nothing else, and a consumer can show it is being written.
    tool_progress: usize,
    tool_call: ToolCall,
    /// A call the completion stopped inside (cancellation, the output budget,
    /// or an unparseable body at EOS), dropped after this many body bytes:
    /// it is neither an action nor answer text.
    tool_cut: usize,
    stop: engine.Outcome,
};
