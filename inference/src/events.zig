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
    tool_call: ToolCall,
    stop: engine.Outcome,
};
