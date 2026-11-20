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
    /// A call dropped unrun: the completion stopped inside it, or its body
    /// did not parse. It is neither an action nor answer text.
    tool_cut: ToolCut,
    stop: engine.Outcome,
};

/// Why a call was dropped, and enough of it to say so. Slices borrow until
/// the sink's `send` returns, except `hint`, which is static.
pub const ToolCut = struct {
    bytes: usize,
    reason: CutReason = .malformed,
    /// Byte offset in the body where reading stopped.
    at: usize = 0,
    /// The tool name the body named, or empty.
    name: []const u8 = "",
    /// The body's first and last bytes, at most `cut_sample` each.
    head: []const u8 = "",
    tail: []const u8 = "",
    /// The profile's one-line reminder of its call syntax; may be empty.
    hint: []const u8 = "",
};

pub const CutReason = enum {
    /// The completion stopped before the call's closing token.
    unclosed,
    /// No tool name could be read.
    no_name,
    /// An escape the call syntax does not define.
    invalid_escape,
    /// Any other break in the call syntax.
    malformed,
};

/// Bytes of a cut body kept at each end for the report.
pub const cut_sample = 512;
