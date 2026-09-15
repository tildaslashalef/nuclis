//! The engine-event sink the agent loop feeds. `inference.events.Event`
//! carries semantic channels; the loop (`loop.zig`) maps them to terminal
//! events and tool calls, so this module holds only the untyped function
//! pointer the engine writes through and no model syntax.
const inference = @import("inference");

/// Adapts application state that already has a `send`-like method for
/// inference events. The agent loop installs one whose handler parses the
/// answer channel; tests install a recorder.
pub const Sink = struct {
    context: *anyopaque,
    call: *const fn (*anyopaque, inference.events.Event) anyerror!void,

    pub fn send(self: *Sink, event: inference.events.Event) !void {
        try self.call(self.context, event);
    }
};
