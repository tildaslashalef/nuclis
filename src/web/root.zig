//! Web search and page fetch through the user's search service: the client
//! (`client.zig`) and the two tools' shared definitions and rendering
//! (`tools.zig`), used by the agent and by `nuclis serve`.
pub const client = @import("client.zig");
pub const tools = @import("tools.zig");

pub const Service = client.Service;
pub const Beat = client.Beat;
pub const validBase = client.validBase;

test {
    _ = client;
    _ = tools;
}
