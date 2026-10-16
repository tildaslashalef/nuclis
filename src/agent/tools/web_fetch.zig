//! `web_fetch`: the shared tool (`web/tools.zig`) on the agent's workspace.
//! Offered only when the workspace has a search service.
const std = @import("std");
const root = @import("root.zig");
const web = @import("../../web/root.zig");
const web_search = @import("web_search.zig");

pub const tool: root.Tool = .{
    .name = web.tools.fetch.name,
    .description = web.tools.fetch.description,
    .parameters = web.tools.fetch.parameters,
    .label = "Fetching",
    .display = "Fetch",
    .subject = "url",
    .run = run,
};

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const service = workspace.web orelse return root.fail(alloc, "web_fetch: no search service is configured", .{});
    var beat = web_search.Beat{ .workspace = workspace };
    return web_search.adopt(try web.tools.runFetch(service, alloc, arguments, beat.seam()));
}
