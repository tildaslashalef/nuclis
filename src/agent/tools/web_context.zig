//! `web_context`: the shared tool (`web/tools.zig`) on the agent's workspace.
//! Offered only when the search service answers `/v1/context`.
const std = @import("std");
const root = @import("root.zig");
const web = @import("../../web/root.zig");
const web_search = @import("web_search.zig");

pub const tool: root.Tool = .{
    .name = web.tools.context.name,
    .description = web.tools.context.description,
    .parameters = web.tools.context.parameters,
    .label = "Reading the web",
    .display = "Context",
    .subject = "query",
    .run = run,
};

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const service = workspace.web orelse return root.fail(alloc, "web_context: no search service is configured", .{});
    var beat = web_search.Beat{ .workspace = workspace };
    return web_search.adopt(try web.tools.runContext(service, alloc, arguments, beat.seam()));
}
