//! `web_search`: the shared tool (`web/tools.zig`) on the agent's workspace.
//! Offered only when the workspace has a search service.
const std = @import("std");
const root = @import("root.zig");
const web = @import("../../web/root.zig");
const interrupt = @import("../../interrupt.zig");

pub const tool: root.Tool = .{
    .name = web.tools.search.name,
    .description = web.tools.search.description,
    .parameters = web.tools.search.parameters,
    .label = "Searching",
    .display = "Search",
    .subject = "query",
    .run = run,
};

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const service = workspace.web orelse return root.fail(alloc, "web_search: no search service is configured", .{});
    var beat = Beat{ .workspace = workspace };
    return adopt(try web.tools.runSearch(service, alloc, arguments, beat.seam()));
}

/// The agent's side of a web request's wait: the workspace's tick (keys,
/// repaint), then Ctrl-C, which cancels the request.
pub const Beat = struct {
    workspace: root.Workspace,

    pub fn seam(self: *Beat) web.Beat {
        return .{ .context = self, .call = call };
    }

    fn call(context: *anyopaque) bool {
        const self: *Beat = @ptrCast(@alignCast(context));
        if (self.workspace.tick) |beat| beat.call(beat.context);
        return interrupt.requested();
    }
};

/// A shared rendering as the agent's result; ownership moves.
pub fn adopt(rendered: web.tools.Rendered) root.Result {
    return .{ .text = rendered.text, .summary = rendered.summary, .is_error = rendered.is_error, .truncated = rendered.truncated };
}
