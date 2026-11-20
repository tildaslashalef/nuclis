//! The command-output kit: every command's text report renders through it,
//! so all of them share one look (a header line, aligned detail sections,
//! width-aware tables, status glyphs and words, counted nouns, one size and
//! duration format). The design follows artoo's CLI;
//! docs/app/command-output.md states the rules. The agent's chat surface (`tui/`) has its own
//! renderer and does not use this.
//!
//! A command takes a `Ui` (from `Ui.detect` on the stream it writes) and
//! passes it to the renderers; `Ui.plain` is the uncoloured, unbounded form
//! the tests pin and `--json` never needs.

pub const style = @import("ui/style.zig");
pub const text = @import("ui/text.zig");
pub const values = @import("ui/values.zig");
pub const table = @import("ui/table.zig");
pub const blocks = @import("ui/blocks.zig");
pub const Ui = @import("ui/Ui.zig");
pub const Step = @import("ui/Step.zig");

test {
    _ = style;
    _ = text;
    _ = values;
    _ = table;
    _ = blocks;
    _ = Ui;
    _ = Step;
}
