//! `src/tui/` — the terminal surface of `nuclis agent`, as a module of the
//! executable rather than a package (docs/spec.md § Decisions). The rule that
//! defines it: **this module imports nothing from `inference`**. It knows
//! about rows, cells, keys, colours, and escape sequences; it knows nothing
//! about tokens, sessions, models, or prompts. `src/agent/` owns those and
//! drives the surface.
//!
//! What lives here today:
//!
//! | Module | Role |
//! | --- | --- |
//! | `terminal` | the terminal lease: raw mode, kitty keys, bracketed paste, size |
//! | `screen` | the live region: repaint, insertion above it, cursor geometry |
//! | `editor` | the prompt buffer: word wrap, paste chips, history, layout |
//! | `event` | the typed events a turn is made of: the producer/consumer seam |
//! | `transcript` | blocks built from events; each closed one written once |
//! | `status` | the instrumented bar, as a value with a pure `paint` |
//! | `choice` | an inline list with a selection: pickers and completion |
//! | `keys` | pure decoding of input bytes into `Key` values |
//! | `view` | display width, wrapping, escape-aware truncation, control-byte sanitization |
//! | `markdown` | small markdown into pre-styled, pre-wrapped rows |
//! | `highlight` | the heuristic code highlighter `markdown` calls |
//! | `theme` | named palettes behind a semantic style enum |
//! | `style` | the same palette applied to one-shot command reports |
//!
//! Only `terminal` touches the OS (termios, `poll`, the window-size ioctl);
//! `screen` writes to a plain `std.Io.Writer` and everything else is a value
//! type over an allocator, which is what makes the golden tests possible
//! without a TTY — including the tests that pin the escape stream itself.
//!
//! Zig note. A `root.zig` that only re-exports is the module's public face:
//! importers say `tui.view.lines(...)` instead of reaching into a file path,
//! and the `test` block below pulls every file's tests into `zig build test`
//! (Zig only compiles what is referenced, so an unreferenced file's tests
//! would silently never run).
pub const terminal = @import("terminal.zig");
pub const screen = @import("screen.zig");
pub const editor = @import("editor.zig");
pub const transcript = @import("transcript.zig");
pub const status = @import("status.zig");
pub const choice = @import("choice.zig");
pub const event = @import("event.zig");
pub const keys = @import("keys.zig");
pub const view = @import("view.zig");
pub const graphemes = @import("graphemes.zig");
pub const markdown = @import("markdown.zig");
pub const highlight = @import("highlight.zig");
pub const diff = @import("diff.zig");
pub const theme = @import("theme.zig");
pub const style = @import("style.zig");
pub const banner = @import("banner.zig");

test {
    _ = terminal;
    _ = screen;
    _ = editor;
    _ = transcript;
    _ = status;
    _ = choice;
    _ = event;
    _ = keys;
    _ = view;
    _ = graphemes;
    _ = markdown;
    _ = highlight;
    _ = diff;
    _ = theme;
    _ = style;
    _ = banner;
}
