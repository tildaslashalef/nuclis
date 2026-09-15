//! Colour for the whole executable: the agent's renderer and the one-shot
//! command reports (`style.zig`) both paint through the semantic `Style`
//! enum, never through a colour name. Three layers, each replaceable on its
//! own:
//!
//! 1. **Palette** — named colours (`Palette`), each with its canonical 24-bit
//!    value and the 256-colour approximation published with it. `Name` lists
//!    the palettes this build knows; `agent.theme` selects one.
//! 2. **Role** — what a semantic style asks the palette for: a foreground
//!    slot, a background slot, and attributes. A theme never changes a role,
//!    so it changes colour, never layout.
//! 3. **Level** — how much of the role the terminal can show: truecolor, the
//!    256-colour approximation, the sixteen ANSI colours, or attributes only
//!    (`NO_COLOR`, a dumb terminal). A colour-only role names a substitute
//!    attribute for that last level (`plain`): inline code underlined, a
//!    painted bar reversed.
//!
//! Beside colour is one more axis: the **glyph set**. Every decorative
//! character (bullets, rules, spinner) is named in `Glyphs` with a Unicode
//! and an ASCII table, chosen by the locale's codeset or `NUCLIS_ASCII`. Every
//! SGR sequence is composed at compile time, so `paint` is a table lookup.
const std = @import("std");

pub const Kind = enum { truecolor, c256, c16, plain };

/// Semantic styles the renderer asks for. Mapping them through one role
/// table keeps palette decisions in this file instead of scattered across
/// draw code.
pub const Style = enum {
    header,
    user,
    assistant,
    thinking,
    thinking_header,
    italic,
    dim,
    bold,
    code,
    code_block,
    heading,
    link,
    bullet,
    quote,
    status,
    status_busy,
    status_bg,
    editor_bg,
    accent,
    keyword,
    string,
    comment,
    number,
    error_text,
    /// Command reports (`style.zig`): a good status word, a warning word,
    /// and a field label.
    success,
    warning,
    label,
    /// Agent block kinds the transcript renders: tool call, tool result,
    /// and the structured diff rows.
    tool_call,
    tool_result,
    diff_add,
    diff_remove,
    diff_header,
    /// The bytes that changed within a paired diff line: a reverse video
    /// highlight, so it reads at every colour level (attributes are the one
    /// axis that never goes away).
    diff_change,
    /// An inline pill: the editor's paste chip.
    chip,
    /// The prefill/decode counter in the status bar.
    progress,
    /// The selected row of an inline choice list (picker, completion).
    choice_selected,
};

pub const reset = "\x1b[0m";
/// Restores the default foreground without clearing a painted background.
pub const fg_default = "\x1b[39m";

/// One palette colour: the theme's canonical 24-bit value, the 256-colour
/// approximation published with it, and the ANSI-16 slot it falls back to.
/// The first two are the theme's own values (pinned by a test against the
/// upstream file); `ansi` is *ours*, because a sixteen-colour terminal paints
/// from the user's own palette and the best a theme can do is name the slot
/// whose meaning matches (bright red for red, bright black for a grey).
pub const Color = struct { hex: u24, index: u8, ansi: u8 };

/// The colours a theme names. The shape is Gruvbox's, which every base16-ish
/// palette maps onto: a background scale from hard to light, a foreground
/// scale from light to faint, and eight accents in a bright and a dim
/// variant. A theme fills every slot; roles pick from them.
pub const Palette = struct {
    bg0_h: Color,
    bg0: Color,
    bg0_s: Color,
    bg1: Color,
    bg2: Color,
    bg3: Color,
    bg4: Color,
    fg0: Color,
    fg1: Color,
    fg2: Color,
    fg3: Color,
    fg4: Color,
    gray: Color,
    gray_dim: Color,
    red: Color,
    red_dim: Color,
    green: Color,
    green_dim: Color,
    yellow: Color,
    yellow_dim: Color,
    blue: Color,
    blue_dim: Color,
    purple: Color,
    purple_dim: Color,
    aqua: Color,
    aqua_dim: Color,
    orange: Color,
    orange_dim: Color,
};

/// Which palette slot a role reads.
pub const Slot = std.meta.FieldEnum(Palette);

/// Gruvbox dark, from morhetz/gruvbox (MIT; see THIRD_PARTY_NOTICES.md).
/// The `_dim` accents are its "neutral" variants, the others its "bright"
/// ones; the indices are the community 256-colour approximations published
/// in the same file. The test at the end of this file pins every value.
///
/// The `ansi` column is the sixteen-colour fallback and follows the terminal
/// palette gruvbox ships: the neutral accents take the normal slots (1–6),
/// the bright ones their bright twins (9–14), the greys 0 and 8, the
/// foreground scale 7 and 15. Orange has no ANSI slot of its own and borrows
/// yellow's, which is what every sixteen-colour gruvbox port does. The dark
/// background slots split deliberately: `bg0`/`bg0_h` are the terminal's own
/// black, while the slots that exist *to be seen against it* (the editor box,
/// the status bar, a code block, a chip) take bright black, which is the only
/// way a painted area stays visible with sixteen colours.
pub const gruvbox_dark: Palette = .{
    .bg0_h = .{ .hex = 0x1d2021, .index = 234, .ansi = 0 },
    .bg0 = .{ .hex = 0x282828, .index = 235, .ansi = 0 },
    .bg0_s = .{ .hex = 0x32302f, .index = 236, .ansi = 8 },
    .bg1 = .{ .hex = 0x3c3836, .index = 237, .ansi = 8 },
    .bg2 = .{ .hex = 0x504945, .index = 239, .ansi = 8 },
    .bg3 = .{ .hex = 0x665c54, .index = 241, .ansi = 8 },
    .bg4 = .{ .hex = 0x7c6f64, .index = 243, .ansi = 8 },
    .fg0 = .{ .hex = 0xfbf1c7, .index = 229, .ansi = 15 },
    .fg1 = .{ .hex = 0xebdbb2, .index = 223, .ansi = 15 },
    .fg2 = .{ .hex = 0xd5c4a1, .index = 250, .ansi = 7 },
    .fg3 = .{ .hex = 0xbdae93, .index = 248, .ansi = 7 },
    .fg4 = .{ .hex = 0xa89984, .index = 246, .ansi = 7 },
    .gray = .{ .hex = 0x928374, .index = 245, .ansi = 8 },
    .gray_dim = .{ .hex = 0x928374, .index = 244, .ansi = 8 },
    .red = .{ .hex = 0xfb4934, .index = 167, .ansi = 9 },
    .red_dim = .{ .hex = 0xcc241d, .index = 124, .ansi = 1 },
    .green = .{ .hex = 0xb8bb26, .index = 142, .ansi = 10 },
    .green_dim = .{ .hex = 0x98971a, .index = 106, .ansi = 2 },
    .yellow = .{ .hex = 0xfabd2f, .index = 214, .ansi = 11 },
    .yellow_dim = .{ .hex = 0xd79921, .index = 172, .ansi = 3 },
    .blue = .{ .hex = 0x83a598, .index = 109, .ansi = 12 },
    .blue_dim = .{ .hex = 0x458588, .index = 66, .ansi = 4 },
    .purple = .{ .hex = 0xd3869b, .index = 175, .ansi = 13 },
    .purple_dim = .{ .hex = 0xb16286, .index = 132, .ansi = 5 },
    .aqua = .{ .hex = 0x8ec07c, .index = 108, .ansi = 14 },
    .aqua_dim = .{ .hex = 0x689d6a, .index = 72, .ansi = 6 },
    .orange = .{ .hex = 0xfe8019, .index = 208, .ansi = 11 },
    .orange_dim = .{ .hex = 0xd65d0e, .index = 166, .ansi = 3 },
};

/// The themes this build knows, by their configuration name (`agent.theme`).
/// Adding one is adding a `Palette` and an enum member; the roles, and so
/// the layout, are untouched.
pub const Name = enum {
    @"gruvbox-dark",

    pub fn palette(self: Name) Palette {
        return switch (self) {
            .@"gruvbox-dark" => gruvbox_dark,
        };
    }
};

pub const default_name: Name = .@"gruvbox-dark";

// ----- glyphs -----

/// Every decorative character the surface draws, named once. A renderer asks
/// the theme for a glyph instead of writing the code point, so the ASCII
/// fallback is a table swap rather than a branch at each call site. Widths
/// differ between the tables (`[x]` is three cells where `☑` is one), so
/// layout code measures a glyph with `view.width` and never assumes one cell.
pub const Glyphs = struct {
    /// Horizontal rule, one cell per repetition.
    rule: []const u8,
    /// Unordered list markers, by nesting level.
    bullet: []const u8,
    bullet2: []const u8,
    bullet3: []const u8,
    task_done: []const u8,
    task_todo: []const u8,
    /// Block-quote prefix, drawn before the quoted text.
    quote: []const u8,
    /// Table cell separator and the joint of the header divider.
    table_bar: []const u8,
    table_joint: []const u8,
    /// Truncation indicator ("… 12 lines above").
    ellipsis: []const u8,
    /// Fold arrows for the thinking block.
    fold_open: []const u8,
    fold_closed: []const u8,
    /// Before a settled tool call: the arrow that marks it as the agent
    /// reaching for a tool.
    call: []const u8,
    /// Status-bar decorations: the idle marker, the labels of context,
    /// token counts, prefill, decode, and reasoning effort.
    idle: []const u8,
    context: []const u8,
    tokens: []const u8,
    prefill: []const u8,
    decode: []const u8,
    effort: []const u8,
    /// Frames of the busy spinner, advanced once per repaint.
    spinner: []const []const u8,
};

pub const GlyphSet = enum { unicode, ascii };

pub const unicode_glyphs: Glyphs = .{
    .rule = "─",
    .bullet = "•",
    .bullet2 = "◦",
    .bullet3 = "▪",
    .task_done = "☑",
    .task_todo = "☐",
    .quote = "▌",
    .table_bar = "│",
    .table_joint = "┼",
    .ellipsis = "…",
    .fold_open = "▾",
    .fold_closed = "▸",
    .call = "→",
    .idle = "◆",
    .context = "▤",
    .tokens = "⇅",
    .prefill = "⇤",
    .decode = "⇥",
    .effort = "✦",
    .spinner = &.{ "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" },
};

/// The same surface in ASCII, for a terminal whose locale is not UTF-8.
pub const ascii_glyphs: Glyphs = .{
    .rule = "-",
    .bullet = "*",
    .bullet2 = "-",
    .bullet3 = "+",
    .task_done = "[x]",
    .task_todo = "[ ]",
    .quote = "|",
    .table_bar = "|",
    .table_joint = "+",
    .ellipsis = "...",
    .fold_open = "v",
    .fold_closed = ">",
    .call = "->",
    .idle = "*",
    .context = "#",
    .tokens = "=",
    .prefill = "<",
    .decode = ">",
    .effort = "*",
    .spinner = &.{ "|", "/", "-", "\\" },
};

/// SGR attributes, as a set rather than a sequence: the order they are
/// emitted in is this file's business.
const Attrs = struct {
    bold: bool = false,
    dim: bool = false,
    italic: bool = false,
    underline: bool = false,
    reverse: bool = false,

    fn merge(self: Attrs, other: Attrs) Attrs {
        return .{
            .bold = self.bold or other.bold,
            .dim = self.dim or other.dim,
            .italic = self.italic or other.italic,
            .underline = self.underline or other.underline,
            .reverse = self.reverse or other.reverse,
        };
    }
};

/// What one semantic style asks for, independent of palette and level.
const Role = struct {
    fg: ?Slot = null,
    bg: ?Slot = null,
    attrs: Attrs = .{},
    /// Extra attributes for the attribute-only level, where the colours are
    /// dropped: a role whose meaning is carried by colour alone names a
    /// substitute here (underline for inline code, reverse for a bar).
    plain: Attrs = .{},
};

fn role(style: Style) Role {
    return switch (style) {
        .header => .{ .fg = .aqua, .attrs = .{ .bold = true } },
        .user => .{ .fg = .orange, .attrs = .{ .bold = true } },
        .assistant => .{ .fg = .aqua, .attrs = .{ .bold = true } },
        .thinking => .{ .fg = .fg4, .attrs = .{ .italic = true } },
        .thinking_header => .{ .fg = .green },
        .italic => .{ .attrs = .{ .italic = true } },
        // Dim means *this colour*, not this colour faded. Gruvbox's gray is
        // already a low-contrast foreground; stacking the SGR dim attribute
        // on top of it halves the intensity again and left the help block and
        // every notice barely legible on a dark background (reported
        // 2026-09-12). The attribute now appears only where there is no
        // colour to carry the meaning.
        .dim => .{ .fg = .gray, .plain = .{ .dim = true } },
        .bold => .{ .attrs = .{ .bold = true } },
        .code => .{ .fg = .aqua, .plain = .{ .underline = true } },
        .code_block => .{ .fg = .fg2, .bg = .bg1 },
        .heading => .{ .fg = .orange, .attrs = .{ .bold = true } },
        .link => .{ .fg = .blue, .attrs = .{ .underline = true } },
        .bullet => .{ .fg = .orange, .attrs = .{ .bold = true } },
        .quote => .{ .fg = .purple, .attrs = .{ .italic = true } },
        .status => .{ .fg = .fg1 },
        .status_busy => .{ .fg = .aqua, .attrs = .{ .bold = true } },
        .status_bg => .{ .bg = .bg1, .plain = .{ .reverse = true } },
        .editor_bg => .{ .bg = .bg0_s, .plain = .{ .reverse = true } },
        .accent => .{ .fg = .green },
        .keyword => .{ .fg = .orange, .attrs = .{ .bold = true } },
        .string => .{ .fg = .green },
        .comment => .{ .fg = .gray, .plain = .{ .dim = true } },
        .number => .{ .fg = .purple },
        .error_text => .{ .fg = .red, .attrs = .{ .bold = true } },
        .success => .{ .fg = .green },
        .warning => .{ .fg = .orange },
        .label => .{ .fg = .fg4 },
        .tool_call => .{ .fg = .blue, .attrs = .{ .bold = true } },
        .tool_result => .{ .fg = .fg2 },
        .diff_add => .{ .fg = .green },
        .diff_remove => .{ .fg = .red },
        .diff_header => .{ .fg = .yellow, .attrs = .{ .bold = true } },
        .diff_change => .{ .attrs = .{ .reverse = true } },
        .chip => .{ .fg = .fg1, .bg = .bg2, .plain = .{ .reverse = true } },
        .progress => .{ .fg = .yellow },
        .choice_selected => .{ .fg = .orange, .bg = .bg2, .attrs = .{ .bold = true }, .plain = .{ .reverse = true } },
    };
}

pub const Theme = struct {
    kind: Kind,
    name: Name = default_name,
    glyph_set: GlyphSet = .unicode,

    /// Chooses the strongest colour level the environment advertises with
    /// the default palette. Ghostty sets COLORTERM=truecolor; 256-colour
    /// terminals advertise "256color" in TERM; a plain `xterm` or `vt100`
    /// keeps the sixteen ANSI colours; anything unknown degrades to
    /// attributes only (bold/dim/italic), so the interface stays readable
    /// everywhere. NO_COLOR wins over all of them.
    pub fn detect(environ: *const std.process.Environ.Map) Theme {
        return detectNamed(environ, default_name);
    }

    /// The same detection for a configured palette (`agent.theme`). The
    /// palette never affects the level: a theme cannot turn colour on where
    /// the terminal or the user said no.
    pub fn detectNamed(environ: *const std.process.Environ.Map, name: Name) Theme {
        return .{ .kind = detectKind(environ), .name = name, .glyph_set = detectGlyphs(environ) };
    }

    pub fn paint(self: Theme, style: Style) []const u8 {
        return tables[@intFromEnum(self.name)][@intFromEnum(self.kind)][@intFromEnum(style)];
    }

    /// The glyph table to draw with. A value, not a pointer: `Glyphs` is a
    /// struct of static strings and `Theme` is copied everywhere anyway.
    pub fn glyphs(self: Theme) Glyphs {
        return switch (self.glyph_set) {
            .unicode => unicode_glyphs,
            .ascii => ascii_glyphs,
        };
    }
};

fn detectKind(environ: *const std.process.Environ.Map) Kind {
    if (environ.get("NO_COLOR") != null) return .plain;
    const colorterm = environ.get("COLORTERM") orelse "";
    if (std.mem.eql(u8, colorterm, "truecolor") or std.mem.eql(u8, colorterm, "24bit")) return .truecolor;
    const term = environ.get("TERM") orelse "";
    if (term.len == 0 or std.mem.eql(u8, term, "dumb")) return .plain;
    // "…-256color" and the direct-colour entries advertise the wide palette;
    // everything else that is a terminal at all has the sixteen ANSI slots.
    if (std.ascii.indexOfIgnoreCase(term, "256color") != null or std.ascii.indexOfIgnoreCase(term, "direct") != null) return .c256;
    return .c16;
}

/// UTF-8 is claimed by the locale, never guessed from the terminal: a
/// codeset suffix (`en_US.UTF-8`, `C.utf8`) is the only portable statement a
/// process gets. `NUCLIS_ASCII=1` forces the fallback for a terminal whose
/// font lacks the glyphs even though its locale is fine.
fn detectGlyphs(environ: *const std.process.Environ.Map) GlyphSet {
    if (environ.get("NUCLIS_ASCII")) |value| {
        if (!std.mem.eql(u8, value, "0")) return .ascii;
    }
    const locale = environ.get("LC_ALL") orelse environ.get("LC_CTYPE") orelse environ.get("LANG") orelse "";
    if (std.ascii.indexOfIgnoreCase(locale, "utf-8") != null or std.ascii.indexOfIgnoreCase(locale, "utf8") != null) return .unicode;
    return .ascii;
}

// ----- compile-time table construction -----

const style_count = @typeInfo(Style).@"enum".fields.len;
const Table = [style_count][]const u8;

/// Every (palette, level) pair resolved once at compile time. Indexed by
/// `@intFromEnum` in `paint`, which is why `Name` and `Kind` are dense enums.
const tables = blk: {
    // One `comptimePrint` per colour channel per style per level: the
    // default 1,000 branches is spent long before the first table is done.
    @setEvalBranchQuota(200_000);
    var all: [@typeInfo(Name).@"enum".fields.len][@typeInfo(Kind).@"enum".fields.len]Table = undefined;
    for (std.enums.values(Name)) |name| {
        for (std.enums.values(Kind)) |kind| {
            var table: Table = undefined;
            for (std.enums.values(Style)) |style| table[@intFromEnum(style)] = sgr(name.palette(), kind, role(style));
            all[@intFromEnum(name)][@intFromEnum(kind)] = table;
        }
    }
    break :blk all;
};

fn color(comptime palette: Palette, comptime slot: Slot) Color {
    return @field(palette, @tagName(slot));
}

/// One SGR sequence for a role at a level: attributes first, then the
/// foreground, then the background, all as parameters of a single escape.
/// Empty when nothing is left to say (a colour-only role at the plain level).
fn sgr(comptime palette: Palette, comptime kind: Kind, comptime r: Role) []const u8 {
    comptime {
        var params: []const u8 = "";
        const attrs = if (kind == .plain) r.attrs.merge(r.plain) else r.attrs;
        if (attrs.bold) params = param(params, "1");
        if (attrs.dim) params = param(params, "2");
        if (attrs.italic) params = param(params, "3");
        if (attrs.underline) params = param(params, "4");
        if (attrs.reverse) params = param(params, "7");
        if (kind != .plain) {
            if (r.fg) |slot| params = param(params, channel(color(palette, slot), kind, 38));
            if (r.bg) |slot| params = param(params, channel(color(palette, slot), kind, 48));
        }
        return if (params.len == 0) "" else "\x1b[" ++ params ++ "m";
    }
}

fn param(comptime params: []const u8, comptime next: []const u8) []const u8 {
    return if (params.len == 0) next else params ++ ";" ++ next;
}

/// `38` selects the foreground, `48` the background. The sixteen-colour
/// level does not use those two at all: it names one of the eight normal
/// (30/40) or eight bright (90/100) slots directly.
fn channel(comptime c: Color, comptime kind: Kind, comptime which: u8) []const u8 {
    return switch (kind) {
        .truecolor => std.fmt.comptimePrint("{d};2;{d};{d};{d}", .{ which, c.hex >> 16, (c.hex >> 8) & 0xff, c.hex & 0xff }),
        .c256 => std.fmt.comptimePrint("{d};5;{d}", .{ which, c.index }),
        .c16 => blk: {
            const base: u8 = if (which == 38) (if (c.ansi < 8) 30 else 82) else (if (c.ansi < 8) 40 else 92);
            break :blk std.fmt.comptimePrint("{d}", .{base + c.ansi});
        },
        .plain => unreachable,
    };
}

test "styles resolve to static sequences per support level" {
    const th = Theme{ .kind = .truecolor };
    try std.testing.expectEqualStrings("\x1b[1;38;2;142;192;124m", th.paint(.header));
    try std.testing.expectEqualStrings("\x1b[3;38;2;168;153;132m", th.paint(.thinking));
    try std.testing.expectEqualStrings("\x1b[48;2;60;56;54m", th.paint(.status_bg));
    // Foreground before background, one escape for both.
    try std.testing.expectEqualStrings("\x1b[38;2;213;196;161;48;2;60;56;54m", th.paint(.code_block));
    const dim = Theme{ .kind = .c256 };
    // Dim is a colour, not a faded one: no SGR 2 where there is colour.
    try std.testing.expectEqualStrings("\x1b[38;5;245m", dim.paint(.dim));
    try std.testing.expectEqualStrings("\x1b[38;5;250;48;5;237m", dim.paint(.code_block));
    // Sixteen colours: the normal slots are 30/40, the bright ones 90/100,
    // and neither uses the 38/48 extension.
    const ansi = Theme{ .kind = .c16 };
    try std.testing.expectEqualStrings("\x1b[90m", ansi.paint(.dim)); // gray → bright black
    try std.testing.expectEqualStrings("\x1b[37;100m", ansi.paint(.code_block)); // fg2 on bg1
    try std.testing.expectEqualStrings("\x1b[1;96m", ansi.paint(.header)); // aqua → bright cyan
    try std.testing.expectEqualStrings("\x1b[91m", ansi.paint(.diff_remove)); // red → bright red
    const bare = Theme{ .kind = .plain };
    // …and the attribute is what carries it where there is none.
    try std.testing.expectEqualStrings("\x1b[2m", bare.paint(.dim));
    try std.testing.expectEqualStrings("\x1b[3m", bare.paint(.thinking));
    try std.testing.expectEqualStrings("", bare.paint(.thinking_header));
    // Colour-only roles keep their meaning at the plain level.
    try std.testing.expectEqualStrings("\x1b[4m", bare.paint(.code));
    try std.testing.expectEqualStrings("\x1b[7m", bare.paint(.status_bg));
    try std.testing.expectEqualStrings("\x1b[1;7m", bare.paint(.choice_selected));
    for (std.enums.values(Style)) |style| {
        for (std.enums.values(Kind)) |kind| {
            for (std.enums.values(Name)) |name| {
                _ = (Theme{ .kind = kind, .name = name }).paint(style); // every triple resolves at comptime
            }
        }
    }
    try std.testing.expectEqualStrings("\x1b[0m", reset);
}

test "the gruvbox dark palette matches the canonical values" {
    // morhetz/gruvbox `colors/gruvbox.vim`: truecolor value and the
    // 256-colour approximation of every entry the theme uses. Pinned so a
    // second palette, or an edit to a role, cannot drift them.
    const p = Name.@"gruvbox-dark".palette();
    const expected = [_]struct { Color, []const u8 }{
        .{ p.bg0_h, "1d2021 234" },      .{ p.bg0, "282828 235" },        .{ p.bg0_s, "32302f 236" },
        .{ p.bg1, "3c3836 237" },        .{ p.bg2, "504945 239" },        .{ p.bg3, "665c54 241" },
        .{ p.bg4, "7c6f64 243" },        .{ p.fg0, "fbf1c7 229" },        .{ p.fg1, "ebdbb2 223" },
        .{ p.fg2, "d5c4a1 250" },        .{ p.fg3, "bdae93 248" },        .{ p.fg4, "a89984 246" },
        .{ p.gray, "928374 245" },       .{ p.gray_dim, "928374 244" },   .{ p.red, "fb4934 167" },
        .{ p.red_dim, "cc241d 124" },    .{ p.green, "b8bb26 142" },      .{ p.green_dim, "98971a 106" },
        .{ p.yellow, "fabd2f 214" },     .{ p.yellow_dim, "d79921 172" }, .{ p.blue, "83a598 109" },
        .{ p.blue_dim, "458588 66" },    .{ p.purple, "d3869b 175" },     .{ p.purple_dim, "b16286 132" },
        .{ p.aqua, "8ec07c 108" },       .{ p.aqua_dim, "689d6a 72" },    .{ p.orange, "fe8019 208" },
        .{ p.orange_dim, "d65d0e 166" },
    };
    var buffer: [32]u8 = undefined;
    for (expected) |pair| {
        const printed = try std.fmt.bufPrint(&buffer, "{x:0>6} {d}", .{ pair[0].hex, pair[0].index });
        try std.testing.expectEqualStrings(pair[1], printed);
    }
}

test "detection picks the strongest color level the environment advertises" {
    const a = std.testing.allocator;
    var map = std.process.Environ.Map.init(a);
    defer map.deinit();
    try map.put("COLORTERM", "truecolor");
    try map.put("TERM", "xterm-256color");
    try std.testing.expectEqual(Kind.truecolor, Theme.detect(&map).kind);
    try std.testing.expectEqual(default_name, Theme.detect(&map).name);

    var no_truecolor = std.process.Environ.Map.init(a);
    defer no_truecolor.deinit();
    try no_truecolor.put("TERM", "xterm-256color");
    try std.testing.expectEqual(Kind.c256, Theme.detect(&no_truecolor).kind);
    // A named palette never changes the level.
    try std.testing.expectEqual(Kind.c256, Theme.detectNamed(&no_truecolor, .@"gruvbox-dark").kind);
    // A terminal without the wide palette still has the sixteen ANSI slots.
    try no_truecolor.put("TERM", "xterm");
    try std.testing.expectEqual(Kind.c16, Theme.detect(&no_truecolor).kind);
    try no_truecolor.put("TERM", "screen.xterm-new");
    try std.testing.expectEqual(Kind.c16, Theme.detect(&no_truecolor).kind);
    try no_truecolor.put("TERM", "xterm-direct");
    try std.testing.expectEqual(Kind.c256, Theme.detect(&no_truecolor).kind);

    try no_truecolor.put("NO_COLOR", "1");
    try std.testing.expectEqual(Kind.plain, Theme.detect(&no_truecolor).kind);

    var empty = std.process.Environ.Map.init(a);
    defer empty.deinit();
    try std.testing.expectEqual(Kind.plain, Theme.detect(&empty).kind);
}

test "the glyph set follows the locale, and NUCLIS_ASCII forces it" {
    const a = std.testing.allocator;
    var map = std.process.Environ.Map.init(a);
    defer map.deinit();
    // No locale at all: a process that cannot claim UTF-8 does not draw it.
    try std.testing.expectEqual(GlyphSet.ascii, Theme.detect(&map).glyph_set);
    try map.put("LANG", "en_US.UTF-8");
    try std.testing.expectEqual(GlyphSet.unicode, Theme.detect(&map).glyph_set);
    try map.put("LC_ALL", "C");
    try std.testing.expectEqual(GlyphSet.ascii, Theme.detect(&map).glyph_set); // LC_ALL wins
    try map.put("LC_ALL", "C.utf8");
    try std.testing.expectEqual(GlyphSet.unicode, Theme.detect(&map).glyph_set);
    try map.put("NUCLIS_ASCII", "1");
    try std.testing.expectEqual(GlyphSet.ascii, Theme.detect(&map).glyph_set);
    try map.put("NUCLIS_ASCII", "0");
    try std.testing.expectEqual(GlyphSet.unicode, Theme.detect(&map).glyph_set);
    // The glyph set is independent of the colour level in both directions.
    try map.put("NO_COLOR", "1");
    try std.testing.expectEqual(GlyphSet.unicode, Theme.detect(&map).glyph_set);
    try std.testing.expectEqual(Kind.plain, Theme.detect(&map).kind);
}

test "every glyph has an ASCII counterpart and none of them is empty" {
    // A missing fallback would print nothing where the Unicode table draws a
    // bullet, so the tables are checked field by field rather than by eye.
    inline for (@typeInfo(Glyphs).@"struct".fields) |field| {
        const uni = @field(unicode_glyphs, field.name);
        const ascii = @field(ascii_glyphs, field.name);
        if (field.type == []const []const u8) {
            try std.testing.expect(uni.len > 0 and ascii.len > 0);
            for (ascii) |frame| try std.testing.expect(frame.len > 0 and frame.len < 3);
        } else {
            try std.testing.expect(uni.len > 0 and ascii.len > 0);
            // The fallback must be printable ASCII, which is the whole point.
            for (ascii) |byte| try std.testing.expect(byte >= 0x20 and byte < 0x7f);
        }
    }
    const th = Theme{ .kind = .plain, .glyph_set = .ascii };
    try std.testing.expectEqualStrings("[x]", th.glyphs().task_done);
    try std.testing.expectEqualStrings("☑", (Theme{ .kind = .plain }).glyphs().task_done);
}
