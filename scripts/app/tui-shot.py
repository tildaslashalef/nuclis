#!/usr/bin/env python3
"""The screenshot harness: drives `nuclis agent` in a Rex terminal and
captures its screen, so the surface can be looked at without a person at
the keyboard.

The terminal is Rex's own, on the Rex server, with no multiplexer between
it and the agent: what it captures is what Rex shows, kitty graphics and
all (rexterm.py, beside this file). It needs no window; the session shows
up in the Rex app's sidebar to be watched. The block is pinned to the size
asked for and to fixed default colours, so captures do not depend on the
app's theme. Each capture is written under .zig-cache/tui/ as `<name>.txt`
(the text), `<name>.ansi` (the escape stream that redraws it),
`<name>.tagged.txt` (every styled run as `[fg=#hex bg=#hex bold]…[/]`,
readable without a terminal) and `<name>.html` (Rex's styled page); with
`--png`, also `<name>.png`, that page rendered by headless Chrome. Images
drawn with kitty graphics are not in any capture, only in the live
terminal. A burst captures at a fixed rate for a while and reports which
frames changed and the interval between them, which is how an animation's
cadence is measured.

Steps, in order, as arguments:
  keys=<text>            type the text (literally; no key names)
  key=<name>             one key in Rex's syntax: enter, tab, escape, ctrl+c,
                         alt+enter, up (tmux's names, C-c or Up, also work)
  enter                  the same as key=enter
  wait=<seconds>         sleep
  capture=<name>         capture the screen now
  burst=<name>,<s>,<hz>  capture every 1/hz seconds for s seconds
  until=<text>,<s>       wait up to s seconds for the text to appear on screen,
                         and print how long it took (to 0.05 s)
  buffer=<name>          save the newest OSC 52 copy to <name>.buffer.txt
  paste=<text>           a bracketed paste of the text (a dropped file's path)
  resize=<cols>x<rows>   resize the terminal, as a window resize does

The session is left running unless `--stop` is given: open it from the Rex
sidebar, or read it with `rex capture -s nuclis-<session>`. See
docs/development.md § Looking at the agent without a person at the keyboard.
"""

import argparse
import json
import re
import sys
import time
import unittest
from typing import Any
from pathlib import Path

from rexterm import Terminal, html_page, png, rex_key

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / ".zig-cache" / "tui"
DEFAULT_COMMAND = "./zig-out/bin/nuclis agent"

# ---- the SGR tagger (pure; covered by --self-test) --------------------------

BASIC = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
SGR = re.compile(r"\x1b\[([0-9;:]*)m")
OTHER_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")


def _color(params, i):
    """Parses the colour after a 38/48 at params[i]; returns (name, consumed)."""
    if i < len(params) and params[i] == "5" and i + 1 < len(params):
        return "c%s" % params[i + 1], 2
    if i < len(params) and params[i] == "2" and i + 3 < len(params):
        r, g, b = (int(x or 0) for x in params[i + 1 : i + 4])
        return "#%02x%02x%02x" % (r, g, b), 4
    return None, 0


def apply_sgr(state, params):
    """Folds one SGR parameter list into `state` (a dict) and returns it."""
    p = [x for x in params.replace(":", ";").split(";")] if params else ["0"]
    i = 0
    while i < len(p):
        n = int(p[i] or 0)
        i += 1
        if n == 0:
            state.clear()
        elif n in (1, 2, 3, 4, 7, 9):
            state[{1: "bold", 2: "dim", 3: "italic", 4: "underline", 7: "reverse", 9: "strike"}[n]] = True
        elif n == 22:
            state.pop("bold", None)
            state.pop("dim", None)
        elif n in (23, 24, 27, 29):
            state.pop({23: "italic", 24: "underline", 27: "reverse", 29: "strike"}[n], None)
        elif 30 <= n <= 37:
            state["fg"] = BASIC[n - 30]
        elif 90 <= n <= 97:
            state["fg"] = "bright-" + BASIC[n - 90]
        elif n == 39:
            state.pop("fg", None)
        elif 40 <= n <= 47:
            state["bg"] = BASIC[n - 40]
        elif 100 <= n <= 107:
            state["bg"] = "bright-" + BASIC[n - 100]
        elif n == 49:
            state.pop("bg", None)
        elif n in (38, 48):
            name, used = _color(p, i)
            if name:
                state["fg" if n == 38 else "bg"] = name
            i += used
    return state


def tag_of(state):
    parts = []
    for key in ("fg", "bg"):
        if key in state:
            parts.append("%s=%s" % (key, state[key]))
    for flag in ("bold", "dim", "italic", "underline", "reverse", "strike"):
        if state.get(flag):
            parts.append(flag)
    return "[" + " ".join(parts) + "]" if parts else ""


def tagged(ansi):
    """The escape stream as text with `[attrs]…[/]` around every styled run.
    A tag is emitted only when the style changes and only before visible
    text, so an unstyled line is unchanged."""
    out = []
    state = {}
    open_tag = ""
    pending = ""
    pos = 0
    for line in ansi.split("\n"):
        pos = 0
        while pos < len(line):
            m = SGR.match(line, pos)
            if m:
                apply_sgr(state, m.group(1))
                pending = tag_of(state)
                pos = m.end()
                continue
            m = OTHER_ESCAPE.match(line, pos)
            if m:
                pos = m.end()
                continue
            if pending != open_tag:
                if open_tag:
                    out.append("[/]")
                if pending:
                    out.append(pending)
                open_tag = pending
            out.append(line[pos])
            pos += 1
        if open_tag:
            out.append("[/]")
            open_tag = ""
        pending = tag_of(state)
        out.append("\n")
    return "".join(out).rstrip("\n") + "\n"


def plain(ansi):
    return OTHER_ESCAPE.sub("", SGR.sub("", ansi))


# ---- the run ----------------------------------------------------------------


def write_capture(out_dir, name, ansi):
    (out_dir / (name + ".ansi")).write_text(ansi)
    (out_dir / (name + ".txt")).write_text(plain(ansi))
    (out_dir / (name + ".tagged.txt")).write_text(tagged(ansi))


def screen_html(fragment, rows):
    """Rex's HTML capture cut to its last `rows` lines. The capture is one
    pre-formatted block with the screen's trailing blank rows left out; a
    styled run never spans lines."""
    head, sep, body = fragment.partition(">")
    body = body.rstrip("\n")
    if body.endswith("</div>"):
        body = body[: -len("</div>")]
    body = body.rstrip("\n")
    return head + sep + ("\n".join(body.split("\n")[-rows:]) if rows else "") + "</div>"


def shown_rows(ansi):
    """How many of the screen's rows are left once its trailing blank rows
    are dropped, as Rex's HTML capture drops them."""
    lines = plain(ansi).split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return len(lines)


def write_page(session, out_dir, name, ansi, with_png):
    page = out_dir / (name + ".html")
    shown = shown_rows(ansi)
    page.write_text(html_page(screen_html(session.capture("html"), shown), session.columns, session.rows))
    if with_png and not png(page.resolve(), out_dir / (name + ".png"), session.columns, session.rows):
        print("png: no headless Chrome found, or it failed; the text captures continue", file=sys.stderr)


def burst(session, out_dir, name, seconds, hz):
    frames = []
    interval = 1.0 / hz
    start = time.perf_counter()
    next_at = start
    while time.perf_counter() - start < seconds:
        now = time.perf_counter()
        if now < next_at:
            time.sleep(next_at - now)
        frames.append((time.perf_counter() - start, session.screen()))
        next_at += interval
    changed = [i for i in range(1, len(frames)) if frames[i][1] != frames[i - 1][1]]
    for i in [0] + changed:
        write_capture(out_dir, "%s-%02d" % (name, i), frames[i][1])
    gaps = [frames[changed[k]][0] - frames[changed[k - 1]][0] for k in range(1, len(changed))]
    mean = sum(gaps) / len(gaps) if gaps else None
    report = {
        "name": name,
        "frames": len(frames),
        "seconds": seconds,
        "hz": hz,
        "changed": [round(frames[i][0], 3) for i in changed],
        "mean_interval_s": round(mean, 3) if mean is not None else None,
    }
    (out_dir / (name + ".burst.json")).write_text(json.dumps(report, indent=1) + "\n")
    print(
        "burst %s: %d frames, %d changed, mean interval %s"
        % (name, len(frames), len(changed), ("%.3fs" % mean) if mean is not None else "n/a")
    )


def wait_for(session, text, seconds):
    started = time.time()
    deadline = started + seconds
    while time.time() < deadline:
        if text in plain(session.screen()):
            # Measured from the step's start, at the polling step's resolution.
            print("until %r: seen after %.2fs" % (text, time.time() - started))
            return True
        time.sleep(0.05)
    print("until: %r not seen within %ss" % (text, seconds), file=sys.stderr)
    return False


def parse_step(arg) -> tuple[str, Any]:
    if arg == "enter":
        return ("key", "enter")
    if "=" not in arg:
        raise SystemExit("bad step: %s" % arg)
    kind, value = arg.split("=", 1)
    if kind in ("keys", "key", "capture", "buffer", "paste"):
        return (kind, value)
    if kind == "wait":
        return (kind, float(value))
    if kind == "resize":
        columns, rows = (int(x) for x in value.lower().split("x"))
        return (kind, (columns, rows))
    if kind == "burst":
        name, seconds, hz = value.split(",")
        return (kind, (name, float(seconds), float(hz)))
    if kind == "until":
        text, seconds = value.rsplit(",", 1)
        return (kind, (text, float(seconds)))
    raise SystemExit("bad step: %s" % arg)


def run(args):
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    columns, rows = (int(x) for x in args.size.lower().split("x"))
    command = args.command
    if command.startswith("./"):
        # A repository-relative binary, whatever --cwd the app runs in.
        head, _, tail = command.partition(" ")
        command = str(ROOT / head[2:]) + (" " + tail if tail else "")
    name = "nuclis-" + args.session
    session = Terminal(name, (columns, rows), command, ROOT / args.cwd, env=args.env)
    session.start()
    ok = True
    for step in [parse_step(s) for s in args.steps]:
        kind, value = step
        if kind == "keys":
            session.send_text(value)
        elif kind == "key":
            session.send_key(value)
        elif kind == "paste":
            session.paste(value)
        elif kind == "wait":
            time.sleep(value)
        elif kind == "resize":
            session.resize(*value)
        elif kind == "capture":
            ansi = session.screen()
            write_capture(out_dir, value, ansi)
            write_page(session, out_dir, value, ansi, args.png)
            print("capture %s" % value)
        elif kind == "burst":
            burst(session, out_dir, *value)
        elif kind == "until":
            ok = wait_for(session, *value) and ok
        elif kind == "buffer":
            (out_dir / (value + ".buffer.txt")).write_text(session.buffer())
            print("buffer %s" % value)
    if args.stop:
        session.stop()
    else:
        print("session left running: %s in the Rex sidebar; rex capture -s %s; rex kill %s" % (name, name, name))
    return 0 if ok else 1


# ---- self-test ----------------------------------------------------------------


class TaggerTest(unittest.TestCase):
    def test_truecolor_and_reset(self):
        self.assertEqual(tagged("a\x1b[38;2;255;0;0mred\x1b[0mb\n"), "a[fg=#ff0000]red[/]b\n")

    def test_bold_dim_and_background(self):
        self.assertEqual(tagged("\x1b[1m\x1b[48;5;236mx\x1b[22my\x1b[m\n"), "[bg=c236 bold]x[/][bg=c236]y[/]\n")

    def test_basic_colors_and_other_escapes(self):
        self.assertEqual(tagged("\x1b[32m\x1b[Kok\x1b[39m done\n"), "[fg=green]ok[/] done\n")
        self.assertEqual(plain("\x1b[32mok\x1b[0m\x1b]8;;http://x\x1b\\l\x1b]8;;\x1b\\"), "okl")

    def test_style_carries_across_lines(self):
        self.assertEqual(tagged("\x1b[2ma\nb\x1b[0m\n"), "[dim]a[/]\n[dim]b[/]\n")

    def test_empty_and_unstyled(self):
        self.assertEqual(tagged("plain\n\nrows\n"), "plain\n\nrows\n")

    def test_screen_html(self):
        page = '<div style="white-space: pre;">a\nb <div style="display: inline;">c</div>\nd\n</div>'
        self.assertEqual(
            screen_html(page, 2), '<div style="white-space: pre;">b <div style="display: inline;">c</div>\nd</div>'
        )

    def test_shown_rows(self):
        self.assertEqual(shown_rows("a\n\x1b[31mb\x1b[0m\n  \n\n"), 2)
        self.assertEqual(shown_rows("\n\n"), 0)

    def test_keys(self):
        self.assertEqual(rex_key("C-c"), "ctrl+c")
        self.assertEqual(rex_key("M-Enter"), "alt+enter")
        self.assertEqual(rex_key("Up"), "up")
        self.assertEqual(rex_key("ctrl+o"), "ctrl+o")
        self.assertEqual(rex_key("x"), "x")

    def test_steps(self):
        self.assertEqual(parse_step("enter"), ("key", "enter"))
        self.assertEqual(parse_step("burst=warm,30,10"), ("burst", ("warm", 30.0, 10.0)))
        self.assertEqual(parse_step("until=ready,40"), ("until", ("ready", 40.0)))
        self.assertEqual(parse_step("keys=a=b"), ("keys", "a=b"))
        self.assertEqual(parse_step("buffer=clip"), ("buffer", "clip"))
        self.assertEqual(parse_step("resize=100x30"), ("resize", (100, 30)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="the steps, in order")
    ap.add_argument("--size", default="160x45", help="columns x rows (default 160x45)")
    ap.add_argument("--session", default="shot", help="Rex session name, after nuclis- (default shot)")
    ap.add_argument("--command", default=DEFAULT_COMMAND, help="what to run (default: %s)" % DEFAULT_COMMAND)
    ap.add_argument("--cwd", default=".", help="working directory of the command, relative to the repository")
    ap.add_argument("--out", default=str(OUT_DIR.relative_to(ROOT)), help="where captures go")
    ap.add_argument("--png", action="store_true", help="also render each capture to <name>.png (headless Chrome)")
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an environment variable for the command (repeatable)",
    )
    ap.add_argument("--stop", action="store_true", help="kill the Rex session at the end")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TaggerTest)
        return 0 if unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful() else 1
    if not args.steps:
        ap.error("no steps given")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
