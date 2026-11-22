"""A command in a Rex terminal, driven and read through the `rex` CLI: what
the screenshot harness (tui-shot.py) and the README's GIF (agent-demo.py)
run the agent in.

Rex runs the terminal on its server, so it needs no window: a session made
here shows up in the Rex app's sidebar, to be watched or typed into, but
nothing is opened or focused. The block is pinned to the requested size and
default colours, so captures are the same on every machine and in every
app theme. The terminal is Rex's own (TERM_PROGRAM=rex, truecolor, kitty
graphics), with no multiplexer in between. Needs the `rex` CLI on PATH.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

# The default colours and the sixteen-colour palette the block is pinned to
# (Gruvbox dark, hard); 16-255 are xterm's.
FOREGROUND = "#ebdbb2"
BACKGROUND = "#1d2021"
PALETTE16 = [
    "#282828", "#cc241d", "#98971a", "#d79921", "#458588", "#b16286", "#689d6a", "#a89984",
    "#928374", "#fb4934", "#b8bb26", "#fabd2f", "#83a598", "#d3869b", "#8ec07c", "#ebdbb2",
]  # fmt: skip


def palette():
    """All 256 colours as #rrggbb: the pinned sixteen, xterm's cube and greys."""
    steps = [0, 95, 135, 175, 215, 255]
    cube = ["#%02x%02x%02x" % (steps[i // 36], steps[i // 6 % 6], steps[i % 6]) for i in range(216)]
    greys = ["#%02x%02x%02x" % ((8 + 10 * i,) * 3) for i in range(24)]
    return PALETTE16 + cube + greys


# tmux key names, still accepted, to Rex's key syntax.
TMUX_KEYS = {
    "Enter": "enter",
    "Tab": "tab",
    "BTab": "shift+tab",
    "Escape": "escape",
    "BSpace": "backspace",
    "Space": "space",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "Home": "home",
    "End": "end",
    "PageUp": "page_up",
    "PageDown": "page_down",
}


def rex_key(name):
    """A key as Rex's `send-key` spells it: `ctrl+c`, `alt+enter`, `up`.
    tmux's names (`C-c`, `M-Enter`, `Up`) are translated."""
    mods = []
    while len(name) > 2 and name[1] == "-" and name[0] in "CMS":
        mods.append({"C": "ctrl", "M": "alt", "S": "shift"}[name[0]])
        name = name[2:]
    key = TMUX_KEYS.get(name, name if len(name) == 1 else name.lower())
    return "+".join(mods + [key])


def rex(*args, check=True):
    if shutil.which("rex") is None:
        raise SystemExit("rex: not on PATH (the Rex app's CLI; see docs/development.md)")
    r = subprocess.run(["rex", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit("rex %s: %s" % (args[0], (r.stderr or r.stdout).strip()))
    return r


class Terminal:
    """One Rex session holding one terminal block that runs `command` (a
    shell command line) in `cwd`. The session's label is `name`."""

    def __init__(self, name, size, command, cwd, env=()):
        self.name = name
        self.columns, self.rows = size
        self.command = command
        self.cwd = cwd
        self.env = list(env)
        self.block = None

    def start(self):
        rex("kill", self.name, check=False)
        # The environment is set here, not by a login shell, so a run sees
        # this process's PATH whatever the account's shell config does.
        env = ["PATH=%s" % os.environ.get("PATH", ""), "COLORTERM=truecolor", "NUCLIS_NO_NOTIFY=1"] + self.env
        argv = ["/usr/bin/env", *env, "/bin/sh", "-c", "exec " + self.command]
        # --keep-open keeps the last screen (a panic and its trace) readable.
        out = rex("new", self.name, "-c", str(self.cwd), "--keep-open", "--shell", "none", "--json", "--", *argv).stdout
        self.block = json.loads(out)["initial_windows"][0]["block_ids"][0]
        self.call(
            "set_theme",
            {
                "foreground": FOREGROUND,
                "background": BACKGROUND,
                "palette": ["%d=%s" % (i, c) for i, c in enumerate(PALETTE16)],
                "scheme": "dark",
            },
        )
        self.resize(self.columns, self.rows)

    def resize(self, columns, rows):
        """Resizes the terminal, as a window resize does; the program gets
        its SIGWINCH."""
        self.columns, self.rows = columns, rows
        self.call("resize", {"columns": columns, "rows": rows})
        # The size applies at once while no app has taken the block over.
        size = self.call("size")
        if (size.get("columns"), size.get("rows")) != (self.columns, self.rows):
            print(
                "warning: terminal is %sx%s, wanted %dx%d (an app showing the session owns its size)"
                % (size.get("columns"), size.get("rows"), self.columns, self.rows),
                file=sys.stderr,
            )

    def call(self, method, args=None):
        cmd = ["block", "call", "-b", self.block, method]
        if args is not None:
            cmd.append(json.dumps(args))
        out = rex(*cmd).stdout.strip()
        return json.loads(out) if out else {}

    def send_text(self, text):
        self.call("write", {"data": text})

    def send_key(self, key):
        rex("send-key", "-b", self.block, rex_key(key))

    def paste(self, text):
        """A bracketed paste, as a drop or Cmd-V arrives; the agent asked
        for the markers, so it reads them."""
        self.send_text("\x1b[200~" + text + "\x1b[201~")

    def capture(self, fmt="vt"):
        """The screen and scrollback, in `text`, `vt` or `html`."""
        return rex("capture", "-b", self.block, "--format", fmt).stdout

    def screen(self):
        """The visible screen with its styles: the last `rows` lines of the
        capture, which always ends with the whole screen."""
        lines = self.capture("vt").replace("\r\n", "\n").rstrip("\n").split("\n")
        return "\n".join(lines[-self.rows :]) + "\n"

    def buffer(self):
        """The newest text a program copied with OSC 52, or empty when
        there is none (the server keeps a copy for about a minute)."""
        r = rex("block", "call", "-b", self.block, "clipboard_read", check=False)
        if r.returncode != 0:
            return ""
        return json.loads(r.stdout).get("data", "")

    def exited(self):
        return self.call("process").get("last_exit") is not None

    def stop(self):
        rex("kill", self.name, check=False)


def html_page(fragment, columns, rows):
    """Rex's HTML capture as a page in the block's colours, `columns` wide.
    The capture names palette colours as `--vt-palette-N` variables."""
    variables = "".join("--vt-palette-%d:%s;" % (i, c) for i, c in enumerate(palette()))
    return (
        "<!doctype html><meta charset=utf-8><style>"
        ":root{" + variables + "}"
        "html,body{margin:0;background:%s;color:%s}"
        "body>div{padding:12px;width:%dch;height:%dem;overflow:hidden;"
        "font:14px/1.25 Menlo,'SF Mono',monospace}"
        "</style><body>%s</body>" % (BACKGROUND, FOREGROUND, columns, rows * 1.25, fragment)
    )


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def chrome():
    return CHROME if os.path.exists(CHROME) else shutil.which("google-chrome") or shutil.which("chromium")


def png(html_path, png_path, columns, rows):
    """Renders the page to a PNG with headless Chrome; False without one."""
    browser = chrome()
    if browser is None:
        return False
    # Menlo at 14px is 8.43px a column; the margin is the page's padding.
    width = int(columns * 8.43) + 24 + 8
    height = int(rows * 14 * 1.25) + 24 + 8
    # A profile of its own, so a running Chrome's is never touched.
    profile = tempfile.mkdtemp(prefix="nuclis-shot-chrome-")
    r = subprocess.run(
        [
            browser,
            "--headless",
            "--user-data-dir=%s" % profile,
            "--no-first-run",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            "--window-size=%d,%d" % (width, height),
            "--screenshot=%s" % png_path,
            html_path.as_uri(),
        ],
        capture_output=True,
        text=True,
    )
    shutil.rmtree(profile, ignore_errors=True)
    return r.returncode == 0 and png_path.exists()
