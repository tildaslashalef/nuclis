#!/usr/bin/env python3
"""Record the README's GIF: `nuclis agent` with Qwen3.8-27B fixing a failing
test in a scratch clone of the generated playground (scripts/app/playground.py).

The shell runs in a Rex terminal (rexterm.py, the one the screenshot
harness uses) under this script's `--record` mode: a pty between the
terminal and the shell that writes everything the shell's side prints into
an asciicast v2 file, from the first clear of the screen on. `agg` renders
that to docs/media/agent.gif. vhs's browser terminal stalls the TUI's
frames, so it is not used. Needs the `rex` CLI, agg (`brew install agg`),
and a release build in zig-out/. Run from anywhere:
`python3 scripts/app/agent-demo.py`.
"""

import argparse
import codecs
import fcntl
import json
import os
import pathlib
import pty
import re
import select
import signal
import subprocess
import sys
import termios
import time
import tty

import playground
from rexterm import Terminal

ROOT = pathlib.Path(__file__).resolve().parents[2]
NUCLIS = ROOT / "zig-out/bin/nuclis"
WORK = pathlib.Path("/tmp/playground")
CAST = ROOT / ".zig-cache/demo/agent.cast"
GIF = ROOT / "docs/media/agent.gif"
COLUMNS, ROWS = 110, 32
TASK = "The tests fail. Find the bug, fix it, and run the tests again."
TERM: Terminal | None = None  # the Rex terminal main() starts


def term() -> Terminal:
    assert TERM is not None, "main() starts the terminal first"
    return TERM


CLEAR = b"\x1b[2J"


def record(cast, columns, rows, argv):
    """Runs inside the Rex terminal: `argv` on a pty sized like the
    terminal, the terminal's keys to it, its output to the terminal and,
    from its first clear of the screen on, to `cast` as asciicast v2."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(argv[0], argv)

    def winch(*_):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, fcntl.ioctl(0, termios.TIOCGWINSZ, b"\0" * 8))

    signal.signal(signal.SIGWINCH, winch)
    winch()
    saved = termios.tcgetattr(0)
    tty.setraw(0)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    start = None
    with open(cast, "w") as out:
        while True:
            try:
                ready, _, _ = select.select([0, fd], [], [])
            except InterruptedError:
                continue
            if fd in ready:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    data = b""
                if not data:
                    break
                os.write(1, data)
                if start is None and CLEAR in data:
                    data = b"\x1b[H" + data[data.index(CLEAR) :]
                    start = time.monotonic()
                    header = {"version": 2, "width": columns, "height": rows, "timestamp": int(time.time())}
                    header["env"] = {"TERM": "xterm-256color", "SHELL": "/bin/bash"}
                    out.write(json.dumps(header) + "\n")
                if start is not None:
                    text = decoder.decode(data)
                    if text:
                        out.write(json.dumps([round(time.monotonic() - start, 4), "o", text]) + "\n")
                        out.flush()
            if 0 in ready:
                data = os.read(0, 65536)
                if data:
                    os.write(fd, data)
    os.waitpid(pid, 0)
    termios.tcsetattr(0, termios.TCSAFLUSH, saved)


def screen():
    return term().capture("text")


def until(pattern, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if re.search(pattern, screen()):
            return
        time.sleep(0.5)
    sys.exit("agent-demo: %r not seen within %ss" % (pattern, seconds))


def type_text(text, delay=0.035):
    for ch in text:
        term().send_text(ch)
        time.sleep(delay)


def enter():
    term().send_key("enter")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", metavar="CAST", help=argparse.SUPPRESS)
    parser.add_argument("--speed", default="1", help="agg playback speed (default 1, real time)")
    parser.add_argument("--idle", default="1.5", help="agg idle-time limit in seconds (default 1.5)")
    if "--record" in sys.argv:
        at = sys.argv.index("--record")
        return record(sys.argv[at + 1], COLUMNS, ROWS, sys.argv[sys.argv.index("--", at) + 1 :])
    args = parser.parse_args()

    # The scratch project: the playground with rect_perimeter broken and
    # committed, so the agent's fix is the only diff.
    subprocess.run(["rm", "-rf", str(WORK)], check=True)
    subprocess.run(["git", "clone", "-q", str(playground.ensure()), str(WORK)], check=True)
    rect = WORK / "src/shapes/rect.py"
    rect.write_text(rect.read_text().replace("return 2 * (width + height)", "return width + height"))
    subprocess.run(["git", "-C", str(WORK), "commit", "-qam", "demo: a broken perimeter"], check=True)

    CAST.parent.mkdir(parents=True, exist_ok=True)
    GIF.parent.mkdir(parents=True, exist_ok=True)
    global TERM
    # PS1 is set on bash itself: the /bin/sh in between drops it.
    shell = "%s %s --record %s -- env 'PS1=$ ' bash --norc --noprofile" % (
        sys.executable,
        pathlib.Path(__file__).resolve(),
        CAST,
    )
    env = ["PATH=%s:%s" % (NUCLIS.parent, os.environ.get("PATH", ""))]
    TERM = Terminal("nuclis-demo", (COLUMNS, ROWS), shell, WORK, env=env)
    TERM.start()
    time.sleep(1)
    try:
        # Clear inside the recording so the cast starts from a blank screen.
        type_text("clear", 0)
        enter()
        time.sleep(1)
        type_text("make test 2>&1 | tail -3")
        enter()
        time.sleep(2.5)
        type_text("nuclis agent")
        enter()
        until(r"◆ ready", 180)
        time.sleep(1.5)
        type_text(TASK)
        time.sleep(0.5)
        enter()
        time.sleep(10)
        until(r"◆ ready", 900)
        time.sleep(4)
        TERM.send_key("ctrl+c")
        time.sleep(1.5)
        # A fresh screen for the ending; no pager, so the diff is plain text.
        type_text("clear", 0)
        enter()
        time.sleep(0.5)
        type_text("git --no-pager diff --color")
        enter()
        time.sleep(3)
        type_text("make test 2>&1 | tail -3")
        enter()
        time.sleep(4)
    finally:
        TERM.stop()
    subprocess.run(
        [
            "agg",
            "--speed",
            args.speed,
            "--idle-time-limit",
            args.idle,
            "--font-size",
            "16",
            "--last-frame-duration",
            "4",
            str(CAST),
            str(GIF),
        ],
        check=True,
    )
    print("wrote %s (%.1f MB)" % (GIF, GIF.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
