#!/usr/bin/env python3
"""Record the README's GIF: `nuclis agent` with Qwen3.8-27B fixing a failing
test in a scratch clone of the generated playground (scripts/playground.py).

The agent runs in a detached tmux session (the terminal the screenshot
harness uses); `tmux pipe-pane` streams everything it writes through this
script's `--stamp` mode into an asciicast v2 file, and `agg` renders that
to docs/media/agent.gif. vhs's browser terminal stalls the TUI's frames,
so it is not used. Needs tmux, agg (`brew install agg`), and a release
build in zig-out/. Run from anywhere: `python3 scripts/agent-demo.py`.
"""

import argparse
import codecs
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import playground

ROOT = pathlib.Path(__file__).resolve().parent.parent
NUCLIS = ROOT / "zig-out/bin/nuclis"
WORK = pathlib.Path("/tmp/playground")
CAST = ROOT / ".zig-cache/demo/agent.cast"
GIF = ROOT / "docs/media/agent.gif"
SOCKET = "nuclis-demo"
COLUMNS, ROWS = 110, 32
TASK = "The tests fail. Find the bug, fix it, and run the tests again."


def stamp(columns, rows):
    """stdin bytes to asciicast v2 on stdout, one output event per read."""
    header = {
        "version": 2,
        "width": columns,
        "height": rows,
        "timestamp": int(time.time()),
        "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
    }
    print(json.dumps(header), flush=True)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    start = time.monotonic()
    while True:
        chunk = os.read(0, 65536)
        if not chunk:
            break
        text = decoder.decode(chunk)
        if text:
            print(json.dumps([round(time.monotonic() - start, 4), "o", text]), flush=True)


def tmux(*args, check=True):
    return subprocess.run(["tmux", "-L", SOCKET, *args], check=check, capture_output=True, text=True).stdout


def screen():
    return tmux("capture-pane", "-p", "-t", "demo")


def until(pattern, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if re.search(pattern, screen()):
            return
        time.sleep(0.5)
    sys.exit("agent-demo: %r not seen within %ss" % (pattern, seconds))


def type_text(text, delay=0.035):
    for ch in text:
        tmux("send-keys", "-t", "demo", "-l", ch)
        time.sleep(delay)


def enter():
    tmux("send-keys", "-t", "demo", "Enter")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stamp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--speed", default="2", help="agg playback speed (default 2)")
    parser.add_argument("--idle", default="1.5", help="agg idle-time limit in seconds (default 1.5)")
    args = parser.parse_args()
    if args.stamp:
        return stamp(COLUMNS, ROWS)

    # The scratch project: the playground with rect_perimeter broken and
    # committed, so the agent's fix is the only diff.
    subprocess.run(["rm", "-rf", str(WORK)], check=True)
    subprocess.run(["git", "clone", "-q", str(playground.ensure()), str(WORK)], check=True)
    rect = WORK / "src/shapes/rect.py"
    rect.write_text(rect.read_text().replace("return 2 * (width + height)", "return width + height"))
    subprocess.run(["git", "-C", str(WORK), "commit", "-qam", "demo: a broken perimeter"], check=True)

    CAST.parent.mkdir(parents=True, exist_ok=True)
    GIF.parent.mkdir(parents=True, exist_ok=True)
    tmux("kill-server", check=False)
    env = ["-e", "COLORTERM=truecolor", "-e", "NUCLIS_NO_NOTIFY=1", "-e", "PS1=$ "]
    tmux(
        "new-session",
        "-d",
        "-s",
        "demo",
        "-x",
        str(COLUMNS),
        "-y",
        str(ROWS),
        "-c",
        str(WORK),
        *env,
        "bash --norc --noprofile",
    )
    time.sleep(1)
    # Set before the recording starts: tmux's `-e` does not carry PATH into
    # the pane's shell.
    tmux("send-keys", "-t", "demo", 'export PATH="%s:$PATH"; clear' % NUCLIS.parent, "Enter")
    time.sleep(0.5)
    tmux(
        "pipe-pane", "-O", "-t", "demo", "%s %s --stamp > %s" % (sys.executable, pathlib.Path(__file__).resolve(), CAST)
    )
    time.sleep(0.5)
    try:
        # Clear inside the recording so the cast starts from a blank screen.
        tmux("send-keys", "-t", "demo", "clear", "Enter")
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
        tmux("send-keys", "-t", "demo", "C-c")
        time.sleep(1.5)
        # A fresh screen for the ending; no pager, so the diff is plain text.
        tmux("send-keys", "-t", "demo", "clear", "Enter")
        time.sleep(0.5)
        type_text("git --no-pager diff --color")
        enter()
        time.sleep(3)
        type_text("make test 2>&1 | tail -3")
        enter()
        time.sleep(4)
    finally:
        tmux("pipe-pane", "-t", "demo", check=False)
        tmux("kill-server", check=False)
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
