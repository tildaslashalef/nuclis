#!/usr/bin/env python3
"""The screenshot harness: drives `nuclis agent` in a detached tmux session
and captures its screen, so the surface can be looked at without a person
at the keyboard.

tmux is a full terminal emulator, so what it captures is what a terminal
would show: the live region's scrolling, the synchronized frames, the
colours. The session runs with COLORTERM=truecolor, the way Ghostty sets it,
so the theme resolves to the same palette. Each capture is written three
ways under .zig-cache/tui/: `<name>.txt` (the text), `<name>.ansi` (the
escape stream tmux replays), and `<name>.tagged.txt` (every styled run as
`[fg=#hex bg=#hex bold]…[/]`, readable without a terminal). A burst captures
at a fixed rate for a while and reports which frames changed and the
interval between them, which is how an animation's cadence is measured.

Steps, in order, as arguments:
  keys=<text>            type the text (literally; no key names)
  key=<name>             one tmux key name: Enter, Tab, Escape, C-c, C-o, Up
  enter                  the same as key=Enter
  wait=<seconds>         sleep
  capture=<name>         capture the screen now
  burst=<name>,<s>,<hz>  capture every 1/hz seconds for s seconds
  until=<text>,<s>       wait up to s seconds for the text to appear on screen

`--ghostty` also opens one Ghostty window attached to the session and, on
each capture, photographs it to `<name>.png` with `screencapture` (needs the
screen-recording permission for the terminal running this script; without
it the PNG step reports and the text captures continue).

The session is left running unless `--stop` is given; attach to it with
`tmux -L nuclis attach -t <session>`. See docs/development.md § The agent's
transcript.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / '.zig-cache' / 'tui'
SOCKET = 'nuclis'
DEFAULT_COMMAND = './zig-out/bin/nuclis agent'

TMUX_CONF = """
set -g default-terminal "tmux-256color"
set -as terminal-overrides ",*:RGB"
set -g status off
set -g escape-time 0
set -g focus-events on
set -g history-limit 10000
set -g remain-on-exit on
set -g window-size latest
"""

# ---- the SGR tagger (pure; covered by --self-test) --------------------------

BASIC = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
SGR = re.compile(r'\x1b\[([0-9;:]*)m')
OTHER_ESCAPE = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])')


def _color(params, i):
    """Parses the colour after a 38/48 at params[i]; returns (name, consumed)."""
    if i < len(params) and params[i] == '5' and i + 1 < len(params):
        return 'c%s' % params[i + 1], 2
    if i < len(params) and params[i] == '2' and i + 3 < len(params):
        r, g, b = (int(x or 0) for x in params[i + 1:i + 4])
        return '#%02x%02x%02x' % (r, g, b), 4
    return None, 0


def apply_sgr(state, params):
    """Folds one SGR parameter list into `state` (a dict) and returns it."""
    p = [x for x in params.replace(':', ';').split(';')] if params else ['0']
    i = 0
    while i < len(p):
        n = int(p[i] or 0)
        i += 1
        if n == 0:
            state.clear()
        elif n in (1, 2, 3, 4, 7, 9):
            state[{1: 'bold', 2: 'dim', 3: 'italic', 4: 'underline', 7: 'reverse', 9: 'strike'}[n]] = True
        elif n == 22:
            state.pop('bold', None)
            state.pop('dim', None)
        elif n in (23, 24, 27, 29):
            state.pop({23: 'italic', 24: 'underline', 27: 'reverse', 29: 'strike'}[n], None)
        elif 30 <= n <= 37:
            state['fg'] = BASIC[n - 30]
        elif 90 <= n <= 97:
            state['fg'] = 'bright-' + BASIC[n - 90]
        elif n == 39:
            state.pop('fg', None)
        elif 40 <= n <= 47:
            state['bg'] = BASIC[n - 40]
        elif 100 <= n <= 107:
            state['bg'] = 'bright-' + BASIC[n - 100]
        elif n == 49:
            state.pop('bg', None)
        elif n in (38, 48):
            name, used = _color(p, i)
            if name:
                state['fg' if n == 38 else 'bg'] = name
            i += used
    return state


def tag_of(state):
    parts = []
    for key in ('fg', 'bg'):
        if key in state:
            parts.append('%s=%s' % (key, state[key]))
    for flag in ('bold', 'dim', 'italic', 'underline', 'reverse', 'strike'):
        if state.get(flag):
            parts.append(flag)
    return '[' + ' '.join(parts) + ']' if parts else ''


def tagged(ansi):
    """The escape stream as text with `[attrs]…[/]` around every styled run.
    A tag is emitted only when the style changes and only before visible
    text, so an unstyled line is unchanged."""
    out = []
    state = {}
    open_tag = ''
    pending = ''
    pos = 0
    for line in ansi.split('\n'):
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
                    out.append('[/]')
                if pending:
                    out.append(pending)
                open_tag = pending
            out.append(line[pos])
            pos += 1
        if open_tag:
            out.append('[/]')
            open_tag = ''
        pending = tag_of(state)
        out.append('\n')
    return ''.join(out).rstrip('\n') + '\n'


def plain(ansi):
    return OTHER_ESCAPE.sub('', SGR.sub('', ansi))


# ---- tmux --------------------------------------------------------------------

class Session:
    def __init__(self, name, size, command, cwd, keep_conf=False):
        self.name = name
        self.columns, self.rows = size
        self.command = command
        self.cwd = cwd
        conf = tempfile.NamedTemporaryFile('w', suffix='.tmux.conf', delete=False)
        conf.write(TMUX_CONF)
        conf.close()
        self.conf = conf.name

    def tmux(self, *args, check=True, capture=False):
        cmd = ['tmux', '-L', SOCKET, '-f', self.conf] + list(args)
        return subprocess.run(cmd, check=check, capture_output=capture, text=True)

    def start(self):
        self.tmux('kill-session', '-t', self.name, check=False, capture=True)
        env = ['-e', 'COLORTERM=truecolor', '-e', 'NUCLIS_NO_NOTIFY=1']
        self.tmux('new-session', '-d', '-s', self.name, '-x', str(self.columns), '-y', str(self.rows),
                  '-c', str(self.cwd), *env, self.command)
        # The pane must be exactly the requested size: the status bar is off
        # and no client is attached, so the window keeps its creation size.
        info = self.tmux('display-message', '-p', '-t', self.name, '#{pane_width}x#{pane_height}', capture=True).stdout.strip()
        if info != '%dx%d' % (self.columns, self.rows):
            print('warning: pane is %s, wanted %dx%d' % (info, self.columns, self.rows), file=sys.stderr)

    def send_text(self, text):
        self.tmux('send-keys', '-t', self.name, '-l', text)

    def send_key(self, key):
        self.tmux('send-keys', '-t', self.name, key)

    def screen(self):
        return self.tmux('capture-pane', '-t', self.name, '-p', '-e', capture=True).stdout

    def alive(self):
        return self.tmux('has-session', '-t', self.name, check=False, capture=True).returncode == 0

    def stop(self):
        self.tmux('kill-session', '-t', self.name, check=False, capture=True)
        os.unlink(self.conf)


# ---- Ghostty ----------------------------------------------------------------

class Ghostty:
    """One Ghostty window attached to the session, photographed on capture."""

    def __init__(self, session):
        self.session = session
        self.window_id = None

    def open(self):
        subprocess.run(['open', '-na', 'Ghostty', '--args',
                        '--window-width=%d' % self.session.columns, '--window-height=%d' % self.session.rows,
                        '-e', 'tmux', '-L', SOCKET, 'attach', '-t', self.session.name], check=True)
        deadline = time.time() + 10
        while time.time() < deadline and self.window_id is None:
            time.sleep(0.5)
            self.window_id = self.find_window()
        if self.window_id is None:
            print('ghostty: window not found; PNG captures skipped', file=sys.stderr)

    def find_window(self):
        script = ('ObjC.import("CoreGraphics");'
                  'const list = ObjC.deepUnwrap($.CGWindowListCopyWindowInfo($.kCGWindowListOptionOnScreenOnly, 0));'
                  'const w = list.filter(w => w.kCGWindowOwnerName === "Ghostty" && (w.kCGWindowName || "").includes("%s"));'
                  'JSON.stringify(w.map(w => w.kCGWindowNumber));' % self.session.name)
        r = subprocess.run(['osascript', '-l', 'JavaScript', '-e', script], capture_output=True, text=True)
        try:
            ids = json.loads(r.stdout.strip() or '[]')
        except json.JSONDecodeError:
            return None
        return ids[0] if ids else None

    def shot(self, path):
        if self.window_id is None:
            return False
        r = subprocess.run(['screencapture', '-x', '-o', '-l', str(self.window_id), str(path)], capture_output=True, text=True)
        if r.returncode != 0 or not path.exists():
            print('ghostty: screencapture failed (%s); grant screen recording to the terminal' % r.stderr.strip(), file=sys.stderr)
            self.window_id = None
            return False
        return True


# ---- the run ----------------------------------------------------------------

def write_capture(out_dir, name, ansi):
    (out_dir / (name + '.ansi')).write_text(ansi)
    (out_dir / (name + '.txt')).write_text(plain(ansi))
    (out_dir / (name + '.tagged.txt')).write_text(tagged(ansi))


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
    for i in ([0] + changed):
        write_capture(out_dir, '%s-%02d' % (name, i), frames[i][1])
    gaps = [frames[changed[k]][0] - frames[changed[k - 1]][0] for k in range(1, len(changed))]
    mean = sum(gaps) / len(gaps) if gaps else None
    report = {'name': name, 'frames': len(frames), 'seconds': seconds, 'hz': hz,
              'changed': [round(frames[i][0], 3) for i in changed],
              'mean_interval_s': round(mean, 3) if mean is not None else None}
    (out_dir / (name + '.burst.json')).write_text(json.dumps(report, indent=1) + '\n')
    print('burst %s: %d frames, %d changed, mean interval %s' % (
        name, len(frames), len(changed), ('%.3fs' % mean) if mean is not None else 'n/a'))


def wait_for(session, text, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if text in plain(session.screen()):
            return True
        time.sleep(0.2)
    print('until: %r not seen within %ss' % (text, seconds), file=sys.stderr)
    return False


def parse_step(arg):
    if arg == 'enter':
        return ('key', 'Enter')
    if '=' not in arg:
        raise SystemExit('bad step: %s' % arg)
    kind, value = arg.split('=', 1)
    if kind in ('keys', 'key', 'capture'):
        return (kind, value)
    if kind == 'wait':
        return (kind, float(value))
    if kind == 'burst':
        name, seconds, hz = value.split(',')
        return (kind, (name, float(seconds), float(hz)))
    if kind == 'until':
        text, seconds = value.rsplit(',', 1)
        return (kind, (text, float(seconds)))
    raise SystemExit('bad step: %s' % arg)


def run(args):
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    columns, rows = (int(x) for x in args.size.lower().split('x'))
    session = Session(args.session, (columns, rows), args.command, ROOT / args.cwd)
    session.start()
    ghostty = Ghostty(session) if args.ghostty else None
    if ghostty:
        ghostty.open()
    ok = True
    for step in [parse_step(s) for s in args.steps]:
        kind, value = step
        if kind == 'keys':
            session.send_text(value)
        elif kind == 'key':
            session.send_key(value)
        elif kind == 'wait':
            time.sleep(value)
        elif kind == 'capture':
            write_capture(out_dir, value, session.screen())
            if ghostty:
                ghostty.shot(out_dir / (value + '.png'))
            print('capture %s' % value)
        elif kind == 'burst':
            burst(session, out_dir, *value)
        elif kind == 'until':
            ok = wait_for(session, *value) and ok
    if args.stop:
        session.stop()
    else:
        print('session left running: tmux -L %s attach -t %s' % (SOCKET, args.session))
    return 0 if ok else 1


# ---- self-test ----------------------------------------------------------------

class TaggerTest(unittest.TestCase):
    def test_truecolor_and_reset(self):
        self.assertEqual(tagged('a\x1b[38;2;255;0;0mred\x1b[0mb\n'), 'a[fg=#ff0000]red[/]b\n')

    def test_bold_dim_and_background(self):
        self.assertEqual(tagged('\x1b[1m\x1b[48;5;236mx\x1b[22my\x1b[m\n'), '[bg=c236 bold]x[/][bg=c236]y[/]\n')

    def test_basic_colors_and_other_escapes(self):
        self.assertEqual(tagged('\x1b[32m\x1b[Kok\x1b[39m done\n'), '[fg=green]ok[/] done\n')
        self.assertEqual(plain('\x1b[32mok\x1b[0m\x1b]8;;http://x\x1b\\l\x1b]8;;\x1b\\'), 'okl')

    def test_style_carries_across_lines(self):
        self.assertEqual(tagged('\x1b[2ma\nb\x1b[0m\n'), '[dim]a[/]\n[dim]b[/]\n')

    def test_empty_and_unstyled(self):
        self.assertEqual(tagged('plain\n\nrows\n'), 'plain\n\nrows\n')

    def test_steps(self):
        self.assertEqual(parse_step('enter'), ('key', 'Enter'))
        self.assertEqual(parse_step('burst=warm,30,10'), ('burst', ('warm', 30.0, 10.0)))
        self.assertEqual(parse_step('until=ready,40'), ('until', ('ready', 40.0)))
        self.assertEqual(parse_step('keys=a=b'), ('keys', 'a=b'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('steps', nargs='*', help='the steps, in order')
    ap.add_argument('--size', default='160x45', help='columns x rows (default 160x45)')
    ap.add_argument('--session', default='shot', help='tmux session name (default shot)')
    ap.add_argument('--command', default=DEFAULT_COMMAND, help='what to run (default: %s)' % DEFAULT_COMMAND)
    ap.add_argument('--cwd', default='.', help='working directory of the command, relative to the repository')
    ap.add_argument('--out', default=str(OUT_DIR.relative_to(ROOT)), help='where captures go')
    ap.add_argument('--ghostty', action='store_true', help='also attach a Ghostty window and photograph it on capture')
    ap.add_argument('--stop', action='store_true', help='kill the session at the end')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TaggerTest)
        return 0 if unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful() else 1
    if not args.steps:
        ap.error('no steps given')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
