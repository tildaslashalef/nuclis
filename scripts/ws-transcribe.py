#!/usr/bin/env python3
"""Live transcription through `nuclis serve`'s WebSocket route, standard library only.

    python3 scripts/ws-transcribe.py clip.wav                      # a server on 127.0.0.1:9000
    python3 scripts/ws-transcribe.py clip.flac --speed 4 --language en-US --lookahead 3
    python3 scripts/ws-transcribe.py --check --model DIR --clips ID,ID,...   # the gate
    python3 scripts/ws-transcribe.py --measure --clips ID --beside-chat gemma-4-e4b-qat

A clip is streamed to `GET /v1/audio/transcriptions/stream` as 16 kHz mono
s16le in binary messages of `--piece-ms` (20 ms), each sent once its audio
has ended at real time or `--speed` times it (0: as fast as the socket
takes it), then `{"type": "finish"}`; each
event is printed with its time since the stream began. A partial's delay is
its arrival behind the moment the audio it covers (its `audio_seconds`) had
been sent; the final's is its arrival after `finish`. The raw socket speaks
RFC 6455 itself: the handshake, masked frames, pings answered.

Reads 16 kHz mono WAV (16-bit PCM or 32-bit float, converted to 16-bit as
sent); anything else (the corpus's FLAC) through macOS's `afconvert`.

`--check` is the gate `nemotron-asr-stream-route`: it starts `--nuclis serve
--model <--serve-model>`, streams each corpus clip (`make speech-corpus`)
on one connection, each under its locale (a `language` message before its
audio), and requires every final's text to equal `nuclis transcribe
--json --model <--model>` of the same 16-bit samples at the same lookahead
and locale, exactly.

`--measure` starts the server with `--model <--beside-chat> --pin
<--serve-model>` (or alone without `--beside-chat`), streams the clips at
real time `--rounds` times after a warm-up, and reports each partial's and
the final's delay; with `--beside-chat` it measures the chat model's
streamed tokens per second alone and while the clip streams beside it.
docs/guide/api.md § Live transcription, § Measured rates.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import pathlib
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests/fixtures/speech-corpus.json"
SPEECH = ROOT / ".reference/speech"
GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
RATE = 16000
STORY = "Write a long story about a lighthouse keeper who finds a message in a bottle. Tell it in detail."


class Failure(Exception):
    pass


# ----- audio -----


def read_wav16(data: bytes) -> bytes:
    """The s16le samples of a 16 kHz mono WAV, 16-bit PCM or 32-bit float."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise Failure("not a WAV file")
    at = 12
    fmt = b""
    while at + 8 <= len(data):
        tag, size = data[at : at + 4], struct.unpack_from("<I", data, at + 4)[0]
        body = data[at + 8 : at + 8 + size]
        if tag == b"fmt ":
            fmt = body
        elif tag == b"data":
            if len(fmt) < 16:
                raise Failure("a WAV data chunk before its format")
            kind, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
            if kind == 0xFFFE and len(fmt) >= 26:
                # WAVE_FORMAT_EXTENSIBLE: the subformat GUID starts with the format.
                kind = struct.unpack_from("<H", fmt, 24)[0]
            if channels != 1 or rate != RATE:
                raise Failure(f"{channels} channels at {rate} Hz; this client sends 16 kHz mono")
            if kind == 1 and bits == 16:
                return bytes(body[: len(body) // 2 * 2])
            if kind == 3 and bits == 32:
                floats = struct.unpack(f"<{len(body) // 4}f", body[: len(body) // 4 * 4])
                return struct.pack(f"<{len(floats)}h", *(max(-32768, min(32767, round(x * 32768))) for x in floats))
            raise Failure(f"WAV format {kind} with {bits} bits; this client reads 16-bit PCM or 32-bit float")
        at += 8 + size + (size & 1)
    raise Failure("a WAV file without data")


def load_pcm(path: pathlib.Path) -> bytes:
    """16 kHz mono s16le of `path`: a WAV read here, anything else through afconvert."""
    if path.suffix.lower() == ".wav":
        return read_wav16(path.read_bytes())
    with tempfile.TemporaryDirectory() as directory:
        out = pathlib.Path(directory) / "clip.wav"
        command = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1", str(path), str(out)]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise Failure(f"{' '.join(command)}: {done.stderr.strip()}")
        return read_wav16(out.read_bytes())


def write_wav16(path: pathlib.Path, pcm: bytes) -> None:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, RATE, 2 * RATE, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    path.write_bytes(header + pcm)


# ----- RFC 6455, the client's side -----


class WebSocket:
    """One client connection: masked frames out, the server's frames in."""

    def __init__(self, host: str, port: int, target: str) -> None:
        self.sock = socket.create_connection((host, port), timeout=60)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET {target} HTTP/1.1\r\nhost: {host}:{port}\r\nupgrade: websocket\r\nconnection: Upgrade\r\n"
            f"sec-websocket-key: {key}\r\nsec-websocket-version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        self.buffer = b""
        while b"\r\n\r\n" not in self.buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise Failure("the server closed during the handshake")
            self.buffer += chunk
        head, self.buffer = self.buffer.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        if not lines[0].startswith("HTTP/1.1 101"):
            raise Failure(f"handshake refused: {lines[0]}; {self.buffer.decode(errors='replace')}")
        headers = {k.strip().lower(): v.strip() for k, v in (line.split(":", 1) for line in lines[1:])}
        expected = base64.b64encode(hashlib.sha1(key.encode() + GUID).digest()).decode()
        if headers.get("sec-websocket-accept") != expected:
            raise Failure("the handshake's accept value is wrong")
        self.send_lock = threading.Lock()

    def send(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            head = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n <= 0xFFFF:
            head = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload)) if n < 4096 else _mask(payload, mask)
        with self.send_lock:
            self.sock.sendall(head + mask + masked)

    def send_json(self, value: dict[str, Any]) -> None:
        self.send(0x1, json.dumps(value).encode())

    def _take(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError
            self.buffer += chunk
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def recv(self) -> tuple[int, bytes]:
        """The next message (the server never fragments); pings are answered here."""
        while True:
            b0, b1 = self._take(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack("!H", self._take(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._take(8))[0]
            if b1 & 0x80:
                raise Failure("the server masked a frame")
            payload = self._take(n)
            opcode = b0 & 0x0F
            if opcode == 0x9:
                self.send(0xA, payload)
                continue
            return opcode, payload

    def close(self, code: int = 1000) -> None:
        try:
            self.send(0x8, struct.pack("!H", code))
        except OSError:
            pass


def _mask(payload: bytes, mask: bytes) -> bytes:
    key = int.from_bytes(mask * ((len(payload) + 3) // 4), "big")
    padded = payload + b"\0" * (-len(payload) % 4)
    return (int.from_bytes(padded, "big") ^ key).to_bytes(len(padded), "big")[: len(payload)]


# ----- one stream -----


@dataclass
class Event:
    at: float
    body: dict[str, Any]


@dataclass
class Result:
    events: list[Event] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0
    final: dict[str, Any] | None = None
    partial_delays_ms: list[float] = field(default_factory=list)
    final_delay_ms: float = 0.0


class Session:
    """A connection with a reader thread collecting events as they arrive."""

    def __init__(self, host: str, port: int, query: str) -> None:
        self.ws = WebSocket(host, port, "/v1/audio/transcriptions/stream" + (f"?{query}" if query else ""))
        self.events: list[Event] = []
        self.closed: tuple[int, str] | None = None
        self.changed = threading.Condition()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        try:
            while True:
                opcode, payload = self.ws.recv()
                now = time.monotonic()
                with self.changed:
                    if opcode == 0x8:
                        code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 1005
                        self.closed = (code, payload[2:].decode(errors="replace"))
                        self.changed.notify_all()
                        return
                    if opcode == 0x1:
                        self.events.append(Event(now, json.loads(payload)))
                        self.changed.notify_all()
        except (EOFError, OSError):
            with self.changed:
                self.closed = self.closed or (1006, "connection lost")
                self.changed.notify_all()

    def wait_for(self, kind: str, after: int, timeout: float) -> Event:
        """The first event of `kind` at index `after` or later."""
        deadline = time.monotonic() + timeout
        with self.changed:
            while True:
                for e in self.events[after:]:
                    if e.body.get("type") == kind:
                        return e
                    if e.body.get("type") == "error":
                        raise Failure(f"error event: {e.body}")
                if self.closed:
                    raise Failure(f"closed {self.closed[0]}: {self.closed[1]}")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise Failure(f"no {kind} event in {timeout:.0f} s")
                self.changed.wait(left)

    def stream(self, pcm: bytes, speed: float, piece_ms: int, language: str | None = None) -> Result:
        """One utterance: the samples paced at `speed`, then finish; waits for the final."""
        if language is not None:
            self.ws.send_json({"type": "language", "language": language})
        with self.changed:
            first = len(self.events)
        piece = RATE * piece_ms // 1000 * 2
        result = Result(started=time.monotonic())
        for at in range(0, len(pcm), piece):
            if speed > 0:
                # As a microphone delivers it: a piece once its audio has ended.
                due = result.started + min(at + piece, len(pcm)) / 2 / RATE / speed
                pause = due - time.monotonic()
                if pause > 0:
                    time.sleep(pause)
            self.ws.send(0x2, pcm[at : at + piece])
        result.finished = time.monotonic()
        self.ws.send_json({"type": "finish"})
        final = self.wait_for("final", first, 120)
        with self.changed:
            result.events = [e for e in self.events[first:] if e.at <= final.at]
        result.final = final.body
        result.final_delay_ms = (final.at - result.finished) * 1000
        if speed > 0:
            for e in result.events:
                if e.body.get("type") == "partial":
                    covered = result.started + float(e.body["audio_seconds"]) / speed
                    result.partial_delays_ms.append((e.at - covered) * 1000)
        return result

    def close(self) -> None:
        self.ws.close()
        self.thread.join(timeout=5)
        self.ws.sock.close()


def show(result: Result) -> None:
    delays = iter(result.partial_delays_ms)
    for e in result.events:
        kind = e.body.get("type")
        stamp = f"+{e.at - result.started:7.3f} s"
        if kind == "partial":
            delay = next(delays, None)
            behind = f" {delay:6.0f} ms behind" if delay is not None else ""
            print(f"{stamp}  partial @{e.body['audio_seconds']:.2f} s{behind}  {e.body['text']}")
        elif kind == "final":
            words = len(e.body.get("words") or [])
            print(
                f"{stamp}  final   {result.final_delay_ms:.0f} ms after finish  [{e.body.get('language')}, {words} words]"
            )
            print(f"           {e.body['text']}")
        else:
            print(f"{stamp}  {json.dumps(e.body)}")


# ----- the server -----


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    def __init__(self, nuclis: str, args: list[str]) -> None:
        self.port = free_port()
        command = [nuclis, "serve", "--host", "127.0.0.1", "--port", str(self.port), "--quiet", *args]
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.time() + 600
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/health", timeout=2).close()
                return
            except OSError:
                if self.process.poll() is not None or time.time() > deadline:
                    err = self.process.stderr.read().decode() if self.process.stderr else ""
                    raise Failure(f"{' '.join(command)} did not start: {err}") from None
                time.sleep(0.3)

    def stop(self) -> None:
        self.process.send_signal(2)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()


# ----- the chat beside it -----


@dataclass
class ChatRun:
    stamps: list[float] = field(default_factory=list)
    error: str | None = None


def chat_stream(port: int, model: str, max_tokens: int, run: ChatRun) -> None:
    """A streamed chat completion; the arrival time of each content chunk."""
    body = {
        "model": model,
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": STORY}],
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            run.error = f"{response.status} {response.read().decode(errors='replace')}"
            return
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                run.stamps.append(time.monotonic())
    finally:
        connection.close()


def rate(stamps: list[float], start: float, end: float) -> float:
    """Content chunks per second between the first and the last inside [start, end]."""
    inside = [t for t in stamps if start <= t <= end]
    if len(inside) < 2:
        return 0.0
    return (len(inside) - 1) / (inside[-1] - inside[0])


# ----- modes -----


def corpus_clips(ids: str) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = json.loads(CORPUS.read_text())["clips"]
    known = {c["id"]: c for c in clips}
    missing = [i for i in ids.split(",") if i not in known]
    if missing:
        raise Failure(f"not in {CORPUS.name}: {', '.join(missing)}")
    return [known[i] for i in ids.split(",")]


def clip_pcm(clip: dict[str, Any]) -> bytes:
    path = SPEECH / clip["file"]
    if not path.is_file():
        raise Failure(f"{path}: missing; run make speech-corpus")
    return load_pcm(path)


def check(args: argparse.Namespace) -> int:
    """The gate: every final equals `nuclis transcribe --json` of the same samples."""
    clips = corpus_clips(args.clips)
    server = Server(args.nuclis, ["--model", args.serve_model])
    failures = 0
    try:
        session = Session("127.0.0.1", server.port, f"lookahead={args.lookahead}")
        with tempfile.TemporaryDirectory() as directory:
            for clip in clips:
                pcm = clip_pcm(clip)
                wav = pathlib.Path(directory) / f"{clip['id']}.wav"
                write_wav16(wav, pcm)
                command = [args.nuclis, "transcribe", str(wav), "--json", "--backend", "metal"]
                command += ["--lookahead", str(args.lookahead), "--language", clip["language"], "--model", args.model]
                done = subprocess.run(command, capture_output=True, text=True, check=False)
                if done.returncode != 0:
                    raise Failure(f"{' '.join(command)}: exit {done.returncode}\n{done.stderr}")
                expected = json.loads(done.stdout)["text"]
                result = session.stream(pcm, args.speed, args.piece_ms, clip["language"])
                got = (result.final or {}).get("text")
                partials = sum(1 for e in result.events if e.body.get("type") == "partial")
                ok = got == expected
                failures += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {clip['id']}: {len(pcm) / 2 / RATE:.1f} s, {partials} partials")
                if not ok:
                    print(f"     transcribe: {expected}\n     stream:     {got}")
        session.close()
    finally:
        server.stop()
    print(f"{len(clips) - failures} of {len(clips)} finals equal nuclis transcribe's")
    return 1 if failures else 0


def summarize(results: list[Result]) -> str:
    partials = [d for r in results for d in r.partial_delays_ms]
    finals = [r.final_delay_ms for r in results]
    if not partials:
        return "no partials"
    q = statistics.quantiles(partials, n=10) if len(partials) >= 2 else [partials[0]] * 9
    return (
        f"{len(partials)} partials behind the audio: median {statistics.median(partials):.0f} ms, "
        f"p90 {q[8]:.0f} ms, max {max(partials):.0f} ms; final after finish: median "
        f"{statistics.median(finals):.0f} ms (range {min(finals):.0f}-{max(finals):.0f})"
    )


def measure(args: argparse.Namespace) -> int:
    clips = corpus_clips(args.clips)
    serve_args = ["--pin", args.serve_model]
    if args.beside_chat:
        serve_args = ["--model", args.beside_chat, *serve_args]
    server = Server(args.nuclis, serve_args)
    try:
        query = f"lookahead={args.lookahead}"
        audio = [(clip, clip_pcm(clip)) for clip in clips]
        session = Session("127.0.0.1", server.port, query)
        session.stream(audio[0][1], 4, args.piece_ms, audio[0][0]["language"])  # warm-up
        alone: list[Result] = []
        for _ in range(args.rounds):
            for clip, pcm in audio:
                alone.append(session.stream(pcm, 1, args.piece_ms, clip["language"]))
        print(f"alone: {summarize(alone)}")
        if args.verbose:
            show(alone[0])
        if args.beside_chat:
            solo = ChatRun()
            chat_stream(server.port, args.beside_chat, args.max_tokens, solo)
            if solo.error:
                raise Failure(f"chat: {solo.error}")
            print(f"{args.beside_chat} alone: {rate(solo.stamps, 0, float('inf')):.1f} tokens/s")
            beside: list[Result] = []
            during: list[float] = []
            after: list[float] = []
            for _ in range(args.rounds):
                for clip, pcm in audio:
                    run = ChatRun()
                    thread = threading.Thread(
                        target=chat_stream, args=(server.port, args.beside_chat, args.max_tokens, run)
                    )
                    thread.start()
                    while len(run.stamps) < 20 and thread.is_alive():
                        time.sleep(0.01)
                    result = session.stream(pcm, 1, args.piece_ms, clip["language"])
                    thread.join()
                    if run.error:
                        raise Failure(f"chat: {run.error}")
                    beside.append(result)
                    end = result.events[-1].at
                    during.append(rate(run.stamps, result.started, end))
                    after.append(rate(run.stamps, end, float("inf")))
            print(f"beside {args.beside_chat}: {summarize(beside)}")
            print(
                f"{args.beside_chat} while a clip streams: {statistics.median(during):.1f} tokens/s "
                f"(range {min(during):.1f}-{max(during):.1f}); after: {statistics.median(after):.1f}"
            )
            if args.verbose:
                show(beside[0])
        session.close()
    finally:
        server.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", nargs="?", type=pathlib.Path, help="a WAV (16 kHz mono) or anything afconvert reads")
    parser.add_argument("--url", default="127.0.0.1:9000", help="host:port of a running server")
    parser.add_argument("--speed", type=float, default=1.0, help="times real time; 0 sends at once")
    parser.add_argument("--piece-ms", type=int, default=20, help="audio per binary message")
    parser.add_argument("--lookahead", type=int, default=3, choices=(0, 3, 6, 13))
    parser.add_argument("--language", help="a locale (en-US, de-DE, ...); default auto")
    parser.add_argument("--model", help="a transcription model's name (a clip); its directory (--check)")
    parser.add_argument("--json", action="store_true", help="print the events as JSON lines")
    parser.add_argument("--check", action="store_true", help="the gate: finals against nuclis transcribe")
    parser.add_argument("--measure", action="store_true", help="delays at real time, alone and beside a chat")
    parser.add_argument("--nuclis", default=str(ROOT / "zig-out/bin/nuclis"), help="the binary (--check, --measure)")
    parser.add_argument("--serve-model", default="nemotron-asr-0.6b", help="what the server opens")
    parser.add_argument("--clips", help="comma-separated corpus ids (--check, --measure)")
    parser.add_argument("--beside-chat", help="a language model streaming a chat answer meanwhile (--measure)")
    parser.add_argument("--max-tokens", type=int, default=1200, help="the chat answer's length (--measure)")
    parser.add_argument("--rounds", type=int, default=3, help="times each clip is streamed (--measure)")
    parser.add_argument("--verbose", action="store_true", help="print one stream's events (--measure)")
    args = parser.parse_args()
    try:
        if args.check or args.measure:
            if not args.clips:
                parser.error("--check and --measure need --clips")
            if args.check and not args.model:
                parser.error("--check needs --model, the checkpoint directory transcribe reads")
            return check(args) if args.check else measure(args)
        if args.clip is None:
            parser.error("give a clip, or --check / --measure")
        host, _, port = str(args.url).rpartition(":")
        query = f"lookahead={args.lookahead}" + (f"&model={args.model}" if args.model else "")
        session = Session(host or "127.0.0.1", int(port), query)
        result = session.stream(load_pcm(args.clip), args.speed, args.piece_ms, args.language)
        session.close()
        if args.json:
            for e in result.events:
                print(json.dumps({"at": round(e.at - result.started, 4), **e.body}))
        else:
            show(result)
        return 0
    except Failure as failure:
        print(f"ws-transcribe: {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
