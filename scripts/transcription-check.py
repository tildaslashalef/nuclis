#!/usr/bin/env python3
"""`POST /v1/audio/transcriptions` against `nuclis transcribe`, and long
recordings through it. Two gates (gates.json) and the measurements of
docs/guide/api.md § Transcription (files):

    transcription-check.py route --model DIR   # the corpus: route text == CLI text, per clip
    transcription-check.py long --model DIR    # synthetic 5- and 30-minute recordings
    transcription-check.py long --model DIR --minutes 30 --chat gemma-4-e4b-qat   # beside a chat

Each run starts a fresh `nuclis serve` (`--nuclis`, default the fresh
./zig-out/bin/nuclis) under a temporary NUCLIS_HOME whose registry names
the checkpoint directory `transcription-check` (with `--chat`, the user's
own home and catalogue names instead), so the user's configuration is not
read. The corpus is `make speech-corpus` (`.reference/speech/`).

`route`: every clip goes through the route (`response_format=json`,
`language` its locale) and through `nuclis transcribe --json --language`
on the same backend; the two texts must be equal, byte for byte.

`long`: a recording is built from the corpus clips of `--languages`
(default `en-US`, the 40 LibriSpeech clips: one speaker's language, as a
note-taking app records) in a seeded order, 16-bit 16 kHz mono, each clip
followed by a silence of 1.5 to 3 s, until it reaches the duration
(`--minutes`, default 5 and 30), under
`.zig-cache/transcription/` (never committed). It is transcribed with
`response_format=verbose_json` while the server's RSS is sampled. Checks:
every segment lies within one clip's span, widened by one 80 ms row, so
every join is a segment boundary; the words of each clip's segments equal
the clip transcribed alone (`nuclis transcribe` of the same samples) but
for at most `--max-drift` of the words (edits over words: the encoder's
and the decoder's state carry from one clip into the next, so a clip's
first words can differ; the word error rate against the corpus's
references is reported for both); and the peak
RSS of the longest recording exceeds the shortest's by at most the bytes
its larger upload adds plus `--rss-margin-mb`. Then once with
`stream=true`: the time to the first segment event. With `--chat`, a
chat stream runs during the transcription and alone, for its rate.

Standard library only; `afconvert` (macOS) decodes LibriSpeech's FLAC.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import pathlib
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import Any, cast

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import wer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / ".zig-cache/transcription"
RATE = 16000
ROW = 0.08
ENTRY = "transcription-check"
Json = dict[str, Any]


class Failure(Exception):
    pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return cast(int, s.getsockname()[1])


class Server:
    """A fresh `nuclis serve` on a free port, the transcription model open."""

    def __init__(self, nuclis: str, model: str, chat: str | None = None) -> None:
        self.port = free_port()
        env = dict(os.environ)
        self.home: tempfile.TemporaryDirectory[str] | None = None
        if chat is None:
            self.home = tempfile.TemporaryDirectory(prefix="nuclis-transcription-")
            config = {"schema_version": 1, "models": {ENTRY: {"kind": "transcription", "path": model}}}
            pathlib.Path(self.home.name, "nuclis.json").write_text(json.dumps(config))
            env["NUCLIS_HOME"] = self.home.name
            command = [nuclis, "serve", "--port", str(self.port), "--quiet", "--model", ENTRY]
            self.model = ENTRY
        else:
            command = [nuclis, "serve", "--port", str(self.port), "--quiet", "--model", model, "--model", chat]
            self.model = model
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
        deadline = time.monotonic() + 600
        while True:
            try:
                connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                connection.request("GET", "/v1/health")
                connection.getresponse().read()
                connection.close()
                return
            except OSError:
                if self.process.poll() is not None or time.monotonic() > deadline:
                    stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                    raise Failure(f"serve did not start: {stderr}") from None
                time.sleep(0.2)

    def rss(self) -> int:
        """The server's resident set, in bytes."""
        done = subprocess.run(["ps", "-o", "rss=", "-p", str(self.process.pid)], capture_output=True, text=True)
        return int(done.stdout.strip() or 0) * 1024

    def close(self) -> None:
        self.process.send_signal(2)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
        if self.home is not None:
            self.home.cleanup()


def multipart(fields: list[tuple[str, str]], audio: bytes, filename: str) -> tuple[bytes, str]:
    boundary = "nuclis-check-6f1d2b"
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(audio)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe_http(port: int, fields: list[tuple[str, str]], audio: bytes, filename: str) -> Json:
    body, content_type = multipart(fields, audio, filename)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    connection.request("POST", "/v1/audio/transcriptions", body, {"content-type": content_type})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    if response.status != 200:
        raise Failure(f"HTTP {response.status}: {data[:400]!r}")
    return cast(Json, json.loads(data))


@dataclass
class Streamed:
    first_segment_s: float | None = None
    segments: list[Json] = field(default_factory=list)
    deltas: list[str] = field(default_factory=list)
    done: Json | None = None
    progress: int = 0


def transcribe_stream(port: int, fields: list[tuple[str, str]], audio: bytes, filename: str) -> Streamed:
    body, content_type = multipart([*fields, ("stream", "true")], audio, filename)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    started = time.monotonic()
    connection.request("POST", "/v1/audio/transcriptions", body, {"content-type": content_type})
    response = connection.getresponse()
    if response.status != 200:
        raise Failure(f"HTTP {response.status}: {response.read()[:400]!r}")
    out = Streamed()
    for raw in response:
        line = raw.decode().rstrip("\n")
        if line.startswith(": progress"):
            out.progress += 1
        if not line.startswith("data: "):
            continue
        event = cast(Json, json.loads(line[len("data: ") :]))
        kind = event.get("type")
        if kind == "transcript.text.segment":
            if out.first_segment_s is None:
                out.first_segment_s = time.monotonic() - started
            out.segments.append(event)
        elif kind == "transcript.text.delta":
            out.deltas.append(cast(str, event["delta"]))
        elif kind == "transcript.text.done":
            out.done = event
        else:
            raise Failure(f"unexpected event {line[:200]}")
    connection.close()
    return out


def cli(nuclis: str, model: str, path: pathlib.Path, language: str, backend: str) -> Json:
    command = [
        nuclis,
        "transcribe",
        str(path),
        "--json",
        "--backend",
        backend,
        "--model",
        model,
        "--language",
        language,
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise Failure(f"{' '.join(command)}: exit {done.returncode}\n{done.stderr}")
    return cast(Json, json.loads(done.stdout))


def corpus(speech: pathlib.Path) -> list[Json]:
    clips = cast(list[Json], json.loads(wer.CORPUS.read_text())["clips"])
    for clip in clips:
        if not (speech / clip["file"]).is_file():
            raise Failure(f"{speech / clip['file']}: missing; run make speech-corpus")
    return clips


def route(args: argparse.Namespace) -> int:
    clips = corpus(args.speech)
    server = Server(args.nuclis, args.model)
    started = time.monotonic()
    different = 0
    audio_s = 0.0
    try:
        for clip in clips:
            path = args.speech / clip["file"]
            served = transcribe_http(
                server.port, [("model", server.model), ("language", clip["language"])], path.read_bytes(), path.name
            )
            alone = cli(args.nuclis, args.model, path, clip["language"], args.backend)
            audio_s += cast(float, alone["audio_seconds"])
            same = served["text"] == alone["text"]
            different += not same
            print(f"{clip['id']:<34} {'same' if same else 'DIFFERENT'}")
            if not same:
                print(f"  route {served['text']!r}\n  cli   {alone['text']!r}")
    finally:
        server.close()
    print(
        f"\n{len(clips) - different} of {len(clips)} clips equal ({audio_s:.1f} s of audio, {time.monotonic() - started:.1f} s)"
    )
    return 1 if different else 0


# ----- long recordings -----


def read_wav(data: bytes) -> list[float] | bytes:
    """A 16 kHz mono WAV's samples: 16-bit as bytes, float as values."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise Failure("not a WAV")
    at = 12
    tag = bits = 0
    while at + 8 <= len(data):
        chunk, size = data[at : at + 4], struct.unpack_from("<I", data, at + 4)[0]
        body = data[at + 8 : at + 8 + size]
        if chunk == b"fmt ":
            tag, channels, rate = struct.unpack_from("<HHI", body)
            bits = struct.unpack_from("<H", body, 14)[0]
            if tag == 0xFFFE:
                tag = struct.unpack_from("<H", body, 24)[0]
            if channels != 1 or rate != RATE:
                raise Failure(f"a WAV of {channels} channels at {rate} Hz")
        elif chunk == b"data":
            if tag == 1 and bits == 16:
                return body
            if tag == 3 and bits == 32:
                return list(struct.unpack(f"<{len(body) // 4}f", body))
            raise Failure(f"WAV format {tag}/{bits}")
        at += 8 + size + (size & 1)
    raise Failure("a WAV without data")


def pcm16(samples: list[float] | bytes) -> bytes:
    if isinstance(samples, bytes):
        return samples
    return struct.pack(f"<{len(samples)}h", *(max(-32768, min(32767, round(s * 32768))) for s in samples))


def wav16(pcm: bytes) -> bytes:
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        RATE,
        2 * RATE,
        2,
        16,
        b"data",
        len(pcm),
    )
    return header + pcm


def clip_pcm(path: pathlib.Path) -> bytes:
    """The clip as 16-bit samples, cached under .zig-cache/transcription/clips."""
    cached = CACHE / "clips" / (path.stem + ".wav")
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".wav":
            data = path.read_bytes()
        else:
            with tempfile.TemporaryDirectory() as directory:
                out = pathlib.Path(directory, "clip.wav")
                subprocess.run(
                    ["afconvert", "-f", "WAVE", "-d", "LEF32@16000", "-c", "1", str(path), str(out)], check=True
                )
                data = out.read_bytes()
        cached.write_bytes(wav16(pcm16(read_wav(data))))
    return cast(bytes, read_wav(cached.read_bytes()))


@dataclass
class Placed:
    clip: Json
    start: float
    end: float


def build(clips: list[Json], speech: pathlib.Path, minutes: float) -> tuple[pathlib.Path, list[Placed]]:
    """The recording of about `minutes`, and where each clip sits in it."""
    order = random.Random(1)
    target = minutes * 60 * RATE
    pieces: list[bytes] = []
    placed: list[Placed] = []
    samples = 0
    while samples < target:
        shuffled = clips[:]
        order.shuffle(shuffled)
        for clip in shuffled:
            pcm = clip_pcm(speech / clip["file"])
            placed.append(Placed(clip, samples / RATE, (samples + len(pcm) // 2) / RATE))
            pieces.append(pcm)
            samples += len(pcm) // 2
            silence = order.choice((1.5, 2.0, 2.5, 3.0))
            pieces.append(bytes(2 * round(silence * RATE)))
            samples += round(silence * RATE)
            if samples >= target:
                break
    languages = "-".join(sorted({c["language"] for c in clips}))
    path = CACHE / f"long-{minutes:g}min-{languages}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav16(b"".join(pieces)))
    return path, placed


def alone_texts(args: argparse.Namespace, clips: list[Json]) -> dict[str, list[str]]:
    """Each clip's words, transcribed alone from the same 16-bit samples under `auto`."""
    out: dict[str, list[str]] = {}
    for clip in clips:
        clip_pcm(args.speech / clip["file"])
        path = CACHE / "clips" / (pathlib.Path(clip["file"]).stem + ".wav")
        out[clip["id"]] = wer.normalize(cli(args.nuclis, args.model, path, "auto", args.backend)["text"])
    return out


@dataclass
class Long:
    minutes: float
    upload: int
    audio_s: float
    wall_s: float
    peak_rss: int
    idle_rss: int
    segments: int
    score: Score


@dataclass
class Score:
    """Segments outside every clip's span; edits against the clips alone over
    their words, and clips unequal; the long recording's and the clips
    alone's edits against the references, over the references' words."""

    misplaced: list[str]
    drift: int = 0
    words: int = 0
    unequal: int = 0
    errors: int = 0
    alone_errors: int = 0
    reference_words: int = 0


def check_segments(result: Json, placed: list[Placed], alone: dict[str, list[str]]) -> Score:
    misplaced: list[str] = []
    by_clip: list[list[str]] = [[] for _ in placed]
    k = 0
    for segment in cast(list[Json], result["segments"]):
        start, end = cast(float, segment["start"]), cast(float, segment["end"])
        while k < len(placed) and placed[k].end + ROW < start:
            k += 1
        if k == len(placed) or not (placed[k].start - ROW <= start and end <= placed[k].end + ROW + 1e-9):
            misplaced.append(f"{start:.2f}-{end:.2f} {segment['text'][:60]!r}")
            continue
        by_clip[k].append(cast(str, segment["text"]))
    score = Score(misplaced)
    for p, texts in zip(placed, by_clip, strict=True):
        got = wer.normalize(" ".join(texts))
        want = alone[p.clip["id"]]
        reference = wer.normalize(p.clip.get("normalized") or p.clip.get("text", ""))
        e = wer.edits(want, got)
        score.drift += e
        score.words += len(want)
        score.unequal += e > 0
        score.errors += wer.edits(reference, got)
        score.alone_errors += wer.edits(reference, want)
        score.reference_words += len(reference)
    return score


def chat_rate(port: int, model: str, stop: threading.Event | None, out: list[float]) -> None:
    """Streams a long chat answer; appends its content chunks per second (while `stop` is unset, if given)."""
    body = {
        "model": model,
        "stream": True,
        "temperature": 0,
        "max_tokens": 1200,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": "Tell a long story about a lighthouse keeper, in plain prose."}],
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    connection.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    response = connection.getresponse()
    times: list[float] = []
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = cast(Json, json.loads(line[6:]))
        choices = cast(list[Json], chunk.get("choices") or [])
        if choices and cast(Json, choices[0].get("delta", {})).get("content"):
            if stop is None or not stop.is_set():
                times.append(time.monotonic())
    connection.close()
    if len(times) > 1:
        out.append((len(times) - 1) / (times[-1] - times[0]))


def long(args: argparse.Namespace) -> int:
    languages = str(args.languages).split(",")
    clips = [c for c in corpus(args.speech) if "all" in languages or c["language"] in languages]
    minutes = sorted(float(m) for m in str(args.minutes).split(","))
    print("transcribing each clip alone…")
    alone = alone_texts(args, clips)
    results: list[Long] = []
    for m in minutes:
        path, placed = build(clips, args.speech, m)
        audio = path.read_bytes()
        server = Server(args.nuclis, args.model, args.chat)
        try:
            idle = server.rss()
            peak = [idle]
            running = threading.Event()
            running.set()

            def sample(server: Server = server, peak: list[int] = peak, running: threading.Event = running) -> None:
                while running.is_set():
                    peak[0] = max(peak[0], server.rss())
                    time.sleep(0.05)

            sampler = threading.Thread(target=sample)
            sampler.start()
            rates: list[float] = []
            chat_thread: threading.Thread | None = None
            stop = threading.Event()
            if args.chat:
                chat_thread = threading.Thread(target=chat_rate, args=(server.port, args.chat, stop, rates))
                chat_thread.start()
                time.sleep(2)
            started = time.monotonic()
            fields = [("model", server.model), ("response_format", "verbose_json"), ("language", "auto")]
            result = transcribe_http(server.port, fields, audio, path.name)
            wall = time.monotonic() - started
            stop.set()
            running.clear()
            sampler.join()
            if chat_thread is not None:
                chat_thread.join()
                alone_rates: list[float] = []
                chat_rate(server.port, args.chat, None, alone_rates)
                print(
                    f"  chat: {rates[0] if rates else 0:.1f} tokens/s beside the transcription, {alone_rates[0]:.1f} alone"
                )
            streamed = transcribe_stream(server.port, fields[:1] + fields[2:], audio, path.name)
        finally:
            server.close()
        score = check_segments(result, placed, alone)
        if (
            streamed.done is None
            or streamed.done["text"] != result["text"]
            or "".join(streamed.deltas) != result["text"]
        ):
            raise Failure("the stream's text differs from the response's")
        audio_s = cast(float, result["duration"])
        r = Long(m, len(audio), audio_s, wall, peak[0], idle, len(cast(list[Json], result["segments"])), score)
        results.append(r)
        print(
            f"{m:g} min: {len(placed)} clips, {r.segments} segments, {audio_s:.1f} s of audio in {wall:.1f} s "
            f"({audio_s / wall:.1f}× real time); first segment {streamed.first_segment_s or 0:.2f} s into the stream "
            f"({streamed.progress} progress comments); upload {len(audio) / 2**20:.1f} MiB; RSS {idle / 2**20:.0f} MiB idle, "
            f"peak {peak[0] / 2**20:.0f} MiB; {score.drift} word edits against the clips alone in {score.words} words "
            f"({score.drift / max(1, score.words):.2%}), {score.unequal} clips unequal; WER against the references "
            f"{score.errors / max(1, score.reference_words):.2%} (the clips alone {score.alone_errors / max(1, score.reference_words):.2%}); "
            f"{len(score.misplaced)} segments across a join"
        )
        for line in score.misplaced[:10]:
            print(f"  across a join: {line}")
    failed = False
    for r in results:
        if r.score.misplaced:
            print(f"FAIL {r.minutes:g} min: {len(r.score.misplaced)} segments cross a join", file=sys.stderr)
            failed = True
        if r.score.drift > args.max_drift * r.score.words:
            print(f"FAIL {r.minutes:g} min: {r.score.drift} word edits, above {args.max_drift:.1%}", file=sys.stderr)
            failed = True
    if len(results) > 1 and not args.chat:
        short, longest = results[0], results[-1]
        allowed = (longest.upload - short.upload) + args.rss_margin_mb * 2**20
        grown = longest.peak_rss - short.peak_rss
        print(
            f"peak RSS {short.minutes:g} → {longest.minutes:g} min: +{grown / 2**20:.0f} MiB (allowed +{allowed / 2**20:.0f})"
        )
        if grown > allowed:
            print("FAIL: peak RSS grows with the duration beyond the upload", file=sys.stderr)
            failed = True
    return 1 if failed else 0


class Tests(unittest.TestCase):
    def test_multipart(self) -> None:
        body, content_type = multipart([("model", "m")], b"RIFF", "a.wav")
        self.assertTrue(content_type.endswith("boundary=nuclis-check-6f1d2b"))
        self.assertIn(b'name="model"\r\n\r\nm\r\n', body)
        self.assertTrue(body.endswith(b"RIFF\r\n--nuclis-check-6f1d2b--\r\n"))

    def test_wav(self) -> None:
        pcm = struct.pack("<3h", 0, 16384, -32768)
        self.assertEqual(read_wav(wav16(pcm)), pcm)
        self.assertEqual(pcm16([0.0, 0.5, -1.0, 2.0]), struct.pack("<4h", 0, 16384, -32768, 32767))

    def test_placement(self) -> None:
        placed = [Placed({"id": "a"}, 0.0, 2.0), Placed({"id": "b"}, 4.0, 6.0)]
        alone = {"a": ["one", "two"], "b": ["three"]}
        result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "One"},
                {"start": 1.2, "end": 2.08, "text": "two."},
                {"start": 3.92, "end": 5.0, "text": "Three!"},
            ]
        }
        score = check_segments(result, placed, alone)
        self.assertEqual((score.misplaced, score.drift, score.words, score.unequal), ([], 0, 3, 0))
        across = {"segments": [{"start": 1.0, "end": 4.5, "text": "two three"}]}
        score = check_segments(across, placed, alone)
        self.assertEqual(len(score.misplaced), 1)
        self.assertEqual((score.drift, score.unequal), (3, 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("route", "long", "self-test"))
    parser.add_argument("--nuclis", default=str(ROOT / "zig-out/bin/nuclis"), help="the binary to run")
    parser.add_argument("--model", help="the checkpoint directory (with --chat, a catalogue name)")
    parser.add_argument("--backend", default="metal", choices=("cpu", "metal"), help="the CLI's, as the server's")
    parser.add_argument("--speech", type=pathlib.Path, default=wer.SPEECH, help="the fetched corpus")
    parser.add_argument("--minutes", default="5,30", help="long: the recordings' durations")
    parser.add_argument("--languages", default="en-US", help="long: the clips' locales, comma-separated, or all")
    parser.add_argument("--max-drift", type=float, default=0.03, help="long: edits against the clips alone, over words")
    parser.add_argument("--rss-margin-mb", type=int, default=64, help="long: peak RSS growth allowed beyond the upload")
    parser.add_argument("--chat", help="long: a language model to stream a chat beside it (the user's home)")
    args = parser.parse_args()
    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1
    if not args.model:
        parser.error("--model is required")
    if not shutil.which("afconvert") and args.mode == "long":
        parser.error("long needs afconvert (macOS)")
    try:
        return route(args) if args.mode == "route" else long(args)
    except Failure as failure:
        print(f"transcription-check: FAILED: {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
