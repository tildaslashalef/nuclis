#!/usr/bin/env python3
"""The voice loop through one `nuclis serve`: ears, brain, and voice together, as Muon runs them.

    python3 scripts/voice-loop.py beside --rounds 3     # speech beside a chat stream
    python3 scripts/voice-loop.py loop --rounds 3       # a spoken question answered aloud

Both start `nuclis serve --model <--chat> --pin <--voice-model> --pin
<--ears>` (the user's home and models) and warm each model once.

`beside` measures the chat model's streamed tokens per second alone, then
while `--sentences` are spoken beside it (each its own request, sent
together, as a client queues an answer's sentences), and the speech's
lead over playback.

`loop` speaks a question with the server's own voice (once, cached under
`.zig-cache/voice-loop/`, resampled to 16 kHz by macOS's `afconvert`),
streams it at real time into `GET /v1/audio/transcriptions/stream`
followed by 0.4 s of silence (where a client's end-of-turn detector
would send `finish`), sends
the final transcript to chat, and sends each sentence of the streamed
answer to `POST /v1/audio/speech` as soon as it ends. It reports the
delays after `finish`: the final transcript, the first token, the first
sentence, the first audio byte; and the playback.

Playback is modelled, not played: audio starts when its first byte
arrives and plays at real time, each request's after the one before it.
The lead is how far the audio received is ahead of the playback cursor
when a chunk arrives; an underrun is a chunk that arrives after the
cursor ran out (the listener hears a gap). docs/guide/api.md § Measured
rates.
"""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / ".zig-cache/voice-loop"
SPEECH_RATE = 24000

_spec = importlib.util.spec_from_file_location("ws_transcribe", ROOT / "scripts/ws-transcribe.py")
assert _spec and _spec.loader
ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ws)

TRAILING_MS = 400
QUESTION = "What is the tallest mountain in Africa, and how long does it take to climb it?"
SYSTEM = "You are a voice assistant. Answer in three or four short spoken sentences, without lists or markup."
SENTENCES = [
    "The lighthouse stood at the edge of the world, and every night its keeper climbed the stairs.",
    "He counted them as he went, one hundred and twelve, the same as the night before.",
    "At the top he trimmed the wick, wiped the glass, and looked out over the black water.",
    "Somewhere out there a ship was waiting for his light, and he would not let it down.",
    "When the morning came he wrote a single line in the logbook and went to sleep.",
]


# ----- speech -----


@dataclass
class Speech:
    """One speech request: the arrival time and length in seconds of each audio chunk."""

    sent: float = 0.0
    chunks: list[tuple[float, float]] = field(default_factory=list)
    error: str | None = None

    @property
    def seconds(self) -> float:
        return sum(s for _, s in self.chunks)


def speak(port: int, model: str, voice: str, text: str, out: Speech) -> None:
    body = {"model": model, "input": text, "voice": voice, "response_format": "pcm", "seed": 1}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        out.sent = time.monotonic()
        connection.request("POST", "/v1/audio/speech", json.dumps(body), {"content-type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            out.error = f"{response.status} {response.read().decode(errors='replace')}"
            return
        odd = 0
        while True:
            data = response.read1(65536)
            if not data:
                break
            whole = (odd + len(data)) // 2
            odd = (odd + len(data)) % 2
            out.chunks.append((time.monotonic(), whole / SPEECH_RATE))
    finally:
        connection.close()


@dataclass
class Playback:
    """Requests played one after another at real time from their first byte."""

    leads: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    first: float | None = None
    audio: float = 0.0

    def play(self, requests: list[Speech]) -> None:
        chunks = [c for r in requests for c in r.chunks]
        end = None
        for at, seconds in chunks:
            if end is None:
                self.first = at
                end = at
            elif at > end:
                self.gaps.append(at - end)
                end = at
            else:
                self.leads.append(end - at)
            end += seconds
            self.audio += seconds

    def summary(self) -> str:
        lead = (
            f"lead min {min(self.leads):.2f} s, median {statistics.median(self.leads):.2f} s"
            if self.leads
            else "no lead"
        )
        gaps = f"{len(self.gaps)} underruns ({sum(self.gaps) * 1000:.0f} ms)" if self.gaps else "no underrun"
        return f"{self.audio:.1f} s of audio, {lead}, {gaps}"


def speak_all(port: int, args: argparse.Namespace, texts: list[str]) -> list[Speech]:
    """Each text its own request, all sent together (the server speaks them in order)."""
    requests = [Speech() for _ in texts]
    threads = []
    for text, request in zip(texts, requests):
        thread = threading.Thread(target=speak, args=(port, args.voice_model, args.voice, text, request))
        thread.start()
        threads.append(thread)
        time.sleep(0.01)  # arrival order
    for thread in threads:
        thread.join()
    for r in requests:
        if r.error:
            raise ws.Failure(f"speech: {r.error}")
    return requests


# ----- chat -----


def chat(port: int, model: str, messages: list[dict[str, str]], max_tokens: int, on_text: Any) -> None:
    body = {
        "model": model,
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "messages": messages,
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ws.Failure(f"chat: {response.status} {response.read().decode(errors='replace')}")
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choices = json.loads(line[6:]).get("choices") or []
            text = (choices[0].get("delta") or {}).get("content") if choices else None
            if text:
                on_text(time.monotonic(), text)
    finally:
        connection.close()


def chat_rate(port: int, args: argparse.Namespace) -> Any:
    run = ws.ChatRun()
    ws.chat_stream(port, args.chat, args.max_tokens, run)
    if run.error:
        raise ws.Failure(f"chat: {run.error}")
    return run


# ----- modes -----


def start_server(args: argparse.Namespace) -> Any:
    return ws.Server(args.nuclis, ["--model", args.chat, "--pin", args.voice_model, "--pin", args.ears])


def warm(port: int, args: argparse.Namespace) -> None:
    speak_all(port, args, ["Warming up."])
    chat(port, args.chat, [{"role": "user", "content": "Say hi."}], 8, lambda _t, _x: None)


def beside(args: argparse.Namespace) -> int:
    server = start_server(args)
    try:
        warm(server.port, args)
        alone = chat_rate(server.port, args)
        print(f"{args.chat} alone: {ws.rate(alone.stamps, 0, float('inf')):.1f} tokens/s")
        texts = SENTENCES[: args.sentences]
        solo = Playback()
        requests = speak_all(server.port, args, texts)
        solo.play(requests)
        made = requests[-1].chunks[-1][0] - requests[0].sent
        print(
            f"speech alone: first byte {(requests[0].chunks[0][0] - requests[0].sent) * 1000:.0f} ms, {solo.audio / made:.2f}x real time; {solo.summary()}"
        )
        for _ in range(args.rounds):
            run = ws.ChatRun()
            thread = threading.Thread(target=ws.chat_stream, args=(server.port, args.chat, args.max_tokens, run))
            thread.start()
            time.sleep(1.0)
            requests = speak_all(server.port, args, texts)
            playback = Playback()
            playback.play(requests)
            thread.join()
            if run.error:
                raise ws.Failure(f"chat: {run.error}")
            start, end = requests[0].sent, requests[-1].chunks[-1][0]
            first = (requests[0].chunks[0][0] - requests[0].sent) * 1000
            print(
                f"beside: {args.chat} {ws.rate(run.stamps, start, end):.1f} tokens/s while speaking ({end - start:.1f} s), "
                f"{ws.rate(run.stamps, end, float('inf')):.1f} after; speech first byte {first:.0f} ms, "
                f"{playback.audio / (end - start):.2f}x real time; {playback.summary()}"
            )
    finally:
        server.stop()
    return 0


def question_pcm(port: int, args: argparse.Namespace) -> bytes:
    """The question as 16 kHz s16le, spoken once by the server's voice."""
    CACHE.mkdir(parents=True, exist_ok=True)
    wav16 = CACHE / f"question-{args.voice}.wav"
    if not wav16.is_file():
        body = {"model": args.voice_model, "input": QUESTION, "voice": args.voice, "response_format": "pcm", "seed": 1}
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        connection.request("POST", "/v1/audio/speech", json.dumps(body), {"content-type": "application/json"})
        response = connection.getresponse()
        pcm = response.read()
        connection.close()
        if response.status != 200 or not pcm:
            raise ws.Failure(f"the question was not spoken: {response.status}")
        wav24 = CACHE / "question-24k.wav"
        with wave.open(str(wav24), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SPEECH_RATE)
            w.writeframes(pcm)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(wav24), str(wav16)], check=True)
    # A turn ends in silence: the client's end-of-turn detector waits for it.
    return ws.load_pcm(wav16) + bytes(2 * 16000 * TRAILING_MS // 1000)


@dataclass
class Turn:
    spoken_end: float = 0.0
    final: float = 0.0
    transcript: str = ""
    first_token: float = 0.0
    first_sentence: float = 0.0
    stamps: list[float] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    requests: list[Speech] = field(default_factory=list)


SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def turn(port: int, args: argparse.Namespace, session: Any, pcm: bytes) -> Turn:
    t = Turn()
    result = session.stream(pcm, 1.0, 20, "en-US")
    t.spoken_end = result.finished
    t.final = result.finished + result.final_delay_ms / 1000
    t.transcript = (result.final or {}).get("text", "")
    pending = ""
    threads: list[threading.Thread] = []

    def queue(text: str) -> None:
        request = Speech()
        t.requests.append(request)
        t.sentences.append(text)
        thread = threading.Thread(target=speak, args=(port, args.voice_model, args.voice, text, request))
        thread.start()
        threads.append(thread)

    def on_text(at: float, text: str) -> None:
        nonlocal pending
        if not t.stamps:
            t.first_token = at
        t.stamps.append(at)
        pending += text
        while (m := SENTENCE_END.search(pending)) is not None:
            sentence, pending = pending[: m.end()].strip(), pending[m.end() :]
            if sentence:
                if not t.first_sentence:
                    t.first_sentence = at
                queue(sentence)

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t.transcript}]
    chat(port, args.chat, messages, args.max_tokens, on_text)
    if pending.strip():
        queue(pending.strip())
    for thread in threads:
        thread.join()
    for r in t.requests:
        if r.error:
            raise ws.Failure(f"speech: {r.error}")
    return t


def loop(args: argparse.Namespace) -> int:
    server = start_server(args)
    try:
        warm(server.port, args)
        pcm = question_pcm(server.port, args)
        session = ws.Session("127.0.0.1", server.port, "lookahead=3")
        session.stream(pcm[: 16000 * 2], 0, 20, "en-US")  # warm the recognizer
        rows = []
        for i in range(args.rounds):
            t = turn(server.port, args, session, pcm)
            playback = Playback()
            playback.play(t.requests)
            first_audio = playback.first or 0.0
            last_token = t.stamps[-1] if t.stamps else t.first_token
            row = {
                "final": (t.final - t.spoken_end) * 1000,
                "first_token": (t.first_token - t.spoken_end) * 1000,
                "first_sentence": (t.first_sentence - t.spoken_end) * 1000,
                "first_audio": (first_audio - t.spoken_end) * 1000,
                "rate": (len(t.stamps) - 1) / (last_token - t.first_token) if len(t.stamps) > 1 else 0.0,
            }
            rows.append(row)
            if i == 0:
                print(f"heard: {t.transcript!r}")
                print(f"answer: {' '.join(t.sentences)!r}")
            print(
                f"round {i + 1}: after finish: final {row['final']:.0f} ms, first token {row['first_token']:.0f} ms, "
                f"first sentence {row['first_sentence']:.0f} ms, first audio {row['first_audio']:.0f} ms; "
                f"{args.chat} {row['rate']:.1f} tokens/s over {len(t.stamps)} chunks; {len(t.sentences)} sentences, {playback.summary()}"
            )
        session.close()
        for key in ("final", "first_token", "first_sentence", "first_audio"):
            values = [r[key] for r in rows]
            print(f"{key}: median {statistics.median(values):.0f} ms (range {min(values):.0f}-{max(values):.0f})")
    finally:
        server.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["beside", "loop"])
    parser.add_argument("--nuclis", default=str(ROOT / "zig-out/bin/nuclis"))
    parser.add_argument("--chat", default="gemma-4-e4b-qat")
    parser.add_argument("--voice-model", default="qwen3-tts-1.7b")
    parser.add_argument("--voice", default="ryan")
    parser.add_argument("--ears", default="nemotron-asr-0.6b")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sentences", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    try:
        return beside(args) if args.mode == "beside" else loop(args)
    except ws.Failure as e:
        print(f"voice-loop: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
