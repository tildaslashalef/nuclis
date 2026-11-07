#!/usr/bin/env python3
"""Does speech stay ahead of its playback beside a chat prefill?

Starts a fresh `nuclis serve` per brain (or uses `--url`), sends a chat whose
system prompt opens with a unique line so nothing is reused, and 0.3 s later
one speech request read as a stream. The audio is timestamped per 100 ms and
played through a modelled player: playback starts with the first byte, runs
at real time, and pauses while it has nothing (a stall). `--beside` picks what
runs beside it (`alone` is the cold chat with no speech, for what the prefill
costs); the method is docs/guide/api.md § The voice loop.

    python3 scripts/bench/speech-beside-chat.py --brain gemma-4-e4b-qat --runs 4
"""

import argparse
import http.client
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

NUCLIS = str(Path(__file__).resolve().parents[2] / "zig-out" / "bin" / "nuclis")
SENTENCE = (
    "The old lighthouse on the northern cape has guided fishing boats home through "
    "fog and winter storms for nearly two hundred years, and its keeper still climbs "
    "the stairs every evening."
)
WORDS = "harbour tide lantern gull rope anchor current chart compass keel mast sail reef shoal".split()
RATE = 24000
BYTES_PER_100MS = RATE * 2 // 10


def filler(tokens: int, unique: str) -> str:
    """A system prompt of about `tokens` tokens (these words are one token each)."""
    rng = random.Random(unique)
    return f"Session {unique}.\n" + " ".join(rng.choice(WORDS) for _ in range(tokens))


def connection(url: str) -> http.client.HTTPConnection:
    parsed = urlparse(url)
    return http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port, timeout=300)


def chat(url, model, system, effort, out):
    """Streams one completion; `out` gets the first token's time and the usage."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": "Name one sea bird."}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 48,
        "temperature": 0,
        "reasoning_effort": effort,
    }
    start = time.monotonic()
    c = connection(url)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    response = c.getresponse()
    first = None
    while line := response.readline():
        if not line.startswith(b"data: {"):
            continue
        chunk = json.loads(line[6:])
        delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
        if first is None and (delta.get("content") or delta.get("reasoning_content")):
            first = time.monotonic() - start
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
    out["first_token"] = first
    out["total"] = time.monotonic() - start
    c.close()


def speak(url, voice_model, voice):
    """One speech request: the modelled player's stalls and the generate time."""
    body = {"model": voice_model, "input": SENTENCE, "voice": voice, "response_format": "pcm", "seed": 0}
    start = time.monotonic()
    c = connection(url)
    c.request("POST", "/v1/audio/speech", json.dumps(body), {"content-type": "application/json"})
    response = c.getresponse()
    if response.status != 200:
        raise SystemExit(f"speech: {response.status} {response.read()[:300]!r}")
    received = 0
    first_byte = start
    play_end = start  # the moment the player runs out of what it has
    stalls = []
    when = []
    while True:
        data = response.read1(BYTES_PER_100MS)
        now = time.monotonic()
        if not data:
            break
        if received == 0:
            first_byte = now
            play_end = now
        if now > play_end + 0.005:
            stalls.append(now - play_end)
            when.append(now - first_byte)
            play_end = now
        play_end += len(data) / (2 * RATE)
        received += len(data)
    end = time.monotonic()
    c.close()
    seconds = received / (2 * RATE)
    return {
        "audio": seconds,
        "first_byte": first_byte - start,
        "generate": end - first_byte,
        "x": (end - first_byte) / seconds,
        "stalls": len(stalls),
        "silent": sum(stalls),
        "worst": max(stalls, default=0),
        "at": [round(t, 1) for t in when],
    }


def one(url, brain, voice_model, voice, beside, tokens):
    chatted = {}
    thread = None
    alone = beside == "alone"
    if beside != "nothing":
        effort = "none" if beside.endswith("off") else "low"
        unique = "cached" if beside == "cached" else f"{random.getrandbits(48):012x}"
        if beside == "cached":
            effort = "none"
        args = (url, brain, filler(tokens, unique), effort, chatted)
        thread = threading.Thread(target=chat, args=args)
        thread.start()
        time.sleep(0.3)
    heard = None if alone else speak(url, voice_model, voice)
    if thread:
        thread.join()
    return heard, chatted


def serve(args):
    command = [args.nuclis, "serve", "--quiet", "--port", str(args.port), "--memory", str(args.memory)]
    for pin in [args.brain, args.voice_model]:
        command += ["--pin", pin]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{args.port}"
    for _ in range(600):
        try:
            c = connection(url)
            c.request("GET", "/v1/health")
            if c.getresponse().status == 200:
                return process, url
        except OSError:
            pass
        if process.poll() is not None:
            raise SystemExit(f"the server exited with {process.returncode}")
        time.sleep(0.5)
    raise SystemExit("the server did not come up")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", default="gemma-4-e4b-qat")
    parser.add_argument("--voice-model", default="qwen3-tts-1.7b-customvoice")
    parser.add_argument("--voice", default="sohee")
    parser.add_argument("--beside", nargs="+", default=["nothing", "alone", "cached", "cold-off", "cold-low"])
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2700)
    parser.add_argument("--port", type=int, default=9482)
    parser.add_argument("--memory", type=int, default=40)
    parser.add_argument("--nuclis", default=NUCLIS)
    parser.add_argument("--url", help="a server already running; none is started")
    parser.add_argument("--json", help="also write every run here")
    args = parser.parse_args()
    process = None
    url = args.url
    if not url:
        process, url = serve(args)
    records = []
    try:
        print(f"{args.brain}, {args.tokens} tokens")
        print(
            "| beside | first byte s | generate s | x | stalls | silent s | worst s | first token s | prompt (reused) | stalls at s |"
        )
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for beside in args.beside:
            if beside == "cached":
                one(url, args.brain, args.voice_model, args.voice, "cached", args.tokens)
            for _ in range(args.runs):
                heard, chatted = one(url, args.brain, args.voice_model, args.voice, beside, args.tokens)
                usage = chatted.get("usage", {})
                heard = heard or dict.fromkeys(["first_byte", "generate", "x", "stalls", "silent", "worst", "at"], 0)
                reused = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                print(
                    f"| {beside} | {heard['first_byte']:.2f} | {heard['generate']:.2f} | {heard['x']:.2f} "
                    f"| {heard['stalls']} | {heard['silent']:.2f} | {heard['worst']:.2f} "
                    f"| {chatted.get('first_token') or 0:.2f} | {usage.get('prompt_tokens', 0)} ({reused}) | {heard['at']} |",
                    flush=True,
                )
                records.append({"beside": beside, "heard": heard, "chat": chatted})
    finally:
        if process:
            process.terminate()
            process.wait()
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    sys.exit(main())
