#!/usr/bin/env python3
"""Does speech stay ahead of its playback beside a chat prefill?

Starts a fresh `nuclis serve` per brain (or uses `--url`), sends a chat whose
system prompt opens with a unique line so nothing is reused, and 0.3 s later
one speech request read as a stream. The audio is timestamped per 100 ms and
played through a modelled player: playback starts with the first byte, runs
at real time, and pauses while it has nothing (a stall). `--beside` picks what
runs beside it (`alone` is the cold chat with no speech, for what the prefill
costs); the method is docs/benchmarks/record/api.md § The voice loop.

    python3 scripts/bench/speech-beside-chat.py --brain gemma-4-e4b-qat --runs 4
"""

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import serve  # noqa: E402

NUCLIS = str(ROOT / "zig-out" / "bin" / "nuclis")
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


def where(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname or serve.HOST, parsed.port or 80


def chat(url, model, system, effort, out):
    """Streams one completion; `out` gets the first token's time and the usage."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": "Name one sea bird."}],
        "stream_options": {"include_usage": True},
        "max_tokens": 48,
        "temperature": 0,
        "reasoning_effort": effort,
    }
    host, port = where(url)
    start = time.monotonic()
    first = None
    for at, chunk in serve.chat_stream(port, body, timeout=300, host=host):
        delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
        if first is None and (delta.get("content") or delta.get("reasoning_content")):
            first = at - start
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
    out["first_token"] = first
    out["total"] = time.monotonic() - start


def speak(url, voice_model, voice):
    """One speech request: the modelled player's stalls and the generate time."""
    body = {"model": voice_model, "input": SENTENCE, "voice": voice, "response_format": "pcm", "seed": 0}
    host, port = where(url)
    start = time.monotonic()
    received = 0
    first_byte = start
    play_end = start  # the moment the player runs out of what it has
    stalls = []
    when = []
    try:
        for now, data in serve.speech_request(port, body, piece=BYTES_PER_100MS, timeout=300, host=host):
            if received == 0:
                first_byte = now
                play_end = now
            if now > play_end + 0.005:
                stalls.append(now - play_end)
                when.append(now - first_byte)
                play_end = now
            play_end += len(data) / (2 * RATE)
            received += len(data)
    except serve.Failed as failure:
        raise SystemExit(f"speech: {failure}") from None
    end = time.monotonic()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", default="gemma-4-e4b-qat")
    parser.add_argument("--voice-model", default="qwen3-tts-1.7b-customvoice")
    parser.add_argument("--voice", default="sohee")
    parser.add_argument("--beside", nargs="+", default=["nothing", "alone", "cached", "cold-off", "cold-low"])
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2700)
    parser.add_argument("--memory", type=int, default=40)
    parser.add_argument("--nuclis", default=NUCLIS)
    parser.add_argument("--url", help="a server already running; none is started")
    parser.add_argument("--json", help="also write every run here")
    args = parser.parse_args()
    server = None
    url = args.url
    if not url:
        pins = ["--pin", args.brain, "--pin", args.voice_model]
        server = serve.Server(args.nuclis, ["--memory", str(args.memory), *pins])
        url = f"http://{serve.HOST}:{server.port}"
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
        if server:
            server.stop()
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    sys.exit(main())
