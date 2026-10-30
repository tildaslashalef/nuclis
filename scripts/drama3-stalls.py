"""Short-phrase stall check for Drama 3 through `nuclis speak` and Nemotron.

Saves each reference clip (a WAV beside a .txt transcript) as a voice under a
throwaway NUCLIS_HOME whose `models` links to the real one, speaks every phrase
with every voice and seed, and scores each take: a stall when Nemotron hears
under 60 % of the phrase's letters, or the take ends in more than 1.5 s of
silence. The method is docs/models/drama3.md § Stalls, and what nuclis adds.

    python3 scripts/drama3-stalls.py --refs ~/Code/ttslab/voicerefs/doo-v4-12s \
        --refs ~/Code/ttslab/voicerefs/nda-v2-12s --seeds 1 2 3 4 --out .zig-cache/stalls
"""

import argparse
import array
import json
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

NUCLIS = str(Path(__file__).resolve().parent.parent / "zig-out" / "bin" / "nuclis")

# Short Egyptian phrases; the lab's four worst first.
PHRASES = [
    "إزيك يا باشا؟ عامل إيه؟",
    "طب استنى بس.",
    "الجو حلو قوي النهاردة.",
    "إيه الحلاوة دي؟ مش مصدق!",
    "يا ستي ولا يهمك.",
    "معلش اتأخرت عليكم.",
    "تعالى نتمشى شوية على الكورنيش.",
    "هو إنت بتتكلم بجد ولا بتهزر؟",
]


def normalize(text: str) -> str:
    text = re.sub(r"[ً-ْٰـ]", "", text)  # diacritics, tatweel
    text = re.sub("[إأآا]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w]", "", text)


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def tail_silence(path: Path, threshold: float = 0.02, window: int = 2048) -> tuple[float, float]:
    """The take's length and its trailing run of windows under `threshold` RMS, in seconds."""
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        samples = array.array("h", w.readframes(w.getnframes()))
    quiet = 0
    for end in range(len(samples), 0, -window):
        chunk = samples[max(0, end - window) : end]
        rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5 / 32768
        if rms >= threshold:
            break
        quiet += len(chunk)
    return len(samples) / rate, quiet / rate


def run(command: list[str], env: dict[str, str]) -> str:
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        sys.exit(f"{' '.join(command[:3])} failed:\n{done.stderr}")
    return done.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-phrase stall check for Drama 3.")
    parser.add_argument("--refs", action="append", required=True, help="a clip's path without .wav/.txt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--phrases", type=int, default=len(PHRASES), help="the first N phrases")
    parser.add_argument("--model", default="drama-3")
    parser.add_argument("--nuclis", default=NUCLIS, help="the binary to run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out: Path = args.out
    home = out / "home"
    (home / "voices").mkdir(parents=True, exist_ok=True)
    models = home / "models"
    if not models.exists():
        models.symlink_to(Path(os.environ.get("NUCLIS_HOME", Path.home() / ".nuclis")) / "models")
    env = {**os.environ, "NUCLIS_HOME": str(home.resolve())}

    voices = []
    for ref in args.refs:
        ref = Path(ref).expanduser()
        name = re.sub(r"[^a-z0-9_-]", "-", ref.name.lower())
        if not (home / "voices" / name).exists():
            text = ref.with_suffix(".txt").read_text().strip()
            run([args.nuclis, "voice", "add", name, str(ref.with_suffix(".wav")), "--text", text], env)
        voices.append(name)

    rows = []
    for voice in voices:
        for i, phrase in enumerate(PHRASES[: args.phrases]):
            for seed in args.seeds:
                wav = out / f"{voice}-p{i}-s{seed}.wav"
                log = run(
                    [
                        args.nuclis,
                        "speak",
                        "--model",
                        args.model,
                        "--voice",
                        voice,
                        "--seed",
                        str(seed),
                        "-o",
                        str(wav),
                        phrase,
                    ],
                    env,
                )
                heard = json.loads(
                    run([args.nuclis, "transcribe", str(wav), "--language", "ar", "--json"], env) or "{}"
                ).get("text", "")
                seconds, tail = tail_silence(wav)
                reference, hypothesis = normalize(phrase), normalize(heard)
                coverage = min(len(hypothesis) / max(len(reference), 1), 1.0)
                row = {
                    "voice": voice,
                    "phrase": i,
                    "seed": seed,
                    "text": phrase,
                    "heard": heard,
                    "seconds": round(seconds, 2),
                    "tail_silence": round(tail, 2),
                    "cer": round(distance(reference, hypothesis) / max(len(reference), 1), 3),
                    "coverage": round(coverage, 3),
                    "retried": "retried" in log,
                    "stall": coverage < 0.6 or tail > 1.5,
                }
                rows.append(row)
                print(
                    f"{voice:14} p{i} s{seed}: {seconds:5.2f} s, tail {tail:4.2f} s, coverage {coverage:.2f}{'  STALL' if row['stall'] else ''}",
                    flush=True,
                )
    (out / "scored.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    stalls = [r for r in rows if r["stall"]]
    print(
        f"\n{len(stalls)} stalls of {len(rows)} takes; {sum(r['retried'] for r in rows)} takes retried a silent start"
    )
    for r in stalls:
        print(f"  {r['voice']} p{r['phrase']} s{r['seed']}: {r['text']} -> {r['heard']}")


if __name__ == "__main__":
    main()
