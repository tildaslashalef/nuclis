#!/usr/bin/env python3
"""Word error rate of `nuclis transcribe` on the speech corpus.

Runs the given binary (by default the fresh `./zig-out/bin/nuclis`) as

    nuclis transcribe <clip> --json --backend cpu --model <dir> --language <locale>

over the clips of `tests/fixtures/speech-corpus.json` (fetched by `make
speech-corpus` into `.reference/speech/`), each under its own locale unless
`--auto`, and reports the word error rate per language and over all: the
edits (substitutions, deletions, insertions) of a word-level alignment over
the reference's words, summed over clips before dividing.

Both sides are normalized alike: NFKC, case folded, `ё` as `е`, apostrophes
dropped, every other punctuation mark a space. The reference is FLEURS's
normalized transcript where the corpus has one (what the speaker said, which
can differ from the written sentence), else the transcript (LibriSpeech's).

`--max-wer` makes it a gate: the overall rate must not exceed it. Standard
library only. docs/models/nemotron-asr.md § Transducer against the trace,
and word error rate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import unicodedata
import unittest
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests/fixtures/speech-corpus.json"
SPEECH = ROOT / ".reference/speech"
APOSTROPHES = "'’ʼ`´"


def normalize(text: str) -> list[str]:
    """The words of `text`, normalized for scoring."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    out: list[str] = []
    for ch in text:
        if ch in APOSTROPHES:
            continue
        out.append(" " if unicodedata.category(ch).startswith("P") or ch.isspace() else ch)
    return "".join(out).split()


def edits(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over words: substitutions, deletions, insertions."""
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i] + [0] * len(hypothesis)
        for j, other in enumerate(hypothesis, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other))
        previous = current
    return previous[-1]


@dataclass
class Result:
    clip: str
    language: str
    reference: list[str]
    hypothesis: list[str]
    errors: int
    seconds: float
    transcribe_ms: float
    detected: str | None


def transcribe(
    nuclis: str, model: str, clip: dict[str, Any], speech: pathlib.Path, auto: bool, lookahead: int
) -> Result:
    path = speech / clip["file"]
    if not path.is_file():
        raise SystemExit(f"{path}: missing; run make speech-corpus")
    command = [nuclis, "transcribe", str(path), "--json", "--backend", "cpu", "--lookahead", str(lookahead)]
    command += ["--model", model, "--language", "auto" if auto else clip["language"]]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"{' '.join(command)}: exit {done.returncode}\n{done.stderr}")
    report = json.loads(done.stdout)
    reference = normalize(clip.get("normalized") or clip["text"])
    hypothesis = normalize(report["text"])
    return Result(
        clip=clip["id"],
        language=clip["language"],
        reference=reference,
        hypothesis=hypothesis,
        errors=edits(reference, hypothesis),
        seconds=report["audio_seconds"],
        transcribe_ms=report["timings"]["transcribe_ms"],
        detected=report["language"],
    )


def summarize(results: list[Result]) -> dict[str, dict[str, float]]:
    """Per language and `all`: clips, words, errors, rate, audio and transcription seconds."""
    table: dict[str, dict[str, float]] = {}
    for key in sorted({r.language for r in results}) + ["all"]:
        chosen = [r for r in results if key in ("all", r.language)]
        words = sum(len(r.reference) for r in chosen)
        errors = sum(r.errors for r in chosen)
        table[key] = {
            "clips": len(chosen),
            "words": words,
            "errors": errors,
            "wer": errors / words if words else 0.0,
            "audio_s": sum(r.seconds for r in chosen),
            "transcribe_s": sum(r.transcribe_ms for r in chosen) / 1000,
        }
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nuclis", default="./zig-out/bin/nuclis", help="the binary to run")
    parser.add_argument("--model", help="the checkpoint directory (required unless --self-test)")
    parser.add_argument("--speech", type=pathlib.Path, default=SPEECH, help="the fetched corpus (.reference/speech)")
    parser.add_argument("--clips", help="comma-separated corpus ids; default every clip")
    parser.add_argument("--auto", action="store_true", help="the prompt auto instead of each clip's locale")
    parser.add_argument("--lookahead", type=int, default=3, choices=(0, 3, 6, 13))
    parser.add_argument("--max-wer", type=float, help="fail when the overall rate exceeds this")
    parser.add_argument("--json", type=pathlib.Path, help="write every clip's result and the summary here")
    parser.add_argument("--verbose", action="store_true", help="print each clip's words")
    parser.add_argument("--self-test", action="store_true", help="run the normalizer's and the alignment's tests")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1
    if not args.model:
        parser.error("--model is required")
    corpus = json.loads(CORPUS.read_text())
    clips: list[dict[str, Any]] = corpus["clips"]
    if args.clips:
        wanted = args.clips.split(",")
        known = {c["id"]: c for c in clips}
        missing = [w for w in wanted if w not in known]
        if missing:
            raise SystemExit(f"not in the corpus: {', '.join(missing)}")
        clips = [known[w] for w in wanted]
    started = time.monotonic()
    results: list[Result] = []
    for clip in clips:
        result = transcribe(args.nuclis, args.model, clip, args.speech, args.auto, args.lookahead)
        results.append(result)
        rate = result.errors / max(1, len(result.reference))
        print(
            f"{result.clip:<34} {result.language:<6} {result.errors:>3}/{len(result.reference):<3} {rate:6.1%}  {result.detected or '-'}"
        )
        if args.verbose and result.errors:
            print(f"  ref {' '.join(result.reference)}\n  hyp {' '.join(result.hypothesis)}")
    elapsed = time.monotonic() - started
    table = summarize(results)
    print(f"\n{'language':<8} {'clips':>5} {'words':>6} {'errors':>6} {'WER':>7} {'audio s':>8} {'CPU s':>7}")
    for key, row in table.items():
        print(
            f"{key:<8} {row['clips']:>5.0f} {row['words']:>6.0f} {row['errors']:>6.0f} {row['wer']:>7.2%} "
            f"{row['audio_s']:>8.1f} {row['transcribe_s']:>7.1f}"
        )
    print(f"wall time {elapsed:.1f} s (each clip a fresh process: model load included)")
    if args.json:
        payload = {"results": [r.__dict__ for r in results], "summary": table, "wall_s": elapsed}
        args.json.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    if args.max_wer is not None and table["all"]["wer"] > args.max_wer:
        print(f"WER {table['all']['wer']:.2%} exceeds the bound {args.max_wer:.2%}", file=sys.stderr)
        return 1
    return 0


class Tests(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(
            normalize("Il n'a pas chiffré ces réductions, disant…"),
            ["il", "na", "pas", "chiffré", "ces", "réductions", "disant"],
        )
        self.assertEqual(normalize("Das Cochamó-Tal – Chiles"), ["das", "cochamó", "tal", "chiles"])
        self.assertEqual(normalize("ЁМКОСТИ, ёлка"), ["емкости", "елка"])
        self.assertEqual(normalize("HE BEGAN A CONFUSED COMPLAINT"), normalize("He began a confused complaint."))

    def test_edits(self) -> None:
        self.assertEqual(edits(["a", "b", "c"], ["a", "b", "c"]), 0)
        self.assertEqual(edits(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(edits(["a", "b", "c"], ["a", "c"]), 1)
        self.assertEqual(edits(["a", "b"], ["a", "b", "c", "d"]), 2)
        self.assertEqual(edits([], ["a"]), 1)


if __name__ == "__main__":
    sys.exit(main())
