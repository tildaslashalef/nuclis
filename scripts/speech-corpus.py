#!/usr/bin/env python3
"""Fetch, or pin, the speech corpus: public clips with human transcripts.

The corpus (`tests/fixtures/speech-corpus.json`) names each clip by the
archive it comes from (a URL at a pinned revision), its member path inside
it, and the SHA-256 of the clip's bytes, with its reference transcript. The
clips themselves are never committed; `make speech-corpus` runs

    python3 scripts/speech-corpus.py fetch

which streams each archive that still has a missing clip, extracts only the
listed members into `.reference/speech/`, stops reading once it has them
all, and refuses any clip whose digest differs. Standard library only.

`pin` rebuilds the JSON from the archives (downloaded whole into
`.reference/speech/archives/`, digests recorded): 40 LibriSpeech test-clean
utterances, one per speaker, and 5 FLEURS test utterances in each of four
languages, each the first in sorted order whose duration lies within bounds.
It is run once, when the corpus is chosen; the JSON is the record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import IO, Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests/fixtures/speech-corpus.json"
OUT = ROOT / ".reference/speech"
USER_AGENT = "nuclis-speech-corpus/1"

FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}/data"
# FLEURS directory, the model's locale (`prompt_dictionary` key), clips taken.
FLEURS = [("de_de", "de-DE", 5), ("fr_fr", "fr-FR", 5), ("es_419", "es-US", 5), ("ru_ru", "ru-RU", 5)]
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
# Duration bounds of a chosen clip, in samples at 16 kHz.
MIN_SAMPLES = 4 * 16000
MAX_SAMPLES = 12 * 16000


def open_url(url: str) -> IO[bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)  # noqa: S310 - pinned https URLs only


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- fetch -----------------------------------------------------------------


def fetch(corpus: dict[str, Any], out: pathlib.Path) -> int:
    """Extracts every missing clip; returns the number of clips refused."""
    sources = {s["id"]: s for s in corpus["sources"]}
    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    present = 0
    bad = 0
    for clip in corpus["clips"]:
        path = out / clip["file"]
        if path.is_file():
            if sha256(path.read_bytes()) == clip["sha256"]:
                present += 1
                continue
            print(f"{clip['file']}: digest differs, fetching again", file=sys.stderr)
        by_source.setdefault(clip["source"], {})[clip["member"]] = clip
    for source_id, wanted in by_source.items():
        source = sources[source_id]
        print(f"{source_id}: {len(wanted)} clips from {source['url']}", file=sys.stderr)
        with open_url(source["url"]) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                clip = wanted.pop(member.name, None)
                if clip is None:
                    continue
                handle = archive.extractfile(member)
                assert handle is not None
                data = handle.read()
                if sha256(data) != clip["sha256"]:
                    print(f"{clip['file']}: SHA-256 {sha256(data)} is not the pinned {clip['sha256']}", file=sys.stderr)
                    bad += 1
                    continue
                path = out / clip["file"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                present += 1
                if not wanted:
                    break
        for name in wanted:
            print(f"{source_id}: {name} is not in the archive", file=sys.stderr)
            bad += 1
    print(f"{present} of {len(corpus['clips'])} clips present and verified in {out}", file=sys.stderr)
    return bad


# ---- pin -------------------------------------------------------------------


@dataclass
class Archive:
    path: pathlib.Path
    sha256: str
    size: int


def download(url: str, path: pathlib.Path) -> Archive:
    """Downloads `url` to `path` once; the digest is of the bytes on disk."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        with open_url(url) as response, partial.open("wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        partial.rename(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return Archive(path, digest.hexdigest(), path.stat().st_size)


def flac_samples(data: bytes) -> tuple[int, int]:
    """Sample rate and total samples from a FLAC STREAMINFO block."""
    if data[:4] != b"fLaC" or data[4] & 0x7F != 0:
        raise ValueError("not a FLAC stream starting with STREAMINFO")
    info = data[8:42]
    rate = int.from_bytes(info[10:13], "big") >> 4
    total = int.from_bytes(info[13:18], "big") & ((1 << 36) - 1)
    return rate, total


def wav_samples(data: bytes) -> tuple[int, int, int]:
    """Sample rate, channels, and sample frames of a PCM RIFF/WAVE file."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    at = 12
    rate = channels = bits = 0
    while at + 8 <= len(data):
        tag, size = data[at : at + 4], int.from_bytes(data[at + 4 : at + 8], "little")
        body = data[at + 8 : at + 8 + size]
        if tag == b"fmt ":
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
        elif tag == b"data":
            return rate, channels, size // (channels * bits // 8)
        at += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def pin_librispeech(archives: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive = download(LIBRISPEECH_URL, archives / "librispeech-test-clean.tar.gz")
    texts: dict[str, str] = {}
    audio: dict[str, bytes] = {}
    with tarfile.open(archive.path, mode="r:gz") as tar:
        for member in tar:
            handle = tar.extractfile(member) if member.isfile() else None
            if handle is None:
                continue
            if member.name.endswith(".trans.txt"):
                for line in handle.read().decode().splitlines():
                    key, _, text = line.partition(" ")
                    texts[key] = text
            elif member.name.endswith(".flac"):
                audio[member.name] = handle.read()
    by_speaker: dict[str, list[str]] = {}
    for name in sorted(audio):
        by_speaker.setdefault(pathlib.PurePosixPath(name).name.split("-")[0], []).append(name)
    clips: list[dict[str, Any]] = []
    for speaker in sorted(by_speaker, key=int):
        for name in by_speaker[speaker]:
            data = audio[name]
            rate, samples = flac_samples(data)
            if rate != 16000 or not MIN_SAMPLES <= samples <= MAX_SAMPLES:
                continue
            key = pathlib.PurePosixPath(name).stem
            clips.append(
                {
                    "id": f"librispeech-{key}",
                    "source": "librispeech-test-clean",
                    "member": name,
                    "file": f"librispeech/{key}.flac",
                    "sha256": sha256(data),
                    "bytes": len(data),
                    "samples": samples,
                    "language": "en-US",
                    "text": texts[key],
                }
            )
            break
    source = {
        "id": "librispeech-test-clean",
        "url": LIBRISPEECH_URL,
        "sha256": archive.sha256,
        "bytes": archive.size,
        "license": "CC BY 4.0",
        "attribution": "LibriSpeech ASR corpus (Panayotov, Chen, Povey, Khudanpur, 2015), OpenSLR SLR12",
    }
    return source, clips


def pin_fleurs(
    archives: pathlib.Path, directory: str, locale: str, count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    url = f"{FLEURS_BASE}/{directory}/audio/test.tar.gz"
    archive = download(url, archives / f"fleurs-{directory}-test.tar.gz")
    with open_url(f"{FLEURS_BASE}/{directory}/test.tsv") as response:
        rows = list(csv.reader(io.TextIOWrapper(response, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    # Columns: sentence id, file, raw transcription, normalized, characters, samples, gender.
    chosen: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in sorted(rows, key=lambda r: (int(r[0]), r[1])):
        if len(chosen) == count:
            break
        if row[0] in seen or not MIN_SAMPLES <= int(row[5]) <= MAX_SAMPLES:
            continue
        seen.add(row[0])
        chosen[f"test/{row[1]}"] = row
    clips: list[dict[str, Any]] = []
    with tarfile.open(archive.path, mode="r:gz") as tar:
        for member in tar:
            row = chosen.get(member.name)
            handle = tar.extractfile(member) if row is not None else None
            if row is None or handle is None:
                continue
            data = handle.read()
            rate, channels, samples = wav_samples(data)
            if rate != 16000 or channels != 1:
                raise ValueError(f"{member.name}: {rate} Hz, {channels} channels")
            stem = pathlib.PurePosixPath(member.name).stem
            clips.append(
                {
                    "id": f"fleurs-{directory}-{row[0]}",
                    "source": f"fleurs-{directory}-test",
                    "member": member.name,
                    "file": f"fleurs/{directory}/{stem}.wav",
                    "sha256": sha256(data),
                    "bytes": len(data),
                    "samples": samples,
                    "language": locale,
                    "text": row[2],
                    "normalized": row[3],
                }
            )
    if len(clips) != count:
        raise ValueError(f"{directory}: found {len(clips)} of {count} chosen clips")
    clips.sort(key=lambda c: c["id"])
    source = {
        "id": f"fleurs-{directory}-test",
        "url": url,
        "sha256": archive.sha256,
        "bytes": archive.size,
        "license": "CC BY 4.0",
        "attribution": f"FLEURS (Conneau et al., 2022), google/fleurs at {FLEURS_REVISION[:8]}, {directory} test split",
    }
    return source, clips


def pin(archives: pathlib.Path) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    source, chosen = pin_librispeech(archives)
    sources.append(source)
    clips += chosen
    for directory, locale, count in FLEURS:
        source, chosen = pin_fleurs(archives, directory, locale, count)
        sources.append(source)
        clips += chosen
    return {
        "$comment": "The speech corpus: scripts/speech-corpus.py (`make speech-corpus` fetches it into "
        ".reference/speech/). Each clip is a member of its source archive, pinned by SHA-256, with its "
        "reference transcript; FLEURS clips also carry FLEURS's normalized transcript.",
        "sources": sources,
        "clips": clips,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="fetch", choices=["fetch", "pin"])
    parser.add_argument("--out", type=pathlib.Path, default=OUT, help="where clips go (default .reference/speech)")
    args = parser.parse_args()
    if args.command == "pin":
        corpus = pin(args.out / "archives")
        CORPUS.write_text(json.dumps(corpus, indent=1, ensure_ascii=False) + "\n")
        print(f"pinned {len(corpus['clips'])} clips into {CORPUS.relative_to(ROOT)}", file=sys.stderr)
        return 0
    corpus = json.loads(CORPUS.read_text())
    return 1 if fetch(corpus, args.out) else 0


if __name__ == "__main__":
    sys.exit(main())
