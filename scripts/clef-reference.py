#!/usr/bin/env python3
# pyright: reportMissingImports=false
# (torch, transformers, and PIL live only in the reference venv the docstring names.)
"""Capture clef-flash oracle fixtures from Cloudflare's own `joint_schema_model.py`.

Never imported by the build. Run it in the reference venv:

    uv venv --python 3.12 .reference/clef-venv
    VIRTUAL_ENV=.reference/clef-venv uv pip install torch==2.11.0 transformers==5.10.2 \
        safetensors huggingface_hub pillow torchvision numpy
    .reference/clef-venv/bin/python scripts/clef-reference.py sequence
    .reference/clef-venv/bin/python scripts/clef-reference.py head

Neither mode loads the backbone. `sequence` builds every request's model input
with the reference's `encode_record` (the tokenizer and processor of the pulled
repository), and writes inference/src/models/fixtures/clef/sequences.json: per
request the ids, the question and option spans, and for images the grid and
the token offset. `head` reads what nuclis dumped for each request
(`.zig-cache/clef/dumps/<name>.hidden.f32`, the backbone's final hidden rows
after `output_norm`, and `<name>.lexical.f32`, each option's mean output-head
row), runs the reference's `JointSchemaHead` in F32 on the CPU, and writes
fixtures/clef/head.json: per request the logits per question. `head
--synthetic --dumps D --out F` first writes seeded random dumps to D, which
checks the head alone, before any backbone runs.
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("USE_TF", "0")

REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
REPO = "Cloudflare/clef-flash"
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "inference/src/models/fixtures/clef"
DUMPS = ROOT / ".zig-cache/clef/dumps"
HIDDEN = 4096

LOG = (
    "".join(
        "2026-09-%02d 12:%02d:%02d worker-%d INFO request id=%05d path=/api/v1/items/%d status=200 latency_ms=%d\n"
        % (1 + i % 28, i % 60, (i * 7) % 60, i % 4, i, i * 13 % 997, 20 + i * 37 % 180)
        for i in range(70)
    )
    + "2026-09-29 12:00:00 worker-2 ERROR database connection refused; retrying\n"
)

CONVERSATION = [
    {
        "role": "user" if i % 2 == 0 else "assistant",
        "content": ("Can you check order %d? It has not arrived yet." % (4100 + i))
        if i % 2 == 0
        else ("Order %d shipped on Monday and should arrive within three business days." % (4100 + i)),
    }
    for i in range(12)
] + [{"role": "user", "content": "This is the third late order. Close my account, I am done."}]

# `expect` maps a question to the option an obvious answer picks: the sanity set.
REQUESTS = [
    {
        "name": "card_invoice",
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
            },
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        },
        "expect": {"status": "overdue", "large": "true"},
    },
    {
        "name": "card_systemone",
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle the message?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            },
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
        "expect": {"department": "technical", "urgency": "2", "outage": "true"},
    },
    {
        "name": "ticket_department",
        "state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, system errors",
                    "other": "everything else",
                },
            }
        },
        "expect": {"department": "billing"},
    },
    {
        "name": "noul_error",
        "state": "The server returned HTTP 500 for every request since 09:00.",
        "questions": {"error": {"type": "noul", "instructions": "Is something failing?"}},
        "expect": {"error": "true"},
    },
    {
        "name": "score_sentiment",
        "state": {"review": "Absolutely wonderful. Fast delivery, perfect quality, I will buy again!"},
        "questions": {
            "sentiment": {
                "type": "score",
                "instructions": "How positive is the review?",
                "criteria": ["very negative", "negative", "neutral", "positive", "very positive"],
            }
        },
        "expect": {"sentiment": "4"},
    },
    {
        "name": "log_long",
        "state": LOG,
        "questions": {
            "error": {"type": "noul", "instructions": "Does the log contain an ERROR line?"},
            "worker": {
                "type": "choice",
                "instructions": "Which worker logged the error?",
                "criteria": {"worker-0": None, "worker-1": None, "worker-2": None, "worker-3": None},
            },
        },
        "expect": {"error": "true", "worker": "worker-2"},
    },
    {
        "name": "multi_question",
        "state": CONVERSATION,
        "questions": {
            "churn": {"type": "noul", "instructions": "Does the customer want to close the account?"},
            "topic": {
                "type": "choice",
                "instructions": "What is the conversation about?",
                "criteria": {"shipping": "late or missing orders", "billing": "charges", "login": "account access"},
            },
            "anger": {
                "type": "score",
                "instructions": "How angry is the customer?",
                "criteria": ["calm", "annoyed", "furious"],
            },
            "language": {"type": "choice", "criteria": {"en": "English", "de": "German", "fr": "French"}},
            "refund": {"type": "noul", "instructions": "Did the assistant offer a refund?"},
            "speaker": {
                "type": "choice",
                "instructions": "Who spoke last?",
                "criteria": {"user": "the customer", "assistant": "the support agent"},
            },
        },
        "expect": {"churn": "true", "topic": "shipping", "language": "en", "refund": "false", "speaker": "user"},
    },
    {
        "name": "noul_criteria",
        "state": {"temperature_c": 41, "city": "Cairo"},
        "questions": {
            "hot": {
                "type": "noul",
                "criteria": {"true": "Above 35 degrees Celsius.", "false": "35 degrees Celsius or below."},
            }
        },
        "expect": {"hot": "true"},
    },
    {
        "name": "image_red",
        "state": {"task": "Describe the attached image."},
        "images": [{"color": [220, 20, 20], "size": [320, 240]}],
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What is the dominant color of the image?",
                "criteria": {"red": None, "green": None, "blue": None},
            }
        },
        "expect": {"color": "red"},
    },
    {
        "name": "image_two",
        "state": "Two images are attached.",
        "images": [{"color": [20, 20, 220], "size": [256, 256]}, {"color": [20, 200, 20], "size": [512, 384]}],
        "questions": {
            "first": {
                "type": "choice",
                "instructions": "What color is the first image?",
                "criteria": {"red": None, "green": None, "blue": None},
            }
        },
        "expect": {"first": "blue"},
    },
]


def reference_module():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "joint_schema_model.py", revision=REVISION)
    sys.path.insert(0, str(pathlib.Path(path).parent))
    import joint_schema_model

    return joint_schema_model


def model_dir() -> pathlib.Path:
    home = pathlib.Path(os.environ.get("NUCLIS_HOME", pathlib.Path.home() / ".nuclis"))
    return home / "models" / REPO


def image(spec):
    from PIL import Image

    return Image.new("RGB", tuple(spec["size"]), tuple(spec["color"]))


def record(request) -> dict:
    out = {k: v for k, v in request.items() if k not in ("name", "expect", "images")}
    if "images" in request:
        out["images"] = [image(spec) for spec in request["images"]]
    return out


def sequence(args) -> None:
    from transformers import AutoProcessor

    jsm = reference_module()
    processor = AutoProcessor.from_pretrained(model_dir())
    tokenizer = processor.tokenizer
    entries = []
    for request in REQUESTS:
        encoded = jsm.encode_record(tokenizer, record(request), processor=processor)
        entry = {
            "name": request["name"],
            "request": {k: v for k, v in request.items() if k not in ("name", "expect")},
            "expect": request["expect"],
            "rendered_state": jsm.render(request["state"]),
            "input_ids": list(encoded.input_ids),
            "questions": [
                {
                    "id": q.question_id,
                    "type": q.question_type,
                    "span": list(q.question_span),
                    "option_spans": [list(s) for s in q.option_spans],
                    "option_ids": list(q.option_ids),
                }
                for q in encoded.questions
            ],
        }
        if encoded.media is not None:
            grid = encoded.media["image_grid_thw"].tolist()
            entry["media"] = {
                "token_offset": encoded.media["token_offset"],
                "image_grid_thw": grid,
                "image_tokens": [t * h * w // 4 for t, h, w in grid],
            }
        entries.append(entry)
        print(f"{request['name']}: {len(encoded.input_ids)} tokens, {len(encoded.questions)} questions")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = {
        "repo": REPO,
        "revision": REVISION,
        "versions": versions(),
        "requests": entries,
    }
    # One request per line: the ids would otherwise take a line each.
    head = {k: v for k, v in out.items() if k != "requests"}
    rows = ",\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
    text = json.dumps(head, ensure_ascii=False)[:-1] + ', "requests": [\n' + rows + "\n]}\n"
    (FIXTURES / "sequences.json").write_text(text)


def head(args) -> None:
    import numpy as np
    import torch
    from safetensors.torch import load_file

    jsm = reference_module()
    directory = model_dir()
    config = json.loads((directory / "joint_head_config.json").read_text())
    weights = load_file(directory / "joint_head.safetensors")
    module = jsm.JointSchemaHead(**config)
    module.load_state_dict(weights, strict=True)
    module = module.float().eval()
    bf16 = jsm.JointSchemaHead(**config)
    bf16.load_state_dict(weights, strict=True)
    bf16 = bf16.to(torch.bfloat16).eval()
    sequences = json.loads((FIXTURES / "sequences.json").read_text())
    dumps = args.dumps or DUMPS
    if args.synthetic:
        synthesize(dumps, sequences)
    results = []
    for entry in sequences["requests"]:
        hidden_path = dumps / f"{entry['name']}.hidden.f32"
        lexical_path = dumps / f"{entry['name']}.lexical.f32"
        if not hidden_path.exists():
            print(f"{entry['name']}: no dump, skipped")
            continue
        rows = len(entry["input_ids"])
        hidden = torch.from_numpy(np.fromfile(hidden_path, dtype="<f4").reshape(rows, HIDDEN))
        lexical = torch.from_numpy(np.fromfile(lexical_path, dtype="<f4").reshape(-1, HIDDEN))
        encoded = encoded_record(jsm, entry)
        with torch.inference_mode():
            logits = run(module, hidden, lexical, encoded, torch.float32)
            half = run(bf16, hidden, lexical, encoded, torch.bfloat16)
        per_question = [row.tolist() for row in logits]
        gap = max(float((a.float() - b.float()).abs().max()) for a, b in zip(logits, half))
        picks = {
            q["id"]: q["option_ids"][int(torch.tensor(row).argmax())]
            for q, row in zip(entry["questions"], per_question)
        }
        ok = all(picks[k] == v for k, v in entry["expect"].items())
        verdict = "" if args.synthetic else (" ok" if ok else " UNEXPECTED")
        print(f"{entry['name']}: picks {picks}{verdict}; bf16 head max |Δ| {gap:.4f}")
        results.append({"name": entry["name"], "logits": per_question, "bf16_max_abs_gap": gap, "picks": picks})
    out = {"repo": REPO, "revision": REVISION, "versions": versions(), "results": results}
    target = args.out or FIXTURES / "head.json"
    target.write_text(json.dumps(out, indent=1) + "\n")


def synthesize(dumps: pathlib.Path, sequences) -> None:
    """Seeded stand-ins for the backbone's rows (unit-variance, like rows after
    an RMS norm) and the lexical means, so the head is checked alone."""
    import numpy as np

    rng = np.random.default_rng(34)
    dumps.mkdir(parents=True, exist_ok=True)
    for entry in sequences["requests"]:
        rows = len(entry["input_ids"])
        options = sum(len(q["option_spans"]) for q in entry["questions"])
        rng.standard_normal((rows, HIDDEN), dtype=np.float32).astype("<f4").tofile(
            dumps / f"{entry['name']}.hidden.f32"
        )
        (0.02 * rng.standard_normal((options, HIDDEN), dtype=np.float32)).astype("<f4").tofile(
            dumps / f"{entry['name']}.lexical.f32"
        )


def encoded_record(jsm, entry):
    questions = tuple(
        jsm.EncodedQuestion(
            question_id=q["id"],
            question_type=q["type"],
            question_span=tuple(q["span"]),
            option_spans=tuple(tuple(s) for s in q["option_spans"]),
            option_ids=tuple(q["option_ids"]),
        )
        for q in entry["questions"]
    )
    return jsm.EncodedRecord(input_ids=tuple(entry["input_ids"]), questions=questions, record_id=entry["name"])


def run(module, hidden, lexical, encoded, dtype):
    """`JointSchemaHead.forward` reads the output head's rows by token id; the
    dump holds each option's mean row instead, so a stand-in table maps one
    fresh id per option to its mean and the spans index it."""
    import torch

    options = [s for q in encoded.questions for s in q.option_spans]
    assert lexical.shape[0] == len(options), (lexical.shape, len(options))
    ids = torch.tensor(encoded.input_ids).unsqueeze(0).clone()
    table = torch.zeros(len(options), HIDDEN)
    # Each option span is rewritten to one id repeated, so the span mean is
    # exactly the dumped mean.
    for index, (start, end) in enumerate(options):
        ids[0, start:end] = index
        table[index] = lexical[index]
    mask = torch.ones_like(ids)
    return module(hidden.unsqueeze(0).to(dtype), ids, mask, [encoded], table.to(dtype))[0]


def versions() -> dict:
    import tokenizers
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["sequence", "head"])
    parser.add_argument("--dumps", type=pathlib.Path, help="head: the dumps to read (default .zig-cache/clef/dumps)")
    parser.add_argument("--out", type=pathlib.Path, help="head: where the logits go (default the fixture)")
    parser.add_argument(
        "--synthetic", action="store_true", help="head: write seeded random dumps first (checks the head alone)"
    )
    args = parser.parse_args()
    {"sequence": sequence, "head": head}[args.mode](args)


if __name__ == "__main__":
    main()
