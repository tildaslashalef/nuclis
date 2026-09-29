#!/usr/bin/env python3
"""Capture Laya oracle fixtures from the pinned `laya` Python package on the CPU.

Never imported by the build. Run it in the reference venv:

    uv venv --python 3.12 .zig-cache/reference/laya-venv
    VIRTUAL_ENV=.zig-cache/reference/laya-venv uv pip install laya==0.3.20
    USE_TF=0 .zig-cache/reference/laya-venv/bin/python scripts/laya-reference.py

It loads the pulled checkpoint (no second download) from a staged copy under
.zig-cache/reference/laya-model, since the package may rewrite
tokenizer_config.json in place. Writes inference/src/models/fixtures/laya/:
`requests.json` (versions, per request the sequence, markers, logits, and the
package's answer; per tensor its rows and offset) with `activations.f32`
(little-endian F32 rows those offsets index), and `tokens.json` (text → ids
through the package's own tokenizer, `add_special_tokens=False`).
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

os.environ.setdefault("USE_TF", "0")

LAYA_VERSION = "0.3.20"
COMMIT = "55cf4c4e"
WEIGHTS_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
ROOT = pathlib.Path(__file__).resolve().parents[1]
ENCODER_LAYERS = (0, 1, 3, 27)

DEPARTMENT = {"type": "choice", "instructions": "Which department should handle this?",
              "criteria": {"billing": "invoices, payments, refunds",
                           "technical": "bugs, outages, system errors",
                           "other": "everything else"}}
TICKET = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."

LOG = "".join(
    "2026-09-%02d 12:%02d:%02d worker-%d INFO request id=%05d path=/api/v1/items/%d status=200 latency_ms=%d\n"
    % (1 + i % 28, i % 60, (i * 7) % 60, i % 4, i, i * 13 % 997, 20 + i * 37 % 180) for i in range(40)
) + "2026-09-29 12:00:00 worker-2 ERROR database connection refused; retrying\n"

CONVERSATION = [
    {"role": "user" if i % 2 == 0 else "assistant",
     "content": ("Can you check order %d? It has not arrived yet." % (4100 + i)) if i % 2 == 0
     else ("Order %d shipped on Monday and should arrive within three business days." % (4100 + i))}
    for i in range(40)
] + [{"role": "user", "content": "This is the third late order. Close my account, I am done."}]

TOPICS = ["billing", "refunds", "shipping", "returns", "warranty", "login", "password", "security",
          "privacy", "outage", "performance", "integrations", "api", "mobile", "desktop", "pricing",
          "upgrades", "cancellation", "feedback", "other"]

REQUESTS = [
    ("choice_described", TICKET, DEPARTMENT),
    ("choice_labels", "The new dashboard is fast, clean, and finally shows what I need.",
     {"type": "choice", "instructions": "What is the sentiment of this message?",
      "criteria": ["positive", "negative", "neutral"]}),
    ("score", TICKET,
     {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]}),
    ("noul", TICKET,
     {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}),
    ("json_state",
     {"customer": "Zoë Müller", "plan": "pro", "seats": 12, "open_tickets": [
         {"id": 881, "subject": "SSO login fails with error 500", "age_days": 3},
         {"id": 885, "subject": "Invoice PDF shows the wrong VAT number", "age_days": 1}]},
     {"type": "noul", "instructions": "Does this customer have an unresolved technical problem?",
      "criteria": {"true": "a login, outage, or error is still open", "false": "only billing or no issues"}}),
    ("long_text", LOG,
     {"type": "choice", "instructions": "What is the most severe level in this log?",
      "criteria": {"info": "only routine requests", "warning": "degraded but working", "error": "a failure"}}),
    ("long_list", CONVERSATION,
     {"type": "noul", "instructions": "Does the user want to close their account?"}),
    ("choice_20", "My phone app logs me out every few minutes and I have to type my password again.",
     {"type": "choice", "instructions": "Which topic does this message belong to?",
      "criteria": {t: "questions and requests about %s, including anything related to %s for any product or plan"
                   % (t, t) for t in TOPICS}}),
]

# Tokenizer text set: accents (composed and decomposed), CJK, emoji, code, the
# splitter's edge cases, NFC-sensitive input, and every added-token kind.
TEXTS = [
    "", "hello", " hello", "Hello world", "don't DON'T we're I'M 'S",
    "Grüße, café, naïve, façade", "café ẹ́ ḍ̇",
    "\u212a \u2126 \u212b \u00c5 A\u030a", "\u1100\u1161\u11a8 \uac00\u11a8 \ud55c\uad6d\uc5b4",
    "\u0958 \u0915\u093c \U0001d160", "a\u0301\u0327 e\u0327\u0301\u0301 A\u0300\u0301",
    "\u4e16\u754c\u4f60\u597d\uff0c\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",
    "\U0001f642 \U0001f44d\U0001f3fd \U0001f468\u200d\U0001f469\u200d\U0001f467 \U0001f1ef\U0001f1f5",
    "fn main() {\n    let x: u32 = 42;\n\treturn x * 2;\n}\n",
    "def f(a, b):\n        return {'k': [1, 2, 3]}\n",
    "12345 3.14159 1,000,000 ²٣",
    "a" + " " * 30 + "b" + " " * 50 + "c" + " " * 25,
    "  leading and trailing  ", "\n\n\ttabs\t\tand\r\nnewlines\n", " \t\n x", "x   y　z w v",
    "\x1c\x1d\x1e\x1f\x0b\x0c", "!!! ??? ... --- ### @@@",
    "[CLS] [SEP] [PAD] [UNK] <|endoftext|> <|padding|>", "a   [MASK] b [MASK]c", "[unused0] [unused82]",
    "|||IP_ADDRESS||| |||EMAIL_ADDRESS|||x|||PHONE_NUMBER|||", "<|padding|≯ ≠ ≯",
    "user@example.com https://example.com/a?b=c&d=e#f",
    "ǅungla ǲ ﬁ ﬀ ｆｕｌｌ ; `",
    json.dumps({"customer": "Zoë Müller", "plan": "pro"}, ensure_ascii=False),
]


def check(agent, name, qtype, logits, answer) -> None:
    """The package batches with padding and rounds to 4 places; its answer
    must be calibration applied to this forward's logits."""
    import numpy as np
    from laya.common import temp_bucket
    k = len(logits)
    t = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
    z = logits / t
    p = np.exp(z - z.max())
    p /= p.sum()
    got = [answer["noul"]] if answer["type"] == "noul" else list(answer["probabilities"].values())
    want = [p[1]] if answer["type"] == "noul" else list(p)
    if max(abs(a - b) for a, b in zip(got, want)) > 1e-4:
        sys.exit("%s: package answer %s differs from calibrated logits %s" % (name, got, want))


def stage(model: pathlib.Path, staged: pathlib.Path) -> None:
    """Copies every support file and links the weights, so the package's
    in-place tokenizer_config rewrite can never touch the pulled set."""
    if staged.exists():
        shutil.rmtree(staged)
    for rel in ("rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json"):
        (staged / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model / rel, staged / rel)
    (staged / "model.safetensors").symlink_to(model / "model.safetensors")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=pathlib.Path,
                        default=pathlib.Path.home() / ".nuclis/models/convaiinnovations/laya")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "inference/src/models/fixtures/laya")
    parser.add_argument("--skip-digest", action="store_true", help="do not re-hash the 800 MB weights")
    args = parser.parse_args()

    import laya
    import numpy as np
    import tokenizers
    import torch
    import transformers
    from laya.common import render_options, serialize_state

    from importlib.metadata import version
    if version("laya") != LAYA_VERSION:
        parser.error("laya %s installed, %s pinned" % (version("laya"), LAYA_VERSION))
    if not args.skip_digest and sha256(args.model / "model.safetensors") != WEIGHTS_SHA256:
        parser.error("model.safetensors differs from the pinned root set (commit %s)" % COMMIT)

    staged = ROOT / ".zig-cache/reference/laya-model"
    stage(args.model, staged)
    torch.manual_seed(0)
    agent = laya.Agent(str(staged), device="cpu")
    assert not agent.amp_enabled and agent.dtype == torch.float32
    model = agent.model.eval()

    captured = {}

    def keep(name):
        def hook(_module, _inputs, output):
            captured[name] = (output[0] if isinstance(output, tuple) else output).detach()[0].float().clone()
        return hook

    for i in ENCODER_LAYERS:
        model.encoder.layers[i].register_forward_hook(keep("encoder.%d" % i))
    for i, layer in enumerate(model.head.layers):
        layer.register_forward_hook(keep("head.%d" % i))

    blob = bytearray()
    requests = []
    for name, state, question in REQUESTS:
        agent._check_question(name, question)
        internal = {name: agent._to_internal(question)}
        [item] = agent._encode_state(state, [name], internal)
        ids, markers = item["ids"], item["markers"]
        n = len(ids)
        captured.clear()
        with torch.no_grad():
            enc = model.encoder(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, n, dtype=torch.long))
            captured["final"] = enc.last_hidden_state.detach()[0].float().clone()
            logits, _ = model(torch.tensor([ids]), torch.ones(1, n, dtype=torch.long),
                              torch.tensor([markers]), torch.ones(1, len(markers), dtype=torch.bool),
                              torch.tensor([item["qtype"]]))
        logits = logits[0].float().numpy()
        answer = agent.system_one(state, {name: question})
        check(agent, name, item["qtype"], logits, answer["answers"][name])

        # Rows kept per tensor: the first row, up to three markers (first, second,
        # last), and the last row; head tensors keep only the marker rows.
        kept_markers = sorted(set([markers[0], markers[min(1, len(markers) - 1)], markers[-1]]))
        encoder_rows = sorted(set([0, *kept_markers, n - 1]))
        tensors = []
        for tname in ["encoder.%d" % i for i in ENCODER_LAYERS] + ["final", "head.0", "head.1"]:
            rows = kept_markers if tname.startswith("head") else encoder_rows
            data = captured[tname][rows].numpy().astype("<f4")
            tensors.append({"name": tname, "rows": rows, "offset": len(blob) // 4})
            blob += data.tobytes()

        opts = render_options(internal[name])
        mask = agent.tok.mask_token
        requests.append({
            "name": name,
            "state": state,
            "question": question,
            # The exact texts build_sequence tokenizes: the head, each option
            # after a space, and the serialized state (list states keep their tail).
            "head_text": "%s question: %s" % (internal[name]["t"], str(internal[name]["ins"]).replace(mask, " ")),
            "options": [o.replace(mask, " ") for o in opts],
            "state_text": serialize_state(state).replace(mask, " "),
            "truncate_left": isinstance(state, list),
            "qtype": item["qtype"],
            "ids": ids,
            "markers": markers,
            "logits": [float(x) for x in logits],
            "answer": answer["answers"][name],
            "usage": answer["usage"],
            "tensors": tensors,
        })
        print("%-18s %3d tokens %2d options  logits %s" % (name, n, len(markers), np.round(logits, 4)))

    tok = agent.tok
    token_cases = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in TEXTS]

    args.out.mkdir(parents=True, exist_ok=True)
    versions = {
        "laya": LAYA_VERSION, "torch": torch.__version__, "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__, "python": sys.version.split()[0],
    }
    meta = {
        "source": "scripts/laya-reference.py", "repo": "convaiinnovations/laya", "commit": COMMIT,
        "weights_sha256": WEIGHTS_SHA256, "versions": versions, "device": "cpu", "dtype": "float32",
        "hidden": int(model.encoder.config.hidden_size),
    }
    (args.out / "requests.json").write_text(json.dumps({**meta, "requests": requests}, ensure_ascii=False, indent=1) + "\n")
    (args.out / "activations.f32").write_bytes(bytes(blob))
    (args.out / "tokens.json").write_text(json.dumps({**meta, "token_cases": token_cases}, ensure_ascii=False, indent=1) + "\n")
    total = sum(p.stat().st_size for p in args.out.iterdir())
    print("versions:", versions)
    print("wrote %s (%d bytes)" % (args.out, total))
    if total > 1 << 20:
        sys.exit("fixtures exceed 1 MiB")


if __name__ == "__main__":
    main()
