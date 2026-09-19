#!/usr/bin/env python3
"""Capture synthetic text/token fixtures from a separately started pinned server.

Only loopback is allowed. No generation requests or workspace text are sent.
Writes inference/src/profiles/fixtures/<profile>-text.json after all checks:
the server must run the pinned reference revision on the pinned artifact of
the chosen profile, with the artifact's own chat template (no override).
"""
import argparse
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request

REVISION = "7620399f58aebfd2196b74021f9581bcf7218cb9"

# The conversations every profile renders; the same set so the two fixtures
# are comparable. `continued` is two assistant messages in a row, which
# Gemma's template merges into one model turn and Qwen's keeps apart.
CONVERSATIONS = [
    ("single", [{"role": "user", "content": "Hello"}]),
    ("system", [{"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Explain slices."}]),
    ("merged_system", [{"role": "system", "content": "  Be precise.\n"},
                       {"role": "developer", "content": "\tUse Zig.  "},
                       {"role": "system", "content": " \n"},
                       {"role": "user", "content": " Explain slices. "}]),
    ("history", [{"role": "user", "content": "Compute 1+1."},
                 {"role": "assistant", "content": " 2 ", "reasoning_content": " Add the integers. "},
                 {"role": "user", "content": "Now double it."},
                 {"role": "assistant", "content": "4"},
                 {"role": "user", "content": "Why?"}]),
    ("continued", [{"role": "user", "content": "Count."},
                   {"role": "assistant", "content": "one"},
                   {"role": "assistant", "content": " two"},
                   {"role": "user", "content": "Go on."}]),
    ("unicode", [{"role": "user", "content": "  Grüße 世界 🙂 é  "}]),
    ("empty", [{"role": "user", "content": " \t\r\n\v\f"}]),
]

TEXTS = ["", "hello", " hello", "Hello\nworld", "\t  a\r\nb\n",
         "pub fn add(a: i32, b: i32) i32 { return a + b; }",
         "1234567890 3.14159", "don't I'M we're", "Grüße 世界 🙂 é"]

PROFILES = {
    "qwen38": {
        "model_sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
        "model_size": 16464440224,
        "template_sha256": "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce",
        # Five reasoning efforts: the template takes `reasoning_effort` and
        # resolves `high` to `xhigh` itself.
        "efforts": ("off", "low", "medium", "high", "xhigh"),
        "kwargs": lambda effort: {"enable_thinking": effort != "off",
                                  "reasoning_effort": "xhigh" if effort == "off" else effort,
                                  "preserve_thinking": True},
        "preserve_thinking": True,
        "marker_text": "<|im_start|>assistant\n<think>\n</think><|im_end|>",
    },
    "gemma4": {
        "model_sha256": "90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e",
        "model_size": 7366423360,
        "template_sha256": "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b",
        # The template only switches thinking on or off: `off` and one
        # thinking effort are captured; the Zig test asserts the other two
        # render like `medium`.
        "efforts": ("off", "medium"),
        "kwargs": lambda effort: {"enable_thinking": effort != "off"},
        "preserve_thinking": False,
        "marker_text": "<|turn>model\n<|channel>thought\n<channel|>Hi<turn|>",
    },
    "muse_glimmer": {
        "model_sha256": "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
        "model_size": 15878222368,
        "template_sha256": "114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e",
        # Four reasoning strengths and no off: the template takes
        # `reasoning_strength` as a system-prompt line. Conversations without
        # a system message get the template's synthesized one, whose date line
        # the server fills from its clock (`strftime_now`) on capture day.
        "efforts": ("low", "medium", "high", "xhigh"),
        "kwargs": lambda effort: {"reasoning_strength": effort},
        "preserve_thinking": True,
        "marker_text": "<|start|>assistant to=self<|message|>Hi<|eom|><|start|>assistant to=user<|message|>Hello<|eot|>",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18087")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="qwen38")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    address = urllib.parse.urlparse(args.url)
    if (address.scheme != "http" or address.hostname != "127.0.0.1"
            or address.username or address.password or address.path not in ("", "/")
            or address.query or address.fragment):
        parser.error("--url must be an HTTP endpoint on 127.0.0.1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(endpoint, body=None):
        req = urllib.request.Request(args.url.rstrip("/") + endpoint,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=60) as response:
            return json.load(response)

    props = request("/props")
    if REVISION[:7] not in props["build_info"]:
        raise RuntimeError("expected pinned reference revision")
    # The server reports the template without the file's trailing newline
    # (Gemma's ends in one); the digest is the file's, so try both forms.
    template = props["chat_template"]
    if profile["template_sha256"] not in (hashlib.sha256(template.encode()).hexdigest(),
                                          hashlib.sha256((template + "\n").encode()).hexdigest()):
        raise RuntimeError("expected pinned artifact template, without server overrides")
    model = pathlib.Path(props["model_path"])
    if model.stat().st_size != profile["model_size"]:
        raise RuntimeError("expected pinned artifact size")
    with model.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != profile["model_sha256"]:
            raise RuntimeError("model checksum mismatch")

    def tokens(text, special):
        return request("/tokenize", {"content": text, "add_special": False,
                                     "parse_special": special})["tokens"]

    cases = []
    for name, messages in CONVERSATIONS:
        for effort in profile["efforts"]:
            prompt = request("/apply-template", {"messages": messages,
                "chat_template_kwargs": profile["kwargs"](effort)})["prompt"]
            cases.append({"name": name + "_" + effort, "effort": effort,
                          "messages": messages, "prompt": prompt,
                          "tokens": tokens(prompt, True)})

    token_cases = []
    for text in TEXTS + [profile["marker_text"]]:
        for special in (False, True):
            ids = tokens(text, special)
            decoded = request("/detokenize", {"tokens": ids})["content"]
            if decoded != text:
                raise RuntimeError("reference token round trip changed synthetic input")
            token_cases.append({"text": text, "parse_special": special,
                                "tokens": ids, "detokenized": decoded})
    result = {"reference_revision": REVISION, "model_sha256": profile["model_sha256"],
              "template_sha256": profile["template_sha256"],
              "add_special": False, "preserve_thinking": profile["preserve_thinking"],
              "prompt_cases": cases, "token_cases": token_cases}
    target = pathlib.Path(__file__).resolve().parents[1] / f"inference/src/profiles/fixtures/{args.profile}-text.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    print(f"Saved {len(cases)} prompt cases and {len(token_cases)} tokenizer cases")


if __name__ == "__main__":
    main()
