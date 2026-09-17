#!/usr/bin/env python3
"""Prove that another chat template renders like a profile's pinned one.

A profile is selected by the SHA-256 of the artifact's `tokenizer.chat_template`.
A file converted with a different revision of the same template (a finetune,
an older converter) carries a different digest, and the engine refuses it
rather than guess. This script turns such a digest into an accepted alias by
evidence: with the reference server (pinned revision, loopback only) holding
the *other* file, it replays every case of the profile's pinned fixtures
(`<profile>-text.json`, `<profile>-tools.json`) through `/apply-template` and
`/tokenize` and requires the prompts and tokens to be byte-identical. Only
then does it record the alias in `inference/src/profiles/fixtures/<profile>-aliases.json`
with the served file's identity; the Zig profile lists the same digests and
its test checks the two agree.

The template digest is read from the served file's own GGUF header, so it is
exactly what `profiles.forDocument` computes at load.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import pathlib
import struct
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "inference/src/profiles/fixtures"


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = load_sibling("profile-tools-fixtures")
TEXT = load_sibling("tokenizer-fixtures")


def header_strings(path, wanted):
    """The wanted string keys of a GGUF header, read without mapping tensors."""
    out = {}
    with path.open("rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))

        def rstr():
            (n,) = rd("Q")
            return f.read(n)

        def rval(kind):
            if kind in (0, 1, 7):
                return rd("bB?"[kind] if kind != 7 else "B")[0]
            if kind in (2, 3):
                return rd("hH"[kind - 2])[0]
            if kind in (4, 5):
                return rd("iI"[kind - 4])[0]
            if kind == 6:
                return rd("f")[0]
            if kind == 8:
                return rstr()
            if kind == 9:
                (et,) = rd("I")
                (n,) = rd("Q")
                return [rval(et) for _ in range(n)]
            if kind in (10, 11):
                return rd("qQ"[kind - 10])[0]
            if kind == 12:
                return rd("d")[0]
            raise ValueError(f"unknown GGUF value type {kind}")

        magic, version, _tensors, kv = rd("IIQQ")
        if magic != 0x46554747 or version != 3:
            raise RuntimeError("not a GGUF v3 file")
        for _ in range(kv):
            key = rstr().decode()
            (kind,) = rd("I")
            value = rval(kind)
            if key in wanted:
                out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18087")
    parser.add_argument("--profile", choices=sorted(TOOLS.PROFILES), required=True)
    args = parser.parse_args()
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
    if TOOLS.REVISION[:7] not in props["build_info"]:
        raise RuntimeError("expected pinned reference revision")
    model = pathlib.Path(props["model_path"])
    header = header_strings(model, {"tokenizer.chat_template", "general.name"})
    template = header["tokenizer.chat_template"]
    digest = hashlib.sha256(template).hexdigest()
    pinned = TOOLS.PROFILES[args.profile]["template_sha256"]
    if digest == pinned:
        raise RuntimeError("the served file carries the pinned template; nothing to alias")
    served = props["chat_template"]
    if served not in (template.decode(), template.decode().rstrip("\n")):
        raise RuntimeError("the server renders a template other than the file's (an override?)")

    text = json.loads((FIXTURES / f"{args.profile}-text.json").read_text())
    tools = json.loads((FIXTURES / f"{args.profile}-tools.json").read_text())
    if pinned not in (text["template_sha256"], tools["template_sha256"]):
        raise RuntimeError("fixtures are not the pinned template's")

    def tokens(content, special):
        return request("/tokenize", {"content": content, "add_special": False,
                                     "parse_special": special})["tokens"]

    mismatches = []
    for case in text["prompt_cases"]:
        prompt = request("/apply-template", {
            "messages": case["messages"],
            "chat_template_kwargs": TEXT.PROFILES[args.profile]["kwargs"](case["effort"])})["prompt"]
        if prompt != case["prompt"]:
            mismatches.append(("text", case["name"], case["prompt"], prompt))
        elif tokens(prompt, True) != case["tokens"]:
            mismatches.append(("tokens", case["name"], case["tokens"], "differs"))
    for case in tools["cases"]:
        body = {"messages": TOOLS.to_request(case["messages"]),
                "chat_template_kwargs": TOOLS.PROFILES[args.profile]["kwargs"](case["effort"])}
        if case.get("tools"):
            body["tools"] = TOOLS.request_tools(case["tools"])
        prompt = request("/apply-template", body)["prompt"]
        if prompt != case["prompt"]:
            mismatches.append(("tools", case["name"], case["prompt"], prompt))
    for case in text["token_cases"]:
        if tokens(case["text"], case["parse_special"]) != case["tokens"]:
            mismatches.append(("token_case", case["text"], case["tokens"], "differs"))
    if mismatches:
        for kind, name, expected, got in mismatches[:5]:
            print(f"--- {kind} {name!r}\n{expected!r}\n+++\n{got!r}\n")
        raise SystemExit(f"{len(mismatches)} case(s) differ: {digest} is not an alias of {pinned}")

    size = model.stat().st_size
    with model.open("rb") as source:
        file_digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            file_digest.update(chunk)
    entry = {
        "template_sha256": digest,
        "template_bytes": len(template),
        "source": {"name": header.get("general.name", b"").decode(), "file": model.name,
                   "size": size, "sha256": file_digest.hexdigest()},
        "reference_revision": TOOLS.REVISION,
        "checked": datetime.date.today().isoformat(),
        "prompt_cases": len(text["prompt_cases"]),
        "tool_cases": len(tools["cases"]),
        "token_cases": len(text["token_cases"]),
    }
    target = FIXTURES / f"{args.profile}-aliases.json"
    result = json.loads(target.read_text()) if target.exists() else {
        "profile": args.profile, "template_sha256": pinned, "aliases": []}
    if result["template_sha256"] != pinned:
        raise RuntimeError("the aliases file pins another template; re-check every alias")
    result["aliases"] = [a for a in result["aliases"] if a["template_sha256"] != digest] + [entry]
    result["aliases"].sort(key=lambda a: a["template_sha256"])
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"{digest} renders every case like {pinned}: recorded in {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
