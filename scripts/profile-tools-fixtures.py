#!/usr/bin/env python3
"""Capture pinned prompt fixtures for a profile's native tool path.

Same rules as tokenizer-fixtures.py: only loopback, the pinned reference
revision, the pinned artifact, and the artifact's own chat template (no
override). This script adds the cases tokenizer-fixtures.py cannot: a tools
block and tool-call/tool-result history, for every reasoning effort the
template takes. Writes inference/src/profiles/fixtures/<profile>-tools.json.

The reference renders through its chat-format layer, not the raw template:
for Gemma 4 that layer appends `<|turn>model\n` when the template leaves the
prompt at a closed turn, and the fixture records that rendering.

The fixture stores messages in the Zig profile's normalized shape
(`tool_calls[].arguments` is a JSON object *string*), so the render test
parses it straight into `profiles.Message`; the script converts to the
reference's OpenAI-shaped request only for `/apply-template`.
"""
import argparse
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request

REVISION = "7620399f58aebfd2196b74021f9581bcf7218cb9"

PROFILES = {
    "qwen38": {
        "model_sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
        "model_size": 16464440224,
        "template_sha256": "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce",
        "efforts": ("off", "low", "medium", "xhigh"),
        "kwargs": lambda effort: {"enable_thinking": effort != "off",
                                  "reasoning_effort": "xhigh" if effort == "off" else effort,
                                  "preserve_thinking": True},
        "preserve_thinking": True,
    },
    "gemma4": {
        "model_sha256": "90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e",
        "model_size": 7366423360,
        "template_sha256": "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b",
        "efforts": ("off", "medium"),
        "kwargs": lambda effort: {"enable_thinking": effort != "off"},
        # The template supports preserving reasoning and the reference server
        # enables it by default (`--no-reasoning-preserve` turns it off):
        # a call-bearing assistant message keeps its thought at any age.
        "preserve_thinking": True,
    },
}


def tool(name, description, parameters):
    # `parameters` is stored as the compact JSON string `profiles.ToolDefinition`
    # carries; the reference request re-parses it to an object.
    return {"name": name, "description": description,
            "parameters": json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))}


READ = tool("read_file",
            "Read a bounded region of a UTF-8 text file in the workspace.",
            {"type": "object",
             "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}},
             "required": ["path"]})
BASH = tool("bash",
            "Run one shell command in the workspace.",
            {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]})
# A declaration exercising the schema subset the profile renders: a
# description, integer/boolean/enum/array/nested-object properties.
SEARCH = tool("search",
              "Search the workspace.",
              {"type": "object",
               "properties": {"query": {"type": "string", "description": "Text to find."},
                              "limit": {"type": "integer"},
                              "regex": {"type": "boolean"},
                              "mode": {"type": "string", "enum": ["files", "content"]},
                              "paths": {"type": "array", "items": {"type": "string"}},
                              "options": {"type": "object",
                                          "properties": {"case_sensitive": {"type": "boolean"}},
                                          "required": ["case_sensitive"]}},
               "required": ["query"]})
TOOLS = {"qwen38": [READ, BASH], "gemma4": [READ, BASH, SEARCH]}


def call(cid, name, arguments):
    # `arguments` is a Python object here; the fixture stores the compact JSON
    # string the Zig type carries.
    return {"id": cid, "name": name, "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))}


def assistant(content, reasoning, calls=()):
    m = {"role": "assistant", "content": content}
    if reasoning:
        m["reasoning_content"] = reasoning
    if calls:
        m["tool_calls"] = list(calls)
    return m


def user(content):
    return {"role": "user", "content": content}


def result(cid, content):
    return {"role": "tool", "content": content, "tool_call_id": cid}


def system(content):
    return {"role": "system", "content": content}


COMMON_CASES = [
    ("tools_only", [user("read a then run it")]),
    ("one_call", [
        user("read a"),
        assistant("Reading now.", "I should read it.",
                  [call(1, "read_file", {"path": "a.zig", "offset": 3})]),
        result(1, "line one\nline two"),
        user("now what?"),
    ]),
    ("two_calls", [
        user("do both"),
        assistant("", "Plan both calls.",
                  [call(1, "read_file", {"path": "a.zig"}),
                   call(2, "bash", {"command": "zig build test"})]),
        result(1, "file body"),
        result(2, "ok"),
        user("next"),
    ]),
    ("empty_args_call", [
        user("go"),
        assistant("", "", [call(7, "bash", {})]),
        result(7, "done"),
        user("again"),
    ]),
    ("nested_value_call", [
        user("one call"),
        assistant("", "Check the shape.",
                  [call(1, "read_file", {"path": "a\nb\"c", "offset": 3})]),
        result(1, "body"),
        user("thanks"),
    ]),
]

# Gemma renders calls and results inside one model turn, so the shapes that
# matter are where the turn ends: on results (the loop continues), on
# results with content (the template closes the turn and the reference
# reopens one), and across two steps of one loop.
GEMMA_CASES = COMMON_CASES + [
    ("system_and_tools", [system("Be terse."), user("hi")]),
    ("nested_args_call", [
        user("search"),
        assistant("", "Shape check.",
                  [call(1, "search", {"query": "a\"b", "limit": 3, "regex": True, "ratio": 1.5,
                                      "mode": "files", "paths": ["x", "y"],
                                      "options": {"case_sensitive": False}, "nothing": None})]),
        result(1, "none"),
        user("ok"),
    ]),
    ("ends_on_results", [
        user("read a"),
        assistant("", "Plan.", [call(1, "read_file", {"path": "a.zig"})]),
        result(1, "line one\nline two"),
    ]),
    ("ends_on_results_no_reasoning", [
        user("read a"),
        assistant("", "", [call(1, "read_file", {"path": "a.zig"})]),
        result(1, "body"),
    ]),
    ("ends_on_results_with_content", [
        user("read a"),
        assistant("Reading now.", "Plan.", [call(1, "read_file", {"path": "a.zig"})]),
        result(1, "body"),
    ]),
    ("two_steps", [
        user("do both"),
        assistant("", "First.", [call(1, "read_file", {"path": "a.zig"})]),
        result(1, "file body"),
        assistant("", "Second.", [call(2, "bash", {"command": "zig build test"})]),
        result(2, "ok"),
    ]),
    ("answer_after_results", [
        user("read a"),
        assistant("", "Plan.", [call(1, "read_file", {"path": "a.zig"})]),
        result(1, "body"),
        assistant("Done.", "Afterthought."),
        user("thanks"),
    ]),
]
CASES = {"qwen38": COMMON_CASES, "gemma4": GEMMA_CASES}


def to_request(messages):
    """The Zig-normalized shape into the reference's OpenAI-shaped request."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                calls.append({"id": str(c["id"]), "type": "function",
                              "function": {"name": c["name"],
                                           "arguments": json.loads(c["arguments"])}})
            copy = {k: v for k, v in m.items() if k != "tool_calls"}
            copy["tool_calls"] = calls
            out.append(copy)
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"],
                        "tool_call_id": str(m["tool_call_id"])})
        else:
            out.append(m)
    return out


def request_tools(tools):
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": json.loads(t["parameters"])}}
            for t in tools]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18087")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="qwen38")
    args = parser.parse_args()
    PROFILE = PROFILES[args.profile]
    tools = TOOLS[args.profile]
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
    template = props["chat_template"]
    if PROFILE["template_sha256"] not in (hashlib.sha256(template.encode()).hexdigest(),
                                          hashlib.sha256((template + "\n").encode()).hexdigest()):
        raise RuntimeError("expected pinned artifact template, without server overrides")
    model = pathlib.Path(props["model_path"])
    if model.stat().st_size != PROFILE["model_size"]:
        raise RuntimeError("expected pinned artifact size")

    cases = []
    for name, messages in CASES[args.profile]:
        # `tools_only` carries definitions; every case does, so the tools
        # block is pinned and the call history is what varies.
        for effort in PROFILE["efforts"]:
            prompt = request("/apply-template", {
                "messages": to_request(messages),
                "tools": request_tools(tools),
                "chat_template_kwargs": PROFILE["kwargs"](effort),
            })["prompt"]
            cases.append({"name": name, "effort": effort, "messages": messages,
                          "tools": tools, "prompt": prompt})

    result = {"reference_revision": REVISION, "model_sha256": PROFILE["model_sha256"],
              "template_sha256": PROFILE["template_sha256"],
              "add_special": False, "preserve_thinking": PROFILE["preserve_thinking"],
              "cases": cases}
    target = (pathlib.Path(__file__).resolve().parents[1]
              / f"inference/src/profiles/fixtures/{args.profile}-tools.json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    print(f"Saved {len(cases)} tool prompt cases")


if __name__ == "__main__":
    main()
