#!/usr/bin/env python3
"""Capture pinned prompt fixtures for the Qwen3.8 native tool path.

Same rules as tokenizer-fixtures.py: only loopback, the pinned reference
revision, the pinned artifact, and the artifact's own chat template (no
override). This script adds the cases tokenizer-fixtures.py cannot: a tools
block and tool-call/tool-result history, for every reasoning effort the
template takes. Writes inference/src/profiles/fixtures/qwen38-tools.json.

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

PROFILE = {
    "model_sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
    "model_size": 16464440224,
    "template_sha256": "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce",
}
EFFORTS = ("off", "low", "medium", "xhigh")


def kwarg(effort):
    return {"enable_thinking": effort != "off",
            "reasoning_effort": "xhigh" if effort == "off" else effort,
            "preserve_thinking": True}


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
TOOLS = [READ, BASH]


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


CASES = [
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
    for name, messages in CASES:
        # `tools_only` carries definitions; every case does, so the tools
        # block is pinned and the call history is what varies.
        for effort in EFFORTS:
            prompt = request("/apply-template", {
                "messages": to_request(messages),
                "tools": request_tools(TOOLS),
                "chat_template_kwargs": kwarg(effort),
            })["prompt"]
            cases.append({"name": name, "effort": effort, "messages": messages,
                          "tools": TOOLS, "prompt": prompt})

    result = {"reference_revision": REVISION, "model_sha256": PROFILE["model_sha256"],
              "template_sha256": PROFILE["template_sha256"],
              "add_special": False, "preserve_thinking": True, "cases": cases}
    target = pathlib.Path(__file__).resolve().parents[1] / "inference/src/profiles/fixtures/qwen38-tools.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    print(f"Saved {len(cases)} tool prompt cases")


if __name__ == "__main__":
    main()
