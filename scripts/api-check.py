#!/usr/bin/env python3
"""Checks `nuclis serve`'s OpenAI-compatible routes against a real model: the
shapes the OpenAI SDKs parse, a conversation whose later requests reuse the
earlier ones (`cached_tokens`), a tool round trip, an image, and the
refusals. Not a gate: it needs a model and a GPU.

    make api-check                              # starts ./zig-out/bin/nuclis serve
    make api-check ARGS='--model gemma-4-e4b-qat --image-model gemma-4-12b-qat'
    python3 scripts/api-check.py --url http://127.0.0.1:9000/v1   # a running server

Standard library only, so it runs without the SDKs installed; it asserts
the fields they read instead.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
Json = dict[str, Any]


class Failure(Exception):
    pass


def check(condition: bool, what: str) -> None:
    if not condition:
        raise Failure(what)


class Client:
    def __init__(self, url: str, model: str | None) -> None:
        self.url = url.rstrip("/")
        self.model = model

    def call(self, method: str, path: str, body: Json | None = None) -> tuple[int, Json]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data, {"content-type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                return response.status, cast(Json, json.load(response))
        except urllib.error.HTTPError as error:
            return error.code, cast(Json, json.load(error))

    def stream(self, body: Json) -> tuple[list[Json], list[str]]:
        """A streamed request's chunks and its SSE comments; asserts the framing."""
        payload = {"model": self.model, **body} if self.model else body
        request = urllib.request.Request(
            self.url + "/chat/completions",
            json.dumps({**payload, "stream": True}).encode(),
            {"content-type": "application/json"},
        )
        chunks: list[Json] = []
        comments: list[str] = []
        done = False
        with urllib.request.urlopen(request, timeout=1800) as response:
            check(response.headers.get("content-type") == "text/event-stream", "content-type")
            for raw in response:
                line = raw.decode().rstrip("\n")
                if not line:
                    continue
                if line.startswith(":"):
                    comments.append(line)
                    continue
                check(line.startswith("data: ") and not done, f"SSE line {line[:80]!r}")
                data = line[len("data: ") :]
                if data == "[DONE]":
                    done = True
                    continue
                chunk = cast(Json, json.loads(data))
                check("error" not in chunk, f"stream error {chunk}")
                chunks.append(chunk)
        check(done, "the stream ends with [DONE]")
        return chunks, comments

    def chat(self, body: Json, model: str | None = None) -> tuple[int, Json]:
        chosen = model or self.model
        return self.call("POST", "/chat/completions", {"model": chosen, **body} if chosen else body)


def completion_shape(r: Json) -> Json:
    """The fields an SDK's ChatCompletion reads; returns the message."""
    check(str(r.get("id", "")).startswith("chatcmpl-"), f"id: {r.get('id')}")
    check(r.get("object") == "chat.completion", "object")
    check(isinstance(r.get("created"), int) and isinstance(r.get("model"), str), "created, model")
    choice = cast(Json, r["choices"][0])
    check(choice.get("index") == 0 and choice.get("finish_reason") in ("stop", "length", "tool_calls"), "choice")
    message = cast(Json, choice["message"])
    check(message.get("role") == "assistant", "role")
    usage = cast(Json, r["usage"])
    check(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "usage totals")
    check(isinstance(usage["prompt_tokens_details"]["cached_tokens"], int), "cached_tokens")
    check(isinstance(usage["completion_tokens_details"]["reasoning_tokens"], int), "reasoning_tokens")
    return message


def check_models(client: Client) -> None:
    status, r = client.call("GET", "/models")
    check(status == 200 and r.get("object") == "list", "GET /v1/models")
    languages = [m for m in cast(list[Json], r["data"]) if m.get("nuclis", {}).get("kind") == "language"]
    check(len(languages) > 0, "no language model listed")
    print(f"  models: {len(languages)} language model(s)")


def check_conversation(client: Client) -> None:
    messages: list[Json] = [{"role": "developer", "content": "You are terse. Answer in one sentence."}]
    previous = 0
    for i, question in enumerate(["Name a prime number above 50.", "Is it odd?", "Give the next prime after it."]):
        messages.append({"role": "user", "content": question})
        status, r = client.chat({"messages": messages, "reasoning_effort": "none", "max_tokens": 120})
        check(status == 200, f"turn {i}: {status} {r}")
        message = completion_shape(r)
        usage = cast(Json, r["usage"])
        cached = cast(int, usage["prompt_tokens_details"]["cached_tokens"])
        if i > 0:
            check(cached >= previous * 0.8, f"turn {i}: cached {cached} of a {previous}-token earlier prompt")
        print(f"  turn {i}: prompt {usage['prompt_tokens']}, cached {cached}, generated {usage['completion_tokens']}")
        previous = cast(int, usage["prompt_tokens"])
        messages.append({"role": "assistant", "content": message["content"]})


def check_tools(client: Client) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    messages: list[Json] = [{"role": "user", "content": "What is the weather in Cairo right now? Use the tool."}]
    status, r = client.chat(
        {"messages": messages, "tools": tools, "reasoning_effort": "low", "max_completion_tokens": 800}
    )
    check(status == 200, f"tool call: {status} {r}")
    message = completion_shape(r)
    check(r["choices"][0]["finish_reason"] == "tool_calls", f"finish_reason {r['choices'][0]['finish_reason']}")
    calls = cast(list[Json], message["tool_calls"])
    call = calls[0]
    check(str(call["id"]).startswith("call_") and call["type"] == "function", "call id and type")
    check(isinstance(json.loads(call["function"]["arguments"]), dict), "arguments are a JSON object")
    messages.append(
        {
            "role": "assistant",
            "content": message["content"],
            "reasoning_content": message.get("reasoning_content"),
            "tool_calls": calls,
        }
    )
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"temp_c": 31, "sky": "clear"})})
    status, r = client.chat(
        {"messages": messages, "tools": tools, "reasoning_effort": "low", "max_completion_tokens": 800}
    )
    check(status == 200, f"tool result: {status} {r}")
    answer = completion_shape(r)
    usage = cast(Json, r["usage"])
    check("31" in (answer["content"] or ""), f"the answer uses the result: {answer['content']!r}")
    print(
        f"  tools: {call['function']['name']}({call['function']['arguments']}), then cached {usage['prompt_tokens_details']['cached_tokens']} of {usage['prompt_tokens']}"
    )


def check_stream(client: Client) -> None:
    chunks, _ = client.stream(
        {
            "messages": [{"role": "user", "content": "Count from 1 to 5, comma separated."}],
            "reasoning_effort": "low",
            "max_tokens": 400,
            "stream_options": {"include_usage": True},
        }
    )
    check(all(c.get("object") == "chat.completion.chunk" for c in chunks), "chunk object")
    check(len({c["id"] for c in chunks}) == 1, "one id across the chunks")
    choices = [cast(Json, c["choices"][0]) for c in chunks if c["choices"]]
    check(choices[0]["delta"].get("role") == "assistant", "the first delta carries the role")
    content = "".join(str(c["delta"].get("content") or "") for c in choices)
    reasoning = "".join(str(c["delta"].get("reasoning_content") or "") for c in choices)
    check("5" in content, f"content {content!r}")
    check(
        choices[-1]["finish_reason"] == "stop" and all(c["finish_reason"] is None for c in choices[:-1]),
        "finish_reason on the last choice only",
    )
    usage = cast(Json, chunks[-1].get("usage"))
    check(not chunks[-1]["choices"] and usage["completion_tokens"] > 0, "a last chunk with the usage")
    print(
        f"  stream: {len(choices)} chunks, {len(reasoning)} reasoning and {len(content)} answer characters, usage {usage['prompt_tokens']}/{usage['completion_tokens']}"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    chunks, _ = client.stream(
        {
            "messages": [{"role": "user", "content": "What is the weather in Cairo? Use the tool."}],
            "tools": tools,
            "reasoning_effort": "none",
            "max_tokens": 400,
        }
    )
    calls = [call for c in chunks for call in cast(list[Json], c["choices"][0]["delta"].get("tool_calls") or [])]
    check(
        len(calls) >= 1 and calls[0]["index"] == 0 and str(calls[0]["id"]).startswith("call_"),
        f"streamed calls {calls}",
    )
    check(isinstance(json.loads(calls[0]["function"]["arguments"]), dict), "streamed arguments are whole JSON")
    check(chunks[-1]["choices"][0]["finish_reason"] == "tool_calls", "finish_reason tool_calls")
    print(f"  stream: {calls[0]['function']['name']}({calls[0]['function']['arguments']})")


def check_image(client: Client, model: str | None) -> None:
    image = base64.b64encode((ROOT / "site/icons/icon-192.png").read_bytes()).decode()
    content = [
        {"type": "text", "text": "What colours are in this icon? One sentence."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
    ]
    status, r = client.chat(
        {"messages": [{"role": "user", "content": content}], "reasoning_effort": "none", "max_tokens": 80}, model
    )
    check(status == 200, f"image: {status} {r}")
    message = completion_shape(r)
    print(f"  image ({r['model']}): {str(message['content'])[:90]!r}")


def check_refusals(client: Client) -> None:
    user = [{"role": "user", "content": "x"}]
    cases: list[tuple[Json, int, str, str | None]] = [
        ({"messages": user, "response_format": {"type": "json_schema"}}, 400, "unsupported_feature", "response_format"),
        ({"messages": user, "tool_choice": "required"}, 400, "unsupported_feature", "tool_choice"),
        ({"messages": user, "n": 2}, 400, "unsupported_feature", "n"),
        (
            {"messages": [{"role": "tool", "tool_call_id": "nope", "content": "x"}]},
            400,
            "invalid_request",
            "messages[0]",
        ),
        ({"model": "laya", "messages": user}, 400, "not_a_language_model", "model"),
        ({"model": "no-such-model", "messages": user}, 404, "model_not_found", "model"),
    ]
    for body, status, code, param in cases:
        got, r = client.call("POST", "/chat/completions", body)
        error = cast(Json, r.get("error", {}))
        check(got == status and error.get("code") == code and error.get("param") == param, f"{body}: {got} {r}")
        check(
            error.get("type") in ("invalid_request_error", "server_error") and bool(error.get("message")),
            f"error body {r}",
        )
    print(f"  refusals: {len(cases)} answered with their code and field")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return cast(int, s.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="a running server's /v1 URL; without it, one is started")
    parser.add_argument("--model", help="the language model (default: the server's engine.model)")
    parser.add_argument("--image-model", help="the model the image check uses (default: --model)")
    parser.add_argument("--binary", default=str(ROOT / "zig-out/bin/nuclis"))
    args = parser.parse_args()

    server: subprocess.Popen[bytes] | None = None
    url = cast(str | None, args.url)
    if url is None:
        port = free_port()
        command = [args.binary, "serve", "--port", str(port), "--quiet"]
        if args.model:
            command += ["--chat-model", args.model]
        server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        url = f"http://127.0.0.1:{port}/v1"
        deadline = time.time() + 600
        while True:
            try:
                urllib.request.urlopen(url + "/health", timeout=2).close()
                break
            except OSError:
                if server.poll() is not None or time.time() > deadline:
                    print(f"serve did not start: {server.stderr.read().decode() if server.stderr else ''}")
                    return 1
                time.sleep(0.5)
    client = Client(url, cast(str | None, args.model))
    try:
        print(f"api-check against {url}")
        check_models(client)
        check_conversation(client)
        check_tools(client)
        check_stream(client)
        check_image(client, cast(str | None, args.image_model) or client.model)
        check_refusals(client)
        print("api-check: ok")
        return 0
    except Failure as failure:
        print(f"api-check: FAILED: {failure}")
        return 1
    finally:
        if server is not None:
            server.send_signal(2)
            server.wait(timeout=60)


if __name__ == "__main__":
    sys.exit(main())
