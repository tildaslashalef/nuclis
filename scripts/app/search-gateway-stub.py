#!/usr/bin/env python3
"""A local stand-in for the search service behind the agent's web tools:
the v1 contract in docs/app/agent.md § The web tools, answered from canned data,
for the screenshot harness, `nuclis serve` checks, and any run where the
real service is out of reach. Standard library only; nothing leaves the
machine.

    python3 scripts/app/search-gateway-stub.py              # http://127.0.0.1:8099
    python3 scripts/app/search-gateway-stub.py --port 9100
    python3 scripts/app/search-gateway-stub.py --no-context # a host without /v1/context

Point the configuration at it: `web.url` http://127.0.0.1:8099 (http is
accepted on loopback only), then `agent.web` or `serve.web` true.

The answers are deterministic. Any query searches the three pages below,
the first two with snippets; a query of `busy` answers `busy`, `nothing` an
empty list, `broken` `provider_unavailable` (every provider skipped), one
containing `fallback` is answered by Brave with Parallel skipped
(`providers_failed`), and one containing `pyhton` is searched as corrected
(`altered_query`). Context answers canned passages for any query, none for
`nothing`; `--no-context` leaves `/v1/context` out of health and answers it
`501 not_available`. `freshness` is checked as the contract says. The first
answer to a search, a context, or a fetch with `q` carries
`X-Search-Cache: miss`, a repeat `hit`.

A fetch of one of the pages serves its text, paged by `start` and
`max_chars`, as the local fetcher does (`status` and `content_type` set);
with `q`, as Parallel does: the paragraphs that share a word with the
question, `focused` true, `status` and `content_type` null. A private or
local host is `blocked_address`, a `.zip` is `unsupported_content`,
`challenged.example` is the bot-challenge `upstream_failed`, a host named
`slow.example` waits 26 s (past the service's 25 s budget, so it answers
`upstream_timeout`), and any other URL is `upstream_failed`.
"""

import argparse
import ipaddress
import json
import re
import sys
import threading
import time
import urllib.parse
from typing import cast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "stub-0.6.0"
MAX_QUERY_BYTES = 400
FOCUSED_CHARS = 8_000
MAX_RESULTS, DEFAULT_RESULTS = 10, 8
MAX_CHARS, DEFAULT_CHARS = 100_000, 20_000
MIN_TOKENS, MAX_TOKENS, DEFAULT_TOKENS = 1024, 16384, 8192
MAX_URLS, DEFAULT_URLS = 20, 20
FRESHNESS = re.compile(r"^(pd|pw|pm|py|\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2})$")

ZIG_HTTP = """# std.http.Client

A HTTP(S) client: connection pooling, TLS through `std.crypto.tls`, and
redirects it can follow or hand back.

## Making a request

```zig
var client: std.http.Client = .{ .allocator = gpa, .io = io };
defer client.deinit();
var req = try client.request(.GET, uri, .{});
defer req.deinit();
try req.sendBodiless();
var response = try req.receiveHead(&.{});
```

## Reading the body

`response.reader(&buffer)` returns a reader over the body; read it to the
end so the connection returns to the pool.

- `keep_alive`: reuse the connection (default true).
- `redirect_behavior`: `.unhandled` returns redirects to the caller.
"""

PEP8 = "# PEP 8 – Style Guide for Python Code\n\n" + "\n\n".join(
    f"## Section {n}\n\n" + " ".join(f"Guideline {n}.{k}: keep lines short and names clear." for k in range(1, 60))
    for n in range(1, 25)
)

PAGES = {
    "https://ziglang.org/documentation/master/std/#std.http.Client": (
        "std.http.Client — Zig standard library",
        "text/html",
        ZIG_HTTP,
    ),
    "https://peps.python.org/pep-0008/": ("PEP 8 – Style Guide for Python Code", "text/html", PEP8),
    "https://example.org/notes.txt": ("", "text/plain", "plain notes\nsecond line\n"),
}

RESULTS = [
    {
        "title": "std.http.Client — Zig standard library",
        "url": "https://ziglang.org/documentation/master/std/#std.http.Client",
        "snippet": "A HTTP(S) client: connection pooling, TLS, and redirects it can follow or hand back.",
        "snippets": [
            "var req = try client.request(.GET, uri, .{}); then sendBodiless and receiveHead.",
            "redirect_behavior: .unhandled returns redirects to the caller.",
            "keep_alive reuses the connection; read the body to the end to return it to the pool.",
        ],
        "published": None,
    },
    {
        "title": "PEP 8 – Style Guide for Python Code",
        "url": "https://peps.python.org/pep-0008/",
        "snippet": "This document gives coding conventions for the Python code comprising the standard library.",
        "snippets": ["Limit all lines to a maximum of 79 characters."],
        "published": "2001-07-05",
    },
    {
        "title": "Plain notes",
        "url": "https://example.org/notes.txt",
        "snippet": "A plain text file, served as text.",
        "snippets": [],
        "published": None,
    },
]

SOURCES = [
    {
        "url": "https://ziglang.org/documentation/master/std/#std.http.Client",
        "title": "std.http.Client — Zig standard library",
        "published": None,
        "snippets": [
            "A HTTP(S) client: connection pooling, TLS through std.crypto.tls, and redirects it can follow or hand back.",
            "var client: std.http.Client = .{ .allocator = gpa, .io = io };\ndefer client.deinit();\nvar req = try client.request(.GET, uri, .{});",
        ],
    },
    {
        "url": "https://peps.python.org/pep-0008/",
        "title": "PEP 8 – Style Guide for Python Code",
        "published": "2001-07-05",
        "snippets": ["Limit all lines to a maximum of 79 characters.", "Use 4 spaces per indentation level."],
    },
    {
        "url": "https://stackoverflow.com/questions/1/challenged",
        "title": "How do I read a response body? - Stack Overflow",
        "published": "2025-03-02",
        "snippets": ["Read the body to the end so the connection goes back to the pool."],
    },
]

CHALLENGED = "challenged.example"

LOCAL_SUFFIXES = (".home.arpa", ".local", ".localhost", ".internal")


class Failure(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def integer(params, name, default, low, high):
    raw = params.get(name, [""])[0]
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise Failure(400, "bad_request", f"{name} must be an integer") from None
    if value < low:
        raise Failure(400, "bad_request", f"{name} must be at least {low}")
    return min(value, high)


def checked_query(params):
    query = params.get("q", [""])[0].strip()
    if not query:
        raise Failure(400, "bad_request", "q is empty")
    if len(query.encode()) > MAX_QUERY_BYTES:
        raise Failure(400, "bad_request", f"q is longer than {MAX_QUERY_BYTES} bytes")
    freshness = params.get("freshness", [""])[0]
    if freshness and not FRESHNESS.match(freshness):
        raise Failure(400, "bad_request", "freshness must be pd, pw, pm, py, or YYYY-MM-DDtoYYYY-MM-DD")
    return query


def search(params):
    query = checked_query(params)
    count = integer(params, "n", DEFAULT_RESULTS, 1, MAX_RESULTS)
    if query == "busy":
        raise Failure(503, "busy", "every request slot stayed taken for 2 s")
    if query == "broken":
        raise Failure(
            503, "provider_unavailable", "every provider was skipped: parallel (out of credit), brave (rate-limited)"
        )
    results = [] if query == "nothing" else RESULTS[:count]
    fallback = "fallback" in query
    return {
        "query": query,
        "altered_query": query.replace("pyhton", "python") if "pyhton" in query else None,
        "provider": "brave" if fallback else "parallel",
        "results": results,
        "providers_failed": ["parallel"] if fallback else [],
    }


def context(params, offered):
    if not offered:
        raise Failure(501, "not_available", "this host offers no /v1/context")
    query = checked_query(params)
    integer(params, "tokens", DEFAULT_TOKENS, MIN_TOKENS, MAX_TOKENS)
    urls = integer(params, "urls", DEFAULT_URLS, 1, MAX_URLS)
    return {
        "query": query,
        "provider": "parallel",
        "sources": [] if query == "nothing" else SOURCES[:urls],
        "providers_failed": [],
    }


def blocked(host):
    name = host.lower().rstrip(".")
    try:
        return not ipaddress.ip_address(name.strip("[]")).is_global
    except ValueError:
        return name.endswith(LOCAL_SUFFIXES) or "." not in name


def fetch(params):
    url = params.get("url", [""])[0]
    if not url:
        raise Failure(400, "bad_request", "url is empty")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise Failure(400, "bad_request", "url must be an http or https URL with a host")
    if blocked(parts.hostname):
        raise Failure(403, "blocked_address", f"{parts.hostname} is not a public address")
    if parts.path.endswith(".zip"):
        raise Failure(415, "unsupported_content", "application/zip is not extracted")
    if parts.hostname == CHALLENGED:
        raise Failure(
            502,
            "upstream_failed",
            f"{CHALLENGED} answered with a bot challenge (Cloudflare) that only a browser passes; "
            "try /v1/context, whose provider has the page's text",
        )
    if parts.hostname == "slow.example":
        time.sleep(26)
        raise Failure(504, "upstream_timeout", "slow.example did not answer within 25 s")
    page = PAGES.get(url)
    if page is None:
        raise Failure(502, "upstream_failed", f"{parts.hostname} answered 404 Not Found")
    title, content_type, text = page
    question = params.get("q", [""])[0].strip()
    if "q" in params:
        return focused(url, title, text, question)
    start = integer(params, "start", 0, 0, sys.maxsize)
    max_chars = integer(params, "max_chars", DEFAULT_CHARS, 1, MAX_CHARS)
    piece = text[start : start + max_chars]
    return {
        "url": url,
        "final_url": url,
        "provider": "local",
        "providers_failed": [],
        "status": 200,
        "content_type": content_type,
        "title": title,
        "published": None,
        "focused": False,
        "text": piece,
        "start": start,
        "chars": len(text),
        "truncated": start + len(piece) < len(text),
    }


def focused(url, title, text, question):
    if not question:
        raise Failure(400, "bad_request", "q is empty")
    if len(question.encode()) > MAX_QUERY_BYTES:
        raise Failure(400, "bad_request", f"q is longer than {MAX_QUERY_BYTES} bytes")
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) >= 3}
    parts = [p for p in text.split("\n\n") if words & set(re.findall(r"\w+", p.lower()))]
    answer = "\n\n".join(parts)[:FOCUSED_CHARS]
    return {
        "url": url,
        "final_url": url,
        "provider": "parallel",
        "providers_failed": [],
        "status": None,
        "content_type": None,
        "title": title,
        "published": None,
        "focused": True,
        "text": answer,
        "start": 0,
        "chars": len(answer),
        "truncated": False,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"search-gateway-stub/{VERSION}"

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = urllib.parse.parse_qs(query, keep_blank_values=True)
        server = cast("Server", self.server)
        offered = server.context_offered
        cache = None
        try:
            if path == "/v1/search":
                status, body = 200, search(params)
                cache = server.seen(self.path)
            elif path == "/v1/context":
                status, body = 200, context(params, offered)
                cache = server.seen(self.path)
            elif path == "/v1/fetch":
                status, body = 200, fetch(params)
                if "q" in params:
                    cache = server.seen(self.path)
            elif path == "/v1/health":
                endpoints = ["/v1/search", "/v1/fetch", "/v1/health"] + (["/v1/context"] if offered else [])
                providers = {"parallel": "ok", "brave": "ok", "local": "ok"}
                routes = {"/v1/search": ["parallel", "brave"], "/v1/fetch": ["parallel", "local"]}
                if offered:
                    routes["/v1/context"] = ["parallel", "brave"]
                status, body = (
                    200,
                    {
                        "version": VERSION,
                        "providers": providers,
                        "routes": routes,
                        "endpoints": endpoints,
                        "billed": {"parallel": server.billed},
                    },
                )
            else:
                raise Failure(404, "not_found", f"no endpoint {path}")
        except Failure as failure:
            status, body = failure.status, {"error": {"code": failure.code, "message": failure.message}}
        self.answer(status, body, cache)
        shown = f" cache={cache}" if cache else ""
        print(f"{self.path} {status}{shown} ua={self.headers.get('User-Agent', '-')}", file=sys.stderr, flush=True)

    def do_POST(self):
        self.answer(405, {"error": {"code": "method_not_allowed", "message": "only GET"}})

    def answer(self, status, body, cache=None):
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if cache:
            self.send_header("X-Search-Cache", cache)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):  # noqa: A002 (the base class's name)
        pass


class Server(ThreadingHTTPServer):
    """The stub's state: whether it offers context, and the searches seen,
    for the cache header and health's `billed` (one per miss)."""

    def __init__(self, address, context_offered):
        super().__init__(address, Handler)
        self.context_offered = context_offered
        self.billed = 0
        self._seen = set()
        self._lock = threading.Lock()

    def seen(self, target):
        with self._lock:
            if target in self._seen:
                return "hit"
            self._seen.add(target)
            self.billed += 1
            return "miss"


def main():
    parser = argparse.ArgumentParser(description="A local stand-in for the search service behind the web tools.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--no-context", action="store_true", help="play a host without /v1/context")
    args = parser.parse_args()
    server = Server((args.host, args.port), not args.no_context)
    server.daemon_threads = True
    print(f"search-gateway-stub on http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
