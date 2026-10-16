#!/usr/bin/env python3
"""A local stand-in for the search service behind the agent's web tools:
holocron's v1 contract (docs/app/agent.md § 11) answered from canned data,
for the screenshot harness, `nuclis serve` checks, and any run where the
lab's service is out of reach. Standard library only; nothing leaves the
machine.

    python3 scripts/search-gateway-stub.py              # http://127.0.0.1:8099
    python3 scripts/search-gateway-stub.py --port 9100

Point the configuration at it: `web.url` http://127.0.0.1:8099 (http is
accepted on loopback only), then `agent.web` or `serve.web` true.

The answers are deterministic. Any query searches the three pages below;
a query of `busy` answers `busy`, `nothing` an empty list, `broken` the
no-engines failure. A fetch of one of the pages serves its text, paged by
`start` and `max_chars`; a private or lab host is `blocked_address`, a
`.pdf` is `unsupported_content`, a host named `slow.example` waits 25 s
(past the service's 20 s budget, so it answers `upstream_timeout`), and any
other URL is `upstream_failed`.
"""

import argparse
import ipaddress
import json
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "stub-0.1.0"
MAX_QUERY_BYTES = 400
MAX_RESULTS, DEFAULT_RESULTS = 10, 8
MAX_CHARS, DEFAULT_CHARS = 100_000, 20_000

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
        "engine": "duckduckgo",
        "published": None,
    },
    {
        "title": "PEP 8 – Style Guide for Python Code",
        "url": "https://peps.python.org/pep-0008/",
        "snippet": "This document gives coding conventions for the Python code comprising the standard library.",
        "engine": "brave",
        "published": "2001-07-05",
    },
    {
        "title": "Plain notes",
        "url": "https://example.org/notes.txt",
        "snippet": "A plain text file, served as text.",
        "engine": "mojeek",
        "published": None,
    },
]

LAB_SUFFIXES = (".lab.nuclis.dev", ".home.arpa", ".local", ".localhost", ".internal")
LAB_NAMES = {"localhost", "kamino", "coruscant", "unifi"}


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


def search(params):
    query = params.get("q", [""])[0].strip()
    if not query:
        raise Failure(400, "bad_request", "q is empty")
    if len(query.encode()) > MAX_QUERY_BYTES:
        raise Failure(400, "bad_request", f"q is longer than {MAX_QUERY_BYTES} bytes")
    count = integer(params, "n", DEFAULT_RESULTS, 1, MAX_RESULTS)
    if query == "busy":
        raise Failure(503, "busy", "every request slot stayed taken for 2 s")
    if query == "broken":
        raise Failure(503, "no_engines", "no results, and these engines failed: duckduckgo, brave, mojeek")
    results = [] if query == "nothing" else RESULTS[:count]
    return {
        "query": query,
        "provider": "searxng",
        "results": results,
        "engines_failed": ["startpage"] if results else [],
    }


def blocked(host):
    name = host.lower().rstrip(".")
    try:
        return not ipaddress.ip_address(name.strip("[]")).is_global
    except ValueError:
        return name in LAB_NAMES or name.endswith(LAB_SUFFIXES) or "." not in name


def fetch(params):
    url = params.get("url", [""])[0]
    if not url:
        raise Failure(400, "bad_request", "url is empty")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise Failure(400, "bad_request", "url must be an http or https URL with a host")
    if blocked(parts.hostname):
        raise Failure(403, "blocked_address", f"{parts.hostname} is not a public address")
    if parts.path.endswith(".pdf"):
        raise Failure(415, "unsupported_content", "application/pdf is not extracted")
    if parts.hostname == "slow.example":
        time.sleep(25)
        raise Failure(504, "upstream_timeout", "slow.example did not answer within 20 s")
    page = PAGES.get(url)
    if page is None:
        raise Failure(502, "upstream_failed", f"{parts.hostname} answered 404")
    title, content_type, text = page
    start = integer(params, "start", 0, 0, sys.maxsize)
    max_chars = integer(params, "max_chars", DEFAULT_CHARS, 1, MAX_CHARS)
    piece = text[start : start + max_chars]
    return {
        "url": url,
        "final_url": url,
        "status": 200,
        "content_type": content_type,
        "title": title,
        "text": piece,
        "start": start,
        "chars": len(text),
        "truncated": start + len(piece) < len(text),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"search-gateway-stub/{VERSION}"

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = urllib.parse.parse_qs(query, keep_blank_values=True)
        try:
            if path == "/v1/search":
                status, body = 200, search(params)
            elif path == "/v1/fetch":
                status, body = 200, fetch(params)
            elif path == "/v1/health":
                status, body = 200, {"version": VERSION, "providers": {"searxng": "ok"}}
            else:
                raise Failure(404, "not_found", f"no endpoint {path}")
        except Failure as failure:
            status, body = failure.status, {"error": {"code": failure.code, "message": failure.message}}
        self.answer(status, body)
        print(f"{path} {status} ua={self.headers.get('User-Agent', '-')}", file=sys.stderr, flush=True)

    def do_POST(self):
        self.answer(405, {"error": {"code": "method_not_allowed", "message": "only GET"}})

    def answer(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):  # noqa: A002 (the base class's name)
        pass


def main():
    parser = argparse.ArgumentParser(description="A local stand-in for the search service behind the web tools.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"search-gateway-stub on http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
