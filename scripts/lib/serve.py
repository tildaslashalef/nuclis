"""What the scripts that start or call `nuclis serve` share.

`Server` starts one on a free loopback port and waits for `/v1/health`; its
stderr goes to a temporary file, never a pipe nobody reads (a long run that
filled the pipe would block the server), and its tail is in the error when
startup fails. `chat_stream` and `speech_request` stream a response with
each piece's arrival time; `multipart` builds a form upload.

Standard library only. Import with:

    sys.path.insert(0, str(ROOT / "scripts/lib"))
    import serve
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from collections.abc import Iterator
from typing import IO, Any

HOST = "127.0.0.1"
Json = dict[str, Any]


class Failed(Exception):
    """The server did not start, or a request was answered with an error."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


class Server:
    """`nuclis serve --quiet` on a free port with `args`; a context manager.

    `env` replaces the environment (the caller's own when None); startup
    waits at most `timeout` seconds, read off the clock.
    """

    def __init__(self, nuclis: str, args: list[str], env: dict[str, str] | None = None, timeout: float = 600) -> None:
        self.port = free_port()
        self.url = f"http://{HOST}:{self.port}/v1"
        self.command = [nuclis, "serve", "--host", HOST, "--port", str(self.port), "--quiet", *args]
        self._stderr: IO[bytes] = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.DEVNULL, stderr=self._stderr, env=os.environ if env is None else env
        )
        deadline = time.monotonic() + timeout
        while True:
            try:
                connection = http.client.HTTPConnection(HOST, self.port, timeout=2)
                connection.request("GET", "/v1/health")
                connection.getresponse().read()
                connection.close()
                return
            except OSError:
                if self.process.poll() is not None or time.monotonic() > deadline:
                    tail = self.stderr_tail()
                    self.stop()
                    raise Failed(f"{' '.join(self.command)} did not start:\n{tail}") from None
                time.sleep(0.2)

    def stderr_tail(self, lines: int = 20) -> str:
        self._stderr.seek(0)
        return "\n".join(self._stderr.read().decode(errors="replace").splitlines()[-lines:])

    def rss(self) -> int:
        """The server's resident set, in bytes."""
        done = subprocess.run(["ps", "-o", "rss=", "-p", str(self.process.pid)], capture_output=True, text=True)
        return int(done.stdout.strip() or 0) * 1024

    def stop(self) -> None:
        """Ctrl-C, then a kill after 60 s."""
        if self.process.poll() is None:
            self.process.send_signal(2)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._stderr.close()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def multipart(
    fields: list[tuple[str, str]], file_field: str, filename: str, content_type: str, data: bytes
) -> tuple[bytes, str]:
    """A multipart/form-data body with `fields` then one file; the body and its content type."""
    boundary = "nuclis-form-6f1d2b"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{n}"\r\n\r\n{v}\r\n'.encode() for n, v in fields]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode()
    )
    parts += [data, f"\r\n--{boundary}--\r\n".encode()]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _post(
    port: int, path: str, body: Json, timeout: float, host: str
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    connection.request("POST", path, json.dumps(body), {"content-type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        text = response.read().decode(errors="replace")[:400]
        connection.close()
        raise Failed(f"{path}: {response.status} {text}")
    return connection, response


def chat_stream(port: int, body: Json, timeout: float = 600, host: str = HOST) -> Iterator[tuple[float, Json]]:
    """A streamed chat completion (`body` gets `stream: true`): each chunk with its arrival time."""
    connection, response = _post(port, "/v1/chat/completions", {**body, "stream": True}, timeout, host)
    try:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                yield time.monotonic(), json.loads(line[6:])
    finally:
        connection.close()


def content(chunk: Json) -> str | None:
    """A chat chunk's content delta, if it has one."""
    choices = chunk.get("choices") or []
    return (choices[0].get("delta") or {}).get("content") if choices else None


def speech_request(
    port: int, body: Json, piece: int = 65536, timeout: float = 600, host: str = HOST
) -> Iterator[tuple[float, bytes]]:
    """`POST /v1/audio/speech`: the audio as it arrives, at most `piece` bytes a read, with arrival times."""
    connection, response = _post(port, "/v1/audio/speech", body, timeout, host)
    try:
        while data := response.read1(piece):
            yield time.monotonic(), data
    finally:
        connection.close()


class Tests(unittest.TestCase):
    def test_multipart(self) -> None:
        body, kind = multipart([("language", "en")], "file", "a.wav", "audio/wav", b"RIFF")
        boundary = kind.split("boundary=")[1]
        self.assertTrue(body.startswith(f"--{boundary}\r\n".encode()))
        self.assertIn(b'name="language"\r\n\r\nen\r\n', body)
        self.assertIn(b'name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n', body)
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))

    def test_content(self) -> None:
        self.assertEqual(content({"choices": [{"delta": {"content": "hi"}}]}), "hi")
        self.assertIsNone(content({"choices": []}))

    def test_a_server_that_exits_names_its_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "nuclis")
            with open(fake, "w") as f:
                f.write("#!/bin/sh\necho 'error: no model' >&2\nexit 1\n")
            os.chmod(fake, 0o755)
            started = time.monotonic()
            with self.assertRaises(Failed) as caught:
                Server(fake, [], timeout=30)
            self.assertLess(time.monotonic() - started, 30)
        self.assertIn("error: no model", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
