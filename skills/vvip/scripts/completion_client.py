"""Direct native completion SSE with absolute and output-sensitive deadlines.

Stdlib only; no redirects, proxy discovery, retries, or release inference.
The OS hostname resolver is outside Python socket timeout control.
"""
from contextlib import contextmanager
import http.client
import json
import math
import os
import socket
import threading
import time
from urllib.parse import urlsplit


def has_output(chunk):
    return any(choice.get("token_ids") or choice.get("text")
               for choice in chunk.get("choices", []))


@contextmanager
def completion_events(base, payload, timeout, *, ttft_timeout=None, idle_timeout=None):
    ttft_timeout = timeout if ttft_timeout is None else ttft_timeout
    idle_timeout = timeout if idle_timeout is None else idle_timeout
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
           for v in (timeout, ttft_timeout, idle_timeout)):
        raise ValueError("timeouts must be positive finite seconds")
    url = urlsplit(base)
    if (url.scheme not in {"http", "https"} or not url.hostname
            or url.username is not None or url.password is not None or url.query or url.fragment):
        raise ValueError("use a direct HTTP(S) base URL without credentials, query, or fragment")
    started = time.monotonic()
    total_deadline = started + timeout
    phase_deadline = started + ttft_timeout
    phase, expired, stopped = "first_output", None, False
    condition = threading.Condition()
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=min(timeout, ttft_timeout))
    response = watchdog = None

    try:
        connection.connect()
        transport = connection.sock  # Retain ownership even for Connection: close.

        def check_deadline():
            nonlocal expired
            if expired is None:
                now = time.monotonic()
                if now >= total_deadline:
                    expired = "total"
                elif now >= phase_deadline:
                    expired = phase
            if expired is not None:
                raise TimeoutError(f"{expired} deadline exceeded")

        def guard():
            nonlocal expired
            with condition:
                while not stopped:
                    remaining = min(total_deadline, phase_deadline) - time.monotonic()
                    if remaining <= 0:
                        expired = "total" if total_deadline <= phase_deadline else phase
                        # close() alone may wait for the file object's blocked
                        # read. shutdown() wakes it, including partial SSE lines.
                        try:
                            transport.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        return
                    condition.wait(remaining)

        check_deadline()
        # The watchdog owns wall deadlines; do not accidentally impose the
        # shorter TTFT budget on output reads after the first real token.
        transport.settimeout(timeout)
        watchdog = threading.Thread(target=guard, daemon=True)
        watchdog.start()
        headers = {"Content-Type": "application/json"}
        if os.environ.get("VVIP_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["VVIP_API_KEY"]
        connection.request("POST", url.path.rstrip("/") + "/v1/completions",
                           body=json.dumps(payload).encode(), headers=headers)
        response = connection.getresponse()
        check_deadline()
        if response.status != 200:
            raise RuntimeError(f"completion HTTP status {response.status}")

        def chunks():
            nonlocal phase, phase_deadline
            while True:
                line = response.readline(1024 * 1024 + 1)
                check_deadline()
                if not line:
                    return
                if len(line) > 1024 * 1024:
                    raise RuntimeError("SSE line exceeds 1 MiB")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    yield None
                    return
                chunk = json.loads(data)
                if not isinstance(chunk, dict) or "error" in chunk:
                    raise RuntimeError("invalid completion SSE or server error")
                if has_output(chunk):
                    with condition:
                        check_deadline()
                        phase = "output_idle"
                        phase_deadline = time.monotonic() + idle_timeout
                        condition.notify()
                yield chunk

        yield chunks()
    except (OSError, http.client.HTTPException):
        if expired is None and time.monotonic() >= min(total_deadline, phase_deadline):
            expired = "total" if total_deadline <= phase_deadline else phase
        if expired is not None:
            raise TimeoutError(f"{expired} deadline exceeded") from None
        raise
    finally:
        with condition:
            stopped = True
            condition.notify()
        if watchdog is not None:
            watchdog.join()
        if response is not None:
            response.close()
        connection.close()
