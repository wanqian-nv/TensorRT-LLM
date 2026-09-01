#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stable front door for the Anthropic-compatibility servers.
#
# A serving job lives for hours and lands on whatever node Slurm gives it, so
# its URL changes every time it is rescheduled. Users need one address that
# never changes. This gateway holds that address and forwards to whichever
# backend is currently healthy, so `ANTHROPIC_BASE_URL` is written once and
# never edited again.
#
#   ./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml
#
# It starts idle and holds no GPUs. Two unauthenticated POST endpoints decide
# whether a serving job exists at all, so the front door can stay up for a week
# while the expensive half of the deployment comes and goes:
#
#   curl -XPOST http://<gateway>:8333/_gateway/start_server   # find GPUs, serve
#   curl -XPOST http://<gateway>:8333/_gateway/stop_server    # give them back
#
# Between those two the gateway behaves exactly as before: it relays a
# successor in before the wall clock and reclaims the predecessor. Outside
# them it is a proxy for whatever it happens to find, and nothing more.
#
# Standard library only, on purpose: the gateway has to outlive every serving
# job, so it runs outside the TRT-LLM container on whatever long-lived host is
# available. Requiring httpx or uvicorn there would mean a venv, which means
# outbound network -- one more thing that host has to provide.

import argparse
import asyncio
import collections
import glob
import json
import logging
import os
import re
import sys
import time
import urllib.parse

LOG = logging.getLogger("gateway")

# Hop-by-hop request headers plus the ones this gateway owns. Request framing
# headers (content-length, transfer-encoding, te, trailer) deliberately survive:
# request bodies are relayed byte for byte. SSE response framing is normalized
# separately so the gateway can append a valid terminal error event.
STRIP_REQUEST_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "upgrade",
    "host",
    "x-api-key",
    "authorization",
    "accept-encoding",
}

MAX_HEAD_BYTES = 64 * 1024
RELAY_CHUNK = 64 * 1024
PENDING_VISIBILITY_GRACE = 60

# The gateway now submits a serving job on its own from a cold start, so a
# deployment sbatch refuses outright -- expired reservation, full account,
# broken YAML -- would otherwise queue one every supervisor sweep for as long
# as the gateway lives. Doubles per consecutive failure, up to ten minutes; any
# backend coming up, or an explicit start_server, clears it.
SUBMIT_BACKOFF_BASE = 30
SUBMIT_BACKOFF_MAX = 600

# `serve.sh quit` only touches a control file. The controller notices within two
# seconds, TERMs the launcher, and cleanup_workers spends another five waiting
# before it escalates -- so rc=0 says the request was filed, not that the GPUs
# came back. stop_server watches Slurm for this long before it says "released",
# because saying it wrongly is the one thing that endpoint must not do.
RELEASE_VERIFY_SECONDS = 45
RELEASE_POLL_INTERVAL = 3
# States in which Slurm is still holding nodes for the job.
HOLDING_STATES = {"RUNNING", "PENDING", "CONFIGURING", "SUSPENDED", "RESIZING",
                  "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
                  "STOPPED"}

# Labels reach the filesystem: serve.sh builds the run directory name out of
# one. The control endpoints are unauthenticated, so a caller-supplied label is
# reduced to something that cannot escape the trace root.
LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Anthropic-shaped so a client's error handling reports something meaningful
# instead of a bare transport failure.
ERROR_BODIES = {
    401: ("authentication_error", "unknown api key; ask the gateway owner to "
                                  "add your username to users.txt"),
    404: ("not_found_error", "no such gateway endpoint"),
    405: ("invalid_request_error", "this endpoint only accepts POST"),
    409: ("invalid_request_error", "this gateway runs with --no-relay, so it "
                                   "does not start or stop serving jobs"),
    502: ("api_error", "backend refused the connection"),
    503: ("overloaded_error", "no healthy backend right now; the serving job "
                              "is rotating, retry shortly"),
}

REASONS = {200: "OK", 401: "Unauthorized", 404: "Not Found",
           405: "Method Not Allowed", 409: "Conflict", 502: "Bad Gateway",
           503: "Service Unavailable"}

SSE_ROTATED = (b'event: error\n'
               b'data: {"type":"error","error":{"type":"overloaded_error",'
               b'"message":"backend rotated mid-stream; the response is '
               b'incomplete, please resend"}}\n\n')


# ---------------------------------------------------------------------------
# Fleet state
# ---------------------------------------------------------------------------
class Backend:
    """One serving job, as seen through its registration file."""

    def __init__(self, record):
        self.job_id = str(record["job_id"])
        self.url = record["url"].rstrip("/")
        self.run_dir = record.get("run_dir", "")
        self.state = record.get("state", "")
        self.end_time = float(record.get("end_time") or 0)
        self.heartbeat = float(record.get("heartbeat") or 0)
        self.healthy = False
        self.timeouts = 0        # consecutive probe timeouts
        self.healthy_since = 0.0
        # Probing resolves the URL once; every request reuses host/port.
        match = re.match(r"^http://([^:/]+):(\d+)$", self.url)
        if not match:
            raise ValueError("unusable url %r" % self.url)
        self.host = match.group(1)
        self.port = int(match.group(2))

    def refresh(self, record):
        self.state = record.get("state", self.state)
        self.heartbeat = float(record.get("heartbeat") or 0)


class Fleet:
    """Everything the request path and the supervisor share."""

    def __init__(self, args):
        self.args = args
        self.backends = {}          # job_id -> Backend
        self.active = None          # job_id currently taking new requests
        self.draining = {}          # job_id -> reclaim deadline (unix ts)
        # A default dict on purpose. Requests outlive their backend's entry --
        # discovery can retire a job while its streams are still draining -- and
        # the release below runs in a finally that also closes the client
        # socket. A KeyError there would leak the connection, so counting must
        # not be able to raise.
        self.inflight = collections.defaultdict(int)
        self.users = set()
        self.users_mtime = 0.0
        # (job_id, since), retained through registration until healthy, plus
        # whether it has been seen RUNNING yet -- the queue wait and the
        # registration wait need separate clocks.
        self.pending = None
        self.pending_running = False
        # Replaced, but not yet cleared for reclaim: draining ends in
        # `serve.sh quit`, so it waits until the successor has proven itself.
        self.superseded = set()
        self.started = time.time()

        # Lifecycle is explicit. The gateway comes up idle and holds no GPUs
        # until somebody POSTs /_gateway/start_server; stop_server hands them
        # all back and leaves the gateway as the only thing running. Routing is
        # deliberately NOT gated on this: a gateway restarted underneath a live
        # serving job still forwards to it, it just does not relay or reclaim
        # it until asked. So this says "who owns the job lifecycle", not "is
        # anything being served" -- `active` already answers that.
        self.desired = "stopped"
        # The fleet directory is namespaced <trace_root>/_fleet/<cluster>_<model>,
        # so its name is the deployment's. Reported in /_gateway/health purely
        # so that `serve.sh gateway --stop` can check it is talking to the
        # gateway it thinks it is before releasing anybody's GPUs.
        self.name = os.path.basename(os.path.normpath(args.fleet_dir))
        # Everything that submits or cancels a job holds this. Both the control
        # endpoints and the supervisor do so across awaits, and interleaving
        # them loses writes: two concurrent starts would each see no pending
        # job and submit one.
        self.control = asyncio.Lock()
        # Job ids stop_server has released. `serve.sh quit` is asynchronous --
        # the controller polls its control directory every two seconds and then
        # takes the server down -- so these keep heartbeating for a while after
        # being told to go. Without this they would win the next election and
        # take traffic to a server that is shutting down. Entries for jobs that
        # never registered are kept on purpose: that is exactly the job that
        # might still be mid-registration when scancel reaches it.
        self.stopping = set()
        # Consecutive failures to get a serving job up, and the earliest time
        # another attempt is worth making.
        self.submit_failures = 0
        self.retry_after = 0.0

    def track_pending(self, job_id, now):
        """Start the clock on a successor that is not serving yet."""
        self.pending = (job_id, now)
        self.pending_running = False

    def forget_pending(self):
        self.pending = None
        self.pending_running = False

    def live(self):
        """Backends eligible for traffic and for relay.

        Everything except what stop_server released. Those records outlive the
        request to release them, and every decision here -- election, relay,
        the cold-start submit -- would read them as a healthy serving job.
        """
        return {job_id: backend for job_id, backend in self.backends.items()
                if job_id not in self.stopping}

    def submit_failed(self, now):
        """Record a failed attempt at a serving job; return the backoff."""
        self.submit_failures += 1
        delay = SUBMIT_BACKOFF_BASE * 2 ** min(self.submit_failures - 1, 5)
        self.retry_after = now + min(SUBMIT_BACKOFF_MAX, delay)
        return int(self.retry_after - now)

    def submit_succeeded(self):
        self.submit_failures = 0
        self.retry_after = 0.0

    # -- users ------------------------------------------------------------
    def reload_users(self):
        path = self.args.users
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            if self.users:
                LOG.warning("users file disappeared: %s (keeping %d entries)",
                            path, len(self.users))
            return
        if mtime == self.users_mtime:
            return
        names = set()
        with open(path) as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    names.add(line)
        self.users_mtime = mtime
        if names != self.users:
            LOG.info("users reloaded: %d entries", len(names))
        self.users = names

    # -- discovery --------------------------------------------------------
    def discover(self):
        """Rebuild the backend table from the registration directory.

        Each serving job owns exactly one file named after its Slurm job id, so
        there is never more than one writer per file and the gateway never has
        to coordinate with anybody. The union is just the directory listing.
        """
        now = time.time()
        seen = set()
        for path in glob.glob(os.path.join(self.args.fleet_dir, "*.json")):
            try:
                with open(path) as handle:
                    record = json.load(handle)
            except (OSError, ValueError):
                # Mid-rename, or truncated. The writer replaces the file
                # atomically, so the next sweep gets a whole one.
                continue
            job_id = str(record.get("job_id", ""))
            if not job_id:
                continue
            if now - float(record.get("heartbeat") or 0) > self.args.stale_after:
                continue
            seen.add(job_id)
            if job_id in self.backends:
                self.backends[job_id].refresh(record)
            else:
                try:
                    self.backends[job_id] = Backend(record)
                except (KeyError, ValueError) as exc:
                    LOG.warning("ignoring %s: %s", path, exc)
                    continue
                self.inflight.setdefault(job_id, 0)
                LOG.info("backend appeared: %s at %s (ends %s)", job_id,
                         self.backends[job_id].url,
                         fmt_time(self.backends[job_id].end_time))

        for job_id in [j for j in self.backends if j not in seen]:
            LOG.info("backend gone: %s (no heartbeat for %ds)", job_id,
                     self.args.stale_after)
            self.backends.pop(job_id, None)
            self.draining.pop(job_id, None)
            self.superseded.discard(job_id)
            self.stopping.discard(job_id)
            # Keep the counter while anything is still streaming off this
            # backend; a later sweep collects it once the count reaches zero.
            if not self.inflight.get(job_id):
                self.inflight.pop(job_id, None)

        # Discovery, probing and supervision run on independent timers, so
        # retiring a backend has to clear the pointer to it here rather than
        # waiting for the next election. Otherwise `active` names a job that is
        # no longer in the table, and everything that dereferences it raises.
        if self.active is not None and self.active not in self.backends:
            LOG.warning("active backend %s retired; serving 503", self.active)
            self.active = None

    # -- election ---------------------------------------------------------
    def elect(self):
        """Pick the healthy backend that will live the longest.

        Choosing by end time is what makes relay work without anybody
        orchestrating it: a freshly started job outlives the one it replaces,
        so the moment it passes /health it wins the election on its own.
        """
        candidates = [j for j, b in self.live().items() if b.healthy]
        if self.pending and self.pending[0] in candidates:
            LOG.info("successor %s is healthy", self.pending[0])
            self.forget_pending()
        winner = max(candidates,
                     key=lambda j: self.backends[j].end_time) if candidates else None
        if winner == self.active:
            return
        previous = self.active
        self.active = winner
        if winner is None:
            LOG.warning("no healthy backend; serving 503")
        else:
            # A backend just came up, so whatever made the last submit fail is
            # not worth backing off from any more. On the edge and not on every
            # sweep, on purpose: while a backend is already active and relay
            # submits keep being refused, the backoff is the thing that should
            # be slowing them down, and resetting it here every 5s would leave
            # it with nothing to do on the one path that submits most often.
            self.submit_succeeded()
            LOG.info("active backend -> %s (%s)", winner,
                     self.backends[winner].url)
            # Won the election back: whatever replaced it is gone or sicker, so
            # it is no longer a candidate for reclaim.
            self.superseded.discard(winner)
            if winner in self.draining:
                self.draining.pop(winner, None)
                LOG.info("cancelled drain of re-elected backend %s", winner)

        # Only a forward handover marks the predecessor. Falling back to an
        # older job after the active backend fails is reversible: the newer job
        # may merely be restarting and must remain eligible to win back routing.
        if winner is not None and previous and previous in self.backends:
            if (self.backends[winner].end_time
                    > self.backends[previous].end_time):
                self.superseded.add(previous)
                LOG.info("superseded %s; reclaim held until %s is stable",
                         previous, winner)
            else:
                LOG.warning("failed back from %s to older backend %s; keeping "
                            "%s available for recovery", previous, winner,
                            previous)


def fmt_time(ts):
    if not ts:
        return "unknown"
    return time.strftime("%H:%M:%S", time.localtime(ts))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def read_head(reader):
    """Read up to and including the blank line ending an HTTP head."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = await reader.read(8192)
        if not chunk:
            return None, b""
        buf += chunk
        if len(buf) > MAX_HEAD_BYTES:
            raise ValueError("head exceeds %d bytes" % MAX_HEAD_BYTES)
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def parse_request_head(head):
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        raise ValueError("malformed request line: %r" % lines[0])
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise ValueError("malformed header: %r" % line)
        headers.append((name.strip(), value.strip()))
    return parts[0], parts[1], headers


def header_value(headers, name):
    name = name.lower()
    for key, value in headers:
        if key.lower() == name:
            return value
    return None


def extract_key(headers):
    key = header_value(headers, "x-api-key")
    if key:
        return key.strip()
    auth = header_value(headers, "authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def build_response(status, reason, body, extra_headers=()):
    head = ["HTTP/1.1 %d %s" % (status, reason),
            "Content-Type: application/json",
            "Content-Length: %d" % len(body),
            "Connection: close"]
    head.extend(extra_headers)
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


def error_response(status, retry_after=None, message=None):
    kind, default = ERROR_BODIES[status]
    body = json.dumps({"type": "error",
                       "error": {"type": kind,
                                 "message": message or default}}).encode()
    extra = ["Retry-After: %d" % retry_after] if retry_after else []
    return build_response(status, REASONS[status], body, extra)


def json_response(payload, status=200):
    body = json.dumps(payload, indent=2).encode()
    return build_response(status, REASONS[status], body)


def clean_label(label, fallback):
    """Reduce a caller-supplied label to something safe in a path."""
    label = LABEL_UNSAFE.sub("-", (label or "").strip())[:32].strip("-._")
    return label or fallback


def parse_response_head(head):
    lines = head.decode("latin-1").split("\r\n")
    if not lines or not lines[0].startswith("HTTP/"):
        raise ValueError("malformed upstream status line")
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise ValueError("malformed upstream header: %r" % line)
        headers.append((name.strip(), value.strip()))
    return lines[0], headers


def rewrite_sse_head(status_line, headers):
    """Make downstream SSE framing independent from the upstream framing."""
    owned = {"connection", "content-length", "keep-alive", "trailer",
             "transfer-encoding"}
    lines = [status_line]
    lines.extend("%s: %s" % (name, value) for name, value in headers
                 if name.lower() not in owned)
    lines.extend(("Transfer-Encoding: chunked", "Connection: close"))
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def chunk_frame(payload):
    return b"%x\r\n%s\r\n" % (len(payload), payload)


class SseTracker:
    """Recognize terminal Anthropic events across arbitrary transport reads."""

    def __init__(self):
        self.buffer = bytearray()
        self.current_event = None
        self.saw_stop = False
        self.saw_error = False

    @property
    def terminal(self):
        return self.saw_stop or self.saw_error

    def feed(self, payload):
        self.buffer.extend(payload)
        while True:
            newline = self.buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self.buffer[:newline]).rstrip(b"\r")
            del self.buffer[:newline + 1]
            if not line:
                if self.current_event == b"message_stop":
                    self.saw_stop = True
                elif self.current_event == b"error":
                    self.saw_error = True
                self.current_event = None
                continue
            name, sep, value = line.partition(b":")
            if not sep or name != b"event":
                continue
            self.current_event = value.lstrip(b" ")


class BufferedUpstream:
    """Expose bytes already read with the response head before the socket."""

    def __init__(self, reader, initial):
        self.reader = reader
        self.buffer = bytearray(initial)

    async def read(self, size):
        if self.buffer:
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data
        return await self.reader.read(size)

    async def read_exact(self, size):
        parts = []
        remaining = size
        while remaining:
            data = await self.read(remaining)
            if not data:
                return b"".join(parts), False
            parts.append(data)
            remaining -= len(data)
        return b"".join(parts), True

    async def read_line(self):
        while True:
            newline = self.buffer.find(b"\r\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[:newline + 2]
                return line
            if len(self.buffer) > MAX_HEAD_BYTES:
                raise ValueError("upstream framing line is too long")
            data = await self.reader.read(8192)
            if not data:
                return None
            self.buffer.extend(data)


async def emit_sse_payload(writer, tracker, payload):
    if not payload:
        return
    tracker.feed(payload)
    writer.write(chunk_frame(payload))
    await writer.drain()


async def relay_chunked_sse(source, writer, tracker):
    while True:
        line = await source.read_line()
        if line is None:
            return False
        try:
            size = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            LOG.warning("invalid upstream chunk size: %r", line)
            return False
        if size < 0:
            return False
        if size == 0:
            # Consume trailers, but do not forward them: rewrite_sse_head removes
            # Trailer and the gateway owns the downstream terminal chunk.
            while True:
                trailer = await source.read_line()
                if trailer is None:
                    return False
                if not trailer:
                    return True

        remaining = size
        while remaining:
            payload = await source.read(min(RELAY_CHUNK, remaining))
            if not payload:
                return False
            remaining -= len(payload)
            await emit_sse_payload(writer, tracker, payload)
        ending, complete = await source.read_exact(2)
        if not complete or ending != b"\r\n":
            return False


async def relay_sized_sse(source, writer, tracker, size):
    remaining = size
    while remaining:
        payload = await source.read(min(RELAY_CHUNK, remaining))
        if not payload:
            return False
        remaining -= len(payload)
        await emit_sse_payload(writer, tracker, payload)
    return True


async def relay_close_delimited_sse(source, writer, tracker):
    while True:
        payload = await source.read(RELAY_CHUNK)
        if not payload:
            return True
        await emit_sse_payload(writer, tracker, payload)


# ---------------------------------------------------------------------------
# Request path
# ---------------------------------------------------------------------------
class Gateway:

    def __init__(self, fleet):
        self.fleet = fleet

    async def handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        started = time.time()
        try:
            head, rest = await read_head(reader)
            if head is None:
                return
            method, path, headers = parse_request_head(head)
        except (ValueError, ConnectionError) as exc:
            LOG.debug("bad request from %s: %s", peer, exc)
            await close(writer)
            return

        if path.startswith("/_gateway/"):
            try:
                await self.serve_introspection(method, path, headers, writer)
            except Exception:
                # These endpoints shell out to Slurm, so they have more ways to
                # fail than the proxy path does. An escape here would reach
                # asyncio's own handler, and the caller -- who has just asked
                # for GPUs to be released -- would get an empty reply and read
                # it as "the gateway is down".
                LOG.exception("%s %s failed", method, path)
                await respond(writer, error_response(
                    502, message="the gateway failed while handling this "
                                 "request; see its log"))
            return

        key = extract_key(headers) or "kf-anonymous"
        # Allowlist disabled on purpose. The check below rejected any key that
        # was not a line in the users file, and an OpenAI-native client sends
        # whatever OPENAI_API_KEY happens to be -- usually an sk-... string. It
        # would be turned away here, by the gateway, so the request left no
        # trace at all in the backend's server.log and looked like it had never
        # been sent. `key` is still resolved above: the access line and the
        # upstream user header keep attributing every request.
        # Re-enable by uncommenting; the users file is still loaded and watched,
        # and /_gateway/fleet still checks it.
        # if key not in self.fleet.users:
        #     LOG.info("401 %s %s user=%r", method, path, key)
        #     await respond(writer, error_response(401))
        #     return

        # Resolved once. The election loop may move `active` while this request
        # is in flight; everything below must keep talking about the same
        # backend, or the inflight count is incremented on one and decremented
        # on another. Read through .get(): this runs outside the try below, so
        # a lookup that raises here would leak the client socket.
        job_id = self.fleet.active
        backend = self.fleet.backends.get(job_id) if job_id else None
        if backend is None:
            # "Retry shortly" is the wrong advice when the gateway is idle by
            # request: nothing is coming unless somebody asks for it. Say which
            # of the two it is, in the body the client already prints.
            stopped = self.fleet.desired == "stopped" and not self.fleet.live()
            LOG.info("503 %s %s user=%s (%s)", method, path, key,
                     "stopped" if stopped else "no backend")
            message = ("no serving job is running; POST /_gateway/start_server "
                       "to start one") if stopped else None
            await respond(writer, error_response(503, retry_after=20,
                                                 message=message))
            return
        self.fleet.inflight[job_id] += 1
        status = "-"
        # Closing the client socket sits in its own finally so that no amount of
        # bookkeeping trouble above can leak the connection.
        try:
            try:
                status = await self.proxy(backend, method, path, headers, rest,
                                          reader, writer, key)
            except (ConnectionError, OSError) as exc:
                LOG.warning("upstream %s failed: %s", backend.url, exc)
                status = "502"
                await respond(writer, error_response(502))
            finally:
                self.fleet.inflight[job_id] -= 1
                LOG.info("%s %s %s user=%s backend=%s %.1fs", status, method,
                         path, key, job_id, time.time() - started)
        finally:
            await close(writer)

    async def serve_introspection(self, method, path, headers, writer):
        path, _, query = path.partition("?")
        if path == "/_gateway/health":
            if self.fleet.active is not None:
                # "ok" would be true and useless: a monitor keyed on it cannot
                # tell a job this gateway is relaying from one it merely found,
                # and the second kind goes down at its wall clock with no
                # successor. Both are serving; only one has an owner.
                status = ("ok" if self.fleet.desired == "running"
                          else "unmanaged")
            elif self.fleet.desired == "running":
                status = "no_backend"
            else:
                # Not a fault. Nobody has asked for a serving job, so a monitor
                # should not page on it -- which "no_backend" invites.
                status = "stopped"
            payload = {"status": status,
                       "deployment": self.fleet.name,
                       "active": self.fleet.active,
                       "desired": self.fleet.desired,
                       "pending": self.fleet.pending[0] if self.fleet.pending
                                  else None,
                       "uptime_s": round(time.time() - self.fleet.started)}
            await respond(writer, json_response(payload))
            return
        if path in ("/_gateway/start_server", "/_gateway/stop_server"):
            await self.serve_control(method, path, query, writer)
            return
        if path == "/_gateway/fleet":
            if extract_key(headers) not in self.fleet.users:
                await respond(writer, error_response(401))
                return
            now = time.time()
            payload = {
                "deployment": self.fleet.name,
                "active": self.fleet.active,
                "desired": self.fleet.desired,
                "pending_successor": self.fleet.pending[0] if self.fleet.pending
                                     else None,
                "submit_failures": self.fleet.submit_failures,
                "submit_retry_in_s": max(0, round(self.fleet.retry_after - now)),
                "backends": {
                    job_id: {
                        "url": b.url,
                        "healthy": b.healthy,
                        "healthy_for_s": round(now - b.healthy_since)
                                         if b.healthy_since else None,
                        "probe_timeouts": b.timeouts,
                        "state": b.state,
                        "ends_at": fmt_time(b.end_time),
                        "ends_in_s": round(b.end_time - now),
                        "last_beat_s": round(now - b.heartbeat, 1),
                        "inflight": self.fleet.inflight.get(job_id, 0),
                        "superseded": job_id in self.fleet.superseded,
                        "draining": job_id in self.fleet.draining,
                        "stopping": job_id in self.fleet.stopping,
                    }
                    for job_id, b in sorted(self.fleet.backends.items())
                },
            }
            await respond(writer, json_response(payload))
            return
        # A mistyped control path used to answer "backend refused the
        # connection", which sends whoever typed it looking at the serving job.
        await respond(writer, error_response(404))

    async def serve_control(self, method, path, query, writer):
        """start_server / stop_server: who owns the serving job right now.

        Unauthenticated, like /v1 -- see the note in `handle`. That makes the
        method the only thing between a mistyped GET, a link preview or a
        crawler and a released allocation, so anything but POST is refused
        before it reaches Slurm.
        """
        if method != "POST":
            LOG.info("405 %s %s", method, path)
            await respond(writer, error_response(405))
            return
        if self.fleet.args.no_relay:
            # --no-relay promises not to touch job lifecycles at all. Honouring
            # half of that promise would be worse than refusing.
            LOG.warning("409 %s: gateway runs with --no-relay", path)
            await respond(writer, error_response(409))
            return
        if path.endswith("/start_server"):
            label = urllib.parse.parse_qs(query).get("label", [""])[0]
            status, payload = await start_server(self.fleet,
                                                 clean_label(label, "start"))
        else:
            status, payload = await stop_server(self.fleet)
        await respond(writer, json_response(payload, status))

    async def proxy(self, backend, method, path, headers, rest, reader, writer,
                    user):
        up_reader, up_writer = await asyncio.open_connection(backend.host,
                                                             backend.port)
        try:
            up_writer.write(self.upstream_head(backend, method, path, headers,
                                               user))
            if rest:
                up_writer.write(rest)
            await up_writer.drain()

            # Nothing here parses the request body. The pump runs until the
            # client stops sending or the response finishes, so content-length
            # and chunked bodies both work without being understood.
            pump = asyncio.create_task(relay(reader, up_writer))
            try:
                return await self.relay_response(up_reader, writer)
            finally:
                pump.cancel()
        finally:
            await close(up_writer)

    def upstream_head(self, backend, method, path, headers, user):
        lines = ["%s %s HTTP/1.1" % (method, path),
                 "Host: %s:%d" % (backend.host, backend.port),
                 # Close framing gives non-SSE responses an unambiguous EOF.
                 # SSE responses are decoded and reframed below.
                 "Connection: close",
                 "Accept-Encoding: identity",
                 "X-Gateway-User: %s" % user]
        for name, value in headers:
            if name.lower() in STRIP_REQUEST_HEADERS:
                continue
            lines.append("%s: %s" % (name, value))
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")

    async def relay_response(self, up_reader, writer):
        head, rest = await read_head(up_reader)
        if head is None:
            raise ConnectionError("upstream closed before sending a response")
        status_line, headers = parse_response_head(head)
        parts = status_line.split(" ")
        if len(parts) < 2:
            raise ConnectionError("malformed upstream status line")
        status = parts[1]
        content_type = header_value(headers, "content-type") or ""
        content_encoding = header_value(headers, "content-encoding") or ""
        transfer_encoding = header_value(headers, "transfer-encoding") or ""
        encodings = [value.strip().lower()
                     for value in transfer_encoding.split(",") if value.strip()]
        is_sse = "text/event-stream" in content_type.lower()
        supported_transfer = not encodings or encodings == ["chunked"]

        # Reframing an encoding the gateway does not decode would mix an
        # unencoded injected event into that stream. Accept-Encoding: identity
        # prevents content encoding for the normal server; retain raw relay as a
        # safe fallback for encoded bodies or unknown transfer codings.
        if (not is_sse or content_encoding.lower() not in ("", "identity")
                or not supported_transfer):
            writer.write(head + b"\r\n\r\n" + rest)
            await writer.drain()
            while True:
                chunk = await up_reader.read(RELAY_CHUNK)
                if not chunk:
                    return status
                writer.write(chunk)
                await writer.drain()

        writer.write(rewrite_sse_head(status_line, headers))
        await writer.drain()
        source = BufferedUpstream(up_reader, rest)
        tracker = SseTracker()
        if "chunked" in encodings:
            clean_end = await relay_chunked_sse(source, writer, tracker)
        else:
            content_length = header_value(headers, "content-length")
            if content_length is None:
                clean_end = await relay_close_delimited_sse(
                    source, writer, tracker)
            else:
                try:
                    length = int(content_length)
                except ValueError:
                    length = -1
                if length < 0:
                    clean_end = False
                else:
                    clean_end = await relay_sized_sse(source, writer, tracker,
                                                      length)

        if not tracker.terminal:
            ending = "clean end" if clean_end else "truncated upstream framing"
            LOG.warning("stream reached %s without message_stop or error; "
                        "injecting error", ending)
            await emit_sse_payload(writer, tracker, SSE_ROTATED)
            status += "!"
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        return status


async def relay(reader, writer):
    try:
        while True:
            chunk = await reader.read(RELAY_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, OSError, asyncio.CancelledError):
        pass


async def respond(writer, payload):
    try:
        writer.write(payload)
        await writer.drain()
    except (ConnectionError, OSError):
        pass
    await close(writer)


async def close(writer):
    try:
        writer.close()
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------
async def probe(backend, timeout):
    """GET /health, classified into three outcomes rather than a boolean.

    "dead" and "timeout" look the same to a boolean probe but mean opposite
    things. A refused connection or an unresolvable host says the process is
    gone -- unambiguous, act at once. A timeout usually says the server is too
    busy to answer a health check, and a server that busy is normally still
    generating tokens fine; taking it out of rotation would turn "slow" into
    "503" with nowhere better to send the traffic.

    A non-200 is not ambiguous either: /health only fails when the engine
    reports itself broken, which trtllm-serve follows with a shutdown.
    """
    writer = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(backend.host, backend.port), timeout)
        writer.write(b"GET /health HTTP/1.1\r\nHost: %s\r\n"
                     b"Connection: close\r\n\r\n"
                     % backend.host.encode("latin-1"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout)
        return "ok" if b" 200 " in line else "dead"
    except asyncio.TimeoutError:
        return "timeout"
    except (ConnectionError, OSError):
        return "dead"
    finally:
        if writer is not None:
            await close(writer)


def apply_probe(backend, result, unhealthy_after):
    if result == "ok":
        backend.timeouts = 0
        if not backend.healthy:
            backend.healthy = True
            backend.healthy_since = time.time()
            LOG.info("backend %s healthy", backend.job_id)
        return
    if result == "dead":
        if backend.healthy:
            LOG.warning("backend %s unreachable; dropping it now",
                        backend.job_id)
        backend.timeouts = 0
        backend.healthy = False
        backend.healthy_since = 0.0
        return
    backend.timeouts += 1
    if backend.healthy and backend.timeouts >= unhealthy_after:
        LOG.warning("backend %s timed out %d times in a row; marking unhealthy",
                    backend.job_id, backend.timeouts)
        backend.healthy = False
        backend.healthy_since = 0.0
    elif backend.healthy:
        LOG.info("backend %s health probe timed out (%d/%d)", backend.job_id,
                 backend.timeouts, unhealthy_after)


async def discovery_loop(fleet):
    while True:
        try:
            fleet.reload_users()
            fleet.discover()
        except Exception:
            LOG.exception("discovery failed")
        await asyncio.sleep(fleet.args.discover_interval)


async def health_loop(fleet):
    while True:
        try:
            backends = list(fleet.backends.values())
            if backends:
                results = await asyncio.gather(
                    *[probe(b, fleet.args.probe_timeout) for b in backends],
                    return_exceptions=True)
                for backend, result in zip(backends, results):
                    if not isinstance(result, str):
                        result = "timeout"
                    apply_probe(backend, result, fleet.args.unhealthy_after)
            fleet.elect()
        except Exception:
            LOG.exception("health loop failed")
        await asyncio.sleep(fleet.args.health_interval)


async def run_serve_sh(fleet, *serve_args):
    """Run serve.sh. Returns (None, message) if it could not be run at all.

    Guarded the same way run_slurm_command is, and for a sharper reason now
    that release_job calls this from a request handler: an unguarded exec
    failure -- serve.sh moved, a transient EAGAIN -- would escape through
    stop_server, leave the client with an empty reply, and strand every job it
    had already marked as being torn down. Every caller compares against 0, so
    None fails closed.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            fleet.args.serve_sh, *serve_args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    except OSError as exc:
        LOG.error("cannot run %s: %s", fleet.args.serve_sh, exc)
        return None, str(exc)
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


async def run_slurm_command(*command):
    try:
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
    except OSError as exc:
        LOG.error("cannot run %s: %s", command[0], exc)
        return None, ""
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


async def slurm_job_status(job_id):
    """Return (state, reason), ("GONE", ""), or None on query failure."""
    code, out = await run_slurm_command("squeue", "-h", "-j", job_id, "-o",
                                        "%T|%r")
    if code is not None and code != 0 and "invalid job id" in out.lower():
        return "GONE", ""
    if code is None or code != 0:
        LOG.warning("cannot query successor %s: %s", job_id, out)
        return None
    if not out:
        return "GONE", ""
    state, _, reason = out.splitlines()[0].partition("|")
    return state.strip().upper(), reason.strip()


async def submit_successor(fleet, now, label):
    code, out = await run_serve_sh(fleet, "submit", "--yaml", fleet.args.yaml,
                                   "--label", label)
    match = re.search(r"Submitted batch job (\d+)", out)
    if code == 0 and match:
        fleet.track_pending(match.group(1), now)
        fleet.submit_succeeded()
        LOG.info("successor submitted: job %s", match.group(1))
        return True
    LOG.error("submit failed (rc=%s, attempt %d): %s; next attempt in %ds",
              code, fleet.submit_failures + 1, out, fleet.submit_failed(now))
    return False


async def wait_for_release(job_id, timeout):
    """Watch Slurm until the job stops holding nodes. None if it cannot tell."""
    deadline = time.time() + timeout
    while True:
        status = await slurm_job_status(job_id)
        if status is None:
            return None
        if status[0] not in HOLDING_STATES:
            return True
        if time.time() >= deadline:
            return False
        await asyncio.sleep(RELEASE_POLL_INTERVAL)


async def release_job(fleet, job_id, run_dir):
    """Stop one serving job. Returns "released", "held" or "failed".

    `quit` is the clean path: the controller stops the server, writes a final
    state and exits, which under sbatch ends the job and returns the nodes. It
    needs a live control directory, so a job that never got that far, or whose
    controller is already gone, falls back to scancel.

    A filed `quit` is not a released allocation, so this waits for Slurm to
    agree. What it will not do is escalate a `quit` that worked into a scancel:
    a controller started by hand inside somebody's `salloc` exits exactly as
    asked and leaves the allocation behind, because the allocation is that
    person's shell. Reporting "held" sends the operator to `scancel` with the
    job id; guessing would take their session down with the server.
    """
    if run_dir:
        code, out = await run_serve_sh(fleet, "quit", run_dir)
        if code == 0:
            released = await wait_for_release(job_id,
                                              fleet.args.release_timeout)
            if released:
                LOG.info("released %s via quit (%s)", job_id, run_dir)
                return "released"
            if released is None:
                LOG.warning("quit %s was filed but squeue would not confirm "
                            "the release; treating it as done", job_id)
                return "released"
            LOG.warning("%s stopped serving but still holds its allocation "
                        "%ds later; leaving it to be cancelled by hand",
                        job_id, fleet.args.release_timeout)
            return "held"
        LOG.warning("quit %s failed (rc=%s): %s; falling back to scancel",
                    job_id, code, out)
    code, out = await run_slurm_command("scancel", job_id)
    if code == 0:
        LOG.info("released %s via scancel", job_id)
        return "released"
    LOG.error("scancel %s failed (rc=%s): %s", job_id, code, out)
    return "failed"


async def start_server(fleet, label):
    """Hand the job lifecycle to the supervisor and get a serving job coming.

    Idempotent on purpose: this is what a person retries when they are not sure
    whether the first call landed. A serving job that already exists is adopted
    rather than duplicated -- including one somebody submitted by hand, and
    including the one still loading weights from the previous call.
    """
    async with fleet.control:
        was = fleet.desired
        fleet.desired = "running"
        # An explicit start is also an explicit "try again now": whoever asked
        # has usually just fixed whatever sbatch was refusing.
        fleet.submit_succeeded()

        live = fleet.live()
        if live or fleet.pending:
            # Name the job this actually adopts: the one still loading if there
            # is one, else the one taking traffic, else the one that will win
            # the next election.
            if fleet.pending:
                job_id = fleet.pending[0]
            elif fleet.active in live:
                job_id = fleet.active
            else:
                job_id = max(live, key=lambda j: live[j].end_time)
            LOG.info("start_server: adopting job %s (was %s)", job_id, was)
            return 200, {
                "action": "adopted",
                "desired": "running",
                "job_id": job_id,
                "active": fleet.active,
                "message": "a serving job is already present; the gateway now "
                           "relays and reclaims it",
            }

        LOG.info("start_server: submitting a serving job (was %s)", was)
        if await submit_successor(fleet, time.time(), label):
            return 200, {
                "action": "submitted",
                "desired": "running",
                "job_id": fleet.pending[0],
                "active": None,
                "message": "queued; poll /_gateway/health until status is ok. "
                           "It waits for nodes and then loads weights, so "
                           "minutes to hours depending on the queue",
            }
        # The supervisor keeps trying on its own, so this is a report, not a
        # dead end -- but a script that fired it should be able to tell.
        return 503, {
            "action": "submit_failed",
            "desired": "running",
            "job_id": None,
            "active": None,
            "message": "sbatch refused the job; the gateway will retry in %ds. "
                       "Its log has the sbatch output"
                       % max(0, round(fleet.retry_after - time.time())),
        }


async def stop_server(fleet):
    """Release every serving job and leave the gateway as the only process."""
    async with fleet.control:
        was = fleet.desired
        fleet.desired = "stopped"
        fleet.submit_succeeded()

        targets = []
        # A submitted job that has not registered yet has no run directory to
        # drive, so scancel is the only handle on it.
        if fleet.pending:
            job_id = fleet.pending[0]
            if job_id not in fleet.backends:
                targets.append((job_id, ""))
            fleet.forget_pending()
        for job_id, backend in sorted(fleet.backends.items()):
            targets.append((job_id, backend.run_dir))

        # Every bit of state below is settled before the first await, so that
        # the discovery and health loops -- which run between those awaits and
        # would otherwise re-elect a backend that is on its way out -- see a
        # consistent picture from the moment this call starts tearing down.
        inflight = sum(fleet.inflight.get(job_id, 0) for job_id, _ in targets)
        fleet.stopping.update(job_id for job_id, _ in targets)
        fleet.draining.clear()
        fleet.superseded.clear()
        if fleet.active in fleet.stopping:
            # Stop routing first. These servers are about to go down mid-stream
            # and a clean 503 is a better answer than a truncated response.
            fleet.active = None

        # Concurrently, because each one waits on Slurm for up to
        # RELEASE_VERIFY_SECONDS and a deployment with several backends would
        # otherwise outlast the caller's HTTP timeout.
        outcomes = await asyncio.gather(
            *[release_job(fleet, job_id, run_dir) for job_id, run_dir in targets],
            return_exceptions=True)

        released, held, failed = [], [], []
        for (job_id, _), outcome in zip(targets, outcomes):
            if isinstance(outcome, BaseException):
                LOG.error("releasing %s raised: %r", job_id, outcome)
                outcome = "failed"
            {"released": released, "held": held}.get(outcome, failed).append(job_id)

        # A job nothing could be done to is still serving, so put it back in
        # rotation rather than leaving it latched out: `stopping` means "on its
        # way out", and this one is not. Otherwise it would be unroutable for
        # the rest of its wall clock AND invisible to the next start_server,
        # which would then submit a second allocation alongside it.
        for job_id in failed:
            fleet.stopping.discard(job_id)
        if fleet.active is None:
            fleet.elect()

        LOG.info("stop_server: released %s (was %s, interrupted %d request(s))"
                 "%s%s", ", ".join(released) or "nothing", was, inflight,
                 "; STILL ALLOCATED: %s" % ", ".join(held) if held else "",
                 "; FAILED on %s" % ", ".join(failed) if failed else "")

        if failed:
            # The one thing this endpoint must not do is leave a GPU job nobody
            # knows about, so name it.
            message = ("could not stop %s; check it with squeue and scancel it "
                       "by hand" % ", ".join(failed))
        elif held:
            # Normal for a controller started by hand inside an salloc: it did
            # what it was told, and the allocation is not its to give back.
            message = ("stopped, but %s still holds its allocation -- scancel "
                       "it if you want the nodes back" % ", ".join(held))
        else:
            message = "POST /_gateway/start_server to bring a serving job back"
        return (502 if failed else 200), {
            "action": "stopped",
            "desired": "stopped",
            "released": released,
            "still_allocated": held,
            "failed": failed,
            "interrupted_requests": inflight,
            "message": message,
        }


async def supervise_pending(fleet, now):
    """Keep a submitted successor tracked until it is actually healthy."""
    if not fleet.pending:
        return
    job_id, submitted_at = fleet.pending
    backend = fleet.backends.get(job_id)
    if backend is not None:
        failed_attempt = (" exited with status " in backend.state
                          or backend.state.startswith("stopped;"))
        if not backend.healthy and failed_attempt:
            LOG.warning("successor %s failed to start; restarting its retained "
                        "allocation", job_id)
            code, out = await run_serve_sh(fleet, "restart", backend.run_dir)
            if code == 0:
                fleet.track_pending(job_id, now)
            else:
                LOG.error("restart %s failed (rc=%s): %s; retrying in %ds",
                          job_id, code, out, fleet.submit_failed(now))
                cancel_code, cancel_out = await run_slurm_command(
                    "scancel", job_id)
                if cancel_code == 0:
                    fleet.forget_pending()
                else:
                    LOG.error("scancel %s failed (rc=%s): %s", job_id,
                              cancel_code, cancel_out)
        return

    # sbatch can take a moment to publish a new job into squeue. Do not mistake
    # that visibility gap for an immediate terminal failure.
    if now - submitted_at < PENDING_VISIBILITY_GRACE:
        return
    status = await slurm_job_status(job_id)
    if status is None:
        return
    state, reason = status
    terminal = {
        "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
        "OUT_OF_MEMORY", "TIMEOUT"
    }
    if state == "GONE" or state in terminal:
        # `pending` is cleared the moment a successor passes /health, so
        # reaching a terminal state while still pending means it never served.
        # Counted like a refused sbatch: a node that boot-fails in seconds
        # would otherwise be resubmitted every sweep forever. Any backend
        # coming up clears the count, so a cluster that fails intermittently
        # never accumulates one.
        LOG.warning("successor %s is no longer runnable (%s); retrying in %ds",
                    job_id, state, fleet.submit_failed(now))
        fleet.forget_pending()
        return

    # Slurm reports RUNNING as soon as it hands the node over, which is before
    # the prolog finishes and therefore before the batch script exists to
    # register anything. Restart the grace clock instead of reading that as a
    # launcher failure: node setup here regularly outlasts
    # PENDING_VISIBILITY_GRACE, and cancelling mid-prolog throws away a
    # successor that was about to come up -- repeatedly, since every retry
    # meets the same prolog. A prolog cannot stall forever, because Slurm caps
    # it with PrologEpilogTimeout and fails the job into a terminal state the
    # branch above already handles.
    if state == "RUNNING" and "prolog" in reason.lower():
        LOG.info("successor %s is still in prolog; deferring its failure check",
                 job_id)
        fleet.pending = (job_id, now)
        return

    # Slurm reports RUNNING the moment the batch script starts, and serve.sh
    # needs another 10-20s to validate the allocation and write its first fleet
    # record. Measuring that gap from submit time hides it entirely while the
    # queue is short and makes it certain once the queue wait passes the grace
    # period -- which is exactly the cold start start_server now performs. A
    # job that queued for twenty minutes would be scancelled seconds after
    # Slurm handed the nodes over, and the retry would queue behind everyone
    # else and meet the same fate. Give the transition its own grace period.
    if state == "RUNNING" and not fleet.pending_running:
        LOG.info("successor %s started; %ds to register before it counts as a "
                 "launcher failure", job_id, PENDING_VISIBILITY_GRACE)
        fleet.pending = (job_id, now)
        fleet.pending_running = True
        return

    # A RUNNING job reaches cmd_run and registers before it starts loading the
    # model. If it remains invisible here, the launcher failed before that
    # point. Held jobs cannot make progress either. Cancel before retrying so a
    # delayed job cannot later appear as a duplicate successor.
    should_cancel = state == "RUNNING" or "held" in reason.lower()
    if should_cancel:
        LOG.warning("successor %s is %s without registering (%s); cancelling "
                    "and retrying in %ds", job_id, state, reason or "no reason",
                    fleet.submit_failed(now))
        code, out = await run_slurm_command("scancel", job_id)
        if code == 0:
            fleet.forget_pending()
        else:
            LOG.error("scancel %s failed (rc=%s): %s", job_id, code, out)


async def supervisor_loop(fleet):
    while True:
        try:
            # Under the same lock as start_server and stop_server, and the
            # state is read inside it: a stop landing halfway through a sweep
            # must not have its teardown followed by that sweep's submit.
            async with fleet.control:
                if fleet.desired == "running":
                    await supervise(fleet)
                elif fleet.active is not None:
                    # Serving a job this gateway did not start and will not
                    # relay. Repeated every sweep on purpose: the consequence
                    # arrives hours later, at that job's wall clock, and by
                    # then this log is the only thing that saw it coming.
                    LOG.warning("backend %s is serving but unmanaged: no "
                                "successor will be submitted before it ends. "
                                "POST /_gateway/start_server to adopt it",
                                fleet.active)
        except Exception:
            LOG.exception("supervisor failed")
        await asyncio.sleep(fleet.args.supervisor_interval)


async def supervise(fleet):
    """One sweep of job lifecycle. Only ever reached while `desired` is running."""
    now = time.time()
    await supervise_pending(fleet, now)

    # Jobs stop_server released are excluded from every decision below: they
    # keep heartbeating on their way out, and each one of these would otherwise
    # read them as a serving job that needs relaying or reclaiming.
    live = fleet.live()

    # Relay: submit the next job early enough that it finishes loading weights
    # before this one hits the wall clock.
    backend = live.get(fleet.active) if fleet.active else None
    if backend is not None and not fleet.args.no_relay:
        remaining = backend.end_time - now
        # A submitted/loading successor is represented by `pending`. An older
        # job that took traffic and then failed remains discoverable so it can
        # recover, but must not block a replacement successor forever.
        successors = [
            j for j, candidate in live.items()
            if j != fleet.active and candidate.healthy
        ]
        if backend.end_time <= 0:
            # Registration could not determine the wall clock. Routing still
            # works; relaying on `0 - now` would read as "already expired" and
            # submit a job every single sweep.
            LOG.warning("backend %s has no end time; relay disabled for it",
                        fleet.active)
        elif (remaining < fleet.args.lead_time and not successors
              and not fleet.pending and now >= fleet.retry_after):
            LOG.info("%s ends in %ds; submitting successor",
                     fleet.active, int(remaining))
            await submit_successor(fleet, now, "relay")
    elif (not fleet.args.no_relay and not live and not fleet.pending
          and now >= fleet.retry_after):
        # Somebody asked for a serving job and there is none. Either
        # start_server's own submit failed, or the fleet was lost afterwards --
        # recovery and cold start want the same thing, and neither can depend
        # on a live active backend, because a cancelled pending job may vanish
        # just as its predecessor reaches the wall clock. What used to gate
        # this is now `desired`, which the supervisor loop checks before
        # calling in here.
        LOG.warning("no serving job is up; submitting one")
        await submit_successor(fleet, now, "recovery")

    # Promote superseded backends to draining, but only once the successor has
    # held up. Handing over routing is reversible and happens the instant the
    # successor is healthy; releasing the predecessor's allocation is not, so it
    # waits. Without this, a successor that passes one probe and then dies takes
    # the predecessor down with it and leaves nothing serving until the next job
    # finishes loading.
    winner = fleet.backends.get(fleet.active) if fleet.active else None
    if winner is not None and fleet.superseded:
        stable_for = now - winner.healthy_since if winner.healthy_since else 0
        if winner.healthy and stable_for >= fleet.args.promote_after:
            for job_id in sorted(fleet.superseded):
                fleet.superseded.discard(job_id)
                if job_id == fleet.active:
                    LOG.warning("refusing to drain active backend %s", job_id)
                    continue
                backend = fleet.backends.get(job_id)
                if backend is None:
                    continue
                deadline = backend.end_time - 60
                fleet.draining[job_id] = deadline
                LOG.info("draining %s (%s stable %ds, inflight=%d, reclaim "
                         "by %s)", job_id, fleet.active, int(stable_for),
                         fleet.inflight.get(job_id, 0), fmt_time(deadline))

    # Reclaim: the drained job is already past being useful, and its allocation
    # is worth releasing a little early. Skipped under --no-relay, which
    # promises not to touch job lifecycles at all -- submitting and reclaiming
    # are two halves of the same authority.
    if fleet.args.no_relay:
        return
    for job_id, deadline in list(fleet.draining.items()):
        if job_id == fleet.active:
            LOG.warning("cancelled stale drain of active backend %s", job_id)
            fleet.draining.pop(job_id, None)
            continue
        backend = fleet.backends.get(job_id)
        if backend is None:
            fleet.draining.pop(job_id, None)
            continue
        inflight = fleet.inflight.get(job_id, 0)
        if inflight and now <= deadline:
            continue
        why = "drained" if not inflight else "deadline"
        LOG.info("reclaiming %s (%s, inflight=%d)", job_id, why, inflight)
        code, out = await run_serve_sh(fleet, "quit", backend.run_dir)
        if code == 0:
            # Same reason stop_server does it: the controller polls its control
            # directory every two seconds, so this job answers /health for a
            # moment longer. Without this it is still a candidate, and a
            # successor that fails inside that window hands routing straight
            # back to the job that was just told to exit. Only on success --
            # a quit that never landed must stay routable, because that job is
            # still serving.
            fleet.stopping.add(job_id)
        else:
            LOG.error("quit %s failed (rc=%s): %s", job_id, code, out)
        fleet.draining.pop(job_id, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="stable front door for the Anthropic-compatibility servers")
    parser.add_argument("--fleet-dir", required=True,
                        help="directory the serving jobs register into")
    parser.add_argument("--users", required=True,
                        help="allowlist, one username per line")
    parser.add_argument("--yaml", default="",
                        help="deployment YAML the supervisor resubmits")
    parser.add_argument("--serve-sh", default="",
                        help="path to serve.sh (defaults next to this file)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8333)
    parser.add_argument("--lead-time", type=int, default=2700,
                        help="seconds before the wall clock to submit the "
                             "successor (default 45min)")
    parser.add_argument("--stale-after", type=int, default=30,
                        help="drop a backend after this long without a "
                             "heartbeat")
    parser.add_argument("--discover-interval", type=float, default=5.0)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--supervisor-interval", type=float, default=30.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument("--unhealthy-after", type=int, default=20,
                        help="consecutive probe timeouts before a backend is "
                             "taken out of rotation; a refused connection is "
                             "acted on immediately regardless")
    parser.add_argument("--release-timeout", type=float,
                        default=RELEASE_VERIFY_SECONDS,
                        help="how long stop_server waits for Slurm to confirm "
                             "a job has given its nodes back before reporting "
                             "it as still allocated")
    parser.add_argument("--promote-after", type=float, default=180.0,
                        help="seconds a successor must stay healthy before its "
                             "predecessor may be reclaimed")
    parser.add_argument("--no-relay", action="store_true",
                        help="proxy only; never submit a successor and never "
                             "reclaim a drained job")
    args = parser.parse_args(argv)
    if not args.serve_sh:
        args.serve_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "serve.sh")
    if not args.no_relay and not args.yaml:
        parser.error("--yaml is required unless --no-relay is given")
    return args


async def main_async(args):
    fleet = Fleet(args)
    os.makedirs(args.fleet_dir, exist_ok=True)
    fleet.reload_users()
    # Says it once per start, where an operator will see it: the source comment
    # in Gateway.handle is not somewhere anyone looks before restarting this.
    LOG.warning("user allowlist is DISABLED (%s is loaded but not enforced); "
                "every caller on this network can use this gateway", args.users)
    fleet.discover()

    gateway = Gateway(fleet)
    server = await asyncio.start_server(gateway.handle, args.host, args.port)
    LOG.info("listening on %s:%d", args.host, args.port)
    LOG.info("fleet dir: %s", args.fleet_dir)
    LOG.info("relay: %s", "off" if args.no_relay
             else "lead time %ds from %s" % (args.lead_time, args.yaml))
    if args.no_relay:
        LOG.info("job control: off; start_server and stop_server answer 409")
    else:
        LOG.info("job control: idle. No serving job is submitted until "
                 "POST /_gateway/start_server; stop_server releases them all")

    async with server:
        await asyncio.gather(discovery_loop(fleet),
                             health_loop(fleet),
                             supervisor_loop(fleet))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")
    args = parse_args(sys.argv[1:])
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        LOG.info("interrupted")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
