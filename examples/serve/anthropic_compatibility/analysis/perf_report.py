#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the four-section performance report for one serving run.

Reads whatever a run directory happens to hold -- the Anthropic audit log, the
per-request perf metrics the controller drained, the attention-DP routing
trace, and the per-rank iteration logs -- and emits four CSVs plus a single
self-contained HTML report.

    python3 analysis/perf_report.py <attempt_dir>… [--out LABEL]

Every report lands in one place -- `REPORTS_ROOT/<label>`, defaulting to the
run's own name -- and never beside the run it describes. An attempt directory
is raw capture: it gets re-run, rsynced and cleaned up, while a report is the
thing you link to. Collecting them also gives `build_index.py` a directory to
rank, which it can only do over siblings, since the index links relatively.

Sections, and the CSV each one leaves behind for re-analysis:

    1  requests.csv    one row per request
    2  sessions.csv    one row per Claude Code session
    3  ctx_iters.csv   one row per (prefill instance, iteration)
       ctx_rank_iters.csv  the same, split per rank -- what the pooled
                       rows average away (hit rate, KV util, ctx tokens)
                       Iteration-grain files are gzipped past 50k rows and
                       gain a .gz suffix; pandas reads them unchanged.
    4  summary.csv     one row for the whole run

Sources are optional: a run missing its audit log still gets sections 3 and 4.
Missing values stay empty rather than becoming zero, because a zero cache-hit
rate and an unmeasured one lead to opposite conclusions.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import gzip
import io
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Iteration-log line, e.g.
#   iter = 1, global_rank = 2, rank = 2, num_scheduled_requests = 1, ...
_ITER_LINE = re.compile(r"\biter = \d+, global_rank = ")
_KV_PAIR = re.compile(r"(\w+) = ")
# [batchmgr][RANK 0] Max KV cache blocks per sequence: 16384 [window size=...],
# tokens per block=64, primary blocks=36506, secondary blocks=13359, ...
_KV_CAPACITY = re.compile(
    r"\[batchmgr\]\[RANK (\d+)\].*tokens per block=(\d+), primary blocks=(\d+)")
# The v2 manager's startup banner. Its presence is the only way to recognise
# a v2 run whose logs predate the cross-tier counters, and on those runs
# alloc_total - alloc_new is a zero that means nothing rather than no
# evictions -- see _kv_deltas.
_KV_V2 = re.compile(r"KV cache manager v2")

HIT_RATE_FLOOR = 0.90  # below this a request is called out for investigation

# Mirrors KVCacheAwareADPRouter's default `match_rate_threshold`. Below this
# ratio the router forces every rank's match to 0, so all scores tie and the
# decision falls through to the tie-breaks -- which is what makes a routing
# reason readable at all. Not read from the worker config: a deployment that
# overrides it will mislabel ties, so the reason table names the assumption.
ROUTER_MATCH_RATE_THRESHOLD = 0.1

# Prefix reuse is block-aligned, so a prompt's trailing partial block never
# matches. One block is therefore the only shortfall that is not a loss.
# Worth keeping conservative: it is the line between "the cache is fine" and
# "the cache dropped something", and a percentage there hides partial evictions
# in exact proportion to how long the prompt is.
BLOCK_TOLERANCE = 128

# Where every report goes: `_reports` beside this checkout, not beside the runs
# it describes. The trace root holds raw capture that gets rsynced and cleaned
# up, and it is read-only to some callers; a report is the thing you link to and
# open, so it lives where the editor already has the tree loaded. `.gitignore`
# keeps it out of `git status`. A different trace root (computelab has its own)
# sets PERF_REPORTS_DIR; build_index.py reads the same variable, and the two
# must agree or the index misses reports.
REPORTS_ROOT = Path(os.environ.get(
    "PERF_REPORTS_DIR",
    "/lustre/fsw/portfolios/coreai/users/serli/workspace/TensorRT-LLM"
    "/examples/serve/anthropic_compatibility/_reports"))

_ATTEMPT_DIR = re.compile(r"attempt-\d+")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _f_list(value: Any) -> list[float] | None:
    """A slash-joined per-pool-group vector, as the worker writes it.

    Slash rather than a list literal because the iteration log is parsed as
    ``key = value, `` pairs and a ``[1, 2, 3]`` would split across three of
    them. None when absent, never an empty list: a run written before the
    per-group fields existed must not read as a run with zero pool groups.
    """
    if value in (None, ""):
        return None
    out = [_f(part) for part in str(value).split("/")]
    return None if any(v is None for v in out) else out


def _parse_iso(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_bound(value: str | None) -> float | None:
    """One end of the analysis window, as epoch seconds.

    A naive stamp is read as *local* time, which is the iteration logs'
    convention (`timestamp = 2026-08-21 16:59:23`); the audit log writes UTC
    with an explicit offset and is unambiguous either way. Pass the offset when
    the two machines disagree -- the report prints the resolved window in both
    forms so a mistake is visible rather than silently shifting every figure.
    """
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(f"cannot read {value!r} as an ISO-8601 instant")
    return (stamp if stamp.tzinfo else stamp.astimezone()).timestamp()


def _inside(value: float | None, window: tuple[float | None, float | None]) -> bool:
    """Whether an epoch stamp falls in the window, with no window meaning yes.

    An unplaceable row -- no stamp at all -- is kept when nothing is being
    filtered and dropped when something is, because a row that cannot be dated
    cannot be claimed to belong to the window. ``_apply_window`` counts those
    separately so the report can say how many were lost that way.
    """
    since, until = window
    if since is None and until is None:
        return True
    if value is None:
        return False
    return ((since is None or value >= since)
            and (until is None or value <= until))


def _pct(values: Iterable[float | None], q: float) -> float | None:
    kept = sorted(v for v in values if v is not None)
    if not kept:
        return None
    return kept[min(len(kept) - 1, int(len(kept) * q))]


def _stats(values: Iterable[float | None]) -> dict[str, float | None]:
    kept = [v for v in values if v is not None]
    if not kept:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None}
    return {
        "n": len(kept),
        "mean": statistics.fmean(kept),
        "p50": _pct(kept, 0.50),
        "p90": _pct(kept, 0.90),
        "p99": _pct(kept, 0.99),
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Total wall time covered by at least one interval.

    Summing per-request durations would double-count concurrent requests, so
    the server-busy figure has to be a union rather than a sum.
    """
    ordered = sorted(i for i in intervals if i[0] is not None and i[1] is not None)
    total = 0.0
    cur_start = cur_end = None
    for start, end in ordered:
        if end < start:
            continue
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


# --------------------------------------------------------------------------
# source discovery
# --------------------------------------------------------------------------
class Run:
    """Everything one attempt directory offers, with the gaps named."""

    def __init__(self, attempt_dir: Path):
        self.dir = attempt_dir
        self.audit = attempt_dir / "anthropic_audit.jsonl"
        # disagg_config's perf_metrics_output_dir sends these to a
        # ``perf_metrics/`` subdirectory on newer runs; older ones drop them
        # beside the logs. Both layouts are in the wild, so take either.
        self.perf = sorted(attempt_dir.glob("perf_metrics-*.jsonl")) + sorted(
            attempt_dir.glob("perf_metrics/perf_metrics-*.jsonl"))
        self.route = sorted(attempt_dir.glob("adp_route_trace*.jsonl"))
        # Client ids seen under more than one worker; set by build_requests.
        # Non-zero means those requests carry no routing columns rather than
        # another worker's, which is the tradeoff that keeps the join honest.
        self.route_client_ambiguous = 0
        # Worker logs. Disaggregated runs name them by role; aggregated runs
        # write a single server.log.
        self.iter_logs = sorted(attempt_dir.glob("ctx-*.log")) + sorted(
            attempt_dir.glob("gen-*.log"))
        self.disagg = bool(self.iter_logs)
        if not self.iter_logs and (attempt_dir / "server.log").exists():
            self.iter_logs = [attempt_dir / "server.log"]
        self.name = attempt_dir.parent.name
        self.notes: list[str] = []

    def prefill_logs(self) -> list[Path]:
        """Logs whose iterations are prefill work.

        Disaggregated: the context workers. Aggregated: the one server, whose
        iterations mix prefill and decode -- section 3 says so rather than
        pretending the numbers mean the same thing.
        """
        ctx = [p for p in self.iter_logs if p.name.startswith("ctx-")]
        return ctx or self.iter_logs

    def decode_logs(self) -> list[Path]:
        """Generation workers. Empty on an aggregated run, where decode shares
        the one loop with prefill and cannot be separated by file."""
        return [p for p in self.iter_logs if p.name.startswith("gen-")]

    def worker_name(self, path: Path) -> str:
        return path.stem

    def kv_manager_v2(self) -> bool:
        """Whether the workers ran the v2 KV cache manager.

        Read from the startup banner rather than inferred from the cross-tier
        counters, because those counters only exist on logs written after they
        were added: a v2 run from before that would otherwise keep reporting
        ``alloc_total - alloc_new`` as a measured zero eviction count, which is
        the one number on this report that has been actively misleading.

        Only the head of one log is scanned. The banner is printed during
        engine construction, before any iteration line, and these files reach
        several gigabytes.
        """
        cached = getattr(self, "_kv_v2", None)
        if cached is None:
            cached = False
            for path in (self.prefill_logs() + self.decode_logs())[:1]:
                with path.open(encoding="latin-1", errors="replace") as handle:
                    for line in handle:
                        if _KV_V2.search(line):
                            cached = True
                            break
                        if _ITER_LINE.search(line):
                            break
            self._kv_v2 = cached
        return cached


def parse_iter_log(path: Path) -> list[dict]:
    """One row per (rank, iteration) from a worker's stdout log.

    The log is `key = value, ` pairs ending in a Python dict literal, and the
    file can carry stray control bytes from the launcher's tee, so it is read
    as latin-1 and matched loosely.

    One log holds more than one engine lifetime. Sizing the KV pool builds a
    throwaway engine first, runs it, measures, and tears it down before the
    real one starts -- and each gets its own ``PyExecutor``, so each restarts
    ``iter`` at 1. Rows are tagged with an ``instance`` that advances whenever a
    rank's counter goes backwards. Nothing is dropped: the estimation pass is
    real work the GPU did. It is only kept *apart*, so its iteration 1 does not
    merge with the real engine's into one row averaging two different pools.
    """
    rows: list[dict] = []
    instance = 0
    seen_iter: dict[int, int] = {}
    with path.open(encoding="latin-1", errors="replace") as handle:
        for line in handle:
            if not _ITER_LINE.search(line):
                continue
            tail = line[line.find("iter = "):].rstrip()
            states: dict[str, Any] = {}
            marker = tail.find("states = ")
            if marker >= 0:
                try:
                    states = ast.literal_eval(tail[marker + len("states = "):])
                except (ValueError, SyntaxError):
                    states = {}
                tail = tail[:marker]
            row: dict[str, Any] = {}
            for chunk in tail.split(", "):
                if " = " not in chunk:
                    continue
                key, _, value = chunk.partition(" = ")
                row[key.strip()] = value.strip().rstrip(",")
            if "iter" not in row:
                continue
            fetched = row.get("currank_total_requests", "")
            cur, _, glob = fetched.partition("/")
            iteration, rank = int(row["iter"]), int(row.get("rank", -1))
            # Not advancing is the restart signal, not merely going backwards:
            # within one engine a rank's counter strictly increases, and the
            # estimation pass can be a single iteration, so the real engine's
            # first line repeats the value rather than dropping below it.
            if iteration <= seen_iter.get(rank, 0):
                instance += 1
                seen_iter.clear()
            seen_iter[rank] = iteration
            rows.append({
                "instance": instance,
                "iter": iteration,
                "rank": rank,
                "global_rank": int(row.get("global_rank", -1)),
                "num_scheduled_requests": int(row.get("num_scheduled_requests", 0)),
                # Added by this fork; absent on older logs.
                "num_paused_requests": _f(row.get("num_paused_requests")),
                "kv_cache_util": _f(row.get("kv_cache_util")),
                "kv_hit_rate": _f(row.get("kv_hit_rate")),
                "kv_reused_blocks": _f(row.get("kv_reused_blocks")),
                "kv_missed_blocks": _f(row.get("kv_missed_blocks")),
                "kv_alloc_total_blocks": _f(row.get("kv_alloc_total_blocks")),
                "kv_alloc_new_blocks": _f(row.get("kv_alloc_new_blocks")),
                # Levels, not counters: how the pool's slots stand right now.
                # Absent before the fields were added, and None is the whole
                # point -- a run without them must not report a full pool as an
                # empty one.
                "kv_free_blocks": _f(row.get("kv_free_blocks")),
                "kv_evictable_blocks": _f(row.get("kv_evictable_blocks")),
                # The same two, split per pool group. The sums above cannot
                # show one saturated group, and a saturated group is what
                # drives eviction: the manager sizes each eviction off a single
                # group's free-slot count. Absent on logs written before the
                # worker emitted them.
                "kv_free_blocks_by_pg": _f_list(row.get("kv_free_blocks_by_pg")),
                "kv_evictable_blocks_by_pg": _f_list(
                    row.get("kv_evictable_blocks_by_pg")),
                # Cumulative, so these are differenced in _kv_deltas like the
                # reuse counters, never read as a level.
                "kv_offload_blocks": _f(row.get("kv_offload_blocks")),
                "kv_onboard_blocks": _f(row.get("kv_onboard_blocks")),
                "kv_host_dropped_blocks": _f(row.get("kv_host_dropped_blocks")),
                "fetched_currank": _f(cur),
                "fetched_global": _f(glob),
                "host_step_time_ms": _f(row.get("host_step_time", "").rstrip("ms")),
                "device_step_time_ms": _f(
                    row.get("prev_device_step_time", "").rstrip("ms")),
                "timestamp": row.get("timestamp"),
                "num_ctx_requests": states.get("num_ctx_requests"),
                "num_ctx_tokens": states.get("num_ctx_tokens"),
                "num_generation_tokens": states.get("num_generation_tokens"),
                "cached_kv_tokens": states.get("cached_kv_tokens"),
            })
    return rows


def parse_kv_capacity(path: Path) -> tuple[int | None, int | None]:
    """(primary_blocks, tokens_per_block) for one worker.

    The line is printed twice: once for the estimation dry run and once for
    the real allocation. The last one is the one that holds.
    """
    blocks = tokens = None
    with path.open(encoding="latin-1", errors="replace") as handle:
        for line in handle:
            hit = _KV_CAPACITY.search(line)
            if hit:
                tokens, blocks = int(hit.group(2)), int(hit.group(3))
    return blocks, tokens


# --------------------------------------------------------------------------
# section 1 -- per request
# --------------------------------------------------------------------------
def _index_by_ids(rows: list[dict], id_fields: Iterable[str]) -> dict[str, dict]:
    """Index rows under every id they carry.

    Disaggregated and aggregated runs bridge on different ids -- the engine
    request id in one, the serve-edge client id in the other -- so indexing
    everything and resolving by priority beats picking one and silently
    joining nothing.
    """
    index: dict[str, dict] = {}
    for row in rows:
        for field in id_fields:
            value = row.get(field)
            if value not in (None, ""):
                index.setdefault(str(value), row)
    return index


def _tool_latencies(audit_rows: list[dict]) -> dict[str, list[dict]]:
    """Resolve every tool call to the turn whose request carried its result.

    Scans ``tool_results_in_request`` -- the whole history -- rather than only
    the last message, because Claude Code often appends a system message after
    a tool result, which hides it from a last-message scan. Popping from
    ``pending`` keeps each id matchable once, so re-reading history costs
    nothing; measured, this takes the unmatched rate from ~21% to ~0%.
    """
    by_session: dict[Any, list[dict]] = defaultdict(list)
    for row in audit_rows:
        by_session[row.get("client_session_id")].append(row)

    out: dict[str, list[dict]] = defaultdict(list)
    for rows in by_session.values():
        rows.sort(key=lambda r: r.get("started_at") or "")
        pending: dict[str, dict] = {}
        for row in rows:
            started = _parse_iso(row.get("started_at"))
            for result in row.get("tool_results_in_request") or []:
                call = pending.pop(result.get("tool_use_id"), None)
                if call is None:
                    continue
                gap = None
                if started is not None and call["finished"] is not None:
                    gap = (started - call["finished"]) * 1000.0
                out[call["audit_id"]].append({
                    "tool": call["name"],
                    "latency_ms": gap,
                    "is_error": bool(result.get("is_error")),
                })
            finished = _parse_iso(row.get("finished_at"))
            for call in (row.get("response") or {}).get("tool_calls_emitted") or []:
                if call.get("id"):
                    pending[call["id"]] = {
                        "audit_id": row["audit_request_id"],
                        "name": call.get("name"),
                        "finished": finished,
                    }
        for call in pending.values():  # launched, never came back
            out[call["audit_id"]].append(
                {"tool": call["name"], "latency_ms": None, "is_error": False})
    return out


def _perf_role(path: Path, records: list[dict], disagg: bool) -> str:
    """Which hop a perf_metrics file came from, when the name does not say.

    Two writers are in the wild. The proxy's poller wraps each record and names
    the file after the hop it drained -- ``perf_metrics-{ctx,gen}-N.jsonl`` --
    so the name is the answer. A worker writing its own
    ``perf_metrics_output_dir`` names the file after the *process*, so both
    hops land in ``perf_metrics-server-<host>-<pid>-<stamp>.jsonl`` and the
    name says nothing. Sniff the content instead: only the generation hop
    receives a KV transfer, and only the context hop chunks its prefill.
    """
    token = path.stem.split("-")[1] if "-" in path.stem else ""
    if token in ("ctx", "gen"):
        return token
    if not disagg or token in ("proxy", "disagg"):
        return ""  # aggregated, or already-paired proxy records
    for record in records:
        inner = record.get("perf_metrics") or {}
        if "kv_cache_transfer_start" in (inner.get("timing_metrics") or {}):
            return "gen"
        if (record.get("time_breakdown_metrics") or {}).get("ctx_chunk_metrics"):
            return "ctx"
    return ""


def _load_perf(run: "Run") -> list[dict]:
    """Every perf record the run offers, unwrapped, role-tagged and paired.

    Cached on the run: ``build_requests`` and ``gpu_forward_ms`` both need it,
    and one of these files runs to millions of lines.
    """
    cached = getattr(run, "_perf_cache", None)
    if cached is not None:
        return cached
    rows: list[dict] = []
    for path in run.perf:
        raw = _read_jsonl(path)
        # The poller wraps; a worker writing the file itself does not, and
        # then the record *is* the line.
        wrapped = [w for w in raw if "record" in w]
        records = [w["record"] for w in wrapped] + [w for w in raw if "record" not in w]
        role = _perf_role(path, records, run.disagg)
        for entry in raw:
            record = entry.get("record")
            if record is None:
                record, drained = entry, None
            else:
                drained = entry.get("drained_at")
            record["_source"] = entry.get("source") or role
            record["_drained_at"] = drained
            rows.append(record)
    run._perf_cache = _pair_worker_records(rows)
    return run._perf_cache


def _pair_worker_records(records: list[dict]) -> list[dict]:
    """Fold per-worker disaggregated records into the proxy's nested shape.

    ``/perf_metrics`` drains on read, so when a proxy polls it first the proxy
    file holds both hops already paired. Without one, each worker drains its
    own hop into ``perf_metrics-{ctx,gen}-N.jsonl`` and the two records --
    sharing a ``ctx_request_id``, each carrying its own per-process
    ``request_id`` -- have to be joined here. Indexing them instead keeps
    whichever hop was read first and drops the other, which costs the whole
    generation side: KV transfer, decode timing, decode-side block counts.
    """
    out: list[dict] = []
    paired: dict[str, dict] = {}
    for record in records:
        role = (record.get("_source") or "").split("-")[0]
        key = record.get("ctx_request_id")
        already_nested = ("ctx_perf_metrics" in record
                          or "gen_perf_metrics" in record)
        if already_nested or role not in ("ctx", "gen") or key in (None, ""):
            # An unpairable generation record still has to be nested: _phase's
            # flat fallback reads a bare record as the *context* hop, which
            # would file decode timing and decode GPU time under prefill.
            out.append({"gen_perf_metrics": record,
                        "_source": record.get("_source"),
                        "_drained_at": record.get("_drained_at"),
                        "request_id": record.get("request_id")}
                       if role == "gen" and not already_nested else record)
            continue
        merged = paired.get(str(key))
        if merged is None:
            merged = {"ctx_request_id": key, "_source": "paired"}
            paired[str(key)] = merged
            out.append(merged)
        merged[f"{role}_perf_metrics"] = record
        drained = [t for t in (merged.get("_drained_at"),
                               record.get("_drained_at")) if t is not None]
        merged["_drained_at"] = max(drained) if drained else None
    return out


def build_requests(run: Run) -> list[dict]:
    audit = _read_jsonl(run.audit)
    perf_rows = _load_perf(run)

    # Proxy records nest both hops; aggregated records are flat.
    perf_index: dict[str, dict] = {}
    for record in perf_rows:
        ctx = record.get("ctx_perf_metrics") or {}
        gen = record.get("gen_perf_metrics") or {}
        for key in (ctx.get("ctx_request_id"), gen.get("ctx_request_id"),
                    record.get("ctx_request_id"), record.get("request_id")):
            if key not in (None, ""):
                perf_index.setdefault(str(key), record)

    decisions: list[dict] = []
    for path in run.route:
        worker = path.stem.replace("adp_route_trace-", "").replace(
            "adp_route_trace", "server")
        for batch in _read_jsonl(path):
            for decision in batch.get("decisions") or []:
                decision["_worker"] = worker
                decision["_iter"] = batch.get("iter")
                decision["_log_iter"] = batch.get("log_iter")
                decision["_load_before"] = batch.get("load_before")
                decisions.append(decision)
    # ``req_id`` is engine-global and safe to flatten. ``client_id`` is not: it
    # is a per-worker counter that restarts at 1 on every context server, so
    # merging both into one dict lets ``setdefault`` attach worker A's decision
    # to worker B's request the moment a run has more than one ctx instance --
    # silently, since the ids collide only in the small-integer range. Kept in a
    # separate index, and a client id that more than one worker used resolves to
    # nothing rather than to whichever file happened to sort first.
    _label_route_reasons(decisions)
    route_index = _index_by_ids(decisions, ("req_id",))
    client_index: dict[str, dict] = {}
    client_ambiguous: set[str] = set()
    for decision in decisions:
        value = decision.get("client_id")
        if value in (None, ""):
            continue
        key = str(value)
        earlier = client_index.get(key)
        if earlier is None:
            client_index[key] = decision
        elif earlier.get("_worker") != decision.get("_worker"):
            client_ambiguous.add(key)
    run.route_client_ambiguous = len(client_ambiguous)

    def route_for(*ids: Any) -> dict:
        """The routing decision for a request, engine id first.

        Both passes run over every id before falling through, rather than
        resolving each id across both indexes in turn: an engine-id hit is
        always right, and a client-id hit is only a fallback for the aggregated
        deployment, where the serve edge and the engine share one counter.
        """
        keys = [str(value) for value in ids if value not in (None, "")]
        for key in keys:
            found = route_index.get(key)
            if found is not None:
                return found
        for key in keys:
            if key in client_ambiguous:
                continue
            found = client_index.get(key)
            if found is not None:
                return found
        return {}

    tool_map = _tool_latencies(audit)
    rows: list[dict] = []
    for record in audit:
        rid = record.get("disagg_request_id") or record.get("engine_request_id")
        ctx_rid = record.get("ctx_request_id")
        perf = perf_index.get(str(rid)) or perf_index.get(str(ctx_rid)) or {}
        route = route_for(rid, ctx_rid)
        rows.append(_request_row(record, perf, route, tool_map))

    # Requests the audit never saw (direct-to-worker probes, or a run without
    # capture) still deserve a row, keyed on whatever the engine knows.
    seen = {str(r["rid"]) for r in rows if r["rid"] is not None}
    for record in perf_rows:
        ctx = record.get("ctx_perf_metrics") or {}
        rid = (ctx.get("ctx_request_id") or record.get("ctx_request_id")
               or record.get("request_id"))
        if rid is None or str(rid) in seen:
            continue
        rows.append(_request_row({}, record, route_for(rid), {}))
    rows.sort(key=lambda r: (r.get("started_at") or 0, str(r.get("rid"))))
    for index, row in enumerate(rows):
        row["request_index"] = index
    return rows


def _phase(perf: dict, side: str) -> dict:
    """Timing/KV for one hop; proxy records nest, aggregated ones are flat."""
    nested = perf.get(f"{side}_perf_metrics")
    if nested is None:
        nested = perf if side == "ctx" else {}
    inner = (nested or {}).get("perf_metrics") or {}
    return {
        "timing": inner.get("timing_metrics") or {},
        "kv": inner.get("kv_cache_metrics") or {},
        "breakdown": (nested or {}).get("time_breakdown_metrics") or {},
        "first_iter": inner.get("first_iter"),
        "last_iter": inner.get("last_iter"),
    }


def _label_route_reasons(decisions: list[dict]) -> None:
    """Stamp each routing decision with *why* that rank won.

    Replays the router's scoring loop per batch. The load figures in the trace
    are a pre-batch snapshot, so the tie-break on active tokens can only be
    reproduced by re-accumulating `effective_added` across the batch exactly as
    `route_requests` does -- reading `load_before` directly says every rank was
    idle and gets the tie-breaks wrong.
    """
    batches: dict[tuple, list[dict]] = defaultdict(list)
    for decision in decisions:
        batches[(decision.get("_worker"), decision.get("_iter"))].append(decision)

    for batch in batches.values():
        load = batch[0].get("_load_before") or {}
        tokens = [float(v) for v in (load.get("num_active_tokens") or [])]
        for decision in batch:
            matches = decision.get("match_lens")
            rank = decision.get("best_rank")
            if decision.get("phase") != "scored" or not matches:
                decision["_reason"] = f"{decision.get('phase') or 'unknown'} (unscored)"
            else:
                req = decision.get("req_tokens") or 0
                eligible = sorted(int(k) for k in matches)
                # The gate the router applies before scoring: under it, every
                # match is forced to 0 and the scores tie by construction.
                gated = (max(matches.values()) / max(req, 1)
                         ) > ROUTER_MATCH_RATE_THRESHOLD
                score = {r: req - (matches[str(r)] if gated else 0) for r in eligible}
                best = min(score.values())
                winners = [r for r in eligible if score[r] == best]
                if len(winners) == 1:
                    # A short candidate list means the fair-share cap dropped a
                    # rank for the rest of the batch -- possibly the one holding
                    # the prefix, which the trace cannot show either way.
                    decision["_reason"] = (
                        "best match (candidates capped)"
                        if tokens and len(matches) < len(tokens) else "best match")
                else:
                    least = min((tokens[r] for r in winners if r < len(tokens)),
                                default=0.0)
                    tied = [r for r in winners
                            if r < len(tokens) and tokens[r] == least]
                    decision["_reason"] = ("tie, fewest active tokens"
                                           if len(tied) == 1
                                           else "tie, req_id shuffle")
            if rank is not None and rank < len(tokens):
                tokens[rank] += max(
                    (decision.get("req_tokens") or 0)
                    - ((matches or {}).get(str(rank)) or 0), 0)


# Counters the attention-DP context pad leaves untouched. The pad is a real
# request with real blocks, but kv_cache_manager_v2 calls stop_committing() on
# it, so nothing reaches the radix tree and every one of these stands still.
_PAD_FROZEN = ("kv_reused_blocks", "kv_missed_blocks",
               "kv_alloc_total_blocks", "kv_alloc_new_blocks")


def _is_adp_pad(entry: dict, previous: dict | None, max_num_tokens: int | None) -> bool:
    """Is this rank-iteration the attention-DP context pad rather than work?

    Under attention DP every rank must step together, so a rank with nothing
    to do is handed a dummy sized at exactly ``max_num_tokens``
    (py_executor._pad_attention_dp_dummy_request). On a context-only
    disaggregated worker that dummy is always the CONTEXT flavour, and
    ``model_engine`` filters ``is_attention_dp_dummy`` only inside its
    *generation* branch -- the context loop that feeds ``num_ctx_tokens =
    len(input_ids)`` does not. So the pad lands in the iteration log looking
    exactly like a full chunk of real prefill.

    It is not a rounding error. Measured on two runs it was 84.5% and 87.6% of
    every context token the log reports, which is the whole of the 6.5x and
    8.1x gap between this column and the tokens the requests actually needed:
    removing it left a residual of *zero* on one run. Counting it inflates the
    per-rank token totals and pins mean utilisation near 1.0 whatever the
    worker was really doing -- the more idle the rank, the busier it looks.

    THE TEST IS A CONJUNCTION, AND BOTH HALVES ARE LOAD-BEARING.

    Size alone would be wrong: 8.6% of the rank-iterations at exactly
    ``max_num_tokens`` are genuine full chunks, and they are kept because their
    counters move (a real 8192-token chunk allocates 8192/128 = 64 blocks, seen
    as ``kv_missed_blocks`` +68 with the boundary block). Frozen counters alone
    would also be wrong: an iteration that is a pure cache hit allocates
    nothing either, so it too would stand still. Only both together identify
    the pad.

    THE PREMISE, stated so it can be re-checked rather than assumed: the dummy
    is sized at *exactly* ``max_num_tokens``. py_executor sets
    ``token_num = min(max_num_tokens, engine.max_num_tokens, engine.max_seq_len,
    kv_cache_manager.max_seq_len)`` and then clamps it to
    ``block_capacity - extra_kv_tokens``. On these deployments the first term
    wins and the clamp is inert, but a configuration where the clamp bites
    would size the pad below the budget and this test would silently stop
    finding it -- undercounting pads, not overcounting them, so the symptom is
    the ratio below creeping back up.

    THE CHECK that catches exactly that: with the pads removed, the summed
    context tokens should equal the audit's ``isl_new`` for the same window.
    Measured 0 residual on 08-26 and +0.29% on 08-23 (the latter being 30
    requests whose ctx perf records carry ``ctx_request_id: null`` and which
    the audit therefore never saw). If that agreement drifts, re-derive the
    dummy's size before trusting any prefill figure in section 3.
    """
    if not max_num_tokens:
        return False
    if (entry.get("num_ctx_tokens") or 0) != max_num_tokens:
        return False
    seen = [entry.get(k) for k in _PAD_FROZEN]
    if any(v is None for v in seen):
        return False
    if previous is None:
        # A rank's first logged step has nothing to stand still against. The
        # counters are cumulative from zero, so a real chunk of this size must
        # already have allocated max_num_tokens/tokens_per_block blocks; all
        # four still at zero means nothing was committed and this is the pad.
        # Measured, this is 7 rank-iterations on one run -- the four of the
        # KV-sizing dry-run engine plus three ranks idle on the real engine's
        # first step -- and skipping it leaves exactly 7 x 8192 unexplained.
        return all(v == 0 for v in seen)
    return all(entry.get(k) == previous.get(k) for k in _PAD_FROZEN)


def _request_row(audit: dict, perf: dict, route: dict, tool_map: dict) -> dict:
    ctx, gen = _phase(perf, "ctx"), _phase(perf, "gen")
    usage = audit.get("usage") or {}
    isl_cached = usage.get("cache_read_input_tokens")
    isl_new = usage.get("input_tokens")
    isl_total = None
    if isl_cached is not None and isl_new is not None:
        isl_total = isl_cached + isl_new

    def span(phase: dict, a: str, b: str) -> float | None:
        x, y = _f(phase["timing"].get(a)), _f(phase["timing"].get(b))
        return None if x is None or y is None else (y - x) * 1000.0

    started = _parse_iso(audit.get("started_at"))
    tools = tool_map.get(audit.get("audit_request_id"), [])
    tool_lat = [t["latency_ms"] for t in tools if t["latency_ms"] is not None]

    ttft = _f(audit.get("server_ttft_ms"))
    e2e = _f(audit.get("duration_ms"))
    # The audit stamps TTFT at the first streamed delta, so non-streaming
    # requests have none. The engine-side pair is always present and is a
    # different measurement -- it excludes the HTTP hop -- so it gets its own
    # column rather than being folded into ttft_ms.
    ttft_engine = None
    arrival = _f(perf.get("disagg_server_arrival_time"))
    first_tok = _f(perf.get("disagg_server_first_token_time"))
    if arrival is not None and first_tok is not None:
        ttft_engine = (first_tok - arrival) * 1000.0
    else:
        ttft_engine = span(ctx, "arrival_time", "first_token_time")
    return {
        "request_index": None,
        "rid": audit.get("disagg_request_id") or audit.get("engine_request_id")
               or (perf.get("ctx_perf_metrics") or {}).get("ctx_request_id")
               or perf.get("request_id"),
        "session_id": audit.get("client_session_id"),
        "audit_request_id": audit.get("audit_request_id"),
        "ctx_request_id": audit.get("ctx_request_id"),
        "anthropic_message_id": audit.get("anthropic_message_id"),
        "started_at": started,
        "finished_at": _parse_iso(audit.get("finished_at")),
        "status": audit.get("status"),
        "stream": audit.get("stream"),
        # --- tokens ---
        "isl_total": isl_total,
        "isl_cached": isl_cached,
        "isl_new": isl_new,
        "osl": usage.get("output_tokens"),
        "kv_hit_rate": (isl_cached / isl_total) if isl_total else None,
        # --- prefix-reuse split (LCP tracker; see the skill's step 4a) ---
        # opportunity = lcp / this prompt   -- the ceiling the prompt allows
        # realization = cached / lcp        -- how much of that the cache kept
        # Their product is kv_hit_rate. realization above 1.0 is physically
        # impossible and proves the tracker compared against the wrong prompt.
        "prompt_lcp_tokens": audit.get("prompt_lcp_tokens"),
        "previous_prompt_tokens": audit.get("previous_prompt_tokens"),
        "lcp_opportunity": audit.get("current_reuse_opportunity_ratio"),
        "lcp_retention": audit.get("previous_prompt_retention_ratio"),
        "cache_realization": (
            isl_cached / audit["prompt_lcp_tokens"]
            if isl_cached is not None and audit.get("prompt_lcp_tokens") else None),
        "message_capture_file": audit.get("message_capture_file"),
        # --- latency ---
        "ttft_ms": ttft,
        "ttft_engine_ms": ttft_engine,
        "e2e_ms": e2e,
        "decode_ms": (e2e - ttft) if (e2e is not None and ttft is not None) else None,
        "prefill_queue_ms": span(ctx, "arrival_time", "first_scheduled_time"),
        "prefill_ms": span(ctx, "first_scheduled_time", "first_token_time"),
        "kv_transfer_ms": span(gen, "kv_cache_transfer_start", "kv_cache_transfer_end"),
        "kv_transfer_bytes": gen["timing"].get("kv_cache_size"),
        "engine_decode_ms": span(gen, "first_token_time", "last_token_time"),
        "ctx_first_iter": ctx["first_iter"],
        "ctx_last_iter": ctx["last_iter"],
        "gen_first_iter": gen["first_iter"],
        "gen_last_iter": gen["last_iter"],
        # --- engine KV ---
        "ctx_blocks_total": ctx["kv"].get("num_total_allocated_blocks"),
        "ctx_blocks_new": ctx["kv"].get("num_new_allocated_blocks"),
        "ctx_blocks_reused": ctx["kv"].get("num_reused_blocks"),
        "gen_blocks_total": gen["kv"].get("num_total_allocated_blocks"),
        # --- routing ---
        "routed_rank": route.get("best_rank"),
        "route_phase": route.get("phase"),
        "route_iter": route.get("_iter"),
        "route_log_iter": route.get("_log_iter"),
        "match_len_chosen": (route.get("match_lens") or {}).get(
            str(route.get("best_rank"))),
        "match_len_best": max((route.get("match_lens") or {}).values(), default=None),
        # The whole per-rank probe, not just its extremes. `match_len_best` is a
        # max over the dict and so cannot tell "every rank was asked and none
        # had it" from "the rank that had it was never asked" -- and those are a
        # capacity problem and a routing problem respectively.
        "route_match_lens": "|".join(
            f"{k}:{v}" for k, v in sorted((route.get("match_lens") or {}).items())),
        "cache_affinity_active": route.get("cache_affinity_active"),
        "route_reason": route.get("_reason"),
        "effective_added": route.get("effective_added"),
        # --- tools ---
        "tool_names": "|".join(sorted({t["tool"] for t in tools if t["tool"]})),
        "tool_call_count": len(tools),
        "tool_unmatched": sum(1 for t in tools if t["latency_ms"] is None),
        "tool_latency_max_ms": max(tool_lat) if tool_lat else None,
        "tool_latency_sum_ms": sum(tool_lat) if tool_lat else None,
        # filled in a second pass, needs the session ordering
        "gap_to_next_turn_ms": None,
        "session_turn_index": None,
    }


def annotate_sessions(rows: list[dict]) -> None:
    """Fill the fields that only make sense once a session is ordered."""
    by_session: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[row.get("session_id")].append(row)
    for turns in by_session.values():
        turns.sort(key=lambda r: (r.get("started_at") or 0))
        for index, row in enumerate(turns, start=1):
            row["session_turn_index"] = index
        for cur, nxt in zip(turns, turns[1:]):
            if cur.get("finished_at") is not None and nxt.get("started_at") is not None:
                cur["gap_to_next_turn_ms"] = (
                    nxt["started_at"] - cur["finished_at"]) * 1000.0
    for turns in by_session.values():
        for prev, cur in zip(turns, turns[1:]):
            # How much of the previous prompt the cache still held. kv_hit_rate
            # divides by *this* prompt, so a turn that appends a lot reads as a
            # miss even when the cache lost nothing; this divides by what was
            # actually reusable. Healthy continuations sit at ~1.0.
            if prev["isl_total"] and cur["isl_cached"] is not None:
                cur["realization"] = cur["isl_cached"] / prev["isl_total"]
    for row in rows:
        row.setdefault("realization", None)
        best, chosen = row.get("match_len_best"), row.get("match_len_chosen")
        # Tokens the router gave up by not sending this request to the rank
        # holding the longest prefix -- the price of load balancing, in tokens.
        row["affinity_regret"] = (
            best - chosen if best is not None and chosen is not None else None)
        row["low_hit_rate"] = (
            row["kv_hit_rate"] is not None and row["kv_hit_rate"] < HIT_RATE_FLOOR)


# --------------------------------------------------------------------------
# section 2 -- per session
# --------------------------------------------------------------------------
def _per_turn(total_value: int | None, turns: int) -> float | None:
    return total_value / turns if total_value is not None and turns else None


def build_sessions(rows: list[dict], kv_capacity: dict) -> list[dict]:
    by_session: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[row.get("session_id")].append(row)

    bytes_per_token = kv_capacity.get("bytes_per_token")
    capacity_tokens = kv_capacity.get("capacity_tokens")

    out = []
    for session, turns in sorted(by_session.items(), key=lambda kv: str(kv[0])):
        turns.sort(key=lambda r: (r.get("started_at") or 0))
        starts = [t["started_at"] for t in turns if t["started_at"] is not None]
        ends = [t["finished_at"] for t in turns if t["finished_at"] is not None]
        span_ms = (max(ends) - min(starts)) * 1000.0 if starts and ends else None
        e2e_sum = sum(t["e2e_ms"] for t in turns if t["e2e_ms"] is not None) or None
        busy_ms = _union_seconds(
            [(t["started_at"], t["finished_at"]) for t in turns]) * 1000.0
        isl_max = max((t["isl_total"] for t in turns if t["isl_total"] is not None),
                      default=None)

        def total(field: str) -> int | None:
            vals = [t[field] for t in turns if t[field] is not None]
            return sum(vals) if vals else None

        out.append({
            "session_id": session,
            "turns": len(turns),
            "started_at": min(starts) if starts else None,
            "finished_at": max(ends) if ends else None,
            "run": turns[0].get("run"),
            "span_ms": span_ms,
            "isl_cached_total": total("isl_cached"),
            "isl_new_total": total("isl_new"),
            "osl_total": total("osl"),
            # Per request as well as summed. A session total conflates "long
            # turns" with "many turns" -- a 92-turn session tops every total
            # column without any of its turns being large -- so the table shows
            # the mean and sessions.csv keeps both.
            "isl_cached_mean": _per_turn(total("isl_cached"), len(turns)),
            "isl_new_mean": _per_turn(total("isl_new"), len(turns)),
            "osl_mean": _per_turn(total("osl"), len(turns)),
            "kv_hit_rate": (
                (total("isl_cached") or 0) / total("isl_total")
                if total("isl_total") else None),
            "ttft_sum_ms": total("ttft_ms"),
            "decode_sum_ms": total("decode_ms"),
            "e2e_sum_ms": e2e_sum,
            # Everything inside the session span with no request in flight:
            # client-side tool execution plus think time. Derived from the span
            # rather than from tool matching, so unmatched tool calls cannot
            # hide time -- and from a union rather than a sum, because turns of
            # one session do overlap and subtracting the sum drives it
            # negative.
            "client_time_ms": (
                (span_ms - busy_ms) if span_ms is not None else None),
            "tool_calls": total("tool_call_count"),
            "tool_latency_sum_ms": total("tool_latency_sum_ms"),
            "isl_max": isl_max,
            # Resident KV high-water mark: later prompts contain earlier ones
            # as a prefix, so the longest turn bounds the session.
            "kv_bytes_peak": (
                isl_max * bytes_per_token
                if isl_max is not None and bytes_per_token else None),
            "sessions_per_rank": (
                capacity_tokens // isl_max
                if isl_max and capacity_tokens else None),
            "ranks_used": "|".join(sorted({
                str(t["routed_rank"]) for t in turns
                if t["routed_rank"] is not None})),
        })
    return out


# --------------------------------------------------------------------------
# section 3 -- prefill server, per iteration
# --------------------------------------------------------------------------
def _rank_hit_rates(ranks: list[dict], prev: dict[int, dict]) -> list[float | None]:
    """Each rank's own hit rate for one iteration, index-aligned with ``ranks``.

    Must be called before :func:`_kv_deltas`, which advances ``prev``. The
    pooled rate that ``_kv_deltas`` produces sums the counters across ranks
    first, so a rank missing every block while its peers hit disappears into
    the average -- which is exactly the case worth seeing.
    """
    out: list[float | None] = []
    for entry in ranks:
        before = prev.get(entry["rank"])
        if not before or entry["kv_reused_blocks"] is None:
            out.append(None)
            continue
        reused = entry["kv_reused_blocks"] - (before["kv_reused_blocks"] or 0)
        missed = (entry["kv_missed_blocks"] or 0) - (before["kv_missed_blocks"] or 0)
        out.append(reused / (reused + missed) if reused + missed else None)
    return out


TIER_FIELDS = (("offload", "kv_offload_blocks"),
               ("onboard", "kv_onboard_blocks"),
               ("host_dropped", "kv_host_dropped_blocks"))


def _rank_tier_deltas(ranks: list[dict], prev: dict[int, dict]) -> list[dict]:
    """Each rank's own cross-tier block movement for one iteration.

    Must be called before :func:`_kv_deltas`, which advances ``prev``. The
    pooled version sums the ranks first, which answers "did the worker offload"
    but not "which rank did", and on an attention-DP worker those are different
    questions: one rank holding the long-lived prefixes offloads while its
    peers never touch the host tier.
    """
    out: list[dict] = []
    for entry in ranks:
        before = prev.get(entry["rank"])
        row = {}
        for key, field in TIER_FIELDS:
            row[key] = (None if not before or entry[field] is None
                        or before[field] is None
                        else (entry[field] or 0) - (before[field] or 0))
        out.append(row)
    return out


def _pooled_cum_hit(ranks: list[dict]) -> float | None:
    """The instance's lifetime hit rate: counters summed over ranks, divided once.

    Pooling before dividing is the point -- averaging each rank's ratio would
    weight a rank that acquired a hundred blocks the same as one that acquired
    a million. Needs no ``prev``: the log's counters run since engine start, so
    a single line already carries the answer.
    """
    reused = missed = 0.0
    seen = False
    for entry in ranks:
        if entry["kv_reused_blocks"] is None:
            continue
        seen = True
        reused += entry["kv_reused_blocks"]
        missed += entry["kv_missed_blocks"] or 0
    if not seen or reused + missed == 0:
        return None
    return reused / (reused + missed)


def _cum_hits(ranks: list[dict]) -> list[float | None]:
    """Each rank's own lifetime ratio, index-aligned with ``ranks``."""
    return [entry["kv_hit_rate"] for entry in ranks]


def _kv_deltas(ranks: list[dict], prev: dict[int, dict]
               ) -> tuple[float, float, float, dict[str, float], bool, bool, bool]:
    """Per-iteration KV deltas summed across one instance's ranks.

    ``have_alloc`` and ``have_tier`` are tracked apart from ``have_delta``
    because logs written before each set of counters existed would otherwise
    difference None-as-zero and report a confident zero where there is no
    measurement.

    ``have_tier`` doubles as the KV-manager version test, which is why it
    matters beyond presence. On v2 every call site assigns
    ``alloc_total_blocks`` and ``alloc_new_blocks`` the same value, so
    ``evicted`` below is identically zero there and means nothing; the tier
    counters are v2-only, so their presence says "believe host_dropped, not
    evicted".
    """
    reused = missed = evicted = 0.0
    tier = {"offload": 0.0, "onboard": 0.0, "host_dropped": 0.0}
    have_delta = have_alloc = have_tier = False
    for entry in ranks:
        before = prev.get(entry["rank"])
        if before:
            have_delta = True
            reused += (entry["kv_reused_blocks"] or 0) - (before["kv_reused_blocks"] or 0)
            missed += (entry["kv_missed_blocks"] or 0) - (before["kv_missed_blocks"] or 0)
            if (entry["kv_alloc_total_blocks"] is not None
                    and before["kv_alloc_total_blocks"] is not None):
                have_alloc = True
                evicted += ((entry["kv_alloc_total_blocks"] - entry["kv_alloc_new_blocks"])
                            - (before["kv_alloc_total_blocks"]
                               - before["kv_alloc_new_blocks"]))
            if (entry["kv_host_dropped_blocks"] is not None
                    and before["kv_host_dropped_blocks"] is not None):
                have_tier = True
                for key, field in (("offload", "kv_offload_blocks"),
                                   ("onboard", "kv_onboard_blocks"),
                                   ("host_dropped", "kv_host_dropped_blocks")):
                    tier[key] += (entry[field] or 0) - (before[field] or 0)
        prev[entry["rank"]] = entry
    return reused, missed, evicted, tier, have_delta, have_alloc, have_tier


def _pool_capacity(entries: list[dict]) -> dict[tuple[int, int], float]:
    """Total KV slots per (engine instance, rank), from the level counters.

    The v2 manager never prints the line :func:`parse_kv_capacity` looks for,
    so capacity has to come out of the levels themselves. ``free + evictable +
    pinned = max`` and pinned is never negative, so the largest ``free +
    evictable`` ever observed *is* max -- exact on any iteration where nothing
    was pinned, which every run has.

    Dividing instead -- ``max = (free + evictable) / (1 - util)`` -- looks
    equivalent and is not: util is printed to three decimals, and at the low
    pin rates these workers actually run the rounding dominates and the
    estimate wanders by tens of slots.

    Keyed on the instance too, not the rank alone. A log holds more than one
    engine lifetime and the pool-sizing pass runs a pool several times smaller;
    charging its rows against the real engine's capacity reports a nearly empty
    throwaway pool as three-quarters full.
    """
    capacity: dict[tuple[int, int], float] = {}
    for entry in entries:
        free, evictable = entry["kv_free_blocks"], entry["kv_evictable_blocks"]
        if free is None or evictable is None:
            continue
        key = (entry["instance"], entry["rank"])
        capacity[key] = max(capacity.get(key, 0.0), free + evictable)
    return capacity


_ROLE_LINE = re.compile(
    r"deepseek_role=(\w+),\s*compress_ratio=(\d+),\s*role=\S+?,\s*"
    r"pool_group_id=(\d+),\s*layer_group_id=(\d+)")

# COMPRESSOR_KV / INDEXER_COMPRESSOR_SCORE / ... all describe one thing at the
# grain a reader cares about. Collapsed so a legend entry stays legible; the
# caption keeps the unabridged list.
_ROLE_FAMILY = (("INDEXER_COMPRESSOR", "COMPRESSOR"), ("COMPRESSOR", "COMPRESSOR"),
                ("INDEXER_COMPRESS", "COMPRESS"), ("COMPRESS", "COMPRESS"),
                ("SWA", "SWA"))


def parse_pool_group_roles(path: Path) -> dict[int, dict[str, Any]]:
    """Which KV content each pool group holds, from the worker's own mapping line.

    ``DeepseekV4CacheManager`` prints one line per (role, compress_ratio) naming
    the ``pool_group_id`` it landed in, so the mapping is read rather than
    assumed. That matters: pool group index is not a semantic order but the
    ascending sort of each group's slot-size vector, so it is neither the order
    the roles are declared in nor the order a hand-written ``pool_ratio``
    comment is likely to guess.

    Empty for any model that does not print the line, and the caller falls back
    to bare ``pgN`` labels rather than inventing names.
    """
    groups: dict[int, dict[str, Any]] = {}
    with path.open(encoding="latin-1", errors="replace") as handle:
        for line in handle:
            found = _ROLE_LINE.search(line)
            if not found:
                continue
            role, ratio, pg_id, _layer = found.groups()
            entry = groups.setdefault(int(pg_id), {"roles": set(), "ratios": set()})
            entry["roles"].add(role)
            entry["ratios"].add(int(ratio))
    for entry in groups.values():
        families: list[str] = []
        for role in sorted(entry["roles"]):
            for prefix, family in _ROLE_FAMILY:
                if role.startswith(prefix):
                    if family not in families:
                        families.append(family)
                    break
        entry["label"] = "+".join(sorted(families))
        entry["roles"] = sorted(entry["roles"])
        entry["ratios"] = sorted(entry["ratios"])
    return groups


def _pool_capacity_by_pg(entries: list[dict]
                         ) -> dict[tuple[int, int, int], float]:
    """Slots per (engine instance, rank, pool group), same estimator as pooled.

    ``free + evictable`` peaks when nothing is pinned, so its maximum over the
    run is that group's slot count. Kept per group precisely because the groups
    are not the same size: the ratio splits *bytes*, and a group whose blocks
    are large gets proportionally fewer slots, so a summed capacity is
    dominated by the smallest-block group and says nothing about the others.
    """
    capacity: dict[tuple[int, int, int], float] = {}
    for entry in entries:
        free = entry.get("kv_free_blocks_by_pg")
        evictable = entry.get("kv_evictable_blocks_by_pg")
        if not free or not evictable or len(free) != len(evictable):
            continue
        for pg, (f, e) in enumerate(zip(free, evictable)):
            key = (entry["instance"], entry["rank"], pg)
            capacity[key] = max(capacity.get(key, 0.0), f + e)
    return capacity


def _final_capacity_by_pg(capacity: dict[tuple[int, int, int], float]
                          ) -> list[float] | None:
    """Per-pool-group slot counts of the engine that served the traffic.

    Summed per group rather than taken from the pooled estimate, and the two
    are not interchangeable. ``max_t sum_g x_g(t) <= sum_g max_t x_g(t)``: the
    pooled form is exact only if every group is unpinned on the *same*
    iteration, while this one needs each group unpinned at *some* iteration.
    They agree when the log starts at engine startup, where nothing is pinned
    yet, and diverge on a windowed report or a rolled log -- with the pooled
    form understating.
    """
    if not capacity:
        return None
    last = max(instance for instance, _, _ in capacity)
    groups = sorted({pg for instance, _, pg in capacity if instance == last})
    return [max((value for (i, _, pg), value in capacity.items()
                 if i == last and pg == group), default=0.0)
            for group in groups]


def _final_capacity(capacity: dict[tuple[int, int], float]) -> float | None:
    """Per-rank slot count of the engine that served the traffic -- the last."""
    if not capacity:
        return None
    last = max(instance for instance, _ in capacity)
    return max(value for (instance, _), value in capacity.items() if instance == last)


def _pool_filled(entry: dict, capacity: dict[tuple[int, int], float]) -> float | None:
    """Share of a rank's pool holding content, pinned or merely retained.

    ``1 - free/max`` rather than ``evictable/max`` deliberately. A block whose
    content is reusable but currently locked by an in-flight request counts as
    pinned, not evictable, so ``evictable`` dips whenever traffic is active and
    reads as the cache shrinking while it is in fact growing -- measured, it
    fell from 478 to 40 slots across two iterations that added content. Free is
    the quantity that only moves when the pool genuinely fills.
    """
    total = capacity.get((entry["instance"], entry["rank"]))
    free = entry["kv_free_blocks"]
    if not total or free is None:
        return None
    return 1.0 - free / total


def build_gen_iters(run: Run, max_batch_size: int) -> list[dict]:
    """One row per (generation instance, iteration).

    Kept apart from the prefill table on purpose: the two workers print the
    same field names for different quantities. ``num_scheduled_requests`` is
    prefills on one side and decode steps on the other, and the budget a
    prefill iteration fills is a token count (``max_num_tokens``) while a
    decode iteration fills batch slots (``max_batch_size``). Pooling them would
    divide by two different denominators under one heading.
    """
    rows: list[dict] = []
    for path in run.decode_logs():
        worker = run.worker_name(path)
        manager_v2 = run.kv_manager_v2()
        entries = parse_iter_log(path)
        capacity = _pool_capacity(entries)
        # Keyed on (instance, iter), not iter alone: the pool-sizing engine and
        # the real one both count from 1, and merging their first iterations
        # would average two differently sized pools into one row.
        per_iter: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for entry in entries:
            per_iter[(entry["instance"], entry["iter"])].append(entry)

        prev: dict[int, dict] = {}
        current_instance = None
        for key in sorted(per_iter):
            instance, iteration = key
            if instance != current_instance:
                # Every counter restarts with the engine, so differencing this
                # instance's first iteration against the previous instance's
                # last would subtract a live total from a fresh zero and report
                # a large negative delta.
                prev.clear()
                current_instance = instance
            ranks = per_iter[key]
            batch = [r["num_scheduled_requests"] for r in ranks]
            tokens = [r["num_generation_tokens"] or 0 for r in ranks]
            batch_mean = statistics.fmean(batch) if batch else 0.0
            batch_total, tokens_total = sum(batch), sum(tokens)
            per_rank_hit = _rank_hit_rates(ranks, prev)
            (reused, missed, evicted, tier,
             have_delta, have_alloc, have_tier) = _kv_deltas(ranks, prev)
            device = [r["device_step_time_ms"] for r in ranks
                      if r["device_step_time_ms"] is not None]
            util = [r["kv_cache_util"] for r in ranks if r["kv_cache_util"] is not None]
            filled = [f for f in (_pool_filled(r, capacity) for r in ranks)
                      if f is not None]
            rows.append({
                "worker": worker,
                "instance": instance,
                "iter": iteration,
                "ranks": len(ranks),
                "has_decode": batch_total > 0,
                "decode_batch_mean": batch_mean,
                "decode_batch_max": max(batch, default=0),
                "decode_batch_total": batch_total,
                "batch_occupancy": batch_mean / max_batch_size if max_batch_size else None,
                "imbalance": (max(batch) - batch_mean) / batch_mean if batch_mean else None,
                "gen_tokens_mean": statistics.fmean(tokens) if tokens else 0.0,
                "gen_tokens_max": max(tokens, default=0),
                "gen_tokens_total": tokens_total,
                # 1.0 without speculative decoding; with MTP it is the accepted
                # draft length plus one, and it slides back toward 1.0 as
                # acceptance degrades -- which neither batch size nor step time
                # would reveal on its own.
                "tokens_per_request": tokens_total / batch_total if batch_total else None,
                "cached_kv_tokens": sum(r["cached_kv_tokens"] or 0 for r in ranks),
                "paused_requests": sum(int(r["num_paused_requests"] or 0) for r in ranks),
                "kv_util_mean": statistics.fmean(util) if util else None,
                "kv_util_max": max(util, default=None),
                "kv_util_spread": _spread(util),
                "kv_hit_rate_spread": _spread(per_rank_hit),
                "kv_hit_rate_cum": _pooled_cum_hit(ranks),
                "kv_hit_rate_cum_spread": _spread(_cum_hits(ranks)),
                "device_step_spread": _spread(device),
                "gen_tokens_spread": _spread(tokens),
                "kv_hit_rate_iter": (
                    reused / (reused + missed) if have_delta and (reused + missed) else None),
                # v2 assigns alloc_total and alloc_new the same value at every
                # call site, so this difference is identically zero there and
                # would read as a measured "no evictions". The tier counters
                # only exist on v2, so their presence is the version test.
                "kv_evicted_tokens": (evicted * 128 if have_alloc
                                      and not manager_v2 else None),
                "kv_capacity_blocks": capacity.get((instance, ranks[0]["rank"])),
                "kv_free_blocks": _opt_sum(ranks, "kv_free_blocks"),
                "kv_evictable_blocks": _opt_sum(ranks, "kv_evictable_blocks"),
                "kv_pool_filled": statistics.fmean(filled) if filled else None,
                "kv_offload_blocks_iter": tier["offload"] if have_tier else None,
                "kv_onboard_blocks_iter": tier["onboard"] if have_tier else None,
                "kv_host_dropped_blocks_iter": (tier["host_dropped"]
                                                if have_tier else None),
                "host_step_time_ms": statistics.fmean(
                    [r["host_step_time_ms"] for r in ranks
                     if r["host_step_time_ms"] is not None] or [0.0]),
                "device_step_time_ms": statistics.fmean(device) if device else None,
                "timestamp": ranks[0]["timestamp"],
            })
    return rows


def _opt_sum(ranks: list[dict], field: str) -> float | None:
    """Sum a field across ranks, or None when no rank reported it.

    Summing with ``or 0`` would turn "this log predates the field" into a
    confident zero, which for a pool level is the opposite of the truth.
    """
    kept = [r[field] for r in ranks if r.get(field) is not None]
    return sum(kept) if kept else None


def _spread(values: list[float]) -> float | None:
    """(max - mean) / mean across an instance's ranks.

    Zero when every rank carried the same amount; with n ranks and only one
    doing the work it reaches n - 1. Reported next to the mean because a mean
    alone cannot distinguish four ranks at 25% from one rank at 100%.
    """
    kept = [v for v in values if v is not None]
    # Undefined below two ranks, not zero. Under attention-DP a prefill lands
    # on one rank, so the other ranks report no hit rate that iteration; a
    # single sample would otherwise score a spread of 0 and read as "perfectly
    # balanced" when nothing was compared.
    if len(kept) < 2:
        return None
    mean = statistics.fmean(kept)
    return (max(kept) - mean) / mean if mean else None


def build_ctx_iters(run: Run, max_num_tokens: int
                    ) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """One row per (prefill instance, iteration), plus the KV capacity found.

    Utilization is the mean context tokens across *all* of that instance's
    ranks over the per-rank token budget, counting an attention-DP pad as the
    0 tokens of real work it did rather than dropping it from the average --
    see the comment on ctx_tokens below for why neither dropping it nor taking
    its logged size works. Imbalance is (max - mean) / mean across the same
    ranks: 0 means every rank got the same prefill work this iteration, and
    ranks - 1 means one rank got all of it.
    """
    rows: list[dict] = []
    # Per (worker, rank, iteration). The aggregate rows pool a worker's ranks,
    # which is right for occupancy but wrong for a hit rate: pooling hides the
    # one rank that is missing while its peers hit.
    rank_rows: list[dict] = []
    # Per (worker, rank, pool group, iteration). Eviction is decided one pool
    # group at a time, so this is the only grain at which "the pool filled"
    # is the same question the allocator asks.
    pg_rows: list[dict] = []
    capacity: dict[str, Any] = {}
    # previous entry per (instance, rank), for the pad test -- the pad is
    # recognised by what stands still since that rank last stepped
    pad_prev: dict[tuple, dict] = {}
    for path in run.prefill_logs():
        worker = run.worker_name(path)
        manager_v2 = run.kv_manager_v2()
        blocks, tokens_per_block = parse_kv_capacity(path)
        if blocks and tokens_per_block:
            capacity = {
                "primary_blocks": blocks,
                "tokens_per_block": tokens_per_block,
                "capacity_tokens": blocks * tokens_per_block,
            }
        tokens_per_block = tokens_per_block or 128

        entries = parse_iter_log(path)
        # Stamp the attention-DP pad flag once, walking each rank's own
        # timeline, so the pooled and per-rank paths below agree and neither
        # has to care about loop order.
        for entry in sorted(entries, key=lambda e: (str(e["instance"]), e["rank"],
                                                    e["iter"])):
            key = (entry["instance"], entry["rank"])
            entry["_adp_pad"] = _is_adp_pad(entry, pad_prev.get(key), max_num_tokens)
            pad_prev[key] = entry
        slots = _pool_capacity(entries)
        pg_slots = _pool_capacity_by_pg(entries)
        pg_roles = parse_pool_group_roles(path)
        if pg_roles:
            capacity = {**capacity, "pool_group_roles": pg_roles}
        if slots and not capacity.get("capacity_tokens"):
            # v2 prints no capacity line at all, so this is the only place the
            # pool size comes from on a v2 run. Slots, not tokens: with more
            # than one pool group the slot sizes differ, and multiplying a slot
            # count by tokens_per_block overstates the pool -- on the DeepSeek
            # pilot by 1.38x against the engine's own quota arithmetic.
            capacity = {**capacity, "slots_per_rank": _final_capacity(slots),
                        "slots_per_rank_by_pg": _final_capacity_by_pg(pg_slots),
                        "tokens_per_block": tokens_per_block}
        # Keyed on (instance, iter), not iter alone: the pool-sizing engine and
        # the real one both count from 1, and merging their first iterations
        # would average two differently sized pools into one row.
        per_iter: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for entry in entries:
            per_iter[(entry["instance"], entry["iter"])].append(entry)

        prev: dict[int, dict] = {}
        current_instance = None
        for key in sorted(per_iter):
            instance, iteration = key
            if instance != current_instance:
                # Every counter restarts with the engine, so differencing this
                # instance's first iteration against the previous instance's
                # last would subtract a live total from a fresh zero and report
                # a large negative delta.
                prev.clear()
                current_instance = instance
            ranks = per_iter[key]
            # A pad contributes 0 tokens, but it still occupies its rank for
            # the iteration, so it stays in the denominator. Dropping the pad
            # row instead divides by the busy ranks alone, and then a rank
            # working alone beside three idle ones reads utilization 1.000,
            # imbalance 0.000 -- indistinguishable from all four running a
            # full chunk. On this run that is 85% of prefilling iterations,
            # and it moved utilization 0.129 -> 0.413 (p99 0.521 -> 1.000)
            # and imbalance 2.810 -> 0.066, 87% of it exact zero.
            # Counting the pad at its logged size is the other wrong
            # answer: it is emitted at exactly max_num_tokens, which pins peak
            # at the ceiling and makes imbalance identically 1/utilization - 1
            # (verified on 14,006 of 14,006 pad-bearing iterations), a column
            # with no information the utilization column did not already have.
            ctx_tokens = [0 if r.get("_adp_pad") else (r["num_ctx_tokens"] or 0)
                          for r in ranks]
            mean = statistics.fmean(ctx_tokens) if ctx_tokens else 0.0
            # Zeroed pads cannot raise the max, so this is still the largest
            # real chunk, and has_prefill below gates on the same iterations
            # it did before.
            peak = max(ctx_tokens, default=0)

            per_rank_hit = _rank_hit_rates(ranks, prev)
            per_rank_tier = _rank_tier_deltas(ranks, prev)
            for entry, hit, tier_delta in zip(ranks, per_rank_hit, per_rank_tier):
                rank_rows.append({
                    "worker": worker,
                    "rank": entry["rank"],
                    "series": f'{worker} r{entry["rank"]}',
                    "instance": entry["instance"],
                    "iter": iteration,
                    "timestamp": entry["timestamp"],
                    # Both forms, because they answer different questions and
                    # neither substitutes for the other. The delta is what
                    # moved on this iteration -- bursty, mostly zero, and the
                    # only way to see *when*. The cumulative counter is the
                    # log's own field passed straight through, and it is what
                    # the charts draw: a count series cannot survive the
                    # time-window averaging in _downsample, whereas a monotone
                    # curve can, and "how many blocks by now" is read off it
                    # directly while the rate is its slope.
                    **{f"kv_{key}_blocks_iter": tier_delta[key]
                       for key, _ in TIER_FIELDS},
                    **{f"kv_{key}_blocks_cum": entry[field]
                       for key, field in TIER_FIELDS},
                    "kv_hit_rate_iter": hit,
                    # The log's own lifetime ratio: reused/(reused+missed)
                    # over every block this rank has acquired since the
                    # engine started. Charted in place of the per-iteration
                    # delta because the delta divides by the handful of
                    # blocks a single iteration happened to acquire, and a
                    # chunked prefill's continuation chunk acquires only
                    # fresh blocks by construction -- it reads exactly 0
                    # however warm the request was. On this run 15.7% of
                    # prefilling iterations read 0 that way, pulling an
                    # unweighted mean to 0.780 on a run whose counters say
                    # 0.941. The counters only rise, so this is a running
                    # total, not a rate: free to move early, nearly pinned
                    # late, when the denominator is already millions.
                    "kv_hit_rate_cum": entry["kv_hit_rate"],
                    # Named for what it is rather than what the log calls it:
                    # this is 1 - available/max, and available counts a retained
                    # reusable block as free, so it only ever measures slots
                    # pinned by in-flight requests.
                    "kv_cache_util": entry["kv_cache_util"],
                    "kv_free_blocks": entry["kv_free_blocks"],
                    "kv_evictable_blocks": entry["kv_evictable_blocks"],
                    "kv_capacity_blocks": slots.get((entry["instance"], entry["rank"])),
                    "kv_pool_filled": _pool_filled(entry, slots),
                    "num_ctx_tokens": entry["num_ctx_tokens"],
                    "is_adp_pad": entry.get("_adp_pad"),
                    "utilization": ((entry["num_ctx_tokens"] or 0) / max_num_tokens
                                    if max_num_tokens else None),
                })
                free_pg = entry.get("kv_free_blocks_by_pg") or []
                evict_pg = entry.get("kv_evictable_blocks_by_pg") or []
                for pg, free_slots in enumerate(free_pg):
                    total = pg_slots.get((entry["instance"], entry["rank"], pg))
                    named = (pg_roles.get(pg) or {}).get("label")
                    tag = f"pg{pg}" + (f" {named}" if named else "")
                    pg_rows.append({
                        "worker": worker,
                        "rank": entry["rank"],
                        "pool_group": pg,
                        "pool_group_roles": ",".join(
                            (pg_roles.get(pg) or {}).get("roles") or []) or None,
                        "series": f'{worker} r{entry["rank"]} {tag}',
                        "instance": entry["instance"],
                        "iter": iteration,
                        "timestamp": entry["timestamp"],
                        "kv_free_blocks": free_slots,
                        "kv_evictable_blocks": (evict_pg[pg]
                                                if pg < len(evict_pg) else None),
                        "kv_capacity_blocks": total,
                        "kv_pool_filled": (1.0 - free_slots / total
                                           if total else None),
                    })
            (reused, missed, evicted, tier,
             have_delta, have_alloc, have_tier) = _kv_deltas(ranks, prev)

            device = [r["device_step_time_ms"] for r in ranks
                      if r["device_step_time_ms"] is not None]
            filled = [f for f in (_pool_filled(r, slots) for r in ranks)
                      if f is not None]
            rows.append({
                "worker": worker,
                "instance": instance,
                "iter": iteration,
                "ranks": len(ranks),
                # An aggregated server spends most iterations in decode with no
                # context tokens at all. Averaging utilization over those
                # dilutes it to noise, so section 3 reports over prefilling
                # iterations and says how many that was.
                "has_prefill": peak > 0,
                "ctx_tokens_mean": mean,
                "ctx_tokens_max": peak,
                "utilization": mean / max_num_tokens if max_num_tokens else None,
                # Zero when every rank carried the same load; grows with skew.
                "imbalance": (peak - mean) / mean if mean else None,
                "kv_hit_rate_iter": (
                    reused / (reused + missed) if have_delta and (reused + missed) else None),
                # v2 assigns alloc_total and alloc_new the same value at every
                # call site, so this difference is identically zero there and
                # would read as a measured "no evictions". Gated on the manager
                # version rather than on have_tier: the tier counters are newer
                # than v2 itself, and a v2 run predating them still needs this
                # blanked. kv_host_dropped_blocks_iter is the answer instead,
                # where the run is new enough to have it.
                "kv_evicted_tokens": (evicted * tokens_per_block if have_alloc
                                      and not manager_v2 else None),
                "kv_capacity_blocks": slots.get((instance, ranks[0]["rank"])),
                "kv_free_blocks": _opt_sum(ranks, "kv_free_blocks"),
                "kv_evictable_blocks": _opt_sum(ranks, "kv_evictable_blocks"),
                "kv_pool_filled": statistics.fmean(filled) if filled else None,
                # The spread that is worth reading. Pool fill is rank-resident
                # and persists across iterations, so a difference between ranks
                # is a real difference in what they hold -- unlike the pinned
                # and utilization spreads, which are structural under
                # attention-DP because one prefill occupies one rank.
                "kv_pool_filled_spread": _spread(filled),
                "kv_offload_blocks_iter": tier["offload"] if have_tier else None,
                "kv_onboard_blocks_iter": tier["onboard"] if have_tier else None,
                "kv_host_dropped_blocks_iter": (tier["host_dropped"]
                                                if have_tier else None),
                "kv_util_mean": statistics.fmean(
                    [r["kv_cache_util"] for r in ranks if r["kv_cache_util"] is not None]
                    or [0.0]),
                "kv_util_spread": _spread([r["kv_cache_util"] for r in ranks]),
                "kv_hit_rate_spread": _spread(per_rank_hit),
                "kv_hit_rate_cum": _pooled_cum_hit(ranks),
                "kv_hit_rate_cum_spread": _spread(_cum_hits(ranks)),
                "device_step_spread": _spread(device),
                "paused_requests": sum(int(r["num_paused_requests"] or 0) for r in ranks),
                "scheduled_requests": sum(r["num_scheduled_requests"] for r in ranks),
                "device_step_time_ms": statistics.fmean(device) if device else None,
                "host_step_time_ms": statistics.fmean(
                    [r["host_step_time_ms"] for r in ranks
                     if r["host_step_time_ms"] is not None] or [0.0]),
                "timestamp": ranks[0]["timestamp"],
            })
    return rows, capacity, rank_rows, pg_rows


# --------------------------------------------------------------------------
# section 4 -- whole run
# --------------------------------------------------------------------------
def gpu_forward_ms(run: Run) -> dict[str, float | None]:
    """GPU forward time per role, de-duplicated by batch.

    One forward serves the whole batch, and its elapsed time is stamped onto
    every request in that batch, so summing across requests multiplies by the
    batch size. ``forward_start_time`` is written into every metric entry and
    is identical within a batch, which makes it the batch key. Clocks are
    per-process, so ctx and gen are de-duplicated separately.
    """
    batches: dict[str, dict[float, float]] = {"prefill": {}, "decode": {}}
    missing = 0
    for record in _load_perf(run):
        for side, role in (("ctx", "prefill"), ("gen", "decode")):
            phase = _phase(record, side)
            breakdown = phase["breakdown"]
            if not breakdown:
                if phase["timing"]:
                    missing += 1
                continue
            for entry in (breakdown.get("ctx_chunk_metrics") or []) + (
                    breakdown.get("step_metrics") or []):
                key, value = entry.get("forward_start_time"), entry.get("gpu_forward_time")
                if key is not None and value:
                    batches[role][key] = value
    return {
        "prefill_ms": sum(batches["prefill"].values()) or None,
        "decode_ms": sum(batches["decode"].values()) or None,
        "prefill_batches": len(batches["prefill"]),
        "decode_batches": len(batches["decode"]),
        "phases_without_gpu_time": missing,
    }


def _tier_total(iters: list[dict], name: str) -> float | None:
    """Run total for one cross-tier counter, or None if the run never had it."""
    field = f"kv_{name}_blocks_iter"
    kept = [row[field] for row in iters if row.get(field) is not None]
    return sum(kept) if kept else None


def _ratio(top: float | None, bottom: float | None) -> float | None:
    return None if top is None or not bottom else top / bottom


def build_summary(runs: list[Run], requests: list[dict], sessions: list[dict],
                  ctx_iters: list[dict], gen_iters: list[dict],
                  capacity: dict, rank_iters: list[dict] | None = None) -> dict:
    """Totals for one run, or for several merged.

    Time is accumulated per run and then summed rather than measured across
    the merged set: two serving jobs recorded hours apart are disjoint, and
    spanning them would charge the gap between them to the workload. Token
    counts and hit rate pool freely, having no time dimension.
    """
    wall = busy = 0.0
    for one in runs:
        rows = [r for r in requests if r.get("run") in (one.name, None)]
        starts = [r["started_at"] for r in rows if r["started_at"] is not None]
        ends = [r["finished_at"] for r in rows if r["finished_at"] is not None]
        if starts and ends:
            wall += max(ends) - min(starts)
        busy += _union_seconds([(r["started_at"], r["finished_at"]) for r in rows])
    wall = wall or None
    gpu = {"prefill_ms": 0.0, "decode_ms": 0.0, "prefill_batches": 0,
           "decode_batches": 0, "phases_without_gpu_time": 0}
    for one in runs:
        # Batch keys are per-process clocks, so de-duplication has to happen
        # inside a run before the totals are added up.
        part = gpu_forward_ms(one)
        gpu["prefill_ms"] += part["prefill_ms"] or 0.0
        gpu["decode_ms"] += part["decode_ms"] or 0.0
        for key in ("prefill_batches", "decode_batches", "phases_without_gpu_time"):
            gpu[key] += part[key]
    gpu["prefill_ms"] = gpu["prefill_ms"] or None
    gpu["decode_ms"] = gpu["decode_ms"] or None

    prefilling = [c for c in ctx_iters if c.get("has_prefill")]
    cached = sum(r["isl_cached"] or 0 for r in requests)
    total_isl = sum(r["isl_total"] or 0 for r in requests)
    return {
        "runs": len(runs),
        "requests": len(requests),
        "sessions": len(sessions),
        "wall_s": wall,
        # Wall time with at least one request in flight. A union, not a sum:
        # concurrent requests overlap.
        "server_busy_s": busy or None,
        # Everything else inside the window -- tool execution and client think
        # time -- with no request in flight.
        "client_idle_s": (wall - busy) if wall is not None and busy else None,
        "gpu_prefill_s": (gpu["prefill_ms"] / 1000.0) if gpu["prefill_ms"] else None,
        "gpu_decode_s": (gpu["decode_ms"] / 1000.0) if gpu["decode_ms"] else None,
        "gpu_prefill_batches": gpu["prefill_batches"],
        "gpu_decode_batches": gpu["decode_batches"],
        "phases_without_gpu_time": gpu["phases_without_gpu_time"],
        "kv_hit_rate": (cached / total_isl) if total_isl else None,
        "isl_cached_total": cached or None,
        "isl_new_total": sum(r["isl_new"] or 0 for r in requests) or None,
        "osl_total": sum(r["osl"] or 0 for r in requests) or None,
        "low_hit_rate_requests": sum(1 for r in requests if r["low_hit_rate"]),
        "prefill_iters": len(ctx_iters),
        # Over prefilling iterations only, matching what section 3 plots. An
        # aggregated server spends 99.9% of its loop in decode; averaging
        # utilization over those turns 0.136 into 0.0001 and reads as an idle
        # prefill path.
        "prefill_iters_with_work": len(prefilling),
        "prefill_utilization_mean": (
            statistics.fmean([c["utilization"] for c in prefilling
                              if c["utilization"] is not None]) if prefilling else None),
        "prefill_imbalance_mean": (
            statistics.fmean([c["imbalance"] for c in prefilling
                              if c["imbalance"] is not None]) if prefilling else None),
        # 0 evictions is a measurement; only an absent iteration log is missing.
        "kv_evicted_tokens_total": (
            sum(c["kv_evicted_tokens"] or 0 for c in ctx_iters)
            if any(c["kv_evicted_tokens"] is not None for c in ctx_iters) else None),
        "kv_capacity_tokens": capacity.get("capacity_tokens"),
        "kv_capacity_blocks": capacity.get("slots_per_rank"),
        # Peak over the per-rank series, not the rank-mean and not the last
        # row. The pool fills monotonically until it starts evicting, so a mean
        # over time reports roughly half of how full it got; a "last row"
        # depends on which worker lands last in the list; and averaging across
        # ranks hides the one that fills first, which is where eviction starts.
        # Ranks diverge in practice -- 2.8% against 3.9% on the relay pilot.
        "kv_pool_filled_peak": max(
            (c["kv_pool_filled"] for c in (rank_iters or ctx_iters)
             if c.get("kv_pool_filled") is not None), default=None),
        # Cross-tier movement, the three that replace the dead eviction proxy
        # on v2. offload left the GPU but survives on the host tier; onboard is
        # a hit that had to be copied back; host_dropped fell out of the
        # hierarchy altogether and is the only real loss of reusable prefix.
        **{f"kv_{name}_blocks_total": _tier_total(ctx_iters, name)
           for name in ("offload", "onboard", "host_dropped")},
        # The two ratios worth acting on: the share of hits that cost a copy,
        # and the share of written content that was lost outright.
        "kv_onboard_share_of_reuse": _ratio(
            _tier_total(ctx_iters, "onboard"),
            sum(r["isl_cached"] or 0 for r in requests) / 128 or None),
        "decode_iters": len(gen_iters),
        "decode_batch_mean": (
            statistics.fmean([g["decode_batch_total"] for g in gen_iters
                              if g.get("has_decode")] or [0.0]) if gen_iters else None),
        "decode_occupancy_mean": (
            statistics.fmean([g["batch_occupancy"] for g in gen_iters
                              if g.get("has_decode") and g["batch_occupancy"] is not None]
                             or [0.0]) if gen_iters else None),
        "tokens_per_request_mean": (
            statistics.fmean([g["tokens_per_request"] for g in gen_iters
                              if g["tokens_per_request"] is not None] or [0.0])
            if gen_iters else None),
        "decode_paused_total": (
            sum(g["paused_requests"] for g in gen_iters) if gen_iters else None),
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
INK, MUTED, ACCENT, WARN = "#1b2733", "#7a8794", "#3b7dd8", "#c2410c"


def _chart(ctx_iters: list[dict], field: str, title: str, ylabel: str,
           pct: bool = False, origin: float | None = None,
           time_field: str = "timestamp", series_field: str = "worker",
           marker_only: bool = False, toggle: bool = False,
           mean_line: bool = False,
           vlines: list[float] | None = None, vlabel: str | None = None) -> str:
    """One series per prefill instance against wall clock, as an inline PNG.

    Three choices worth stating, because the obvious version of this chart
    lies. The x axis is elapsed seconds, not the iteration counter: iteration
    duration spans three orders of magnitude on agent traffic (median 0.6s,
    p99 580s), so an index axis gives a ten-minute idle the same width as a
    20ms step. The line breaks wherever the gap exceeds ten times the median,
    because a continuous line across a ten-minute silence claims the metric
    held that value while nothing was running. Points are marked, so a burst
    of five events does not read as one thick segment.

    ``toggle`` renders each series as its own transparent overlay above a
    shared axes image, so the legend can switch individual ranks off. Four
    overlapping rank curves are unreadable where they cross, and the question
    asked of these charts is usually about one rank against the others. The
    axes are drawn once and every layer reuses that exact box, so the overlays
    register regardless of how the browser scales them.

    ``mean_line`` draws the series mean as a horizontal rule. On a metric with
    a structural ceiling -- token-budget utilization, or an imbalance pinned at
    ``ranks - 1`` -- the eye tracks the excursions and misreads the level.

    ``vlines`` are absolute timestamps drawn as thin grey rules behind the
    curves, for marking when some event happened against a metric measured
    elsewhere. Thin and translucent on purpose: these arrive in the thousands,
    so any one rule is unreadable and what the layer actually shows is where
    the events bunch up. Overlaying them assumes the two clocks agree --
    request stamps come from the serve edge, the curve from the worker log,
    and _parse_stamp reads the worker's naive stamp as *local* time. On the
    run this was written for the two agreed to within one second because the
    analysis host and the worker share a timezone; on a host that does not,
    this layer silently shifts by the offset. The count is printed on the
    chart so a shifted layer is at least visible as one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    plotted = None
    for row in ctx_iters:
        raw = row.get(time_field)
        stamp = _parse_stamp(raw) if isinstance(raw, str) else _f(raw)
        value = _f(row.get(field))
        if value is None or stamp is None:
            continue
        plotted = stamp if plotted is None else min(plotted, stamp)
        series[row.get(series_field) or ""].append((stamp, value))
    # Anchor on the run, not on the first point that happens to be plotted:
    # otherwise a workload whose first prefill lands ten minutes in reads as
    # starting at zero, and two runs cannot be laid over each other.
    origin = origin if origin is not None else plotted
    if not series:
        return ""

    fig, axis = plt.subplots(figsize=(9, 2.6), dpi=130)
    for stamp in (vlines or []):
        axis.axvline((stamp - origin) / 60.0, color="#94a3b8", linewidth=0.35,
                     alpha=0.30, zorder=0.5)
    colors = [ACCENT, WARN, "#0f766e", "#7c3aed"]
    drawn: list[tuple[str, list[tuple[float, float]]]] = []
    handles: list[Any] = []
    for index, (name, points) in enumerate(sorted(series.items())):
        points.sort()
        points = _downsample(points)
        gaps = [b[0] - a[0] for a, b in zip(points, points[1:])]
        limit = (statistics.median(gaps) * 10) if gaps else None
        xs: list[float | None] = []
        ys: list[float | None] = []
        for offset, (stamp, value) in enumerate(points):
            if offset and limit and (stamp - points[offset - 1][0]) > max(limit, 1.0):
                xs.append(None)
                ys.append(None)
            xs.append((stamp - origin) / 60.0)
            ys.append(value)
        drawn.append((name, [(x, y) for x, y in zip(xs, ys) if x is not None]))
        handles.append(axis.plot(
            xs, ys, linewidth=0 if marker_only else 1.0, label=name,
            color=colors[index % len(colors)],
            marker="o", markersize=2.6 if marker_only else 1.8, markevery=1)[0])
    mean_value = None
    if mean_line:
        flat = [y for _, pts in drawn for _, y in pts]
        if flat:
            mean_value = statistics.fmean(flat)
            axis.axhline(mean_value, color=INK, linestyle=(0, (5, 3)),
                         linewidth=0.9, zorder=1.5)
            axis.annotate(f"mean {mean_value * 100:.1f}%" if pct
                          else f"mean {mean_value:,.3g}",
                          xy=(1.0, mean_value), xycoords=("axes fraction", "data"),
                          xytext=(-3, 3), textcoords="offset points",
                          ha="right", va="bottom", fontsize=7, color=INK)
    axis.set_title(title, fontsize=10, color=INK, loc="left")
    if vlines and vlabel:
        axis.annotate(f"{len(vlines):,} {vlabel}", xy=(0.0, 1.0),
                      xycoords="axes fraction", xytext=(2, -3),
                      textcoords="offset points", ha="left", va="top",
                      fontsize=7, color="#94a3b8")
    axis.set_xlabel("minutes into the run", fontsize=8, color=MUTED)
    axis.set_ylabel(ylabel, fontsize=8, color=MUTED)
    axis.tick_params(labelsize=7, colors=MUTED)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#d8dee5")
    axis.grid(axis="y", color="#eef1f4", linewidth=0.8)
    axis.set_axisbelow(True)
    if pct:
        axis.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    if len(series) > 1:
        axis.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    # Read the axes box and limits *after* layout so a cursor position can be
    # mapped back to a value. Cheaper than an SVG carrying one <title> per
    # point, and the rendered image is untouched.
    box = axis.get_position()
    payload = {
        "box": [box.x0, 1.0 - box.y1, box.width, box.height],
        "xlim": list(axis.get_xlim()),
        "ylim": list(axis.get_ylim()),
        "pct": bool(pct),
        "series": [{"name": name,
                    "pts": [[round(x, 3), round(y, 5)] for x, y in pts]}
                   for name, pts in drawn],
    }
    def render(transparent: bool = False) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", transparent=transparent)
        return base64.b64encode(buffer.getvalue()).decode()

    if not (toggle and len(drawn) > 1):
        image = render()
        plt.close(fig)
        return ('<figure class="chart"><img alt="%s" src="data:image/png;base64,%s">'
                '<script type="application/json">%s</script></figure>'
                % (title, image, json.dumps(payload, separators=(",", ":"))))

    # Layered: the axes once, then one transparent overlay per series. The
    # limits are frozen first so every layer maps data to pixels identically --
    # without that, matplotlib re-autoscales to whichever single series is
    # visible and the overlays no longer line up with the axes beneath them.
    axis.set_xlim(*payload["xlim"])
    axis.set_ylim(*payload["ylim"])
    legend = axis.get_legend()
    if legend is not None:
        legend.set_visible(False)
    for handle in handles:
        handle.set_visible(False)
    base = render()

    # set_title(..., loc="left") stores the text on the left-hand title artist,
    # not on ``axis.title``, so clearing that one leaves the heading painted
    # onto every overlay and it stacks up four deep.
    axis.set_title("", loc="left")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)
    axis.tick_params(labelleft=False, labelbottom=False, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    mean_artists = [a for a in axis.lines if a not in handles] + list(axis.texts)
    for artist in mean_artists:
        artist.set_visible(False)

    layers = []
    for handle, (name, _) in zip(handles, drawn):
        handle.set_visible(True)
        layers.append((name, handle.get_color(), render(transparent=True)))
        handle.set_visible(False)
    plt.close(fig)

    chips = "".join(
        f'<button class="chip on" data-series="{name}">'
        f'<span style="background:{color}"></span>{name}</button>'
        for name, color, _ in layers)
    stack = "".join(
        f'<img class="layer" data-series="{name}" alt="" '
        f'src="data:image/png;base64,{data}">' for name, _, data in layers)
    return ('<figure class="chart layered">'
            '<div class="stack"><img class="base" alt="%s" '
            'src="data:image/png;base64,%s">%s</div>'
            '<div class="legend">%s</div>'
            '<script type="application/json">%s</script></figure>'
            % (title, base, stack, chips,
               json.dumps(payload, separators=(",", ":"))))


def _annotate_thread_gaps(requests: list[dict], threads: dict | None) -> int:
    """Per request, wall clock until the next turn *of its own thread* starts.

    Not "the next request in the session". A session id is a client handle,
    not a conversation: several threads run under one at the same time, plus
    the subagents they launch. Ordering a session by start time and
    differencing neighbours therefore pairs turns from unrelated threads, and
    on this run 5,024 of 13,211 such gaps came out *negative* -- the next
    request to start had started before this one finished, because it belonged
    to a thread that was already running. That column is kept as
    ``legacy_gap_to_next_in_session_ms``.

    This one takes the earliest child in the rebuilt thread, so it is a real
    interval by construction (build_threads only links a parent that finished
    before its child started, so no negatives exist). Where a turn fans out to
    several children -- a parallel tool batch, or a subagent launch -- the
    earliest is the one that ends the idle period.

    A negative value here is not a long tail, it is a broken parent link, so
    they are dropped rather than averaged in and the count is returned for the
    caller to surface. On a healthy rebuild it is 0.
    """
    kids: dict[str, list[dict]] = defaultdict(list)
    for node in (threads or {}).values():
        if node.get("parent"):
            kids[node["parent"]].append(node)
    gaps, negative = {}, 0
    for parent, group in kids.items():
        first = min(group, key=lambda n: _f(n.get("started_at")) or 0.0)
        value = _f(first.get("true_gap_ms"))
        if value is None:
            continue
        if value < 0:
            negative += 1
            continue
        gaps[parent] = value
    for row in requests:
        row["gap_to_child_ms"] = gaps.get(row.get("audit_request_id"))
    return negative


def _tok(value: float) -> str:
    return f"{value:,.0f}"


def _ratio_fmt(value: float) -> str:
    return f"{value:,.2f}" if value < 100 else f"{value:,.0f}"


def _sec(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 60:
        return f"{value:.1f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.1f}h"


def _distributions(panels: list[tuple[str, list[float], str, Any]],
                   cols: int = 3) -> str:
    """One log-x histogram per metric, as a single inline PNG.

    One panel each rather than one shared axis: these are tokens, a bare ratio
    and a duration, and nothing makes them comparable on one y. Log x on all of
    them because each spans three decades or more -- gap to next turn runs 49ms
    to 2.5h -- and on a linear axis the entire distribution lands in the first
    pixel column with a lone outlier at the far edge, which is the shape of the
    axis, not of the data.

    Percentile rules rather than a summary table because the question these
    answer is where the mass sits, not what the mean is: every one of these is
    heavy-tailed enough that the mean falls above p50 and describes no actual
    request. The mean is printed anyway, next to p50, so the size of that gap
    is visible instead of implied.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live = [(label, [v for v in values if v is not None and v > 0], xlabel, fmt)
            for label, values, xlabel, fmt in panels]
    live = [panel for panel in live if len(panel[1]) > 1]
    if not live:
        return ""
    rows = math.ceil(len(live) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9, 2.1 * rows + 0.3), dpi=130)
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for axis in flat[len(live):]:
        axis.set_visible(False)
    for axis, (label, values, xlabel, fmt) in zip(flat, live):
        lo, hi = min(values), max(values)
        if hi <= lo:
            hi = lo * 1.1
        low, high = math.log10(lo), math.log10(hi)
        edges = [10 ** (low + (high - low) * i / 40) for i in range(41)]
        axis.hist(values, bins=edges, color=ACCENT, alpha=0.75, linewidth=0)
        axis.set_xscale("log")
        stat = _stats(values)
        for q, color, style in ((stat["p50"], INK, (0, (4, 2))),
                                (stat["p90"], WARN, (0, (2, 2))),
                                (stat["p99"], WARN, (0, (1, 2)))):
            if q:
                axis.axvline(q, color=color, linewidth=0.9, linestyle=style,
                             zorder=3)
        axis.set_title(label, fontsize=9, color=INK, loc="left")
        axis.set_xlabel(xlabel, fontsize=7, color=MUTED)
        axis.tick_params(labelsize=6.5, colors=MUTED)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#d8dee5")
        axis.grid(axis="y", color="#eef1f4", linewidth=0.8)
        axis.set_axisbelow(True)
        text = "\n".join([
            f'n {stat["n"]:,}',
            f'p50 {fmt(stat["p50"])}   mean {fmt(stat["mean"])}',
            f'p90 {fmt(stat["p90"])}',
            f'p99 {fmt(stat["p99"])}',
        ])
        # Headroom before the annotation, not after: the tallest bar is often
        # under the top-right corner where the numbers go, and a bbox alone
        # then hides the mode it is describing.
        top = axis.get_ylim()[1]
        axis.set_ylim(top=top * 1.42)
        axis.text(0.97, 0.96, text, transform=axis.transAxes, ha="right",
                  va="top", fontsize=6.5, color=MUTED, linespacing=1.5,
                  bbox={"facecolor": "white", "edgecolor": "none",
                        "alpha": 0.82, "pad": 1.5}, zorder=4)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return ('<figure class="dist"><img alt="request metric distributions" '
            'src="data:image/png;base64,%s"></figure>'
            % base64.b64encode(buffer.getvalue()).decode())


# Idle to saturated. Steps rather than a continuous ramp because the counts
# that matter are small integers -- most sessions never exceed two concurrent
# requests -- and a continuous scale spends its whole range separating values
# that occur once.
CONCURRENCY_SHADES = ["#eef2f6", "#bcd4f0", "#8ab4e6", "#5a95dc",
                      "#3b7dd8", "#1e4f8f"]


def _inflight(intervals: list[tuple[float, float]]
              ) -> list[tuple[float, float, int]]:
    """Step function of how many intervals are open, as (from, to, count)."""
    events: list[tuple[float, int]] = []
    for begin, finish in intervals:
        events.append((begin, 1))
        events.append((finish, -1))
    events.sort()
    out: list[tuple[float, float, int]] = []
    count, previous = 0, None
    for stamp, delta in events:
        if previous is not None and stamp > previous:
            out.append((previous, stamp, count))
        count += delta
        previous = stamp
    return out


def _request_chart(requests: list[dict], sessions: list[dict],
                   origin: float | None) -> str:
    """Session spans over wall clock, shaded by that session's own concurrency.

    One bar per session, positioned and sized by real time, so overlapping
    sessions are visible as bars sharing an x range. The shading along a bar is
    how many of that session's requests were in flight at that moment: pale
    where the session is open but idle -- the client running a tool, which on
    agent traffic is most of a session -- and darker where it had work on the
    server. Without it a session reads as continuously busy for its whole span.

    The panel below counts requests in flight across all sessions, which is the
    load the server actually saw, and shares the x axis so a peak can be traced
    back to the sessions that caused it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if origin is None:
        return ""
    by_session: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in requests:
        if (row.get("started_at") is not None
                and row.get("finished_at") is not None):
            by_session[row["session_id"]].append(
                (row["started_at"], row["finished_at"]))

    lanes = []
    for row in sessions:
        spans = by_session.get(row["session_id"])
        if spans:
            lanes.append((min(b for b, _ in spans), row, spans))
    if not lanes:
        return ""
    lanes.sort(key=lambda item: item[0])

    everything = [span for _, _, spans in lanes for span in spans]
    last = max(finish for _, finish in everything)
    unit, label = ((3600.0, "hours") if last - origin > 3 * 3600
                   else (60.0, "minutes"))
    width = (last - origin) / unit

    rows = len(lanes)
    top_height = min(20.0, max(1.4, 0.17 * rows))
    figure_height = top_height + 2.0
    fig, (axis, load) = plt.subplots(
        2, 1, figsize=(9, figure_height), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [top_height, 1.7], "hspace": 0.08})

    thinnest = width * 0.0015
    for y, (_, row, spans) in enumerate(lanes):
        for begin, finish, count in _inflight(spans):
            shade = CONCURRENCY_SHADES[min(count, len(CONCURRENCY_SHADES) - 1)]
            axis.barh(y, max((finish - begin) / unit, thinnest),
                      left=(begin - origin) / unit, height=0.74,
                      color=shade, linewidth=0)
        axis.text(width * 1.008, y, f'{row["turns"]}', va="center",
                  fontsize=6, color=MUTED)
    axis.set_ylim(rows - 0.4, -0.6)
    step = max(1, rows // 30)
    axis.set_yticks(range(0, rows, step))
    axis.set_yticklabels([f"s{i + 1}" for i in range(0, rows, step)], fontsize=6)
    axis.set_title("Session spans — shaded by that session's concurrent "
                   "requests (request count at the right)",
                   fontsize=10, color=INK, loc="left", pad=20)
    axis.legend(handles=[Patch(facecolor=CONCURRENCY_SHADES[0], label="idle")]
                + [Patch(facecolor=CONCURRENCY_SHADES[i], label=str(i))
                   for i in range(1, len(CONCURRENCY_SHADES) - 1)]
                + [Patch(facecolor=CONCURRENCY_SHADES[-1],
                         label=f"{len(CONCURRENCY_SHADES) - 1}+")],
                ncol=len(CONCURRENCY_SHADES), fontsize=6, frameon=False,
                loc="lower left", bbox_to_anchor=(0, 1.0),
                handlelength=1.2, columnspacing=1.0)

    steps = _inflight(everything)
    xs = [(begin - origin) / unit for begin, _, _ in steps] + [width]
    ys = [count for _, _, count in steps] + [0]
    load.fill_between(xs, ys, step="post", color=ACCENT, alpha=0.28, linewidth=0)
    load.step(xs, ys, where="post", color=ACCENT, linewidth=0.9)
    peak = max(ys)
    load.set_ylim(0, peak * 1.15 or 1)
    load.set_ylabel("in flight", fontsize=7, color=MUTED)
    load.set_xlabel(f"{label} since the first request", fontsize=8, color=MUTED)
    load.set_xlim(0, width * 1.02)
    load.text(0.004, 0.9, f"peak {peak}", transform=load.transAxes,
              fontsize=7, color=MUTED, va="top")

    for target in (axis, load):
        target.tick_params(labelsize=7, colors=MUTED)
        for spine in ("top", "right", "left"):
            target.spines[spine].set_visible(False)
        target.spines["bottom"].set_color("#d8dee5")
        target.grid(axis="x", color="#eef1f4", linewidth=0.8)
        target.set_axisbelow(True)
    # Explicit margins rather than tight_layout or bbox_inches. Both would
    # leave the axes box unknown -- one warns because the legend sits outside
    # the axes, the other crops the canvas afterwards -- and the box is what
    # maps a cursor position back to a time.
    fig.subplots_adjust(left=0.075, right=0.955, hspace=0.08,
                        top=1.0 - 0.62 / figure_height,
                        bottom=0.42 / figure_height)
    upper, lower = axis.get_position(), load.get_position()
    payload = {
        "box": [upper.x0, 1.0 - upper.y1, upper.width, upper.y1 - lower.y0],
        "xlim": [0.0, width * 1.02],
        "unit": "h" if unit == 3600.0 else "min",
        # Read back at the cursor instead of a nearest-point search: this chart
        # draws spans, not points, and the question it answers is "how many
        # requests were in flight at this instant".
        "xonly": True,
        "steps": [[round((begin - origin) / unit, 4), count]
                  for begin, _, count in steps],
    }
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return ('<figure class="chart">'
            '<img alt="session spans and in-flight requests" '
            'src="data:image/png;base64,%s">'
            '<script type="application/json">%s</script></figure>'
            % (base64.b64encode(buffer.getvalue()).decode(),
               json.dumps(payload, separators=(",", ":"))))


def _parse_stamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def _downsample(points: list[tuple[float, float]], budget: int = 1500
                ) -> list[tuple[float, float]]:
    """Thin a dense series by averaging within equal time windows.

    Only dense series need this -- a few hundred prefill events are plotted as
    they are. Windows are cut on time rather than on position so a burst and a
    silence are not averaged together.
    """
    if len(points) <= budget:
        return points
    span = points[-1][0] - points[0][0]
    if span <= 0:
        return points[:budget]
    width = span / budget
    out: list[tuple[float, float]] = []
    bucket: list[tuple[float, float]] = []
    edge = points[0][0] + width
    for point in points:
        if point[0] > edge and bucket:
            out.append((statistics.fmean(p[0] for p in bucket),
                        statistics.fmean(p[1] for p in bucket)))
            bucket = []
            edge = point[0] + width
        bucket.append(point)
    if bucket:
        out.append((statistics.fmean(p[0] for p in bucket),
                    statistics.fmean(p[1] for p in bucket)))
    return out


def _num(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None or value == "":
        return '<span class="na">—</span>'
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value:,}{suffix}" if isinstance(value, int) else str(value)


def _table(rows: list[dict], columns: list[tuple[str, str]],
           flag: str | None = None) -> str:
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body = []
    for row in rows:
        cls = ' class="total"' if row.get("_total") else (
            ' class="flag"' if flag and row.get(flag) else "")
        cells = "".join(
            f"<td>{_num(row.get(key), 3 if 'rate' in key or key in ('utilization', 'imbalance') else 1)}</td>"
            for key, _ in columns)
        body.append(f"<tr{cls}>{cells}</tr>")
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _dur(ms: float | None) -> str:
    """Milliseconds below a second, seconds above it.

    Agent traffic spans five orders of magnitude in one column -- a 12ms decode
    step next to a 69,000ms TTFT -- and reading which is which off raw
    millisecond counts is needless work.
    """
    if ms is None:
        return '<span class="na">—</span>'
    return f"{ms / 1000:,.2f} s" if abs(ms) >= 1000 else f"{ms:,.0f} ms"


def _capacity_total(rank_iters: list[dict]) -> float | None:
    """Total KV pages across every rank of every prefill instance.

    Summed over *distinct* ranks, not over rows: capacity is a level repeated
    on every iteration, so summing the column would multiply it by the
    iteration count. Keyed on the instance too, for the same reason
    :func:`_pool_capacity` is -- one log can hold more than one engine
    lifetime, and the pool-sizing pass runs a much smaller pool.

    An instance with a single logged iteration is left out, because the
    numerator this denominates cannot contain it: the tier counters are
    cumulative and differenced *within* an instance, so one iteration yields
    no delta. Counting its pool anyway inflates the denominator by a pool that
    never moved a page -- measured on the DeepSeek pilot, the sizing pass added
    35,600 slots to a 157,240-slot run and pulled every share down by a fifth.
    """
    seen: dict[tuple, tuple[set, float]] = {}
    for r in rank_iters:
        cap = r.get("kv_capacity_blocks")
        if cap is None:
            continue
        key = (r.get("worker"), r.get("instance"), r.get("rank"))
        iters, _ = seen.setdefault(key, (set(), cap))
        iters.add(r.get("iter"))
    return sum(cap for iters, cap in seen.values() if len(iters) > 1) or None


def _tier_table(ctx_iters: list[dict], rank_iters: list[dict]) -> str:
    """Cross-tier page movement over the run, as totals rather than a spread.

    These are cumulative counters, and after differencing they are rates that
    are zero on almost every iteration -- a p50/p90/p99 over them would say
    nothing. What matters is the run total, and each total against the pool it
    moved through: how much left the GPU, how much was copied back, and how
    much fell out of the hierarchy, which is the only real loss of reusable
    prefix.

    Empty on any run without the counters -- they arrived with the v2 manager,
    and before them the report has nothing to say here rather than zero.
    """
    totals = {name: _tier_total(ctx_iters, name)
              for name in ("offload", "onboard", "host_dropped")}
    if all(v is None for v in totals.values()):
        return ""
    # Every count here is a page -- ``_record_migrated_slots`` and
    # ``_record_dropped_pages`` both increment once per page -- so the
    # denominator has to be pages too, or the quotient is not a share of
    # anything. Pool capacity is the only page-denominated total the run
    # offers. The engine records no "pages written": ``iter_alloc_new_blocks``
    # counts pages on the migration path and logical blocks on the main
    # allocation path, and ``ctx_blocks_new`` / ``isl_cached`` are logical
    # blocks and tokens, which convert to pages at no fixed rate -- a block's
    # data occupies a page in each pool group it touches, and the groups differ
    # in both size and count. Dividing a page count by either overstated the
    # share by that unknown factor, which is what this replaces.
    capacity = _capacity_total(rank_iters)
    shares = {name: _ratio(totals[name], capacity) for name in totals}
    filled = [r["kv_pool_filled"] for r in rank_iters
              if r.get("kv_pool_filled") is not None]
    quiet = all((v or 0) == 0 for v in totals.values())

    def row(label: str, key: str) -> str:
        share = shares[key]
        return (f"<tr><td>{label}</td><td>{_num(totals[key], 0)}</td>"
                f'<td class="lead">{_num(share and share * 100, 2, "%")}'
                f"{' of pool capacity' if share is not None else ''}</td></tr>")

    return (
        "<table><thead><tr><th>cross-tier movement</th><th>pages</th>"
        "<th>share</th></tr></thead><tbody>"
        + row("Offloaded GPU → host", "offload")
        + row("Onboarded host → GPU", "onboard")
        + row("Dropped out of the hierarchy", "host_dropped")
        + "</tbody></table>"
        '<p class="sub">'
        "<b>Onboarded</b> is the cost a healthy hit rate hides: a prefix served "
        "from the host tier is still a hit, but it is paid for with a copy back "
        "onto the GPU before the prefill can run. <b>Dropped</b> is the only one "
        "of the three that loses reusable content outright, and the one a falling "
        "hit rate can be charged to; offloading merely moves it down a tier. "
        "All three are counted <b>per page</b>, one page per pool-group slot, "
        "so the share is against the run's total pool capacity in pages — the "
        "only page-denominated total there is. It is cumulative movement over "
        "a fixed pool, so it reads as turnover: 100% would mean the run moved "
        "a pool's worth of pages, not that the pool was full. "
        "These replace <code>Evicted tokens</code> above, which is "
        "<code>alloc_total − alloc_new</code> — a v1 signal that the v2 manager "
        "sets to zero by construction, since every call site assigns both "
        "counters the same value."
        + (" All three are zero on this run"
           + (f", and the pool only reached "
              f"{max(filled) * 100:.1f}% full, so nothing had to be reclaimed."
              if filled else ".")
           if quiet else "")
        + "</p>")


def _rank_hit(acc: dict) -> float | None:
    """A rank's whole-run hit rate, preferring the log's own counter.

    Last, not mean: the counters are cumulative, so the final reading already
    is the rate over the whole run, and averaging a running total would report
    roughly where it sat halfway. Logs predating the field fall back to the
    mean of the differenced curve, which is all they carry.
    """
    if acc.get("hit"):
        return acc["hit"][-1]
    if acc.get("hit_delta"):
        return statistics.fmean(acc["hit_delta"])
    return None


def _rank_totals(rank_iters: list[dict]) -> str:
    """Cumulative per-rank prefill share over the whole run.

    The per-iteration imbalance column cannot answer "is one rank overloaded".
    Under attention-DP a prefill lands on exactly one rank, so with n ranks
    every prefilling iteration scores (max - mean) / mean = n - 1 whatever the
    routing does -- that number is structural, not skew. The skew is here: how
    the iterations and tokens divided across ranks once the run is over.
    """
    per: dict[tuple[str, int], dict] = {}
    for row in rank_iters:
        if not row.get("num_ctx_tokens") or row.get("is_adp_pad"):
            # The attention-DP pad is a full-budget dummy handed to an idle
            # rank; counting it would make the least busy rank look busiest.
            continue
        acc = per.setdefault((row["worker"], row["rank"]),
                             {"iters": 0, "tokens": 0, "hit": [], "util": []})
        acc["iters"] += 1
        acc["tokens"] += row["num_ctx_tokens"]
        for key, field in (("hit", "kv_hit_rate_cum"),
                           ("hit_delta", "kv_hit_rate_iter"),
                           ("util", "kv_cache_util"),
                           ("filled", "kv_pool_filled")):
            if row.get(field) is not None:
                acc.setdefault(key, []).append(row[field])
    if not per:
        return ""
    total = sum(a["tokens"] for a in per.values()) or 1
    body = []
    for (worker, rank), acc in sorted(per.items()):
        share = acc["tokens"] / total
        peak_filled = max(acc["filled"]) if acc.get("filled") else None
        body.append(
            f"<tr><td>{worker} r{rank}</td><td>{acc['iters']:,}</td>"
            f"<td>{acc['tokens']:,}</td>"
            f'<td class="imb">{share * 100:.1f}%</td>'
            # Last, not mean: the counters are cumulative, so the final
            # reading already is the rank's whole-run rate. Averaging a running
            # total would instead report roughly where it sat halfway.
            f'<td class="lead">{_num(_rank_hit(acc), 3)}</td>'
            f'<td>{_num(statistics.fmean(acc["util"]) if acc["util"] else None, 3)}</td>'
            # Peak, not mean: the pool fills monotonically, so a mean over the
            # run would report roughly half of how full it actually got.
            f'<td>{_num(peak_filled and peak_filled * 100, 1, "%")}</td></tr>')
    shares = [a["tokens"] / total for a in per.values()]
    skew = max(shares) / min(shares) if min(shares) else None
    return ("<table><thead><tr><th>rank</th><th>prefill iters</th>"
            "<th>ctx tokens</th>"
            '<th class="imb">share</th><th>hit rate</th><th>pinned</th>'
            "<th>pool filled</th>"
            f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
            f'<p class="sub">Cumulative over the run, prefilling iterations only. '
            "An even split is 1/ranks each; this run's busiest rank took "
            f"{_num(skew, 1)}× the quietest. This is the routing skew — the "
            "per-iteration imbalance column above cannot show it, because under "
            "attention-DP one prefill occupies one rank and every prefilling "
            "iteration therefore scores <code>ranks − 1</code> regardless of how "
            "well the router balanced.</p>")


def _stat_row(label: str, values: Iterable[float | None], digits: int = 1,
              duration: bool = False, shares: list[float] | None = None,
              spread: Iterable[float | None] | None = None,
              with_spread: bool = False) -> str:
    """One metric row; mean and p50 are emphasised because they are what the
    eye goes to first. Ratios need more than one decimal: a utilization of
    0.036 rendered at one decimal reads as 0.0 and looks like no data.

    ``with_spread`` adds an imbalance column straight after the mean, because
    that pair is what has to be read together: a mean says how much of the
    resource was used, the spread says whether one rank carried it alone.
    """
    s = _stats(values)
    share = _stats(shares) if shares else None
    cells = ""
    for key in ("mean", "p50", "p90", "p99"):
        text = _dur(s[key]) if duration else _num(s[key], digits)
        if share and share[key] is not None:
            text += f' <span class="pct">({share[key] * 100:.0f}%)</span>'
        cells += (f'<td class="{"lead" if key in ("mean", "p50") else ""}">'
                  f'{text}</td>')
        if key == "mean" and with_spread:
            imb = _stats(spread)["mean"] if spread is not None else None
            cells += f'<td class="imb">{_num(imb, 3)}</td>'
    return f"<tr><td>{label}</td><td>{s['n']}</td>{cells}</tr>"


CHART_JS = r'''<script>
// Charts are rendered as images; the axes box and the drawn points travel
// alongside as JSON so a cursor can be mapped back to a value without shipping
// an SVG that carries one <title> per point.
document.querySelectorAll('figure.chart').forEach(function (fig) {
  var img = fig.querySelector('img');
  var data = JSON.parse(fig.querySelector('script').textContent);
  var tip = document.createElement('div'); tip.className = 'tip';
  var rule = document.createElement('div'); rule.className = 'rule';
  fig.appendChild(rule); fig.appendChild(tip);
  // Layered charts: the legend switches overlays on and off. `hidden` is what
  // keeps the readout honest -- a cursor must not report a series the reader
  // has just switched off, because the nearest point would then come from a
  // curve that is not on screen.
  var hidden = {};
  fig.querySelectorAll('.chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      var name = chip.dataset.series;
      var on = chip.classList.toggle('on');
      hidden[name] = !on;
      fig.querySelectorAll('.layer').forEach(function (el) {
        if (el.dataset.series === name) el.classList.toggle('off', !on);
      });
    });
  });
  var fmt = function (v) {
    return data.pct ? (v * 100).toFixed(1) + '%'
                    : (Math.abs(v) >= 1000 ? v.toLocaleString(undefined, {maximumFractionDigits: 0})
                                           : v.toPrecision(3));
  };
  img.addEventListener('mousemove', function (ev) {
    var r = img.getBoundingClientRect();
    var fx = (ev.clientX - r.left) / r.width;
    var inner = (fx - data.box[0]) / data.box[2];
    if (inner < 0 || inner > 1) { tip.classList.remove('on'); rule.classList.remove('on'); return; }
    var x = data.xlim[0] + inner * (data.xlim[1] - data.xlim[0]);
    if (data.xonly) {
      var live = 0;
      for (var i = 0; i < data.steps.length; i++) {
        if (data.steps[i][0] <= x) live = data.steps[i][1]; else break;
      }
      var cx = (data.box[0] + inner * data.box[2]) * r.width;
      tip.textContent = x.toFixed(1) + ' ' + data.unit + '   \u00b7   '
                      + live + ' in flight';
      tip.classList.add('on'); rule.classList.add('on');
      // Follow the cursor instead of anchoring to the top of the axes box.
      // This panel's height scales with the session count -- 396 sessions
      // render a 2860px-tall figure -- and the in-flight strip being read is
      // its bottom sliver, so a fixed top anchor put the number thousands of
      // pixels above the cursor, off screen. Measured against the figure, not
      // the image: the image carries a 6px top margin inside it.
      // offsetWidth is read after the text is set, so the clamp uses the width
      // this readout actually has rather than the previous one's.
      var fr = fig.getBoundingClientRect();
      var half = tip.offsetWidth / 2;
      tip.style.left = Math.max(half, Math.min(r.width - half, cx)) + 'px';
      tip.style.top = (ev.clientY - fr.top) + 'px';
      rule.style.left = cx + 'px';
      return;
    }
    var best = null;
    data.series.forEach(function (s) {
      if (hidden[s.name]) return;
      s.pts.forEach(function (p) {
        var d = Math.abs(p[0] - x);
        if (!best || d < best.d) best = {d: d, x: p[0], y: p[1], name: s.name};
      });
    });
    if (!best) { tip.classList.remove('on'); rule.classList.remove('on'); return; }
    var px = (data.box[0] + (best.x - data.xlim[0]) / (data.xlim[1] - data.xlim[0]) * data.box[2]) * r.width;
    var py = (data.box[1] + (data.ylim[1] - best.y) / (data.ylim[1] - data.ylim[0]) * data.box[3]) * r.height;
    tip.textContent = (data.series.length > 1 ? best.name + '  ' : '')
                    + fmt(best.y) + '   @ ' + best.x.toFixed(1) + ' min';
    tip.style.left = px + 'px'; tip.style.top = py + 'px';
    rule.style.left = px + 'px';
    tip.classList.add('on'); rule.classList.add('on');
  });
  img.addEventListener('mouseleave', function () {
    tip.classList.remove('on'); rule.classList.remove('on');
  });
});
</script>'''

CSS = """
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
color:#1b2733;margin:0;padding:32px 40px;max-width:1180px;background:#fff}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:34px 0 10px;
padding-bottom:6px;border-bottom:1px solid #e6eaee}
h3{font-size:13px;margin:20px 0 6px;color:#7a8794;font-weight:600;
text-transform:uppercase;letter-spacing:.04em}
.sub{color:#7a8794;font-size:12px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 4px}
th{text-align:right;color:#7a8794;font-weight:600;padding:5px 8px;
border-bottom:1px solid #e6eaee;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:4px 8px;border-bottom:1px solid #f2f5f7;
font-variant-numeric:tabular-nums}
tr.flag td{background:#fff5ed}
td.lead{font-weight:650;color:#0f1c2b}
td.imb,th.imb{background:#f2f6fb;font-weight:650;color:#0f1c2b;border-left:1px solid #dde5ef;border-right:1px solid #dde5ef}
span.pct{color:#7a8794;font-weight:400}
details{margin:22px 0 0}summary{cursor:pointer;font-size:16px;font-weight:600;
padding-bottom:6px;border-bottom:1px solid #e6eaee;color:#1b2733}
summary::marker{color:#7a8794}
tr.total td{border-top:2px solid #d8dee5;font-weight:650;background:#fafbfc}
.na{color:#c3cad2}
.kpi{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
.kpi div{border:1px solid #e6eaee;border-radius:6px;padding:9px 14px;min-width:132px}
.kpi b{display:block;font-size:19px;font-variant-numeric:tabular-nums}
.kpi span{font-size:11px;color:#7a8794}
.note{background:#f7f9fb;border-left:3px solid #3b7dd8;padding:9px 13px;
margin:12px 0;font-size:12px;color:#41505f}
img{max-width:100%;display:block;margin:6px 0}
figure.chart{margin:6px 0;position:relative}
figure.dist{margin:10px 0}
figure.chart .stack{position:relative}
figure.chart .stack img{display:block;width:100%}
figure.chart .stack img.layer{position:absolute;left:0;top:0;pointer-events:none}
figure.chart .stack img.layer.off{display:none}
figure.chart .legend{display:flex;flex-wrap:wrap;gap:4px;margin:2px 0 0 4px}
.chip{display:inline-flex;align-items:center;gap:4px;font:inherit;font-size:11px;
color:#7a8794;background:#f7f9fb;border:1px solid #e3e8ee;border-radius:11px;
padding:1px 8px;cursor:pointer;font-variant-numeric:tabular-nums}
.chip span{width:8px;height:8px;border-radius:50%;opacity:.28}
.chip.on{color:#1b2733;background:#fff;border-color:#c3cad2}
.chip.on span{opacity:1}
.tip{position:absolute;pointer-events:none;background:#1b2733;color:#fff;
font-size:11px;padding:3px 7px;border-radius:4px;white-space:nowrap;
transform:translate(-50%,-140%);opacity:0;transition:opacity .08s;
font-variant-numeric:tabular-nums;z-index:5}
.tip.on{opacity:1}
.rule{position:absolute;pointer-events:none;top:0;bottom:0;width:1px;
background:#c3cad2;opacity:0}
.rule.on{opacity:1}
code{background:#f2f5f7;padding:1px 4px;border-radius:3px;font-size:11px}
"""


def rank_thread_assignment(requests: list[dict], threads: dict) -> dict | None:
    """Which rank each conversation landed on, and why.

    A thread is pinned: 94% of them never leave the rank their opening turn was
    routed to, because every later turn carries the whole previous prompt and
    that rank's match is unbeatable. So the root's routing decision, taken once,
    decides where the thread's entire compute goes -- which is why this is
    keyed on roots and totals the whole chain's `isl_new` rather than the root's.
    """
    if not threads:
        return None
    kinds, chain, roots = {}, defaultdict(float), {}
    for row in requests:
        node = threads.get(row["audit_request_id"])
        if not node:
            continue
        tid = node["thread_id"]
        chain[tid] += row.get("isl_new") or 0
        if str(node.get("thread_depth")) == "1":
            roots[tid] = row
            kinds[tid] = node.get("kind") or "unknown"
    by_rank: dict[str, Counter] = defaultdict(Counter)
    reason: dict[str, Counter] = defaultdict(Counter)
    tokens: dict[str, float] = defaultdict(float)
    for tid, row in roots.items():
        rank = row.get("routed_rank")
        if rank in (None, ""):
            continue
        rank = str(rank)
        by_rank[rank][kinds.get(tid) or "unknown"] += 1
        tokens[rank] += chain[tid]
        reason[rank][row.get("route_reason") or "no routing record"] += 1
    if not by_rank:
        return None
    return {"kinds": by_rank, "tokens": tokens, "reason": reason,
            "roots": sum(sum(c.values()) for c in by_rank.values())}


def _cross_thread_reuse(report_dir: Path | None) -> dict | None:
    """`cross_thread_reuse.json`, if that analysis has been run.

    It is a separate script because splitting boilerplate reuse from real
    reuse means reading the captured bodies, which this report never does.
    """
    if report_dir is None:
        return None
    path = report_dir / "cross_thread_reuse.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _probe(encoded: str | None) -> dict[str, float]:
    """`route_match_lens` back into {rank: matched tokens}."""
    out = {}
    for pair in (encoded or "").split("|"):
        rank, _, value = pair.partition(":")
        if value:
            try:
                out[rank] = float(value)
            except ValueError:
                pass
    return out


def kv_miss_cost(requests: list[dict], threads: dict) -> dict | None:
    """What each cause of a KV miss actually costs, as a share of the run.

    For one parent -> child transition the tokens the engine computed split
    exactly two ways::

        child.isl_new = (child.isl_total - parent.isl_total)   new content
                      + (parent.isl_total - child.isl_cached)  recomputed

    The first term is content that never existed before and had to be
    computed. The second is prefix the cache already held on the previous turn
    and failed to serve -- the only part any optimisation could recover. This
    reports that second term per cause, against two denominators: every token
    the run computed, and every millisecond it spent in prefill.

    Prefill time is attributed pro rata on tokens (`prefill_ms * missed /
    isl_new`), which assumes prefill cost is linear in tokens computed. That
    holds well under chunked prefill and is stated on the table rather than
    hidden.

    The point of the table is the denominator. A cause worth a fraction of a
    percent of prefill is not worth engineering effort however unpleasant it
    looks per request, and that is a decision this makes on evidence.
    """
    if not threads:
        return None
    by_aid = {r["audit_request_id"]: r for r in requests}
    buckets: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "tok": 0.0, "ms": 0.0, "hit": [], "aids": [],
                 "missed": {}})
    total_new = total_prefill = 0.0
    classified = 0
    for row in requests:
        new, prefill = _f(row.get("isl_new")), _f(row.get("prefill_ms"))
        total_new += new or 0.0
        total_prefill += prefill or 0.0
        node = threads.get(row["audit_request_id"])
        if not node or not node.get("parent"):
            continue
        parent = by_aid.get(node["parent"])
        pnode = threads.get(node["parent"])
        if parent is None or pnode is None:
            continue
        pisl, cached = _f(parent.get("isl_total")), _f(row.get("isl_cached"))
        if pisl is None or cached is None:
            continue
        classified += 1
        # The ceiling is what the child actually SHARED with the parent, not
        # the parent's whole prompt. Those differ: DeepSeek-V4's template
        # re-roles a trailing system message and defers the generation prompt
        # past it, so a parent that ended on one is not a prefix of its child
        # at all -- measured, the shared prefix stops a median 305 tokens
        # short. Charging the difference to the cache blamed it for tokens the
        # child never sent, and inflated this table 3.4x (12.08M -> 3.52M on
        # the 08-26 run).
        #
        # prompt_lcp_tokens is NOT a parent-to-child LCP, and using it as one
        # is wrong on 36.8% of pairs. AnthropicPromptLcpTracker keeps one prior
        # prompt PER SESSION (anthropic_adapter.py:379-423, a 16-entry LRU
        # against 338 sessions here), so it is the LCP against whatever request
        # ran last in that session -- and a session interleaves several
        # threads, so that is usually not this turn's parent. Applied blindly
        # it silently deleted 993 pairs carrying 5,011,567 real missed tokens
        # (8.44% of everything the run computed) through the `missed <= 0` gate
        # below. It is only usable when the thread parent IS that previous
        # request, and the audit says so itself: previous_prompt_tokens equals
        # the parent's prompt on 99.84% of session-adjacent pairs against 1.86%
        # of the rest. Otherwise fall back to the parent's whole prompt -- safe
        # because tail_role already carries the divergence case, and a
        # user-tail parent whose message list is a prefix of the child's is a
        # token prefix on 99.91% of pairs.
        lcp = _f(row.get("prompt_lcp_tokens"))
        prev_len = _f(row.get("previous_prompt_tokens"))
        adjacent = prev_len is not None and abs(prev_len - pisl) < 1
        ceiling = min(lcp, pisl) if (adjacent and lcp is not None) else pisl
        missed = ceiling - cached
        # A branch point shares more than the parent's prompt: a sibling that
        # ran first leaves blocks the second one legitimately matches past the
        # parent, so the ceiling is not the parent alone. Those come out
        # negative and are not misses.
        if missed <= 0:
            continue
        if node.get("sys_digest") != pnode.get("sys_digest"):
            cause = "SYSTEM_CHANGED"
        elif pnode.get("tail_role") == "system":
            # Server-side, not the client's doing. See tail_role in
            # threads_and_tools.py: the generation prompt and the trailing
            # reminder swap order between the two turns, so the parent's
            # tokens are not a prefix of the child's however warm the cache
            # is. On 594 pairs whose parent message list is an exact prefix of
            # the child's, the token divergence is still >= 88 (exactly 88 on
            # 53.7%); on the same 738 pairs without a system tail it is 0 on
            # 100%. Ranked above the cache-side tests because it is the reason
            # they fire, not a competitor to them.
            cause = "PROMPT_DIVERGENCE"
        elif _f(node.get("n_msgs")) is not None and _f(pnode.get("n_msgs")) is not None \
                and _f(node["n_msgs"]) < _f(pnode["n_msgs"]):
            cause = "HISTORY_REWRITTEN"
        else:
            # Server side, named positively from the router's per-rank probe
            # rather than by elimination. A residual bucket would quietly
            # absorb any mechanism nobody has thought of yet and report it as
            # the server's fault; anything that matches none of the three known
            # shapes is called out as unexplained instead.
            probe = _probe(row.get("route_match_lens"))
            held = probe.get(str(parent.get("routed_rank")))
            chosen = _f(row.get("match_len_chosen"))
            if not probe:
                cause = "UNEXPLAINED (no routing probe)"
            elif held is None:
                cause = "ROUTE_CANDIDATE_ABSENT"
            elif held < pisl - BLOCK_TOLERANCE:
                # NOT named EVICTED any more, and the rename is the point.
                # `held` is the router's probe of the parent's rank, and it
                # equals this request's own isl_cached on 12,033 of 12,063
                # pairs (99.75%; match_len_chosen equals it on 100%), so this
                # test is arithmetically `missed > BLOCK_TOLERANCE` and
                # carries no independent evidence of eviction. The budget says
                # the same: under the old name this bucket claimed 96,824
                # blocks of lost prefix on the 08-26 run, against 24,938
                # blocks ever offloaded and 15,490 ever host-dropped across
                # all four ranks -- 3.9x short. And it does not move with idle
                # time: holding parent.isl_new in [2000, 8000), the rate is
                # 20.7 / 25.7 / 22.2 / 17.6% across gaps of <0.3s / 0.3-2s /
                # 2-60s / >60s.
                #
                # What is left here after PROMPT_DIVERGENCE has taken the
                # system-tail turns is prefix the parent committed that the
                # tree no longer serves. Capacity eviction is one way to get
                # there; on this deployment the bigger one looks to be
                # enable_swa_scratch_reuse (a silent DeepseekV4ForCausalLM
                # model default, absent from ctx_config.yaml), which leaves
                # scratch blocks without a windowed page so _prune_match walks
                # the match back to the last block that has one. Named for the
                # observation rather than for either mechanism.
                cause = "PREFIX_NOT_IN_TREE"
            elif chosen is not None and chosen < held - BLOCK_TOLERANCE:
                cause = "ROUTE_LOSS"
            else:
                cause = "UNEXPLAINED (rank held it and was chosen)"
        b = buckets[cause]
        b["n"] += 1
        b["aids"].append(row["audit_request_id"])
        b["missed"][row["audit_request_id"]] = missed
        b["tok"] += missed
        if _f(row.get("kv_hit_rate")) is not None:
            b["hit"].append(_f(row["kv_hit_rate"]))
        if prefill and new:
            b["ms"] += prefill * min(missed / new, 1.0)
    return {"buckets": dict(buckets), "total_new": total_new,
            "total_prefill": total_prefill, "classified": classified,
            "transitions": sum(1 for r in requests
                               if (threads.get(r["audit_request_id"]) or {}).get("parent"))}


def render_html(runs: list[Run], requests: list[dict], sessions: list[dict],
                ctx_iters: list[dict], gen_iters: list[dict],
                rank_iters: list[dict], summary: dict, notes: list[str],
                rank_pg_iters: list[dict] | None = None,
                capacity_meta: dict | None = None,
                threads: dict | None = None,
                report_dir: Path | None = None) -> str:
    stamps = [_parse_stamp(r.get("timestamp"))
              for r in (ctx_iters + gen_iters) if r.get("timestamp")]
    run_origin = min((t for t in stamps if t is not None), default=None)
    # Filled by _annotate_thread_gaps in main, before the CSVs are written, so
    # every figure below is checkable against requests.csv.
    child_gaps = any(r.get("gap_to_child_ms") is not None for r in requests)
    low = [r for r in requests if r["low_hit_rate"]]
    merged = len(runs) > 1
    shape = "disaggregated" if any(r.disagg for r in runs) else "aggregated"
    title = f"{len(runs)} runs merged" if merged else runs[0].name
    parts = [f"<h1>{title}</h1>",
             f'<p class="sub">{shape} · {summary["requests"]} requests · '
             f'{summary["sessions"]} sessions'
             + ("" if merged else f' · {runs[0].dir}') + "</p>"]
    if merged:
        per_run = []
        for one in runs:
            rows = [r for r in requests if r.get("run") == one.name]
            starts = [r["started_at"] for r in rows if r["started_at"] is not None]
            ends = [r["finished_at"] for r in rows if r["finished_at"] is not None]
            cached = sum(r["isl_cached"] or 0 for r in rows)
            total = sum(r["isl_total"] or 0 for r in rows)
            per_run.append({
                "run": one.name,
                "requests": len(rows),
                "sessions": len({r["session_id"] for r in rows if r["session_id"]}),
                "wall": _dur((max(ends) - min(starts)) * 1000) if starts and ends else "—",
                "kv_hit_rate": (cached / total) if total else None,
                "prefill_iters": sum(1 for c in ctx_iters
                                     if c.get("run") == one.name and c.get("has_prefill")),
            })
        parts.append("<h3>Merged from</h3>")
        parts.append(_table(per_run, [
            ("run", "run"), ("requests", "requests"), ("sessions", "sessions"),
            ("wall", "wall"), ("kv_hit_rate", "hit"),
            ("prefill_iters", "prefill iters")]))
    if notes:
        parts.append('<div class="note">' + "<br>".join(notes) + "</div>")

    # ---- 1 ----
    parts.append("<h2>1 · Requests</h2>")

    # How the requests group into conversations, which every per-turn figure
    # below is keyed on. `session_id` is not the unit: Claude Code fans
    # subagents and background tasks out over one id, so a session is a forest
    # rather than a chain and the ratio here is what says how bushy.
    if threads:
        in_window = {r["audit_request_id"] for r in requests}
        placed = [t for a, t in threads.items() if a in in_window]
        n_thread = len({t["thread_id"] for t in placed})
        n_sess = len({t["session_id"] for t in placed if t["session_id"]})
        if n_thread:
            parts.append(
                '<p class="sub">'
                f'<b>{len(requests):,}</b> requests · <b>{n_thread:,}</b> threads · '
                f'<b>{n_sess:,}</b> sessions — '
                f'{len(requests) / n_thread:.1f} turns per thread, '
                f'{n_thread / n_sess:.1f} threads per session. '
                "A thread is one conversation, rebuilt by prefix-matching the "
                "captured prompts; a session id can carry several of them at once."
                '</p>')
        reuse = _cross_thread_reuse(report_dir)
        if reuse:
            new_all = sum(r["isl_new"] or 0 for r in requests) or 1
            parts.append(
                '<p class="sub">Opening turns reused <b>'
                f'{reuse["root_cached"]:,.0f}</b> tokens from other threads '
                f'({reuse["roots"]:,} thread openings, excluding title calls). '
                f'<b>{reuse["beyond"] / new_all * 100:.2f}%</b> of every token this '
                "run computed was saved by sharing that goes past the "
                "system-prompt-and-tools block "
                f'({reuse["beyond"]:,.0f} tokens); the other '
                f'{reuse["root_cached"] - reuse["beyond"]:,.0f} is boilerplate every '
                "thread on the same client build shares.</p>")
    parts.append("<h3>Latency</h3><table><thead><tr><th>metric</th><th>n</th>"
                 "<th>mean</th><th>p50</th><th>p90</th><th>p99</th></tr></thead><tbody>")
    for label, field, ratio in (("TTFT (client)", "ttft_ms", False),
                                ("TTFT (engine)", "ttft_engine_ms", False),
                                ("Decode", "decode_ms", True),
                                ("E2E", "e2e_ms", False),
                                ("Prefill queue", "prefill_queue_ms", True),
                                ("Prefill", "prefill_ms", True),
                                ("KV transfer", "kv_transfer_ms", True),
                                ("Gap to next turn",
                                 "gap_to_child_ms" if child_gaps
                                 else "gap_to_next_turn_ms", False),
                                ("Tool call (max/turn)", "tool_latency_max_ms", False)):
        shares = None
        if ratio:
            # Share is taken per request and then summarised, not as a ratio of
            # two summaries: the p90 of one phase and the p90 of E2E belong to
            # different requests.
            shares = [r[field] / r["e2e_ms"]
                      for r in requests
                      if r[field] is not None and r["e2e_ms"]]
        parts.append(_stat_row(label, [r[field] for r in requests],
                               duration=True, shares=shares))
    parts.append("</tbody></table>")
    parts.append('<p class="sub">'
                 "Bracketed figures are that phase's share of E2E. "
                 "<b>E2E is server-side only</b> — the interval from the request "
                 "entering the handler to its last byte, so tool execution, which "
                 "happens between requests, is never inside it. It does include "
                 "queueing, tokenisation, detokenisation and transport, so the "
                 "phases will not sum to it.<br>"
                 "<b>Gap to next turn</b> is the wall clock between one turn "
                 "finishing and the next turn "
                 + ("<i>of the same thread</i> starting" if child_gaps
                    else "starting <i>in the same session</i>")
                 + " — all client-side time, whatever it went on. "
                 + ("" if child_gaps else
                    "<b>Session ordering is the wrong key and this row shows it:</b> "
                    "one session id carries several concurrent threads, so "
                    "differencing neighbours pairs turns from unrelated "
                    "conversations and a large share of these come out negative. "
                    "Re-run with <code>--threads</code> for the thread-keyed "
                    "version, which cannot be negative by construction. ")
                 + "<b>Tool call</b> is the "
                 "interval for one specific <code>tool_use_id</code> to come back, "
                 "matched by id. They coincide when a turn emits one tool and the "
                 "result arrives in the very next turn, and diverge when a turn "
                 "emits several in parallel (one gap, many tool intervals), when a "
                 "background result lands several turns later (tool interval longer "
                 "than any one gap), or when a turn calls no tool at all.</p>")

    parts.append("<h3>Tokens &amp; cache</h3><table><thead><tr><th>metric</th><th>n</th>"
                 "<th>mean</th><th>p50</th><th>p90</th><th>p99</th></tr></thead><tbody>")
    for label, field, digits in (("ISL total", "isl_total", 0),
                                 ("ISL cached", "isl_cached", 0),
                                 ("ISL new", "isl_new", 0), ("OSL", "osl", 0),
                                 ("KV hit rate", "kv_hit_rate", 3),
                                 ("Affinity regret (tok)", "affinity_regret", 0)):
        parts.append(_stat_row(label, [r[field] for r in requests], digits=digits))
    parts.append("</tbody></table>")

    parts.append(f'<p class="sub">{len(low)} of {len(requests)} requests fell '
                 f"below {HIT_RATE_FLOOR:.0%}. A low ratio is usually a turn "
                 "appending new content to a prefix the cache served in full, which "
                 "costs nothing recoverable — "
                 'the table below prices each cause instead of listing '
                 'the requests.</p>'
                 if low else '<p class="sub">No request fell below '
                 f'{HIT_RATE_FLOOR:.0%}.</p>')

    # Section 1's payoff, and the reason it sits here rather than in an
    # appendix: the hit-rate stats above say a ratio is low, and this says
    # whether that costs anything. It replaces a per-request listing of every
    # low-hit request, which named requests without ever pricing them -- and on
    # agent traffic almost all of those are a turn appending content to a prefix
    # the cache served in full, which is free.
    cost = kv_miss_cost(requests, threads or {})
    if cost and report_dir is not None:
        # The table in section 1 is a per-cause summary; this is the roster
        # behind it, so a cause can be chased back to individual turns.
        by_aid_all = {r["audit_request_id"]: r for r in requests}
        _write_csv(report_dir / "kv_miss_causes.csv", [
            {"audit_request_id": aid,
             "rid": by_aid_all.get(aid, {}).get("rid"),
             "cause": cause,
             "started_at": by_aid_all.get(aid, {}).get("started_at"),
             "gap_from_parent_ms": _f(
                 (threads or {}).get(aid, {}).get("true_gap_ms")),
             # Same formula as the section 1 table, deliberately. These
             # disagreed once (by 3,579,062 tokens on one bucket) because the
             # table gained the ceiling and this did not.
             "missed_tokens": bucket["missed"].get(aid),
             "isl_total": by_aid_all.get(aid, {}).get("isl_total"),
             "isl_cached": by_aid_all.get(aid, {}).get("isl_cached"),
             "kv_hit_rate": by_aid_all.get(aid, {}).get("kv_hit_rate"),
             "routed_rank": by_aid_all.get(aid, {}).get("routed_rank"),
             "thread_id": (threads or {}).get(aid, {}).get("thread_id")}
            for cause, bucket in sorted(cost["buckets"].items())
            for aid in bucket["aids"]])
    if cost and cost["buckets"]:
        tok_all, ms_all = cost["total_new"], cost["total_prefill"]
        order = ["PROMPT_DIVERGENCE", "PREFIX_NOT_IN_TREE",
                 "ROUTE_CANDIDATE_ABSENT", "ROUTE_LOSS",
                 "SYSTEM_CHANGED", "HISTORY_REWRITTEN"]
        label = {
            "PROMPT_DIVERGENCE": "Prompt diverged — parent turn ended on a "
                                 "<code>role:\"system\"</code> block",
            "PREFIX_NOT_IN_TREE": "Shared prefix was not in the reuse tree",
            "ROUTE_CANDIDATE_ABSENT": "Router never asked the rank that held it",
            "ROUTE_LOSS": "Router had the rank that held it and chose another",
            "SYSTEM_CHANGED": "System prompt changed — prefix invalidated from token 0",
            "HISTORY_REWRITTEN": "History rewritten — compaction or truncation"}
        detail = ['<h3>What each cause of a KV miss costs</h3>',
                  "<table><thead><tr><th>cause</th><th>turns</th>"
                  "<th>hit rate p50</th><th>hit rate mean</th>"
                  "<th>share of all tokens computed</th>"
                  "<th>share of all prefill</th></tr></thead><tbody>"]
        tot_tok = tot_ms = 0.0
        for key in order + [k for k in cost["buckets"] if k not in order]:
            b = cost["buckets"].get(key)
            if not b:
                continue
            tot_tok += b["tok"]
            tot_ms += b["ms"]
            hits = sorted(b["hit"])
            p50 = f"{hits[len(hits) // 2]:.3f}" if hits else "—"
            mean = f"{sum(hits) / len(hits):.3f}" if hits else "—"
            detail.append(
                f"<tr><td>{label.get(key, key)}</td><td>{b['n']:,}</td>"
                f"<td>{p50}</td><td>{mean}</td>"
                f"<td class=lead>{b['tok'] / tok_all * 100 if tok_all else 0:.2f}%</td>"
                f"<td class=lead>{b['ms'] / ms_all * 100 if ms_all else 0:.2f}%</td></tr>")
        all_hits = sorted(h for b in cost["buckets"].values() for h in b["hit"])
        t_p50 = f"{all_hits[len(all_hits) // 2]:.3f}" if all_hits else "—"
        t_mean = f"{sum(all_hits) / len(all_hits):.3f}" if all_hits else "—"
        detail.append(
            "<tr class=total><td>all causes</td>"
            f"<td>{sum(b['n'] for b in cost['buckets'].values()):,}</td>"
            f"<td>{t_p50}</td><td>{t_mean}</td>"
            f"<td class=lead>{tot_tok / tok_all * 100 if tok_all else 0:.2f}%</td>"
            f"<td class=lead>{tot_ms / ms_all * 100 if ms_all else 0:.2f}%</td></tr>")
        detail.append("</tbody></table>")
        detail.append(
            '<p class="sub">For one turn the tokens the engine computed split exactly '
            "two ways: <code>isl_new = (this ISL &minus; parent ISL) + (parent ISL "
            "&minus; this cached)</code>. The first term is content that never existed "
            "before; only the second was ever recoverable, and it is what this counts. "
            "Prefill is attributed pro rata on tokens, which assumes prefill cost is "
            "linear in tokens computed &mdash; true enough under chunked prefill. "
            f"Denominators are the whole run: {tok_all:,.0f} tokens computed and "
            f"{_dur(ms_all)} of prefill across {len(requests):,} requests. "
            f"{cost['classified']:,} of {cost['transitions']:,} parent&rarr;child "
            "transitions could be costed; the rest are thread roots or lack a parent "
            "in the window.</p>")
        wall = _f(summary.get("wall_s"))
        anchor = (f" That is <b>{tot_ms / 1000 / wall * 100:.2f}% of the run's wall "
                  f"clock</b> ({_dur(tot_ms)} of {wall / 3600:.1f} h)." if wall else "")
        detail.append(
            '<p class="sub"><b>Read the share, not the count.</b> A cause worth a '
            "fraction of a percent of prefill is not worth engineering effort however "
            "bad it looks on a single request. A turn that appends a lot of new content "
            "shows a low hit rate and costs nothing recoverable &mdash; it is absent "
            "from this table by construction, which is the point."
            + anchor +
            " A share of prefill is not a share of the run: prefill itself is only part "
            "of it, so carry the wall-clock figure into any decision about whether to "
            "act.</p>")
        parts.append(f'<div id="kv-miss">{"".join(detail)}</div>')
    elif cost is None:
        parts.append('<div class="note">Per-cause KV-miss cost needs the rebuilt '
                     "threads: run <code>analysis/threads_and_tools.py</code>, then "
                     "re-run this with <code>--threads &lt;report&gt;/_threadstudy/"
                     "threads.csv</code>.</div>")

    # ---- 2 ----
    parts.append("<h2>2 · Sessions</h2>")
    turns_total = sum(x["turns"] for x in sessions)
    parts.append(f'<p class="sub">{len(sessions)} sessions, {turns_total} turns '
                 f'({turns_total / len(sessions):.1f} per session).</p>'
                 if sessions else '<p class="sub">No sessions.</p>')
    session_origin = min((r["started_at"] for r in requests
                          if r.get("started_at") is not None), default=None)
    parts.append(_request_chart(requests, sessions, session_origin))
    parts.append('<p class="sub">One bar per session over wall clock, so '
                 "overlapping sessions share an x range. Shading along a bar is "
                 "that session's own requests in flight at that moment — pale "
                 "means the session was open but idle, which is the client running "
                 "a tool, and on agent traffic is most of a session. The panel "
                 "below counts requests in flight across all sessions: that is the "
                 "load the server saw, and it shares the x axis so a peak traces "
                 "back to the sessions that caused it.</p>")
    # Per request, not per session. A session mean flattens the thing worth
    # seeing: these distributions are heavy-tailed and multi-modal, and
    # averaging a 92-turn session into one row destroys both. The per-session
    # rows are still written to sessions.csv.
    panels = [
        ("new ISL", [_f(r.get("isl_new")) for r in requests], "tokens", _tok),
        ("OSL", [_f(r.get("osl")) for r in requests], "tokens", _tok),
        ("new ISL / OSL",
         [_f(r["isl_new"]) / _f(r["osl"])
          for r in requests
          if _f(r.get("isl_new")) is not None and _f(r.get("osl"))],
         "ratio", _ratio_fmt),
        ("ISL (cached + new)", [_f(r.get("isl_total")) for r in requests],
         "tokens", _tok),
    ]
    if child_gaps:
        panels.append(
            ("Gap to next turn (same thread)",
             [r["gap_to_child_ms"] / 1000.0 for r in requests
              if r.get("gap_to_child_ms") is not None],
             "seconds", _sec))
    parts.append(_distributions(panels))
    parts.append('<p class="sub">'
                 "One panel per metric over <b>requests</b>, log x, counts on y. "
                 "The dashed rules are p50 (dark), p90 and p99 (orange). "
                 "<code>ISL</code> is the whole prompt and splits exactly into "
                 "<code>cached ISL + new ISL</code>; only <b>new ISL</b> was "
                 "computed, so it, not ISL, is what prefill cost tracks. "
                 "<b>new ISL / OSL</b> is per request, which is not the ratio of "
                 "the two panels beside it — a turn that appends a tool result "
                 "and emits one token sits far right, and there are many of them. "
                 + ("<b>Gap to next turn</b> is wall clock from this turn "
                    "finishing to the next turn <i>of the same thread</i> "
                    "starting, i.e. tool execution plus client think time. It is "
                    "taken from the rebuilt threads, not from session ordering: "
                    "one session id carries several concurrent threads, so "
                    "differencing neighbours within a session pairs unrelated "
                    "turns and yields a negative gap on 38% of them. Turns with "
                    "no child in the window — thread tails — are absent. "
                    if child_gaps else
                    "<b>Gap to next turn</b> is omitted: it needs the rebuilt "
                    "threads, so re-run with <code>--threads</code>. The "
                    "session-ordered version is not a substitute, because one "
                    "session id carries several concurrent threads. ")
                 + "Per-session rows, including the client-time and fit-per-rank "
                 "columns this replaced, are in <code>sessions.csv</code>."
                 + ("" if summary["kv_capacity_tokens"] else
                    " KV capacity was not in this run's logs, so the fit-per-rank "
                    "column there is blank.") + "</p>")

    # ---- 3 ----
    parts.append("<h2>3 · Prefill server</h2>")
    prefilling = [c for c in ctx_iters if c.get("has_prefill")]
    # KV occupancy is a level, not a rate: the cache stays full of retained
    # prefixes on iterations that prefill nothing, so this one curve is drawn
    # over every iteration while the rest are restricted to prefilling ones.
    occupancy = list(ctx_iters)
    if prefilling:
        parts.append(f'<p class="sub">{len(prefilling):,} of {len(ctx_iters):,} '
                     "iterations carried context tokens. Everything below except "
                     "KV utilization is restricted to those; utilization is an "
                     "occupancy level and holds on every iteration. Charts drawn "
                     "from the iteration log and from the request log each start "
                     "at their own first record: the worker and the serve edge "
                     "keep clocks hours apart, so a shared x is the same elapsed "
                     "time into the run, not the same instant.</p>")
        ctx_iters = prefilling
    rank_occupancy = [r for r in rank_iters
                      if not prefilling or r["num_ctx_tokens"]] or rank_iters
    if occupancy:
        # Per rank, not pooled. A four-rank mean of 0.28 is the same number
        # whether all four sit at 0.28 or one sits at 1.0 and three at 0.04,
        # and only the second is a capacity problem.
        #
        # 1 - free/max, not the log's kv_cache_util. That field is
        # 1 - available/max and available counts a retained reusable block as
        # free, so it measures only what in-flight requests have pinned and
        # reads identically on an empty pool and a full one. This curve is the
        # one that answers "when does the pool fill and start evicting"; the
        # pinned share is kept below as a statistic, where it belongs.
        pg_rows = [r for r in (rank_pg_iters or [])
                   if r.get("kv_pool_filled") is not None]
        pg_filled = _chart(pg_rows, "kv_pool_filled",
                           "KV pool filled per rank and pool group",
                           "1 − free / capacity", pct=True, origin=run_origin,
                           series_field="series", toggle=True) if pg_rows else ""
        if pg_filled:
            parts.append(pg_filled)
            roles = (summary.get("pool_group_roles")
                     or (capacity_meta or {}).get("pool_group_roles") or {})
            if roles:
                spelled = "; ".join(
                    f"<b>pg{pg}</b> = {', '.join(info['roles'])}"
                    + (f" (compress_ratio {', '.join(str(r) for r in info['ratios'])})"
                       if info.get("ratios") else "")
                    for pg, info in sorted(roles.items()))
                parts.append(
                    f'<p class="sub">Pool groups hold different KV content: '
                    f"{spelled}. Read from the worker's own "
                    "<code>role-to-pool/lifecycle mapping</code> lines, not assumed: "
                    "the pool group index is the ascending sort of each group's "
                    "slot-size vector, so it follows neither the order the roles are "
                    "declared in nor the order of the <code>pool_ratio</code> list a "
                    "config comment is likely to name.</p>")
            parts.append(
                '<p class="sub">One curve per (rank, pool group). This is the '
                "grain the allocator actually works at: an eviction is sized off "
                "a <em>single</em> group's free-slot count, so one group pinned "
                "at 0 evicts on every allocation while the summed curve below "
                "still reads roomy. The groups are not the same size — the pool "
                "ratio splits bytes, so a group whose blocks are large gets "
                "proportionally fewer slots — which is why they are not pooled "
                "and why a group's own capacity is its own high-water "
                "<code>free + evictable</code>. Counts are slots, one slot per "
                "page: a token block spanning three groups occupies three "
                "slots, so these are not logical KV blocks.</p>")
        # Drawn only when the per-(rank, pool group) chart above is not: this
        # is the same quantity summed over groups, and that sum is precisely
        # what hides a saturated small group -- the thing the split exists to
        # show. The paragraph below still applies, since it defines
        # kv_pool_filled, which the pool-group curves and the stats table use.
        filled = any(r.get("kv_pool_filled") is not None for r in rank_occupancy)
        if not pg_filled:
            parts.append(_chart(
                rank_occupancy, "kv_pool_filled", "KV pool filled per rank",
                "1 − free / capacity", pct=True, origin=run_origin,
                series_field="series", toggle=True) or _chart(
                    rank_occupancy or occupancy,
                    "kv_util_mean" if not rank_occupancy else "kv_cache_util",
                    "KV slots pinned by in-flight requests, per rank",
                    "pinned / max blocks", pct=True, origin=run_origin,
                    series_field="series" if rank_occupancy else "worker",
                    toggle=True))
        if filled:
            parts.append(
                '<p class="sub">Share of each rank\'s KV slots holding content, '
                "pinned or merely retained for reuse — <code>1 − free/capacity</code>. "
                "Not the log's <code>kv_cache_util</code>, which is "
                "<code>1 − available/max</code> where <code>available = free + "
                "evictable</code>: a block whose last reference was dropped stays in "
                "the eviction LRU still holding matchable content and counts there as "
                "free, so that field only ever measures what is pinned right now and "
                "reads the same on a full pool as on an empty one. It is reported "
                "below as <b>Pinned by in-flight</b>. Capacity is recovered as the "
                "largest <code>free + evictable</code> observed, which is exact "
                "because pinned is never negative.</p>")
        else:
            parts.append(
                '<p class="sub">Slots pinned by requests in flight — the log\'s '
                "<code>kv_cache_util</code>, which is <code>1 − available/max</code> "
                "with <code>available = free + evictable</code>, so a block retained "
                "for reuse counts as free and this reads the same on a full pool as "
                "on an empty one. <b>It is not an occupancy figure.</b> This run's "
                "logs predate <code>kv_free_blocks</code> / "
                "<code>kv_evictable_blocks</code>, so how full the pool actually got "
                "is not measured here at all.</p>")
        # The KV manager's own reused/missed block counters, differenced per
        # rank. Pooling ranks would average away the rank that is missing while
        # its peers hit -- the one case the curve exists to show.
        # Prefer the log's cumulative ratio; older logs carry only the
        # counters, and there the differenced curve is all there is.
        hit_cum = any(r.get("kv_hit_rate_cum") is not None
                      for r in rank_occupancy)
        hit_field = "kv_hit_rate_cum" if hit_cum else "kv_hit_rate_iter"
        iter_hits = _chart(rank_occupancy, hit_field,
                           "KV hit rate per rank (server block counters"
                           + (", cumulative)" if hit_cum else ")"),
                           "reused / (reused+missed)"
                           + (", since engine start" if hit_cum else ""),
                           pct=True,
                           origin=run_origin, series_field="series",
                           toggle=True)
        parts.append(iter_hits)
        notes_html = ["Per rank, from the KV manager's own block counters in "
                      "the worker log: of every block the rank has acquired "
                      "since the engine started, the share that came from "
                      "reuse. Block-level, so a partially matched block counts "
                      "wholly as a miss. The counters only rise, so this is a "
                      "running total rather than a rate — it moves freely "
                      "through warmup and barely at all late, once the "
                      "denominator is millions of blocks, so a late dip is "
                      "damped rather than shown. It is the correctly weighted "
                      "figure nonetheless: every block counts once, which the "
                      "per-iteration mean in the table below does not do. The "
                      "per-request view lives in section 1, where a single "
                      "request can be named."
                      if hit_cum else
                      "Per rank, from the KV manager's own block counters in "
                      "the worker log: of the blocks acquired between two "
                      "iterations, the share that came from reuse. Block-level, "
                      "so a partially matched block counts wholly as a miss. "
                      "The per-request view lives in section 1, where a single "
                      "request can be named."]
        if not iter_hits:
            notes_html.append(
                "No hit-rate curve: this run's logs predate "
                "<code>kv_reused_blocks</code> / <code>kv_missed_blocks</code>, "
                "which the KV manager reports per rank on every iteration. "
                "There is no fallback — the hit rate is simply not measured on "
                "this run.")
        if notes_html:
            parts.append('<p class="sub">' + " ".join(notes_html) + "</p>")
    if ctx_iters:
        parts.append(_chart(ctx_iters, "utilization", "Token-budget utilization",
                            "mean ctx tokens / budget", pct=True,
                            origin=run_origin, mean_line=True))
        parts.append(_chart(ctx_iters, "imbalance", "Rank imbalance",
                            "(max − mean) / mean", origin=run_origin,
                            mean_line=True))
        parts.append("<table><thead><tr><th>metric</th><th>n</th><th>mean</th>"
                     '<th class="imb">imbalance</th>'
                     "<th>p50</th><th>p90</th><th>p99</th></tr></thead><tbody>")
        for label, field, digits, spread in (
                ("Utilization", "utilization", 3, "imbalance"),
                ("Hit rate", "kv_hit_rate_cum", 3, "kv_hit_rate_cum_spread"),
                ("Pool filled", "kv_pool_filled", 3, "kv_pool_filled_spread"),
                ("Evicted tokens", "kv_evicted_tokens", 0, None),
                ("Pinned by in-flight", "kv_util_mean", 3, "kv_util_spread"),
                ("Device step ms", "device_step_time_ms", 1, "device_step_spread"),
                ("Host step ms", "host_step_time_ms", 1, None)):
            if field == "kv_hit_rate_cum":
                # A running total has no distribution. Its p90 would be a time
                # quantile -- where the curve happened to sit 90% of the way
                # through -- not a spread over comparable samples, and printing
                # one invites it to be read as tail behaviour. Only the endpoint
                # carries meaning, so only the endpoint is shown. The imbalance
                # cell survives because it is a spread across ranks at one
                # instant, which stays a real quantity: it is the gap between
                # the ranks' whole-run rates.
                have = [c.get(field) for c in ctx_iters if c.get(field) is not None]
                imb = [c.get(spread) for c in ctx_iters if c.get(spread) is not None]
                if not have:
                    # Log predates the cumulative field; the differenced curve
                    # is all there is, so report it as the distribution it is.
                    parts.append(_stat_row(
                        label, [c.get("kv_hit_rate_iter") for c in ctx_iters],
                        digits=digits, with_spread=True,
                        spread=[c.get("kv_hit_rate_spread") for c in ctx_iters]))
                    continue
                parts.append(
                    f"<tr><td>{label}</td><td>{len(have)}</td>"
                    f'<td class="lead">{_num(have[-1] if have else None, 3)}</td>'
                    f'<td class="imb">{_num(imb[-1] if imb else None, 3)}</td>'
                    "<td>&mdash;</td><td>&mdash;</td><td>&mdash;</td></tr>")
                continue
            parts.append(_stat_row(
                label, [c[field] for c in ctx_iters], digits=digits,
                duration="ms" in label, with_spread=True,
                spread=[c.get(spread) for c in ctx_iters] if spread else None))
        parts.append("</tbody></table>")
        parts.append(_tier_table(occupancy, rank_iters))
        # Drawn over every iteration, not just the prefilling ones: a block can
        # be offloaded on an iteration that scheduled no context work, and
        # restricting the curve would make the counter appear to jump.
        # When a turn's parent prefix was evicted, the rank that prefilled it
        # had to give the blocks up -- which is the same pool pressure this
        # curve is made of. Marking when those turns *started* puts the two on
        # one axis instead of asking the reader to hold both in their head.
        unserved_at = sorted(
            t for t in (_f(by_aid_all.get(aid, {}).get("started_at"))
                        for aid in ((cost or {}).get("buckets", {})
                                    .get("PREFIX_NOT_IN_TREE", {}).get("aids", [])))
            if t is not None) if cost else []
        tier_charts = [
            _chart(rank_iters, f"kv_{key}_blocks_cum", title,
                   "blocks since engine start", origin=run_origin,
                   series_field="series", toggle=True,
                   vlines=marks, vlabel=vlabel)
            for key, title, marks, vlabel in (
                ("offload", "Blocks offloaded GPU → host, per rank",
                 unserved_at, "turns whose shared prefix was not in the tree (grey)"),
                ("onboard", "Blocks onboarded host → GPU, per rank", None, None))]
        if any(tier_charts):
            parts.extend(c for c in tier_charts if c)
            parts.append(
                '<p class="sub">Cumulative, straight from the KV manager\'s own '
                "counters in the worker log — the slope is the rate and the "
                "separation between ranks is the imbalance. Cumulative rather "
                "than per-iteration because the per-iteration series is zero on "
                "most iterations and dense enough to be time-window averaged "
                "before it is drawn, which turns a burst into a small number "
                "rather than a spike; the per-iteration deltas are in "
                "<code>ctx_rank_iters.csv</code> as "
                "<code>kv_offload_blocks_iter</code> and "
                "<code>kv_onboard_blocks_iter</code> for anyone who wants the "
                "bursts. The grey rules on the offload chart mark the start of "
                "every turn whose shared prefix was not in the reuse tree "
                "(<code>PREFIX_NOT_IN_TREE</code> in section 1, rostered in "
                "<code>kv_miss_causes.csv</code>). Read them <i>against</i> "
                "the curves, not as their cause: on the run this was built "
                "for they do not line up, and that mismatch is part of why "
                "the bucket is no longer called eviction. "
                "<b>Offload</b> is a block leaving GPU memory but "
                "surviving on the host tier, so it is still reusable at the "
                "cost of a copy back; <b>onboard</b> is that copy back actually "
                "happening, i.e. a hit that the GPU no longer held. Neither is "
                "a loss — <b>dropped</b> in the table above is. A counter that "
                "steps down rather than up means the engine restarted inside "
                "the window, since these run since engine start.</p>")
        parts.append(_rank_totals(rank_iters))
        parts.append('<p class="sub">'
                     "The imbalance column is (max − mean) / mean across that "
                     "instance's ranks, averaged over iterations: 0 means every rank "
                     "carried the same amount that iteration. For "
                     "<b>utilization</b> under attention-DP it is pinned near "
                     "<code>ranks − 1</code> by construction — one prefill occupies "
                     "one rank — so read routing skew off the per-rank table above, "
                     "not off this column. <b>Pinned by in-flight</b> is structural "
                     "for the same reason and to the same value: the rank running the "
                     "prefill holds every pinned slot and the others hold none, which "
                     "scores exactly <code>ranks − 1</code>. Measured on the relay "
                     "pilot, 89.5% of iterations sat at 3.000 on four ranks, because "
                     "on 136 of 151 iterations exactly one rank had anything pinned "
                     "at all. It says nothing about routing. For <b>hit rate</b> and "
                     "<b>pool filled</b> the column is meaningful: those are "
                     "rank-resident state that persists across iterations, and a "
                     "spread there means the ranks really do hold different amounts — "
                     "pool fill spread was 0.126 on the same run, against 3.000 for "
                     "pinned. "
                     + ("<b>Evicted tokens</b> is blank: it is "
                        "<code>alloc_total − alloc_new</code>, and the v2 manager "
                        "assigns those two counters the same value at every call "
                        "site, so the difference is zero by construction and "
                        "measures nothing. "
                        if runs and runs[0].kv_manager_v2() else
                        "Evicted tokens come from "
                        "<code>alloc_total − alloc_new</code>, which counts blocks "
                        "taken while still holding reusable content — an upper "
                        "bound, since a freed partial block lands there too. ") +
                     "<b>Hit rate</b> is the run's lifetime figure, "
                     "<code>reused / (reused+missed)</code> over every block the "
                     "instance acquired, read straight off the counters rather "
                     "than averaged: it carries no p50/p90/p99 because a running "
                     "total has no distribution — those cells would report where "
                     "the curve sat at a moment, not a spread. Averaging the "
                     "per-iteration deltas instead, as this row used to, divides "
                     "by whatever handful of blocks one iteration happened to "
                     "acquire, so a chunked prefill's continuation chunk — which "
                     "acquires only fresh blocks by construction — scores 0 "
                     "however warm its request was, and enough of those drag the "
                     "mean far below the rate the run actually achieved. "
                     "<code>device step</code> brackets the whole loop body, so it "
                     "tracks wall clock rather than GPU-busy time.</p>")
    else:
        parts.append('<p class="sub">No prefill iteration log found.</p>')

    if gen_iters:
        decoding = [g for g in gen_iters if g.get("has_decode")] or gen_iters
        parts.append("<h3>Decode</h3>")
        parts.append(f'<p class="sub">{len(decoding):,} of {len(gen_iters):,} '
                     "generation iterations carried a decode batch.</p>")
        parts.append(_chart(decoding, "batch_occupancy", "Batch occupancy",
                            "batch / max_batch_size", pct=True, origin=run_origin))
        parts.append(_chart(decoding, "device_step_time_ms", "Decode step time",
                            "ms", origin=run_origin))
        parts.append("<table><thead><tr><th>metric</th><th>n</th><th>mean</th>"
                     '<th class="imb">imbalance</th>'
                     "<th>p50</th><th>p90</th><th>p99</th></tr></thead><tbody>")
        for label, field, digits, spread in (
                ("Batch occupancy", "batch_occupancy", 3, "imbalance"),
                ("Decode batch", "decode_batch_total", 1, "imbalance"),
                ("Tokens / request", "tokens_per_request", 3, None),
                ("Gen tokens", "gen_tokens_total", 1, "gen_tokens_spread"),
                ("Paused", "paused_requests", 1, None),
                ("KV util", "kv_util_mean", 3, "kv_util_spread"),
                ("Device step ms", "device_step_time_ms", 1, "device_step_spread"),
                ("Host step ms", "host_step_time_ms", 1, None)):
            parts.append(_stat_row(
                label, [g[field] for g in decoding], digits=digits,
                duration="ms" in label, with_spread=True,
                spread=[g.get(spread) for g in decoding] if spread else None))
        parts.append("</tbody></table>")
        tpr = _stats([g["tokens_per_request"] for g in decoding])["p50"]
        parts.append('<p class="sub">'
                     "Decode throughput is bounded by batch slots, so occupancy uses "
                     "<code>max_batch_size</code> — not the token budget prefill is "
                     "measured against. <code>tokens / request</code> is 1.0 without "
                     "speculative decoding and becomes the accepted draft length plus "
                     "one with MTP; it decays toward 1.0 as acceptance degrades, which "
                     "batch size and step time both hide."
                     + (f" Median here is {tpr:.2f}." if tpr is not None else "")
                     + " Cross-check against section 1: "
                     "<code>device step ÷ tokens-per-request</code> should track "
                     "<code>decode_ms ÷ osl</code>, since each step emits that many "
                     "tokens per request. A gap is time spent outside the loop — "
                     "queueing, detokenisation, transport.</p>")

    # Where each conversation was sent, and why. It belongs in section 3
    # because it is what the `imbalance` column above is actually made of: the
    # router balances *requests*, but a thread is pinned for its whole life, so
    # what a rank really takes on is the compute of every conversation whose
    # opening turn landed there.
    assign = rank_thread_assignment(requests, threads or {})
    if assign:
        kinds = assign["kinds"]
        order = [k for k in ("main", "subagent", "title", "sdk", "no-tools",
                             "other", "unknown")
                 if any(c.get(k) for c in kinds.values())]
        total_tok = sum(assign["tokens"].values()) or 1.0
        parts.append("<h3>Thread roots per rank</h3>")
        parts.append("<table><thead><tr><th>rank</th>"
                     + "".join(f"<th>{k}</th>" for k in order)
                     + "<th>threads</th><th>thread ISL new</th><th>share</th>"
                     "</tr></thead><tbody>")
        for rank in sorted(kinds, key=lambda r: int(r) if r.isdigit() else r):
            c = kinds[rank]
            tok = assign["tokens"][rank]
            parts.append(
                f"<tr><td>{rank}</td>"
                + "".join(f"<td>{c.get(k, 0):,}</td>" for k in order)
                + f"<td>{sum(c.values()):,}</td><td>{tok:,.0f}</td>"
                f"<td class=lead>{tok / total_tok * 100:.1f}%</td></tr>")
        totals = Counter()
        for c in kinds.values():
            totals.update(c)
        parts.append(
            "<tr class=total><td>all</td>"
            + "".join(f"<td>{totals.get(k, 0):,}</td>" for k in order)
            + f"<td>{assign['roots']:,}</td><td>{total_tok:,.0f}</td>"
            "<td class=lead>100.0%</td></tr></tbody></table>")
        parts.append(
            '<p class="sub"><b>ISL new is the whole chain, not the opening turn.</b> '
            "A thread stays on the rank its first turn was routed to — every later "
            "turn carries the previous prompt, so that rank's match is unbeatable — "
            "which makes one routing decision own all of a conversation's compute. "
            "Read this against <code>imbalance</code> above: near-equal thread counts "
            "with unequal token totals is not the router misdividing, it is one kind "
            "of conversation being heavier than another and affinity keeping each "
            "kind together.</p>")

        parts.append("<h3>Why each thread landed there</h3>")
        reasons = assign["reason"]
        cols = sorted({k for c in reasons.values() for k in c},
                      key=lambda k: -sum(c.get(k, 0) for c in reasons.values()))
        parts.append("<table><thead><tr><th>rank</th>"
                     + "".join(f"<th>{k}</th>" for k in cols)
                     + "</tr></thead><tbody>")
        for rank in sorted(reasons, key=lambda r: int(r) if r.isdigit() else r):
            parts.append(f"<tr><td>{rank}</td>"
                         + "".join(f"<td>{reasons[rank].get(k, 0):,}</td>"
                                   for k in cols) + "</tr>")
        parts.append("<tr class=total><td>all</td>"
                     + "".join(f"<td>{sum(c.get(k, 0) for c in reasons.values()):,}</td>"
                               for k in cols) + "</tr></tbody></table>")
        parts.append(
            '<p class="sub">Replayed from the routing trace. <b>best match</b> is one '
            "rank scoring strictly lowest — cache affinity deciding. <b>tie, req_id "
            "shuffle</b> is every candidate scoring identically <i>and</i> carrying "
            "the same load, so the winner is the first entry of a permutation seeded "
            f"on the request id; that happens when a prompt matches under "
            f"{ROUTER_MATCH_RATE_THRESHOLD:.0%} of itself anywhere, which forces every "
            "match to zero. <b>candidates capped</b> means the fair-share cap had "
            "already dropped a rank for the rest of that batch — possibly the one "
            "holding the prefix, which the trace cannot show. The gate assumes the "
            "router's default threshold; a deployment that overrides it will mislabel "
            "the ties.</p>")

    # ---- 4 ----
    parts.append("<h2>4 · Run totals</h2>")
    wall = summary["wall_s"]

    def share(value: float | None) -> str:
        return f"{value / wall * 100:.1f}%" if value and wall else "—"

    parts.append('<div class="kpi">' + "".join(
        f"<div><b>{_dur(v * 1000 if v is not None else None) if dur else _num(v, 3)}</b>"
        f"<span>{k}{'' if not pctof or pctof == '—' else ' · ' + pctof}</span></div>"
        for k, v, dur, pctof in (
            ("wall", wall, True, None),
            ("server busy", summary["server_busy_s"], True,
             share(summary["server_busy_s"])),
            ("client idle", summary["client_idle_s"], True,
             share(summary["client_idle_s"])),
            ("GPU prefill", summary["gpu_prefill_s"], True,
             share(summary["gpu_prefill_s"])),
            ("GPU decode", summary["gpu_decode_s"], True,
             share(summary["gpu_decode_s"])),
            ("KV hit rate", summary["kv_hit_rate"], False, None))) + "</div>")

    # Absolute seconds answer "how long"; the share answers "where did the run
    # go", and only the second one survives comparison between runs of
    # different length.
    server_cpu = None
    gpu_total = (summary["gpu_prefill_s"] or 0) + (summary["gpu_decode_s"] or 0)
    if summary["server_busy_s"] and gpu_total:
        server_cpu = summary["server_busy_s"] - gpu_total
    decomposition = [
        ("wall", wall, 0),
        ("server busy", summary["server_busy_s"], 1),
        ("GPU prefill", summary["gpu_prefill_s"], 2),
        ("GPU decode", summary["gpu_decode_s"], 2),
        ("server CPU (busy − GPU)", server_cpu, 2),
        ("client idle", summary["client_idle_s"], 1),
    ]
    body = "".join(
        f'<tr><td style="padding-left:{depth * 18 + 8}px">{label}</td>'
        f'<td class="lead">{_dur(value * 1000) if value is not None else "—"}</td>'
        f'<td class="lead">{"100.0%" if depth == 0 and value else share(value)}</td></tr>'
        for label, value, depth in decomposition)
    parts.append("<table><thead><tr><th>time</th><th>duration</th>"
                 f"<th>share of wall</th></tr></thead><tbody>{body}</tbody></table>")
    parts.append(_table([summary], [
        ("requests", "requests"), ("sessions", "sessions"),
        ("isl_cached_total", "cached ISL"), ("isl_new_total", "new ISL"),
        ("osl_total", "OSL"), ("low_hit_rate_requests", f"< {HIT_RATE_FLOOR:.0%}"),
        ("kv_evicted_tokens_total", "evicted tok"),
        ("kv_host_dropped_blocks_total", "dropped pg"),
        ("kv_onboard_blocks_total", "onboard pg"),
        ("kv_pool_filled_peak", "pool filled"),
        ("prefill_utilization_mean", "util"), ("prefill_imbalance_mean", "imbalance"),
        ("gpu_prefill_batches", "prefill batches"),
        ("gpu_decode_batches", "decode batches")]))
    parts.append('<div class="note">'
                 "<b>Time decomposition.</b> <code>wall</code> = first request start to "
                 "last finish. <code>server busy</code> is the union of request "
                 "intervals, not their sum, so concurrency is not double-counted; "
                 "<code>client idle</code> is the remainder — tool calls and client "
                 "think time. GPU time is de-duplicated by batch "
                 "(<code>forward_start_time</code>): one forward serves the whole "
                 "batch and its elapsed time is stamped on every request in it, so "
                 "summing per request would multiply by batch size."
                 + ("" if summary["gpu_decode_s"] else
                    " <b>GPU decode is blank</b> because the generation worker runs the "
                    "overlap scheduler, which never emits "
                    "<code>time_breakdown_metrics</code>; only a worker with "
                    "<code>disable_overlap_scheduler: true</code> reports it.")
                 + "</div>")

    return (f"<!doctype html><meta charset=utf-8><title>{title}</title>"
            f"<style>{CSS}</style>" + "".join(parts) + CHART_JS)


# Above this many rows a CSV is written gzipped. Iteration-grain files run to
# millions of rows on agent traffic -- one run's per-rank file is 123 MB plain,
# 6 MB compressed -- and nothing reads them by eye at that size. pandas and the
# csv module both open .csv.gz by extension, so the only cost is the suffix.
GZIP_ROWS = 50_000


def _write_csv(path: Path, rows: list[dict]) -> Path:
    """Write ``rows``; returns the path actually written (may gain ``.gz``)."""
    if len(rows) > GZIP_ROWS:
        path = path.with_suffix(path.suffix + ".gz")
    if not rows:
        path.write_text("")
        return path
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    opener = (gzip.open if path.suffix == ".gz" else open)
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _budget(attempt_dir: Path, key: str, names: tuple[str, ...],
            default: int) -> int:
    for name in names:
        config = attempt_dir / name
        if config.exists():
            hit = re.search(rf"^{key}:\s*(\d+)", config.read_text(), re.MULTILINE)
            if hit:
                return int(hit.group(1))
    return default


def _apply_window(run: Run, loaded: dict,
                  window: tuple[float | None, float | None]) -> dict:
    """Drop everything outside the window, from every source at once.

    Filtering happens *after* the rows are built, never while reading, because
    the iteration metrics are deltas against the previous iteration: cutting the
    log first would charge the whole idle stretch to the first surviving
    iteration, inventing a burst of allocations and evictions that never
    happened. Built first, then filtered, every delta is still the one-step
    delta it claims to be.

    ``gpu_forward_ms`` re-reads the perf records rather than the request rows,
    so the run's perf cache has to be narrowed too -- by request id, since those
    records carry a per-process monotonic clock that cannot be compared to wall
    time. A record whose request was dropped is dropped with it, which is what
    keeps section 4's GPU time inside the window.
    """
    kept = {"requests": [], "ctx_iters": [], "gen_iters": [], "rank_iters": [],
            "rank_pg_iters": []}
    undated = 0
    for row in loaded["requests"]:
        if _inside(row.get("started_at"), window):
            kept["requests"].append(row)
        elif row.get("started_at") is None:
            undated += 1
    for key in ("ctx_iters", "gen_iters", "rank_iters", "rank_pg_iters"):
        kept[key] = [row for row in loaded[key]
                     if _inside(_parse_stamp(row.get("timestamp")), window)]

    ids = {str(row["rid"]) for row in kept["requests"] if row.get("rid") is not None}
    ids |= {str(row["ctx_request_id"]) for row in kept["requests"]
            if row.get("ctx_request_id") is not None}
    narrowed = []
    for record in _load_perf(run):
        ctx = record.get("ctx_perf_metrics") or {}
        gen = record.get("gen_perf_metrics") or {}
        if any(str(key) in ids for key in
               (ctx.get("ctx_request_id"), gen.get("ctx_request_id"),
                record.get("ctx_request_id"), record.get("request_id"))
               if key not in (None, "")):
            narrowed.append(record)
    run._perf_cache = narrowed

    excluded = {key: len(loaded[key]) - len(kept[key]) for key in kept}
    excluded["undated_requests"] = undated
    return {**loaded, **kept, "excluded": excluded}


def load_run(attempt_dir: Path,
             window: tuple[float | None, float | None] = (None, None)) -> dict:
    """Everything one attempt directory yields, tagged with its run name."""
    run = Run(attempt_dir)
    token_budget = _budget(attempt_dir, "max_num_tokens",
                           ("ctx_config.yaml", "server_config.yaml"), 8192)
    batch_budget = _budget(attempt_dir, "max_batch_size",
                           ("gen_config.yaml", "server_config.yaml"), 128)
    ctx_iters, capacity, rank_iters, rank_pg_iters = build_ctx_iters(
        run, token_budget)
    gen_iters = build_gen_iters(run, batch_budget)
    requests = build_requests(run)
    for rows in (ctx_iters, gen_iters, requests, rank_iters, rank_pg_iters):
        for row in rows:
            row["run"] = run.name
    loaded = {"run": run, "requests": requests, "ctx_iters": ctx_iters,
              "gen_iters": gen_iters, "capacity": capacity,
              "rank_iters": rank_iters, "rank_pg_iters": rank_pg_iters}
    if window[0] is None and window[1] is None:
        return {**loaded, "excluded": {}}
    return _apply_window(run, loaded, window)


def _run_name(attempt_dir: Path) -> str:
    """The run a directory belongs to, named or not.

    Both layouts are in the wild -- `<root>/<run>/attempt-NNN` and, since the
    dated rollout, `<root>/<YYYY-MM>/<DD>/<run>/attempt-NNN` -- so the run is
    recognised by shape rather than by depth, and a run directory passed
    without its attempt still names itself.
    """
    resolved = attempt_dir.resolve()
    return (resolved.parent.name if _ATTEMPT_DIR.fullmatch(resolved.name)
            else resolved.name)


def resolve_out(out: Path | None, attempt_dirs: list[Path]) -> Path:
    """Report directory: a label under REPORTS_ROOT, unless --out is absolute.

    A relative --out is read as a label, not a path, so `--out glm5.2_pilot`
    lands where every other report lives instead of wherever the shell happened
    to be standing. An absolute path is the one way out of the root, for the
    case where somebody deliberately wants the report elsewhere.

    Two attempts of one run share a label and the later overwrites the earlier,
    which is the wanted behaviour for a re-run and not for a comparison; give
    the second an explicit label when both are being kept.
    """
    if out is not None:
        return out if out.is_absolute() else REPORTS_ROOT / out
    names = [_run_name(d) for d in attempt_dirs]
    # A merge is named after the run it starts from, which is only ever a
    # stand-in: pass --out when the merge has a subject of its own.
    label = (names[0] if len(names) == 1
             else f"_merged_{len(names)}runs_{names[0]}")
    return REPORTS_ROOT / label


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempt_dir", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"report label under {REPORTS_ROOT}, or an "
                             "absolute path to write outside it "
                             "(default: the run's own name)")
    parser.add_argument("--threads", type=Path, default=None,
                        help="threads.csv from analysis/threads_and_tools.py; "
                             "adds section 1's per-cause KV-miss cost table, "
                             "which needs each request's parent turn")
    parser.add_argument("--causes", type=Path, default=None,
                        help="CSV of request_index,cause to annotate the "
                             "low-hit-rate table")
    parser.add_argument("--since", default=None, metavar="ISO",
                        help="ignore everything before this instant, e.g. "
                             "2026-08-21T12:57:49+00:00 -- for a run whose "
                             "traffic came in waves separated by idle hours, "
                             "which would otherwise be charged to the workload "
                             "(naive stamps are read as local time)")
    parser.add_argument("--until", default=None, metavar="ISO",
                        help="ignore everything after this instant")
    args = parser.parse_args()

    window = (parse_bound(args.since), parse_bound(args.until))
    loaded = [load_run(d, window) for d in args.attempt_dir]
    runs = [entry["run"] for entry in loaded]
    out = resolve_out(args.out, args.attempt_dir)
    out.mkdir(parents=True, exist_ok=True)

    requests = [r for entry in loaded for r in entry["requests"]]
    ctx_iters = [r for entry in loaded for r in entry["ctx_iters"]]
    gen_iters = [r for entry in loaded for r in entry["gen_iters"]]
    rank_iters = [r for entry in loaded for r in entry["rank_iters"]]
    rank_pg_iters = [r for entry in loaded for r in entry["rank_pg_iters"]]
    # Prefer a run that found the v1 capacity line, since that one yields
    # tokens; fall back to any run that at least recovered the slot count from
    # the level counters, which is all a v2 log offers.
    capacity = next(
        (e["capacity"] for e in loaded if e["capacity"].get("capacity_tokens")),
        next((e["capacity"] for e in loaded if e["capacity"].get("slots_per_rank")),
             {}))

    requests.sort(key=lambda r: (r.get("started_at") or 0, str(r.get("rid"))))
    for index, row in enumerate(requests):
        row["request_index"] = index
    # Sessions are grouped on the id alone, deliberately. The gateway hands a
    # live conversation to a successor job, so one session really does span
    # runs -- keying on (run, session) would cut a single conversation into
    # fragments and report each as a fresh cold start.
    annotate_sessions(requests)

    if args.causes and args.causes.exists():
        with args.causes.open(encoding="utf-8") as handle:
            causes = {str(r["request_index"]): r.get("cause", "")
                      for r in csv.DictReader(handle)}
        for row in requests:
            row["cause"] = causes.get(str(row["request_index"]), "")

    # Optional: the rebuilt threads, which section 1's KV-miss cost table needs
    # to know each request's previous turn. Absent, the section says how to get it.
    threads: dict = {}
    if args.threads and args.threads.exists():
        with args.threads.open(encoding="utf-8") as handle:
            threads = {t["audit_request_id"]: t for t in csv.DictReader(handle)}

    sessions = build_sessions(requests, capacity)
    for session in sessions:
        session["runs"] = len({r["run"] for r in requests
                               if r.get("session_id") == session["session_id"]})
    summary = build_summary(runs, requests, sessions, ctx_iters, gen_iters,
                            capacity, rank_iters)

    notes = []
    if window[0] is not None or window[1] is not None:
        def edge(value: float | None, fallback: str) -> str:
            if value is None:
                return fallback
            stamp = datetime.fromtimestamp(value).astimezone()
            return (f"{stamp.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S} UTC "
                    f"({stamp:%H:%M:%S} local)")
        dropped = {key: sum(e["excluded"].get(key, 0) for e in loaded)
                   for key in ("requests", "ctx_iters", "gen_iters",
                               "undated_requests")}
        notes.append(
            f"Analysis window: {edge(window[0], 'run start')} → "
            f"{edge(window[1], 'run end')}. Everything outside it is excluded "
            f"from every figure below — {dropped['requests']:,} requests, "
            f"{dropped['ctx_iters']:,} prefill and {dropped['gen_iters']:,} "
            "decode iterations, and the GPU time of the requests dropped. Wall "
            "clock is measured inside the window only, so an idle stretch "
            "outside it is not charged to the workload."
            + (f" {dropped['undated_requests']:,} requests carried no timestamp "
               "(engine-only rows the audit never saw) and could not be placed "
               "in the window, so they are dropped too."
               if dropped["undated_requests"] else ""))
    if len(runs) > 1:
        spanning = sum(1 for s in sessions if s.get("runs", 1) > 1)
        notes.append(
            f"Merged from {len(runs)} runs. Time is summed per run, never "
            "measured across them, so the gaps between serving jobs are not "
            "charged to the workload."
            + (f" {spanning} sessions span more than one run — the gateway "
               "relayed them to a successor job, and their client-idle time "
               "includes the restart." if spanning else "")
            + " Sessions pair across runs because the id comes from the client;"
              " <code>rid</code> does not, being a per-process counter that"
              " restarts, so the key in the merged CSVs is"
              " <code>(run, rid)</code>. Every cross-source join happens inside"
              " a run before merging, so the collisions are cosmetic.")
    ambiguous = sum(r.route_client_ambiguous for r in runs)
    if ambiguous:
        notes.append(
            f"{ambiguous:,} client ids appeared under more than one context "
            "worker, so the routing columns are blank for the requests that "
            "resolve only by client id. <code>client_id</code> is a per-worker "
            "counter that restarts at 1 on every instance, so attaching one "
            "worker's decision to another's request would be worse than "
            "attaching none.")
    if not any(r.audit.exists() for r in runs):
        notes.append("No <code>anthropic_audit.jsonl</code>: sections 1 and 2 carry "
                     "only what the engine reported.")
    if summary["phases_without_gpu_time"]:
        notes.append(f"{summary['phases_without_gpu_time']} request phases reported "
                     "timing but no GPU breakdown (overlap scheduler).")
    if not capacity.get("capacity_tokens"):
        notes.append(
            "KV capacity in <em>tokens</em> not found in the worker logs "
            "(<code>Max KV cache blocks per sequence</code>), so per-rank session "
            "capacity is blank."
            + (f" The pool size in <em>slots</em> is known — "
               f"{capacity['slots_per_rank']:,.0f} per rank, recovered from the "
               "level counters — but it is deliberately not converted to tokens: "
               "with more than one pool group the slot sizes differ, and "
               "multiplying by <code>tokens_per_block</code> overstates the pool."
               if capacity.get("slots_per_rank") else "")
            + (" That single number is itself a poor summary: the pool ratio "
               "splits <em>bytes</em>, so a group whose blocks are large gets "
               "proportionally fewer slots. Per pool group the split is "
               + " / ".join(f"{v:,.0f}"
                            for v in capacity["slots_per_rank_by_pg"])
               + ", so the largest group can dominate the total while a small "
               "one saturates and evicts."
               if capacity.get("slots_per_rank_by_pg") else ""))

    negative_gaps = _annotate_thread_gaps(requests, threads)
    if negative_gaps:
        notes.append(
            f"{negative_gaps} thread parent links have the child starting before "
            "the parent finished. That is impossible by build_threads' own rule "
            "and means the rebuild is stale or wrong; those gaps are dropped "
            "rather than reported.")

    _write_csv(out / "requests.csv", requests)
    _write_csv(out / "sessions.csv", sessions)
    _write_csv(out / "ctx_iters.csv", ctx_iters)
    _write_csv(out / "ctx_rank_iters.csv", rank_iters)
    if rank_pg_iters:
        _write_csv(out / "ctx_rank_pg_iters.csv", rank_pg_iters)
    _write_csv(out / "gen_iters.csv", gen_iters)
    _write_csv(out / "summary.csv", [summary])
    (out / "REPORT.html").write_text(
        render_html(runs, requests, sessions, ctx_iters, gen_iters, rank_iters,
                    summary, notes, rank_pg_iters, capacity, threads, out),
        encoding="utf-8")

    label = (f"{len(runs)} runs merged" if len(runs) > 1 else runs[0].name)
    print(f"{label}: {len(requests)} requests, {len(sessions)} sessions, "
          f"{len(ctx_iters)} prefill / {len(gen_iters)} decode iterations -> {out}")
    for note in notes:
        print("  note: " + re.sub(r"<[^>]+>", "", note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
