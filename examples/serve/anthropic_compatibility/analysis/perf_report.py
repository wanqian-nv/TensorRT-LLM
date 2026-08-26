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
from collections import defaultdict
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
        "cache_affinity_active": route.get("cache_affinity_active"),
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

    Utilization is the mean context tokens across that instance's ranks over
    the per-rank token budget. Imbalance is (max - mean) / mean across the
    same ranks: 0 means every rank got the same prefill work this iteration.
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
            ctx_tokens = [r["num_ctx_tokens"] or 0 for r in ranks]
            mean = statistics.fmean(ctx_tokens) if ctx_tokens else 0.0
            peak = max(ctx_tokens, default=0)

            per_rank_hit = _rank_hit_rates(ranks, prev)
            for entry, hit in zip(ranks, per_rank_hit):
                rank_rows.append({
                    "worker": worker,
                    "rank": entry["rank"],
                    "series": f'{worker} r{entry["rank"]}',
                    "instance": entry["instance"],
                    "iter": iteration,
                    "timestamp": entry["timestamp"],
                    "kv_hit_rate_iter": hit,
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
           mean_line: bool = False) -> str:
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
        if not row.get("num_ctx_tokens"):
            continue
        acc = per.setdefault((row["worker"], row["rank"]),
                             {"iters": 0, "tokens": 0, "hit": [], "util": []})
        acc["iters"] += 1
        acc["tokens"] += row["num_ctx_tokens"]
        for key, field in (("hit", "kv_hit_rate_iter"), ("util", "kv_cache_util"),
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
            f'<td class="lead">{_num(statistics.fmean(acc["hit"]) if acc["hit"] else None, 3)}</td>'
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


def render_html(runs: list[Run], requests: list[dict], sessions: list[dict],
                ctx_iters: list[dict], gen_iters: list[dict],
                rank_iters: list[dict], summary: dict, notes: list[str],
                rank_pg_iters: list[dict] | None = None,
                capacity_meta: dict | None = None) -> str:
    stamps = [_parse_stamp(r.get("timestamp"))
              for r in (ctx_iters + gen_iters) if r.get("timestamp")]
    run_origin = min((t for t in stamps if t is not None), default=None)
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
    parts.append("<h3>Latency</h3><table><thead><tr><th>metric</th><th>n</th>"
                 "<th>mean</th><th>p50</th><th>p90</th><th>p99</th></tr></thead><tbody>")
    for label, field, ratio in (("TTFT (client)", "ttft_ms", False),
                                ("TTFT (engine)", "ttft_engine_ms", False),
                                ("Decode", "decode_ms", True),
                                ("E2E", "e2e_ms", False),
                                ("Prefill queue", "prefill_queue_ms", True),
                                ("Prefill", "prefill_ms", True),
                                ("KV transfer", "kv_transfer_ms", True),
                                ("Gap to next turn", "gap_to_next_turn_ms", False),
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
                 "finishing and the next starting in the same session — all "
                 "client-side time, whatever it went on. <b>Tool call</b> is the "
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
                 f"below {HIT_RATE_FLOOR:.0%} — listed with their cause in "
                 '<a href="#low-hit">the appendix</a>.</p>'
                 if low else '<p class="sub">No request fell below '
                 f'{HIT_RATE_FLOOR:.0%}.</p>')

    # ---- 2 ----
    parts.append("<h2>2 · Sessions</h2>")
    turns_total = sum(x["turns"] for x in sessions)
    parts.append(f'<p class="sub">{len(sessions)} sessions, {turns_total} turns '
                 f'({turns_total / len(sessions):.1f} per session).</p>'
                 if sessions else '<p class="sub">No sessions.</p>')
    footer = {"session_id": f"mean of {len(sessions)}", "_total": True}
    for key in ("turns", "span_ms", "isl_cached_mean", "isl_new_mean", "osl_mean",
                "kv_hit_rate", "ttft_sum_ms", "decode_sum_ms", "client_time_ms",
                "isl_max", "sessions_per_rank"):
        vals = [x[key] for x in sessions if x[key] is not None]
        footer[key] = statistics.fmean(vals) if vals else None
    footer["ranks_used"] = ""
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
    for row in sessions + [footer]:
        for key in ("ttft_sum_ms", "decode_sum_ms", "client_time_ms", "span_ms"):
            row[key + "_fmt"] = _dur(row.get(key))
    session_columns = [
        ("session_id", "session"), ("turns", "reqs"), ("span_ms_fmt", "span"),
        ("isl_cached_mean", "cached ISL"), ("isl_new_mean", "new ISL"),
        ("osl_mean", "OSL"), ("kv_hit_rate", "hit"),
        ("ttft_sum_ms_fmt", "prefill"), ("decode_sum_ms_fmt", "decode"),
        ("client_time_ms_fmt", "client"), ("isl_max", "max ISL"),
        ("sessions_per_rank", "fit/rank")]
    # Only worth a column when something routed: without an ADP route trace the
    # rank a request landed on is unknown, and an empty column reads as "one
    # rank" rather than "not measured".
    if any(x.get("ranks_used") for x in sessions):
        session_columns.append(("ranks_used", "ranks"))
    parts.append(_table(sessions + [footer], session_columns))
    parts.append('<p class="sub">'
                 "<code>reqs</code> is the request count; <code>cached ISL</code>, "
                 "<code>new ISL</code> and <code>OSL</code> are means per request, "
                 "not session totals — a 92-turn session would otherwise top every "
                 "one of them without any single turn being large. Totals are in "
                 "<code>sessions.csv</code>. "
                 + ("" if any(x.get("ranks_used") for x in sessions) else
                    "The <code>ranks</code> column is omitted: no ADP route trace "
                    "was recorded, so which rank served a request is unknown. "
                    "Set <code>TRTLLM_ADP_ROUTE_TRACE</code> to get it. ")
                 + "<code>client</code> = session span − union of its request "
                 "intervals, i.e. the time inside the session with no request in "
                 "flight: "
                 "tool execution plus client think time, measured without relying on "
                 "tool matching. <code>fit/rank</code> = KV capacity ÷ longest prompt, "
                 "i.e. how many such sessions one rank holds at once."
                 + ("" if summary["kv_capacity_tokens"] else
                    " Capacity was not in this run's logs, so it is blank.") + "</p>")

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
        iter_hits = _chart(rank_occupancy, "kv_hit_rate_iter",
                           "KV hit rate per rank (server block counters)",
                           "reused / (reused+missed)", pct=True,
                           origin=run_origin, series_field="series",
                           toggle=True)
        parts.append(iter_hits)
        notes_html = ["Per rank, from the KV manager's own block counters in "
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
                ("Hit rate", "kv_hit_rate_iter", 3, "kv_hit_rate_spread"),
                ("Pool filled", "kv_pool_filled", 3, "kv_pool_filled_spread"),
                ("Evicted tokens", "kv_evicted_tokens", 0, None),
                ("Pinned by in-flight", "kv_util_mean", 3, "kv_util_spread"),
                ("Device step ms", "device_step_time_ms", 1, "device_step_spread"),
                ("Host step ms", "host_step_time_ms", 1, None)):
            parts.append(_stat_row(
                label, [c[field] for c in ctx_iters], digits=digits,
                duration="ms" in label, with_spread=True,
                spread=[c.get(spread) for c in ctx_iters] if spread else None))
        parts.append("</tbody></table>")
        parts.append(_tier_table(occupancy, rank_iters))
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

    # An appendix, and closed by default: it is a per-request drill-down, read
    # after the four sections have said whether anything needs drilling into.
    if low:
        detail = []
        detail.append(f"<h3>Hit rate below {HIT_RATE_FLOOR:.0%} "
                     f"({len(low)} of {len(requests)})</h3>")
        if low:
            healthy = _stats([r["ttft_ms"] or r["ttft_engine_ms"]
                              for r in requests if not r["low_hit_rate"]])["p50"]
            rows = []
            for row in sorted(low, key=lambda r: -(r["ttft_ms"] or r["ttft_engine_ms"] or 0)):
                ttft = row["ttft_ms"] or row["ttft_engine_ms"]
                rows.append({
                    "request_index": row["request_index"],
                    "session_turn_index": row["session_turn_index"],
                    "ttft": _dur(ttft),
                    "vs_healthy": (f"{ttft / healthy:.1f}×"
                                   if ttft and healthy else "—"),
                    "kv_hit_rate": row["kv_hit_rate"],
                    "isl_total": row["isl_total"],
                    "cause": row.get("cause") or "",
                    "low_hit_rate": True,
                })
            detail.append(_table(rows, [
                ("request_index", "#"), ("session_turn_index", "turn"),
                ("ttft", "TTFT"), ("vs_healthy", "vs healthy p50"),
                ("kv_hit_rate", "hit"), ("isl_total", "ISL"),
                ("cause", "cause")], flag="low_hit_rate"))
            if not any(r["cause"] for r in rows):
                detail.append('<p class="sub">The <code>cause</code> column is filled by '
                             "the analyst, not the script: separating an evicted prefix "
                             "from a rewritten prompt needs the captured request bodies. "
                             "Write one line per row into a CSV of "
                             "<code>request_index,cause</code> and re-run with "
                             "<code>--causes</code>.</p>")
        else:
            detail.append('<p class="sub">None.</p>')
        parts.append(f'<details id="low-hit"><summary>Hit rate below '
                     f'{HIT_RATE_FLOOR:.0%} — {len(low)} of {len(requests)} '
                     f'requests</summary>{"".join(detail)}</details>')
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
                    summary, notes, rank_pg_iters, capacity),
        encoding="utf-8")

    label = (f"{len(runs)} runs merged" if len(runs) > 1 else runs[0].name)
    print(f"{label}: {len(requests)} requests, {len(sessions)} sessions, "
          f"{len(ctx_iters)} prefill / {len(gen_iters)} decode iterations -> {out}")
    for note in notes:
        print("  note: " + re.sub(r"<[^>]+>", "", note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
