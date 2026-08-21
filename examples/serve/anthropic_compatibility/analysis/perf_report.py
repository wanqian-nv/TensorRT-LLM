#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the four-section performance report for one serving run.

Reads whatever a run directory happens to hold -- the Anthropic audit log, the
per-request perf metrics the controller drained, the attention-DP routing
trace, and the per-rank iteration logs -- and emits four CSVs plus a single
self-contained HTML report.

    python3 analysis/perf_report.py <attempt_dir> [--out DIR]

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
import re
import statistics
from collections import defaultdict
from datetime import datetime
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

HIT_RATE_FLOOR = 0.90  # below this a request is called out for investigation


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


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
        self.perf = sorted(attempt_dir.glob("perf_metrics-*.jsonl"))
        self.route = sorted(attempt_dir.glob("adp_route_trace*.jsonl"))
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


def parse_iter_log(path: Path) -> list[dict]:
    """One row per (rank, iteration) from a worker's stdout log.

    The log is `key = value, ` pairs ending in a Python dict literal, and the
    file can carry stray control bytes from the launcher's tee, so it is read
    as latin-1 and matched loosely.
    """
    rows: list[dict] = []
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
            rows.append({
                "iter": int(row["iter"]),
                "rank": int(row.get("rank", -1)),
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
            out.append(record)
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
    perf_rows = []
    for path in run.perf:
        for wrapped in _read_jsonl(path):
            record = wrapped.get("record") or {}
            record["_source"] = wrapped.get("source")
            record["_drained_at"] = wrapped.get("drained_at")
            perf_rows.append(record)
    perf_rows = _pair_worker_records(perf_rows)

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
    route_index = _index_by_ids(decisions, ("req_id", "client_id"))

    tool_map = _tool_latencies(audit)
    rows: list[dict] = []
    for record in audit:
        rid = record.get("disagg_request_id") or record.get("engine_request_id")
        ctx_rid = record.get("ctx_request_id")
        perf = perf_index.get(str(rid)) or perf_index.get(str(ctx_rid)) or {}
        route = route_index.get(str(rid)) or route_index.get(str(ctx_rid)) or {}
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
        rows.append(_request_row({}, record, route_index.get(str(rid), {}), {}))
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
               ) -> tuple[float, float, float, bool, bool]:
    """Per-iteration KV deltas summed across one instance's ranks.

    ``have_alloc`` is tracked apart from ``have_delta`` because logs written
    before the alloc counters existed would otherwise difference None-as-zero
    and report a confident zero evictions where there is no measurement.
    """
    reused = missed = evicted = 0.0
    have_delta = have_alloc = False
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
        prev[entry["rank"]] = entry
    return reused, missed, evicted, have_delta, have_alloc


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
        per_iter: dict[int, list[dict]] = defaultdict(list)
        for entry in parse_iter_log(path):
            per_iter[entry["iter"]].append(entry)

        prev: dict[int, dict] = {}
        for iteration in sorted(per_iter):
            ranks = per_iter[iteration]
            batch = [r["num_scheduled_requests"] for r in ranks]
            tokens = [r["num_generation_tokens"] or 0 for r in ranks]
            batch_mean = statistics.fmean(batch) if batch else 0.0
            batch_total, tokens_total = sum(batch), sum(tokens)
            per_rank_hit = _rank_hit_rates(ranks, prev)
            reused, missed, evicted, have_delta, have_alloc = _kv_deltas(ranks, prev)
            device = [r["device_step_time_ms"] for r in ranks
                      if r["device_step_time_ms"] is not None]
            util = [r["kv_cache_util"] for r in ranks if r["kv_cache_util"] is not None]
            rows.append({
                "worker": worker,
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
                "kv_evicted_tokens": evicted * 128 if have_alloc else None,
                "host_step_time_ms": statistics.fmean(
                    [r["host_step_time_ms"] for r in ranks
                     if r["host_step_time_ms"] is not None] or [0.0]),
                "device_step_time_ms": statistics.fmean(device) if device else None,
                "timestamp": ranks[0]["timestamp"],
            })
    return rows


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
                    ) -> tuple[list[dict], dict, list[dict]]:
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
    capacity: dict[str, Any] = {}
    for path in run.prefill_logs():
        worker = run.worker_name(path)
        blocks, tokens_per_block = parse_kv_capacity(path)
        if blocks and tokens_per_block:
            capacity = {
                "primary_blocks": blocks,
                "tokens_per_block": tokens_per_block,
                "capacity_tokens": blocks * tokens_per_block,
            }
        tokens_per_block = tokens_per_block or 128

        per_iter: dict[int, list[dict]] = defaultdict(list)
        for entry in parse_iter_log(path):
            per_iter[entry["iter"]].append(entry)

        prev: dict[int, dict] = {}
        for iteration in sorted(per_iter):
            ranks = per_iter[iteration]
            ctx_tokens = [r["num_ctx_tokens"] or 0 for r in ranks]
            mean = statistics.fmean(ctx_tokens) if ctx_tokens else 0.0
            peak = max(ctx_tokens, default=0)

            per_rank_hit = _rank_hit_rates(ranks, prev)
            for entry, hit in zip(ranks, per_rank_hit):
                rank_rows.append({
                    "worker": worker,
                    "rank": entry["rank"],
                    "series": f'{worker} r{entry["rank"]}',
                    "iter": iteration,
                    "timestamp": entry["timestamp"],
                    "kv_hit_rate_iter": hit,
                    "kv_cache_util": entry["kv_cache_util"],
                    "num_ctx_tokens": entry["num_ctx_tokens"],
                    "utilization": ((entry["num_ctx_tokens"] or 0) / max_num_tokens
                                    if max_num_tokens else None),
                })
            reused, missed, evicted, have_delta, have_alloc = _kv_deltas(ranks, prev)

            device = [r["device_step_time_ms"] for r in ranks
                      if r["device_step_time_ms"] is not None]
            rows.append({
                "worker": worker,
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
                "kv_evicted_tokens": evicted * tokens_per_block if have_alloc else None,
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
    return rows, capacity, rank_rows


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
    for path in run.perf:
        for wrapped in _read_jsonl(path):
            record = wrapped.get("record") or {}
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


def build_summary(runs: list[Run], requests: list[dict], sessions: list[dict],
                  ctx_iters: list[dict], gen_iters: list[dict],
                  capacity: dict) -> dict:
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
            sum(c["kv_evicted_tokens"] or 0 for c in ctx_iters) if ctx_iters else None),
        "kv_capacity_tokens": capacity.get("capacity_tokens"),
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
           marker_only: bool = False) -> str:
    """One series per prefill instance against wall clock, as an inline PNG.

    Three choices worth stating, because the obvious version of this chart
    lies. The x axis is elapsed seconds, not the iteration counter: iteration
    duration spans three orders of magnitude on agent traffic (median 0.6s,
    p99 580s), so an index axis gives a ten-minute idle the same width as a
    20ms step. The line breaks wherever the gap exceeds ten times the median,
    because a continuous line across a ten-minute silence claims the metric
    held that value while nothing was running. Points are marked, so a burst
    of five events does not read as one thick segment.
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
        axis.plot(xs, ys, linewidth=0 if marker_only else 1.0, label=name,
                  color=colors[index % len(colors)],
                  marker="o", markersize=2.6 if marker_only else 1.8, markevery=1)
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return ('<figure class="chart"><img alt="%s" src="data:image/png;base64,%s">'
            '<script type="application/json">%s</script></figure>'
            % (title, base64.b64encode(buffer.getvalue()).decode(),
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
        for key, field in (("hit", "kv_hit_rate_iter"), ("util", "kv_cache_util")):
            if row.get(field) is not None:
                acc[key].append(row[field])
    if not per:
        return ""
    total = sum(a["tokens"] for a in per.values()) or 1
    body = []
    for (worker, rank), acc in sorted(per.items()):
        share = acc["tokens"] / total
        body.append(
            f"<tr><td>{worker} r{rank}</td><td>{acc['iters']:,}</td>"
            f"<td>{acc['tokens']:,}</td>"
            f'<td class="imb">{share * 100:.1f}%</td>'
            f'<td class="lead">{_num(statistics.fmean(acc["hit"]) if acc["hit"] else None, 3)}</td>'
            f'<td>{_num(statistics.fmean(acc["util"]) if acc["util"] else None, 3)}</td></tr>')
    shares = [a["tokens"] / total for a in per.values()]
    skew = max(shares) / min(shares) if min(shares) else None
    return ("<table><thead><tr><th>rank</th><th>prefill iters</th>"
            "<th>ctx tokens</th>"
            '<th class="imb">share</th><th>hit rate</th><th>KV util</th>'
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
      tip.style.left = cx + 'px';
      tip.style.top = (data.box[1] * r.height + 10) + 'px';
      rule.style.left = cx + 'px';
      tip.classList.add('on'); rule.classList.add('on');
      return;
    }
    var best = null;
    data.series.forEach(function (s) {
      s.pts.forEach(function (p) {
        var d = Math.abs(p[0] - x);
        if (!best || d < best.d) best = {d: d, x: p[0], y: p[1], name: s.name};
      });
    });
    if (!best) return;
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
                rank_iters: list[dict], summary: dict, notes: list[str]) -> str:
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
        parts.append(_chart(rank_occupancy or occupancy, "kv_util_mean"
                            if not rank_occupancy else "kv_cache_util",
                            "KV cache utilization per rank",
                            "used / max blocks", pct=True, origin=run_origin,
                            series_field="series" if rank_occupancy else "worker"))
        # The KV manager's own reused/missed block counters, differenced per
        # rank. Pooling ranks would average away the rank that is missing while
        # its peers hit -- the one case the curve exists to show.
        iter_hits = _chart(rank_occupancy, "kv_hit_rate_iter",
                           "KV hit rate per rank (server block counters)",
                           "reused / (reused+missed)", pct=True,
                           origin=run_origin, series_field="series")
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
                            "mean ctx tokens / budget", pct=True, origin=run_origin))
        parts.append(_chart(ctx_iters, "imbalance", "Rank imbalance",
                            "(max − mean) / mean", origin=run_origin))
        parts.append("<table><thead><tr><th>metric</th><th>n</th><th>mean</th>"
                     '<th class="imb">imbalance</th>'
                     "<th>p50</th><th>p90</th><th>p99</th></tr></thead><tbody>")
        for label, field, digits, spread in (
                ("Utilization", "utilization", 3, "imbalance"),
                ("Hit rate", "kv_hit_rate_iter", 3, "kv_hit_rate_spread"),
                ("Evicted tokens", "kv_evicted_tokens", 0, None),
                ("KV util", "kv_util_mean", 3, "kv_util_spread"),
                ("Device step ms", "device_step_time_ms", 1, "device_step_spread"),
                ("Host step ms", "host_step_time_ms", 1, None)):
            parts.append(_stat_row(
                label, [c[field] for c in ctx_iters], digits=digits,
                duration="ms" in label, with_spread=True,
                spread=[c.get(spread) for c in ctx_iters] if spread else None))
        parts.append("</tbody></table>")
        parts.append(_rank_totals(rank_iters))
        parts.append('<p class="sub">'
                     "The imbalance column is (max − mean) / mean across that "
                     "instance's ranks, averaged over iterations: 0 means every rank "
                     "carried the same amount that iteration. For "
                     "<b>utilization</b> under attention-DP it is pinned near "
                     "<code>ranks − 1</code> by construction — one prefill occupies "
                     "one rank — so read routing skew off the per-rank table above, "
                     "not off this column. For <b>KV util</b> and <b>hit rate</b> it "
                     "is meaningful: those are rank-resident state, and a spread "
                     "there means the ranks really do hold different amounts. "
                     "Evicted tokens come from "
                     "<code>alloc_total − alloc_new</code>, which counts blocks taken "
                     "while still holding reusable content — an upper bound, since a "
                     "freed partial block lands there too. "
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


def load_run(attempt_dir: Path) -> dict:
    """Everything one attempt directory yields, tagged with its run name."""
    run = Run(attempt_dir)
    token_budget = _budget(attempt_dir, "max_num_tokens",
                           ("ctx_config.yaml", "server_config.yaml"), 8192)
    batch_budget = _budget(attempt_dir, "max_batch_size",
                           ("gen_config.yaml", "server_config.yaml"), 128)
    ctx_iters, capacity, rank_iters = build_ctx_iters(run, token_budget)
    gen_iters = build_gen_iters(run, batch_budget)
    requests = build_requests(run)
    for rows in (ctx_iters, gen_iters, requests, rank_iters):
        for row in rows:
            row["run"] = run.name
    return {"run": run, "requests": requests, "ctx_iters": ctx_iters,
            "gen_iters": gen_iters, "capacity": capacity,
            "rank_iters": rank_iters}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempt_dir", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--causes", type=Path, default=None,
                        help="CSV of request_index,cause to annotate the "
                             "low-hit-rate table")
    args = parser.parse_args()

    loaded = [load_run(d) for d in args.attempt_dir]
    runs = [entry["run"] for entry in loaded]
    out = args.out or (args.attempt_dir[0] / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    requests = [r for entry in loaded for r in entry["requests"]]
    ctx_iters = [r for entry in loaded for r in entry["ctx_iters"]]
    gen_iters = [r for entry in loaded for r in entry["gen_iters"]]
    rank_iters = [r for entry in loaded for r in entry["rank_iters"]]
    capacity = next((e["capacity"] for e in loaded if e["capacity"].get("capacity_tokens")),
                    {})

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
    summary = build_summary(runs, requests, sessions, ctx_iters, gen_iters, capacity)

    notes = []
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
    if not any(r.audit.exists() for r in runs):
        notes.append("No <code>anthropic_audit.jsonl</code>: sections 1 and 2 carry "
                     "only what the engine reported.")
    if summary["phases_without_gpu_time"]:
        notes.append(f"{summary['phases_without_gpu_time']} request phases reported "
                     "timing but no GPU breakdown (overlap scheduler).")
    if not capacity.get("capacity_tokens"):
        notes.append("KV capacity not found in the worker logs "
                     "(<code>Max KV cache blocks per sequence</code>), so per-rank "
                     "session capacity is blank.")

    _write_csv(out / "requests.csv", requests)
    _write_csv(out / "sessions.csv", sessions)
    _write_csv(out / "ctx_iters.csv", ctx_iters)
    _write_csv(out / "ctx_rank_iters.csv", rank_iters)
    _write_csv(out / "gen_iters.csv", gen_iters)
    _write_csv(out / "summary.csv", [summary])
    (out / "REPORT.html").write_text(
        render_html(runs, requests, sessions, ctx_iters, gen_iters, rank_iters,
                    summary, notes),
        encoding="utf-8")

    label = (f"{len(runs)} runs merged" if len(runs) > 1 else runs[0].name)
    print(f"{label}: {len(requests)} requests, {len(sessions)} sessions, "
          f"{len(ctx_iters)} prefill / {len(gen_iters)} decode iterations -> {out}")
    for note in notes:
        print("  note: " + re.sub(r"<[^>]+>", "", note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
