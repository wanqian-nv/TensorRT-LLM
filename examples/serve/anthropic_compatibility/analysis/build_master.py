#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One row per request, with everything: ids, thread, turn, gaps, tools, cache.

    python3 analysis/build_master.py <reports_dir>

Writes `requests_master.csv`. The study produced three tables at three grains
-- per request, per tool call, per thread -- and only the first is worth
keeping separate: this folds the thread-level attributes down onto each
request and the tool-level rows up into per-request aggregates.

`_threadstudy/tool_calls.csv` stays as the one drill-down, because 17,537 tool
calls do not fit one per request; the `tool_*` columns here summarise it.
Rows cover every captured request, not just the report window, so a thread
whose root predates the window is still complete -- `in_window` says which
rows carry engine metrics.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# from requests.csv, kept verbatim -- the ids you cross-reference with, then
# the engine-side measurements
IDS = ["request_index", "rid", "ctx_request_id", "anthropic_message_id",
       "audit_request_id", "run", "message_capture_file"]
# Four columns perf_report.py derives by sorting a session by wall clock, which
# is the assumption the thread rebuild exists to replace. They are carried
# through for comparison and renamed `legacy_*` so nothing picks them up by
# habit -- measured on one run, the share of rows each gets wrong:
#   session_turn_index        96.7%  disagrees with thread_turn
#   gap_to_next_turn_ms       22.8%  negative (paired with a concurrent sibling)
#   cache_realization         21.7%  above 1.0, which is physically impossible
#   realization                9.5%  same
# `prompt_lcp_tokens`, `previous_prompt_tokens`, `lcp_opportunity` and
# `lcp_retention` come from the same gateway-side tracker but showed no
# out-of-range values, so they keep their names; treat them with the same
# suspicion on a session that interleaves threads.
LEGACY = {"session_turn_index": "legacy_session_turn_index",
          "gap_to_next_turn_ms": "legacy_gap_to_next_in_session_ms",
          "cache_realization": "legacy_cache_realization",
          "realization": "legacy_realization"}

PERF = ["status", "stream", "isl_total", "isl_cached", "isl_new", "osl",
        "kv_hit_rate", "prompt_lcp_tokens", "previous_prompt_tokens",
        "lcp_opportunity", "lcp_retention", "cache_realization", "realization",
        "low_hit_rate", "cause",
        "ttft_ms", "ttft_engine_ms", "e2e_ms", "decode_ms", "prefill_queue_ms",
        "prefill_ms", "kv_transfer_ms", "kv_transfer_bytes", "engine_decode_ms",
        "ctx_first_iter", "ctx_last_iter", "gen_first_iter", "gen_last_iter",
        "ctx_blocks_total", "ctx_blocks_new", "ctx_blocks_reused",
        "gen_blocks_total",
        "routed_rank", "route_phase", "route_iter", "route_log_iter",
        "match_len_chosen", "match_len_best", "cache_affinity_active",
        "effective_added", "affinity_regret"]


def kind_of(fact: dict) -> str:
    head = fact.get("sys_head") or ""
    ntools = len(fact.get("tool_names") or [])
    if "cc_is_subagent=true" in head:
        return "subagent"
    if ntools == 0:
        return "title" if fact.get("sys_chars") == 1327 else "no-tools"
    if "You are Claude Code" in head:
        return "main"
    if "You are a Claude agent" in head:
        return "sdk"
    return "other"


def main() -> int:
    root = Path(sys.argv[1])
    work = root / "_threadstudy"

    reqs = {r["audit_request_id"]: r
            for r in csv.DictReader((root / "requests.csv").open(encoding="utf-8"))}
    threads = list(csv.DictReader((work / "threads.csv").open(encoding="utf-8")))
    kinds = {k["thread_id"]: k
             for k in csv.DictReader((work / "thread_kinds.csv").open(encoding="utf-8"))}
    calls = list(csv.DictReader((work / "tool_calls.csv").open(encoding="utf-8")))
    facts = {}
    with (work / "capture_facts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            facts[rec["audit_request_id"]] = rec

    # tool rows up: aggregate onto the request that emitted them
    by_emitter = defaultdict(list)
    for c in calls:
        if c["emitter_aid"]:
            by_emitter[c["emitter_aid"]].append(c)

    # The wait that *follows* a turn, so one row carries the gap on both sides.
    # A turn can have more than one child -- 164 do here, from retries and
    # parallel continuations -- so take the earliest, which is the next turn in
    # the obvious sense, and publish the fan-out in `n_children` rather than
    # letting an arbitrary branch stand in for it. Keying a plain dict on the
    # parent silently kept whichever child was written last.
    kids = defaultdict(list)
    for t in threads:
        if t["parent"]:
            kids[t["parent"]].append(t)
    child_gap, n_children = {}, {}
    for parent, group in kids.items():
        first = min(group, key=lambda t: _f(t["started_at"]) or 0.0)
        child_gap[parent] = _f(first["true_gap_ms"])
        n_children[parent] = len(group)

    rows = []
    for t in threads:
        aid = t["audit_request_id"]
        req = reqs.get(aid, {})
        kind = kinds.get(t["thread_id"], {})
        fact = facts.get(aid, {})
        mine = by_emitter.get(aid, [])
        surf = [_f(c["t_surface_ms"]) for c in mine if _f(c["t_surface_ms"]) is not None]
        job = [_f(c["t_complete_ms"]) for c in mine if _f(c["t_complete_ms"]) is not None]
        modes = sorted({c["wait_mode"] for c in mine})
        parent = t["parent"]

        rows.append({
            **{k: req.get(k, "") for k in IDS},
            "audit_request_id": aid,
            "in_window": t["in_window"],
            # --- session / thread / turn ---
            "session_id": t["session_id"],
            "thread_id": t["thread_id"],
            "thread_turn": t["thread_depth"],
            "thread_total_turns": kind.get("turns", ""),
            "thread_kind": kind_of(fact),
            "is_subagent": t["is_subagent"],
            "subagent_type": kind.get("subagent_type", ""),
            "thread_n_tools": kind.get("n_tools", ""),
            "thread_opening": (kind.get("opening", "") or "")[:120],
            "parent_aid": parent or "",
            "parent_rid": reqs.get(parent, {}).get("rid", "") if parent else "",
            "link_key": t["link_key"],
            # --- timing ---
            "started_at": t["started_at"],
            "finished_at": t["finished_at"],
            "gap_from_parent_ms": t["true_gap_ms"],
            "gap_to_child_ms": child_gap.get(aid, ""),
            "n_children": n_children.get(aid, 0),
            "legacy_session_turn_index": req.get("session_turn_index", ""),
            "legacy_gap_to_next_in_session_ms": req.get("gap_to_next_turn_ms", ""),
            # --- tools this turn emitted ---
            "tool_count": len(mine),
            "tool_names": "|".join(sorted({c["tool"] for c in mine if c["tool"]})),
            "tool_wait_modes": "|".join(modes),
            "tool_bg_count": sum(1 for c in mine
                                 if c["wait_mode"] in ("handle_now", "blocked_out")),
            "tool_surface_max_ms": max(surf) if surf else "",
            "tool_job_max_ms": max(job) if job else "",
            "chain_len": t["chain_len"],
            "parent_chain_len": t["parent_chain_len"],
            **{LEGACY.get(k, k): req.get(k, "") for k in PERF},
        })

    rows.sort(key=lambda r: (_f(r["started_at"]) or 0.0, str(r["rid"])))
    dest = root / "requests_master.csv"
    with dest.open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        out.writeheader()
        out.writerows(rows)
    inw = sum(1 for r in rows if r["in_window"] == "True")
    print(f"{dest}: {len(rows):,} rows ({inw:,} in-window), "
          f"{len(rows[0])} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
