#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 + 1: rebuild conversation threads, then account for every tool call.

    python3 analysis/threads_and_tools.py <reports_dir> <attempt_dir>…

Stage 0 -- `session_id` is not a thread. 512 of this run's 620 sessions carry
concurrent in-flight requests (subagent fan-out), so "the next request in the
session" is often a different branch: 21.5% of the per-session gaps in
`requests.csv` are negative. Threads are rebuilt here by prefix-matching the
per-message digests, so every request gets its true parent and a gap measured
against that parent.

Stage 1 -- every tool call is resolved to (emitting turn, surfacing turn) and
classified by whether the client actually had to wait for it. Three tools
return a handle immediately and finish later (`Bash run_in_background`,
`Agent`, `Monitor`), one blocks for a poll window (`TaskOutput`), and one
blocks for its full timeout before being backgrounded (`Bash` that overran).
Folding those into one "tool latency" number is what makes the existing
`tool_latency_ms` column unreadable, so they are kept apart.

Writes `threads.csv`, `tool_calls.csv` and a `stage01_report.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)

# Claude Code rewrites the trailing message between turns (thinking
# retention), so a clean continuation shows a common prefix of exactly
# len(parent) - 1. Inherited from classify_causes.py, which measured it.
TAIL_EDIT_SLACK = 1

# --- result-text signatures, all verbatim from this run's captures ---------
BG_LAUNCH = re.compile(r"^Command running in background with ID: (\w+)")
BG_TIMEOUT = re.compile(
    r"^Command did not complete within its (\d+)s timeout and was moved to "
    r"the background \(ID: (\w+)\)")
BG_AGENT = re.compile(r"^Async agent launched successfully")
BG_MONITOR = re.compile(r"^Monitor started \(task (\w+), timeout (\d+)ms\)")
BG_WAKEUP = re.compile(r"^Next wakeup scheduled for ")
POLL_STATUS = re.compile(r"<retrieval_status>(\w+)</retrieval_status>")
POLL_TASK = re.compile(r"<task_id>(\w+)</task_id>")
POLL_STATE = re.compile(r"<status>(\w+)</status>")


def classify_result(tool: str, head: str) -> tuple[str, dict]:
    """(waiting mode, extra facts) for one tool_result.

    handle_now  -- returned a handle at once; the real work outlives the call
    blocked_out -- the client blocked for the tool's whole timeout, then it
                   was backgrounded; the wait is real and equals the timeout
    poll_hit / poll_miss -- a retrieval against work already running
    sync        -- an ordinary call the client waited on
    """
    m = BG_TIMEOUT.match(head)
    if m:
        return "blocked_out", {"task_id": m.group(2),
                               "timeout_s": float(m.group(1))}
    m = BG_LAUNCH.match(head)
    if m:
        return "handle_now", {"task_id": m.group(1)}
    if BG_AGENT.match(head):
        return "handle_now", {}
    m = BG_MONITOR.match(head)
    if m:
        return "handle_now", {"task_id": m.group(1),
                              "timeout_s": float(m.group(2)) / 1000.0}
    if BG_WAKEUP.match(head):
        return "handle_now", {}
    if tool == "TaskOutput" or POLL_STATUS.search(head):
        status = POLL_STATUS.search(head)
        task = POLL_TASK.search(head)
        state = POLL_STATE.search(head)
        got = status.group(1) if status else "?"
        extra = {"poll_status": got}
        if task:
            extra["task_id"] = task.group(1)
        if state:
            extra["task_state"] = state.group(1)
        return ("poll_hit" if got == "success" else "poll_miss"), extra
    return "sync", {}


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# stage 0 -- threads
# --------------------------------------------------------------------------
def build_threads(facts: dict, meta: dict) -> None:
    """Give every request a parent, a thread id and a gap against that parent.

    The parent is the deepest earlier request in the same session whose whole
    message list prefixes this one's (one trailing rewrite allowed). Indexing
    prefixes by (length, hash) makes this a handful of dict lookups per
    request rather than an all-pairs scan; the largest session here has 840
    turns, which the quadratic form would not survive comfortably.
    """
    by_session: dict[str, list[str]] = defaultdict(list)
    for aid, fact in facts.items():
        by_session[fact.get("client_session_id") or ""].append(aid)

    for session, ids in by_session.items():
        ids.sort(key=lambda a: (meta[a]["started_at"] or 0.0, a))
        exact: dict[tuple, list[str]] = defaultdict(list)
        edited: dict[tuple, list[str]] = defaultdict(list)
        for aid in ids:
            msgs = facts[aid]["msgs"]
            exact[(len(msgs), hash(tuple(msgs)))].append(aid)
            if msgs:
                edited[(len(msgs) - 1, hash(tuple(msgs[:-1])))].append(aid)

        for aid in ids:
            msgs = facts[aid]["msgs"]
            mine = meta[aid]
            parent = None
            # Deepest first: the immediate parent, not a distant ancestor.
            for k in range(len(msgs) - 1, 0, -1):
                key = (k, hash(tuple(msgs[:k])))
                pool = exact.get(key, []) + edited.get(key, [])
                # Only an earlier request can be a parent, and it must be
                # strictly shorter or the two are the same turn re-sent.
                cands = [c for c in pool
                         if c != aid
                         and (meta[c]["started_at"] or 0.0) <= (mine["started_at"] or 0.0)
                         and len(facts[c]["msgs"]) < len(msgs)]
                if cands:
                    parent = max(cands, key=lambda c: (len(facts[c]["msgs"]),
                                                       meta[c]["started_at"] or 0.0))
                    break
            mine["parent"] = parent
            mine["prefix_depth"] = len(facts[parent]["msgs"]) if parent else 0

    # thread id = the root each chain reaches; depth = position in the chain
    for aid in facts:
        seen, node = set(), aid
        while meta[node].get("parent") and node not in seen:
            seen.add(node)
            node = meta[node]["parent"]
        meta[aid]["thread_id"] = node
        meta[aid]["thread_depth"] = len(seen) + 1

    for aid, mine in meta.items():
        parent = mine.get("parent")
        mine["true_gap_ms"] = None
        if parent and meta[parent]["finished_at"] and mine["started_at"]:
            mine["true_gap_ms"] = (
                mine["started_at"] - meta[parent]["finished_at"]) * 1000.0


# --------------------------------------------------------------------------
# stage 1 -- tools
# --------------------------------------------------------------------------
def build_tool_calls(facts: dict, meta: dict, emitted: dict) -> list[dict]:
    """One row per tool_use id, with both timings kept separate."""
    # Identity and input come from whichever capture first carried the block.
    spec: dict[str, dict] = {}
    order = sorted(facts, key=lambda a: (meta[a]["started_at"] or 0.0, a))
    for aid in order:
        for use in facts[aid]["uses"]:
            if use["id"] and use["id"] not in spec:
                spec[use["id"]] = {"name": use["name"], "in": use["in"],
                                   "in_bytes": use["in_bytes"]}
    # First request whose history carries the result is where it surfaced.
    surfaced: dict[str, dict] = {}
    for aid in order:
        for res in facts[aid]["results"]:
            if res["id"] and res["id"] not in surfaced:
                surfaced[res["id"]] = {"aid": aid, **res}
    # ... and the first that carries its completion notice is where the work
    # actually ended. Backgrounded calls surface a handle at once, so for them
    # `t_surface` measures the handshake and this measures the job.
    notified: dict[str, dict] = {}
    for aid in order:
        for note in facts[aid].get("notes") or []:
            if note["id"] and note["id"] not in notified:
                notified[note["id"]] = {"aid": aid, **note}

    rows = []
    for use_id, info in spec.items():
        emitter = emitted.get(use_id)
        res = surfaced.get(use_id)
        mode, extra = ("unmatched", {})
        if res:
            mode, extra = classify_result(info["name"], res["head"])
        row = {
            "tool_use_id": use_id,
            "tool": info["name"],
            "input_bytes": info["in_bytes"],
            "run_in_background": info["in"].get("run_in_background"),
            "subagent_type": info["in"].get("subagent_type"),
            "declared_timeout": info["in"].get("timeout"),
            "emitter_aid": emitter,
            "emitter_session": meta[emitter]["session_id"] if emitter in meta else None,
            "emitter_thread": meta[emitter].get("thread_id") if emitter in meta else None,
            "surfacer_aid": res["aid"] if res else None,
            "wait_mode": mode,
            "is_error": res["err"] if res else None,
            "result_chars": res["chars"] if res else None,
            "task_id": extra.get("task_id"),
            "poll_status": extra.get("poll_status"),
            "task_state": extra.get("task_state"),
            "bg_timeout_s": extra.get("timeout_s"),
            "t_surface_ms": None,
            "turns_later": None,
            "t_complete_ms": None,
            "complete_turns_later": None,
            "notify_status": None,
        }
        note = notified.get(use_id)
        if note:
            row["notify_status"] = note.get("status")
            if emitter in meta and note["aid"] in meta:
                a = meta[emitter]["finished_at"]
                b = meta[note["aid"]]["started_at"]
                if a is not None and b is not None:
                    row["t_complete_ms"] = (b - a) * 1000.0
                da = meta[emitter].get("thread_depth")
                db = meta[note["aid"]].get("thread_depth")
                if (meta[emitter].get("thread_id") == meta[note["aid"]].get("thread_id")
                        and da is not None and db is not None):
                    row["complete_turns_later"] = db - da
        if emitter in meta and res and res["aid"] in meta:
            a, b = meta[emitter]["finished_at"], meta[res["aid"]]["started_at"]
            if a is not None and b is not None:
                row["t_surface_ms"] = (b - a) * 1000.0
            # Distance along the rebuilt thread, not wall-clock ordering.
            depth_a = meta[emitter].get("thread_depth")
            depth_b = meta[res["aid"]].get("thread_depth")
            same = meta[emitter].get("thread_id") == meta[res["aid"]].get("thread_id")
            if same and depth_a is not None and depth_b is not None:
                row["turns_later"] = depth_b - depth_a
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("attempt_dir", type=Path, nargs="+")
    args = parser.parse_args()
    work = args.reports_dir / "_threadstudy"

    with (args.reports_dir / "requests.csv").open(encoding="utf-8") as handle:
        requests = list(csv.DictReader(handle))
    in_window = {r["audit_request_id"] for r in requests}
    by_aid = {r["audit_request_id"]: r for r in requests}
    print(f"requests.csv: {len(requests):,} in-window requests", file=sys.stderr)

    facts: dict[str, dict] = {}
    with (work / "capture_facts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            facts[rec["audit_request_id"]] = rec
    print(f"captures: {len(facts):,}", file=sys.stderr)

    # Timing and the authoritative emitter come from the audit, which covers
    # requests outside the report window too -- a thread must not lose its
    # root just because the root predates the analysis window.
    meta: dict[str, dict] = {}
    emitted: dict[str, str] = {}
    from datetime import datetime
    def parse(stamp):
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            return None
    for attempt in args.attempt_dir:
        with (attempt / "anthropic_audit.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                aid = rec.get("audit_request_id")
                if aid not in facts:
                    continue
                meta[aid] = {
                    "session_id": rec.get("client_session_id"),
                    "started_at": parse(rec.get("started_at")),
                    "finished_at": parse(rec.get("finished_at")),
                    "in_window": aid in in_window,
                }
                for call in (rec.get("response") or {}).get("tool_calls_emitted") or []:
                    if call.get("id"):
                        emitted[call["id"]] = aid
    print(f"audit: {len(meta):,} matched, {len(emitted):,} emitted tool calls",
          file=sys.stderr)

    build_threads(facts, meta)
    tool_rows = build_tool_calls(facts, meta, emitted)
    print(f"tool calls resolved: {len(tool_rows):,}", file=sys.stderr)

    with (work / "threads.csv").open("w", newline="", encoding="utf-8") as handle:
        cols = ["audit_request_id", "session_id", "thread_id", "thread_depth",
                "parent", "prefix_depth", "started_at", "finished_at",
                "true_gap_ms", "in_window", "request_index", "kv_hit_rate",
                "isl_total", "isl_cached", "osl", "match_len_best",
                "match_len_chosen", "routed_rank", "session_turn_index",
                "gap_to_next_turn_ms", "tool_names"]
        out = csv.DictWriter(handle, fieldnames=cols)
        out.writeheader()
        for aid, mine in meta.items():
            src = by_aid.get(aid, {})
            out.writerow({"audit_request_id": aid, **{
                k: mine.get(k) for k in ("session_id", "thread_id", "thread_depth",
                                         "parent", "prefix_depth", "started_at",
                                         "finished_at", "true_gap_ms", "in_window")},
                **{k: src.get(k) for k in ("request_index", "kv_hit_rate", "isl_total",
                                           "isl_cached", "osl", "match_len_best",
                                           "match_len_chosen", "routed_rank",
                                           "session_turn_index",
                                           "gap_to_next_turn_ms", "tool_names")}})

    with (work / "tool_calls.csv").open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=list(tool_rows[0].keys()))
        out.writeheader()
        out.writerows(tool_rows)
    print(f"wrote {work}/threads.csv and {work}/tool_calls.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
