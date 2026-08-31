#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What each reconstructed thread is, and which of them are subagents.

    python3 analysis/thread_kinds.py <reports_dir>

A thread's identity lives in the tool set it was handed: the main agent gets
the full set, a subagent gets whatever its definition allows, and a one-shot
extraction call gets none at all. Threads are grouped on that signature, then
each root is matched back to the `Agent` call whose `prompt` opens it -- which
is what proves a thread is a subagent rather than merely looking like one.

Writes `thread_kinds.csv` and prints the summary.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# The harness wraps every opening message in a <system-reminder> block, so the
# first 160 characters of a thread root are the same boilerplate on 63% of
# threads. Dropping those blocks is what makes the column say what the thread
# is actually doing.
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def norm(text: str) -> str:
    """Whitespace-flatten so a prompt matches however it was re-wrapped."""
    return " ".join((text or "").split())


def task_text(text: str) -> str:
    """The opening with the harness's own preamble removed."""
    return norm(_REMINDER.sub(" ", text or ""))


def main() -> int:
    work = Path(sys.argv[1]) / "_threadstudy"
    rows = list(csv.DictReader((work / "threads.csv").open(encoding="utf-8")))
    calls = list(csv.DictReader((work / "tool_calls.csv").open(encoding="utf-8")))
    facts = {}
    with (work / "capture_facts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            facts[rec["audit_request_id"]] = rec

    by_thread: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_thread[row["thread_id"]].append(row)
    for turns in by_thread.values():
        turns.sort(key=lambda r: _f(r["started_at"]) or 0.0)

    emitter = {c["tool_use_id"]: c["emitter_aid"] for c in calls}
    thread_of = {r["audit_request_id"]: r["thread_id"] for r in rows}
    finished = {r["audit_request_id"]: _f(r["finished_at"]) for r in rows}

    # Every Agent launch, with the prompt it carried.
    launches: dict[str, dict] = {}
    for rec in facts.values():
        for use in rec["uses"]:
            if use["name"] == "Agent" and use["id"] not in launches:
                launches[use["id"]] = {
                    "prompt": norm(use["in"].get("prompt", "")),
                    "subagent_type": use["in"].get("subagent_type") or "",
                    "description": use["in"].get("description") or "",
                }
    agent_calls = [c for c in calls if c["tool"] == "Agent"]

    # A launch owns the earliest thread that (a) opens with its prompt and
    # (b) starts after the launching turn finished. Both halves are needed:
    # every turn of a thread repeats the root's opening message, and a short
    # prompt like "Blame bug 6652876" recurs across re-runs, so text alone
    # matches dozens of threads. One launch takes one thread, and vice versa.
    roots = sorted(((tid, turns[0]) for tid, turns in by_thread.items()),
                   key=lambda kv: _f(kv[1]["started_at"]) or 0.0)
    opening_of = {tid: norm(facts.get(root["audit_request_id"], {})
                            .get("first_user_head", ""))
                  for tid, root in roots}
    spawned: dict[str, dict] = {}
    taken: set[str] = set()
    for use_id, info in sorted(
            launches.items(),
            key=lambda kv: finished.get(emitter.get(kv[0]), 0.0) or 0.0):
        prompt = info["prompt"]
        if len(prompt) < 12:
            continue
        after = finished.get(emitter.get(use_id))
        for tid, root in roots:
            if tid in taken:
                continue
            started = _f(root["started_at"])
            if after is not None and started is not None and started < after:
                continue
            if prompt[:180] in opening_of[tid]:
                spawned[tid] = {"use_id": use_id, **info}
                taken.add(tid)
                break

    out = []
    for tid, turns in by_thread.items():
        root = turns[0]
        fact = facts.get(root["audit_request_id"], {})
        tools = fact.get("tool_names") or []
        opening = norm(fact.get("first_user_head", ""))
        hit = spawned.get(tid)
        out.append({
            "thread_id": tid,
            "session_id": root["session_id"],
            "turns": len(turns),
            "n_tools": len(tools),
            "tool_sig": ",".join(tools)[:400],
            "sys_chars": fact.get("sys_chars"),
            "n_msgs_at_root": fact.get("n_msgs"),
            "spawned_by_agent_call": hit["use_id"] if hit else "",
            "subagent_type": (hit or {}).get("subagent_type") or "",
            "agent_description": (hit or {}).get("description") or "",
            "parent_thread": thread_of.get(emitter.get(hit["use_id"])) if hit else "",
            "opening": (fact.get("first_user_task")
                        or task_text(fact.get("first_user_head", "")))[:160],
            "started_at": root["started_at"],
        })

    with (work / "thread_kinds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    print(f"threads: {len(out):,}   Agent launches seen: {len(agent_calls):,}"
          f"   distinct Agent prompts: {len(launches):,}")
    matched = sum(1 for t in out if t["spawned_by_agent_call"])
    print(f"threads matched to an Agent launch: {matched:,}")
    print()
    print("=== threads grouped by tool-set size ===")
    grp = defaultdict(list)
    for t in out:
        grp[t["n_tools"]].append(t)
    print(f"{'n_tools':>8s} {'threads':>8s} {'requests':>9s} {'med turns':>10s} "
          f"{'max turns':>10s} {'subagent-linked':>16s}")
    for n, ts in sorted(grp.items(), key=lambda kv: -len(kv[1])):
        turns = sorted(t["turns"] for t in ts)
        print(f"{n:8d} {len(ts):8d} {sum(turns):9,d} {turns[len(turns) // 2]:10d} "
              f"{turns[-1]:10d} {sum(1 for t in ts if t['subagent_type']):16d}")
    print()
    print("=== subagent_type of the linked threads ===")
    print(Counter(t["subagent_type"] for t in out if t["subagent_type"]).most_common())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
