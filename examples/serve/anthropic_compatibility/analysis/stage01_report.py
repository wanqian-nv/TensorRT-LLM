#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the stage 0 + 1 findings from threads.csv / tool_calls.csv."""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def q(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(int(len(values) * p), len(values) - 1)]


def main() -> int:
    work = Path(sys.argv[1]) / "_threadstudy"
    threads = list(csv.DictReader((work / "threads.csv").open(encoding="utf-8")))
    calls = list(csv.DictReader((work / "tool_calls.csv").open(encoding="utf-8")))
    by_aid = {t["audit_request_id"]: t for t in threads}
    win = [t for t in threads if t["in_window"] == "True"]
    emit = defaultdict(list)
    for c in calls:
        emit[c["emitter_aid"]].append(c)

    out = ["# Stage 0 + 1 — threads rebuilt, tool calls accounted",
           "",
           f"Source: `{sys.argv[1]}` · {len(threads):,} captured requests "
           f"({len(win):,} inside the report window) · {len(calls):,} tool calls.",
           ""]

    # ---- stage 0 ----------------------------------------------------------
    old = [x for x in (_f(t["gap_to_next_turn_ms"]) for t in win) if x is not None]
    new = [x for x in (_f(t["true_gap_ms"]) for t in win) if x is not None]
    out += ["## Stage 0 — thread reconstruction", "",
            f"- `session_id` count: **{len({t['session_id'] for t in threads}):,}**",
            f"- reconstructed threads: **{len({t['thread_id'] for t in threads}):,}**",
            f"- max thread depth: **{max(int(t['thread_depth']) for t in threads)}**, "
            f"median **{q([int(t['thread_depth']) for t in threads], .5):.0f}**",
            "",
            "Parent links are found by prefix-matching per-message digests with "
            "`cache_control` stripped (that marker migrates forward between turns "
            "and otherwise desynchronises the last two messages of every parent).",
            "",
            "| gap definition | n | negative | p50 | p90 | p99 | >60s | >300s | >600s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for label, g in (("per-`session_id`, next-by-time (old)", old),
                     ("vs reconstructed parent (new)", new)):
        neg = sum(1 for x in g if x < 0)
        out.append(
            f"| {label} | {len(g):,} | {neg:,} ({neg / len(g) * 100:.1f}%) | "
            f"{q(g, .5) / 1000:.2f}s | {q(g, .9) / 1000:.2f}s | {q(g, .99) / 1000:.0f}s | "
            + " | ".join(f"{sum(1 for x in g if x > t * 1000):,}"
                         for t in (60, 300, 600)) + " |")
    later = Counter(c["turns_later"] for c in calls)
    out += ["",
            f"Correctness check: **{later['1'] / len(calls) * 100:.1f}%** of tool "
            f"results ({later['1']:,}/{len(calls):,}) land in exactly the emitting "
            "turn's child. Before stripping `cache_control` that was 78.5%, with "
            "14.5% appearing to land in a sibling — the signature of a broken link.",
            ""]

    # ---- stage 1a ---------------------------------------------------------
    out += ["## Stage 1a — tool inventory", "",
            "`wait_mode` is read from the result text: `sync` the client waited, "
            "`handle_now` a handle came back at once and the job outlived the call, "
            "`blocked_out` the client blocked the full timeout then the job was "
            "backgrounded, `poll_hit`/`poll_miss` a retrieval against running work.",
            "",
            "| tool | calls | sync | handle_now | blocked_out | poll_hit | poll_miss |",
            "|---|---|---|---|---|---|---|"]
    per = defaultdict(list)
    for c in calls:
        per[c["tool"]].append(c)
    for tool, rows in sorted(per.items(), key=lambda kv: -len(kv[1])):
        if len(rows) < 10:
            continue
        m = Counter(r["wait_mode"] for r in rows)
        out.append(f"| `{tool}` | {len(rows):,} | {m['sync']:,} | {m['handle_now']} | "
                   f"{m['blocked_out']} | {m['poll_hit']} | {m['poll_miss']} |")
    small = sum(len(v) for k, v in per.items() if len(v) < 10)
    out += [f"| _{sum(1 for v in per.values() if len(v) < 10)} tools with <10 calls_ "
            f"| {small} | | | | | |", ""]

    # ---- stage 1b ---------------------------------------------------------
    batch = Counter(c["emitter_aid"] for c in calls)
    solo = sum(1 for c in calls if batch[c["emitter_aid"]] == 1)
    out += ["## Stage 1b — how long results took to come back", "",
            f"**{len(calls) - solo:,} of {len(calls):,} calls ({(len(calls) - solo) / len(calls) * 100:.0f}%) "
            "were emitted in a parallel batch**, and a batch surfaces as one "
            "message — every call in it shares the batch's return time, so a fast "
            "tool inherits the slowest one's latency. `Glob` reads p50 17.7s "
            "batched and 0.34s alone. The table below is **solo calls only**.",
            "",
            "| tool | solo calls | p50 | p90 | p99 | max |",
            "|---|---|---|---|---|---|"]
    solo_by = defaultdict(list)
    for c in calls:
        t = _f(c["t_surface_ms"])
        if batch[c["emitter_aid"]] == 1 and t is not None:
            solo_by[c["tool"]].append(t / 1000)
    for tool, v in sorted(solo_by.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 10:
            continue
        out.append(f"| `{tool}` | {len(v):,} | {q(v, .5):.3f}s | {q(v, .9):.2f}s | "
                   f"{q(v, .99):.1f}s | {max(v):.1f}s |")
    out.append("")

    # ---- stage 1c ---------------------------------------------------------
    bg = [c for c in calls if c["wait_mode"] in ("handle_now", "blocked_out")]
    done = [c for c in bg if _f(c["t_complete_ms"]) is not None]
    out += ["## Stage 1c — background work: handshake ≠ job", "",
            f"{len(bg)} calls were backgrounded. Their completion never arrives as "
            "a tool_result — it arrives as a `<task-notification>` block in a later "
            "user message carrying the launching `<tool-use-id>`. Linking on that id "
            f"recovers the real duration for **{len(done)}** of them.",
            "",
            "| tool / mode | n | handshake p50 | with notice | job p50 | job p90 | job max | turns later (p50) |",
            "|---|---|---|---|---|---|---|---|"]
    grp = defaultdict(list)
    for c in bg:
        grp[(c["tool"], c["wait_mode"])].append(c)
    for (tool, mode), rows in sorted(grp.items(), key=lambda kv: -len(kv[1])):
        hs = [_f(r["t_surface_ms"]) / 1000 for r in rows if _f(r["t_surface_ms"]) is not None]
        jb = [_f(r["t_complete_ms"]) / 1000 for r in rows if _f(r["t_complete_ms"]) is not None]
        tn = [_f(r["complete_turns_later"]) for r in rows
              if _f(r["complete_turns_later"]) is not None]
        cells = (f"{len(jb)} | {q(jb, .5):.0f}s | {q(jb, .9):.0f}s | {max(jb):.0f}s | "
                 f"{q(tn, .5):.0f} |") if jb else "0 | — | — | — | — |"
        out.append(f"| `{tool}` / {mode} | {len(rows)} | {q(hs, .5):.2f}s | {cells}")
    out += ["",
            "This is the trap the existing `tool_latency_ms` column falls into: for "
            "the 290 async `Agent` launches it records the **0.14s handshake** while "
            "the job actually ran a median of **167s** and landed a median of **11 "
            "turns later**.", ""]

    # ---- stage 1d ---------------------------------------------------------
    trans = [(g, t["parent"]) for t in win
             if t["parent"] and (g := _f(t["true_gap_ms"])) is not None]
    out += ["## Stage 1d — what actually produces a long wait", "",
            "Each transition's wait is attributed to the tools its **parent** turn "
            "emitted, since those are what the client was executing.",
            "",
            "| threshold | transitions | top emitters | wait_mode mix |",
            "|---|---|---|---|"]
    for th in (30, 60, 300, 600):
        sel = [x for x in trans if x[0] > th * 1000]
        names, modes = Counter(), Counter()
        for _, parent in sel:
            got = emit.get(parent, [])
            if not got:
                names["(no tool call)"] += 1
            for x in got:
                names[x["tool"]] += 1
                modes[x["wait_mode"]] += 1
        top = ", ".join(f"`{k}` {v}" for k, v in names.most_common(4))
        mix = ", ".join(f"{k} {v}" for k, v in modes.most_common(4))
        out.append(f"| >{th}s | {len(sel):,} | {top} | {mix} |")
    out += ["",
            "In the >600s bucket the wait is almost entirely a **harness ceiling, "
            "not a workload duration**: 40 `blocked_out` (Bash sitting out its full "
            "600s timeout) and 27 `poll_miss` (a `TaskOutput` poll window that "
            "expired with the task still running). The single longest wait in the "
            "run, 4,525s, has no tool call behind it at all — it is idle time.",
            "",
            "## Handoff to the eviction study", "",
            f"The population to test is the **{sum(1 for x in trans if x[0] > 60000):,} "
            f"transitions with a >60s parent-relative wait** "
            f"({sum(1 for x in trans if x[0] > 300000):,} over 300s, "
            f"{sum(1 for x in trans if x[0] > 600000):,} over 600s), each now "
            "carrying a true parent to measure prefix survival against.", ""]

    dest = work / "stage01_report.md"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
