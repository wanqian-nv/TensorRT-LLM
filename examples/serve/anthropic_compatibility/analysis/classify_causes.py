#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Label every low-hit-rate request with the reason its prefix did not reuse.

    python3 analysis/classify_causes.py <reports_dir> <attempt_dir>… > causes.csv
    python3 analysis/perf_report.py <attempt_dir>… --causes causes.csv

Reads `requests.csv` from the reports directory and the captured request
bodies from the attempt directories, and emits `request_index,cause`.

The classification is structural rather than arithmetic, which matters on real
traces. The audit's LCP is computed by a tracker that keeps one previous
prompt per session and diffs against whichever request landed last; when a
session interleaves subagent turns that is a different conversation, and the
resulting `realization` exceeds 1.0 — physically impossible, and observed on
83% of one run's anomalies. So the predecessor is found here by matching
message prefixes, which picks the true parent turn regardless of interleaving,
and the labels follow from comparing against that turn:

    predecessor's messages prefix this one, no new system message
        -> the prompt was reusable and the cache did not serve it: cache side
    otherwise
        -> the prompt itself moved: prompt side, and the diff says how

Thresholds for the corroborating arithmetic are inherited from the
`anthropic-trace-analyze` skill and have not been re-derived here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# A healthy continuation keeps ~100% of the previous prompt: measured
# p50 = 1.006 over 180 real continuations. Anything at or above this is
# the cache doing its job, and the low hit ratio is new content.
REALIZATION_OK = 0.9


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fingerprint(path: Path) -> dict | None:
    """Per-message hashes plus the system shape — enough to compare two turns
    without holding either body in memory."""
    try:
        with gzip.open(path) as handle:
            body = json.load(handle)["body"]
    except (OSError, ValueError, KeyError):
        return None
    messages = body.get("messages") or []
    return {
        "messages": [_digest(m) for m in messages],
        "system": _digest(body.get("system")),
        "system_msgs": sum(1 for m in messages if m.get("role") == "system"),
        "tools": len(body.get("tools") or []),
    }


def _common(a: list[str], b: list[str]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


# Claude Code's context management rewrites the most recent message between
# turns (thinking retention), so a clean continuation shows a common prefix of
# exactly len(previous) - 1, never len(previous). Measured on real traces:
# 108->107, 110->109, 7->6, 2->1. Requiring an exact prefix labels every
# continuation a new thread.
TAIL_EDIT_SLACK = 1


def classify(row: dict, prev: dict | None, common: int, realization: float | None,
             cur: dict | None, pinned: bool) -> tuple[str, str]:
    """(label, one-line evidence) for one low-hit-rate request."""
    if str(row.get("session_turn_index")) == "1":
        return "COLD_START", "first turn of the session — cold by construction"
    if cur is None:
        return "NO_CAPTURE", "no captured body to compare"
    if prev is None or common == 0:
        return ("THREAD_START",
                "no earlier turn in this session shares a history prefix — a new "
                "thread (subagent) starting inside an existing session")

    if cur["system"] != prev["system"]:
        return "SYSTEM_CHANGED", "the top-level system field itself changed"
    if cur["system_msgs"] > prev["system_msgs"]:
        added = cur["system_msgs"] - prev["system_msgs"]
        return ("SYSTEM_HOISTED",
                f"{added} system message(s) appended since the previous turn; the "
                "adapter lifts them into the opening block, rewriting the prompt head")
    dropped = len(prev["messages"]) - common
    if dropped > TAIL_EDIT_SLACK:
        return ("HISTORY_REWRITTEN",
                f"{dropped} of the previous turn's {len(prev['messages'])} messages "
                "are gone from this one — truncation, compaction or summarisation")

    # Append-only, same system: the client did nothing that would break reuse.
    if realization is not None and realization >= REALIZATION_OK:
        return ("PROMPT_GROWTH",
                f"the cache held {realization:.0%} of the previous prompt — the low "
                "ratio is newly appended content, not a cache failure")
    if pinned:
        return ("CACHE_CEILING",
                "history is append-only yet cached tokens are pinned across turns — "
                "reuse is capped, check kv_cache_config")
    return ("CACHE_EVICTION",
            "history is append-only and the system prompt is unchanged, so the "
            "prefix was reusable and the cache did not serve it"
            + (f" (kept {realization:.0%} of the previous prompt)"
               if realization is not None else ""))


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("attempt_dir", type=Path, nargs="+")
    args = parser.parse_args()

    with (args.reports_dir / "requests.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    captures = {d.parent.name: d / "anthropic_message_capture" for d in args.attempt_dir}

    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("session_id"):
            by_session[row["session_id"]].append(row)
    for turns in by_session.values():
        turns.sort(key=lambda r: _float(r.get("started_at")) or 0.0)

    targets = {id(r) for r in rows if r.get("low_hit_rate") == "True"}
    # Only sessions that contain an anomaly need their bodies read; building the
    # predecessor chain needs every turn of those sessions, not just the flagged
    # ones.
    needed = {s for s, turns in by_session.items() if any(id(t) in targets for t in turns)}

    prints: dict[int, dict | None] = {}
    for session in needed:
        for row in by_session[session]:
            root = captures.get(row.get("run", ""))
            rel = row.get("message_capture_file")
            prints[id(row)] = (
                fingerprint(root / rel) if root and rel and (root / rel).exists()
                else None)

    out = csv.writer(sys.stdout)
    out.writerow(["request_index", "cause"])
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if id(row) not in targets:
            continue
        turns = by_session.get(row.get("session_id"), [])
        cur = prints.get(id(row))
        prev, best_common = None, 0
        if cur:
            # The true parent is the earlier turn sharing the longest history
            # prefix, not simply the previous turn by time: sessions interleave
            # subagent threads, and an exact-prefix test would reject every
            # continuation because the trailing message is rewritten.
            for other in turns:
                if other is row:
                    break
                fp = prints.get(id(other))
                if not fp:
                    continue
                shared = _common(fp["messages"], cur["messages"])
                if shared > best_common:
                    prev, best_common = fp, shared
        index = turns.index(row) if row in turns else -1
        window = [_float(t.get("isl_cached")) for t in turns[max(0, index - 2):index + 1]]
        pinned = len(window) == 3 and len(set(window)) == 1 and window[0]
        label, why = classify(row, prev, best_common, _float(row.get("realization")),
                              cur, bool(pinned))
        counts[label] += 1
        out.writerow([row["request_index"], f"{label} — {why}"])

    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{label}: {count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
