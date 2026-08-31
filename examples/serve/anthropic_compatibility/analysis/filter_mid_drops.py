#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every mid-thread hit-rate drop, with its parent turn's context alongside.

    python3 analysis/filter_mid_drops.py <reports_dir> [--floor 0.95] [--early 3]

Writes `mid_thread_drops.csv`: the rows from `requests_master.csv` past turn
`--early` whose hit rate is under `--floor` -- the same set the timeline chart
colours as "mid-thread drop", i.e. the drops a cold start cannot explain.

Every column of the master is kept. Added on top are the parent turn's figures
and three derived ones, because the first question about any of these rows is
"compared to what?" and the answer lives on a different row:

  drop_from_parent    parent's hit rate minus this one's
  survival_vs_parent  match_len_best / (parent isl + parent osl) -- of what the
                      cache could still be holding from the previous turn, the
                      share any rank actually had. Server-side, so it is
                      independent of the gateway's own LCP tracker
  routing_loss        match_len_best - match_len_chosen. Non-zero means the
                      prefix was alive on another rank and the router went
                      elsewhere: a routing loss, not an eviction
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

csv.field_size_limit(10 ** 9)

ADD = ["drop_from_parent", "survival_vs_parent", "routing_loss",
       "parent_kv_hit_rate", "parent_isl_total", "parent_osl",
       "parent_tool_names", "parent_tool_wait_modes", "parent_thread_turn"]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports_dir", type=Path)
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--early", type=int, default=3)
    args = ap.parse_args()

    rows = list(csv.DictReader(
        (args.reports_dir / "requests_master.csv").open(encoding="utf-8")))
    cols = list(rows[0].keys())
    by_aid = {r["audit_request_id"]: r for r in rows}

    out = []
    for r in rows:
        hit = _f(r["kv_hit_rate"])
        if r["in_window"] != "True" or hit is None:
            continue
        if int(r["thread_turn"]) <= args.early or hit >= args.floor:
            continue
        p = by_aid.get(r["parent_aid"]) or {}
        ph, pisl, posl = (_f(p.get("kv_hit_rate")), _f(p.get("isl_total")),
                          _f(p.get("osl")))
        best, chosen = _f(r["match_len_best"]), _f(r["match_len_chosen"])
        # what the previous turn left behind: its prompt plus what it generated
        held = (pisl + posl) if (pisl is not None and posl is not None) else None
        out.append({
            **r,
            "drop_from_parent": f"{ph - hit:.6f}" if ph is not None else "",
            "survival_vs_parent": (f"{best / held:.6f}"
                                   if best is not None and held else ""),
            "routing_loss": (f"{best - chosen:.0f}"
                             if best is not None and chosen is not None else ""),
            "parent_kv_hit_rate": p.get("kv_hit_rate", ""),
            "parent_isl_total": p.get("isl_total", ""),
            "parent_osl": p.get("osl", ""),
            "parent_tool_names": p.get("tool_names", ""),
            "parent_tool_wait_modes": p.get("tool_wait_modes", ""),
            "parent_thread_turn": p.get("thread_turn", ""),
        })

    # worst first: the biggest fall from the previous turn
    out.sort(key=lambda r: -(_f(r["drop_from_parent"]) or -1))
    dest = args.reports_dir / "mid_thread_drops.csv"
    with dest.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=cols + ADD)
        w.writeheader()
        w.writerows(out)
    print(f"{dest}: {len(out):,} rows "
          f"(turn > {args.early}, hit rate < {args.floor:.0%}), "
          f"{len(cols) + len(ADD)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
