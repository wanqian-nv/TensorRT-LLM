#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What one thread's opening turn reuses from other threads, and from which.

    python3 analysis/cross_thread_reuse.py <reports_dir> <attempt_dir>…

A thread's first turn has no parent of its own, so every token it hits in the
cache came from some *other* thread. `isl_cached` is therefore an exact,
server-measured figure for cross-thread reuse -- no reconstruction needed.

What it does not say is *with whom*, and that is the interesting part: a run
where every thread only shares the system prompt is a different cache regime
from one where sibling subagents share their whole skill preamble. That is
recovered by comparing the serialised prompts of the roots pairwise -- via a
prefix-hash ladder rather than 274k string compares -- and reading the pair
off the two threads' kinds.

Session-title calls are excluded: they are not conversations, they share
nothing with one (their prompts diverge at character 47), and they would
otherwise be a third of the population.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)

STEP = 512          # ladder granularity in characters; ~140 tokens
BUCKETS = [0, 1, 4096, 8192, 16384, 32768, 65536, 131072]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _serialise(path: Path) -> tuple[str, int] | None:
    """The prompt as the engine sees it, plus where system+tools end."""
    try:
        with gzip.open(path) as handle:
            body = json.load(handle)["body"]
    except (OSError, ValueError, KeyError):
        return None
    head = "\n".join(b.get("text") or "" for b in (body.get("system") or [])
                     if isinstance(b, dict))
    head += "\n" + json.dumps(body.get("tools") or [], sort_keys=True)
    return head + "\n" + json.dumps(body.get("messages") or [], sort_keys=True), len(head)


def _ladder(args):
    aid, path = args
    got = _serialise(path)
    if got is None:
        return None
    text, head_len = got
    steps = [hash(text[:n]) for n in range(STEP, len(text) + STEP, STEP)]
    return aid, steps, head_len, len(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports_dir", type=Path)
    ap.add_argument("attempt_dir", type=Path, nargs="+")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(
        (args.reports_dir / "requests_master.csv").open(encoding="utf-8"))
        if r["in_window"] == "True"]
    roots = [r for r in rows
             if r["thread_turn"] == "1" and r["thread_kind"] != "title"]
    by_aid = {r["audit_request_id"]: r for r in roots}
    caps = {}
    for attempt in args.attempt_dir:
        for aid, r in by_aid.items():
            p = attempt / "anthropic_message_capture" / (r["message_capture_file"] or "")
            if r["message_capture_file"] and p.exists():
                caps[aid] = p
    print(f"{len(roots):,} non-title thread roots, {len(caps):,} with a captured body")

    with mp.Pool(min(48, mp.cpu_count())) as pool:
        got = [g for g in pool.map(_ladder, list(caps.items()), chunksize=4) if g]
    lad = {aid: (steps, head, total) for aid, steps, head, total in got}

    order = sorted(lad, key=lambda a: _f(by_aid[a]["started_at"]) or 0.0)
    rank = {a: i for i, a in enumerate(order)}

    # Deepest shared prefix with any *earlier* root: only those could have put
    # the blocks in the cache. One pass down the ladder, keeping the first
    # depth at which a hash collides with an earlier root's.
    seen: dict[int, dict[int, str]] = defaultdict(dict)   # depth -> hash -> earliest aid
    best: dict[str, tuple[int, str | None]] = {}
    for aid in order:
        steps, head, total = lad[aid]
        hit_depth, partner = 0, None
        for depth, h in enumerate(steps):
            other = seen[depth].get(h)
            if other is None:
                break
            hit_depth, partner = (depth + 1) * STEP, other
        for depth, h in enumerate(steps):
            seen[depth].setdefault(h, aid)
        best[aid] = (min(hit_depth, total), partner)

    def kind_of(aid):
        r = by_aid[aid]
        return r["thread_kind"], r.get("sys_digest") or r.get("thread_n_tools")

    pairs = Counter()
    detail = defaultdict(lambda: {"tok": [], "share": [], "beyond": []})
    for aid in order:
        shared, partner = best[aid]
        r = by_aid[aid]
        cached = _f(r["isl_cached"]) or 0.0
        isl = _f(r["isl_total"]) or 0.0
        _steps, head, total = lad[aid]
        # the system+tools block in tokens, via this request's own char/token
        # ratio -- exact enough to say whether reuse got past the boilerplate
        head_tok = head / total * isl if total else 0.0
        if partner is None:
            tag = "nothing earlier shares a prefix"
        elif shared <= head:
            tag = "unrelated — system + tools only"
        else:
            a, b = r["thread_kind"], by_aid[partner]["thread_kind"]
            sa, sb = r.get("sys_digest"), by_aid[partner].get("sys_digest")
            if a == "subagent" and b == "subagent":
                tag = ("subagent ↔ subagent, same definition" if sa == sb
                       else "subagent ↔ subagent, different definition")
            elif {a, b} == {"subagent", "main"}:
                tag = "subagent ↔ main"
            elif a == b:
                tag = f"{a} ↔ {a}"
            else:
                tag = f"{a} ↔ {b}"
        pairs[tag] += 1
        d = detail[tag]
        d["tok"].append(cached)
        if isl:
            d["share"].append(cached / isl)
        d["beyond"].append(max(0.0, cached - head_tok))

    def q(v, p=0.5):
        return sorted(v)[int(len(v) * p)] if v else 0.0

    print()
    print(f"{'the opening turn shares with':<42s} {'roots':>6s} {'reused p50':>11s} "
          f"{'ratio p50':>10s} {'beyond boilerplate p50':>23s}")
    for tag, n in pairs.most_common():
        d = detail[tag]
        print(f"{tag:<42s} {n:6,d} {q(d['tok']):11,.0f} {q(d['share']):10.3f} "
              f"{q(d['beyond']):23,.0f}")
    all_tok = [v for d in detail.values() for v in d["tok"]]
    all_sh = [v for d in detail.values() for v in d["share"]]
    all_by = [v for d in detail.values() for v in d["beyond"]]
    print(f"{'all':<42s} {len(all_tok):6,d} {q(all_tok):11,.0f} {q(all_sh):10.3f} "
          f"{q(all_by):23,.0f}")
    print()
    print(f"Of {sum(all_tok):,.0f} tokens reused across threads, "
          f"{sum(all_by):,.0f} ({sum(all_by) / sum(all_tok) * 100:.1f}%) is past "
          "the system+tools block; the rest is the boilerplate every thread on "
          "the same client build shares.")

    print()
    print("reused tokens on a thread's opening turn")
    vals = sorted(all_tok)
    for lo, hi in zip(BUCKETS, BUCKETS[1:] + [None]):
        sel = [v for v in vals if v >= lo and (hi is None or v < hi)]
        if not sel:
            continue
        edge = f"{lo:,}–{hi:,}" if hi else f"{lo:,}+"
        print(f"   {edge:>18s} : {len(sel):5,d} roots ({len(sel) / len(vals) * 100:5.1f}%)"
              f"   total {sum(sel):>12,.0f} tok")
    print(f"   {'all':>18s} : {len(vals):5,d} roots            "
          f"   total {sum(vals):>12,.0f} tok")

    # The report renders the headline from this; it cannot compute the split
    # itself without reading the captured bodies.
    dest = args.reports_dir / "cross_thread_reuse.json"
    dest.write_text(json.dumps({
        "roots": len(all_tok),
        "root_cached": sum(all_tok),
        "beyond": sum(all_by),
        "boilerplate": sum(all_tok) - sum(all_by),
        "by_partner": {tag: {"roots": pairs[tag],
                             "reused_p50": q(detail[tag]["tok"]),
                             "ratio_p50": q(detail[tag]["share"]),
                             "beyond_p50": q(detail[tag]["beyond"])}
                       for tag in pairs},
        "buckets": {(f"{lo}-{hi}" if hi else f"{lo}+"):
                    len([v for v in vals if v >= lo and (hi is None or v < hi)])
                    for lo, hi in zip(BUCKETS, BUCKETS[1:] + [None])},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
