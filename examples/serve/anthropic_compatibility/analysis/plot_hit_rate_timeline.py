#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hit rate over wall clock, for thread openings and every low-hit request.

    python3 analysis/plot_hit_rate_timeline.py <reports_dir> [--floor 0.95]

Plots the union of two sets: the first three turns of every thread, and every
request under the hit-rate floor. Colour carries *why a point is on the chart*
rather than which thread it belongs to -- 1,119 threads cannot be told apart by
hue, and cycling a categorical palette past its validated slots is worse than
not colouring at all. Thread identity rides on the faint connectors instead,
which is what actually lets a reader follow one conversation through the cloud.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

csv.field_size_limit(10 ** 9)

W, H = 1280, 660
ML, MR, MT, MB = 62, 26, 18, 52
PW, PH = W - ML - MR, H - MT - MB

# slots 1-3 of the reference categorical palette: the only three that clear the
# all-pairs gates a scatter needs (validated light and dark, --pairs all)
SERIES = [
    ("mid",   "Mid-thread drop (turn &gt; 3)", "#2a78d6", "#3987e5"),
    ("cold",  "Cold start (turn ≤ 3, below floor)", "#eb6834", "#d95926"),
    ("open",  "Healthy opening (turn ≤ 3)", "#1baf7a", "#199e70"),
]


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

    rows = [r for r in csv.DictReader(
        (args.reports_dir / "requests_master.csv").open(encoding="utf-8"))
        if r["in_window"] == "True" and _f(r["kv_hit_rate"]) is not None]

    pts = []
    for r in rows:
        hit, turn = _f(r["kv_hit_rate"]), int(r["thread_turn"])
        early, low = turn <= args.early, hit < args.floor
        if not (early or low):
            continue
        pts.append({
            "t": _f(r["started_at"]), "h": hit, "turn": turn,
            "cls": "cold" if (early and low) else ("open" if early else "mid"),
            "thread": r["thread_id"], "kind": r["thread_kind"],
            "rid": r["rid"], "gap": _f(r["gap_from_parent_ms"]),
            "isl": _f(r["isl_total"]), "cached": _f(r["isl_cached"]),
            "mb": _f(r["match_len_best"]), "mc": _f(r["match_len_chosen"]),
            "tools": r["tool_names"], "sub": r["is_subagent"],
        })
    pts.sort(key=lambda p: p["t"])
    t0, t1 = pts[0]["t"], pts[-1]["t"]

    def X(t):
        return ML + (t - t0) / (t1 - t0) * PW

    def Y(h):
        return MT + (1.0 - h) * PH

    # thread connectors: only threads with more than one plotted point
    by_thread = defaultdict(list)
    for p in pts:
        by_thread[p["thread"]].append(p)
    paths = []
    for tid, ps in by_thread.items():
        if len(ps) < 2:
            continue
        ps.sort(key=lambda p: p["t"])
        paths.append(" ".join(
            ("M" if i == 0 else "L") + f"{X(p['t']):.1f},{Y(p['h']):.1f}"
            for i, p in enumerate(ps)))

    order = {"open": 0, "cold": 1, "mid": 2}   # anomalies drawn last, on top
    pts.sort(key=lambda p: (order[p["cls"]], p["t"]))

    for p in pts:
        p["x"], p["y"] = round(X(p["t"]), 1), round(Y(p["h"]), 1)
    circles = "".join(
        f'<circle class="pt {p["cls"]}" cx="{p["x"]}" cy="{p["y"]}" r="4"/>'
        for p in pts)

    # axes
    hours = []
    step = 3600
    first = (int(t0) // step + 1) * step
    tick = first
    while tick <= t1:
        hours.append(tick)
        tick += step
    xticks = "".join(
        f'<line class="grid" x1="{X(t):.1f}" y1="{MT}" x2="{X(t):.1f}" y2="{MT+PH}"/>'
        f'<text class="ax" x="{X(t):.1f}" y="{MT+PH+20}" text-anchor="middle">'
        f'{datetime.fromtimestamp(t):%H:%M}</text>' for t in hours)
    yticks = "".join(
        f'<line class="grid" x1="{ML}" y1="{Y(v):.1f}" x2="{ML+PW}" y2="{Y(v):.1f}"/>'
        f'<text class="ax" x="{ML-10}" y="{Y(v)+4:.1f}" text-anchor="end">{v*100:.0f}%</text>'
        for v in (0, .2, .4, .6, .8, 1.0))

    counts = {k: sum(1 for p in pts if p["cls"] == k) for k, *_ in SERIES}
    legend = "".join(
        f'<span class="lg"><i style="background:var(--s-{k})"></i>{lab}'
        f'<b>{counts[k]:,}</b></span>' for k, lab, *_ in SERIES)

    # relief for the light-mode contrast WARN on slot 3: a table view
    tbl = []
    for k, lab, *_ in SERIES:
        sel = [p for p in pts if p["cls"] == k]
        hs = sorted(p["h"] for p in sel)
        med = hs[len(hs) // 2] if hs else 0
        gaps = sorted(p["gap"] for p in sel if p["gap"] is not None)
        tbl.append(
            f"<tr><td><i style='background:var(--s-{k})'></i>{lab}</td>"
            f"<td>{len(sel):,}</td><td>{med*100:.1f}%</td>"
            f"<td>{hs[0]*100:.1f}%</td>"
            f"<td>{(gaps[len(gaps)//2]/1000 if gaps else 0):.1f}s</td>"
            f"<td>{(gaps[-1]/1000 if gaps else 0):.0f}s</td></tr>")

    css_series = "".join(
        f"  --s-{k}: {light};\n" for k, _, light, _dark in SERIES)
    css_series_dark = "".join(
        f"    --s-{k}: {dark};\n" for k, _, _light, dark in SERIES)

    doc = f"""<!doctype html><meta charset=utf-8>
<title>Hit rate over time — thread openings and low-hit requests</title>
<style>
.viz-root {{
  color-scheme: light;
  --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#8a8880; --grid:#e6e5e0; --rule:#c9c7c0;
{css_series}}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7;
    --muted:#7d7c74; --grid:#2c2c2a; --rule:#3d3c39;
{css_series_dark}  }}
}}
:root[data-theme="dark"] .viz-root {{
  color-scheme: dark;
  --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#7d7c74; --grid:#2c2c2a; --rule:#3d3c39;
{css_series_dark}}}
body{{margin:0;background:var(--surface-1)}}
.viz-root{{font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
  background:var(--surface-1);color:var(--text-primary);padding:26px 30px 34px;max-width:1340px}}
h1{{font-size:17px;margin:0 0 4px;font-weight:650}}
p.sub{{color:var(--text-secondary);margin:0 0 14px;font-size:12.5px;max-width:96ch}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin:0 0 8px;font-size:12.5px}}
.lg{{display:flex;align-items:center;gap:7px;color:var(--text-secondary)}}
.lg i,td i{{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}}
.lg b{{color:var(--text-primary);font-variant-numeric:tabular-nums;font-weight:600}}
svg{{display:block;overflow:visible}}
.grid{{stroke:var(--grid);stroke-width:1}}
.ax{{fill:var(--muted);font-size:11px}}
.floor{{stroke:var(--rule);stroke-width:1.5;stroke-dasharray:5 4}}
.link{{fill:none;stroke:var(--muted);stroke-width:1;opacity:.16}}
.pt{{stroke:var(--surface-1);stroke-width:1.5;opacity:.8;cursor:pointer}}
.pt.open{{fill:var(--s-open)}} .pt.cold{{fill:var(--s-cold)}} .pt.mid{{fill:var(--s-mid)}}
#halo{{fill:none;stroke:var(--text-primary);stroke-width:2;opacity:0;pointer-events:none}}
#halo.on{{opacity:1}}
#hit{{fill:transparent;cursor:crosshair}}
#tip{{position:fixed;pointer-events:none;background:var(--text-primary);
  color:var(--surface-1);font-size:11.5px;line-height:1.55;padding:8px 11px;border-radius:6px;
  opacity:0;transition:opacity .08s;white-space:nowrap;z-index:9;
  font-variant-numeric:tabular-nums;box-shadow:0 3px 14px rgba(0,0,0,.22)}}
#tip.on{{opacity:1}}
#tip b{{font-weight:650}}
table{{border-collapse:collapse;margin:20px 0 0;font-size:12px}}
th{{text-align:right;color:var(--muted);font-weight:600;padding:5px 12px;
  border-bottom:1px solid var(--grid);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
td{{text-align:right;padding:5px 12px;border-bottom:1px solid var(--grid);
  font-variant-numeric:tabular-nums;color:var(--text-secondary)}}
td:first-child{{color:var(--text-primary);display:flex;align-items:center;gap:8px}}
</style>
<div class="viz-root">
<h1>KV cache hit rate over wall clock</h1>
<p class="sub">Every request in the first {args.early} turns of its thread, plus every request
below the {args.floor:.0%} floor — {len(pts):,} points across {len(by_thread):,} threads over
{(t1-t0)/3600:.1f} hours. Faint connectors join points of the same thread; colour says why a
point is here, not which thread it is (1,119 threads cannot be separated by hue).</p>
<div class="legend">{legend}</div>
<svg viewBox="0 0 {W} {H}" width="100%">
{yticks}{xticks}
<line class="floor" x1="{ML}" y1="{Y(args.floor):.1f}" x2="{ML+PW}" y2="{Y(args.floor):.1f}"/>
<text class="ax" x="{ML+PW}" y="{Y(args.floor)-7:.1f}" text-anchor="end">{args.floor:.0%} floor</text>
{"".join(f'<path class="link" d="{d}"/>' for d in paths)}
{circles}
<line class="grid" x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}"/>
<circle id="halo" r="8"/>
<rect id="hit" x="{ML}" y="{MT}" width="{PW}" height="{PH}"/>
</svg>
<table><thead><tr><th>series</th><th>points</th><th>median hit</th><th>min hit</th>
<th>median wait before</th><th>max wait before</th></tr></thead>
<tbody>{"".join(tbl)}</tbody></table>
</div>
<div id="tip"></div>
<script>
const D = {json.dumps(pts, separators=(",", ":"))};
const tip  = document.getElementById('tip');
const halo = document.getElementById('halo');
const hit  = document.getElementById('hit');
const svg  = hit.ownerSVGElement;
const fmt = t => new Date(t*1000).toLocaleTimeString([], {{hour:'2-digit',minute:'2-digit',second:'2-digit'}});

// Nearest-point layer. 2,982 dots at r=4 overlap in a third of the 8px cells
// -- the densest holds 33 -- so per-dot hit targets would be unlandable. One
// transparent rect takes the pointer and a coarse grid finds the nearest dot
// within RADIUS, which is the ~24px target the dots themselves cannot offer.
const CELL = 24, RADIUS = 22;
const grid = new Map();
D.forEach((d, i) => {{
  const k = ((d.x/CELL)|0) + ':' + ((d.y/CELL)|0);
  (grid.get(k) || grid.set(k, []).get(k)).push(i);
}});

function nearest(px, py) {{
  const gx = (px/CELL)|0, gy = (py/CELL)|0;
  let best = -1, bd = RADIUS*RADIUS;
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {{
    const c = grid.get((gx+dx) + ':' + (gy+dy));
    if (!c) continue;
    for (const i of c) {{
      const d = D[i], s = (d.x-px)*(d.x-px) + (d.y-py)*(d.y-py);
      if (s < bd) {{ bd = s; best = i; }}
    }}
  }}
  return best;
}}

let cur = -1;
hit.addEventListener('mousemove', e => {{
  const r = svg.getBoundingClientRect();
  const sx = {W} / r.width, sy = {H} / r.height;
  const i = nearest((e.clientX - r.left) * sx, (e.clientY - r.top) * sy);
  if (i !== cur) {{
    cur = i;
    if (i < 0) {{ halo.classList.remove('on'); tip.classList.remove('on'); }}
    else {{
      const d = D[i];
      halo.setAttribute('cx', d.x); halo.setAttribute('cy', d.y);
      halo.classList.add('on');
      tip.innerHTML =
        `<b>${{(d.h*100).toFixed(1)}}%</b> hit &nbsp;·&nbsp; turn ${{d.turn}} of ${{d.kind}}` +
        `<br>${{fmt(d.t)}} &nbsp;·&nbsp; rid ${{d.rid||'—'}}` +
        `<br>wait before: ${{d.gap==null?'—':(d.gap/1000).toFixed(1)+'s'}}` +
        `<br>cached ${{(d.cached||0).toLocaleString()}} / ${{(d.isl||0).toLocaleString()}} tok` +
        `<br>match best/chosen: ${{(d.mb||0).toLocaleString()}} / ${{(d.mc||0).toLocaleString()}}` +
        (d.tools ? `<br>tools: ${{d.tools}}` : '');
      tip.classList.add('on');
    }}
  }}
  if (i >= 0) {{
    tip.style.left = Math.min(e.clientX + 16, innerWidth - tip.offsetWidth - 10) + 'px';
    tip.style.top  = Math.max(e.clientY - tip.offsetHeight - 14, 8) + 'px';
  }}
}});
hit.addEventListener('mouseleave', () => {{
  cur = -1; halo.classList.remove('on'); tip.classList.remove('on');
}});
</script>"""
    dest = args.reports_dir / "hit_rate_timeline.html"
    dest.write_text(doc, encoding="utf-8")
    print(f"{dest}: {len(pts):,} points, {len(by_thread):,} threads, "
          f"{len(paths):,} connectors")
    for k, lab, *_ in SERIES:
        print(f"   {lab.replace('&gt;','>'):38s} {counts[k]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
