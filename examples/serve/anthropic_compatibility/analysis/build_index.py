#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Index every report under a directory into one comparison page.

    python3 analysis/build_index.py <reports_dir>

Each immediate subdirectory holding a ``summary.csv`` becomes a row. Shares of
wall clock sit next to the absolute figures because only the share survives
comparison between runs of different length.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

CSS = """
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;
color:#1b2733;margin:0;padding:32px 40px;max-width:1120px}
h1{font-size:19px;margin:0 0 4px}.sub{color:#7a8794;font-size:12px;margin:0 0 18px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:right;color:#7a8794;font-weight:600;padding:6px 10px;
border-bottom:1px solid #e6eaee;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:6px 10px;border-bottom:1px solid #f2f5f7;
font-variant-numeric:tabular-nums}
a{color:#3b7dd8;text-decoration:none}a:hover{text-decoration:underline}
.na{color:#c3cad2}td.lead{font-weight:650;color:#0f1c2b}
"""


def _f(value, fmt="{:,.0f}"):
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return '<span class="na">—</span>'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for entry in sorted(args.reports_dir.iterdir()):
        summary = entry / "summary.csv"
        if not entry.is_dir() or not summary.exists():
            continue
        with summary.open(encoding="utf-8") as handle:
            record = next(csv.DictReader(handle), None)
        if record:
            rows.append((entry.name, record))

    body = []
    for name, r in rows:
        try:
            wall = float(r["wall_s"])
        except (TypeError, ValueError, KeyError):
            wall = None

        def share(key):
            try:
                return f'{float(r[key]) / wall * 100:.0f}%'
            except (TypeError, ValueError, ZeroDivisionError, KeyError):
                return '<span class="na">—</span>'

        safe = html.escape(name)
        body.append(
            f"<tr><td><a href='{safe}/REPORT.html'>{safe}</a></td>"
            f"<td>{r['requests']}</td><td>{r['sessions']}</td>"
            f"<td class='lead'>{_f(wall / 3600 if wall else None, '{:.2f}')}</td>"
            f"<td>{share('server_busy_s')}</td><td>{share('client_idle_s')}</td>"
            f"<td class='lead'>{_f(r['kv_hit_rate'], '{:.3f}')}</td>"
            f"<td>{r['low_hit_rate_requests']}</td>"
            f"<td>{_f(r['prefill_utilization_mean'], '{:.3f}')}</td>"
            f"<td>{_f(r['prefill_imbalance_mean'], '{:.2f}')}</td>"
            f"<td>{r.get('runs', 1)}</td>"
            f"<td><a href='{safe}/requests.csv'>req</a> · "
            f"<a href='{safe}/sessions.csv'>sess</a> · "
            f"<a href='{safe}/summary.csv'>sum</a></td></tr>")

    (args.reports_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>serving run reports</title>"
        f"<style>{CSS}</style><h1>Serving run reports</h1>"
        "<p class='sub'>Derived metrics only — no prompt content. "
        "Busy and idle are shares of wall clock.</p>"
        "<table><thead><tr><th>run</th><th>req</th><th>sess</th><th>wall h</th>"
        "<th>busy</th><th>idle</th><th>hit</th><th>&lt;90%</th><th>util</th>"
        "<th>imbal</th><th>runs</th><th>csv</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>", encoding="utf-8")
    print(f"indexed {len(rows)} runs -> {args.reports_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
