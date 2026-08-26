---
name: trtllm-perf-report
description: Build the four-section performance report for one TRT-LLM serving run — per request, per session, prefill-server iterations, and run totals — as an HTML report plus four CSVs. Use when the user asks to analyze a run's performance, check prefill utilization or rank imbalance, account for where a run's time went, or investigate cache hit rate.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Serving-run performance report

Base directory (all paths relative to it):
`TensorRT-LLM/examples/serve/anthropic_compatibility/`

Reports directory, written `<REPORTS>` below — every report this skill
produces goes here and nowhere else:
`/lustre/fsw/portfolios/coreai/users/serli/workspace/TensorRT-LLM/examples/serve/anthropic_compatibility/_reports/`

It sits beside this checkout, not beside the runs it describes: the trace root
is raw capture that gets rsynced and cleaned up, and it is read-only to some
callers, while a report is the thing you link to and open — so it lives where
the editor already has the tree loaded. `.gitignore` keeps `_reports/` out of
`git status`; it is in the working tree but never in its history.

`perf_report.py` writes there on its own, so **do not pass a path to `--out`**
and never leave a report in the run directory or `/tmp`. `--out` takes a
*label*, which is resolved under that root; omit it and the label is the run's
own name. `build_index.py` reads the same root by default and its index links
relatively, so a report written outside it is invisible to the index.
`PERF_REPORTS_DIR` moves the root for both — set it when working against a
different trace root (computelab has its own), never to redirect a single
report.

`index.html` lists the reports **newest first**, so the run just built is the
top row; it is the intended entry point rather than the file tree.

## Step 0 — Find the run

A run is an **attempt** directory: it holds `ctx-0.log` / `gen-0.log` (disaggregated)
or `server.log` (aggregated). Search under
`/lustre/fsw/portfolios/coreai/users/serli/claude-traces/`. Both layouts are in
the wild — `<root>/<run_name>/attempt-NNN/` and, since serve.sh started dating
them, `<root>/<YYYY-MM>/<DD>/<run_name>/attempt-NNN/` — so search to depth 5:
at depth 3 all but the oldest handful of runs are invisible. Skip `_fleet`,
`_sbatch_logs` and `_reports`. Highest attempt number unless the user names one.

```bash
find /lustre/fsw/portfolios/coreai/users/serli/claude-traces -maxdepth 5 \
     \( -name 'ctx-0.log' -o -name 'server.log' \) -printf '%h\n' | sort -u
```

Nothing is mandatory — the report degrades to whatever is present and names the
gaps at the top.

## Step 1 — One run or several

`perf_report.py` takes any number of attempt directories.

**When the user names more than one run, ask which they want** — the two
answers produce different documents and neither is recoverable from the other:

- **Merged** (the default; take it if the user has no preference) — one set of
  CSVs and one HTML covering the whole workload. Time is summed per run, never
  measured across them, so the gap between serving jobs is not charged to the
  workload. Sessions pair on the client-supplied id, so a conversation the
  gateway relayed to a successor job stays one session.
- **Compared** — one report per run plus an index page ranking them. Use when
  the runs differ in config and the question is which config won.

```bash
# merged: one report over all of them, under a label naming the merge
python3 analysis/perf_report.py <DIR_1> <DIR_2> … --out _merged_<subject>

# compared: one report each — the label defaults to the run name — then index
for d in <DIR_1> <DIR_2> …; do python3 analysis/perf_report.py "$d"; done
python3 analysis/build_index.py
```

A merged report states how many runs went into it and lists each one's
contribution under **Merged from**, so a reader can see whether one run
dominates the totals.

In the merged CSVs the key is `(run, rid)`, not `rid`: engine request ids are
per-process counters that restart with every serving job. Every cross-source
join happens inside a run before merging, so the collisions do not affect any
figure — they only matter if you re-join the CSVs by hand.

```bash
python3 analysis/perf_report.py <ATTEMPT_DIR>
```

Writes to `<REPORTS>/<run name>/`, and prints the path it chose — quote that
path rather than reconstructing it:

| file | grain |
|---|---|
| `REPORT.html` | the four sections, self-contained |
| `requests.csv` | one row per request |
| `sessions.csv` | one row per Claude Code session |
| `ctx_iters.csv` | one row per (prefill instance, iteration), ranks pooled |
| `ctx_rank_iters.csv` | the same split per rank — what the pooled rows average away |
| `summary.csv` | one row for the run |

Iteration-grain files pass 50k rows on any real run and are written gzipped
with a `.csv.gz` suffix — one run's per-rank file is 123 MB plain and 0.7 MB
compressed, since idle iterations repeat. `pd.read_csv` opens them by
extension, unchanged.

Relay the notes the script prints — they say which sources were missing.

## Step 2 — Report

Lead with section 4's decomposition, then whichever section explains it:

```
wall = server_busy + client_idle
server_busy ⊇ gpu_prefill + gpu_decode        remainder is server CPU
```

- **client_idle dominant** → tool execution, not the server. Section 2's
  `client_time_ms` per session.
- **server_busy ≫ gpu** → host-bound. Section 3's `host_step_time_ms` vs
  `device_step_time_ms`.
- **low `utilization`** → prefill is running far under its token budget;
  batching or arrival rate, not the model.
- **high `imbalance`** → routing skew. Section 1's `affinity_regret` says
  whether the router chose skew deliberately (affinity won) or was forced
  into it (fair-share cap).
- **`low_hit_rate_requests` non-zero** → go to Step 3.

Quote p50/p90, never only the mean: agent traffic is long-tailed, and a mean
TTFT three times its median is the normal shape, not an anomaly.

## Step 3 — Cache anomalies: find the cause, not more metrics

Section 1 lists every request below 90% hit rate, sorted by TTFT and showing
how many times the healthy median it cost. That is the whole point of the
table — the hit rate itself is not interesting, the TTFT it bought and the
reason for it are.

For each row, drill into the captured body and **write one line naming the
cause**. The `anthropic-trace-analyze` skill's Step 4 has the procedure:
`cache_realization × opportunity` splits cache-side (the prefix was reusable
and the engine did not serve it) from prompt-side (the prefix itself changed),
and Steps 4b/4c label it. Do not re-derive that logic here.

Run the mechanical pass first — it labels the cases that are decidable from
the captured bodies alone, so only the residue needs reading:

It reads `requests.csv` from the report, so it takes the report directory
first and the attempt directories it was built from second, and the causes
live in the report beside the numbers they explain:

```bash
python3 analysis/classify_causes.py <REPORTS>/<label> <ATTEMPT_DIR>… \
    > <REPORTS>/<label>/causes.csv
```

Its labels: `COLD_START`, `THREAD_START`, `SYSTEM_CHANGED`, `SYSTEM_HOISTED`,
`HISTORY_REWRITTEN`, `PROMPT_GROWTH`, `CACHE_CEILING`, `CACHE_EVICTION`,
`NO_CAPTURE`. Anything it leaves unlabelled is what you read by hand. Append
those lines to the same CSV, then feed the whole file back so the causes live
next to the numbers:

```bash
cat >> <REPORTS>/<label>/causes.csv <<'EOF'
0,turn 1 of the session — cold by construction
118,SYSTEM_HOISTED — a task-notification system message was appended, rewriting the prompt head
EOF
python3 analysis/perf_report.py <ATTEMPT_DIR> --causes <REPORTS>/<label>/causes.csv
```

Turn 1 of a session is cold by construction; say so and move on rather than
counting it as an anomaly. Group identical causes and give one representative
per group when there are more than ten.

## Reading the numbers

Eight things that change the conclusion if missed.

**GPU time is de-duplicated by batch, and cannot be attributed per request.**
One forward serves the whole batch; its elapsed time is stamped on every
request in it. The script keys on `forward_start_time` (identical within a
batch) and sums distinct batches. Summing the per-request column instead
inflates by the batch size and yields a "GPU time" exceeding wall clock.

**`gpu_decode_s` is blank whenever the generation worker runs the overlap
scheduler.** Only a worker with `disable_overlap_scheduler: true` emits
`time_breakdown_metrics`; the context worker has it, the generation worker
does not. This is a measurement gap, not idle GPUs. Recovering it means one
control run with overlap off — which changes the thing being measured.

**`device_step_time_ms` is not GPU-busy time.** Its CUDA events bracket the
whole loop body, so host stalls fall inside the interval and it tracks wall
clock. Use it for per-iteration wall, never as GPU utilization.

**Evicted tokens are an upper bound.** `alloc_total - alloc_new` counts blocks
taken while still holding reusable content, which is exactly eviction — but a
freed *partial* block also lands there without costing anyone a cache hit,
roughly one per finished sequence. A small non-zero floor at low
`kv_util_mean` is that artifact.

**The imbalance column is per-metric, and `utilization`'s value is
structural.** It is `(max - mean) / mean` across an instance's ranks, averaged
over iterations; 0 means every rank carried the same amount that iteration.
For **utilization under attention-DP it is pinned at `ranks - 1`** — one
prefill occupies one rank, so every prefilling iteration scores 3.0 on 4 ranks
no matter how well the router balanced. Do **not** read that as "one rank does
all the work"; it says nothing about routing. For **KV util** the column is
meaningful, because blocks are rank-resident state that persists across
iterations. For **hit rate** it renders `—`: only the prefilling rank reports
one that iteration, and a spread over a single sample is undefined rather than
zero.

Routing skew is a cumulative question, not a per-iteration one — group
`ctx_rank_iters.csv` by rank over the whole run and compare token totals. On
the GLM pilot the per-iteration column read 2.99 (structural) while the
cumulative split was 25/25/101/186 iterations, a real 7.4x skew, with the
busiest rank also the best-hitting one (0.923 vs 0.810) — affinity working.

**Hit rate and KV utilization are charted per rank, from prefill workers
only.** One line per rank; pooling would hide the rank that is missing while
its peers hit, which is the one case the curve exists to show. Section 3 has no
per-request hit-rate curve — that grain lives in section 1, where a request can
be named.

Hit rate is Δ`kv_reused_blocks` / (Δ`reused` + Δ`missed`) per rank per
iteration — of the blocks the KV manager acquired between two iterations, the
share that came from reuse. It needs `kv_reused_blocks` / `kv_missed_blocks` in
the iteration log, added in `ea7e43043`; **on older runs there is no hit rate at
all** and the report says so rather than substituting something else.

It is block-level, so a partially matched block counts wholly as a miss. That
matters on agent traffic, where a turn appends a little to a long cached prefix:
the appended tokens need one fresh block and no reuse, so the iteration scores
near 0 even though almost none of the prompt was recomputed. Expect the mean to
sit well below the p50 for that reason — on the DeepSeek pilot, mean 0.740
against p50 0.971, with 22% of prefilling iterations reading exactly 0. Read the
p50; treat the mean as a statement about allocation pressure, not about how much
prompt was recomputed.

**An idle rank shows `scheduled_requests = 1`, not 0.** Attention-DP pads a
work-free rank with a dummy so all ranks step together, and the counter is the
raw batch size.

**Two TTFT columns, two definitions.** `ttft_ms` is the HTTP edge and exists
only for streaming requests; `ttft_engine_ms` is the engine and always exists.
Never pool them.

## Join keys

The bridging id differs by deployment; the wrong one joins nothing and raises
no error. `perf_report.py` resolves it, but hand analysis of the CSVs needs it:

- **Disaggregated** — `disagg_request_id`, which equals `ctx_request_id` and
  the engine request id on both workers. `engine_request_id` is **null**.
- **Aggregated** — `engine_request_id`, which equals the serve-edge client id.

Counters restart with their process, so joins hold only within one attempt
directory. Failed requests (`http_500`) carry no ids and cannot join; they stay
in `requests.csv` as rows with an empty `rid`.

`perf_metrics-{ctx,gen}-N.jsonl` being empty in a disaggregated run is by
design, not a fault: `/perf_metrics` drains on read and the proxy polls first,
so `perf_metrics-proxy.jsonl` holds everything — with both hops already paired.
Cross-check its row count against `anthropic_audit.jsonl` and
`adp_route_trace-ctx-*.jsonl`; those are independent full counts, and a
shortfall means the proxy's pairing buffer is overflowing.
