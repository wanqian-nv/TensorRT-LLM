---
name: trtllm-perf-report
description: Build the four-section performance report for one TRT-LLM serving run — per request, per session, prefill-server iterations, and run totals — as an HTML report plus four CSVs, then rebuild the conversation threads so every per-turn figure is keyed on (session, thread, turn) rather than on session id alone. Use when the user asks to analyze a run's performance, check prefill utilization or rank imbalance, account for where a run's time went, trace a conversation across turns, or investigate cache hit rate.
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

## Step 2 — Rebuild threads (always, before reading any per-turn figure)

**`session_id` is not a thread.** Claude Code fans subagents and background
tasks out over one session id, so "the next request in this session" is often a
different branch. Measured on one 13.7k-request run: 512 of 620 sessions
carried concurrent in-flight requests, **21.5% of the per-session gaps came out
negative**, and 622 session ids resolved into **1,227 real threads**. Every
per-turn column `perf_report.py` writes — `session_turn_index`,
`gap_to_next_turn_ms`, `realization`, `cache_realization` — is computed by
sorting a session by time and is wrong wherever that assumption breaks.

```bash
L=<REPORTS>/<label>
python3 analysis/extract_threads.py   <ATTEMPT_DIR>… --out $L/_threadstudy
python3 analysis/threads_and_tools.py $L <ATTEMPT_DIR>…
python3 analysis/thread_kinds.py      $L
python3 analysis/build_master.py      $L

# what one thread's opening turn reuses from other threads. Reads the captured
# bodies, so it is its own script; writes cross_thread_reuse.json, which the
# report picks up from the report directory with no extra flag.
python3 analysis/cross_thread_reuse.py $L <ATTEMPT_DIR>…

# then re-run step 1 with the threads, which fills section 1's KV-miss table.
# Same window flags as the first pass -- a different window silently rebuilds
# every CSV against it.
python3 analysis/perf_report.py <ATTEMPT_DIR>… [--since/--until …] \
        --threads $L/_threadstudy/threads.csv
```

Two passes, because `perf_report.py` does not read the captured bodies and so
cannot rebuild threads itself, while the rebuild needs the `requests.csv` the
first pass writes. Without `--threads` the report says so and skips the table.

Writes **`requests_master.csv`** — one row per captured request, 75 columns:
everything `requests.csv` had, plus the thread structure and the tool
accounting. **Use it instead of `requests.csv` for anything per-turn.** It
covers every captured request, not just the window, so a thread whose root
predates the window stays whole; `in_window` marks the rows carrying engine
metrics.

The columns that are not in `requests.csv`:

| column | meaning |
|---|---|
| `thread_id` / `thread_turn` | the thread and the turn index within it — the join key for anything sequential |
| `parent_aid` / `parent_rid` | the previous turn of *this* thread; empty on a root |
| `gap_from_parent_ms` | the client-side wait **before** this turn. Never negative |
| `gap_to_child_ms` | the wait after it, for attributing a wait to the tools that caused it |
| `thread_kind` | `main` / `subagent` / `sdk` / `title` / `no-tools` / `other` |
| `is_subagent` | from the `cc_is_subagent=true` billing-header flag — authoritative |
| `subagent_type` | best-effort only; resolved by matching `Agent` prompts, and only 63 of 277 launches were unambiguous. Use `is_subagent` to decide *whether*, this only to guess *which* |
| `tool_surface_max_ms` / `tool_job_max_ms` | how long the result took to come back, versus how long the backgrounded job actually ran |
| `link_key` | `digest`, `chain-rescue`, or empty for a root |

`_threadstudy/tool_calls.csv` is the one drill-down worth keeping — one row per
tool call (17.5k on that run, so it cannot fold into a per-request table), with
`wait_mode`, `task_id`, `poll_status`, `t_complete_ms`, `notify_status`. The
`tool_*` columns in the master are its aggregates.

### Check the three invariants every time

They are cheap, they take no part in building the threads, and they are the
only thing standing between a harness change and a silently wrong table.

| invariant | expected | what a drop means |
|---|---|---|
| tool result lands in the emitter's **child** turn | ~100% (measured 17,532/17,537) | parent links are breaking |
| `gap_from_parent_ms` never negative | exactly 0 | a sibling branch is being linked as a parent |
| threads rooted mid-history | small and explainable | the harness is editing the prompt head and links are being dropped |

A broken link is not loud: it shows up as **one extra thread whose first turn
has a low hit rate**, which reads exactly like a legitimate cold start. That is
why the invariants matter more than the thread count.

### What the reconstruction keys on

Per-message digests over `body.messages` only, **normalised, not tolerated**.
Four things are canonicalised or removed first, each because leaving it in
measurably broke links:

- **`cache_control`** — the prompt-cache breakpoint migrates forward between
  turns; leaving it in desynchronised the last two messages of every parent and
  cost 21.5% of links.
- **`role: "system"` messages inside `messages`** — harness-injected agent and
  skill directories, rewritten as MCP servers come and go, on 22.4% of
  transitions.
- **`<system-reminder>` blocks inside user messages** — injected at the *head*,
  which shifts every following block and breaks the digest on message 0.
- **the message body's two encodings** — the API takes a body either as `"X"`
  or as `[{"type":"text","text":"X"}]`, and Claude Code alternates between them
  for identical text. Canonicalising to the block form is what let the
  trailing-message tolerance be **deleted**: it had been carried on the belief
  that thinking blocks get rewritten, but this encoding flip was the only thing
  it ever absorbed, and with it normalised the tolerance changes nothing —
  threads, links, negative gaps and result-in-child are identical with and
  without. Every digest link is now an exact prefix match.

Prefer normalising a field to forgiving a mismatch. A tolerance is a licence to
link two requests that genuinely differ, and it hides the day the harness starts
differing for a new reason.

`body.system` and `body.tools` are deliberately **not** in the key: a system
prompt edit is a *cause* of a hit-rate drop, and keying on it would let the
cause destroy the evidence.

Two ceilings guard against linking to something too far away, both set from the
measured shape of a turn (+2 conversation messages on 99.98% of links):

- a candidate more than **8 messages** back is refused. Without it, a request
  whose immediate parent is missing attaches to a shallow ancestor instead —
  observed once, a 75-message request linked to a 5-message turn 3, inventing a
  96-minute gap and a hit rate falling 0.93 → 0.17 that no eviction caused.
- parent and child must agree on `is_subagent`. A subagent's turns are a
  different thread from its launcher's, however causally downstream.

A tool_use-id chain is the fallback for requests the digest still orphans. It
is immune to everything above, but it is *under*-constrained on its own — run
as the primary key it collapsed 1,229 threads to 572 and produced 541 negative
gaps by merging sibling branches — so it only ever rescues, never leads, and it
must share at least one real tool id (an empty chain prefix matches every
request that has not called a tool yet).

### Old columns are prefixed `legacy_` — do not read them by habit

`perf_report.py` derives four per-turn figures by sorting a session by wall
clock. The master carries them through for comparison but renames them, so the
name itself says not to use it. Measured share of rows each gets wrong on one
13.7k-request run:

| legacy column | wrong on | use instead |
|---|---|---|
| `legacy_session_turn_index` | **96.7%** disagree with `thread_turn` | `thread_turn` |
| `legacy_gap_to_next_in_session_ms` | **22.8%** negative | `gap_to_child_ms` |
| `legacy_cache_realization` | **21.7%** above 1.0 — impossible | recompute against `parent_aid` |
| `legacy_realization` | **9.5%** above 1.0 — impossible | recompute against `parent_aid` |

`prompt_lcp_tokens`, `previous_prompt_tokens`, `lcp_opportunity` and
`lcp_retention` come from the same gateway-side tracker — it keeps one previous
prompt per session and diffs against whichever request landed last — but showed
no out-of-range values, so they keep their names. Treat them with the same
suspicion on any session that interleaves threads.

Everything else from `requests.csv` is measurement, not derivation, and is
carried through unchanged: `kv_hit_rate`, `isl_*`, `osl`, the latency columns,
the iteration and block counters, and the routing columns.

### The trap that produces most of the impossible values

Claude Code fires a tool-less **session-title** request concurrently with the
first real turn, and it usually reaches the gateway ~0.6s earlier — measured
p50 −0.60s, with 87% of the pairs in flight at once. So it takes turn 1: 379 of
387 `session_turn_index == 1` rows were title calls, which pushed the real
first turn to "turn 2" and computed its realization against the title call's
705-token prompt. **360 of 379 came out above 1.0, one at 94.2.**

It shares nothing with the conversation — the two prompts diverge at character
47, inside the `cc_version` string of the billing header, so it cannot warm the
cache either. `thread_kind == 'title'` isolates them; the rebuilt `thread_turn`
already excludes them from the conversation's numbering.

### Cross-thread reuse — what an opening turn gets for free

A thread's first turn has no parent, so **every token it hits came from another
thread**: `isl_cached` at `thread_turn == 1` is an exact, server-measured figure
for cross-thread reuse, needing no reconstruction. Title calls are excluded —
they are not conversations and would be a third of the population.

What that number does not say is *with whom*, which is the part worth knowing.
`cross_thread_reuse.py` recovers it by comparing the serialised prompts of the
roots pairwise — a prefix-hash ladder, not 274k string compares — and reading
the relationship off the two threads' kinds. On the reference run:

| the opening turn shares with | roots | reused p50 | ratio p50 | past boilerplate p50 |
|---|---|---|---|---|
| unrelated — system + tools only | 280 | 25,856 | 0.532 | 0 |
| subagent ↔ subagent, same definition | 229 | 8,832 | 0.275 | 0 |
| main ↔ main | 144 | 57,344 | 0.735 | **11,199** |
| nothing earlier shares a prefix | 50 | 128 | 0.275 | 0 |
| sdk ↔ sdk | 26 | 24,576 | 0.579 | 0 |

`roots` counts thread openings, not pairs — each root is classified once
against the single earlier root it shares the deepest prefix with, so the
column sums to the population. One partner serves many: 690 roots resolved to
just 192 distinct partners, the busiest of them to 60.

Three things this says that the aggregate hit rate does not:

- **"Unrelated" is the largest bucket and is not small reuse.** Sharing only
  the system prompt and tool definitions is still ~25k tokens, over half of
  those prompts. Boilerplate is the bulk of cross-thread reuse: of 19.6M tokens
  reused at openings, **83.8% never gets past that block**.
- **`subagent ↔ main` is structurally impossible, and it is 0.** Their prompts
  diverge at **character 44**, inside the `cc_version` string — because
  `cc_is_subagent=true` sits in `system[0]`, ahead of everything. Then the
  persona line differs, then `system[2]` is a 2,164-char agent definition
  against a 58,003-char harness prompt. 44 characters is ~12 tokens, less than
  one block, so the two populations share nothing in cache terms whatever they
  have in common semantically.
- **Only `main ↔ main` reuses real content** — median 11,199 tokens past the
  boilerplate, from repeated runs of the same workflow sharing an opening.

The headline to carry: **cross-thread sharing that goes past the boilerplate
saved 6.89% of every token the run computed** (3.18M of 46.2M). Section 1
prints it. Take the counterfactual carefully — the same run's *within-thread*
reuse is far larger, 96.3% of what an uncached run would have computed, so
cross-thread reuse is 1.6% of total savings while being half of any one opening
turn's prompt.

The leverage is the boilerplate, not the content: those 16.4M tokens live or
die on whether threads of the same `cc_version` keep it warm. That field
partitions the cache — 17 distinct builds in one run — and a build with few
threads pays the ~25k afresh every time.

## Step 3 — Report

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
- **`low_hit_rate_requests` non-zero** → section 1's KV-miss table, not the
  request list. A low ratio is usually a turn appending new content to a prefix
  the cache served whole, which costs nothing recoverable. Step 4.

Quote p50/p90, never only the mean: agent traffic is long-tailed, and a mean
TTFT three times its median is the normal shape, not an anomaly.

### Where conversations land, and why — section 3's two tables

`imbalance` says the ranks carry different amounts. These say what that is made
of. Both need `--threads`; the reason column is replayed from the routing trace.

**Thread roots per rank.** A thread is *pinned*: 94.2% never leave the rank
their opening turn was routed to, because every later turn carries the previous
prompt and that rank's match is unbeatable. One routing decision therefore owns
a whole conversation's compute, which is why the token column totals the entire
chain rather than the opening turn.

```
rank   main  subagent  title  sdk  no-tools  other  threads   thread ISL new   share
0         0       157     70    0        2      1      230        7,211,120   15.6%
1        61        46    167   18        4      8      304        8,194,794   17.8%
2       279        29     58    5        0      0      371       21,654,251   46.9%
3        53        27     84   45        5      0      214        9,102,658   19.7%
all     393       259    379   68       11      9    1,119       46,162,823  100.0%
```

Read the two totals against each other. Thread counts are close (230–371) while
tokens are not: **rank 2 took 33% of the threads and 46.9% of the compute.**
That is not the router misdividing — it is `main` being much heavier than
`title` or `subagent`, and affinity keeping each kind on one rank. Rank 0 is 98%
subagent, rank 2 is 89% main, and `subagent ↔ main` never share a rank because
their prompts diverge at character 44.

**Why each thread landed there.** Replaying the scoring loop per batch:

```
rank   best match   tie, req_id shuffle   candidates capped   tie, fewest tokens   warmup
0             156                    64                   6                    3        1
1             240                    54                   8                    1        1
2             280                    64                  18                    8        1
3             114                    91                   4                    4        1
all           790                   273                  36                   16        4
```

- **best match, 70.6%** — affinity deciding, as configured.
- **tie → req_id shuffle, 24.4%** — every candidate scored identically *and*
  carried the same load, so the winner is the first entry of a permutation
  seeded on the request id. 91% of these are the affinity gate firing: a prompt
  matching under `match_rate_threshold` (0.1) of itself *anywhere* has every
  rank's match forced to 0, so the scores tie by construction. 219 of the 273
  are session-title calls, which share nothing with any conversation. **This is
  where a rank's identity originally comes from** — not policy, a seeded
  permutation, which affinity then locks in.
- **candidates capped, 3.2%** — the fair-share cap had already dropped a rank
  for the rest of that batch.
- **warmup, 4** — exactly `dp_size`. `cold_start_warmup` fires once at router
  init and never again, but those four decisions seed which rank holds which
  kind's boilerplate, and everything after follows from that.

The gate ratio and the cap are the two knobs, and note which one is which:
`kv_cache_routing_load_balance_weight: 0.0` already removes the load term from
scoring, so there is nothing further to relax there — the residual balancing is
`kv_cache_routing_fair_share_multiplier`, a hard candidate-set cut that the
weight does not control.

### The shuffle is cheap; do not chase it

Measured, the whole shuffle path forfeits **22,773 tokens, 0.0493% of everything
the run computed**, and 98% of that is title calls whose best available match
anywhere was 128 tokens — one block. The non-title cases forfeit 501 tokens
between them. A tie is not a mistake: it means no rank had anything worth
having. Price it before treating it as a bug — the same discipline as Step 4.

## Step 4 — Price the cache misses before explaining them

Section 1's **KV-miss table** is the entry point, and the order matters: decide
whether a miss costs anything before spending time on why it happened.

For one turn the tokens the engine computed split exactly two ways:

```
isl_new = (this ISL − parent ISL)      content that never existed: unavoidable
        + (parent ISL − this cached)   prefix the cache held last turn: the miss
```

Only the second term was ever recoverable, and the table reports it per cause
against two denominators — every token the run computed and every millisecond
of prefill — plus the wall-clock share underneath. **A cause worth a fraction
of a percent of prefill is not worth engineering effort however bad it looks on
one request.** On the run this was built against the whole thing came to 7.51%
of tokens, 3.21% of prefill and **0.64% of wall clock**, which settles it.

The ceiling is exact and needs no tokenizer: the rebuilt link guarantees the
parent's messages are a strict prefix of the child's, so when `system` and
`tools` are unchanged the token-level LCP *is* the parent's `isl_total`.
Measured, the miss is **exactly zero on 99.29%** of transitions — the noise
floor is 0, so any non-zero miss is real and no threshold has to be chosen.

Causes are named positively, never by elimination:

| cause | test |
|---|---|
| `SYSTEM_CHANGED` | parent and child `sys_digest` differ |
| `HISTORY_REWRITTEN` | child has fewer messages than the parent |
| `EVICTED` | the rank the parent ran on was probed and no longer holds it |
| `ROUTE_CANDIDATE_ABSENT` | that rank is **missing from the child's `match_lens`** — never asked |
| `ROUTE_LOSS` | that rank still holds it and the router chose another |
| `UNEXPLAINED …` | matches none of the above; called out rather than blamed on the server |

The last three need the per-rank probe, which is why `requests.csv` carries
`route_match_lens` and not only `match_len_best`: **`match_len_best` is a max
over that dict, so it cannot tell "every rank was asked and none had it" from
"the rank that had it was never asked"** — a capacity problem and a routing
problem respectively, and on the reference run the second was three times the
first (0.48% of wall against 0.16%). A residual bucket would have reported both
as one cache problem and pointed at the wrong fix.

Two things that do **not** cause a miss, both checked rather than assumed:

- **A long previous output.** It sits in the denominator, not the numerator —
  the parent's output is generated on the generation worker, which runs with
  `enable_block_reuse: false`, so it was never in the context worker's pool and
  the match stops at the parent's prompt either way. Measured: parent `osl` p50
  325 in the miss group against 379 in the no-miss group, correlation 0.04, and
  86 of 87 misses are larger than the parent's whole output.
- **Block alignment.** 10,988 of 11,070 zero-miss transitions have a parent
  `isl` that is not a multiple of `tokens_per_block`; the partial tail block is
  reusable.

A **negative** miss is not a miss: at a branch point (`n_children > 1`) a
sibling that ran first leaves blocks the second one legitimately matches past
the parent, so the parent alone is not the ceiling. Those are skipped.

### `classify_causes.py` is superseded for this purpose

It labels only requests under the hit-rate floor, and it does its own prefix
matching without the normalisations of Step 2 — so it reads a harness-injected
`<system-reminder>` as a prompt change. On the reference run it labelled cases
`SYSTEM_HOISTED` and `HISTORY_REWRITTEN` whose prompts were **99.9% identical
to their parent's**; the miss was entirely cache-side. Its `cause` column is
still written for the record, but the KV-miss table is what to act on.

It is still worth running — a per-request label is useful once the table says a
cause is worth chasing. It reads `requests.csv`, so the report directory comes
first and the attempt directories second, and the causes land beside the
numbers they explain:

```bash
python3 analysis/classify_causes.py $L <ATTEMPT_DIR>… > $L/causes.csv
python3 analysis/perf_report.py <ATTEMPT_DIR>… [--since/--until …] \
        --threads $L/_threadstudy/threads.csv --causes $L/causes.csv
```

Labels: `COLD_START`, `THREAD_START`, `SYSTEM_CHANGED`, `SYSTEM_HOISTED`,
`HISTORY_REWRITTEN`, `PROMPT_GROWTH`, `CACHE_CEILING`, `CACHE_EVICTION`,
`NO_CAPTURE` — read them against the table's verdict, not instead of it.

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

**That same pad lands in `num_ctx_tokens`, and on a context-only worker it is
most of the column.** The dummy is sized at exactly `max_num_tokens`, and
`model_engine` filters `is_attention_dp_dummy` only inside its *generation*
branch — the context loop feeding `num_ctx_tokens = len(input_ids)` does not.
So an idle rank logs what looks like a full chunk of prefill. Measured, the pad
was **84.5% and 87.6%** of every context token two runs reported. Uncorrected it
inflates the per-rank token totals and pins mean utilisation near 1.0 *the more
idle the worker is* — one run read `prefill_utilization_mean` 0.836 against a
true 0.129, its p90 and p99 both 1.000 against a true 0.257 and 0.521, and its
per-rank shares (26.4/24.3/26.2/23.2) were mostly measuring idleness; corrected
they are 23.4/27.9/18.9/29.9, which then matches an independent count of
`isl_new` by each request's own `routed_rank`.

**Correcting it means counting the pad as 0, not deleting it.** A pad still
occupies its rank for the iteration, so it belongs in the denominator. All
three conventions are available and two of them are wrong:

| convention | util | imbalance | why it is wrong |
|---|---|---|---|
| pad at logged size | 0.836 | 0.219 | pad *is* `max_num_tokens`, so peak is pinned at the ceiling and imbalance collapses to `1/util − 1` — verified on 14,006 of 14,006 pad-bearing iterations. The column carries nothing the util column did not. |
| drop the pad row | 0.413 | 0.066 | divides by the busy ranks alone. One rank on a full chunk beside three idle ones reads util **1.000**, imbalance **0.000** — identical to all four running full. 85% of prefilling iterations look like that, 87% of the column is exact zero. |
| **pad as 0** | **0.129** | **2.810** | one rank of four working scores `(t − t/4)/(t/4)` = exactly 3 = `ranks − 1`. |

Sanity check that the third is the live one: `Pinned by in-flight` imbalance
(2.780 on that run) is structural for the same reason and should sit at the
same magnitude as the utilization imbalance. Under the drop-the-pad convention
they read 2.780 and 0.066 in the same table.

Note where `imbalance` saturates: with one busy rank it is `ranks − 1`
*whatever* that rank computed — `t` cancels. So it counts how many ranks were
busy together, not how unevenly the busy ones were loaded. **Routing skew comes
off the per-rank totals table, never off this column.**

`perf_report.py` detects the pad (`_is_adp_pad`, column `is_adp_pad` in
`ctx_rank_iters`) and zeroes it in section 3. The test is a **conjunction**
and both halves matter: `num_ctx_tokens == max_num_tokens` **and** all four of
`kv_reused/missed/alloc_total/alloc_new_blocks` unchanged since that rank last
stepped, because `kv_cache_manager_v2` calls `stop_committing()` on the dummy.
Size alone would kill the 8.6% of full-budget rank-iterations that are real
work; frozen counters alone would swallow a pure-cache-hit iteration, which
allocates nothing either.

**Its premise, and how to re-check it.** The test assumes the dummy is exactly
`max_num_tokens`. `py_executor` takes a `min(...)` of four budgets and then
clamps to `block_capacity - extra_kv_tokens`; on these deployments the clamp is
inert, but a config where it bites would shrink the pad and this test would
quietly stop finding it. **The check:** with pads removed, summed context
tokens should equal the audit's `isl_new` over the same window — measured 0
residual on one run, +0.29% on another (30 requests whose ctx perf records carry
`ctx_request_id: null`, which the audit never saw). If that agreement drifts,
re-derive the dummy's size before trusting any prefill figure in section 3.

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
