---
name: anthropic-trace-analyze
description: Analyze an Anthropic audit trace from a run directory. Runs analyze_audit.py, always generates a run-summary Markdown table plus per-turn and pooled-distribution dashboards, then inspects turns with actual cache-hit ratio below 80% from captured request bodies. Use when the user asks to analyze a run, compare performance distributions, check cache reuse, or investigate low cache hit rates.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Audit trace analysis skill

Scope: the Anthropic audit log — per-turn cache health and tool loops, with
root-cause drill-down into captured request bodies. For a whole-run picture
that also covers prefill-server iterations and the wall-clock decomposition,
use `trtllm-perf-report`, which calls back here for the anomaly analysis.

**Skill base directory** (all relative paths below are relative to this):
`TensorRT-LLM/examples/serve/anthropic_compatibility/`

---

## Step 0 — Resolve the run directory

A run directory is any directory that directly contains
`anthropic_audit.jsonl`. It does not have to live under the skill base — the
analysis scripts take absolute paths.

Search these trace roots, in order, for a directory matching what the user named
(e.g. "repairbot", "model bringup", a SLURM job ID, or any substring):

1. `/lustre/fsw/portfolios/coreai/users/serli/claude-traces/` — current default.
   Layout is `<root>/<run_name>/attempt-NNN/`, and the audit files live in the
   **attempt** directory, not the run directory. Skip the `_fleet` and
   `_sbatch_logs` bookkeeping directories. If several attempts carry an
   `anthropic_audit.jsonl`, use the highest-numbered one unless the user asks
   for a specific attempt.
2. `<SKILL_BASE>/runs/` — legacy layout, audit files directly in `<run_name>/`.

If the user gives an explicit path, use it as-is. If the match is ambiguous,
list candidates and ask. You need:

- `<run_dir>/anthropic_audit.jsonl`   — required; abort if missing
- `<run_dir>/anthropic_message_capture/requests/`  — required for anomaly drill-down

Set:
```
SKILL_BASE = TensorRT-LLM/examples/serve/anthropic_compatibility
RUN_DIR    = <absolute path to the directory holding anthropic_audit.jsonl>
             # e.g. /lustre/.../claude-traces/serli_080423_365887_..._repairbot/attempt-002
ANALYSIS   = <RUN_DIR>/analysis
```

Locate candidates with:
```bash
find /lustre/fsw/portfolios/coreai/users/serli/claude-traces -maxdepth 3 \
     -name anthropic_audit.jsonl -printf '%h\n' | sort
```

---

## Step 1 — Run the audit analyzer

```bash
cd <SKILL_BASE>
python3 analysis/analyze_audit.py <RUN_DIR>/anthropic_audit.jsonl --out <ANALYSIS>
```

This produces:
- `<ANALYSIS>/timeline.csv`    — one row per request, all metrics
- `<ANALYSIS>/turns.jsonl`     — same data as JSONL
- `<ANALYSIS>/tool_loops.jsonl`
- `<ANALYSIS>/REPORT.md`
- `<ANALYSIS>/TIMELINE.md`

Report the summary line printed by the script (e.g. "wrote 84 timeline rows…").

---

## Step 2 — Generate dashboards

```bash
cd <SKILL_BASE>
python3 analysis/plot_dashboard.py <ANALYSIS> --title "<run label>"
```

This writes `<ANALYSIS>/dashboard.png`. Tell the user the output path.

The dashboard contains 8 panels:
1. ISL breakdown (cached + new) — stacked bar
2. OSL (output tokens)
3. TTFT (time to first token, seconds)
4. Total server latency (seconds)
5. Actual cache-hit ratio — with reuse opportunity overlay; **turns < 80% are
   highlighted in red**
6. Decode TPS/user
7. Tool calls per turn
8. Tool loop gap (client-side tool execution time)

Each panel has a mean/p50/p75/p99 stats box in the top-right corner.

Always generate the pooled distribution dashboard:

```bash
cd <SKILL_BASE>
python3 analysis/plot_distributions.py \
  --series "<run label>=<ANALYSIS>/timeline.csv" \
  --out-dir <ANALYSIS> \
  --title "<run label>"
```

This writes:

- `<ANALYSIS>/distribution_dashboard.png`
- `<ANALYSIS>/distribution_dashboard.html`
- `<ANALYSIS>/run_summary.md`

`run_summary.md` is mandatory for every analysis. It contains one column per
`--series` run and includes:

- Sessions, API requests, completed turns, failed requests, and cancellations
- Trace elapsed time (`max(finished_at) - min(started_at)`) and summed request
  latency (`sum(server_total_ms)`)
- Total processed tokens, total input ISL, cache-read ISL, new/computed ISL,
  and model output tokens
- Warm cache-hit ratio, warm TTFT p50/p95, and decode TPS/user p50
- Total tool calls, total tool-call time, tool-loop gap p50/p95, and tool-result
  error rate

Define total tool-call time as the sum of one matched
response-to-next-request gap per tool-calling turn. Do not count parallel tool
calls emitted by the same turn as separate elapsed-time intervals.

The distribution dashboard pools completed turns across sessions, excludes
missing values rather than replacing them with zero, and shows a histogram plus
ECDF with mean/p50/p75/p99 for:

1. Total ISL
2. Cached ISL
3. New/uncached ISL
4. First agent-turn ISL (the earliest request with
   `tool_definition_count > 0`, one sample per session/task; exclude Claude
   Code's title-generation preflight request)
5. OSL
6. Completed inference turns per session/task
7. TTFT
8. Total server latency
9. TPS/user
10. Actual cache-hit ratio
11. Matched tool-loop gap

Lay out the distribution dashboard three panels per row. Keep the first two
rows workload-only, with TTFT starting the third row. Format the summary
statistics for Total/Cached/New/First-agent-turn ISL with one decimal and a
`K` suffix for values of at least 1,000 tokens (for example, `mean=26.0K`).
Show values below 1,000 as integer tokens without a `K` suffix (for example,
`mean=338`). Retain natural units for the other panels.

For comparisons, repeat `--series "LABEL=/path/to/timeline.csv"` for every run.
Do not split the distribution by session unless the user explicitly requests it.

---

## Step 3 — Find low-reuse turns (< 80% cache hit)

Read `<ANALYSIS>/timeline.csv` with Python. Collect all rows where
`actual_cache_hit_ratio < 0.80`, excluding turns where the low hit is expected:

**Expected-low criteria (skip these):**
- `session_turn_index == 1` — first turn of a session always starts cold
- `session_turn_index == 2` AND `previous_prompt_retention_ratio` is blank or 0
  — second turn before LCP is established
- **the first request of a *different conversation* sharing the same
  `client_session_id`** — legitimately cold even at a high turn index. Detect by
  shape rather than by turn number: `history_message_count` of 1-2 while
  neighbouring turns in that session are much longer, and/or a different
  `tool_definition_count` or system-prompt length. Subagents and auxiliary calls
  (WebFetch summarisation, and similar) all reuse the parent's session header.
- `cache_realization_ratio > 1.0` — the LCP for this turn is unusable, so it
  cannot be classified; count it separately as `LCP_UNRELIABLE` (see 4a)

For each **unexpected** low-reuse turn, capture:
```python
{
  "global_request_index": ...,
  "session_id": ...,
  "session_turn_index": ...,
  "started_at": ...,
  "actual_cache_hit_ratio": ...,       # the anomalous value
  "isl_total": ...,
  "isl_cached": ...,
  "isl_new": ...,
  "previous_prompt_retention_ratio": ...,
  "current_reuse_opportunity_ratio": ...,
  "prompt_lcp_tokens": ...,
  "message_capture_file": ...,         # path inside RUN_DIR/anthropic_message_capture/
}
```

If there are no unexpected low-reuse turns, report that and skip Step 4.

---

## Step 4 — Drill into each anomalous request body

For each turn from Step 3, load its captured request:

```python
import gzip, json
body_path = f"<RUN_DIR>/anthropic_message_capture/{row['message_capture_file']}"
with gzip.open(body_path) as f:
    capture = json.load(f)
body = capture["body"]
```

From `body`, extract:

**System prompt fingerprint:**
```python
sys_blocks = body.get("system", [])
sys_text = " ".join(b.get("text", "") for b in sys_blocks if isinstance(b, dict))
sys_len = len(sys_text)
sys_first_100 = sys_text[:100]
```

**Message history shape:**
```python
messages = body.get("messages", [])
msg_count = len(messages)
# For each message, get role + content length
msg_shape = [(m["role"], sum(len(b.get("text","")) for b in (m["content"] if isinstance(m["content"], list) else [{"text": str(m["content"])}]))) for m in messages]
```

**Tool definitions count:**
```python
tool_count = len(body.get("tools", []))
```

**Context change vs. previous turn:**
Look at the turn BEFORE this one in the timeline (same session, `session_turn_index - 1`).
Compare `isl_total` values. The delta is `isl_total[this] - isl_total[prev]`.

Then classify the root cause.

### Step 4a — Split cache problems from prompt problems FIRST

`actual_cache_hit_ratio = cache_realization_ratio × current_reuse_opportunity_ratio`,
so a low hit rate has two very different causes. Always separate them before
looking at anything else:

```python
realization = cache_read / prompt_lcp_tokens   # timeline.csv: cache_realization_ratio
opportunity = prompt_lcp_tokens / isl_total    # timeline.csv: current_reuse_opportunity_ratio
```

| `realization` | `opportunity` | Verdict |
|---|---|---|
| low (< ~0.8) | high | **cache-side** — the reusable prefix existed but the engine did not serve it. Go to 4b. |
| high (~1.0) | low | **prompt-side** — the engine reused everything available; the prefix itself is short. Go to 4c. |
| low | low | both; treat as prompt-side first, then re-check. |

Do **not** start from `previous_prompt_retention_ratio`. That metric is computed by
`AnthropicPromptLcpTracker` (`anthropic_adapter.py`) by comparing **token ids** of
the previous prompt against the current one. It measures the *rendered* prompt, so
a server-side rewrite of the prompt looks identical to the client having dropped
context. Concluding "the client truncated history" from a low retention ratio is a
known trap — see 4c.

### Which numbers are trustworthy

| Field | Source | Trustworthy? |
|---|---|---|
| `actual_cache_hit_ratio`, `isl_cached`, `isl_total`, `osl_model_tokens` | engine `usage` | **yes** — measured, independent of the tracker |
| `prompt_lcp_tokens` and everything derived from it (`previous_prompt_retention_ratio`, `current_reuse_opportunity_ratio`, `cache_realization_ratio`) | `AnthropicPromptLcpTracker` | **conditionally** — see below |

Report cache health from `actual_cache_hit_ratio` (this is also what the dashboard
plots). Use the LCP-derived split only to *explain* that number, and only after
filtering by the rule below.

`AnthropicPromptLcpTracker` keeps **one** previous prompt per `client_session_id`.
That assumption breaks in two common situations, and when it breaks the LCP is
computed against the wrong reference prompt:

1. **Concurrent requests in one session.** The stored prompt gets overwritten by
   whichever request lands first, and observation is scheduled off the event loop
   (`schedule_anthropic_lcp_observation`), so the order it observes need not match
   request order. Symptom: `started_at` out of order within a session.
2. **Several independent conversations sharing one `client_session_id`.** A main
   agent, its subagents, and auxiliary calls (e.g. a WebFetch page-summarisation
   request) all carry the same header, each with its own system prompt, tool set
   and history — cross-comparison between them is meaningless. Spot them by shape:
   a request with far fewer `messages`, a different `tool_definition_count`, or a
   different system-prompt length than its neighbours in the same session. Their
   first request is legitimately cold and should not be counted as an anomaly.

**Detection rule: `cache_realization_ratio > 1.0` means that turn's LCP is
unusable.** The engine cannot read more cached tokens than the common prefix
contains, so anything above 100% proves the reference prompt was wrong. Values of
270% and 1466% have been observed in real traces. Exclude those turns from
LCP-based reasoning; their `actual_cache_hit_ratio` remains valid.

### Step 4b — Cache-side causes (`realization` is low)

| Condition | Root cause label |
|---|---|
| `realization` < 0.7 with `opportunity` > 0.9 | **CACHE_EVICTION** — the prefix was reusable but the KV entry was evicted (capacity pressure or TTL) |
| `isl_cached` pinned to the same value across many turns | **CACHE_CEILING** — reuse capped; check `kv_cache_config` in `server_config.yaml` |
| None of the above | **CACHE_UNKNOWN** |

### Step 4c — Prompt-side causes (`opportunity` is low)

The prompt the server rendered diverges from the previous one. The client may be
blameless — verify before blaming it.

**Check the inbound bodies first.** Compare this turn's captured body against the
previous turn's:

```python
prev_sys = json.dumps(prev_body.get("system"))
cur_sys  = json.dumps(cur_body.get("system"))
same_system = prev_sys == cur_sys            # top-level system field unchanged?
appended_only = all(json.dumps(a) == json.dumps(b)
                    for a, b in zip(prev_body["messages"], cur_body["messages"]))
```

Also record which messages are new this turn, and their **roles and positions**.

| Condition | Root cause label |
|---|---|
| a new `role:"system"` message appeared at the tail of `messages` | **SYSTEM_HOISTED** — the adapter lifts mid-conversation system messages into the opening block, rewriting the front of the prompt and invalidating the whole prefix. Client-side history is append-only and blameless. |
| top-level `system` field itself changed | **SYSTEM_CHANGED** — the client really did edit the system prompt |
| earlier messages differ (not append-only) | **HISTORY_REWRITTEN** — truncation, compaction, or summarization on the client |
| history is append-only and no new system message | **RENDER_DIVERGENCE** — same inbound body shape but the rendered token stream still diverged; suspect the conversion or chat template |
| None of the above | **PROMPT_UNKNOWN** |

**SYSTEM_HOISTED is by far the most common in Claude Code traces.** Sources of
tail-appended `role:"system"` messages seen in practice: periodic task-list
reminders, background-task completion notifications (`<task-notification>`),
one-off subagent listings, and hook errors. Confirm it with:

```python
# a NEW system message this turn is the signal, not the mere presence of one
new_system = count_system_msgs(cur) > count_system_msgs(prev)
```

### Step 4d — Confirming a prompt-side hypothesis

Byte-comparing JSON bodies is necessary but **not sufficient**: the divergence is
usually introduced after the body, during `convert_anthropic_request` or the chat
template. To locate the layer, use `analysis/repro_lcp.py`, which replays
captured bodies through the real render path and bisects across three levels
(converted request → rendered string → token ids). It gates on reproducing the
recorded LCP exactly before drawing any conclusion.

Two pitfalls when reasoning about prefixes by hand:

- Format percentages with enough precision. A 43-character difference at the end
  of a 359,183-character prefix is 99.988%, which prints as `100.0%` at `%.1f`
  and reads as "identical".
- Estimated chars-per-token is not reliable enough to map a character offset to a
  token offset. Compare token ids directly.

---

## Step 5 — Return the anomaly report

Output a structured report in this format:

```
## Audit Trace Analysis — <run label>

### Summary
- Turns analyzed: <N>
- Sessions: <K>
- Low-reuse turns (< 80%, unexpected): <M>
- Dashboard: <ANALYSIS>/dashboard.png
- Distribution dashboard: <ANALYSIS>/distribution_dashboard.html
- Run summary: <ANALYSIS>/run_summary.md

### Per-turn stats (across all turns)
| Metric | mean | p50 | p75 | p99 |
|--------|------|-----|-----|-----|
| ISL total (tokens) | … | … | … | … |
| OSL (tokens) | … | … | … | … |
| TTFT (s) | … | … | … | … |
| Total latency (s) | … | … | … | … |
| Cache hit ratio (%) | … | … | … | … |
| Decode TPS | … | … | … | … |
| Tool loop gap (s, non-zero) | … | … | … | … |

### Anomalous turns (cache hit < 80%, unexpected)

For each anomalous turn:

#### Turn <global_index> — Session <S#> turn <session_turn_index> — <root_cause_label>

- **Started at**: <started_at>
- **Cache hit**: <actual_cache_hit_ratio>
- **Split**: realization <cache_realization_ratio> × opportunity <current_reuse_opportunity_ratio>
- **Side**: cache-side | prompt-side   (from Step 4a)
- **ISL**: <isl_total> total / <isl_cached> cached / <isl_new> new
- **LCP tokens**: <prompt_lcp_tokens>
- **Root cause**: <label> — <one sentence explanation>
- **Evidence**: <what you observed in the request body — new message roles and
  positions, whether the top-level system field changed, whether history is
  append-only, isl delta>

### Attribution summary

Always include a table counting anomalous turns by root cause, so the dominant
cause is visible at a glance:

| Root cause | turns | share |
|---|---:|---:|
| SYSTEM_HOISTED | … | …% |
| CACHE_EVICTION | … | …% |
| … | … | … |

### Observations & recommendations

<2–4 bullet points with cross-turn patterns, e.g.:
- "SYSTEM_HOISTED turns recur every 4–5 turns, matching the client's periodic
  task-reminder injection; each one re-prefills the full context"
- "Cache evictions cluster around turns X–Y, suggesting KV cache pressure"
- "Wasted prefill totals N tokens (M% of all prefill); TTFT on affected turns is
  Kx the healthy median"
>
```

Do not truncate anomalous turns — report all of them. If M > 10, group identical
root causes together and show one representative example per group.

---

## Reading the tool-loop numbers

Two systematic limits — state them whenever you report tool-gap statistics:

- **`tool_loop_gap` is capped at ~600s.** That is the client-side Bash tool
  timeout (`max 600000` ms), not a real job duration. Long jobs are also commonly
  launched with `run_in_background`, which returns immediately, so their duration
  never enters this metric at all.
- **Unmatched tool loops carry no gap.** Report the unmatched count alongside
  any gap percentile. It should now be near zero; a large one means something
  new is hiding results.

`analyze_audit.py` matches results by scanning `tool_results_in_request` — the
whole history — and popping the id from the pending set on first match. The
earlier last-message-only scan missed every result the client buried behind a
trailing system message, leaving 21–26% of calls unmatched and truncating the
long tail (max gap 59s where the true value was 190s). Re-reading history is
safe precisely because of the pop: an id matches once, and since history is
append-only, that first match is the turn the result arrived in.
