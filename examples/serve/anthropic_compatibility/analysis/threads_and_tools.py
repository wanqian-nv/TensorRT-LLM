#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 + 1: rebuild conversation threads, then account for every tool call.

    python3 analysis/threads_and_tools.py <reports_dir> <attempt_dir>…

Stage 0 -- `session_id` is not a thread. 512 of this run's 620 sessions carry
concurrent in-flight requests (subagent fan-out), so "the next request in the
session" is often a different branch: 21.5% of the per-session gaps in
`requests.csv` are negative. Threads are rebuilt here by prefix-matching the
per-message digests, so every request gets its true parent and a gap measured
against that parent.

Stage 1 -- every tool call is resolved to (emitting turn, surfacing turn) and
classified by whether the client actually had to wait for it. Three tools
return a handle immediately and finish later (`Bash run_in_background`,
`Agent`, `Monitor`), one blocks for a poll window (`TaskOutput`), and one
blocks for its full timeout before being backgrounded (`Bash` that overran).
Folding those into one "tool latency" number is what makes the existing
`tool_latency_ms` column unreadable, so they are kept apart.

Writes `threads.csv`, `tool_calls.csv` and a `stage01_report.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)

# There is deliberately no trailing-message tolerance here. One was carried for
# a while on the belief that Claude Code rewrites the last message between
# turns; measured, the only thing it ever absorbed was the API's two equivalent
# encodings of a message body -- [{"type":"text","text":X}] on one turn and "X"
# on the next, byte-identical text. `_canon` in extract_threads.py normalises
# that, and with it normalised the tolerance changes nothing at all: threads,
# links, negative gaps and the result-in-child invariant are identical with and
# without it. A tolerance that forgives a real mismatch is a way to link two
# requests that genuinely differ, so it is gone rather than kept "just in case".

# One turn appends the assistant's reply plus the user's tool results, and
# sometimes a harness-injected `role: system` message on top. Measured over
# the 12,464 links the well-constrained digest key made on this run:
# +2 messages 77.6%, +3 messages 22.3%, everything else 0.06%; tool calls
# +1..+5 covers 99.2%. The digest key is constrained enough not to need a
# bound, but the chain rescue is not -- without one, an empty chain prefix
# matches every tool-less request, and 11 replayed requests were parented to
# a 1-message stub 35 messages shallower than themselves. These are the
# ceilings that rules out, set well above the observed spread.
RESCUE_MAX_NEW_MSGS = 8
RESCUE_MAX_NEW_CALLS = 8

# The same ceiling on the digest path. It is well constrained about *ancestry*
# -- the candidate's whole history must prefix the child's -- but says nothing
# about *distance*, so when a request's immediate parent is missing from the
# trace it happily attaches to a shallow ancestor instead. Observed once here:
# a 75-message request linked to a 5-message turn 3, inventing a 96-minute gap
# and a hit rate falling 0.93 -> 0.17 that no eviction caused. Once the jump is
# past a few messages the immediate parent is simply absent, and an honest root
# beats a fabricated 35-turn stride.
LINK_MAX_NEW_MSGS = 8

# --- result-text signatures, all verbatim from this run's captures ---------
BG_LAUNCH = re.compile(r"^Command running in background with ID: (\w+)")
BG_TIMEOUT = re.compile(
    r"^Command did not complete within its (\d+)s timeout and was moved to "
    r"the background \(ID: (\w+)\)")
BG_AGENT = re.compile(r"^Async agent launched successfully")
BG_MONITOR = re.compile(r"^Monitor started \(task (\w+), timeout (\d+)ms\)")
BG_WAKEUP = re.compile(r"^Next wakeup scheduled for ")
POLL_STATUS = re.compile(r"<retrieval_status>(\w+)</retrieval_status>")
POLL_TASK = re.compile(r"<task_id>(\w+)</task_id>")
POLL_STATE = re.compile(r"<status>(\w+)</status>")


def classify_result(tool: str, head: str) -> tuple[str, dict]:
    """(waiting mode, extra facts) for one tool_result.

    handle_now  -- returned a handle at once; the real work outlives the call
    blocked_out -- the client blocked for the tool's whole timeout, then it
                   was backgrounded; the wait is real and equals the timeout
    poll_hit / poll_miss -- a retrieval against work already running
    sync        -- an ordinary call the client waited on
    """
    m = BG_TIMEOUT.match(head)
    if m:
        return "blocked_out", {"task_id": m.group(2),
                               "timeout_s": float(m.group(1))}
    m = BG_LAUNCH.match(head)
    if m:
        return "handle_now", {"task_id": m.group(1)}
    if BG_AGENT.match(head):
        return "handle_now", {}
    m = BG_MONITOR.match(head)
    if m:
        return "handle_now", {"task_id": m.group(1),
                              "timeout_s": float(m.group(2)) / 1000.0}
    if BG_WAKEUP.match(head):
        return "handle_now", {}
    if tool == "TaskOutput" or POLL_STATUS.search(head):
        status = POLL_STATUS.search(head)
        task = POLL_TASK.search(head)
        state = POLL_STATE.search(head)
        got = status.group(1) if status else "?"
        extra = {"poll_status": got}
        if task:
            extra["task_id"] = task.group(1)
        if state:
            extra["task_state"] = state.group(1)
        return ("poll_hit" if got == "success" else "poll_miss"), extra
    return "sync", {}


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# stage 0 -- threads
# --------------------------------------------------------------------------
def _kind(fact: dict) -> str:
    """What kind of agent this request belongs to.

    `cc_is_subagent=true` in the billing header is the one authoritative flag;
    the rest is read off the tool set and the persona line, which are what
    actually differ. Carried on every row so a downstream report can group by
    it without re-reading the captured bodies.
    """
    head = fact.get("sys_head") or ""
    if "cc_is_subagent=true" in head:
        return "subagent"
    if not (fact.get("tool_names") or []):
        # The tool-less title generator has one fixed prompt length; anything
        # else tool-less is a probe or a one-shot extraction, not a session.
        return "title" if fact.get("sys_chars") == 1327 else "no-tools"
    if "You are Claude Code" in head:
        return "main"
    if "You are a Claude agent" in head:
        return "sdk"
    return "other"


def _chain(fact: dict) -> tuple:
    """The tool_use ids in this request's history, in order.

    This is the conversation's causal spine and the one part of a prompt no
    harness rewrites: an id is a random token the model minted on an earlier
    turn and every later turn carries it verbatim. Keying threads on it makes
    the reconstruction immune to everything that also *causes* a hit-rate drop
    -- system prompt edits, tool-definition churn, `role: system` messages
    appended into `messages` (measured: 22.4% of transitions), the
    `<system-reminder>` block and its rolling `currentDate`, `cache_control`
    migration, thinking retention. Keying on message digests instead couples
    the two: the cause silently destroys the evidence, and a broken link is
    indistinguishable from a genuinely new thread.
    """
    return tuple(u["id"] for u in fact.get("uses") or [] if u.get("id"))


def build_threads(facts: dict, meta: dict) -> Counter:
    """Give every request a parent, a thread id and a gap against that parent.

    Two keys, in strict precedence -- and the order matters, because measuring
    the other way round is what proved it:

    1. message-digest prefix. Well constrained: the parent's whole history must
       prefix the child's, so sibling branches cannot be confused.
    2. tool_use chain, used *only* to rescue a request the digest left an
       orphan. The chain is immune to everything a harness rewrites (system
       prompt, tool definitions, `role: system` messages appended into
       `messages` -- 22.4% of transitions -- `<system-reminder>` and its
       rolling date, `cache_control`, thinking retention), so it recovers links
       the digest drops when the harness edits the head of the prompt.

    Running the chain as the *primary* key was tried and is wrong: it is
    under-constrained, seeing only tool ids and none of the text, so it merges
    sibling branches. Measured, it collapsed 1,229 threads to 572, produced 541
    negative parent-relative gaps, and chose 218 parents whose history was not
    even a prefix of the child. The rescue below therefore also demands the
    parent had *finished* before the child started, which is what rules a
    concurrently in-flight sibling out.
    """
    checks: Counter = Counter()
    by_session: dict[str, list[str]] = defaultdict(list)
    for aid, fact in facts.items():
        by_session[fact.get("client_session_id") or ""].append(aid)

    for ids in by_session.values():
        ids.sort(key=lambda a: (meta[a]["started_at"] or 0.0, a))
        chains = {a: _chain(facts[a]) for a in ids}
        # Compare the conversation, not the harness's commentary. `messages`
        # carries `role: "system"` entries the harness injects (the agent
        # directory, the skill directory) and rewrites as MCP servers come and
        # go -- measured on 22.4% of transitions. They are not conversation
        # turns, so they are dropped before digesting; measured, this takes the
        # result-in-child invariant from 99.99% to 100.00%.
        # What an agent *is* cannot change mid-conversation: a subagent's
        # turns are a different thread from its launcher's, however causally
        # downstream they are. Measured, no legitimate link crosses this
        # boundary (0 of 12,466) and no thread is internally mixed (0 of
        # 1,227), so the constraint rejects nothing real -- but it is exactly
        # what a chain rescue on an empty prefix got wrong, attaching three
        # subagent threads to the main-agent turn that launched them.
        # Rejections are counted rather than dropped: if this ever fires, the
        # harness has grown a shape that needs looking at.
        is_sub = {a: "cc_is_subagent=true" in (facts[a].get("sys_head") or "")
                  for a in ids}
        # `msgs_clean` is the per-message digest with `<system-reminder>`
        # blocks already dropped by the extractor; combined with dropping
        # `role: system` entries here, what is left is the conversation alone.
        # Measured: 1,229 -> 1,227 roots and 16 -> 14 rooted mid-history, with
        # both invariants unmoved.
        convo = {a: [d for d, role in zip(facts[a].get("msgs_clean")
                                          or facts[a]["msgs"],
                                          facts[a].get("roles") or [])
                     if role != "system"] for a in ids}
        sizes = {a: len(convo[a]) for a in ids}
        exact: dict[tuple, list[str]] = defaultdict(list)
        by_prefix: dict[tuple, list[str]] = defaultdict(list)
        for a in ids:
            msgs = convo[a]
            exact[(len(msgs), hash(tuple(msgs)))].append(a)
            by_prefix[chains[a]].append(a)

        for a in ids:
            mine, msgs, size = meta[a], convo[a], sizes[a]
            started = mine["started_at"] or 0.0
            best, how = None, ""
            for k in range(len(msgs) - 1, 0, -1):
                key = (k, hash(tuple(msgs[:k])))
                raw = exact.get(key, [])
                pool = [c for c in raw
                        if c != a
                        and (meta[c]["started_at"] or 0.0) <= started
                        and sizes[c] < size
                        and size - sizes[c] <= LINK_MAX_NEW_MSGS]
                checks["stride rejected"] += sum(
                    1 for c in raw
                    if c != a and (meta[c]["started_at"] or 0.0) <= started
                    and sizes[c] < size and size - sizes[c] > LINK_MAX_NEW_MSGS)
                cands = [c for c in pool if is_sub[c] == is_sub[a]]
                checks["subagent-boundary rejected"] += len(pool) - len(cands)
                if cands:
                    best = max(cands, key=lambda c: (sizes[c],
                                                     meta[c]["started_at"] or 0.0))
                    how = "digest"
                    break
            if best is None and chains[a]:
                chain = chains[a]
                # k >= 1: the parent has to share an actual tool_use id. The
                # empty prefix matches every request that has not called a tool
                # yet, and it linked three subagent threads to the main-agent
                # turn that launched them -- causally downstream, but a
                # different conversation sharing no prefix at all, so counting
                # it as "turn N+1" would report a fabricated hit-rate cliff.
                for k in range(len(chain) - 1, 0, -1):
                    cands = [c for c in by_prefix.get(chain[:k], [])
                             if c != a
                             and is_sub[c] == is_sub[a]
                             and len(chains[c]) < len(chain)
                             and sizes[c] < size
                             # finished, not merely started: an in-flight
                             # request is a sibling branch, not a parent
                             and (meta[c]["finished_at"] or 0.0) <= started
                             # and it has to be one turn back, not a stub
                             and size - sizes[c] <= RESCUE_MAX_NEW_MSGS
                             and len(chain) - len(chains[c]) <= RESCUE_MAX_NEW_CALLS]
                    if cands:
                        best = max(cands, key=lambda c: (len(chains[c]), sizes[c]))
                        how = "chain-rescue"
                        break
            mine["parent"] = best
            # carried so a downstream report can partition a miss by cause
            # without re-reading the captured bodies
            mine["sys_digest"] = facts[a].get("system")
            mine["kind"] = _kind(facts[a])
            mine["n_msgs"] = len(facts[a]["msgs"])
            # The role of the LAST message in the request body. On DeepSeek-V4
            # a trailing role:"system" is re-roled to latest_reminder by
            # _map_trailing_system_to_reminder, and _render_message then defers
            # the generation prompt past it -- so the generation prompt and the
            # reminder swap order between this turn and the next, and this
            # turn's token sequence stops being a prefix of its child's. The
            # separation is near total: on the 08-26 run P(parent's prefix is
            # unreusable) is 81.6% when this is "system" and 0.03% when it is
            # "user". Carried here so the report can name that cause without
            # re-reading 14k captured bodies.
            mine["tail_role"] = (facts[a].get("roles") or [None])[-1]
            mine["parent_chain_len"] = len(chains[best]) if best else 0
            mine["chain_len"] = len(chains[a])
            mine["link_key"] = how
            mine["is_subagent"] = is_sub[a]
            checks[how or "root"] += 1

    for aid in facts:
        seen, node = set(), aid
        while meta[node].get("parent") and node not in seen:
            seen.add(node)
            node = meta[node]["parent"]
        meta[aid]["thread_id"] = node
        meta[aid]["thread_depth"] = len(seen) + 1

    for aid, mine in meta.items():
        parent = mine.get("parent")
        mine["true_gap_ms"] = None
        if parent and meta[parent]["finished_at"] and mine["started_at"]:
            mine["true_gap_ms"] = (
                mine["started_at"] - meta[parent]["finished_at"]) * 1000.0
    return checks


# --------------------------------------------------------------------------
# stage 1 -- tools
# --------------------------------------------------------------------------
def build_tool_calls(facts: dict, meta: dict, emitted: dict) -> list[dict]:
    """One row per tool_use id, with both timings kept separate."""
    # Identity and input come from whichever capture first carried the block.
    spec: dict[str, dict] = {}
    order = sorted(facts, key=lambda a: (meta[a]["started_at"] or 0.0, a))
    for aid in order:
        for use in facts[aid]["uses"]:
            if use["id"] and use["id"] not in spec:
                spec[use["id"]] = {"name": use["name"], "in": use["in"],
                                   "in_bytes": use["in_bytes"]}
    # First request whose history carries the result is where it surfaced.
    surfaced: dict[str, dict] = {}
    for aid in order:
        for res in facts[aid]["results"]:
            if res["id"] and res["id"] not in surfaced:
                surfaced[res["id"]] = {"aid": aid, **res}
    # ... and the first that carries its completion notice is where the work
    # actually ended. Backgrounded calls surface a handle at once, so for them
    # `t_surface` measures the handshake and this measures the job.
    notified: dict[str, dict] = {}
    for aid in order:
        for note in facts[aid].get("notes") or []:
            if note["id"] and note["id"] not in notified:
                notified[note["id"]] = {"aid": aid, **note}

    rows = []
    for use_id, info in spec.items():
        emitter = emitted.get(use_id)
        res = surfaced.get(use_id)
        mode, extra = ("unmatched", {})
        if res:
            mode, extra = classify_result(info["name"], res["head"])
        row = {
            "tool_use_id": use_id,
            "tool": info["name"],
            "input_bytes": info["in_bytes"],
            "run_in_background": info["in"].get("run_in_background"),
            "subagent_type": info["in"].get("subagent_type"),
            "declared_timeout": info["in"].get("timeout"),
            "emitter_aid": emitter,
            "emitter_session": meta[emitter]["session_id"] if emitter in meta else None,
            "emitter_thread": meta[emitter].get("thread_id") if emitter in meta else None,
            "surfacer_aid": res["aid"] if res else None,
            "wait_mode": mode,
            "is_error": res["err"] if res else None,
            "result_chars": res["chars"] if res else None,
            "task_id": extra.get("task_id"),
            "poll_status": extra.get("poll_status"),
            "task_state": extra.get("task_state"),
            "bg_timeout_s": extra.get("timeout_s"),
            "t_surface_ms": None,
            "turns_later": None,
            "t_complete_ms": None,
            "complete_turns_later": None,
            "notify_status": None,
        }
        note = notified.get(use_id)
        if note:
            row["notify_status"] = note.get("status")
            if emitter in meta and note["aid"] in meta:
                a = meta[emitter]["finished_at"]
                b = meta[note["aid"]]["started_at"]
                if a is not None and b is not None:
                    row["t_complete_ms"] = (b - a) * 1000.0
                da = meta[emitter].get("thread_depth")
                db = meta[note["aid"]].get("thread_depth")
                if (meta[emitter].get("thread_id") == meta[note["aid"]].get("thread_id")
                        and da is not None and db is not None):
                    row["complete_turns_later"] = db - da
        if emitter in meta and res and res["aid"] in meta:
            a, b = meta[emitter]["finished_at"], meta[res["aid"]]["started_at"]
            if a is not None and b is not None:
                row["t_surface_ms"] = (b - a) * 1000.0
            # Distance along the rebuilt thread, not wall-clock ordering.
            depth_a = meta[emitter].get("thread_depth")
            depth_b = meta[res["aid"]].get("thread_depth")
            same = meta[emitter].get("thread_id") == meta[res["aid"]].get("thread_id")
            if same and depth_a is not None and depth_b is not None:
                row["turns_later"] = depth_b - depth_a
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("attempt_dir", type=Path, nargs="+")
    args = parser.parse_args()
    work = args.reports_dir / "_threadstudy"

    with (args.reports_dir / "requests.csv").open(encoding="utf-8") as handle:
        requests = list(csv.DictReader(handle))
    in_window = {r["audit_request_id"] for r in requests}
    by_aid = {r["audit_request_id"]: r for r in requests}
    print(f"requests.csv: {len(requests):,} in-window requests", file=sys.stderr)

    facts: dict[str, dict] = {}
    with (work / "capture_facts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            facts[rec["audit_request_id"]] = rec
    print(f"captures: {len(facts):,}", file=sys.stderr)

    # Timing and the authoritative emitter come from the audit, which covers
    # requests outside the report window too -- a thread must not lose its
    # root just because the root predates the analysis window.
    meta: dict[str, dict] = {}
    emitted: dict[str, str] = {}
    from datetime import datetime
    def parse(stamp):
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            return None
    for attempt in args.attempt_dir:
        with (attempt / "anthropic_audit.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                aid = rec.get("audit_request_id")
                if aid not in facts:
                    continue
                meta[aid] = {
                    "session_id": rec.get("client_session_id"),
                    "started_at": parse(rec.get("started_at")),
                    "finished_at": parse(rec.get("finished_at")),
                    "in_window": aid in in_window,
                }
                for call in (rec.get("response") or {}).get("tool_calls_emitted") or []:
                    if call.get("id"):
                        emitted[call["id"]] = aid
    # A capture with no audit record has no timestamps, so it cannot be placed
    # in a thread; observed on requests the gateway logged but never finished.
    orphans = [aid for aid in facts if aid not in meta]
    for aid in orphans:
        del facts[aid]
    print(f"audit: {len(meta):,} matched, {len(emitted):,} emitted tool calls"
          + (f", {len(orphans)} capture(s) dropped for having no audit record"
             if orphans else ""), file=sys.stderr)

    checks = build_threads(facts, meta)
    if checks["stride rejected"]:
        print(f"stride guard rejected {checks['stride rejected']:,} candidate "
              f"parent(s) more than {LINK_MAX_NEW_MSGS} messages back",
              file=sys.stderr)
    if checks["subagent-boundary rejected"]:
        print(f"subagent-boundary guard rejected "
              f"{checks['subagent-boundary rejected']:,} candidate parent(s)",
              file=sys.stderr)
    print(f"links: {checks['digest']:,} by digest prefix, "
          f"{checks['chain-rescue']:,} rescued by tool-chain, "
          f"{checks['root']:,} thread roots", file=sys.stderr)
    tool_rows = build_tool_calls(facts, meta, emitted)
    print(f"tool calls resolved: {len(tool_rows):,}", file=sys.stderr)

    with (work / "threads.csv").open("w", newline="", encoding="utf-8") as handle:
        cols = ["audit_request_id", "session_id", "thread_id", "thread_depth",
                "parent", "parent_chain_len", "chain_len", "link_key",
                "is_subagent", "kind", "sys_digest", "n_msgs", "tail_role",
                "started_at", "finished_at",
                "true_gap_ms", "in_window", "request_index", "kv_hit_rate",
                "isl_total", "isl_cached", "osl", "match_len_best",
                "match_len_chosen", "routed_rank", "session_turn_index",
                "gap_to_next_turn_ms", "tool_names"]
        out = csv.DictWriter(handle, fieldnames=cols)
        out.writeheader()
        for aid, mine in meta.items():
            src = by_aid.get(aid, {})
            out.writerow({"audit_request_id": aid, **{
                k: mine.get(k) for k in ("session_id", "thread_id", "thread_depth",
                                         "parent", "parent_chain_len", "chain_len",
                                         "link_key", "is_subagent", "kind", "sys_digest",
                                         "n_msgs", "tail_role", "started_at",
                                         "finished_at", "true_gap_ms", "in_window")},
                **{k: src.get(k) for k in ("request_index", "kv_hit_rate", "isl_total",
                                           "isl_cached", "osl", "match_len_best",
                                           "match_len_chosen", "routed_rank",
                                           "session_turn_index",
                                           "gap_to_next_turn_ms", "tool_names")}})

    with (work / "tool_calls.csv").open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=list(tool_rows[0].keys()))
        out.writeheader()
        out.writerows(tool_rows)
    print(f"wrote {work}/threads.csv and {work}/tool_calls.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
