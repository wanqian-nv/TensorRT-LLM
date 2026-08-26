#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pass 1: distil every captured request body into one small record.

    python3 analysis/extract_threads.py <attempt_dir>… --out <dir>

Writes `capture_facts.jsonl`, one line per captured request, carrying only
what stages 0 and 1 need: the per-message digests that let threads be
reconstructed by prefix matching, and every tool_use / tool_result block with
the input fields that decide whether a call was synchronous or backgrounded.

The bodies themselves are 2 GB of gzip; this is ~20 MB and is read many times
downstream, so the split pays for itself immediately.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import re
from pathlib import Path

# Input fields worth keeping per tool. Everything else about a call is
# discarded: the study only needs to know whether the client had to wait for
# it, plus enough identity to group calls that behave alike.
KEEP_INPUT = {
    "Bash": ("run_in_background", "timeout"),
    "Agent": ("subagent_type", "model", "isolation"),
    "Task": ("subagent_type",),
    "TaskCreate": ("subagent_type", "model"),
    "Monitor": ("timeout", "interval"),
    "ScheduleWakeup": ("delaySeconds", "stop"),
    "Workflow": ("name",),
    "Skill": ("skill",),
    "WebFetch": (),
    "SendMessage": ("to",),
}
# Enough of a result to recognise a background handle.
RESULT_HEAD = 400

# A backgrounded task never reports back as a tool_result -- the launching
# call returned its handle immediately. Completion arrives instead as a
# <task-notification> block injected into a later user message, and that block
# carries the launching <tool-use-id>. It is the only link between a launch
# and the moment the work actually finished, so it is extracted here.
NOTIFY = re.compile(
    r"<task-notification>\s*<task-id>(?P<task>[^<]*)</task-id>\s*"
    r"<tool-use-id>(?P<use>[^<]*)</tool-use-id>"
    r"(?:.*?<status>(?P<status>[^<]*)</status>)?", re.S)


# Claude Code moves its prompt-cache breakpoint forward as a conversation
# grows, so a block that carried `cache_control` on one turn has lost it by
# the next. That is a caching annotation, not conversation content: left in,
# it makes the two trailing messages of every parent mismatch its child and
# breaks the prefix match that reconstructs the thread. Stripped, the match
# is exact.
def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(_strip(value), sort_keys=True,
                   default=str).encode()).hexdigest()[:16]


def _result_text(content) -> str:
    """tool_result content is a string on some clients, a block list on others."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text") or "")
            elif isinstance(block, str):
                out.append(block)
        return "\n".join(out)
    return "" if content is None else str(content)


def distil(path: Path) -> dict | None:
    try:
        with gzip.open(path) as handle:
            cap = json.load(handle)
    except (OSError, ValueError):
        return None
    body = cap.get("body") or {}
    messages = body.get("messages") or []

    uses, results, notes = [], [], []
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and "<task-notification>" in (block.get("text") or ""):
                for m in NOTIFY.finditer(block["text"]):
                    notes.append({"i": index, "task_id": m.group("task"),
                                  "id": m.group("use"),
                                  "status": m.group("status")})
            if kind == "tool_use":
                name = block.get("name") or ""
                raw = block.get("input")
                raw = raw if isinstance(raw, dict) else {}
                keep = {k: raw.get(k) for k in KEEP_INPUT.get(name, ())
                        if raw.get(k) is not None}
                uses.append({
                    "i": index, "id": block.get("id"), "name": name,
                    "in": keep,
                    "in_bytes": len(json.dumps(raw, default=str)),
                })
            elif kind == "tool_result":
                text = _result_text(block.get("content"))
                results.append({
                    "i": index, "id": block.get("tool_use_id"),
                    "err": bool(block.get("is_error")),
                    "chars": len(text), "head": text[:RESULT_HEAD],
                })

    return {
        "audit_request_id": cap.get("audit_request_id"),
        "client_session_id": cap.get("client_session_id"),
        "captured_at": cap.get("captured_at"),
        # thread reconstruction inputs
        "msgs": [_digest(m) for m in messages],
        "system": _digest(body.get("system")),
        "system_msgs": sum(1 for m in messages if m.get("role") == "system"),
        "n_tools": len(body.get("tools") or []),
        "n_msgs": len(messages),
        # tool inventory
        "uses": uses,
        "results": results,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=min(48, mp.cpu_count()))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, Path]] = []
    for attempt in args.attempt_dir:
        run = attempt.parent.name
        root = attempt / "anthropic_message_capture" / "requests"
        files += [(run, p) for p in sorted(root.glob("*.json.gz"))]
    print(f"{len(files):,} capture files from {len(args.attempt_dir)} attempt dir(s)")

    dest = args.out / "capture_facts.jsonl"
    written = failed = 0
    with mp.Pool(args.jobs) as pool, dest.open("w", encoding="utf-8") as handle:
        for (run, _), record in zip(
                files, pool.imap(distil, [p for _, p in files], chunksize=64)):
            if record is None:
                failed += 1
                continue
            record["run"] = run
            handle.write(json.dumps(record) + "\n")
            written += 1
    print(f"wrote {written:,} records to {dest} ({failed} unreadable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
