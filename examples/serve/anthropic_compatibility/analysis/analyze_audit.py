#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a per-turn report from a content-free Anthropic audit JSONL file.

This analyzer intentionally uses only ``TRTLLM_ANTHROPIC_AUDIT_LOG``. With
benchmark LCP tracking enabled on the server, it reports adjacent-input reuse
without persisting prompt bodies or token IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 6)


def load_audit_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: audit record must be a JSON object")
            records.append(record)
    return records


def _session_key(record: dict[str, Any]) -> str:
    session_id = record.get("client_session_id")
    return str(session_id) if session_id else "unidentified-session"


def _record_sort_key(record: dict[str, Any]) -> tuple[float, str]:
    parsed = _parse_timestamp(record.get("started_at"))
    return (
        parsed.timestamp() if parsed is not None else float("inf"),
        str(record.get("audit_request_id") or ""),
    )


def _turn_metrics(
    record: dict[str, Any],
    session_id: str,
    turn_index: int,
    global_request_index: int,
) -> dict[str, Any]:
    usage = record.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_creation = usage.get("cache_creation_input_tokens") or 0
    output_tokens = usage.get("output_tokens")
    isl_total = None
    isl_computed = None
    if input_tokens is not None:
        isl_total = input_tokens + cache_read + cache_creation
        isl_computed = isl_total - cache_read
    prompt_lcp_tokens = record.get("prompt_lcp_tokens")
    lcp_prompt_tokens = record.get("lcp_prompt_tokens")

    response = record.get("response") or {}
    emitted_calls = response.get("tool_calls_emitted") or []
    returned_results = record.get("tool_results_in_last_message") or []
    return {
        "global_request_index": global_request_index,
        "session_id": session_id,
        "turn_index": turn_index,
        "session_turn_index": turn_index,
        "client_session_source": record.get("client_session_source"),
        "audit_request_id": record.get("audit_request_id"),
        "anthropic_message_id": record.get("anthropic_message_id"),
        "openai_response_id": record.get("openai_response_id"),
        "engine_request_id": record.get("engine_request_id"),
        "disagg_request_id": record.get("disagg_request_id"),
        "ctx_request_id": record.get("ctx_request_id"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "status": record.get("status"),
        "error": record.get("error"),
        "server_ttft_ms": record.get("server_ttft_ms"),
        "server_total_ms": record.get("duration_ms"),
        "isl_total": isl_total,
        "isl_cached": cache_read if input_tokens is not None else None,
        "isl_computed": isl_computed,
        "osl": output_tokens,
        "actual_cache_hit_ratio": _ratio(cache_read, isl_total),
        "lcp_prompt_tokens": lcp_prompt_tokens,
        "lcp_matches_reported_isl": (
            lcp_prompt_tokens == isl_total
            if lcp_prompt_tokens is not None and isl_total is not None
            else None
        ),
        "previous_prompt_tokens": record.get("previous_prompt_tokens"),
        "prompt_lcp_tokens": prompt_lcp_tokens,
        "previous_prompt_retention_ratio": record.get("previous_prompt_retention_ratio"),
        "current_reuse_opportunity_ratio": record.get("current_reuse_opportunity_ratio"),
        "input_only_cache_realization_ratio": _ratio(cache_read, prompt_lcp_tokens),
        "actual_cached_exceeds_input_lcp": (
            cache_read > prompt_lcp_tokens if prompt_lcp_tokens is not None else None
        ),
        "tool_calls_emitted": len(emitted_calls),
        "tool_call_names": [call.get("name") for call in emitted_calls],
        "tool_call_input_json_bytes": sum(
            call.get("input_json_bytes") or 0 for call in emitted_calls
        ),
        "tool_results_returned": len(returned_results),
        "tool_result_errors": sum(1 for result in returned_results if result.get("is_error")),
        "thinking_chars": response.get("thinking_chars"),
        "visible_text_chars": response.get("text_chars"),
        "history_message_count": record.get("history_message_count"),
        "history_content_block_counts": record.get("history_content_block_counts") or {},
        "tool_definition_count": record.get("tool_definition_count"),
        "message_capture_file": record.get("message_capture_file"),
    }


def analyze_records(
    records: Iterable[dict[str, Any]],
    session_id_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_records = [
        record
        for record in records
        if session_id_filter is None or _session_key(record) == session_id_filter
    ]
    globally_sorted = sorted(selected_records, key=_record_sort_key)
    global_request_indices = {
        id(record): index for index, record in enumerate(globally_sorted, start=1)
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected_records:
        session_id = _session_key(record)
        grouped[session_id].append(record)

    turns: list[dict[str, Any]] = []
    tool_loops: list[dict[str, Any]] = []
    for session_id, session_records in sorted(grouped.items()):
        session_records.sort(key=_record_sort_key)
        pending_calls: dict[str, tuple[int, dict[str, Any], dict[str, Any]]] = {}
        for turn_index, record in enumerate(session_records, start=1):
            turn = _turn_metrics(
                record,
                session_id,
                turn_index,
                global_request_indices[id(record)],
            )
            turns.append(turn)

            # Scan the whole history, not just the last message. Claude Code
            # often appends a system message after a tool result, which hides
            # the result from a last-message scan and leaves the call forever
            # unmatched -- measured at 21-26% of calls. Re-reading history is
            # safe because pending_calls.pop removes an id on first match, so
            # a result can only ever match once, and history is append-only,
            # so that first match is the turn the result actually arrived in.
            # Measured effect: unmatched 20.9% -> 0.0% and 25.7% -> 1.7% on
            # two real traces, recovering the long tail (max gap 59s -> 190s).
            for result in record.get("tool_results_in_request") or []:
                tool_use_id = result.get("tool_use_id")
                pending = pending_calls.pop(tool_use_id, None)
                if pending is None:
                    continue
                response_turn_index, response_record, call = pending
                response_finished = _parse_timestamp(response_record.get("finished_at"))
                result_request_started = _parse_timestamp(record.get("started_at"))
                gap_ms = None
                if response_finished is not None and result_request_started is not None:
                    gap_ms = round(
                        (result_request_started - response_finished).total_seconds() * 1000,
                        3,
                    )
                tool_loops.append(
                    {
                        "session_id": session_id,
                        "tool_use_id": tool_use_id,
                        "tool_name": call.get("name"),
                        "tool_call_input_json_bytes": call.get("input_json_bytes"),
                        "response_turn_index": response_turn_index,
                        "result_turn_index": turn_index,
                        "response_finished_at": response_record.get("finished_at"),
                        "result_request_started_at": record.get("started_at"),
                        "client_tool_roundtrip_gap_ms": gap_ms,
                        "tool_result_is_error": bool(result.get("is_error")),
                        "tool_result_content_chars": result.get("content_chars"),
                        "matched": True,
                    }
                )

            response = record.get("response") or {}
            for call in response.get("tool_calls_emitted") or []:
                tool_use_id = call.get("id")
                if tool_use_id:
                    pending_calls[tool_use_id] = (turn_index, record, call)

        for tool_use_id, (response_turn_index, response_record, call) in pending_calls.items():
            tool_loops.append(
                {
                    "session_id": session_id,
                    "tool_use_id": tool_use_id,
                    "tool_name": call.get("name"),
                    "tool_call_input_json_bytes": call.get("input_json_bytes"),
                    "response_turn_index": response_turn_index,
                    "result_turn_index": None,
                    "response_finished_at": response_record.get("finished_at"),
                    "result_request_started_at": None,
                    "client_tool_roundtrip_gap_ms": None,
                    "tool_result_is_error": None,
                    "tool_result_content_chars": None,
                    "matched": False,
                }
            )

    return turns, tool_loops


TIMELINE_COLUMNS = [
    "global_request_index",
    "session_id",
    "session_turn_index",
    "engine_request_id",
    "audit_request_id",
    "started_at",
    "finished_at",
    "status",
    "error",
    "server_ttft_ms",
    "server_total_ms",
    "server_decode_ms",
    "output_tps_per_user",
    "isl_total",
    "isl_cached",
    "isl_new",
    "osl_model_tokens",
    "actual_cache_hit_ratio",
    "prompt_lcp_tokens",
    "previous_prompt_retention_ratio",
    "current_reuse_opportunity_ratio",
    "cache_realization_ratio",
    "thinking_chars",
    "visible_text_chars",
    "tool_call_count",
    "tool_call_names",
    "tool_call_input_json_bytes",
    "tool_result_turns",
    "tool_result_count",
    "tool_result_content_chars",
    "tool_result_error_count",
    "tool_loop_gap_ms",
    "tool_loop_gap_min_ms",
    "tool_loop_gap_max_ms",
    "tool_loop_matched",
    "tool_loop_details",
    "history_message_count",
    "history_content_block_counts",
    "tool_definition_count",
    "client_session_source",
    "message_capture_file",
]


def _server_decode_metrics(
    osl: int | None,
    server_ttft_ms: float | None,
    server_total_ms: float | None,
) -> tuple[float | None, float | None]:
    """Return post-first-token duration and per-user output throughput."""
    if server_ttft_ms is None or server_total_ms is None:
        return None, None
    decode_ms = server_total_ms - server_ttft_ms
    if decode_ms <= 0:
        return None, None
    if osl is None or osl <= 1:
        return decode_ms, None
    return decode_ms, (osl - 1) * 1000 / decode_ms


def build_timeline_rows(
    turns: Iterable[dict[str, Any]],
    tool_loops: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join per-turn metrics and matching client tool loops into one timeline."""
    loops_by_turn: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for loop in tool_loops:
        key = (loop["session_id"], loop["response_turn_index"])
        loops_by_turn[key].append(loop)

    timeline = []
    for turn in turns:
        loops = loops_by_turn.get(
            (turn["session_id"], turn["session_turn_index"]),
            [],
        )
        gaps = [
            loop["client_tool_roundtrip_gap_ms"]
            for loop in loops
            if loop["client_tool_roundtrip_gap_ms"] is not None
        ]
        unique_gaps = sorted(set(gaps))
        result_turns = sorted(
            {
                loop["result_turn_index"]
                for loop in loops
                if loop["result_turn_index"] is not None
            }
        )
        loop_details = [
            {
                "tool_use_id": loop["tool_use_id"],
                "tool_name": loop["tool_name"],
                "tool_call_input_json_bytes": loop.get("tool_call_input_json_bytes"),
                "result_turn_index": loop["result_turn_index"],
                "client_tool_roundtrip_gap_ms": loop[
                    "client_tool_roundtrip_gap_ms"
                ],
                "tool_result_content_chars": loop["tool_result_content_chars"],
                "tool_result_is_error": loop["tool_result_is_error"],
                "matched": loop["matched"],
            }
            for loop in loops
        ]
        server_decode_ms, output_tps_per_user = _server_decode_metrics(
            turn["osl"],
            turn["server_ttft_ms"],
            turn["server_total_ms"],
        )
        timeline.append(
            {
                "global_request_index": turn["global_request_index"],
                "session_id": turn["session_id"],
                "session_turn_index": turn["session_turn_index"],
                "engine_request_id": turn["engine_request_id"],
                "audit_request_id": turn["audit_request_id"],
                "started_at": turn["started_at"],
                "finished_at": turn["finished_at"],
                "status": turn["status"],
                "error": turn["error"],
                "server_ttft_ms": turn["server_ttft_ms"],
                "server_total_ms": turn["server_total_ms"],
                "server_decode_ms": server_decode_ms,
                "output_tps_per_user": output_tps_per_user,
                "isl_total": turn["isl_total"],
                "isl_cached": turn["isl_cached"],
                "isl_new": turn["isl_computed"],
                "osl_model_tokens": turn["osl"],
                "actual_cache_hit_ratio": turn["actual_cache_hit_ratio"],
                "prompt_lcp_tokens": turn["prompt_lcp_tokens"],
                "previous_prompt_retention_ratio": turn[
                    "previous_prompt_retention_ratio"
                ],
                "current_reuse_opportunity_ratio": turn[
                    "current_reuse_opportunity_ratio"
                ],
                "cache_realization_ratio": turn[
                    "input_only_cache_realization_ratio"
                ],
                "thinking_chars": turn["thinking_chars"],
                "visible_text_chars": turn["visible_text_chars"],
                "tool_call_count": turn["tool_calls_emitted"],
                "tool_call_names": turn["tool_call_names"],
                "tool_call_input_json_bytes": turn["tool_call_input_json_bytes"],
                "tool_result_turns": result_turns,
                "tool_result_count": sum(1 for loop in loops if loop["matched"]),
                "tool_result_content_chars": sum(
                    loop["tool_result_content_chars"] or 0
                    for loop in loops
                    if loop["matched"]
                ),
                "tool_result_error_count": sum(
                    1 for loop in loops if loop["tool_result_is_error"]
                ),
                "tool_loop_gap_ms": unique_gaps[0] if len(unique_gaps) == 1 else None,
                "tool_loop_gap_min_ms": min(gaps) if gaps else None,
                "tool_loop_gap_max_ms": max(gaps) if gaps else None,
                "tool_loop_matched": (
                    all(loop["matched"] for loop in loops) if loops else None
                ),
                "tool_loop_details": loop_details,
                "history_message_count": turn["history_message_count"],
                "history_content_block_counts": turn[
                    "history_content_block_counts"
                ],
                "tool_definition_count": turn["tool_definition_count"],
                "client_session_source": turn["client_session_source"],
                "message_capture_file": turn["message_capture_file"],
            }
        )
    return sorted(timeline, key=lambda row: row["global_request_index"])


def _format_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _markdown_cell(value: Any) -> str:
    if value is None or value == []:
        return "-"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|")


def render_timeline_markdown(timeline: list[dict[str, Any]]) -> str:
    lines = [
        "# Anthropic Request Timeline",
        "",
        "One row is one `/v1/messages` request. `Session turn` is local to a ",
        "Claude Code session; `Global` is the server-observed order across all ",
        "selected sessions. Tool-loop metrics are attached to the turn that ",
        "emitted the tool call.",
        "",
        "| Global | Session | Session turn | Started at | Status | TTFT ms | "
        "Total ms | Decode ms | TPS/user | ISL | Cached | New | Model OSL | LCP | Actual hit | "
        "Prev retained | Reusable | Realization | Tools | Result chars | "
        "Tool gap ms | Tool errors |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in timeline:
        tool_names = ", ".join(name for name in row["tool_call_names"] if name) or "-"
        lines.append(
            "| {global_index} | {session} | {session_turn} | {started_at} | "
            "{status} | {ttft} | {total} | {decode} | {tps} | "
            "{isl} | {cached} | {new} | {osl} | "
            "{lcp} | {hit} | {retained} | {reusable} | {realization} | "
            "{tools} | {result_chars} | {gap} | {errors} |".format(
                global_index=row["global_request_index"],
                session=_markdown_cell(row["session_id"]),
                session_turn=row["session_turn_index"],
                started_at=_markdown_cell(row["started_at"]),
                status=_markdown_cell(row["status"]),
                ttft=_format_number(row["server_ttft_ms"]),
                total=_format_number(row["server_total_ms"]),
                decode=_format_number(row["server_decode_ms"]),
                tps=_format_number(row["output_tps_per_user"], 2),
                isl=_format_number(row["isl_total"], 0),
                cached=_format_number(row["isl_cached"], 0),
                new=_format_number(row["isl_new"], 0),
                osl=_format_number(row["osl_model_tokens"], 0),
                lcp=_format_number(row["prompt_lcp_tokens"], 0),
                hit=_format_percent(row["actual_cache_hit_ratio"]),
                retained=_format_percent(row["previous_prompt_retention_ratio"]),
                reusable=_format_percent(row["current_reuse_opportunity_ratio"]),
                realization=_format_percent(row["cache_realization_ratio"]),
                tools=_markdown_cell(tool_names),
                result_chars=_format_number(row["tool_result_content_chars"], 0),
                gap=_format_number(row["tool_loop_gap_ms"]),
                errors=row["tool_result_error_count"],
            )
        )
    lines.extend(
        [
            "",
            "`Model OSL` contains only model-generated output tokens. Tool-result ",
            "payloads are client generated and are represented as characters ",
            "because the content-free audit does not tokenize each result block ",
            "separately. `TPS/user` is `(Model OSL - 1) / Decode seconds`, where ",
            "`Decode ms = server total ms - server TTFT ms`; it excludes the ",
            "client tool-loop gap and is not Claude CLI end-to-end throughput. ",
            "See `timeline.csv` for all identifiers, context-shape ",
            "fields, and per-tool JSON details.",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown(turns: list[dict[str, Any]], tool_loops: list[dict[str, Any]]) -> str:
    lines = [
        "# Anthropic Audit Report",
        "",
        "This report uses the content-free server audit. It reports actual cache ",
        "usage, adjacent-input reuse opportunity, and coarse server-observed ",
        "tool-loop gaps; it does not report Claude CLI-visible timing.",
        "",
        "## Turns",
        "",
        "| Session | Turn | Status | TTFT ms | Total ms | ISL | Cached ISL | OSL | "
        "Actual hit | LCP | Prev retained | Current reusable | Cache realization | "
        "Tools | Tool errors |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for turn in turns:
        tool_names = ", ".join(name for name in turn["tool_call_names"] if name) or "-"
        lines.append(
            "| {session} | {turn} | {status} | {ttft} | {total} | {isl} | "
            "{cached} | {osl} | {hit} | {lcp} | {retained} | {reusable} | "
            "{realization} | {tools} | {errors} |".format(
                session=turn["session_id"],
                turn=turn["turn_index"],
                status=turn["status"] or "-",
                ttft=_format_number(turn["server_ttft_ms"]),
                total=_format_number(turn["server_total_ms"]),
                isl=_format_number(turn["isl_total"], 0),
                cached=_format_number(turn["isl_cached"], 0),
                osl=_format_number(turn["osl"], 0),
                hit=_format_percent(turn["actual_cache_hit_ratio"]),
                lcp=_format_number(turn["prompt_lcp_tokens"], 0),
                retained=_format_percent(turn["previous_prompt_retention_ratio"]),
                reusable=_format_percent(turn["current_reuse_opportunity_ratio"]),
                realization=_format_percent(turn["input_only_cache_realization_ratio"]),
                tools=tool_names.replace("|", "\\|"),
                errors=turn["tool_result_errors"],
            )
        )

    lines.extend(
        [
            "",
            "## Tool Loops",
            "",
            "| Session | Tool-use turn | Result turn | Tool | Result error | Coarse gap ms | Matched |",
            "| --- | ---: | ---: | --- | --- | ---: | --- |",
        ]
    )
    for loop in tool_loops:
        lines.append(
            "| {session} | {response_turn} | {result_turn} | {tool} | "
            "{error} | {gap} | {matched} |".format(
                session=loop["session_id"],
                response_turn=loop["response_turn_index"],
                result_turn=_format_number(loop["result_turn_index"], 0),
                tool=(loop["tool_name"] or "-").replace("|", "\\|"),
                error=_format_number(loop["tool_result_is_error"], 0),
                gap=_format_number(loop["client_tool_roundtrip_gap_ms"]),
                matched=str(loop["matched"]).lower(),
            )
        )
    lines.extend(["", "## Validation Flags", ""])
    flags = []
    for turn in turns:
        turn_label = f"{turn['session_id']} turn {turn['turn_index']}"
        if turn["lcp_matches_reported_isl"] is False:
            flags.append(f"- {turn_label}: LCP prompt length does not match reported ISL.")
        if turn["actual_cached_exceeds_input_lcp"]:
            flags.append(f"- {turn_label}: actual cached ISL exceeds adjacent-input LCP.")
    lines.extend(flags or ["No LCP consistency flags."])
    lines.append("")
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_timeline_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMELINE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: _csv_value(row.get(column)) for column in TIMELINE_COLUMNS}
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_log", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: <audit-log-stem>-analysis)",
    )
    parser.add_argument(
        "--session-id",
        help="Analyze one client_session_id instead of every session",
    )
    args = parser.parse_args()

    output_dir = args.out or args.audit_log.with_name(f"{args.audit_log.stem}-analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    turns, tool_loops = analyze_records(load_audit_records(args.audit_log), args.session_id)
    timeline = build_timeline_rows(turns, tool_loops)
    _write_jsonl(output_dir / "turns.jsonl", turns)
    _write_jsonl(output_dir / "tool_loops.jsonl", tool_loops)
    _write_timeline_csv(output_dir / "timeline.csv", timeline)
    (output_dir / "REPORT.md").write_text(render_markdown(turns, tool_loops), encoding="utf-8")
    (output_dir / "TIMELINE.md").write_text(
        render_timeline_markdown(timeline),
        encoding="utf-8",
    )
    print(
        f"wrote {len(timeline)} timeline rows and {len(tool_loops)} tool loops "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
