# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapter between the Anthropic Messages API and the OpenAI chat pipeline.

Request direction: :class:`AnthropicMessagesRequest` is translated into a
:class:`ChatCompletionRequest` so the existing ``openai_chat`` path (chat
template, tool parser, reasoning parser, post-processing) is reused verbatim.
Response direction: the resulting :class:`ChatCompletionResponse` is
translated back into an :class:`AnthropicMessagesResponse`.

The adapter is a pure protocol layer: it never touches the tokenizer,
chat templates, or the engine.
"""

import asyncio
import gzip
import inspect
import json
import os
import queue
import re
import threading
import traceback
import uuid
from array import array
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Union

from fastapi.responses import JSONResponse

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.anthropic_protocol import (
    AnthropicContentBlockDeltaEvent,
    AnthropicContentBlockStartEvent,
    AnthropicContentBlockStopEvent,
    AnthropicCountTokensRequest,
    AnthropicError,
    AnthropicErrorEvent,
    AnthropicErrorResponse,
    AnthropicErrorType,
    AnthropicInputJsonDelta,
    AnthropicMessageDelta,
    AnthropicMessageDeltaEvent,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicMessageStartEvent,
    AnthropicMessageStopEvent,
    AnthropicStopReason,
    AnthropicTextBlock,
    AnthropicTextDelta,
    AnthropicThinkingBlock,
    AnthropicThinkingDelta,
    AnthropicToolUseBlock,
    AnthropicUsage,
    anthropic_sse,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
    FunctionDefinition,
    UsageInfo,
)

# OpenAI finish_reason -> Anthropic stop_reason. A string-valued OpenAI
# stop_reason identifies the matched stop sequence and is handled separately.
STOP_REASON_MAP: Dict[str, AnthropicStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}

ANTHROPIC_AUDIT_LOG_ENV = "TRTLLM_ANTHROPIC_AUDIT_LOG"
ANTHROPIC_LCP_TRACKING_ENV = "TRTLLM_ANTHROPIC_LCP_TRACKING"
ANTHROPIC_BENCH_CAPTURE_DIR_ENV = "TRTLLM_ANTHROPIC_BENCH_CAPTURE_DIR"
_ANTHROPIC_BILLING_MARKER = "x-anthropic-billing-header:"
# Real blocks top out at 94 characters; this only bounds the worst case.
_ANTHROPIC_BILLING_MAX_CHARS = 512
# Claude Code's field set is not stable across client versions: cch= was
# dropped, cc_is_subagent= was added, and cc_entrypoint gained sdk-py. Pinning
# an exact field list silently disabled stripping once already, so match the
# cc-prefixed key=value shape instead. What bounds the false-strip risk is the
# marker plus the system[0]-only restriction in _system_text_parts, not a field
# whitelist -- the marker never appears outside system[0] in practice.
_ANTHROPIC_BILLING_SYSTEM_BLOCK = re.compile(
    r"x-anthropic-billing-header:\s*"
    r"(?:cc[0-9a-z_]*=[0-9A-Za-z._-]*;\s*)+"
)
_billing_shape_warned = False
_SESSION_ID_HEADERS = (
    "x-claude-session-id",
    "x-claude-code-session-id",
    "x-session-id",
)
_ANTHROPIC_CAPTURE_QUEUE: queue.Queue[tuple[Path, dict[str, Any]]] = (
    queue.Queue(maxsize=16)
)
_ANTHROPIC_CAPTURE_WRITER_LOCK = threading.Lock()
_ANTHROPIC_CAPTURE_WRITER: threading.Thread | None = None


def anthropic_lcp_tracking_enabled() -> bool:
    """Return whether benchmark-only adjacent-prompt tracking is enabled."""
    return os.environ.get(ANTHROPIC_LCP_TRACKING_ENV, "0") == "1"


def _write_anthropic_message_capture(
    path: Path, capture: dict[str, Any]
) -> None:
    """Write one sensitive benchmark capture with restrictive permissions."""
    capture_root = path.parent.parent
    capture_root.mkdir(parents=True, exist_ok=True)
    capture_root.chmod(0o700)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)

    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    file_descriptor = os.open(
        temporary_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as raw_file:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_file,
                mtime=0,
            ) as capture_file:
                capture_file.write(
                    json.dumps(
                        capture,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
        os.replace(temporary_path, path)
        path.chmod(0o600)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _anthropic_capture_writer_main() -> None:
    while True:
        path, capture = _ANTHROPIC_CAPTURE_QUEUE.get()
        try:
            _write_anthropic_message_capture(path, capture)
        except Exception as error:  # noqa: BLE001 - capture must not break serving
            logger.warning(
                "Failed to write sensitive Anthropic message capture to %s: %s",
                path,
                error,
            )
        finally:
            _ANTHROPIC_CAPTURE_QUEUE.task_done()


def _ensure_anthropic_capture_writer() -> None:
    global _ANTHROPIC_CAPTURE_WRITER
    with _ANTHROPIC_CAPTURE_WRITER_LOCK:
        if (
            _ANTHROPIC_CAPTURE_WRITER is not None
            and _ANTHROPIC_CAPTURE_WRITER.is_alive()
        ):
            return
        _ANTHROPIC_CAPTURE_WRITER = threading.Thread(
            target=_anthropic_capture_writer_main,
            name="anthropic-message-capture-writer",
            daemon=True,
        )
        _ANTHROPIC_CAPTURE_WRITER.start()


def flush_anthropic_message_captures() -> None:
    """Wait until all queued captures are durable (primarily for tests)."""
    _ANTHROPIC_CAPTURE_QUEUE.join()


async def capture_anthropic_message_request(
    audit_record: dict[str, Any], raw_request: Any
) -> None:
    """Queue an opt-in full request capture without doing disk I/O inline."""
    capture_dir_value = os.environ.get(ANTHROPIC_BENCH_CAPTURE_DIR_ENV)
    if not capture_dir_value:
        return

    try:
        raw_body = await raw_request.body()
        body = json.loads(raw_body)
        raw_headers = raw_request.scope.get("headers", ())
        headers = [
            [
                name.decode("latin-1"),
                value.decode("latin-1"),
            ]
            for name, value in raw_headers
        ]
        relative_path = Path("requests") / (
            f"{audit_record['audit_request_id']}.json.gz"
        )
        capture_path = Path(capture_dir_value) / relative_path
        capture = {
            "event": "anthropic_message_request_capture",
            "captured_at": _utc_timestamp(),
            "audit_request_id": audit_record["audit_request_id"],
            "client_session_id": audit_record.get("client_session_id"),
            "method": raw_request.method,
            "path": raw_request.url.path,
            "headers": headers,
            "body": body,
        }
        _ensure_anthropic_capture_writer()
        _ANTHROPIC_CAPTURE_QUEUE.put_nowait((capture_path, capture))
        audit_record["message_capture_file"] = str(relative_path)
    except queue.Full:
        audit_record["message_capture_error"] = "writer_queue_full"
        logger.warning(
            "Dropped sensitive Anthropic message capture request_id=%s: "
            "writer queue is full",
            audit_record["audit_request_id"],
        )
    except Exception as error:  # noqa: BLE001 - capture must not break serving
        audit_record["message_capture_error"] = type(error).__name__
        logger.warning(
            "Failed to queue sensitive Anthropic message capture request_id=%s: %s",
            audit_record["audit_request_id"],
            error,
        )


class AnthropicPromptLcpTracker:
    """Keep one prior prompt per session and return content-free LCP metrics."""

    def __init__(self, max_sessions: int = 16) -> None:
        self._max_sessions = max_sessions
        self._previous_prompts: OrderedDict[str, array] = OrderedDict()
        self._lock = threading.Lock()

    def observe(self, session_id: str | None,
                token_ids: Iterable[int]) -> dict[str, Any]:
        session_key = session_id or "unidentified-session"
        current_prompt = array("I", token_ids)
        with self._lock:
            previous_prompt = self._previous_prompts.get(session_key)
            lcp_tokens = None
            previous_tokens = None
            if previous_prompt is not None:
                previous_tokens = len(previous_prompt)
                lcp_tokens = 0
                for previous_token, current_token in zip(
                        previous_prompt, current_prompt):
                    if previous_token != current_token:
                        break
                    lcp_tokens += 1
            self._previous_prompts[session_key] = current_prompt
            self._previous_prompts.move_to_end(session_key)
            while len(self._previous_prompts) > self._max_sessions:
                self._previous_prompts.popitem(last=False)

        current_tokens = len(current_prompt)
        previous_retention = None
        current_opportunity = None
        if lcp_tokens is not None:
            if previous_tokens:
                previous_retention = round(lcp_tokens / previous_tokens, 6)
            if current_tokens:
                current_opportunity = round(lcp_tokens / current_tokens, 6)
        return {
            "lcp_session_id": session_key,
            "lcp_prompt_tokens": current_tokens,
            "previous_prompt_tokens": previous_tokens,
            "prompt_lcp_tokens": lcp_tokens,
            "previous_prompt_retention_ratio": previous_retention,
            "current_reuse_opportunity_ratio": current_opportunity,
        }


def schedule_anthropic_lcp_observation(
    tracker: AnthropicPromptLcpTracker,
    record: dict[str, Any],
    token_ids: Iterable[int] | None,
) -> asyncio.Task | None:
    """Calculate LCP off the request event loop when benchmark tracking is on."""
    if not anthropic_lcp_tracking_enabled() or token_ids is None:
        return None
    return asyncio.create_task(
        asyncio.to_thread(
            tracker.observe,
            record.get("client_session_id"),
            token_ids,
        )
    )


async def collect_anthropic_lcp_observation(
    record: dict[str, Any], task: asyncio.Task | None
) -> None:
    """Attach completed content-free LCP metrics to an audit record."""
    if task is None:
        return
    try:
        record.update(await task)
    except Exception as error:  # noqa: BLE001 - metrics must not break a response
        record["lcp_tracking_error"] = type(error).__name__
        logger.warning("Anthropic LCP tracking failed: %s", error)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_result_content_chars(content: Any) -> int:
    """Return text characters in a tool result without retaining its content."""
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    return sum(
        len(block.text)
        for block in content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    )


def _tool_results_in_message(message: Any) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return []
    results = []
    for block in message.content:
        if getattr(block, "type", None) != "tool_result":
            continue
        results.append(
            {
                "tool_use_id": block.tool_use_id,
                "is_error": bool(block.is_error),
                "content_chars": _tool_result_content_chars(block.content),
            }
        )
    return results


def create_anthropic_audit_record(
    request: AnthropicMessagesRequest, headers: Any
) -> dict[str, Any]:
    """Create a content-free audit record for one Messages API request.

    The request body is intentionally summarized rather than persisted.  This
    preserves request/response and tool-loop correlation without writing user
    prompts, tool arguments, or tool output to disk.
    """
    session_id = None
    session_source = None
    for header_name in _SESSION_ID_HEADERS:
        value = headers.get(header_name)
        if value:
            session_id = value
            session_source = f"header:{header_name}"
            break

    block_counts: Counter[str] = Counter()
    tool_results = []
    for message in request.messages:
        if not isinstance(message.content, str):
            block_counts.update(block.type for block in message.content)
        tool_results.extend(_tool_results_in_message(message))
    last_message_results = (
        _tool_results_in_message(request.messages[-1]) if request.messages else []
    )
    metadata = request.metadata or {}
    return {
        "event": "anthropic_message_audit",
        "started_at": _utc_timestamp(),
        "_started_at_monotonic": perf_counter(),
        "audit_request_id": f"anthropic-audit-{uuid.uuid4().hex}",
        "client_session_id": session_id,
        "client_session_source": session_source,
        "metadata_user_id_present": metadata.get("user_id") is not None,
        "model": request.model,
        "stream": bool(request.stream),
        "max_tokens": request.max_tokens,
        "history_message_count": len(request.messages),
        "history_content_block_counts": dict(block_counts),
        "tool_results_in_request": tool_results,
        "tool_results_in_last_message": last_message_results,
        "tool_definition_count": len(request.tools or []),
    }


def _usage_audit_summary(usage: AnthropicUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        "output_tokens": usage.output_tokens,
    }


def _response_content_audit_summary(content: List[Any]) -> dict[str, Any]:
    text_chars = 0
    thinking_chars = 0
    tool_calls = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_chars += len(block.text)
        elif block_type == "thinking":
            thinking_chars += len(block.thinking)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "input_json_bytes": len(
                        json.dumps(block.input, separators=(",", ":")).encode("utf-8")
                    ),
                }
            )
    return {
        "text_chars": text_chars,
        "thinking_chars": thinking_chars,
        "tool_calls_emitted": tool_calls,
    }


def finish_anthropic_audit_record(
    record: dict[str, Any],
    *,
    usage: AnthropicUsage | None = None,
    response_id: str | None = None,
    openai_response_id: str | None = None,
    engine_request_id: str | None = None,
    disagg_request_id: str | int | None = None,
    ctx_request_id: str | int | None = None,
    server_ttft_ms: float | None = None,
    response_content: List[Any] | None = None,
    response_summary: dict[str, Any] | None = None,
    status: str = "completed",
    error: str | None = None,
) -> None:
    """Finalize and persist a Messages API audit record when enabled."""
    started_at = record.pop("_started_at_monotonic", None)
    record["finished_at"] = _utc_timestamp()
    if started_at is not None:
        record["duration_ms"] = round((perf_counter() - started_at) * 1000, 3)
    record["status"] = status
    record["error"] = error
    record["anthropic_message_id"] = response_id
    record["openai_response_id"] = openai_response_id
    record["engine_request_id"] = engine_request_id
    record["disagg_request_id"] = disagg_request_id
    record["ctx_request_id"] = ctx_request_id
    record["server_ttft_ms"] = server_ttft_ms
    if usage is not None:
        record["usage"] = _usage_audit_summary(usage)
    if response_summary is not None:
        record["response"] = response_summary
    elif response_content is not None:
        record["response"] = _response_content_audit_summary(response_content)
    _write_anthropic_audit_record(record)


def _write_anthropic_audit_record(record: dict[str, Any]) -> None:
    """Append one audit record when ``TRTLLM_ANTHROPIC_AUDIT_LOG`` is set."""
    path_value = os.environ.get(ANTHROPIC_AUDIT_LOG_ENV)
    if not path_value:
        return
    try:
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            audit_file.write("\n")
        logger.info(
            "Anthropic audit record request_id=%s engine_request_id=%s "
            "disagg_request_id=%s status=%s log=%s",
            record["audit_request_id"],
            record.get("engine_request_id"),
            record.get("disagg_request_id"),
            record["status"],
            path,
        )
    except OSError as error:
        logger.warning("Failed to write Anthropic audit record to %s: %s", path_value, error)


class AnthropicRequestError(ValueError):
    """Invalid Anthropic request; maps to a 400 with an Anthropic envelope."""


class AnthropicResponseError(ValueError):
    """Invalid upstream model response; maps to an Anthropic API error."""


def anthropic_error_response(
    message: str, error_type: AnthropicErrorType = "api_error", status_code: int = 500
) -> JSONResponse:
    envelope = AnthropicErrorResponse(error=AnthropicError(type=error_type, message=message))
    return JSONResponse(content=envelope.model_dump(exclude_none=True), status_code=status_code)


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> ChatCompletionRequest
# ---------------------------------------------------------------------------


def _is_anthropic_billing_system_block(text: str) -> bool:
    """Recognize Claude Code's model-irrelevant per-request billing block."""
    if not text.startswith(_ANTHROPIC_BILLING_MARKER):
        return False
    if (
        len(text) <= _ANTHROPIC_BILLING_MAX_CHARS
        and _ANTHROPIC_BILLING_SYSTEM_BLOCK.fullmatch(text) is not None
    ):
        return True

    # The marker is there but the shape is not one we know: the client changed
    # format again. Warn once so the next drift fails loudly rather than
    # quietly forwarding the block to the model.
    global _billing_shape_warned
    if not _billing_shape_warned:
        _billing_shape_warned = True
        logger.warning(
            "Anthropic billing system block has an unrecognized shape and will "
            "be sent to the model verbatim; the strip pattern needs updating. "
            "Block starts with: %r",
            text[:120],
        )
    return False


def _system_text_parts(system: Optional[Union[str, List[Any]]]) -> List[str]:
    if system is None:
        return []
    if isinstance(system, str):
        return [system] if system else []
    parts = []
    for index, block in enumerate(system):
        text = getattr(block, "text", None)
        # Claude Code puts this dynamic block at system[0]. The raw request is
        # captured before conversion; exclude it only from model-visible input.
        if index == 0 and text and _is_anthropic_billing_system_block(text):
            continue
        if text:
            parts.append(text)
    return parts


def _image_part(block: Any) -> Optional[Dict[str, Any]]:
    source = block.source
    if source.type == "url" and source.url:
        return {"type": "image_url", "image_url": {"url": source.url}}
    if source.type == "base64" and source.data:
        media_type = source.media_type or "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{source.data}"},
        }
    logger.warning("Anthropic image block with empty source dropped")
    return None


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result content payload into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
        else:
            raise AnthropicRequestError(
                f"Content block type {block_type!r} inside tool_result is not supported"
            )
    return "\n".join(parts)


def _convert_messages(request: AnthropicMessagesRequest) -> List[Dict[str, Any]]:
    system_parts = _system_text_parts(request.system)
    converted: List[Dict[str, Any]] = []

    # Only system messages that precede the first real turn belong to the opening
    # prompt. Anthropic clients also append transient system messages (task
    # reminders, background-task notifications) at the *tail* of the history;
    # hoisting those into the opening block rewrites the very front of the prompt
    # and invalidates the whole KV prefix -- tens of thousands of unchanged tokens
    # get re-prefilled every time one arrives. They stay where the client put them.
    in_leading_system_run = True

    for message in request.messages:
        if message.role == "system" and in_leading_system_run:
            if isinstance(message.content, str):
                system_parts.append(message.content)
            else:
                system_parts.extend(
                    block.text
                    for block in message.content
                    if getattr(block, "type", None) == "text"
                )
            continue

        in_leading_system_run = False

        if isinstance(message.content, str):
            converted.append({"role": message.role, "content": message.content})
            continue

        # Content parts accumulated for the current role; flushed before any
        # role:"tool" message so ordering user(pre) -> tool -> user(post) is
        # preserved.
        parts: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []

        def flush_parts():
            if parts:
                converted_message: Dict[str, Any] = {
                    "role": message.role,
                    "content": list(parts),
                }
                if message.role == "assistant" and reasoning_parts:
                    converted_message["reasoning"] = "".join(reasoning_parts)
                    reasoning_parts.clear()
                converted.append(converted_message)
                parts.clear()

        for block in message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append({"type": "text", "text": block.text})
            elif block_type == "image":
                image_part = _image_part(block)
                if image_part is not None:
                    parts.append(image_part)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )
            elif block_type == "tool_result":
                flush_parts()
                tool_result_text = _tool_result_text(block.content)
                if block.is_error:
                    tool_result_text = f"Tool execution failed: {tool_result_text}"
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": tool_result_text,
                    }
                )
            elif block_type == "thinking":
                if message.role != "assistant":
                    raise AnthropicRequestError(
                        "thinking content blocks are only valid in assistant messages"
                    )
                reasoning_parts.append(block.thinking)
            elif block_type == "redacted_thinking":
                raise AnthropicRequestError(
                    "redacted_thinking history is not supported by this server"
                )
            else:
                logger.warning("Unsupported Anthropic content block %r skipped", block_type)

        if message.role == "assistant" and tool_calls:
            text_content = "".join(p["text"] for p in parts if p.get("type") == "text")
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": tool_calls,
            }
            if reasoning_parts:
                assistant_message["reasoning"] = "".join(reasoning_parts)
                reasoning_parts.clear()
            converted.append(assistant_message)
            parts.clear()
        else:
            flush_parts()
            if message.role == "assistant" and reasoning_parts:
                converted.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "".join(reasoning_parts),
                    }
                )

    if system_parts:
        converted.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return converted


def _convert_tools(request: AnthropicMessagesRequest) -> Optional[List[ChatCompletionToolsParam]]:
    if not request.tools:
        return None
    if request.tool_choice is not None and request.tool_choice.type == "none":
        return None
    tools = []
    for tool in request.tools:
        if tool.is_server_tool():
            raise AnthropicRequestError(
                f"Anthropic server tool {tool.name!r} (type={tool.type!r}) "
                "is not supported by this server"
            )
        if tool.input_schema is None:
            if tool.is_schema_client_tool():
                raise AnthropicRequestError(
                    f"Anthropic schema client tool {tool.name!r} "
                    f"(type={tool.type!r}) is recognized, but its built-in "
                    "input schema is not implemented by this server"
                )
            raise AnthropicRequestError(
                f"Client tool {tool.name!r} requires input_schema"
            )
        tools.append(
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                    strict=tool.strict,
                )
            )
        )
    return tools or None


def _convert_tool_choice(
    request: AnthropicMessagesRequest,
    tools: Optional[List[ChatCompletionToolsParam]],
) -> Optional[Union[str, ChatCompletionNamedToolChoiceParam]]:
    choice = request.tool_choice
    if choice is None:
        # check_tool_choice validator defaults to "auto" when tools are set.
        return "auto" if tools else None
    if choice.disable_parallel_tool_use and choice.type != "none":
        raise AnthropicRequestError(
            "tool_choice.disable_parallel_tool_use=true is not supported"
        )
    if choice.type == "none":
        return "none"
    if tools is None:
        raise AnthropicRequestError(
            f"tool_choice type {choice.type!r} requires at least one "
            "client-executable tool; all provided tools were server tools "
            "or the tools list was empty"
        )
    if choice.type == "auto":
        return "auto"
    if choice.type == "any":
        raise AnthropicRequestError(
            "Anthropic tool_choice type 'any' is not supported because the "
            "chat pipeline cannot require an arbitrary tool call"
        )
    if choice.type == "tool":
        if not choice.name:
            raise AnthropicRequestError("tool_choice type 'tool' requires a 'name'")
        tool_names = {t.function.name for t in tools}
        if choice.name not in tool_names:
            raise AnthropicRequestError(f"tool_choice names unknown tool {choice.name!r}")
        return ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=choice.name)
        )
    raise AnthropicRequestError(f"Unsupported tool_choice {choice.type!r}")


# Chat-template kwargs that turn off pruning of earlier-turn reasoning. The
# name differs per model family but the meaning is identical, and a template
# that does not know a key simply ignores it, so both are always sent
# together:
#   * `clear_thinking`  - GLM-family Jinja templates
#   * `drop_thinking`   - DeepSeek-V4 (`DeepseekV4Tokenizer.apply_chat_template`)
_KEEP_ALL_THINKING_KWARGS = {"clear_thinking": False, "drop_thinking": False}


def _thinking_retention_kwargs(
    context_management: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Translate Anthropic context-editing directives into chat-template kwargs.

    Clients that enable extended thinking also declare what should happen to
    reasoning from earlier turns, e.g.::

        {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}

    Chat templates default to pruning that reasoning (everything before the
    last user message), so a ``keep: "all"`` directive has to be forwarded or
    the client's stated intent is silently dropped.

    Only ``keep: "all"`` is translated. Any other retention policy is left to
    the template default rather than guessed at, because dropping *more* than
    asked is the safer failure mode than keeping more than asked.
    """
    if not isinstance(context_management, dict):
        return {}

    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return {}

    for edit in edits:
        if not isinstance(edit, dict):
            continue
        edit_type = edit.get("type")
        # The type carries a dated suffix (clear_thinking_20251015); match on
        # the family so a future revision keeps working.
        if not isinstance(edit_type, str) or not edit_type.startswith("clear_thinking"):
            continue
        keep = edit.get("keep")
        if keep == "all":
            return dict(_KEEP_ALL_THINKING_KWARGS)
        logger.warning(
            "Unsupported %r retention policy %r; falling back to the chat "
            "template default (earlier-turn reasoning is pruned).",
            edit_type,
            keep,
        )
    return {}


def convert_anthropic_request(request: AnthropicMessagesRequest) -> ChatCompletionRequest:
    """Translate an Anthropic Messages request into a chat completion request."""
    messages = _convert_messages(request)
    if not messages:
        raise AnthropicRequestError("messages must not be empty")
    tools = _convert_tools(request)
    tool_choice = _convert_tool_choice(request, tools)

    chat_request: Dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_completion_tokens": request.max_tokens,
        "stream": bool(request.stream),
    }
    if tools is not None:
        chat_request["tools"] = [t.model_dump() for t in tools]
    if tool_choice is not None:
        chat_request["tool_choice"] = (
            tool_choice if isinstance(tool_choice, str) else tool_choice.model_dump()
        )
    if request.temperature is not None:
        chat_request["temperature"] = request.temperature
    if request.top_p is not None:
        chat_request["top_p"] = request.top_p
    if request.top_k is not None:
        chat_request["top_k"] = request.top_k
    if request.stop_sequences:
        chat_request["stop"] = list(request.stop_sequences)

    # Anthropic extended-thinking controls need to reach both the tokenizer
    # template and the reasoning postprocessor.  DeepSeek-V4 selects thinking
    # mode from these template kwargs; without them, configuring the V4
    # reasoning parser alone leaves it in identity mode.
    chat_template_kwargs: Dict[str, Any] = {}
    if request.thinking:
        thinking_type = request.thinking.get("type")
        if thinking_type == "enabled":
            chat_template_kwargs["enable_thinking"] = True
            budget_tokens = request.thinking.get("budget_tokens")
            if (
                isinstance(budget_tokens, bool)
                or not isinstance(budget_tokens, int)
                or budget_tokens < 1024
            ):
                raise AnthropicRequestError(
                    "thinking.type='enabled' requires budget_tokens >= 1024"
                )
            if budget_tokens >= request.max_tokens:
                raise AnthropicRequestError(
                    "thinking budget_tokens must be less than max_tokens"
                )
            chat_request["thinking_token_budget"] = budget_tokens
        elif thinking_type == "adaptive":
            if request.thinking.get("budget_tokens") is not None:
                raise AnthropicRequestError(
                    "thinking.type='adaptive' does not accept budget_tokens"
                )
            chat_template_kwargs["enable_thinking"] = True
        elif thinking_type == "disabled":
            if request.thinking.get("budget_tokens") is not None:
                raise AnthropicRequestError(
                    "thinking.type='disabled' does not accept budget_tokens"
                )
            chat_template_kwargs["enable_thinking"] = False
        else:
            raise AnthropicRequestError(f"Unsupported thinking type {thinking_type!r}")

    if request.output_config:
        reasoning_effort = request.output_config.get("effort")
        if isinstance(reasoning_effort, str) and reasoning_effort:
            chat_template_kwargs["reasoning_effort"] = reasoning_effort
        output_format = request.output_config.get("format")
        if output_format is not None:
            if not isinstance(output_format, dict):
                raise AnthropicRequestError("output_config.format must be an object")
            if output_format.get("type") != "json_schema":
                raise AnthropicRequestError(
                    "Only output_config.format type 'json_schema' is supported"
                )
            schema = output_format.get("schema")
            if not isinstance(schema, dict):
                raise AnthropicRequestError(
                    "output_config.format.schema must be an object"
                )
            chat_request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": schema},
            }

    chat_template_kwargs.update(_thinking_retention_kwargs(request.context_management))

    if chat_template_kwargs:
        chat_request["chat_template_kwargs"] = chat_template_kwargs
    if request.stream:
        chat_request["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True,
        }
    return ChatCompletionRequest(**chat_request)


def convert_anthropic_count_tokens_request(
    request: AnthropicCountTokensRequest,
) -> ChatCompletionRequest:
    """Translate count-tokens input through the Messages request converter."""
    max_tokens = 1
    if request.thinking and request.thinking.get("type") == "enabled":
        budget_tokens = request.thinking.get("budget_tokens")
        if isinstance(budget_tokens, int) and not isinstance(budget_tokens, bool):
            max_tokens = budget_tokens + 1
    messages_request = AnthropicMessagesRequest(
        model=request.model,
        messages=request.messages,
        max_tokens=max_tokens,
        system=request.system,
        tools=request.tools,
        tool_choice=request.tool_choice,
        thinking=request.thinking,
        output_config=request.output_config,
        betas=request.betas,
        # Retention changes how much history is rendered, so the count has to
        # be taken under the same policy as the real request.
        context_management=request.context_management,
    )
    return convert_anthropic_request(messages_request)


# ---------------------------------------------------------------------------
# Response conversion: ChatCompletionResponse -> Anthropic
# ---------------------------------------------------------------------------


def map_stop_reason(finish_reason: Optional[str]) -> AnthropicStopReason:
    if finish_reason is None:
        return "end_turn"
    mapped = STOP_REASON_MAP.get(finish_reason)
    if mapped is None:
        logger.warning("Unmapped finish_reason %r defaulted to 'end_turn'", finish_reason)
        return "end_turn"
    return mapped


def _map_stop_result(
    finish_reason: Optional[str], stop_reason: Optional[Union[int, str]]
) -> tuple[AnthropicStopReason, Optional[str]]:
    if finish_reason == "stop" and isinstance(stop_reason, str):
        return "stop_sequence", stop_reason
    return map_stop_reason(finish_reason), None


def convert_usage(usage: Optional[UsageInfo]) -> AnthropicUsage:
    if usage is None:
        return AnthropicUsage()
    cached = 0
    if usage.prompt_tokens_details is not None:
        cached = usage.prompt_tokens_details.cached_tokens or 0
    input_tokens = max(usage.prompt_tokens - cached, 0)
    anthropic_usage = AnthropicUsage(
        input_tokens=input_tokens,
        output_tokens=usage.completion_tokens or 0,
    )
    if cached > 0:
        anthropic_usage.cache_read_input_tokens = cached
    return anthropic_usage


def convert_chat_response(chat_response: ChatCompletionResponse) -> AnthropicMessagesResponse:
    """Translate a non-streaming chat completion into an Anthropic message."""
    content: List[Any] = []
    stop_reason: AnthropicStopReason = "end_turn"
    stop_sequence: Optional[str] = None

    if chat_response.choices:
        choice = chat_response.choices[0]
        message = choice.message
        reasoning = message.reasoning_content or message.reasoning
        if reasoning:
            content.append(AnthropicThinkingBlock(thinking=reasoning))
        if message.content:
            content.append(AnthropicTextBlock(text=message.content))
        for tool_call in message.tool_calls:
            try:
                tool_input = json.loads(tool_call.function.arguments)
                if not isinstance(tool_input, dict):
                    raise ValueError("arguments is not a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                raise AnthropicResponseError(
                    f"Tool call {tool_call.function.name!r} arguments are not "
                    "a valid JSON object"
                ) from e
            content.append(
                AnthropicToolUseBlock(
                    id=tool_call.id, name=tool_call.function.name, input=tool_input
                )
            )
        stop_reason, stop_sequence = _map_stop_result(
            choice.finish_reason, choice.stop_reason
        )

    if not content:
        # Anthropic responses must carry at least one content block.
        content.append(AnthropicTextBlock(text=""))

    return AnthropicMessagesResponse(
        model=chat_response.model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=convert_usage(chat_response.usage),
    )


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE chunks -> Anthropic SSE events
# ---------------------------------------------------------------------------


class AnthropicStreamReframer:
    """Stateful reframer from OpenAI chat chunks to Anthropic SSE events.

    Consumes the ``data: <ChatCompletionStreamResponse json>`` lines produced
    by the ``openai_chat`` streaming path and emits the Anthropic event
    sequence::

        message_start
        (content_block_start (content_block_delta)* content_block_stop)*
        message_delta
        message_stop

    Invariants maintained: ``message_start`` is emitted exactly once and
    first; every content block is opened before any delta and closed before a
    block of another type (or another tool call) is opened; block indices are
    monotonically increasing; the delta type always matches the open block
    type.
    """

    def __init__(self, model: str, request_started_at_monotonic: float | None = None):
        self.model = model
        self.message_id = f"msg_{uuid.uuid4().hex}"
        self.openai_response_id: Optional[str] = None
        self.request_started_at_monotonic = request_started_at_monotonic
        self.server_ttft_ms: Optional[float] = None
        self.message_started = False
        self.block_index = -1
        self.open_block_type: Optional[str] = None
        self.open_tool_index: Optional[int] = None
        self.stop_reason: AnthropicStopReason = "end_turn"
        self.stop_sequence: Optional[str] = None
        self.final_usage: Optional[AnthropicUsage] = None
        self.text_chars = 0
        self.thinking_chars = 0
        self.tool_calls: List[Dict[str, Any]] = []
        self._tool_calls_by_index: Dict[int, Dict[str, Any]] = {}

    def _mark_first_semantic_delta(self) -> None:
        if (
            self.server_ttft_ms is not None
            or self.request_started_at_monotonic is None
        ):
            return
        self.server_ttft_ms = round(
            (perf_counter() - self.request_started_at_monotonic) * 1000, 3
        )

    # -- block state machine -------------------------------------------------

    def _close_block(self) -> List[str]:
        if self.open_block_type is None:
            return []
        event = AnthropicContentBlockStopEvent(index=self.block_index)
        self.open_block_type = None
        self.open_tool_index = None
        return [anthropic_sse(event)]

    def _open_block(
        self, block: Any, block_type: str, tool_index: Optional[int] = None
    ) -> List[str]:
        frames = self._close_block()
        self.block_index += 1
        self.open_block_type = block_type
        self.open_tool_index = tool_index
        frames.append(
            anthropic_sse(
                AnthropicContentBlockStartEvent(index=self.block_index, content_block=block)
            )
        )
        return frames

    def _ensure_block(self, block_type: str) -> List[str]:
        if self.open_block_type == block_type:
            return []
        if block_type == "text":
            return self._open_block(AnthropicTextBlock(text=""), "text")
        if block_type == "thinking":
            return self._open_block(AnthropicThinkingBlock(thinking=""), "thinking")
        raise ValueError(f"unexpected block type {block_type}")

    # -- chunk handling -------------------------------------------------------

    def _start_message(self, usage: Optional[AnthropicUsage]) -> List[str]:
        if self.message_started:
            return []
        self.message_started = True
        skeleton = AnthropicMessagesResponse(
            id=self.message_id,
            model=self.model,
            content=[],
            usage=usage or AnthropicUsage(),
        )
        # ``stop_reason``/``stop_sequence`` intentionally stay None here and
        # are delivered by the final message_delta.
        return [anthropic_sse(AnthropicMessageStartEvent(message=skeleton))]

    def process_chunk(self, chunk: ChatCompletionStreamResponse) -> List[str]:
        frames: List[str] = []

        if chunk.id:
            self.openai_response_id = chunk.id

        usage = None
        if chunk.usage is not None:
            usage = convert_usage(chunk.usage)
            self.final_usage = usage
        if not self.message_started:
            start_usage = None
            if usage is not None:
                start_usage = AnthropicUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=0,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                )
            frames.extend(self._start_message(start_usage))

        for choice in chunk.choices:
            delta = choice.delta
            reasoning_delta = delta.reasoning_content or delta.reasoning
            if reasoning_delta:
                self._mark_first_semantic_delta()
                self.thinking_chars += len(reasoning_delta)
                frames.extend(self._ensure_block("thinking"))
                frames.append(
                    anthropic_sse(
                        AnthropicContentBlockDeltaEvent(
                            index=self.block_index,
                            delta=AnthropicThinkingDelta(thinking=reasoning_delta),
                        )
                    )
                )
            if delta.content:
                self._mark_first_semantic_delta()
                self.text_chars += len(delta.content)
                frames.extend(self._ensure_block("text"))
                frames.append(
                    anthropic_sse(
                        AnthropicContentBlockDeltaEvent(
                            index=self.block_index, delta=AnthropicTextDelta(text=delta.content)
                        )
                    )
                )
            for tool_call in delta.tool_calls or []:
                function = tool_call.function
                if function is None:
                    continue
                if function.name:
                    self._mark_first_semantic_delta()
                    # A named fragment starts a new tool call. Force a new
                    # block even if a tool_use block is already open so
                    # argument deltas of parallel calls never merge.
                    block = AnthropicToolUseBlock(
                        id=tool_call.id or f"toolu_{uuid.uuid4().hex}",
                        name=function.name,
                        input={},
                    )
                    tool_call_summary = {
                        "id": block.id,
                        "name": block.name,
                        "input_json_bytes": 0,
                    }
                    self.tool_calls.append(tool_call_summary)
                    if tool_call.index is not None:
                        self._tool_calls_by_index[tool_call.index] = tool_call_summary
                    frames.extend(self._open_block(block, "tool_use", tool_index=tool_call.index))
                if function.arguments:
                    if self.open_block_type != "tool_use" or (
                        tool_call.index is not None
                        and self.open_tool_index is not None
                        and tool_call.index != self.open_tool_index
                    ):
                        logger.warning(
                            "Dropping tool argument fragment without a "
                            "matching open tool_use block (index=%s)",
                            tool_call.index,
                        )
                        continue
                    tool_call_summary = self._tool_calls_by_index.get(tool_call.index)
                    if tool_call_summary is not None:
                        tool_call_summary["input_json_bytes"] += len(
                            function.arguments.encode("utf-8")
                        )
                    frames.append(
                        anthropic_sse(
                            AnthropicContentBlockDeltaEvent(
                                index=self.block_index,
                                delta=AnthropicInputJsonDelta(partial_json=function.arguments),
                            )
                        )
                    )
            if choice.finish_reason:
                self.stop_reason, self.stop_sequence = _map_stop_result(
                    choice.finish_reason, choice.stop_reason
                )

        return frames

    def audit_response_summary(self) -> dict[str, Any]:
        """Return a content-free summary of the reframed streamed response."""
        return {
            "text_chars": self.text_chars,
            "thinking_chars": self.thinking_chars,
            "tool_calls_emitted": self.tool_calls,
        }

    def finish(self) -> List[str]:
        frames = self._close_block()
        frames.extend(self._start_message(None))  # degenerate empty stream
        frames.append(
            anthropic_sse(
                AnthropicMessageDeltaEvent(
                    delta=AnthropicMessageDelta(
                        stop_reason=self.stop_reason,
                        stop_sequence=self.stop_sequence,
                    ),
                    usage=self.final_usage or AnthropicUsage(),
                )
            )
        )
        frames.append(anthropic_sse(AnthropicMessageStopEvent()))
        return frames

    def error(self, message: str) -> List[str]:
        """Close any open block, then surface an error event."""
        frames = self._close_block()
        frames.append(
            anthropic_sse(
                AnthropicErrorEvent(error=AnthropicError(type="api_error", message=message))
            )
        )
        return frames


async def _iter_openai_sse_lines(
    openai_sse: AsyncIterator[Union[str, bytes]],
) -> AsyncIterator[str]:
    """Yield complete lines from string frames or arbitrarily chunked bytes.

    The in-process server produces complete SSE strings, while the
    disaggregated frontend relays raw ``aiohttp`` byte chunks.  Network chunks
    need not align with either UTF-8 characters or SSE line boundaries, so
    retain incomplete input until the next chunk arrives.
    """
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    input_type: Optional[type] = None

    async for payload in openai_sse:
        if not isinstance(payload, (str, bytes)):
            raise TypeError(
                f"OpenAI SSE payload must be str or bytes, got {type(payload)!r}"
            )

        current_type = type(payload)
        if input_type is None:
            input_type = current_type
        elif current_type is not input_type:
            raise TypeError("OpenAI SSE stream cannot mix str and bytes payloads")

        if isinstance(payload, bytes):
            buffer += decoder.decode(payload)
        else:
            buffer += payload

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line.rstrip("\r")

    if input_type is bytes:
        buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer.rstrip("\r")


async def reframe_openai_stream(
    openai_sse: AsyncIterator[Union[str, bytes]],
    model: str,
    on_finished: Optional[Callable[[AnthropicStreamReframer, Optional[str]], Any]] = None,
    request_started_at_monotonic: float | None = None,
) -> AsyncIterator[str]:
    """Translate an OpenAI SSE stream into Anthropic SSE frames."""
    reframer = AnthropicStreamReframer(
        model=model,
        request_started_at_monotonic=request_started_at_monotonic,
    )
    finished = False

    async def finish_audit(error: Optional[str] = None) -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        if on_finished is not None:
            result = on_finished(reframer, error)
            if inspect.isawaitable(result):
                await result

    try:
        async for line in _iter_openai_sse_lines(openai_sse):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data:
                continue
            if data == "[DONE]":
                for frame in reframer.finish():
                    yield frame
                await finish_audit()
                return
            try:
                chunk = ChatCompletionStreamResponse(**json.loads(data))
            except (json.JSONDecodeError, ValueError) as e:
                raise AnthropicResponseError("Malformed upstream stream chunk") from e
            for frame in reframer.process_chunk(chunk):
                yield frame
        # Upstream ended without [DONE]; still terminate the message cleanly.
        for frame in reframer.finish():
            yield frame
        await finish_audit()
    except Exception as e:  # noqa: BLE001 - stream must end with an event
        logger.error(
            f"Anthropic stream reframing failed: {e}\n{traceback.format_exc()}"
        )
        await finish_audit(str(e))
        for frame in reframer.error("Internal server error"):
            yield frame
    finally:
        await finish_audit("stream_cancelled")
