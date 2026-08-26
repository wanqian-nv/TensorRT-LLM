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
"""Route-level tests for the Anthropic Messages compatibility handlers."""

import gzip
import json
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from tensorrt_llm.serve.anthropic_adapter import (
    flush_anthropic_message_captures,
)
from tensorrt_llm.serve.anthropic_protocol import AnthropicCountTokensResponse
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

MODEL = "test-model"


def _request(**overrides):
    payload = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


def _chat_response(*, tool_arguments=None):
    tool_calls = []
    if tool_arguments is not None:
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(
                    name="get_weather", arguments=tool_arguments
                ),
            )
        ]
    return ChatCompletionResponse(
        id="chatcmpl-route-test",
        model=MODEL,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant", content="hello", tool_calls=tool_calls
                ),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def _json_chat_response(*, tool_arguments=None, status_code=200):
    return JSONResponse(
        content=_chat_response(tool_arguments=tool_arguments).model_dump(),
        status_code=status_code,
    )


def _streaming_chat_response():
    chunks = [
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"role": "assistant"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"content": "hello"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            usage=UsageInfo(
                prompt_tokens=3, completion_tokens=2, total_tokens=5
            ),
        ),
    ]

    async def source():
        for chunk in chunks:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(source(), media_type="text/event-stream")


def _make_route_client(server_kind, openai_response):
    app = FastAPI()
    if server_kind == "standard":
        server = object.__new__(OpenAIServer)
        server.model = MODEL
        backend = AsyncMock(return_value=openai_response)
        server.openai_chat = backend
    else:
        server = object.__new__(OpenAIDisaggServer)
        server._service = SimpleNamespace(openai_chat_completion=object())
        backend = AsyncMock(return_value=openai_response)
        server._wrap_entry_point = Mock(return_value=backend)
    app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
    return TestClient(app), backend


def _make_count_route_client(server_kind, input_tokens):
    app = FastAPI()
    if server_kind == "standard":
        server = object.__new__(OpenAIServer)
        backend = AsyncMock(return_value=input_tokens)
        server._count_chat_prompt_tokens = backend
    else:
        backend = AsyncMock(
            return_value=AnthropicCountTokensResponse(
                input_tokens=input_tokens)
        )
        server = object.__new__(OpenAIDisaggServer)
        server._service = SimpleNamespace(anthropic_count_tokens=backend)
    app.add_api_route(
        "/v1/messages/count_tokens",
        server.anthropic_count_tokens,
        methods=["POST"],
    )
    return TestClient(app), backend


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_converts_nonstream_response(server_kind):
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post("/v1/messages", json=_request())

    assert response.status_code == 200
    assert response.json() | {"id": "ignored"} == {
        "id": "ignored",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    chat_request = backend.await_args.args[0]
    assert chat_request.model == MODEL
    assert chat_request.max_completion_tokens == 64
    assert not chat_request.stream


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_count_tokens_route(server_kind):
    client, backend = _make_count_route_client(server_kind, 123)

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": MODEL,
            "system": "system prompt",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 123}
    forwarded_request = backend.await_args.args[0]
    assert forwarded_request.model == MODEL
    last_message = forwarded_request.messages[-1]
    content = (last_message.get("content") if isinstance(last_message, dict)
               else last_message.content)
    assert content == "hello"


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_rejects_anthropic_server_tools(server_kind):
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post(
        "/v1/messages",
        json=_request(
            tools=[
                {
                    "name": "web_search",
                    "type": "web_search_20250305",
                }
            ]
        ),
    )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "server tool" in response.json()["error"]["message"]
    backend.assert_not_awaited()


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_hides_invalid_generated_tool_arguments(server_kind):
    client, _ = _make_route_client(
        server_kind, _json_chat_response(tool_arguments="{not-json")
    )

    response = client.post("/v1/messages", json=_request())

    assert response.status_code == 500
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_reframes_streaming_response(server_kind):
    client, backend = _make_route_client(
        server_kind, _streaming_chat_response()
    )

    response = client.post("/v1/messages", json=_request(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("event: message_start\n")
    assert "event: content_block_delta\n" in response.text
    assert '"text":"hello"' in response.text
    assert response.text.rstrip().endswith(
        'event: message_stop\ndata: {"type":"message_stop"}'
    )
    assert backend.await_args.args[0].stream


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_writes_content_free_audit_record(tmp_path, monkeypatch, server_kind):
    audit_path = tmp_path / "anthropic-audit.jsonl"
    monkeypatch.setenv("TRTLLM_ANTHROPIC_AUDIT_LOG", str(audit_path))
    client, _ = _make_route_client(
        server_kind, _json_chat_response(tool_arguments='{"city":"Paris"}')
    )

    response = client.post(
        "/v1/messages",
        json=_request(
            metadata={"user_id": "user-123"},
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_previous",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_previous",
                            "content": "sunny",
                        }
                    ],
                },
            ],
        ),
        headers={"x-claude-session-id": "session-abc"},
    )

    assert response.status_code == 200
    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["client_session_id"] == "session-abc"
    assert record["client_session_source"] == "header:x-claude-session-id"
    assert record["metadata_user_id_present"] is True
    assert record["tool_results_in_last_message"] == [
        {"tool_use_id": "toolu_previous", "is_error": False, "content_chars": 5}
    ]
    assert record["usage"] == {
        "input_tokens": 3,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 2,
    }
    assert record["openai_response_id"] == "chatcmpl-route-test"
    assert record["engine_request_id"] is None
    assert record["disagg_request_id"] is None
    assert record["response"]["tool_calls_emitted"] == [
        {"id": "call_1", "name": "get_weather", "input_json_bytes": 16}
    ]
    assert "sunny" not in audit_path.read_text()
    assert "Paris" not in audit_path.read_text()


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_captures_full_request_offline(
    tmp_path, monkeypatch, server_kind
):
    audit_path = tmp_path / "anthropic-audit.jsonl"
    capture_dir = tmp_path / "private"
    monkeypatch.setenv("TRTLLM_ANTHROPIC_AUDIT_LOG", str(audit_path))
    monkeypatch.setenv(
        "TRTLLM_ANTHROPIC_BENCH_CAPTURE_DIR", str(capture_dir)
    )
    client, _ = _make_route_client(server_kind, _json_chat_response())
    request_body = _request(
        system="private system prompt",
        tools=[
            {
                "name": "get_weather",
                "description": "Get private weather data",
                "input_schema": {"type": "object"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Paris?",
            }
        ],
    )

    response = client.post(
        "/v1/messages",
        json=request_body,
        headers={
            "authorization": "Bearer private-token",
            "x-claude-session-id": "session-capture",
        },
    )
    flush_anthropic_message_captures()

    assert response.status_code == 200
    audit_record = json.loads(audit_path.read_text())
    capture_path = capture_dir / audit_record["message_capture_file"]
    with gzip.open(capture_path, "rt", encoding="utf-8") as capture_file:
        capture = json.load(capture_file)

    assert capture["audit_request_id"] == audit_record["audit_request_id"]
    assert capture["client_session_id"] == "session-capture"
    assert capture["method"] == "POST"
    assert capture["path"] == "/v1/messages"
    assert capture["body"] == request_body
    assert ["authorization", "Bearer private-token"] in capture["headers"]
    assert stat.S_IMODE(capture_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(capture_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(capture_path.stat().st_mode) == 0o600


def test_messages_route_writes_final_stream_audit_record(tmp_path, monkeypatch):
    audit_path = tmp_path / "anthropic-stream-audit.jsonl"
    monkeypatch.setenv("TRTLLM_ANTHROPIC_AUDIT_LOG", str(audit_path))
    client, _ = _make_route_client("standard", _streaming_chat_response())

    response = client.post("/v1/messages", json=_request(stream=True))

    assert response.status_code == 200
    record = json.loads(audit_path.read_text())
    assert record["status"] == "completed"
    assert record["server_ttft_ms"] is not None
    assert 0 <= record["server_ttft_ms"] <= record["duration_ms"]
    assert record["openai_response_id"] is not None
    assert record["usage"]["input_tokens"] == 3
    assert record["usage"]["output_tokens"] == 2
    assert record["response"] == {
        "text_chars": 5,
        "thinking_chars": 0,
        "tool_calls_emitted": [],
    }


def test_messages_route_records_adjacent_prompt_lcp(tmp_path, monkeypatch):
    audit_path = tmp_path / "anthropic-lcp-audit.jsonl"
    monkeypatch.setenv("TRTLLM_ANTHROPIC_AUDIT_LOG", str(audit_path))
    monkeypatch.setenv("TRTLLM_ANTHROPIC_LCP_TRACKING", "1")
    prompt_token_ids = iter(([1, 2, 3, 4], [1, 2, 3, 9, 10, 11]))

    async def backend(_request, raw_request):
        raw_request.state.prompt_token_ids = next(prompt_token_ids)
        return _json_chat_response()

    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server.openai_chat = AsyncMock(side_effect=backend)
    app = FastAPI()
    app.add_api_route("/v1/messages", server.anthropic_messages,
                      methods=["POST"])
    client = TestClient(app)
    headers = {"x-claude-session-id": "session-lcp"}

    assert client.post("/v1/messages", json=_request(),
                       headers=headers).status_code == 200
    assert client.post("/v1/messages", json=_request(),
                       headers=headers).status_code == 200

    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert records[0]["prompt_lcp_tokens"] is None
    assert records[1]["prompt_lcp_tokens"] == 3
    assert records[1]["previous_prompt_retention_ratio"] == 0.75
    assert records[1]["current_reuse_opportunity_ratio"] == 0.5


def test_standard_and_disagg_register_messages_route(monkeypatch, tmp_path):
    # Registering the disagg routes also mounts a Prometheus multiprocess
    # collector, which refuses to build without a real directory.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))

    standard = object.__new__(OpenAIServer)
    standard.app = FastAPI()
    standard.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    standard.use_harmony = False
    standard.register_routes()

    disagg = object.__new__(OpenAIDisaggServer)
    disagg.app = FastAPI()
    disagg._service = SimpleNamespace(
        openai_completion=AsyncMock(), openai_chat_completion=AsyncMock()
    )
    disagg._perf_metrics_collector = SimpleNamespace(
        get_perf_metrics=AsyncMock()
    )
    disagg._disagg_cluster_storage = None
    disagg._coordinator = None
    disagg.register_routes()

    for server in (standard, disagg):
        paths = {route.path for route in server.app.routes}
        assert "/v1/messages" in paths
        assert "/v1/messages/count_tokens" in paths
