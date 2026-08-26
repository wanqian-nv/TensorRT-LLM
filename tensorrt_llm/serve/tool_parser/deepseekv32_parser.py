# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/sgl-project/sglang/blob/0071fe9c407ad59f2803cc319e1bcaa3ac2021f1/python/sglang/srt/function_call/deepseekv32_detector.py
import json
import re
from typing import List

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, StructureInfo, ToolCallItem, _GetInfoFunc


class DeepSeekV32Parser(BaseToolParser):
    """Tool parser for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    needs_raw_special_tokens = True

    _eos_token = "<｜end▁of▁sentence｜>"  # nosec B105

    # Invoke header up to the function name, which is arbitrary text.
    _INVOKE_HEADER_PREFIX = '<｜DSML｜invoke name="'  # nosec B105

    # The marker on its own is ambiguous -- it opens a real invoke header, but
    # it also occurs verbatim in prose. `_invoke_marker_state` tells them apart.
    _INVOKE_MARKER = "<｜DSML｜invoke"  # nosec B105
    # Whitespace is whatever the template rendered, so match it the way
    # `invoke_pattern` does rather than assuming a single space.
    _INVOKE_HEADER_RE = re.compile(r'<｜DSML｜invoke\s+name="')
    # Prefixes of `name="` that a partial header may have reached so far.
    _INVOKE_HEADER_CONTINUATIONS = ('n', 'na', 'nam', 'name', 'name=',
                                    'name="')

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜function_calls>"  # nosec B105
        self.eot_token = "</｜DSML｜function_calls>"  # nosec B105
        self.invoke_begin_regex = r'<｜DSML｜invoke\s+name="([^"]+)"\s*>'
        self.invoke_end_token = "</｜DSML｜invoke>"  # nosec B105
        self.parameter_regex = (
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        self._inside_tool_calls = False
        self._expects_tool_calls_end = False

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text

    def _parse_parameters_from_xml(self, invoke_content: str) -> dict:
        """Parse parameters from either XML-like format or JSON format to dict.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()

        if invoke_content_stripped.startswith("{") and invoke_content_stripped.endswith("}"):
            try:
                parameters = json.loads(invoke_content_stripped)
                if isinstance(parameters, dict):
                    return parameters
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, fall through to XML parsing
                pass

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        param_matches = re.findall(self.parameter_regex, invoke_content, re.DOTALL)
        for param_name, param_type, param_value in param_matches:
            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()
        return parameters

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if self._eos_token in text:
            text = text.replace(self._eos_token, "")
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            # Extract content between function_calls tags
            function_calls_match = re.search(
                re.escape(self.bot_token) + r"(.*?)" + re.escape(self.eot_token),
                text,
                re.DOTALL,
            )
            if not function_calls_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            function_calls_content = function_calls_match.group(1)

            # Find all invoke blocks
            invoke_pattern = r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>'
            invoke_matches = re.findall(invoke_pattern, function_calls_content, re.DOTALL)

            for func_name, invoke_content in invoke_matches:
                # Parse parameters from XML format
                func_args = self._parse_parameters_from_xml(invoke_content)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def _control_tokens(self) -> tuple[str, ...]:
        """Return control strings whose prefixes may span output chunks."""
        return (
            self.bot_token,
            self.eot_token,
            self._INVOKE_MARKER,
            self.invoke_end_token,
            self._eos_token,
        )

    def _invoke_marker_state(self, text: str) -> str:
        """Classify text that starts with the bare invoke marker.

        The marker alone does not make a tool call: a real header continues
        with ``name="``. Prose may also contain the marker and then diverge
        (``<｜DSML｜invoke`` + ``ality``), and treating that as a tool call
        strands it in the buffer and makes ``finish()`` abort the response.

        Returns ``"header"`` when a real invoke header is already present,
        ``"partial"`` while more characters could still complete one, and
        ``"text"`` once the continuation has ruled a header out.
        """
        if self._INVOKE_HEADER_RE.match(text):
            return "header"
        rest = text[len(self._INVOKE_MARKER):]
        if rest == "" or rest.isspace():
            return "partial"
        stripped = rest.lstrip()
        # Whitespace must separate the marker from `name="`, so a continuation
        # that never had any cannot become a header.
        if len(rest) > len(stripped) and any(
                continuation.startswith(stripped)
                for continuation in self._INVOKE_HEADER_CONTINUATIONS):
            return "partial"
        return "text"

    def _ambiguous_suffix_length(self, text: str) -> int:
        """Find the longest suffix that still may become a control token."""
        max_length = 0
        for token in self._control_tokens():
            max_candidate = min(len(text), len(token) - 1)
            for length in range(1, max_candidate + 1):
                if text.endswith(token[:length]):
                    max_length = max(max_length, length)
        return max_length

    def _find_first_control(self) -> tuple[int, str] | None:
        """Find the first complete control string in the current buffer."""
        matches = []
        for token in self._control_tokens():
            position = self._buffer.find(token)
            if position != -1:
                matches.append((position, token))
        return min(matches, default=None, key=lambda match: match[0])

    def _consume_normal_text(self, text: str) -> None:
        """Observe text emitted outside a DSML tool-call section."""
        del text

    def _append_tool_call(self, match: re.Match,
                          calls: list[ToolCallItem]) -> None:
        """Convert one complete invoke block into streaming call deltas."""
        func_name = match.group(1).strip()
        current_params = self._parse_parameters_from_xml(match.group(2))
        current_args_json = json.dumps(current_params, ensure_ascii=False)

        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]

        calls.extend([
            ToolCallItem(
                tool_index=self.current_tool_id,
                name=func_name,
                parameters="",
            ),
            ToolCallItem(
                tool_index=self.current_tool_id,
                name=None,
                parameters=current_args_json,
            ),
        ])

        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": func_name,
            "arguments": current_params,
        }
        self.streamed_args_for_tool[
            self.current_tool_id] = current_args_json
        self.current_tool_id += 1
        self._last_arguments = ""
        self.current_tool_name_sent = False

    def parse_streaming_increment(
            self, new_text: str,
            tools: List[Tool]) -> StreamingParseResult:
        """Parse text and DSML calls without buffering safe normal content."""
        del tools  # Tool names are validated by the serving layer.
        self._buffer += new_text
        normal_parts: list[str] = []
        all_calls: list[ToolCallItem] = []
        invoke_pattern = (
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>')

        while self._buffer:
            if not self._inside_tool_calls:
                control = self._find_first_control()
                if control is None:
                    suffix_length = self._ambiguous_suffix_length(
                        self._buffer)
                    safe_length = len(self._buffer) - suffix_length
                    normal_text = self._buffer[:safe_length]
                    normal_parts.append(normal_text)
                    self._consume_normal_text(normal_text)
                    self._buffer = self._buffer[safe_length:]
                    break

                position, token = control
                if position:
                    normal_text = self._buffer[:position]
                    normal_parts.append(normal_text)
                    self._consume_normal_text(normal_text)
                    self._buffer = self._buffer[position:]

                if token == self._eos_token:
                    self._buffer = self._buffer[len(token):]
                    continue
                if token == self.bot_token:
                    self._buffer = self._buffer[len(token):]
                    self._inside_tool_calls = True
                    self._expects_tool_calls_end = True
                    continue
                if token == self._INVOKE_MARKER:
                    state = self._invoke_marker_state(self._buffer)
                    if state == "partial":
                        # Could still become a header; wait for the next chunk
                        # rather than committing either way.
                        break
                    if state == "text":
                        # Prose that merely looks like a marker. Emit it and
                        # keep scanning after it.
                        normal_parts.append(self._INVOKE_MARKER)
                        self._consume_normal_text(self._INVOKE_MARKER)
                        self._buffer = self._buffer[len(self._INVOKE_MARKER):]
                        continue
                    self._inside_tool_calls = True
                    self._expects_tool_calls_end = False
                    continue

                # Closing DSML tokens outside a tool section are structural
                # leftovers, never user-visible text.
                self._buffer = self._buffer[len(token):]
                continue

            invoke_match = re.search(invoke_pattern, self._buffer,
                                     re.DOTALL)
            if invoke_match is not None:
                prefix = self._buffer[:invoke_match.start()]
                if prefix.strip():
                    # Content the model emitted inside a tool-call section that
                    # is not part of an invoke. Surfacing it as text loses no
                    # tool call -- the invoke that follows is still parsed --
                    # whereas raising here aborts the whole response.
                    normal_parts.append(prefix)
                    self._consume_normal_text(prefix)
                self._append_tool_call(invoke_match, all_calls)
                self._buffer = self._buffer[invoke_match.end():]
                if not self._expects_tool_calls_end:
                    self._inside_tool_calls = False
                continue

            eot_position = self._buffer.find(self.eot_token)
            if eot_position != -1:
                leftover = self._buffer[:eot_position]
                if leftover.strip():
                    # The section is closing with content that never formed a
                    # complete invoke. Two things produce this: genuinely
                    # malformed generation, and a well-formed call whose
                    # argument text quotes the end marker -- the latter only
                    # while streaming, since the marker arrives before the
                    # invoke it belongs to. Neither is worth killing the stream
                    # for, so emit what we have as text and close the section.
                    normal_parts.append(leftover)
                    self._consume_normal_text(leftover)
                self._buffer = self._buffer[eot_position +
                                            len(self.eot_token):]
                self._inside_tool_calls = False
                self._expects_tool_calls_end = False
                continue

            eos_position = self._buffer.find(self._eos_token)
            if eos_position != -1:
                # Generation ended inside a tool-call section. The call is lost
                # either way; delivering the partial text beats failing the
                # request, and finish() still reports a truncated control token.
                leftover = self._buffer[:eos_position]
                if leftover.strip():
                    normal_parts.append(leftover)
                    self._consume_normal_text(leftover)
                self._buffer = self._buffer[eos_position +
                                            len(self._eos_token):]
                self._inside_tool_calls = False
                self._expects_tool_calls_end = False
                continue
            break

        return StreamingParseResult(normal_text="".join(normal_parts),
                                    calls=all_calls)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Flush safe text and reject an incomplete DSML/control sequence."""
        result = self.parse_streaming_increment("", tools)
        if self._inside_tool_calls:
            raise ValueError(
                "Incomplete DeepSeek DSML tool call at end of stream")

        incomplete_control = any(
            token.startswith(self._buffer) for token in self._control_tokens())
        if len(self._buffer) > 1 and incomplete_control:
            raise ValueError(
                "Incomplete DeepSeek DSML/control token at end of stream")

        normal_text = result.normal_text + self._buffer
        self._buffer = ""
        return StreamingParseResult(normal_text=normal_text,
                                    calls=result.calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger=f'<｜DSML｜invoke name="{name}">',
        )
