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
"""Tests for the attention-DP routing-decision trace.

The trace exists to explain KV-affinity routing after the fact, so what it
records has to survive being read months later: the per-rank state the router
saw, what each candidate rank scored, and where the request went. These tests
pin that content, and pin that collecting it does not perturb the routing it
describes.

Mock objects only; no GPU.
"""

import json

from unittest.mock import MagicMock, Mock

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    KVCacheAwareADPRouter,
    RankState,
)


def _mock_dist(tp_rank=0, tp_size=2):
    dist = MagicMock()
    dist.tp_rank = tp_rank
    dist.tp_size = tp_size
    dist.mapping.dp_size = tp_size
    dist.has_cp_helix = False
    return dist


def _make_request_item(req_id, num_tokens=100, client_id=None,
                       target_dp_rank=None):
    item = MagicMock()
    item.id = req_id
    item.child_req_ids = None
    scheduling_params = MagicMock()
    scheduling_params.attention_dp_rank = target_dp_rank
    scheduling_params.attention_dp_relax = True
    item.request = MagicMock()
    item.request.py_scheduling_params = scheduling_params
    item.request.input_token_ids = list(range(num_tokens))
    item.request.num_input_tokens = num_tokens
    item.request.client_id = req_id if client_id is None else client_id
    item.request.cache_salt = None
    item.request.lora_config = None
    return item


def _router(tp_size=2, cold_start_warmup=False):
    return KVCacheAwareADPRouter(
        dist=_mock_dist(tp_size=tp_size),
        kv_cache_manager=MagicMock(),
        cold_start_warmup=cold_start_warmup,
    )


def _states(*pairs):
    return [
        RankState(rank=i, num_active_requests=n, num_active_tokens=t)
        for i, (n, t) in enumerate(pairs)
    ]


def test_trace_off_by_default():
    router = _router()
    router._all_ranks_prefix_matches = [{1: 0}, {1: 0}]
    router.route_requests(_states((0, 0), (0, 0)), [_make_request_item(1)],
                          max_num_active_requests=10)
    assert router._last_route_trace is None


def test_scored_decision_records_every_candidate_rank():
    router = _router()
    router.route_trace_enabled = True
    # Rank 0 holds 80 of the request's 100 tokens; rank 1 holds none.
    router._all_ranks_prefix_matches = [{1: 80}, {1: 0}]

    router.route_requests(_states((0, 0), (0, 0)),
                          [_make_request_item(1, num_tokens=100,
                                              client_id=4242)],
                          max_num_active_requests=10)

    trace = router._last_route_trace
    assert trace["num_unrouted"] == 0
    (decision, ) = trace["decisions"]
    assert decision["phase"] == "scored"
    assert decision["req_id"] == 1
    # client_id is the serve-edge id the Anthropic audit log records as
    # engine_request_id; req_id is the engine-local queue id.
    assert decision["client_id"] == 4242
    assert decision["req_tokens"] == 100
    assert decision["match_lens"] == {0: 80, 1: 0}
    assert set(decision["scores"]) == {0, 1}
    # 80/100 clears the default 0.1 match-rate gate, so the cached rank wins
    # and only the uncached 20 tokens are charged to it.
    assert decision["cache_affinity_active"] is True
    assert decision["best_rank"] == 0
    assert decision["effective_added"] == 20


def test_cache_affinity_gate_recorded_when_match_is_below_threshold():
    router = _router()
    router.route_trace_enabled = True
    # 5 of 1000 tokens is under the 0.1 threshold, so scoring ignores the match.
    router._all_ranks_prefix_matches = [{1: 5}, {1: 0}]

    router.route_requests(_states((0, 0), (0, 0)),
                          [_make_request_item(1, num_tokens=1000)],
                          max_num_active_requests=10)

    (decision, ) = router._last_route_trace["decisions"]
    assert decision["cache_affinity_active"] is False
    # The raw match length is still recorded -- the gate is a scoring decision,
    # not a reason to drop the measurement.
    assert decision["match_lens"][0] == 5


def test_load_before_is_captured_ahead_of_any_mutation():
    router = _router()
    router.route_trace_enabled = True
    router._all_ranks_prefix_matches = [{1: 0, 2: 0}, {1: 0, 2: 0}]

    router.route_requests(
        _states((1, 100), (3, 700)),
        [_make_request_item(1, num_tokens=50),
         _make_request_item(2, num_tokens=50)],
        max_num_active_requests=10)

    load_before = router._last_route_trace["load_before"]
    assert load_before["num_active_requests"] == [1, 3]
    assert load_before["num_active_tokens"] == [100.0, 700.0]


def test_load_after_is_reconstructible_from_load_before_and_decisions():
    # load_after is deliberately not recorded; this is the reconstruction that
    # makes it redundant.
    router = _router()
    router.route_trace_enabled = True
    router._all_ranks_prefix_matches = [{1: 0, 2: 0}, {1: 0, 2: 0}]

    router.route_requests(
        _states((0, 0), (0, 0)),
        [_make_request_item(1, num_tokens=40),
         _make_request_item(2, num_tokens=60)],
        max_num_active_requests=10)

    trace = router._last_route_trace
    tokens = list(trace["load_before"]["num_active_tokens"])
    counts = list(trace["load_before"]["num_active_requests"])
    for decision in trace["decisions"]:
        tokens[decision["best_rank"]] += decision["effective_added"]
        counts[decision["best_rank"]] += 1
    assert sum(counts) == 2
    assert sum(tokens) == 100


def test_warmup_phase_recorded_without_scores():
    # Cold-start warmup places the first requests before scoring runs, so those
    # decisions have no candidate set to record. They still have to appear, or
    # the trace silently omits the start of every run.
    router = _router(cold_start_warmup=True)
    router.route_trace_enabled = True
    router._all_ranks_prefix_matches = [{1: 0}, {1: 0}]

    router.route_requests(_states((0, 0), (0, 0)), [_make_request_item(1)],
                          max_num_active_requests=10)

    (decision, ) = router._last_route_trace["decisions"]
    assert decision["phase"] == "warmup"
    assert "match_lens" not in decision
    assert "scores" not in decision
    assert decision["best_rank"] == 0


def test_strict_placement_recorded_with_its_own_phase():
    router = _router()
    router.route_trace_enabled = True
    router._all_ranks_prefix_matches = [{1: 0}, {1: 0}]
    item = _make_request_item(1, target_dp_rank=1)
    item.request.py_scheduling_params.attention_dp_relax = False

    router.route_requests(_states((0, 0), (0, 0)), [item],
                          max_num_active_requests=10)

    (decision, ) = router._last_route_trace["decisions"]
    assert decision["phase"] == "strict"
    assert decision["best_rank"] == 1


def test_num_unrouted_counts_requests_the_cap_held_back():
    router = _router()
    router.route_trace_enabled = True
    router._all_ranks_prefix_matches = [{1: 0, 2: 0}, {1: 0, 2: 0}]

    # Both ranks are already at the per-rank ceiling, so no rank is eligible
    # and the batch is dropped whole.
    router.route_requests(
        _states((10, 1000), (10, 1000)),
        [_make_request_item(1), _make_request_item(2)],
        max_num_active_requests=10)

    trace = router._last_route_trace
    assert trace["decisions"] == []
    assert trace["num_unrouted"] == 2


def test_idle_batch_produces_no_record():
    router = _router()
    router.route_trace_enabled = True
    router._all_ranks_prefix_matches = [{}, {}]

    router.route_requests(_states((0, 0), (0, 0)), [],
                          max_num_active_requests=10)

    # route_requests runs every iteration, including ones with nothing to
    # place. Emitting there would make the trace as large as the iteration log.
    assert router._last_route_trace is None


def test_trace_is_json_serializable():
    router = _router()
    router.route_trace_enabled = True
    router._all_ranks_prefix_matches = [{1: 30}, {1: 0}]

    router.route_requests(_states((0, 0), (2, 200)),
                          [_make_request_item(1, num_tokens=100)],
                          max_num_active_requests=10)

    reloaded = json.loads(json.dumps(router._last_route_trace))
    # int dict keys survive as strings; consumers must not rely on int keys.
    assert reloaded["decisions"][0]["match_lens"] == {"0": 30, "1": 0}


def test_tracing_does_not_change_routing():
    # route_requests runs independently on every rank with no broadcast, so a
    # divergence would deadlock the collective protocol -- not just skew a
    # metric.
    def run(trace_enabled):
        router = _router(tp_size=4)
        router.route_trace_enabled = trace_enabled
        router._all_ranks_prefix_matches = [
            {i: (i * 37) % 90 for i in range(12)} for _ in range(4)
        ]
        states = _states((0, 0), (2, 400), (1, 150), (5, 900))
        items = [
            _make_request_item(i, num_tokens=50 + (i * 13) % 400)
            for i in range(12)
        ]
        assigned, expected = router.route_requests(
            states, items, max_num_active_requests=8)
        return {r: [it.id for it in v] for r, v in assigned.items()}, expected

    assert run(False) == run(True)


# ---- Executor-side emission ----
#
# _emit_route_trace only reads plain attributes off the executor, so it is
# exercised against a stand-in rather than a constructed PyExecutor.


class _FakeExecutor:

    _emit_route_trace = PyExecutor._emit_route_trace

    def __init__(self, sink, trace, iter_counter=7, is_warmup=False):
        self._route_trace_sink = sink
        self.iter_counter = iter_counter
        self.is_warmup = is_warmup
        self.adp_router = MagicMock()
        self.adp_router._last_route_trace = trace


class _Sink:

    def __init__(self):
        self.lines = []

    def write(self, s):
        self.lines.append(s)

    def flush(self):
        pass


def _trace():
    return {"load_before": {"num_active_requests": [0, 1],
                            "num_active_tokens": [0.0, 10.0]},
            "num_unrouted": 0,
            "decisions": [{"req_id": 1, "client_id": 1, "phase": "scored",
                           "req_tokens": 100, "best_rank": 0,
                           "effective_added": 100}]}


def test_emit_stamps_both_iteration_conventions():
    sink = _Sink()
    ex = _FakeExecutor(sink, _trace(), iter_counter=7)
    ex._emit_route_trace()

    (line, ) = sink.lines
    record = json.loads(line)
    # iter matches IterationStats.iter; log_iter matches the iteration log
    # line that reports the batch these decisions fed, which carries the next
    # counter value because profile_step() prints before the increment.
    assert record["iter"] == 7
    assert record["log_iter"] == 8
    assert record["decisions"][0]["req_id"] == 1


def test_emit_consumes_the_pending_trace():
    sink = _Sink()
    ex = _FakeExecutor(sink, _trace())
    ex._emit_route_trace()
    ex._emit_route_trace()

    # route_requests runs every iteration but only publishes on batches that
    # placed something; without clearing, the last batch would be rewritten
    # under every subsequent iteration number.
    assert len(sink.lines) == 1
    assert ex.adp_router._last_route_trace is None


def test_emit_drops_warmup_batches():
    sink = _Sink()
    ex = _FakeExecutor(sink, _trace(), is_warmup=True)
    ex._emit_route_trace()

    # Warmup routes synthetic capacity-probing requests through the same path.
    assert sink.lines == []
    # Dropped, not held: otherwise it would resurface under a live iteration.
    assert ex.adp_router._last_route_trace is None


def test_emit_is_a_noop_without_a_sink():
    ex = _FakeExecutor(None, _trace())
    ex._emit_route_trace()
    # The trace stays pending so enabling a sink later loses nothing, and
    # ranks without a sink never pay for the clear.
    assert ex.adp_router._last_route_trace is not None


def test_emit_survives_an_unserializable_field():
    sink = _Sink()
    trace = _trace()
    trace["decisions"][0]["client_id"] = object()
    ex = _FakeExecutor(sink, trace)
    ex._emit_route_trace()

    # Observability must not take down the executor loop.
    assert sink.lines == []
