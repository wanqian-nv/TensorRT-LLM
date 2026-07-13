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
"""DKV: layer-split distributed KV storage with a group-replicated KV lifecycle.

All DKV-related helper code lives in this module for now; it will be split
into a package once the streamer/staging components land.
"""

import hashlib
import os
from typing import Any, List, Optional, Tuple

from tensorrt_llm.logger import logger


class DkvInvariantChecker:
    """Debug-only cross-rank consistency checker for DKV.

    DKV correctness relies on every rank independently deriving identical
    scheduling decisions from replicated state. A divergence is silent and
    surfaces much later (e.g. as a mismatched MoE collective). When enabled
    (TRTLLM_DKV_DEBUG=1) this checker fingerprints per-iteration decisions,
    allgathers one digest per rank, and on mismatch dumps a per-rank diff
    and raises at the exact iteration the divergence appeared.
    """

    def __init__(self, dist, enabled: Optional[bool] = None):
        self.dist = dist
        if enabled is None:
            enabled = os.environ.get("TRTLLM_DKV_DEBUG", "0") not in ("0", "false", "False")
        self.enabled = enabled

    def check(self, iter_counter: int, tag: str, items: List[Tuple[Any, ...]]) -> None:
        """Assert that ``items`` is identical on every rank of the TP group.

        ``items`` must be a sorted list of fingerprint tuples, e.g.
        ``[(request_id, context_current_position, context_chunk_size,
        state_value), ...]``, with ADP dummy requests excluded (they are
        created rank-locally and legitimately differ across ranks).
        """
        if not self.enabled:
            return
        # sha256 over repr, NOT builtin hash(): str hashing is salted per
        # process (PYTHONHASHSEED) and would differ across ranks.
        digest = hashlib.sha256(repr(items).encode()).hexdigest()
        all_digests = self.dist.tp_allgather(digest)
        if all(d == all_digests[0] for d in all_digests):
            return
        all_items = self.dist.tp_allgather(items)
        report = [f"DKV invariant violation at iter {iter_counter}, tag={tag}"]
        reference = all_items[0]
        for rank, rank_items in enumerate(all_items[1:], start=1):
            if rank_items == reference:
                continue
            report.append(f"  rank0-only: {[x for x in reference if x not in rank_items]}")
            report.append(f"  rank{rank}-only: {[x for x in rank_items if x not in reference]}")
        msg = "\n".join(report)
        logger.error(msg)
        raise RuntimeError(msg)
