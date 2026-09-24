"""Pinned, isolated Qwen3.8-Flash-Next scheduling experiment.

Load with --scheduler-cls adaptive_scheduler.AdaptiveAsyncScheduler.
Only the per-step token budget changes. See ADAPTIVE_SCHEDULER.md.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler


SOURCE_SHA256 = {
    "v1/core/sched/scheduler.py": "07449dbf8c27749f3af9c68da8c1df5d4315aa5bf74b2f9cfa7764eb92c029cb",
    "v1/core/sched/async_scheduler.py": "e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508",
    "v1/request.py": "0287844f70eeaeb077d714e833a4b449a15e045a6516f8530182e357a5bec82f",
    "config/scheduler.py": "bf399abc7216b4801fbba5fd8f5cbc1960c9458d7519a544ae3015d83dd7a7d6",
    "config/speculative.py": "c84b546976dc3657ecf4a8f8b5ff4533f37f339911988d40fec3280689998794",
    "config/vllm.py": "b845db7f30bf535e7e1ed5257f5d6513f604f4ed56b5466e2d08986b92f53f53",
    "v1/engine/core.py": "f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad",
}
HIGH_BUDGET = 8192
LOW_BUDGET_ENV = "QWEN_ADAPTIVE_PREFILL_LOW"


def _verify_source_files(root: Path) -> None:
    for relative, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unqualified adaptive scheduler source: {relative}")


def _read_low_budget() -> int:
    value = os.environ.get(LOW_BUDGET_ENV, "4096")
    if value not in {"2048", "4096"}:
        raise ValueError(f"{LOW_BUDGET_ENV} must be 2048 or 4096")
    return int(value)


def _require_config(config) -> None:
    scheduler = config.scheduler_config
    spec = config.speculative_config
    parallel = config.parallel_config
    if (
        scheduler.max_num_batched_tokens != 8192
        or scheduler.max_num_seqs != 8
        or scheduler.max_num_scheduled_tokens != 8192
        or scheduler.async_scheduling is not True
        or scheduler.enable_chunked_prefill is not True
        or scheduler.long_prefill_token_threshold != 0
    ):
        raise RuntimeError("Adaptive scheduler requires async chunked prefill, "
                           "8192 input/scheduled capacity, 8 sequences, no prefill threshold")
    if (
        spec is None
        or spec.method != "mtp"
        or spec.num_speculative_tokens != 3
        or config.num_speculative_tokens != 3
        or spec.uses_dynamic_speculative_decoding()
        or spec.max_num_new_slots_for_drafting != 0
    ):
        raise RuntimeError("Adaptive scheduler requires fixed autoregressive MTP3")
    if not config.use_v2_model_runner or os.environ.get("VLLM_PLE_STAGED") != "1":
        raise RuntimeError("Adaptive scheduler requires the current staged PLE/MRV2 path")
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_dbo
        or config.model_config.is_diffusion
    ):
        raise RuntimeError("Adaptive scheduler requires TP1/PP1/DP1 without DBO/diffusion")


class AdaptiveAsyncScheduler(AsyncScheduler):
    """Use larger prefill chunks only while no running request is decoding."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("vllm_config", args[0] if args else None)
        if config is None:
            raise TypeError("vllm_config is required")
        source = inspect.getsourcefile(AsyncScheduler)
        if source is None or AsyncScheduler.schedule is not Scheduler.schedule:
            raise RuntimeError("Adaptive scheduler requires the audited inherited schedule")
        _verify_source_files(Path(source).resolve().parents[3])
        _require_config(config)
        self._adaptive_low_budget = _read_low_budget()
        super().__init__(*args, **kwargs)
        if (
            self.dynamic_sd_lookup is not None
            or self.num_spec_tokens != 3
            or self.num_sampled_tokens_per_step != 1
            or self.max_num_running_reqs != 8
            or not self.use_v2_model_runner
            or not 0 < self.block_size <= self._adaptive_low_budget
        ):
            raise RuntimeError("Resolved scheduler does not match the adaptive contract")

    def schedule(self, throttle_prefills: bool = False):
        # The pinned base uses this same flag to recognize in-flight decoding,
        # including asynchronous output placeholders and resumed requests.
        # Keep the chosen budget constant for the whole base call: Mamba chunk
        # alignment reads it while determining safe cache-write boundaries.
        self.max_num_scheduled_tokens = (
            self._adaptive_low_budget
            if any(not request.is_prefill_chunk for request in self.running)
            else HIGH_BUDGET
        )
        return super().schedule(throttle_prefills=throttle_prefills)
