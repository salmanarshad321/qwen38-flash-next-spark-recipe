# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next decode GEMM selection on Blackwell.

Dispatch follows Kimi-K3 and uses the local ``(N, K)`` shape and token count.
Plans contain measured CUDA graph capture sizes; other token counts use the
standard linear implementation.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

QWEN38NEXT_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4),
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=4, vector_width=4),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=2),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, static_k=10240),
        4: SkinnyGemmConfig(4, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4),
    },
}


# --- local patch: TP=1 shapes for a single GB10, measured against F.linear ---
# Only entries that beat cuBLAS by a LARGE margin are listed, and that restraint is
# load-bearing. A version of this table covering the CUDA graph capture sizes
# ({1, 2, 4, 8}) as well measured 1.5% SLOWER end to end: at those sizes the kernel wins
# by only 1.00-1.11x in a microbenchmark, the microbenchmark overstates the real gain by
# roughly 2.5x (M=1 measured 1.43x there and 1.069x end to end), and what is left does not
# cover the custom-op dispatch. Marginal wins are worth less than nothing here.
QWEN38NEXT_GEMM_PLANS.update(
    {
        # GDN fused in_proj, x36 per forward. 1.43x / 1.05x
        (10240, 2560): {
            1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4),
            3: SkinnyGemmConfig(3, 128, 1, k_unroll=4, vector_width=4),
        },
        # GDN/QSA output projection, x48. 1.05x / 1.07x
        (2560, 6144): {
            1: SkinnyGemmConfig(1, 32, 1, k_unroll=2),
            3: SkinnyGemmConfig(3, 32, 1, k_unroll=2),
        },
        # QSA fused QKV/gate, x36. 1.23x / 1.02x
        (6144, 2560): {
            1: SkinnyGemmConfig(1, 32, 4, k_unroll=2),
            3: SkinnyGemmConfig(3, 32, 2, k_unroll=2, vector_width=4),
        },
        # GDN fused QKVZ, x12. 1.38x / 1.06x
        (12288, 2560): {
            1: SkinnyGemmConfig(1, 32, 4, k_unroll=2, vector_width=4),
            3: SkinnyGemmConfig(3, 32, 4, k_unroll=2, vector_width=4),
        },
        # HyperConnection down/inject, x97 -- the biggest relative win. 2.20x / 1.92x
        (320, 10240): {
            1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
            3: SkinnyGemmConfig(3, 64, 1, k_unroll=4),
        },
        # QSA indexer Q/K and MoE gate, x108. M=1 measured 0.87x -> left to cuBLAS.
        (640, 2560): {
            3: SkinnyGemmConfig(3, 128, 2, k_unroll=4, vector_width=4),
        },
        # NOT included, and the reason is worth keeping: (10240, 320), x97 per forward,
        # measures 1.70x at M=1 -- but adding it made no difference end to end (step rate
        # 13.86 against v1's 13.95-14.09 band). At 6.2 MiB per call the absolute saving is
        # ~2.7 us after the microbenchmark's ~2.5x optimism, which the custom-op dispatch
        # eats. What the entries below have in common is a large ratio AND a large weight:
        # ratio alone does not survive. Also note K=320 needs vector_width 1 or 2, since
        # the kernel wants K divisible by block_size * vector_width -- a sweep that omits
        # those concludes the shape is unsupported, which is how it was first missed.
        # LM head, x1 per forward but 1.27 GiB of weights. 1.40x / 1.05x
        (248320, 2560): {
            1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
            3: SkinnyGemmConfig(3, 64, 1, k_unroll=4),
        },
    }
)
#
# Do not re-tune these configs against the standalone microbenchmark. Four attempts
# (M=2/4/8 coarse grid; adding (10240, 320); M=1/4 wide grid; M=1/2/3/4/8 wide grid, each
# entry individually verified faster) all measured a 1.3-2.3x microbenchmark win and all
# moved the end-to-end decode step by 0.0-0.4%, which is inside boot-to-boot noise.
#
# The reason: the microbenchmark calls one shape in a loop, so its weights stay resident in
# L2. Every "winning" config above is faster than this machine can even read the weights --
# (10240, 2560) is 52.4 MiB, which needs 192 us at 273 GB/s, and the microbenchmark reports
# 164 us. In a real forward the weights are cold every time and the GEMM is bandwidth-bound,
# so config choice cannot move it. The table below is what the shape-level dispatch is
# worth; the remaining decode time is not in these GEMMs.
#
# Compare candidates by STEP RATE (tok/s divided by tokens-per-chunk), never by tok/s:
# MTP acceptance length shifts between boots and swamps the effect being measured.
# --- end local patch ---


def _is_sm103() -> bool:
    # local patch: sm_121 (GB10) also runs this kernel correctly and faster than cuBLAS
    return current_platform.is_device_capability(
        (10, 3)
    ) or current_platform.is_device_capability((12, 1))


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


class _Qwen38NextLowLatencyApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen3_8_flash_next_low_latency_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen38NextLowLatencyLinearMethod(
    _Qwen38NextLowLatencyApply, UnquantizedLinearMethod
):
    pass


class Qwen38NextLowLatencyEmbeddingMethod(
    _Qwen38NextLowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


def _qwen38next_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    plan = QWEN38NEXT_GEMM_PLANS.get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(x.shape[0])
    if (
        config is not None
        and _runtime_ok(x, weight)
        and shape_dynamic_skinny_gemm.is_available()
    ):
        return shape_dynamic_skinny_gemm(x, weight, config)
    return torch.nn.functional.linear(x, weight)


def _qwen38next_low_latency_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen3_8_flash_next_low_latency_gemm",
    op_func=_qwen38next_low_latency_gemm,
    fake_impl=_qwen38next_low_latency_gemm_fake,
)


def enable_qwen38next_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    if dtype != torch.bfloat16 or not _is_sm103():
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = QWEN38NEXT_GEMM_PLANS.get((weight.shape[0], weight.shape[1]))
        if plan is None:
            continue
        if is_linear:
            child.quant_method = Qwen38NextLowLatencyLinearMethod()
        else:
            child.quant_method = Qwen38NextLowLatencyEmbeddingMethod()
        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
