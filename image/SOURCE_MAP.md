# Source and image reconstruction

`image/Dockerfile` and the [SparkRun mod](../recipes/mods/qwen38-source-overlay/) start from the public, digest-pinned official Qwen3.8-Flash-Next vLLM image used by [blazux at `b6dae9f`](https://github.com/blazux/qwen3.8-Flash-DGX/tree/b6dae9f598158370d8a833e54d5deb159288ee8b). The recipe pins its Linux arm64 manifest, `vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`, so an x86 control machine cannot distribute the wrong architecture. The September 24 live image and this base were compared file by file. The mod's `runtime-overrides/vllm/` contains all 16 Python files that differ inside the installed vLLM package; `runtime-overrides/root/` contains the four added companion modules. The full files are distributed here so the recipe does not depend on missing local Docker layers or a second image registry.

The mod's `BASE_SOURCE_SHA256SUMS` rejects an unexpected base. Its `LIVE_SOURCE_SHA256SUMS` records 29 source files in the live image and its two runtime mounts. The mod and Docker build check those hashes after copying. This is a source-equivalent reconstruction, **not** a byte-for-byte copy of the historical image: the deterministic CUDA extension is compiled afresh, Docker layer metadata differs, and benchmark results have not been replicated on this rebuilt image.

| Packaged source | Origin and credit |
| --- | --- |
| `vllm/` snapshots | [vLLM](https://github.com/vllm-project/vllm), Apache-2.0, with the integration and source changes listed below |
| `vllm_ple_mmap.py`, `vllm/…/ple_layer.py`, FLA files, Mamba guard | [blazux](https://github.com/blazux/qwen3.8-Flash-DGX/tree/b6dae9f598158370d8a833e54d5deb159288ee8b) and [Saren-Arterius](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound), Apache-2.0; later local integration and PLE gather changes |
| `vllm/…/mtp.py`, bundled draft vocabulary | [MiaAI Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/tree/d03809008834124e80223c3482f2ddb59577a48f), AGPL-3.0-or-later, adapted locally |
| `vllm_ple_staging.py`, `vllm/…/model_state.py` | Staged PLE path adapted from [pangoleen](https://github.com/pangoleen/qwen3.8-flash-next-dgx-spark/tree/5ac0ac951e66b1239eb65967ba609d0b12349cdf), MIT, and vLLM; local integration |
| `ple_row_cache.py`, `adaptive_scheduler.py` | Local work; see [NOTICE.md](../NOTICE.md) |
| `vllm/…/ops/qsa.py` | vLLM and [lovablelabs' four-query scorer](https://github.com/lovablelabs/vllm/pull/1), Apache-2.0; local adaptation |
| `kernel-src/` | [jschmied's GB10 deterministic kernel](https://github.com/jschmied/qwen38-flash-next-gb10/tree/3f4b6c1df8e3dbc30850241337e7a66f7fd84c93), Apache-2.0, with local updated source; `sources.json` identifies and hashes the five kernel files |
| Other changed vLLM files | vLLM fixes and local integration, including [#55054](https://github.com/vllm-project/vllm/pull/55054), [#55309](https://github.com/vllm-project/vllm/pull/55309), and [#55715](https://github.com/vllm-project/vllm/pull/55715). The [image lineage](../docs/AS_RUN.md#image-lineage) gives layer order. |

The model weights are fetched from [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) at the recipe's pinned revision and are not part of the image. The image contains no Hugging Face cache, endpoint credential, or private proxy configuration.
