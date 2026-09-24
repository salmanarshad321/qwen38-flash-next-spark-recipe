# As-run deployment, verified 2026-09-24

This is an inventory of the live Spark container, read from Docker inspection and the retained benchmark records. It describes what was running at inspection time; it does not imply the same settings are optimal on another machine.

## Identity

| Item | Value |
| --- | --- |
| Container | `qwen38-scheduler-adaptive4096-20260921` |
| Image tag with matching ID | `qwen38-flash-dgx:staged-speed-20260919` |
| Immutable image ID | `sha256:75894119f7a43e1b9ea27ccc64265f646b916b85d07534d00d3b1f395832d28e` |
| vLLM version | `0.1.dev20073+g8e685d198` |
| Checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4`, snapshot `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Advertised model | `qwen3.8-flash-next` |
| Context | Native `262144`; no YaRN extension |

The local Docker `:latest` tag points to an August 30 image (`c1988df3f5f7`); the running container is pinned to the September 19 image ID above. Container creation date and image build date are different.

## Runtime configuration

The model path is a bind-mounted Hugging Face snapshot. Model weights, credentials, cache directories, and private endpoint configuration are **not** included in this repository.

| Setting | Value |
| --- | --- |
| `--max-model-len` | `262144` |
| `--max-num-seqs` | `8` |
| `--gpu-memory-utilization` | `0.80` |
| `--max-num-batched-tokens` | `8192` |
| `--max-num-scheduled-tokens` | `8192` |
| `--scheduler-cls` | `adaptive_scheduler.AdaptiveAsyncScheduler` |
| Speculative config | MTP, 3 draft tokens, local argmax reduction, EAGLE block drop disabled |
| `--async-scheduling` | enabled |
| Prefix caching | enabled |
| `--kv-cache-dtype` | `auto` (BF16 for this checkpoint/runtime) |
| Graph mode | `FULL_AND_PIECEWISE`, compilation mode 3, capture widths `[1,2,3,4,5,6,7,8,12,16,20,24,28,32,40,48,56,64]`, max width 64 |

The scheduler source is read-only mounted at `/opt/qwen-adaptive`. Its input capacity remains 8,192. Before each inherited scheduler call it selects an 8,192-token step budget when only prefills run, or 4,096 when any request is decoding. The module checks hashes of seven installed vLLM files and refuses an unreviewed runtime change.

Selected environment variables read from the container:

```text
VLLM_PLE_MMAP=1
VLLM_PLE_MMAP_FAST_ROWS=0
VLLM_PLE_MMAP_MADVISE=random
VLLM_PLE_MMAP_PREWARM=1
VLLM_PLE_MMAP_WORKERS=32
VLLM_PLE_ROW_CACHE_MIB=64
VLLM_PLE_STAGED=1
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_FP8_HYBRID=0
VLLM_QSA_DET_TOPK=1
VLLM_QSA_QUERY_PACKING=1
QWEN_ADAPTIVE_PREFILL_LOW=4096
```

The mounted draft-vocabulary file is `source/draft_vocab/draft_vocab_en_code_47k.txt`: 47,149 rows, SHA256 `20e36b6e8eae2598019298959a578ef8adc2948bbed7189e43a8da9b9d84a0b1`. It is byte-identical to the [MiaAI Lab artifact](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/d03809008834124e80223c3482f2ddb59577a48f/files/draft_vocab_en_code_47k.txt). The target still verifies against its full vocabulary.

MiaAI's later [September 14 change](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/CHANGELOG.md) adds 23 byte-fallback IDs, making its current file 47,172 rows. The Spark was inspected with the earlier 47,149-row file. This repository preserves that exact as-run artifact.

## Image lineage

The tags below are local build artifacts on this Spark, in the order retained by the current deployment. Their Dockerfiles and reports live in the September tuning directories on the Spark; selected final overlays are preserved under `source/` here.

| Layer | Change carried forward | Origin to credit |
| --- | --- | --- |
| `optimized-20260830` | Local blazux-derived base and early GB10 changes | blazux, vLLM, other credited patches; full historical build context is unavailable |
| `tuned-20260905` | Deterministic top-k kernel | jschmied; vLLM |
| `tuned-loader-20260905` | PLE loading adjustment | Local adaptation |
| `draft-vocab-20260906` | Reduced MTP head | MiaAI Lab AGPL patch adapted locally |
| `speed-20260912` | GDN, mmap advice, speculative and scheduler source changes | vLLM PRs, blazux/Saren, local integration |
| `fastrows-async-20260917` | Parallel PLE rows and async metadata transfer | blazux #25, vLLM #55054 |
| `kernel-20260918` | Updated deterministic kernel and selective QSA gate fusion | jschmied, vLLM #55309 |
| `ple-cache-20260919` | Bounded exact raw-row cache | Local work on blazux-derived mmap source |
| `exact-speed-20260919` | Four-query QSA scorer | lovablelabs/vLLM PR #1, adapted locally |
| `staged-speed-20260919` | Staged PLE and graph path | Adapted from pangoleen and vLLM |
| Runtime mount, Sep 21 | Adaptive scheduler | Local work |

The full source history and licensing relationship for distributed files is in [NOTICE.md](../NOTICE.md). The `optimized-20260830` ancestor prevents a byte-for-byte rebuild solely from this repository; [REPRODUCE.md](REPRODUCE.md) separates the available source from that missing historical artifact.
