# Qwen3.8-Flash-Next on one DGX Spark: my measured setup

This repository records the Qwen3.8-Flash-Next deployment running on my single DGX Spark on **2026-09-24**. It combines the [blazux single-Spark vLLM recipe](https://github.com/blazux/qwen3.8-Flash-DGX) with credited community and vLLM changes, plus locally developed scheduling, PLE caching, integration, and measurement work. [NOTICE.md](NOTICE.md) maps the sources file by file.

The exact historical image is a chain of local images built between August 30 and September 19. Its first local ancestor is not published as an image or a complete build context, so cloning this repository does not rebuild that image byte for byte. A [portable SparkRun recipe](docs/SPARKRUN.md) now applies the captured serving source to the digest-pinned public vLLM base at startup. An [optional image build](image/Dockerfile) uses the same source. The original benchmark figures describe the historical image; the packaged runtime needs its own matched run. See [reproduction notes](docs/REPRODUCE.md).

## SparkRun

```sh
sparkrun registry add https://github.com/salmanarshad321/qwen38-flash-next-spark-recipe.git
sparkrun run @qwen38-flash-next-spark/qwen38-flash-next-nvfp4 -H YOUR_SPARK_HOST --tp 1 --dry-run --trust
```

Then remove `--dry-run` to start it on an idle Spark. The recipe pulls the public vLLM base, applies the bundled mod, and downloads the pinned model; no GHCR login is needed. Read the [setup and limitations](docs/SPARKRUN.md) before launch.

## Running configuration

| Component | As run |
| --- | --- |
| Hardware | One NVIDIA DGX Spark, GB10, 128 GB unified memory |
| Model | `RadixArk/Qwen3.8-Flash-Next-NVFP4`, snapshot `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Runtime | vLLM `0.1.dev20073+g8e685d198`, local image `qwen38-flash-dgx:staged-speed-20260919` |
| Image ID | `sha256:75894119f7a43e1b9ea27ccc64265f646b916b85d07534d00d3b1f395832d28e` |
| Context and cache | Native 262,144-token context; BF16 KV; prefix caching |
| Speculation | MTP 3; 47,149-row draft vocabulary from MiaAI Lab; full target verification |
| PLE | NVMe mmap, 32 gather workers, random page advice, 64 MiB raw-row cache, staged PLE |
| Scheduling | Async scheduling, eight sequences, 8,192 input capacity; adaptive 8,192/4,096 per-step prefill budget |
| Graphs | `FULL_AND_PIECEWISE`, capture widths 1–64 |

The container uses the staged-speed image with a **read-only mount** of [`adaptive_scheduler.py`](source/adaptive_scheduler.py); the scheduler is a runtime addition, not part of the image layers. [Exact launch details](docs/AS_RUN.md).

## Measured on this Spark

The [49-request workload](bench/workload.py) checks short coding, reasoning, 20K-token retrieval, cached retrieval, concurrency, and an active decode stream competing with prefill. Each saved run passed its functional checks and request/token accounting, with zero preemptions.

| Metric | Original 4,096-token budget, Sep 21 | Adaptive, Sep 21 | Adaptive recheck, Sep 24 |
| --- | ---: | ---: | ---: |
| Short deterministic code decode, median | 48.81 tok/s | 48.92 tok/s | 49.18 tok/s |
| Cold 20,027-token prompt, median TTFT | 8.395 s | 7.693 s | 7.628 s |
| Shared-prefix 20,027-token prompt, median TTFT | 0.591 s | 0.595 s | 0.583 s |
| Median longest gap in active stream during competing prefill | 1.358 s | 1.376 s | 1.365 s |

The matched September 21 cold-prompt improvement is **8.35%**. A separate direct-server check of three fresh 18-token prompts reached first token in **0.165–0.180 s**; the 7.6-second figure is specific to the cold 20K-token workload. Read the [method and limitations](docs/BENCHMARKS.md) and [raw results](data/).

## Speed position

This is a fast, usable single-Spark configuration for my workload. It is **not established as the fastest overall**. Speculative decode speed depends heavily on generated content and draft acceptance. [MiaAI's September 9 sweep](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/CHANGELOG.md) reports 61.5 tok/s on its 600-token code test and 46.8 tok/s on prose; [blazux](https://github.com/blazux/qwen3.8-Flash-DGX) reports approximately 34 tok/s on its NVIDIA-checkpoint setup and faster RadixArk decode in a matched checkpoint comparison. Those prompts, checkpoints, settings, and timing methods differ from mine. No cross-repository speed ranking follows from these figures. [Comparison details](docs/BENCHMARKS.md).

## Contents

- [`docs/SPARKRUN.md`](docs/SPARKRUN.md): install and run the portable single-Spark recipe.
- [`recipes/qwen38-flash-next-nvfp4.yaml`](recipes/qwen38-flash-next-nvfp4.yaml): pinned SparkRun v2 launch configuration.
- [`recipes/mods/qwen38-source-overlay/`](recipes/mods/qwen38-source-overlay/): captured runtime source, kernel build, and hash checks.
- [`image/`](image/): optional container build and attribution map.
- [`docs/AS_RUN.md`](docs/AS_RUN.md): live image, checkpoint, environment, and image lineage.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): measurement protocol, results, and limits.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): public starting point and version pins.
- [`source/`](source/): retained scheduler and selected runtime overlays, including credited adaptations.
- [`bench/`](bench/): the API workload used for the numbers above.
- [`data/`](data/): full JSON output for the matched baseline, adaptive candidate, and September 24 recheck.
- [`SHA256SUMS`](SHA256SUMS): checksums of the distributed source and result files.

## License and credit

This repository is licensed under **AGPL-3.0-or-later** because it includes MiaAI Lab's draft-vocabulary artifact and an adapted MTP patch. Other included files retain their upstream Apache-2.0 and MIT provenance, detailed in [NOTICE.md](NOTICE.md). Model weights are not included and retain their own terms. This project is an independently operated configuration; it is not an official blazux, MiaAI Lab, vLLM, or Qwen release.
