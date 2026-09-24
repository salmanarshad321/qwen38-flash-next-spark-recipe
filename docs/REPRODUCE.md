# Reproduction and source status

## Start from the public base

The deployment grew from [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), whose current README contains the supported clean installation instructions. The historical checkout present on the Spark is `d2854bfff0a0b6f46984b0941ed1db6010031295`. The current upstream default has since changed, including its default checkpoint and context length, so following its current quickstart does **not** recreate the September 24 measurements here.

Use the pinned `RadixArk/Qwen3.8-Flash-Next-NVFP4` snapshot and the settings in [AS_RUN.md](AS_RUN.md) for a comparable target. Fetch weights from their original host; no checkpoint files are distributed here. The historical image `qwen38-flash-dgx:optimized-20260830` is the first local ancestor, and its complete build context was not retained. This repository cannot rebuild image `75894119f7a4` byte for byte. It now includes the [installed source differences](../image/SOURCE_MAP.md) between that image and the digest-pinned public vLLM base, plus the kernel build source. The [SparkRun mod](../recipes/mods/qwen38-source-overlay/) applies these files inside the public base before serving, or a source-equivalent image can be built on a DGX Spark:

```sh
docker build -f image/Dockerfile -t qwen38-flash-next-spark-recipe:as-run-20260924 .
```

The image build and SparkRun mod verify 29 live-source hashes. The rebuilt CUDA binary and Docker layers have different hashes from the historical image. The packaged [SparkRun recipe](../recipes/qwen38-flash-next-nvfp4.yaml) uses the public base with the mod on one Spark, with the exact model revision pinned. Its performance and output still need a matched run before claiming equivalence to the saved September measurements.

## Retained overlays

The `source/` tree preserves the final adaptive scheduler, the MiaAI-derived draft-vocabulary patch and its exact token list, the locally adapted staged PLE path, the bounded PLE row cache, and the four-query QSA scorer. The staged PLE source checks exact vLLM hashes and will reject a newer runtime. The two included candidate Dockerfiles name **local parent image tags** and are evidence of how those layers were built, not commands that work against arbitrary upstream images.

Several intermediate image layers used other source replacements and a compiled deterministic kernel. Their origin and order are listed in [AS_RUN.md](AS_RUN.md). The new `recipes/mods/qwen38-source-overlay/` directory preserves their final installed Python sources and the kernel build source. Do not silently substitute a current vLLM release for the pinned preview or turn off source-hash checks to make an overlay build.

## Re-run the published workload on a compatible server

The harness uses the local unauthenticated API inside the Spark and checks `/metrics`. It refuses to start its timing window until the server is idle. On a compatible vLLM server reachable at `127.0.0.1:30000`:

```sh
python3 bench/workload.py local-recheck
```

To use another local address, set `FLASH_BENCH_BASE` to that server's origin, without `/v1`. The result is written under `bench/results/`. The script sends 49 requests by default and takes several minutes; a live user's overlapping requests will make `traffic_clean` fail. It also assumes model ID `qwen3.8-flash-next`, streamed OpenAI-compatible chat completions, and vLLM Prometheus metrics. It does not accept or store a public API credential.

The archived JSON files are the result of the exact September 21 and September 24 server state. Rerunning later can produce different values because cache state, other traffic, and model-serving software change.
