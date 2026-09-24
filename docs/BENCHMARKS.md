# Benchmarks and comparison limits

## Local protocol

[`bench/workload.py`](../bench/workload.py) streams requests against the local vLLM API and checks generated responses. The saved September 21 baseline and adaptive run each contain 49 requests. The September 24 recheck repeats the adaptive run. The protocol includes short deterministic coding cases, one reasoning case, a 20,027-token retrieval prompt with cold and cached variants, concurrent coding requests at widths 1/4/8, and a decoding request competing with a 22K-token prefill. The source code and complete results are included in this repository.

`code_tps` is the median of six short code completions, calculated as `(completion_tokens - 1) / (response_duration - TTFT)`. `TTFT` is the arrival time of the first streamed content or reasoning chunk at the client on the Spark. The cold and warm rows are medians of three requests each. Prompt-cache salts isolate cold requests. The harness warms kernels before timing and checks that vLLM's successful-request and generated-token counter deltas exactly match its own traffic. All three saved runs report `all_pass=true`, `traffic_clean=true`, and zero preemptions.

| Measured median | Baseline Sep 21 | Adaptive Sep 21 | Adaptive Sep 24 |
| --- | ---: | ---: | ---: |
| Short code decode | 48.813 tok/s | 48.917 tok/s | 49.178 tok/s |
| Cold 20,027-token TTFT | 8.395 s | 7.693 s | 7.628 s |
| Shared-prefix 20,027-token TTFT | 0.591 s | 0.595 s | 0.583 s |
| Longest streaming gap during competing prefill | 1.358 s | 1.376 s | 1.365 s |
| Competing 22K-token prefill TTFT | 9.730 s | 9.801 s | 9.736 s |

The matched baseline-to-adaptive cold TTFT change on September 21 is `(8.395 - 7.693) / 8.395 = 8.35%`. The code-speed change is small and does not establish a decode improvement. The first-use mixed request on the adaptive run had a larger gap than the later repetitions; the raw data retains it. No aggregate concurrency-throughput gain is claimed, because request lengths and batching varied.

The 7.6-second figure is a **cold 20K-token prompt**, not interactive latency for a short message. On September 24, three fresh 18-token requests sent directly to the local server had 0.165, 0.180, and 0.178 s TTFT; all three small replies completed in approximately 0.25 s. These direct checks are not included in the saved 49-request dataset and may differ from an external proxy or pi client.

## Published results in context

- [MiaAI Lab's September 9 sweep](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/CHANGELOG.md) reports **61.5 tok/s code** and **46.8 tok/s prose** at one stream with its 47,149-row vocabulary, 600-token prompts, FP8 KV, BF16 recurrent state, 2,048-token chunks, and FULL decode graphs. It also reports **252.6 tok/s aggregate** on code at eight streams. This is a different checkpoint, prompt set, precision mix, and scheduling configuration from ours, even though the draft-vocabulary file is the same.
- [blazux](https://github.com/blazux/qwen3.8-Flash-DGX) reports approximately **34 tok/s** for its current NVIDIA-checkpoint default and, in a same-recipe checkpoint comparison, **37.1 tok/s RadixArk vs 34.5 tok/s NVIDIA** on greedy prose. That comparison also reports more KV space and a modestly higher agentic-tournament score for NVIDIA. Its default serves up to 500K context; ours advertises native 262K.
- [pangoleen](https://github.com/pangoleen/qwen3.8-flash-next-dgx-spark) reports a **52 tok/s mean** across its generation ladder with its published overlays and a **57 tok/s mean** for its strongest locally measured build. Those runs use their own prompt and launch conditions, so they are evidence of another fast route, not a matched defeat of this setup.
- [Felliks](https://github.com/Felliks/qwen38-flash-next-one-dgx-spark) uses the **RadixArk NVFP4 checkpoint, BF16 KV, and native 262K context**, reporting **32.7 tok/s steady single-stream** on a pinned SGLang configuration. [hasso5703's separate vLLM run](https://github.com/hasso5703/dgx-spark-qwen38/blob/main/BENCHMARKS.md) reports **46.3 tok/s greedy median** for RadixArk on its own bench protocol. These share some of our weight/precision choices, but differ in runtime, prompts, graph mode, or serving patches. Neither is an apples-to-apples ranking.
- Our **49.18 tok/s is short deterministic code**, so it should not be placed above their prose number as a general ranking. Draft acceptance and prompt shape strongly affect MTP throughput.

The available evidence supports a narrower conclusion: this is a speed-oriented, quality-checked single-Spark setup for the published workload, with fast short-message first-token latency and a measured cold-prefill improvement from the adaptive scheduler. No published matched comparison I found establishes a faster setup with this **exact RadixArk snapshot, BF16 KV, FP32 recurrent state, and unchanged side-layer precision**; equally, the data do **not** prove this setup is fastest. A defensible ranking would require running the same prompts, output lengths, checkpoint, context settings, and measurement protocol on each candidate, with repeated boots and output-quality checks.

## Raw files

- [`baseline-4096-a.json`](../data/baseline-4096-a.json)
- [`adaptive4096-a.json`](../data/adaptive4096-a.json)
- [`recheck-20260924-a.json`](../data/recheck-20260924-a.json)

These files include the generated completions from controlled benchmark prompts; they contain no model weights, API key, or user conversation transcript.
