# Isolated QSA prefill query-packing experiment

This directory is a candidate, not a deployment. No remote GPU tests have been
run during preparation. The only expected benefit is faster index scoring in
prefill; there is no claim of faster decode or whole-request throughput.

`qsa_ops.baseline.py` is an exact copy of the September 18 deployed source.
`qsa_ops.candidate.py` changes only `qsa_mqa_paged`'s kernel dispatch and appends
two helpers. All 15 other existing functions, module constants and imports have
identical ASTs, including deterministic top-k and output-gate fusion. The fallback
helper is also AST-identical to the original scorer body after removal of its
program-id row assignment.

## Provenance

- Original upstream PR: https://github.com/lovablelabs/vllm/pull/1
- Pinned commit: `6b4ffca2808d92f304059fcd1a81691c18633e2b`
- Pinned source: https://raw.githubusercontent.com/lovablelabs/vllm/6b4ffca2808d92f304059fcd1a81691c18633e2b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
- Upstream full-source SHA256: `392b3e3e1dacd33299a1c987087f8fc2c33c5fc39a1d4ee659971fdd09d002ac`
- Local baseline SHA256: `1d6c3852243bb3ca66c7a2bf942b7aba8f301513c2edd1871f94353f0bae06a0`
- Candidate SHA256: `5c78af90db5478b727910cebc33d8930acf96163d8d9c2cb70c79bd312afcbdc`
- Both appended helper functions were copied verbatim from that upstream source.
- Apache-2.0/SPDX attribution remains in the source.

The upstream experiment used B200 and the same historical vLLM version,
`0.1.dev20073+g8e685d198`. Its exact score checks passed, but the complete model
comparison was rejected by its strict response/tool-contract gate: ten failures
occurred across both baseline and candidate. That does not establish kernel
causality, and it also does not establish model safety. GB10 execution and a
fresh paired full-model quality/speed gate are required before promotion.

## Dispatch contract

Packing is enabled only for 511 query rows, four heads, head dimension 128,
BF16 queries and compressed keys, compression ratio four, compressed page size
400, and 65,600 score columns. Four rows are packed only if they belong to the
same valid request. Mixed groups, invalid requests, and the final three rows use
the original arithmetic through the fallback helper. All other tensor shapes
use the unchanged original kernel.

`VLLM_QSA_QUERY_PACKING=0` disables the new dispatch for diagnosis. The default
inside this candidate is enabled. This is a source/launch-time switch: a process
restart or recompilation is needed to change an already captured graph.

The 128 MiB score workspace and native padded 262,400-token geometry give
511-row score chunks and 65,600 columns, matching the gate. Verify these actual
runtime shapes before attributing model performance to this candidate.

## Tests

CPU preparation checks:

```sh
python3 gate.py --source-only
python3 -m py_compile gate.py qsa_ops.baseline.py qsa_ops.candidate.py
```

Run these inside the exact production image in an isolated GPU test process,
with this directory mounted read-only, and sufficient temporary CUDA memory:

```sh
python3 gate.py --screen
python3 gate.py --benchmark
```

The full gate compares every defined score bit against the original kernel,
checks visibility independently with integer arithmetic, verifies repeated calls
and actual dispatch, and checks ordered indices through the retained deterministic
top-k library. It exercises native 262,144-token context, mixed requests, invalid
pages/requests, row tails, zero visibility, strided views, ties, signed zeros,
narrow/large values, and an FP32 addition-order counterexample. CUDA-graph replay
is checked after mutating query, cache, page tables, request IDs, positions and
sequence lengths. Negative dispatch controls cover rows, columns, heads,
dimensions, dtype, page size and compression ratio.

`--skip-selection` is an explicitly weaker diagnostic mode for environments that
lack `/opt/llm/kernel-det/_C_det.so`; do not treat it as the complete gate.
`VLLM_QSA_DET_LIB` can point to the exact deployed library if its mount differs.

Optional timing measures 511-row **scorer-only** CUDA graph replays, alternating
baseline/candidate order over eight paired groups. Results are JSONL. A scorer
win still needs matched cold/warm request tests and production traffic exclusion.
No finite functional suite proves unchanged behavior on every possible input.
