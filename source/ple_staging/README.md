# Isolated staged PLE / full-decode experiment

Status: local source/CPU gates pass. GPU and model qualification belong to the
parent experiment; this directory does not deploy or launch anything.

The only installed replacements are `vllm_ple_staging.py` and the generated
`nvidia/model_state.py`. No model weights, FP32 recurrent state, BF16 KV,
existing BF16 47,149-row draft head, PLE scale, PLE table representation,
short-conv, QSA, GDN, prefix-cache, scheduler async, or MTP verifier is changed.
Only the current exact-speed row-cache shim is hash-qualified; no historical
allowlist or shim replacement is installed here.

## Design and local differences

Input preparation calls the existing saved PLE hash/gather implementation on
the SAME padded token IDs, query boundaries and n-gram context as production.
Every output byte, INCLUDING padded embeddings, is copied to a single stable
raw-FP8 buffer. Active model forward returns a slice of this buffer. Therefore
the synchronous table lookup stays outside CUDA capture, while the existing
dequantization/projections/short-conv remain unchanged. The buffer is 20 MiB at
the unchanged 8,192-token capacity (10 MiB at the separately supported 4,096).

This deliberately differs from upstream actual-only lookup/zero-padded-tail.
No correctness claim here relies on padded embedding rows being masked out.
Only profile/capture dummy steps zero their entire extent without table reads.
Inactive installation keeps the old forward intact.

The model-state patch adds imports, one bind call, one live/profile staging
call and one capture-dummy call. Its source is byte-identical to production
SHA `f127380fcd884c1fb7b010a2c32065d55c319f65a1268c18ef21376e39ba8425`.
The generator proves unchanged anchors and generated-file reproducibility.

## Preferred graph trial (keep every other production argument)

Set `VLLM_PLE_STAGED=1` and explicitly `VLLM_USE_V2_MODEL_RUNNER=1`.
Keep `VLLM_PLE_MMAP=1`, existing `VLLM_FP8_HYBRID=0`, MTP3, max sequences8,
buffer capacity8,192, scheduled-token budget4,096, prefix caching and every
current splitting op. Change ONLY compilation graph settings:

```json
{
  "mode": 3,
  "cudagraph_mode": "FULL_AND_PIECEWISE",
  "cudagraph_capture_sizes": [1,2,3,4,5,6,7,8,12,16,20,24,28,32,40,48,56,64],
  "max_cudagraph_capture_size": 64
}
```

Supply these as separate `-cc.*` flags or merge the existing splitting list
into the config object; do not replace the complete current config blindly.
The adapter fails if any qualified splitting boundary is missing, including
`vllm::qwen3_8_flash_next_ple_short_conv`, the current QSA and GDN operators.

Expected descriptor inventory: target8 FULL +18 PW, draft-position0 eight
FULL +18 PW, follow-on draft8 FULL =24 FULL +36 PW. Compiler partition artifact
counts can be larger. FULL target and draft-position0 widths are4,8,...32;
follow-on draft widths1,...8. The pinned manager caps FULL independently of
the retained PIECEWISE maximum64. Larger/mixed prefills keep mode3 compilation.

Explicit diagnostic fallback only: mode0/FULL_DECODE_ONLY, exact capture union
`[1,2,3,4,5,6,7,8,12,16,20,24,28,32]`, maximum32. It may regress prefill and
must not be promoted on decode alone. Plain FULL (captured prefills) is refused.

## Additional fail-closed guards

- Exact pinned graph/config/speculator/MTP/executor/worker/hybrid-state sources.
- Resolved SSM dtype must be `float32`, model dtype BF16, convolution state
  auto/BF16 and KV auto/BF16. Ambiguous SSM `auto` is refused; the Qwen config
  updater resolves the checkpoint's FP32 value before model-state binding.
- UniProc only, TP=PP=DP=1, DBO and ubatching off, fixed MTP depth and one bonus.
- Exactly one registered PLE module, identical to its graph-visible embedding;
  complete original FP8 table and retained scalar scale; exact workspace sizes.
- Stable buffer address; preparation outside capture; first-live CUDA stream
  and host-thread identities must remain unchanged.
- Profile dummy does NOT mark graph validation final. Every capture dummy
  revalidates, and first LIVE preparation independently verifies final settings.

Async scheduling stays enabled: in the pinned UniProc executor, worker calls
are inline/serial; async output copies use a different stream but do not read
the PLE staging buffer. A different executor or forward stream is unsupported.

## Source provenance and limits

Adapted from Apache-2.0 source at
[pangoleen commit5ac0ac951e66b1239eb65967ba609d0b12349cdf](https://github.com/pangoleen/qwen3.8-flash-next-dgx-spark/tree/5ac0ac951e66b1239eb65967ba609d0b12349cdf/build/03-staged-ple).
Its exact old-preview runtime is vLLM `0.1.dev20073+g8e685d198`, matching ours;
eight core source hashes matched our installed files before this adaptation.
The current MTP and mmap pins differ intentionally and were reviewed locally.
Upstream mode0 staging results were combined with BF16 recurrent-state and
other changes; those performance/quality claims do NOT transfer to this trial.

The upstream staged README ultimately excludes prefix caching. Later repository
results report successful cache tests but retain an unresolved historical
wrong-topic incident. We do not disable caching or claim that bug was fixed.
Our exact runner's existing GPU `align` pre-copy, request-slot reset and state
snapshot paths remain untouched. The current `disable_eagle_block_drop=true`
setting remains unchanged; its presence alone does not certify every cache case.

[The separate-fork PLE graph corruption PR](https://github.com/local-inference-lab/vllm/pull/600)
concerns missing piecewise boundaries; our current short-conv boundary already
exists and is explicitly required. Its old operator names are not transplanted.

## Validation

CPU-only, no Torch/vLLM imports or GPU activity:

```text
python3 test_source.py
python3 test_staging.py
python3 test_graph_contract.py
python3 test_hardened.py
```

`gate_cuda.py` is an EXCLUSIVE GPU-window test to be run by the parent, not a
model benchmark. Mount this directory at `/gate:ro`, model cache at `/hf:ro`,
set `PYTHONPATH=/gate`, use the exact parent image and invoke:

```text
python3 /gate/gate_cuda.py --model /hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
```

It SHA-checks and executes the actual unchanged production PLE constructor,
hash bodies and GPU history-gather body, reads REAL checkpoint FP8 rows through
the installed mmap shim, and compares every live+padded raw byte with the
production padded invocation. It tests EOS boundaries, zero-length rows,
request-slot permutations, native-context history offsets, rejected speculative
suffixes and rollback1..3, 4k/8k extents, stable storage and unchanged scale.
Raw and Inductor-compiled CUDA replay must observe mutated buffer bytes at all
18 capture widths. `--skip-compile` is a screen only, not a complete gate.

The harness imports no full model and uploads no large weights. Stable staging
uses20 MiB and request history about16 MiB, plus bounded temporary tensors,
compiler/CUDA overhead and the existing CPU row-cache budget. Raw-table mmap
may populate OS page cache; no whole-table prewarm is requested.

Passing this gate does not validate whole-model CUDA capture. Before retention:
verify final graph modes/inventory and actual recurrent-cache tensor dtypes;
run paired quality and timings, long-prefill regression checks, native long
context, MTP acceptance, cache hit/miss and block-boundary±1, accepted-token
rollback across a boundary, new requests after long prefills, cancellation/slot
reuse, mixed prefill/decode, all concurrency counts1–8, vision/tools/reasoning,
and an independent reload. Require meaningful decode improvement and no major
prefill regression. Roll back to the unchanged exact-speed parent on failure.

## Build layout

Build context is THIS directory. Dockerfile requires an explicitly named
`BASE_IMAGE`, checks parent source hashes, and copies only the two overlay
files. It adds no activation ENV, weight copy or precision setting. Runtime
compiler caches must remain private to this image/graph trial.
