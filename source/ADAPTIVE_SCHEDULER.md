# Isolated adaptive prefill-budget candidate

Prepared September 21, 2026 against the running
`qwen38-trial-staged-speed-20260919` image. This directory does not deploy or
start a server. No performance or model-quality result is claimed.

The candidate inherits the installed `AsyncScheduler` and changes only
`self.max_num_scheduled_tokens` immediately before each inherited `schedule` call:

- 8192 when all running requests are still prefilling, or there are no running requests;
- 4096 when any running request is decoding;
- optional low budget 2048, selected once at initialization by
  `QWEN_ADAPTIVE_PREFILL_LOW=2048` (default is 4096).

It forwards `throttle_prefills` unchanged, returns the inherited result, preserves
inherited exceptions, and does not log on individual scheduling steps. Inherited
request ordering, token acceptance, async output handling, KV allocation,
preemption, prefix-cache logic and block alignment remain in place. The static
configuration remains 8192; only the scheduler's current-step field changes.

## Runtime integration

Make `adaptive_scheduler.py` importable in an isolated clone, for example with a
read-only directory mount at `/opt/qwen-adaptive` and that directory prepended to
the existing `PYTHONPATH`. Preserve the current environment and all model/graph
arguments. Replace the existing scheduled-token option and add:

```text
--max-num-scheduled-tokens 8192
--scheduler-cls adaptive_scheduler.AdaptiveAsyncScheduler
--async-scheduling
```

Only `adaptive_scheduler.py` is needed at runtime. The alignment reference and
test file are CPU-only development artifacts.

The installed CLI accepts a dotted class string. `SchedulerConfig.get_scheduler_cls`
resolves it in the engine process, and `EngineCore` constructs it with keyword
arguments including `vllm_config`, `block_size` and `hash_block_size`. The string
travels in the regular configuration; this module contains no captured closure or
unserializable configuration value. It must therefore be importable in every
process using that configuration. The exact installed config has no special case
that disables async scheduling for a custom class. Subclassing `AsyncScheduler`
preserves its output-placeholder and cached-block update hooks.

Initialization verifies seven source-file SHA256 hashes from the installed
scheduler, async scheduler, request class, scheduler/speculative/runtime config
and engine constructor. It also requires:

- input capacity8192, static scheduled ceiling8192, eight sequences;
- resolved async scheduling and chunked prefill, no separate long-prefill cap;
- fixed autoregressive MTP3 with zero additional drafting slots;
- staged PLE and model runner V2;
- TP1/PP1/DP1, no DBO or diffusion, one sampled token per step;
- no dynamic speculative lookup, and a resolved Mamba block size no larger than
  the selected low budget.

Any unexpected source or configuration fails initialization. Do not weaken a pin
to make a changed runtime launch without reviewing the changed contract.

## Alignment and limitations

The base scheduler reads its token budget throughout the synchronous `schedule`
call, including `_mamba_block_aligned_split`. The candidate chooses it once before
delegation and does not mutate it during that call. The unchanged helper rounds
nonfinal chunks and honors prefix/cache junctions. Current 1600-token blocks mean
effective chunks can be 8000/3200/1600 for configured budgets8192/4096/2048.
The model's input buffers, graph descriptors, KV geometry and staging buffers stay
unchanged. Scheduler graph hashing includes input capacity and sequence count,
neither of which this experiment alters.

`is_prefill_chunk` is the flag the base scheduler itself uses to distinguish
running prefills from decoding. It is updated after scheduling with asynchronous
output placeholders included. A naive comparison against prompt length would
handle those placeholders and replayed output tokens incorrectly.

A request arriving after a large prefill chunk starts still waits for that chunk.
The policy can conservatively choose the low budget for a decode about to finish.
Chunk shape and batching can change numerical/sample trajectories even with
unchanged weights, so identical model output is not guaranteed by this wrapper.

## CPU verification and remaining gates

```sh
python3 -m unittest discover -s diagnostics/dgx-speed-20260921/trial \
  -p test_adaptive_scheduler.py -v
```

The tests load the candidate with a fake base to avoid importing Torch or vLLM.
They check inheritance/delegation, throttle forwarding, asynchronous prefill to
decode transitions, bounded settings, source-pin failure, exception propagation,
and changing budgets through the **exact extracted live Mamba alignment helper**.
That helper's provenance is recorded in `adaptive_alignment_reference.py`.

Before retention, compare against the current server and static budgets: cold and
warm-prefix TTFT, streaming gaps with competing prefills, all concurrency widths1–8,
output/function checks, long and vision prefills, cached boundaries±1, cancellation
and slot reuse, memory/preemption counters and unrelated-traffic exclusion. These
CPU checks do not qualify model execution or establish any speed benefit.
