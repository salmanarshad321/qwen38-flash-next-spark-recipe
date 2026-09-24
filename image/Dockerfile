# Reconstructed from the running 2026-09-24 image's installed source files.
# The pinned official vLLM image is the same base used by blazux at b6dae9f.
FROM vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e

LABEL org.opencontainers.image.source="https://github.com/salmanarshad321/qwen38-flash-next-spark-recipe" \
      org.opencontainers.image.description="Qwen3.8-Flash-Next NVFP4 on one DGX Spark; reconstructed as-run source snapshot" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

ARG SP=/usr/local/lib/python3.12/dist-packages

# These are the exact installed Python sources differing from the official base.
# See SOURCE_MAP.md and the mod's LIVE_SOURCE_SHA256SUMS for attribution and verification.
COPY recipes/mods/qwen38-source-overlay/runtime-overrides/vllm/ ${SP}/vllm/
COPY recipes/mods/qwen38-source-overlay/runtime-overrides/root/ ${SP}/

# Build the GB10 deterministic QSA extension from its distributed source.
COPY recipes/mods/qwen38-source-overlay/kernel-src/ /opt/llm/kernel-det/src/
RUN cd /opt/llm/kernel-det/src \
 && DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=121a MAX_JOBS=2 python3 build_det.py \
 && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so

# These two runtime mounts from the original deployment are bundled here.
COPY recipes/mods/qwen38-source-overlay/runtime/adaptive_scheduler.py /opt/qwen-adaptive/adaptive_scheduler.py
COPY recipes/mods/qwen38-source-overlay/runtime/draft_vocab_en_code_47k.txt /draft-vocab/draft_vocab_en_code_47k.txt
ENV PYTHONPATH=/opt/qwen-adaptive \
    VLLM_MTP_DRAFT_VOCAB=/draft-vocab/draft_vocab_en_code_47k.txt

COPY recipes/mods/qwen38-source-overlay/LIVE_SOURCE_SHA256SUMS /opt/qwen-source/LIVE_SOURCE_SHA256SUMS
COPY recipes/mods/qwen38-source-overlay/verify_live_source.py /opt/qwen-source/verify_live_source.py
RUN python3 /opt/qwen-source/verify_live_source.py \
 && python3 -m compileall -q \
      ${SP}/vllm/models/qwen3_8_flash_next \
      ${SP}/vllm_ple_mmap.py \
      ${SP}/vllm_ple_staging.py \
      /opt/qwen-adaptive/adaptive_scheduler.py
