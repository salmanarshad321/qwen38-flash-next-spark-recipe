#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
sp=/usr/local/lib/python3.12/dist-packages

if [ "$(id -u)" != 0 ]; then
  echo "qwen38-source-overlay needs root inside the container" >&2
  exit 1
fi

# This mod is only qualified against the exact official preview image.
python3 "$here/verify_live_source.py" "$here/BASE_SOURCE_SHA256SUMS"

cp -R "$here/runtime-overrides/vllm/." "$sp/vllm/"
cp -R "$here/runtime-overrides/root/." "$sp/"
mkdir -p /opt/llm/kernel-det/src /opt/qwen-adaptive /draft-vocab
cp -R "$here/kernel-src/." /opt/llm/kernel-det/src/
cp "$here/runtime/adaptive_scheduler.py" /opt/qwen-adaptive/adaptive_scheduler.py
cp "$here/runtime/draft_vocab_en_code_47k.txt" /draft-vocab/draft_vocab_en_code_47k.txt

cd /opt/llm/kernel-det/src
DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=121a MAX_JOBS=2 python3 build_det.py
cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so

python3 "$here/verify_live_source.py" "$here/LIVE_SOURCE_SHA256SUMS"
python3 -m py_compile \
  "$sp/vllm/models/qwen3_8_flash_next/nvidia/model_state.py" \
  "$sp/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py" \
  "$sp/vllm_ple_mmap.py" "$sp/vllm_ple_staging.py" \
  /opt/qwen-adaptive/adaptive_scheduler.py
