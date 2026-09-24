# Deliberately has no default: review must name a currently qualified parent.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm
ARG MODEL_DIR=${VLLM_ROOT}/models/qwen3_8_flash_next/nvidia

# Fail closed unless the parent contains the exact audited old-preview sources
# and the exact-speed parent's audited row-cache mmap shim. Before overwrite.
RUN test "$(sha256sum ${MODEL_DIR}/model_state.py | cut -d' ' -f1)" = \
      "f127380fcd884c1fb7b010a2c32065d55c319f65a1268c18ef21376e39ba8425" \
 && test "$(sha256sum ${MODEL_DIR}/ple_layer.py | cut -d' ' -f1)" = \
      "6690edcd2e2b2ae309b214991a060bffbdbbcdc42334e164db0aebcc0a5b397e" \
 && test "$(sha256sum ${VLLM_ROOT}/v1/worker/gpu/model_runner.py | cut -d' ' -f1)" = \
      "f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8" \
 && test "$(sha256sum ${VLLM_ROOT}/v1/worker/gpu/input_batch.py | cut -d' ' -f1)" = \
      "3929c92e42ae90e4410bb4537dcbdcd171a662e44afc1189a9a3e19037a84410" \
 && test "$(sha256sum ${VLLM_ROOT}/config/compilation.py | cut -d' ' -f1)" = \
      "189c1312f052341d8f19ad06f436ae853a885257a2d2e3d8626c3471892051f9" \
 && test "$(sha256sum ${VLLM_ROOT}/v1/worker/gpu/cudagraph_utils.py | cut -d' ' -f1)" = \
      "ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c" \
 && test "$(sha256sum ${VLLM_ROOT}/v1/worker/gpu/spec_decode/autoregressive/speculator.py | cut -d' ' -f1)" = \
      "575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1" \
 && test "$(sha256sum ${VLLM_ROOT}/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py | cut -d' ' -f1)" = \
      "13392ef0a9ed59eb9c2b2bad43d7c61beb212a8805849e46b5130c230275d0a5" \
 && test "$(sha256sum ${MODEL_DIR}/mtp.py | cut -d' ' -f1)" = \
      "6a99a400ab640bb48ca394ae57017adadaff9c250375caa2ada4fe1a221c22bd" \
 && test "$(sha256sum ${VLLM_ROOT}/v1/executor/uniproc_executor.py | cut -d' ' -f1)" = \
      "3cb039b80fe9b03dc1122f93c1f8a4098ef1355104b317ff6df25358f4349494" \
 && test "$(sha256sum ${VLLM_ROOT}/v1/worker/gpu_worker.py | cut -d' ' -f1)" = \
      "923767cea120a027ca36683bd48b7659eadbfa4cbd4acbcf52a80cee0c0a0ec4" \
 && test "$(sha256sum ${VLLM_ROOT}/v1/worker/gpu/model_states/mamba_hybrid.py | cut -d' ' -f1)" = \
      "e2b54a2c06f14b38fa4476d193d174ce0124574fd12bda0b24fca6444dfe8c87" \
 && test "$(sha256sum /usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py | cut -d' ' -f1)" = \
      "f8ab8f4f082ea9e5a4257b96948d887d800165aa0f8d44fe7a7223387b6faa0e"

COPY vllm_ple_staging.py /usr/local/lib/python3.12/dist-packages/vllm_ple_staging.py
COPY model_state_patched.py ${MODEL_DIR}/model_state.py

RUN python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/vllm_ple_staging.py \
      ${MODEL_DIR}/model_state.py

# Intentionally no ENV directives. The installed code is inert unless the
# launcher explicitly supplies VLLM_PLE_STAGED=1; the runtime then also demands
# VLLM_PLE_MMAP=1, VLLM_USE_V2_MODEL_RUNNER=1, TP1/PP1/DP1, the pinned
# 4096/8192-token, 8-sequence, MTP-2/3/4 scheduler and unchanged precision.
# Preferred trial: mode 3/FULL_AND_PIECEWISE, retaining production PW sizes.
# Control only: mode 0/FULL_DECODE_ONLY. Neither is GPU-qualified here.
