FROM qwen38-flash-dgx:kernel-20260918
LABEL local.speed.experiment="exact bounded raw-byte PLE row cache" \
      local.speed.precision-changes="none"
COPY vllm_ple_mmap.original.py /opt/llm/ple-row-cache/vllm_ple_mmap.original.py
RUN cmp /opt/llm/ple-row-cache/vllm_ple_mmap.original.py /usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py
COPY vllm_ple_mmap.py ple_row_cache.py /usr/local/lib/python3.12/dist-packages/
RUN python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py /usr/local/lib/python3.12/dist-packages/ple_row_cache.py
