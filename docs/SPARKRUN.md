# Run with SparkRun

This is a single-node SparkRun v2 recipe for a DGX Spark / GB10 with 128 GB unified memory. It uses the pinned public vLLM preview image and a [SparkRun mod](../recipes/mods/qwen38-source-overlay/) that installs the captured source and compiles the deterministic kernel before serving. SparkRun downloads the pinned [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) snapshot. The mod includes the exact 47,149-row draft vocabulary and adaptive scheduler used in the recorded deployment; no machine-specific mounts or separately published image are required. Model weights are not included in this repository.

Install [SparkRun](https://sparkrun.dev/getting-started/installation/) on the control machine and configure SSH access to the Spark. Docker and an NVIDIA-compatible runtime must be present on the Spark. Reserve roughly 150 GB of free disk for the model download plus the container image. The model's PLE table is read from storage at runtime, so local NVMe is recommended.

```sh
sparkrun registry add https://github.com/salmanarshad321/qwen38-flash-next-spark-recipe.git
sparkrun run @qwen38-flash-next-spark/qwen38-flash-next-nvfp4 -H YOUR_SPARK_HOST --tp 1 --dry-run --trust
sparkrun run @qwen38-flash-next-spark/qwen38-flash-next-nvfp4 -H YOUR_SPARK_HOST --tp 1 --trust
```

Review the [recipe YAML](../recipes/qwen38-flash-next-nvfp4.yaml) and [mod script](../recipes/mods/qwen38-source-overlay/run.sh) before using `--trust`. The mod changes installed Python files and compiles a CUDA extension inside the container, so initial startup takes longer. The recipe explicitly uses root inside a non-privileged container, host IPC and networking, and clears the base image's `vllm serve` entrypoint so SparkRun can pass its generated command. These match the original container's launch environment. The default host port is 8000; override it with `--port` if needed. Running it on a Spark already serving this model will conflict with the existing port and compete for GPU memory.

The [registry manifest](../.sparkrun/registry.yaml) lets SparkRun discover this recipe by name. Alternatively, clone the repo and pass the recipe file path to `sparkrun run`. SparkRun pins the model download to `model_revision`, and the command also passes that SHA to vLLM as `--revision` so offline cache lookup uses the same revision.

## Rebuild the image

On an arm64 DGX Spark with Docker:

```sh
docker build -f image/Dockerfile -t qwen38-flash-next-spark-recipe:as-run-20260924 .
```

The base image is pinned by digest, the Dockerfile compiles the deterministic GB10 kernel from the included source, and the build fails if a packaged source file differs from the [live-source manifest](../recipes/mods/qwen38-source-overlay/LIVE_SOURCE_SHA256SUMS). See the [source map](../image/SOURCE_MAP.md) and [attribution](../NOTICE.md). This reconstructs the serving source from the live image, rather than reproducing historical Docker layers byte for byte. The SparkRun recipe uses the mod path above, so users do not need to build this image.

## Verification scope

The reconstructed image was built on the original DGX Spark. Its 29 source-file hashes matched the running image; the deterministic kernel compiled and loaded. The SparkRun mod was tested separately in a disposable container from the pinned base. `sparkrun recipe validate --strict` and a one-host dry run passed with SparkRun v0.3.9. The dry run does not download weights or start a second server. A full serve and matched benchmark of the mod-built runtime remain necessary before transferring the original image's performance claims to this artifact. SparkRun's generic memory estimate does not account for this setup's NVMe PLE offload.
