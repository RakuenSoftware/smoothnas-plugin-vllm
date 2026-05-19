# smoothnas-plugin-vllm

SmoothNAS plugin for [vLLM](https://github.com/vllm-project/vllm) on AMD ROCm.
It runs vLLM's OpenAI-compatible API inside a SmoothNAS-managed LXC system
container with AMD GPU passthrough, tier-bound Hugging Face cache storage, and
bearer-injected auth from the SmoothNAS UI.

## Variant

| Manifest | Image tag | Profiles | Use when |
|----------|-----------|----------|----------|
| `smoothnas-plugin.yaml` | `ghcr.io/rakuensoftware/smoothnas-plugin-vllm:VER-rocm` | `gpu-amd`, `rocm-runtime` | Host has an AMD GPU with ROCm/KFD support |

The image is built on the official `vllm/vllm-openai-rocm` base image. That is
the current upstream-recommended image for ROCm; older AMD `rocm/vllm` images
are deprecated upstream.

## Why a wrapper image?

vLLM exposes an OpenAI-compatible HTTP API, but SmoothNAS plugins must enforce
the per-plugin bearer token that tierd injects through nginx. The wrapper:

1. Starts `vllm serve` on `127.0.0.1:8081`
2. Listens on `:8080`, the port SmoothNAS proxies
3. Validates `Authorization: Bearer $SMOOTHNAS_BEARER_EXPECTED`
4. Streams valid requests through to vLLM

`SMOOTHNAS_BEARER_EXPECTED` is auto-populated by SmoothNAS and rotated through
the plugin token flow. Operators should not edit it manually.

## Operator workflow

In the SmoothNAS UI:

1. Install the vLLM plugin from the catalog and choose a tier with NVME capacity
2. Select the AMD render node, or leave the GPU field automatic
3. Set `MODEL_ID` to a Hugging Face model id
4. Set `HF_TOKEN` if the model is gated
5. Start the plugin and open `/plugins/vllm/`

Conservative first-run defaults:

- `MODEL_ID=Qwen/Qwen3-0.6B`
- `VLLM_TENSOR_PARALLEL_SIZE=1`
- `VLLM_MAX_MODEL_LEN=32768`
- `VLLM_GPU_MEMORY_UTILIZATION=0.90`
- `VLLM_DTYPE=auto`
- `VLLM_QUANTIZATION=none`
- `VLLM_TRUST_REMOTE_CODE=off`
- `MEMORY_LIMIT=64GiB`

`HSA_OVERRIDE_GFX_VERSION` is exposed for consumer AMD GPUs that need an
override. Leave it empty on supported ROCm hardware.

## Local development

```sh
cd wrapper
go test ./...
go build ./...

docker buildx build \
  --build-arg VLLM_BASE=vllm/vllm-openai-rocm:latest \
  -t smoothnas-plugin-vllm:dev-rocm .
```

To sideload a dev image into a SmoothNAS dev host, edit `artifact.image` in
`smoothnas-plugin.yaml` to your local tag and install the manifest.

## Release flow

`.github/workflows/release.yml` runs on tag push (`v*`):

1. Runs wrapper tests
2. Builds and pushes the ROCm wrapper image to GHCR
3. Rewrites `smoothnas-plugin.yaml` with the released version and image digest
4. Attaches the installable manifest and `index.json` to the GitHub release

`release-please` opens release PRs from conventional commits on `main`.
