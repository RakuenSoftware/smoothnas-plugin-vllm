# smoothnas-plugin-vllm

SmoothNAS plugin for [vLLM](https://github.com/vllm-project/vllm) on NVIDIA
CUDA and AMD ROCm.
It runs vLLM's OpenAI-compatible API inside a SmoothNAS-managed LXC system
container with GPU passthrough, tier-bound Hugging Face cache storage, and
bearer-injected auth from the SmoothNAS UI.

## Variant

| Manifest | Image tag | Profiles | Use when |
|----------|-----------|----------|----------|
| `smoothnas-plugin.yaml` | `ghcr.io/rakuensoftware/smoothnas-plugin-vllm:VER-cuda` | `gpu-nvidia` | Host has an NVIDIA GPU with CUDA support |
| `smoothnas-plugin-rocm.yaml` | `ghcr.io/rakuensoftware/smoothnas-plugin-vllm:VER-rocm` | `gpu-amd`, `rocm-runtime` | Host has an AMD GPU with ROCm/KFD support |

The CUDA image is built on `vllm/vllm-openai`; the ROCm image is built on
`vllm/vllm-openai-rocm`.

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

First-run defaults target Gemma4 Q5 26B on the SmoothNAS runner:

- `MODEL_ID=unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q5_K_XL`
- `VLLM_TOKENIZER=google/gemma-4-26B-A4B-it`
- `VLLM_HF_CONFIG_PATH=google/gemma-4-26B-A4B-it`
- `VLLM_LOAD_FORMAT=gguf`
- `VLLM_TENSOR_PARALLEL_SIZE=1`
- `VLLM_MAX_MODEL_LEN=65536`
- `VLLM_MAX_NUM_SEQS=1` on ROCm for 24 GiB cards
- `VLLM_MAX_NUM_BATCHED_TOKENS=65536` on ROCm for 24 GiB cards
- `VLLM_GPU_MEMORY_UTILIZATION=0.92`
- `VLLM_KV_CACHE_DTYPE=auto` on ROCm; Gemma4 forces TRITON_ATTN and vLLM rejects `turboquant_k8v4` there
- `VLLM_CPU_OFFLOAD_GB=32` on ROCm for 24 GiB cards
- `VLLM_SPECULATIVE_CONFIG={"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}`
- `VLLM_DTYPE=float16`
- `VLLM_QUANTIZATION=none`
- `VLLM_TRUST_REMOTE_CODE=on`
- `MEMORY_LIMIT=64GiB` on ROCm when CPU offload is enabled

Gemma4 tool-use and reasoning parser flags are also exposed for operators who
need OpenAI tool-calling behavior:

- `VLLM_TOOL_CALL_PARSER=gemma4`
- `VLLM_REASONING_PARSER=gemma4`
- `VLLM_ENABLE_AUTO_TOOL_CHOICE=on`
- `VLLM_CHAT_TEMPLATE=/path/to/template.jinja`
- `VLLM_LIMIT_MM_PER_PROMPT={"image":0,"audio":0}`
- `VLLM_ASYNC_SCHEDULING=on`

The ROCm image includes a build-time repair pass for the vLLM 0.21.0 ROCm 7.2.2
stack: it reinstalls the matching PyTorch and vLLM ROCm wheels and force-extracts
the ROCm shared-library payloads that vLLM needs at runtime. This keeps SmoothNAS
installs plug-and-play instead of requiring host-level vLLM or manual container
repairs. `HSA_OVERRIDE_GFX_VERSION` is exposed for consumer AMD GPUs that need an
override. Leave it empty on supported ROCm hardware.

## Local development

```sh
cd wrapper
go test ./...
go build ./...

docker buildx build \
  --build-arg VLLM_BASE=vllm/vllm-openai:latest \
  -t smoothnas-plugin-vllm:dev-cuda .

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
