# SmoothNAS plugin: vLLM with bearer-auth wrapper.
#
# The final image layers a small Go reverse proxy/auth gate in front of
# vLLM's official OpenAI-compatible server image.

ARG VLLM_BASE=vllm/vllm-openai:latest

FROM golang:1.24-alpine AS wrapper-build
WORKDIR /src
COPY wrapper/go.mod wrapper/main.go ./
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /smoothnas-vllm-wrapper .

FROM ${VLLM_BASE}

ARG VLLM_ROCM_TORCH_WHEEL=https://wheels.vllm.ai/rocm/ad7125a431e176d4161099480a66f0169609a690/torch-2.10.0%2Bgit8514f05-cp312-cp312-manylinux_2_35_x86_64.whl
ARG VLLM_ROCM_TRITON_WHEEL=https://wheels.vllm.ai/rocm/ad7125a431e176d4161099480a66f0169609a690/triton-3.6.0-cp312-cp312-manylinux_2_35_x86_64.whl
ARG VLLM_ROCM_TRITON_KERNELS_WHEEL=https://wheels.vllm.ai/rocm/ad7125a431e176d4161099480a66f0169609a690/triton_kernels-1.0.0-py3-none-any.whl
ARG VLLM_ROCM_VLLM_WHEEL=https://wheels.vllm.ai/rocm/ad7125a431e176d4161099480a66f0169609a690/vllm-0.21.0%2Brocm722-cp312-cp312-manylinux_2_34_x86_64.whl

RUN if [ -d /opt/rocm-7.2.2 ]; then \
      set -eux; \
      python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
        "${VLLM_ROCM_TORCH_WHEEL}" \
        "${VLLM_ROCM_TRITON_WHEEL}" \
        "${VLLM_ROCM_TRITON_KERNELS_WHEEL}" \
        "${VLLM_ROCM_VLLM_WHEEL}"; \
      mkdir -p /tmp/rocm-debs; \
      cd /tmp/rocm-debs; \
      apt-get update; \
      apt-get download comgr miopen-hip rccl rocrand rocsolver rocsparse; \
      for deb in ./*.deb; do dpkg-deb -x "${deb}" /; done; \
      ldconfig; \
      export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"; \
      test -s /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so; \
      test -s /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_hip.so; \
      test -s /opt/rocm/lib/libMIOpen.so.1; \
      test -s /opt/rocm/lib/librocsolver.so.0; \
      python3 -c 'import torch, triton, vllm; print("torch", torch.__version__, "hip", torch.version.hip); print("triton", getattr(triton, "__version__", "unknown")); print("vllm", getattr(vllm, "__version__", "unknown"))'; \
      rm -rf /tmp/rocm-debs /var/lib/apt/lists/*; \
    fi

COPY --from=wrapper-build /smoothnas-vllm-wrapper /usr/local/bin/smoothnas-vllm-wrapper

ENV VLLM_BIN=vllm
ENV VLLM_PORT=8081
ENV LISTEN_ADDR=:8080
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH}

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/smoothnas-vllm-wrapper"]
