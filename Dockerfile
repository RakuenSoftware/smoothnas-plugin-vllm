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

RUN if [ -d /opt/rocm/lib ]; then \
      set -eux; \
      mkdir -p /tmp/rocm-debs; \
      cd /tmp/rocm-debs; \
      apt-get update; \
      apt-get download hipblaslt; \
      for deb in ./*.deb; do dpkg-deb -x "${deb}" /; done; \
      ldconfig; \
      export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"; \
      python3 -c 'import importlib.util, pathlib, shutil; spec = importlib.util.find_spec("flash_attn"); locs = list(spec.submodule_search_locations or []) if spec else []; [shutil.move(str(p), str(p.with_name(p.name + ".disabled"))) for p in map(pathlib.Path, locs) if p.exists() and not p.with_name(p.name + ".disabled").exists()]'; \
      test -s /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so; \
      test -s /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_hip.so; \
      test -s /opt/rocm/lib/libMIOpen.so.1; \
      test -s /opt/rocm/lib/librocroller.so.1; \
      test -s /opt/rocm/lib/librocsolver.so.0; \
      python3 -c 'import importlib.util; assert importlib.util.find_spec("flash_attn") is None'; \
      python3 -c 'import torch, triton, triton.backends, vllm; import vllm._rocm_C; print("torch", torch.__version__, "hip", torch.version.hip); print("triton", getattr(triton, "__version__", "unknown")); print("vllm", getattr(vllm, "__version__", "unknown"))'; \
      rm -rf /tmp/rocm-debs /var/lib/apt/lists/*; \
    fi

COPY --from=wrapper-build /smoothnas-vllm-wrapper /usr/local/bin/smoothnas-vllm-wrapper

ENV VLLM_BIN=vllm
ENV VLLM_PORT=8081
ENV LISTEN_ADDR=:8080
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH}

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/smoothnas-vllm-wrapper"]
