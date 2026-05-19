# SmoothNAS plugin: vLLM ROCm with bearer-auth wrapper.
#
# The final image layers a small Go reverse proxy/auth gate in front of
# vLLM's official ROCm OpenAI-compatible server image.

ARG VLLM_BASE=vllm/vllm-openai-rocm:latest

FROM golang:1.24-alpine AS wrapper-build
WORKDIR /src
COPY wrapper/go.mod wrapper/main.go ./
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /smoothnas-vllm-wrapper .

FROM ${VLLM_BASE}

COPY --from=wrapper-build /smoothnas-vllm-wrapper /usr/local/bin/smoothnas-vllm-wrapper

ENV VLLM_BIN=vllm
ENV VLLM_PORT=8081
ENV LISTEN_ADDR=:8080

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/smoothnas-vllm-wrapper"]
