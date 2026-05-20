package main

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestBuildVLLMArgsRequiresModel(t *testing.T) {
	_, err := buildVLLMArgs("8081", nil, mapEnv(map[string]string{}))
	if err == nil || !strings.Contains(err.Error(), "MODEL_ID") {
		t.Fatalf("err = %v, want MODEL_ID error", err)
	}
}

func TestBuildVLLMArgsBuildsServeCommand(t *testing.T) {
	got, err := buildVLLMArgs("8081", []string{"--max-num-seqs", "8"}, mapEnv(map[string]string{
		"MODEL_ID":                     "Qwen/Qwen3-0.6B",
		"VLLM_TOKENIZER":               "Qwen/Qwen3-0.6B",
		"VLLM_HF_CONFIG_PATH":          "Qwen/Qwen3-0.6B",
		"VLLM_LOAD_FORMAT":             "gguf",
		"VLLM_SERVED_MODEL_NAME":       "qwen3",
		"VLLM_DOWNLOAD_DIR":            "/root/.cache/huggingface",
		"VLLM_TENSOR_PARALLEL_SIZE":    "2",
		"VLLM_MAX_MODEL_LEN":           "32768",
		"VLLM_MAX_NUM_SEQS":            "4",
		"VLLM_MAX_NUM_BATCHED_TOKENS":  "131072",
		"VLLM_GPU_MEMORY_UTILIZATION":  "0.9",
		"VLLM_KV_CACHE_DTYPE":          "turboquant_k8v4",
		"VLLM_KV_CACHE_MEMORY_BYTES":   "20G",
		"VLLM_CPU_OFFLOAD_GB":          "4",
		"VLLM_SPECULATIVE_CONFIG":      `{"method":"mtp","model":"draft","num_speculative_tokens":4}`,
		"VLLM_ADDITIONAL_CONFIG":       `{"foo":true}`,
		"VLLM_DTYPE":                   "auto",
		"VLLM_QUANTIZATION":            "fp8",
		"VLLM_TRUST_REMOTE_CODE":       "on",
		"VLLM_ENABLE_PREFIX_CACHING":   "on",
		"VLLM_ENABLE_CHUNKED_PREFILL":  "off",
		"VLLM_KV_SHARING_FAST_PREFILL": "yes",
	}))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	want := []string{
		"serve", "Qwen/Qwen3-0.6B",
		"--host", "127.0.0.1",
		"--port", "8081",
		"--tokenizer", "Qwen/Qwen3-0.6B",
		"--hf-config-path", "Qwen/Qwen3-0.6B",
		"--load-format", "gguf",
		"--served-model-name", "qwen3",
		"--download-dir", "/root/.cache/huggingface",
		"--tensor-parallel-size", "2",
		"--max-model-len", "32768",
		"--max-num-seqs", "4",
		"--max-num-batched-tokens", "131072",
		"--gpu-memory-utilization", "0.9",
		"--kv-cache-dtype", "turboquant_k8v4",
		"--kv-cache-memory-bytes", "20G",
		"--cpu-offload-gb", "4",
		"--speculative-config", `{"method":"mtp","model":"draft","num_speculative_tokens":4}`,
		"--additional-config", `{"foo":true}`,
		"--quantization", "fp8",
		"--trust-remote-code",
		"--enable-prefix-caching",
		"--no-enable-chunked-prefill",
		"--kv-sharing-fast-prefill",
		"--max-num-seqs", "8",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args = %#v want %#v", got, want)
	}
}

func TestBuildVLLMArgsSkipsUnsetOptionalFlags(t *testing.T) {
	got, err := buildVLLMArgs("8081", nil, mapEnv(map[string]string{
		"MODEL_ID":               "Qwen/Qwen3-0.6B",
		"VLLM_DTYPE":             "auto",
		"VLLM_QUANTIZATION":      "none",
		"VLLM_TRUST_REMOTE_CODE": "off",
	}))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	want := []string{
		"serve", "Qwen/Qwen3-0.6B",
		"--host", "127.0.0.1",
		"--port", "8081",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args = %#v want %#v", got, want)
	}
}

func TestAuthHandlerRejectsMissingBearer(t *testing.T) {
	h := authHandler("secret", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("downstream handler should not run")
	}))
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d want 401", rr.Code)
	}
}

func TestAuthHandlerAcceptsRightBearer(t *testing.T) {
	h := authHandler("secret", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer secret")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK || rr.Body.String() != "ok" {
		t.Fatalf("status=%d body=%q", rr.Code, rr.Body.String())
	}
}

func TestWaitForUpstreamReturnsWhenListenerOpens(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := waitForUpstream(ctx, ln.Addr().String(), time.Second); err != nil {
		t.Fatalf("wait: %v", err)
	}
}

func mapEnv(values map[string]string) func(string) string {
	return func(key string) string {
		return values[key]
	}
}
