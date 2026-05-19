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

func TestBuildVLLMArgsBuildsROCmServeCommand(t *testing.T) {
	got, err := buildVLLMArgs("8081", []string{"--max-num-seqs", "8"}, mapEnv(map[string]string{
		"MODEL_ID":                    "Qwen/Qwen3-0.6B",
		"VLLM_TENSOR_PARALLEL_SIZE":   "2",
		"VLLM_MAX_MODEL_LEN":          "32768",
		"VLLM_GPU_MEMORY_UTILIZATION": "0.9",
		"VLLM_DTYPE":                  "auto",
		"VLLM_QUANTIZATION":           "fp8",
		"VLLM_TRUST_REMOTE_CODE":      "on",
	}))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	want := []string{
		"serve", "Qwen/Qwen3-0.6B",
		"--host", "127.0.0.1",
		"--port", "8081",
		"--tensor-parallel-size", "2",
		"--max-model-len", "32768",
		"--gpu-memory-utilization", "0.9",
		"--quantization", "fp8",
		"--trust-remote-code",
		"--max-num-seqs", "8",
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
