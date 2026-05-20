package main

import (
	"context"
	"crypto/subtle"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

func main() {
	listenAddr := envOr("LISTEN_ADDR", ":8080")
	vllmBin := envOr("VLLM_BIN", "vllm")
	vllmPort := envOr("VLLM_PORT", "8081")
	expected := os.Getenv("SMOOTHNAS_BEARER_EXPECTED")
	if expected == "" {
		log.Fatal("SMOOTHNAS_BEARER_EXPECTED is empty; refusing to start without auth")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()

	upstream, err := url.Parse("http://127.0.0.1:" + vllmPort)
	if err != nil {
		log.Fatalf("parse upstream url: %v", err)
	}
	proxy := httputil.NewSingleHostReverseProxy(upstream)
	proxy.FlushInterval = -1
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		if errors.Is(err, context.Canceled) {
			return
		}
		log.Printf("proxy error %s %s: %v", r.Method, r.URL.Path, err)
		http.Error(w, "upstream unavailable", http.StatusBadGateway)
	}

	srv := &http.Server{
		Addr:              listenAddr,
		Handler:           authHandler(expected, proxy),
		ReadHeaderTimeout: 10 * time.Second,
	}
	serverDone := make(chan error, 1)
	go func() { serverDone <- srv.ListenAndServe() }()
	log.Printf("wrapper listening on %s; bearer auth required", listenAddr)

	args, err := buildVLLMArgs(vllmPort, os.Args[1:], os.Getenv)
	if err != nil {
		log.Fatalf("build vLLM args: %v", err)
	}
	cmd := exec.CommandContext(ctx, vllmBin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = childEnv(os.Environ())
	if err := cmd.Start(); err != nil {
		log.Fatalf("start %s: %v", vllmBin, err)
	}
	log.Printf("started %s pid=%d on 127.0.0.1:%s", vllmBin, cmd.Process.Pid, vllmPort)

	if err := waitForUpstream(ctx, "127.0.0.1:"+vllmPort, 60*time.Second); err != nil {
		log.Printf("upstream readiness wait: %v (proxy will keep returning 502 until ready)", err)
	}

	childDone := make(chan error, 1)
	go func() { childDone <- cmd.Wait() }()

	select {
	case err := <-childDone:
		log.Printf("upstream exited: %v", err)
	case <-ctx.Done():
		log.Printf("signal received; shutting down")
	case err := <-serverDone:
		log.Printf("listener exited: %v", err)
	}

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelShutdown()
	_ = srv.Shutdown(shutdownCtx)
	if cmd.Process != nil {
		_ = cmd.Process.Signal(syscall.SIGTERM)
		_, _ = cmd.Process.Wait()
	}
}

func buildVLLMArgs(port string, extra []string, getenv func(string) string) ([]string, error) {
	model := strings.TrimSpace(getenv("MODEL_ID"))
	if model == "" {
		return nil, fmt.Errorf("MODEL_ID is required")
	}
	args := []string{"serve", model, "--host", "127.0.0.1", "--port", port}

	appendValueFlag := func(envKey, flag string) {
		value := strings.TrimSpace(getenv(envKey))
		if value != "" {
			args = append(args, flag, value)
		}
	}
	appendTriStateFlag := func(envKey, onFlag, offFlag string) {
		value, ok := boolEnv(getenv(envKey))
		if !ok {
			return
		}
		if value {
			args = append(args, onFlag)
		} else if offFlag != "" {
			args = append(args, offFlag)
		}
	}
	appendValueFlag("VLLM_TOKENIZER", "--tokenizer")
	appendValueFlag("VLLM_HF_CONFIG_PATH", "--hf-config-path")
	appendValueFlag("VLLM_LOAD_FORMAT", "--load-format")
	appendValueFlag("VLLM_SERVED_MODEL_NAME", "--served-model-name")
	appendValueFlag("VLLM_DOWNLOAD_DIR", "--download-dir")
	appendValueFlag("VLLM_TENSOR_PARALLEL_SIZE", "--tensor-parallel-size")
	appendValueFlag("VLLM_MAX_MODEL_LEN", "--max-model-len")
	appendValueFlag("VLLM_MAX_NUM_SEQS", "--max-num-seqs")
	appendValueFlag("VLLM_MAX_NUM_BATCHED_TOKENS", "--max-num-batched-tokens")
	appendValueFlag("VLLM_GPU_MEMORY_UTILIZATION", "--gpu-memory-utilization")
	appendValueFlag("VLLM_KV_CACHE_DTYPE", "--kv-cache-dtype")
	appendValueFlag("VLLM_KV_CACHE_MEMORY_BYTES", "--kv-cache-memory-bytes")
	appendValueFlag("VLLM_CPU_OFFLOAD_GB", "--cpu-offload-gb")
	appendValueFlag("VLLM_SPECULATIVE_CONFIG", "--speculative-config")
	appendValueFlag("VLLM_TOOL_CALL_PARSER", "--tool-call-parser")
	appendValueFlag("VLLM_REASONING_PARSER", "--reasoning-parser")
	appendValueFlag("VLLM_CHAT_TEMPLATE", "--chat-template")
	appendValueFlag("VLLM_LIMIT_MM_PER_PROMPT", "--limit-mm-per-prompt")
	appendValueFlag("VLLM_ADDITIONAL_CONFIG", "--additional-config")
	if dtype := strings.TrimSpace(getenv("VLLM_DTYPE")); dtype != "" && dtype != "auto" {
		args = append(args, "--dtype", dtype)
	}
	if quant := strings.TrimSpace(getenv("VLLM_QUANTIZATION")); quant != "" && quant != "none" {
		args = append(args, "--quantization", quant)
	}
	if onOff(getenv("VLLM_TRUST_REMOTE_CODE")) {
		args = append(args, "--trust-remote-code")
	}
	appendTriStateFlag("VLLM_ENABLE_PREFIX_CACHING", "--enable-prefix-caching", "--no-enable-prefix-caching")
	appendTriStateFlag("VLLM_ENABLE_CHUNKED_PREFILL", "--enable-chunked-prefill", "--no-enable-chunked-prefill")
	appendTriStateFlag("VLLM_ENABLE_AUTO_TOOL_CHOICE", "--enable-auto-tool-choice", "")
	appendTriStateFlag("VLLM_ASYNC_SCHEDULING", "--async-scheduling", "")
	appendTriStateFlag("VLLM_KV_SHARING_FAST_PREFILL", "--kv-sharing-fast-prefill", "--no-kv-sharing-fast-prefill")
	args = append(args, extra...)
	return args, nil
}

func childEnv(env []string) []string {
	out := env[:0]
	for _, entry := range env {
		key, _, ok := strings.Cut(entry, "=")
		if !ok || wrapperOnlyEnv(key) {
			continue
		}
		out = append(out, entry)
	}
	return out
}

func wrapperOnlyEnv(key string) bool {
	switch key {
	case "MODEL_ID",
		"VLLM_BIN",
		"VLLM_PORT",
		"VLLM_TOKENIZER",
		"VLLM_HF_CONFIG_PATH",
		"VLLM_LOAD_FORMAT",
		"VLLM_SERVED_MODEL_NAME",
		"VLLM_DOWNLOAD_DIR",
		"VLLM_TENSOR_PARALLEL_SIZE",
		"VLLM_MAX_MODEL_LEN",
		"VLLM_MAX_NUM_SEQS",
		"VLLM_MAX_NUM_BATCHED_TOKENS",
		"VLLM_GPU_MEMORY_UTILIZATION",
		"VLLM_KV_CACHE_DTYPE",
		"VLLM_KV_CACHE_MEMORY_BYTES",
		"VLLM_CPU_OFFLOAD_GB",
		"VLLM_SPECULATIVE_CONFIG",
		"VLLM_TOOL_CALL_PARSER",
		"VLLM_REASONING_PARSER",
		"VLLM_CHAT_TEMPLATE",
		"VLLM_LIMIT_MM_PER_PROMPT",
		"VLLM_ADDITIONAL_CONFIG",
		"VLLM_DTYPE",
		"VLLM_QUANTIZATION",
		"VLLM_TRUST_REMOTE_CODE",
		"VLLM_ENABLE_PREFIX_CACHING",
		"VLLM_ENABLE_CHUNKED_PREFILL",
		"VLLM_ENABLE_AUTO_TOOL_CHOICE",
		"VLLM_ASYNC_SCHEDULING",
		"VLLM_KV_SHARING_FAST_PREFILL",
		"MEMORY_LIMIT":
		return true
	default:
		return false
	}
}

func onOff(value string) bool {
	enabled, ok := boolEnv(value)
	return ok && enabled
}

func boolEnv(value string) (bool, bool) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true, true
	case "0", "false", "no", "off":
		return false, true
	default:
		return false, false
	}
}

func authHandler(expected string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		const prefix = "Bearer "
		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, prefix) {
			http.Error(w, "missing bearer", http.StatusUnauthorized)
			return
		}
		got := strings.TrimPrefix(auth, prefix)
		if subtle.ConstantTimeCompare([]byte(got), []byte(expected)) != 1 {
			http.Error(w, "invalid bearer", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func waitForUpstream(ctx context.Context, addr string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		var d net.Dialer
		conn, err := d.DialContext(ctx, "tcp", addr)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timeout waiting for %s after %s: %w", addr, timeout, err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
