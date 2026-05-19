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
	appendValueFlag("VLLM_TENSOR_PARALLEL_SIZE", "--tensor-parallel-size")
	appendValueFlag("VLLM_MAX_MODEL_LEN", "--max-model-len")
	appendValueFlag("VLLM_GPU_MEMORY_UTILIZATION", "--gpu-memory-utilization")
	if dtype := strings.TrimSpace(getenv("VLLM_DTYPE")); dtype != "" && dtype != "auto" {
		args = append(args, "--dtype", dtype)
	}
	if quant := strings.TrimSpace(getenv("VLLM_QUANTIZATION")); quant != "" && quant != "none" {
		args = append(args, "--quantization", quant)
	}
	if onOff(getenv("VLLM_TRUST_REMOTE_CODE")) {
		args = append(args, "--trust-remote-code")
	}
	args = append(args, extra...)
	return args, nil
}

func onOff(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
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
