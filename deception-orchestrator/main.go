package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/aag/deception-orchestrator/internal/state"
)

type MockExecutor struct{}

func (m *MockExecutor) InjectHoneytoken(ctx context.Context, sessionID, tokenType, persona string) error {
	log.Printf("[Executor] InjectHoneytoken: session=%s type=%s persona=%s", sessionID, tokenType, persona)
	return nil
}

func (m *MockExecutor) RedirectToDecoy(ctx context.Context, sessionID, decoyID string) error {
	log.Printf("[Executor] RedirectToDecoy: session=%s decoy=%s", sessionID, decoyID)
	return nil
}

func (m *MockExecutor) ActivateTarPit(ctx context.Context, sessionID string) error {
	log.Printf("[Executor] ActivateTarPit: session=%s", sessionID)
	return nil
}

func (m *MockExecutor) PoisonData(ctx context.Context, sessionID string) error {
	log.Printf("[Executor] PoisonData: session=%s", sessionID)
	return nil
}

func (m *MockExecutor) GenerateIntel(ctx context.Context, sessionID string) (string, error) {
	log.Printf("[Executor] GenerateIntel: session=%s", sessionID)
	return "intel-report-" + sessionID, nil
}

func (m *MockExecutor) TerminateSession(ctx context.Context, sessionID string) error {
	log.Printf("[Executor] TerminateSession: session=%s", sessionID)
	return nil
}

func main() {
	// Create FSM
	executor := &MockExecutor{}
	fsm := state.NewFSM(
		"test-session-001",
		"192.168.1.100",
		"Mozilla/5.0 (Test Agent)",
		"K8S_NODE",
		executor,
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start FSM
	fsm.Start(ctx)

	// HTTP server for health checks and event ingestion
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"healthy","state":"` + string(fsm.GetState()) + `"}`))
	})
	mux.HandleFunc("/event", func(w http.ResponseWriter, r *http.Request) {
		// Parse event and dispatch to FSM
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"accepted"}`))
	})

	server := &http.Server{
		Addr:    ":8080",
		Handler: mux,
	}

	go func() {
		log.Println("Deception Orchestrator starting on :8080")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down...")
	fsm.Stop()
	server.Shutdown(context.Background())
	log.Println("Stopped")
}
