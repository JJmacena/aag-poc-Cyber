package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"aag-deception-orchestrator/internal/api"
	"aag-deception-orchestrator/internal/state"
)

func main() {
	// Configuration from environment
	port := getEnv("PORT", "8001")
	swaggerPort := getEnv("SWAGGER_PORT", "8080")
	kafkaBootstrap := getEnv("KAFKA_BOOTSTRAP", "kafka:9092")
	postgresDSN := getEnv("POSTGRES_DSN", "postgres://aag:aag@postgres:5432/aag")
	redisURL := getEnv("REDIS_URL", "redis://redis:6379")

	log.Printf("Starting Deception Orchestrator on port %s", port)
	log.Printf("Swagger UI on port %s", swaggerPort)
	log.Printf("Kafka: %s", kafkaBootstrap)
	log.Printf("PostgreSQL: %s", postgresDSN)
	log.Printf("Redis: %s", redisURL)

	// Initialize FSM manager
	fsmManager := state.NewFSMManager()

	// Create API handler
	handler := api.NewHandler(fsmManager)

	// Main API server
	mux := http.NewServeMux()
	handler.RegisterRoutes(mux)

	server := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Swagger UI server (static file serving)
	swaggerMux := http.NewServeMux()
	swaggerMux.Handle("/docs/", http.StripPrefix("/docs/", http.FileServer(http.Dir("/app/swagger"))))
	swaggerMux.HandleFunc("/openapi.json", handler.ServeOpenAPI)
	swaggerMux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		// Create a mock gin context for HealthCheck
		// For health check, we can just call a simple HTTP handler
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"healthy","service":"deception-orchestrator","version":"1.0.0"}`))
	})

	swaggerServer := &http.Server{
		Addr:         ":" + swaggerPort,
		Handler:      swaggerMux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start servers
	go func() {
		log.Printf("Main API listening on :%s", port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Main API server error: %v", err)
		}
	}()

	go func() {
		log.Printf("Swagger UI listening on :%s", swaggerPort)
		if err := swaggerServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Swagger server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down servers...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Main API shutdown error: %v", err)
	}
	if err := swaggerServer.Shutdown(ctx); err != nil {
		log.Printf("Swagger shutdown error: %v", err)
	}

	log.Println("Servers stopped")
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}