package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"aag-deception-orchestrator/internal/state"
)

// Handler wraps the FSM manager for HTTP API
type Handler struct {
	fsmManager *state.FSMManager
}

// NewHandler creates a new API handler
func NewHandler(fsmManager *state.FSMManager) *Handler {
	return &Handler{fsmManager: fsmManager}
}

// RegisterRoutes registers all API routes
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(gin.Logger())

	// Health
	router.GET("/health", h.HealthCheck)

	// FSM endpoints
	api := router.Group("/api/v1")
	{
		api.POST("/sessions", h.CreateSession)
		api.GET("/sessions/:sessionId", h.GetSession)
		api.POST("/sessions/:sessionId/events", h.DispatchEvent)
		api.GET("/sessions/:sessionId/state", h.GetState)
		api.GET("/sessions/:sessionId/context", h.GetContext)
		api.POST("/sessions/:sessionId/score", h.UpdateScore)
		// Honeytoken injection
		api.POST("/sessions/:sessionId/honeytoken", h.InjectHoneytoken)
	}

	mux.Handle("/", router)
}

// HealthCheck returns health status
func (h *Handler) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"service":   "deception-orchestrator",
		"version":   "1.0.0",
		"timestamp": time.Now().Unix(),
	})
}

// CreateSessionRequest represents the request to create a new session
type CreateSessionRequest struct {
	SessionID     string                 `json:"session_id" binding:"required"`
	SrcIP         string                 `json:"src_ip" binding:"required"`
	UserAgent     string                 `json:"user_agent"`
	DecoyPersona  string                 `json:"decoy_persona" binding:"required"`
	InitialScore  float64                `json:"initial_score"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// CreateSession creates a new FSM for a session
func (h *Handler) CreateSession(c *gin.Context) {
	var req CreateSessionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Create executor (mock for now)
	executor := &MockExecutor{}

	fsm := state.NewFSM(req.SessionID, req.SrcIP, req.UserAgent, req.DecoyPersona, executor)

	// Add initial metadata
	ctx := context.Background()
	if req.Metadata != nil {
		fsm.GetContext().Metadata = req.Metadata
	}

	// Start FSM
	fsm.Start(ctx)

	// Store in manager
	h.fsmManager.AddFSM(req.SessionID, fsm)

	// Dispatch initial event if score provided
	if req.InitialScore > 0 {
		event := state.Event{
			Type:      state.EventAnomalyDetected,
			SessionID: req.SessionID,
			Timestamp: time.Now(),
			Payload:   map[string]interface{}{"score": req.InitialScore},
			Source:    "behavioral-engine",
		}
		fsm.Dispatch(event)
	}

	c.JSON(http.StatusCreated, gin.H{
		"session_id": req.SessionID,
		"state":      fsm.GetState(),
		"message":    "Session created and FSM started",
	})
}

// GetSession returns session info
func (h *Handler) GetSession(c *gin.Context) {
	sessionID := c.Param("sessionId")
	fsm := h.fsmManager.GetFSM(sessionID)
	if fsm == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	ctx := fsm.GetContext()
	c.JSON(http.StatusOK, gin.H{
		"session_id":      ctx.SessionID,
		"src_ip":          ctx.SrcIP,
		"user_agent":      ctx.UserAgent,
		"decoy_persona":   ctx.DecoyPersona,
		"state":           fsm.GetState(),
		"threat_tier":     ctx.ThreatTier,
		"classification":  ctx.Classification,
		"start_time":      ctx.StartTime,
		"last_event_time": ctx.LastEventTime,
		"event_count":     ctx.EventCount,
		"score_history":   ctx.ScoreHistory,
		"metadata":        ctx.Metadata,
	})
}

// DispatchEvent dispatches an event to the FSM
func (h *Handler) DispatchEvent(c *gin.Context) {
	sessionID := c.Param("sessionId")
	fsm := h.fsmManager.GetFSM(sessionID)
	if fsm == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	var req struct {
		Type    string                 `json:"type" binding:"required"`
		Payload map[string]interface{} `json:"payload"`
		Source  string                 `json:"source"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	event := state.Event{
		Type:      state.EventType(req.Type),
		SessionID: sessionID,
		Timestamp: time.Now(),
		Payload:   req.Payload,
		Source:    req.Source,
	}

	fsm.Dispatch(event)

	c.JSON(http.StatusOK, gin.H{
		"session_id": sessionID,
		"state":      fsm.GetState(),
		"event":      req.Type,
		"message":    "Event dispatched",
	})
}

// GetState returns current FSM state
func (h *Handler) GetState(c *gin.Context) {
	sessionID := c.Param("sessionId")
	fsm := h.fsmManager.GetFSM(sessionID)
	if fsm == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"session_id": sessionID,
		"state":      fsm.GetState(),
	})
}

// GetContext returns FSM context
func (h *Handler) GetContext(c *gin.Context) {
	sessionID := c.Param("sessionId")
	fsm := h.fsmManager.GetFSM(sessionID)
	if fsm == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	ctx := fsm.GetContext()
	c.JSON(http.StatusOK, ctx)
}

// UpdateScore updates session score and triggers anomaly detection
func (h *Handler) UpdateScore(c *gin.Context) {
	sessionID := c.Param("sessionId")
	fsm := h.fsmManager.GetFSM(sessionID)
	if fsm == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	var req struct {
		Score float64 `json:"score" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Update score history
	ctx := fsm.GetContext()
	ctx.ScoreHistory = append(ctx.ScoreHistory, req.Score)

	// Trigger anomaly detection if score exceeds threshold
	if req.Score >= 0.65 {
		event := state.Event{
			Type:      state.EventAnomalyDetected,
			SessionID: sessionID,
			Timestamp: time.Now(),
			Payload:   map[string]interface{}{"score": req.Score},
			Source:    "behavioral-engine",
		}
		fsm.Dispatch(event)
	}

	c.JSON(http.StatusOK, gin.H{
		"session_id": sessionID,
		"score":      req.Score,
		"state":      fsm.GetState(),
		"message":    "Score updated",
	})
}

// ServeOpenAPI serves the OpenAPI spec
func (h *Handler) ServeOpenAPI(w http.ResponseWriter, r *http.Request) {
	// Read from embedded or file
	openAPISpec := getOpenAPISpec()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(openAPISpec)
}

func getOpenAPISpec() map[string]interface{} {
	return map[string]interface{}{
		"openapi": "3.0.3",
		"info": map[string]interface{}{
			"title":       "AAG Deception Orchestrator API",
			"version":     "1.0.0",
			"description": "Adaptive deception orchestration FSM for AI-driven threat response",
		},
		"servers": []map[string]string{
			{"url": "http://localhost:8001", "description": "Local development"},
		},
		"paths": map[string]interface{}{
			"/health": map[string]interface{}{
				"get": map[string]interface{}{
					"summary": "Health check",
					"responses": map[string]interface{}{
						"200": map[string]interface{}{"description": "OK"},
					},
				},
			},
			"/api/v1/sessions": map[string]interface{}{
				"post": map[string]interface{}{
					"summary": "Create new session",
					"requestBody": map[string]interface{}{
						"required": true,
						"content": map[string]interface{}{
							"application/json": map[string]interface{}{
								"schema": map[string]interface{}{
									"$ref": "#/components/schemas/CreateSessionRequest",
								},
							},
						},
					},
					"responses": map[string]interface{}{
						"201": map[string]interface{}{"description": "Created"},
					},
				},
			},
			"/api/v1/sessions/{sessionId}": map[string]interface{}{
				"get": map[string]interface{}{
					"summary": "Get session info",
					"parameters": []map[string]interface{}{
						{"name": "sessionId", "in": "path", "required": true, "schema": map[string]interface{}{"type": "string"}},
					},
					"responses": map[string]interface{}{
						"200": map[string]interface{}{"description": "OK"},
						"404": map[string]interface{}{"description": "Not found"},
					},
				},
			},
			"/api/v1/sessions/{sessionId}/events": map[string]interface{}{
				"post": map[string]interface{}{
					"summary": "Dispatch event to FSM",
					"parameters": []map[string]interface{}{
						{"name": "sessionId", "in": "path", "required": true, "schema": map[string]interface{}{"type": "string"}},
					},
					"requestBody": map[string]interface{}{
						"required": true,
						"content": map[string]interface{}{
							"application/json": map[string]interface{}{
								"schema": map[string]interface{}{
									"$ref": "#/components/schemas/EventRequest",
								},
							},
						},
					},
					"responses": map[string]interface{}{
						"200": map[string]interface{}{"description": "OK"},
					},
				},
			},
			"/api/v1/sessions/{sessionId}/state": map[string]interface{}{
				"get": map[string]interface{}{
					"summary": "Get current FSM state",
					"parameters": []map[string]interface{}{
						{"name": "sessionId", "in": "path", "required": true, "schema": map[string]interface{}{"type": "string"}},
					},
					"responses": map[string]interface{}{
						"200": map[string]interface{}{"description": "OK"},
					},
				},
			},
			"/api/v1/sessions/{sessionId}/context": map[string]interface{}{
				"get": map[string]interface{}{
					"summary": "Get FSM context",
					"parameters": []map[string]interface{}{
						{"name": "sessionId", "in": "path", "required": true, "schema": map[string]interface{}{"type": "string"}},
					},
					"responses": map[string]interface{}{
						"200": map[string]interface{}{"description": "OK"},
					},
				},
			},
			"/api/v1/sessions/{sessionId}/score": map[string]interface{}{
				"post": map[string]interface{}{
					"summary": "Update anomaly score",
					"parameters": []map[string]interface{}{
						{"name": "sessionId", "in": "path", "required": true, "schema": map[string]interface{}{"type": "string"}},
					},
					"requestBody": map[string]interface{}{
						"required": true,
						"content": map[string]interface{}{
							"application/json": map[string]interface{}{
								"schema": map[string]interface{}{
									"$ref": "#/components/schemas/ScoreRequest",
								},
							},
						},
					},
					"responses": map[string]interface{}{
						"200": map[string]interface{}{"description": "OK"},
					},
				},
			},
			"/api/v1/sessions/{sessionId}/honeytoken": map[string]interface{}{
				"post": map[string]interface{}{
					"summary": "Inject honeytoken into session",
					"parameters": []map[string]interface{}{
						{"name": "sessionId", "in": "path", "required": true, "schema": map[string]interface{}{"type": "string"}},
					},
					"requestBody": map[string]interface{}{
						"required": true,
						"content": map[string]interface{}{
							"application/json": map[string]interface{}{
								"schema": map[string]interface{}{
									"$ref": "#/components/schemas/HoneytokenInjectRequest",
								},
							},
						},
					},
					"responses": map[string]interface{}{
						"200": map[string]interface{}{"description": "OK"},
						"404": map[string]interface{}{"description": "Session not found"},
					},
				},
			},
		},
		"components": map[string]interface{}{
			"schemas": map[string]interface{}{
				"CreateSessionRequest": map[string]interface{}{
					"type": "object",
					"required": []string{"session_id", "src_ip", "decoy_persona"},
					"properties": map[string]interface{}{
						"session_id":     map[string]interface{}{"type": "string"},
						"src_ip":         map[string]interface{}{"type": "string"},
						"user_agent":     map[string]interface{}{"type": "string"},
						"decoy_persona":  map[string]interface{}{"type": "string"},
						"initial_score":  map[string]interface{}{"type": "number"},
						"metadata":       map[string]interface{}{"type": "object"},
					},
				},
				"EventRequest": map[string]interface{}{
					"type": "object",
					"required": []string{"type"},
					"properties": map[string]interface{}{
						"type":    map[string]interface{}{"type": "string", "enum": []string{"SESSION_START", "ANOMALY_DETECTED", "HONEYTOKEN_TOUCH", "DECOY_ACCESS", "ESCALATION", "INTEL_COMPLETE", "SESSION_END", "TIMEOUT", "MANUAL_OVERRIDE"}},
						"payload": map[string]interface{}{"type": "object"},
						"source":  map[string]interface{}{"type": "string"},
					},
				},
				"ScoreRequest": map[string]interface{}{
					"type": "object",
					"required": []string{"score"},
					"properties": map[string]interface{}{
						"score": map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
					},
				},
				"HoneytokenInjectRequest": map[string]interface{}{
					"type": "object",
					"required": []string{"session_id", "token_type"},
					"properties": map[string]interface{}{
						"session_id": map[string]interface{}{"type": "string"},
						"token_type": map[string]interface{}{"type": "string"},
						"persona":    map[string]interface{}{"type": "string"},
						"value":      map[string]interface{}{"type": "string"},
					},
				},
			},
		},
	}
}

// HoneytokenInjectRequest represents a request to inject a honeytoken
type HoneytokenInjectRequest struct {
	SessionID  string `json:"session_id" binding:"required"`
	TokenType  string `json:"token_type" binding:"required"`  // e.g., "CPF", "CREDIT_CARD", "API_KEY", "JWT"
	Persona    string `json:"persona"`
	Value      string `json:"value,omitempty"` // Optional: specific value to inject, otherwise auto-generated
}

// HoneytokenInjectResponse represents the response from honeytoken injection
type HoneytokenInjectResponse struct {
	SessionID  string `json:"session_id"`
	TokenType  string `json:"token_type"`
	Injected   bool   `json:"injected"`
	Honeytoken string `json:"honeytoken,omitempty"`
	Message    string `json:"message"`
}

// InjectHoneytoken injects a honeytoken into a session's deception context
func (h *Handler) InjectHoneytoken(c *gin.Context) {
	var req HoneytokenInjectRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	fsm := h.fsmManager.GetFSM(req.SessionID)
	if fsm == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	ctx := fsm.GetContext()

	// Generate honeytoken if not provided
	honeytoken := req.Value
	if honeytoken == "" {
		// In a real implementation, this would use a proper generator
		// For now, return a placeholder
		honeytoken = fmt.Sprintf("[HONEYTOKEN:%s:%s]", req.TokenType, req.SessionID[:8])
	}

	// Store honeytoken in session metadata
	if ctx.Metadata == nil {
		ctx.Metadata = make(map[string]interface{})
	}
	ctx.Metadata["honeytoken_"+req.TokenType] = honeytoken
	ctx.Metadata["honeytoken_injected_at"] = time.Now().Unix()
	ctx.Metadata["honeytoken_persona"] = req.Persona

	// Execute the honeytoken injection action
	executor := &MockExecutor{}
	err := executor.InjectHoneytoken(context.Background(), req.SessionID, req.TokenType, req.Persona)
	if err != nil {
		log.Printf("Failed to inject honeytoken: %v", err)
	}

	c.JSON(http.StatusOK, HoneytokenInjectResponse{
		SessionID:  req.SessionID,
		TokenType:  req.TokenType,
		Injected:   true,
		Honeytoken: honeytoken,
		Message:    "Honeytoken injected successfully",
	})
}

// MockExecutor implements ActionExecutor for testing
type MockExecutor struct{}

func (m *MockExecutor) InjectHoneytoken(ctx context.Context, sessionID, tokenType, persona string) error {
	log.Printf("[MockExecutor] InjectHoneytoken: session=%s type=%s persona=%s", sessionID, tokenType, persona)
	return nil
}

func (m *MockExecutor) RedirectToDecoy(ctx context.Context, sessionID, decoyID string) error {
	log.Printf("[MockExecutor] RedirectToDecoy: session=%s decoy=%s", sessionID, decoyID)
	return nil
}

func (m *MockExecutor) ActivateTarPit(ctx context.Context, sessionID string) error {
	log.Printf("[MockExecutor] ActivateTarPit: session=%s", sessionID)
	return nil
}

func (m *MockExecutor) PoisonData(ctx context.Context, sessionID string) error {
	log.Printf("[MockExecutor] PoisonData: session=%s", sessionID)
	return nil
}

func (m *MockExecutor) GenerateIntel(ctx context.Context, sessionID string) (string, error) {
	log.Printf("[MockExecutor] GenerateIntel: session=%s", sessionID)
	return "intel-report-" + sessionID, nil
}

func (m *MockExecutor) TerminateSession(ctx context.Context, sessionID string) error {
	log.Printf("[MockExecutor] TerminateSession: session=%s", sessionID)
	return nil
}