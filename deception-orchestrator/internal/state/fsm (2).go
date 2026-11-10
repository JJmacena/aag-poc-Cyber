package state

import (
	"context"
	"encoding/json"
	"log"
	"sync"
	"time"
)

// State represents the deception orchestrator FSM states
type State string

const (
	StateMonitoring       State = "MONITORING"
	StateHoneytokenInject State = "HONEYTOKEN_INJECTION"
	StateDecoyRedirect    State = "DECOY_REDIRECT"
	StateTarPit           State = "TAR_PIT"
	StateDataPoisoning    State = "DATA_POISONING"
	StateIntelGeneration  State = "INTEL_GENERATION"
	StateTerminal         State = "TERMINAL"
)

// Event types that trigger state transitions
type EventType string

const (
	EventSessionStart      EventType = "SESSION_START"
	EventAnomalyDetected   EventType = "ANOMALY_DETECTED"
	EventHoneytokenTouch   EventType = "HONEYTOKEN_TOUCH"
	EventDecoyAccess       EventType = "DECOY_ACCESS"
	EventEscalation        EventType = "ESCALATION"
	EventIntelComplete     EventType = "INTEL_COMPLETE"
	EventSessionEnd        EventType = "SESSION_END"
	EventTimeout           EventType = "TIMEOUT"
	EventManualOverride    EventType = "MANUAL_OVERRIDE"
)

// Event represents a state transition trigger
type Event struct {
	Type      EventType
	SessionID string
	Timestamp time.Time
	Payload   map[string]interface{}
	Source    string // "behavioral-engine", "profiler", "manual", etc.
}

// Context holds session-specific metadata for the FSM
type Context struct {
	SessionID      string
	SrcIP          string
	UserAgent      string
	DecoyPersona   string
	ThreatTier     string
	Classification string
	MitreTechniques []string
	HoneytokensDeployed []string
	DecoysAccessed    []string
	StartTime      time.Time
	LastEventTime  time.Time
	EventCount     int
	ScoreHistory   []float64
	Metadata       map[string]interface{}
	mu             sync.RWMutex
}

// ActionExecutor defines the interface for executing deception actions
type ActionExecutor interface {
	InjectHoneytoken(ctx context.Context, sessionID, tokenType, persona string) error
	RedirectToDecoy(ctx context.Context, sessionID, decoyID string) error
	ActivateTarPit(ctx context.Context, sessionID string) error
	PoisonData(ctx context.Context, sessionID string) error
	GenerateIntel(ctx context.Context, sessionID string) (string, error)
	TerminateSession(ctx context.Context, sessionID string) error
}

// Transition defines a state transition
type Transition struct {
	From       State
	Event      EventType
	To         State
	Action     func(ctx context.Context, fsm *FSM, event Event) error
	Guard      func(ctx *Context, event Event) bool
}

// FSM is the deception orchestrator finite state machine
type FSM struct {
	mu           sync.RWMutex
	state        State
	context      *Context
	transitions  map[State]map[EventType]Transition
	executor     ActionExecutor
	eventChan    chan Event
	honeytokenTimer *time.Timer
	stopChan     chan struct{}
	wg           sync.WaitGroup
}

// NewFSM creates a new deception orchestrator FSM
func NewFSM(sessionID, srcIP, userAgent, decoyPersona string, executor ActionExecutor) *FSM {
	ctx := &Context{
		SessionID:    sessionID,
		SrcIP:        srcIP,
		UserAgent:    userAgent,
		DecoyPersona: decoyPersona,
		StartTime:    time.Now(),
		LastEventTime: time.Now(),
		Metadata:     make(map[string]interface{}),
		ScoreHistory: make([]float64, 0),
	}

	fsm := &FSM{
		state:       StateMonitoring,
		context:     ctx,
		executor:    executor,
		eventChan:   make(chan Event, 100),
		stopChan:    make(chan struct{}),
		transitions: make(map[State]map[EventType]Transition),
	}

	fsm.initTransitions()
	return fsm
}

// initTransitions defines the FSM transition table
func (f *FSM) initTransitions() {
	f.transitions = map[State]map[EventType]Transition{
		StateMonitoring: {
			EventAnomalyDetected: {
				From:  StateMonitoring,
				Event: EventAnomalyDetected,
				To:    StateHoneytokenInject,
				Action: f.actionInjectHoneytoken,
				Guard: f.guardAnomalyThreshold,
			},
			EventSessionEnd: {
				From:  StateMonitoring,
				Event: EventSessionEnd,
				To:    StateTerminal,
				Action: f.actionCleanup,
			},
		},
		StateHoneytokenInject: {
			EventHoneytokenTouch: {
				From:  StateHoneytokenInject,
				Event: EventHoneytokenTouch,
				To:    StateDecoyRedirect,
				Action: f.actionRedirectDecoy,
			},
			EventEscalation: {
				From:  StateHoneytokenInject,
				Event: EventEscalation,
				To:    StateTarPit,
				Action: f.actionActivateTarPit,
			},
			EventTimeout: {
				From:  StateHoneytokenInject,
				Event: EventTimeout,
				To:    StateMonitoring,
				Action: f.actionTimeoutReset,
			},
			EventSessionEnd: {
				From:  StateHoneytokenInject,
				Event: EventSessionEnd,
				To:    StateTerminal,
				Action: f.actionCleanup,
			},
		},
		StateDecoyRedirect: {
			EventDecoyAccess: {
				From:  StateDecoyRedirect,
				Event: EventDecoyAccess,
				To:    StateDataPoisoning,
				Action: f.actionPoisonData,
			},
			EventEscalation: {
				From:  StateDecoyRedirect,
				Event: EventEscalation,
				To:    StateTarPit,
				Action: f.actionActivateTarPit,
			},
			EventIntelComplete: {
				From:  StateDecoyRedirect,
				Event: EventIntelComplete,
				To:    StateIntelGeneration,
				Action: f.actionGenerateIntel,
			},
			EventSessionEnd: {
				From:  StateDecoyRedirect,
				Event: EventSessionEnd,
				To:    StateTerminal,
				Action: f.actionCleanup,
			},
		},
		StateTarPit: {
			EventIntelComplete: {
				From:  StateTarPit,
				Event: EventIntelComplete,
				To:    StateIntelGeneration,
				Action: f.actionGenerateIntel,
			},
			EventSessionEnd: {
				From:  StateTarPit,
				Event: EventSessionEnd,
				To:    StateTerminal,
				Action: f.actionCleanup,
			},
		},
		StateDataPoisoning: {
			EventIntelComplete: {
				From:  StateDataPoisoning,
				Event: EventIntelComplete,
				To:    StateIntelGeneration,
				Action: f.actionGenerateIntel,
			},
			EventSessionEnd: {
				From:  StateDataPoisoning,
				Event: EventSessionEnd,
				To:    StateTerminal,
				Action: f.actionCleanup,
			},
		},
		StateIntelGeneration: {
			EventSessionEnd: {
				From:  StateIntelGeneration,
				Event: EventSessionEnd,
				To:    StateTerminal,
				Action: f.actionCleanup,
			},
		},
		StateTerminal: {},
	}
}

// Start begins the FSM event loop
func (f *FSM) Start(ctx context.Context) {
	f.wg.Add(1)
	go f.run(ctx)
}

// Stop gracefully stops the FSM
func (f *FSM) Stop() {
	close(f.stopChan)
	f.wg.Wait()
}

// Dispatch sends an event to the FSM
func (f *FSM) Dispatch(event Event) {
	select {
	case f.eventChan <- event:
	case <-f.stopChan:
	default:
		log.Printf("[FSM %s] Event queue full, dropping event: %s", f.context.SessionID, event.Type)
	}
}

// GetState returns the current state
func (f *FSM) GetState() State {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.state
}

// GetContext returns a copy of the context
func (f *FSM) GetContext() *Context {
	f.mu.RLock()
	defer f.mu.RUnlock()
	ctxCopy := *f.context
	ctxCopy.Metadata = make(map[string]interface{})
	for k, v := range f.context.Metadata {
		ctxCopy.Metadata[k] = v
	}
	return &ctxCopy
}

// MarshalJSON implements json.Marshaler for State
func (s State) MarshalJSON() ([]byte, error) {
	return json.Marshal(string(s))
}

// UnmarshalJSON implements json.Unmarshaler for State
func (s *State) UnmarshalJSON(data []byte) error {
	var str string
	if err := json.Unmarshal(data, &str); err != nil {
		return err
	}
	*s = State(str)
	return nil
}

// run is the main FSM event loop
func (f *FSM) run(ctx context.Context) {
	defer f.wg.Done()
	f.startHoneytokenTimeout()

	for {
		select {
		case <-ctx.Done():
			f.transitionTo(StateTerminal, Event{Type: EventSessionEnd})
			return
		case <-f.stopChan:
			f.transitionTo(StateTerminal, Event{Type: EventSessionEnd})
			return
		case event := <-f.eventChan:
			f.handleEvent(ctx, event)
		}
	}
}

// handleEvent processes an incoming event
func (f *FSM) handleEvent(ctx context.Context, event Event) {
	f.mu.Lock()
	f.context.LastEventTime = event.Timestamp
	f.context.EventCount++
	f.mu.Unlock()

	f.mu.RLock()
	currentState := f.state
	stateTransitions := f.transitions[currentState]
	f.mu.RUnlock()

	if transition, ok := stateTransitions[event.Type]; ok {
		// Check guard condition
		if transition.Guard != nil {
			f.mu.RLock()
			guardPassed := transition.Guard(f.context, event)
			f.mu.RUnlock()
			if !guardPassed {
				log.Printf("[FSM %s] Guard failed for %s -> %s on %s",
					f.context.SessionID, currentState, transition.To, event.Type)
				return
			}
		}

		// Execute transition action
		if transition.Action != nil {
			if err := transition.Action(ctx, f, event); err != nil {
				log.Printf("[FSM %s] Action error: %v", f.context.SessionID, err)
			}
		}

		// Transition state
		f.transitionTo(transition.To, event)
	} else {
		log.Printf("[FSM %s] No transition for %s on %s", f.context.SessionID, currentState, event.Type)
	}
}

// transitionTo changes the FSM state
func (f *FSM) transitionTo(newState State, event Event) {
	f.mu.Lock()
	oldState := f.state
	f.state = newState
	f.mu.Unlock()

	log.Printf("[FSM %s] State transition: %s -> %s (event: %s)",
		f.context.SessionID, oldState, newState, event.Type)

	// Handle state-specific logic
	switch newState {
	case StateHoneytokenInject:
		f.startHoneytokenTimeout()
	case StateTerminal:
		f.stopHoneytokenTimeout()
	}
}

// Guard: Check if anomaly score exceeds threshold
func (f *FSM) guardAnomalyThreshold(ctx *Context, event Event) bool {
	score, ok := event.Payload["score"].(float64)
	if !ok {
		return false
	}
	// Tier 1 (0.8) -> immediate honeytoken, Tier 2 (0.65) -> honeytoken
	return score >= 0.65
}

// Action implementations
func (f *FSM) actionInjectHoneytoken(ctx context.Context, fsm *FSM, event Event) error {
	tokenType := "aws_cred"
	if tt, ok := event.Payload["token_type"].(string); ok {
		tokenType = tt
	}
	return fsm.executor.InjectHoneytoken(ctx, fsm.context.SessionID, tokenType, fsm.context.DecoyPersona)
}

func (f *FSM) actionRedirectDecoy(ctx context.Context, fsm *FSM, event Event) error {
	decoyID := "decoy-1"
	if did, ok := event.Payload["decoy_id"].(string); ok {
		decoyID = did
	}
	return fsm.executor.RedirectToDecoy(ctx, fsm.context.SessionID, decoyID)
}

func (f *FSM) actionActivateTarPit(ctx context.Context, fsm *FSM, event Event) error {
	return fsm.executor.ActivateTarPit(ctx, fsm.context.SessionID)
}

func (f *FSM) actionPoisonData(ctx context.Context, fsm *FSM, event Event) error {
	return fsm.executor.PoisonData(ctx, fsm.context.SessionID)
}

func (f *FSM) actionGenerateIntel(ctx context.Context, fsm *FSM, event Event) error {
	_, err := fsm.executor.GenerateIntel(ctx, fsm.context.SessionID)
	return err
}

func (f *FSM) actionCleanup(ctx context.Context, fsm *FSM, event Event) error {
	fsm.stopHoneytokenTimeout()
	return nil
}

func (f *FSM) actionTimeoutReset(ctx context.Context, fsm *FSM, event Event) error {
	fsm.startHoneytokenTimeout()
	return nil
}

// Honeytoken timeout management
func (f *FSM) startHoneytokenTimeout() {
	f.stopHoneytokenTimeout()
	f.honeytokenTimer = time.AfterFunc(5*time.Minute, func() {
		f.Dispatch(Event{
			Type:      EventTimeout,
			SessionID: f.context.SessionID,
			Timestamp: time.Now(),
			Source:    "timer",
		})
	})
}

func (f *FSM) stopHoneytokenTimeout() {
	if f.honeytokenTimer != nil {
		f.honeytokenTimer.Stop()
	}
}