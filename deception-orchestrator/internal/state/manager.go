package state

import (
	"context"
	"sync"
)

// FSMManager manages multiple FSM instances
type FSMManager struct {
	mu    sync.RWMutex
	fsms  map[string]*FSM
	wg    sync.WaitGroup
	ctx   context.Context
	cancel context.CancelFunc
}

// NewFSMManager creates a new FSM manager
func NewFSMManager() *FSMManager {
	ctx, cancel := context.WithCancel(context.Background())
	return &FSMManager{
		fsms:   make(map[string]*FSM),
		ctx:    ctx,
		cancel: cancel,
	}
}

// AddFSM adds an FSM to the manager
func (m *FSMManager) AddFSM(sessionID string, fsm *FSM) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.fsms[sessionID] = fsm
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		// FSM runs in its own goroutine via Start()
	}()
}

// GetFSM retrieves an FSM by session ID
func (m *FSMManager) GetFSM(sessionID string) *FSM {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.fsms[sessionID]
}

// RemoveFSM removes an FSM from the manager
func (m *FSMManager) RemoveFSM(sessionID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if fsm, ok := m.fsms[sessionID]; ok {
		fsm.Stop()
		delete(m.fsms, sessionID)
	}
}

// ListFSMs returns all active sessions
func (m *FSMManager) ListFSMs() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	sessions := make([]string, 0, len(m.fsms))
	for id := range m.fsms {
		sessions = append(sessions, id)
	}
	return sessions
}

// Shutdown stops all FSMs
func (m *FSMManager) Shutdown() {
	m.cancel()
	m.mu.Lock()
	for _, fsm := range m.fsms {
		fsm.Stop()
	}
	m.mu.Unlock()
	m.wg.Wait()
}