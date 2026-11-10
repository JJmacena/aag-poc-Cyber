import React, { useState, useEffect } from 'react';
import './App.css';

interface SessionScore {
  session_id: string;
  final_score: number;
  threat_tier: string;
  if_score: number;
  transformer_score: number;
  llm_score: number;
  honeytoken_boost: number;
  timestamp: string;
}

const App: React.FC = () => {
  const [sessions, setSessions] = useState<SessionScore[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<SessionScore | null>(null);

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/health');
      if (!response.ok) throw new Error('API not available');

      // Mock data for demo
      const mockSessions: SessionScore[] = [
        {
          session_id: 'sess-001',
          final_score: 0.85,
          threat_tier: 'TIER-1',
          if_score: 0.78,
          transformer_score: 0.82,
          llm_score: 0.91,
          honeytoken_boost: 0.1,
          timestamp: new Date().toISOString()
        },
        {
          session_id: 'sess-002',
          final_score: 0.42,
          threat_tier: 'BENIGN',
          if_score: 0.35,
          transformer_score: 0.45,
          llm_score: 0.40,
          honeytoken_boost: 0.0,
          timestamp: new Date(Date.now() - 60000).toISOString()
        },
        {
          session_id: 'sess-003',
          final_score: 0.68,
          threat_tier: 'TIER-2',
          if_score: 0.62,
          transformer_score: 0.71,
          llm_score: 0.65,
          honeytoken_boost: 0.05,
          timestamp: new Date(Date.now() - 120000).toISOString()
        }
      ];
      setSessions(mockSessions);
    } catch (err) {
      setError('Failed to connect to Behavioral Engine API');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
    const interval = setInterval(fetchSessions, 5000);
    return () => clearInterval(interval);
  }, []);

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'TIER-1': return '#dc3545';
      case 'TIER-2': return '#fd7e14';
      case 'TIER-3': return '#ffc107';
      case 'BENIGN': return '#28a745';
      default: return '#6c757d';
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>AAG PoC Dashboard</h1>
        <p>Autonomous Agent Guard - Triple-Layer Detection</p>
      </header>

      <main className="main">
        <section className="panel">
          <h2>Session Scores</h2>
          {error && <div className="error">{error}</div>}

          <table className="sessions-table">
            <thead>
              <tr>
                <th>Session ID</th>
                <th>Final Score</th>
                <th>Threat Tier</th>
                <th>IF Score</th>
                <th>Transformer</th>
                <th>LLM Judge</th>
                <th>Honeytoken</th>
                <th>Time</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={9}>Loading...</td></tr>
              ) : sessions.length === 0 ? (
                <tr><td colSpan={9}>No sessions</td></tr>
              ) : (
                sessions.map(session => (
                  <tr key={session.session_id} onClick={() => setSelectedSession(session)}>
                    <td>{session.session_id}</td>
                    <td className="score">{session.final_score.toFixed(2)}</td>
                    <td>
                      <span
                        className="tier-badge"
                        style={{ backgroundColor: getTierColor(session.threat_tier) }}
                      >
                        {session.threat_tier}
                      </span>
                    </td>
                    <td>{session.if_score.toFixed(2)}</td>
                    <td>{session.transformer_score.toFixed(2)}</td>
                    <td>{session.llm_score.toFixed(2)}</td>
                    <td>{session.honeytoken_boost.toFixed(2)}</td>
                    <td>{new Date(session.timestamp).toLocaleTimeString()}</td>
                    <td><button onClick={(e) => { e.stopPropagation(); setSelectedSession(session); }}>View</button></td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </section>

        {selectedSession && (
          <section className="panel detail-panel">
            <h3>Session Detail: {selectedSession.session_id}</h3>
            <div className="detail-grid">
              <div className="detail-item">
                <label>Final Score</label>
                <div className="score-bar">
                  <div
                    className="score-fill"
                    style={{ width: `${selectedSession.final_score * 100}%`, backgroundColor: getTierColor(selectedSession.threat_tier) }}
                  ></div>
                </div>
                <span>{selectedSession.final_score.toFixed(2)}</span>
              </div>
              <div className="detail-item">
                <label>Isolation Forest</label>
                <div className="score-bar">
                  <div className="score-fill" style={{ width: `${selectedSession.if_score * 100}%` }}></div>
                </div>
                <span>{selectedSession.if_score.toFixed(2)}</span>
              </div>
              <div className="detail-item">
                <label>Transformer</label>
                <div className="score-bar">
                  <div className="score-fill" style={{ width: `${selectedSession.transformer_score * 100}%` }}></div>
                </div>
                <span>{selectedSession.transformer_score.toFixed(2)}</span>
              </div>
              <div className="detail-item">
                <label>LLM Judge</label>
                <div className="score-bar">
                  <div className="score-fill" style={{ width: `${selectedSession.llm_score * 100}%` }}></div>
                </div>
                <span>{selectedSession.llm_score.toFixed(2)}</span>
              </div>
            </div>
            <button onClick={() => setSelectedSession(null)}>Close</button>
          </section>
        )}

        <section className="panel links">
          <h2>API Documentation</h2>
          <div className="links-grid">
            <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
              <div className="link-card">
                <h3>Behavioral Engine</h3>
                <p>Port 8000 - /score, /features/names, /models/info</p>
              </div>
            </a>
            <a href="http://localhost:8001/docs" target="_blank" rel="noopener noreferrer">
              <div className="link-card">
                <h3>Deception Orchestrator</h3>
                <p>Port 8001 - FSM Sessions, Events, State</p>
              </div>
            </a>
            <a href="http://localhost:8002/docs" target="_blank" rel="noopener noreferrer">
              <div className="link-card">
                <h3>Agent Profiler</h3>
                <p>Port 8002 - /attribute, /embed, /families</p>
              </div>
            </a>
            <a href="http://localhost:8003/docs" target="_blank" rel="noopener noreferrer">
              <div className="link-card">
                <h3>LLM Judge</h3>
                <p>Port 8003 - /judge, /judge/batch, /prompt/template</p>
              </div>
            </a>
            <a href="http://localhost:8080/docs" target="_blank" rel="noopener noreferrer">
              <div className="link-card">
                <h3>Deception Swagger UI</h3>
                <p>Port 8080 - Alternative Swagger UI</p>
              </div>
            </a>
            <a href="http://localhost:3001" target="_blank" rel="noopener noreferrer">
              <div className="link-card">
                <h3>Grafana</h3>
                <p>Port 3001 - Visualization (admin/aag)</p>
              </div>
            </a>
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;