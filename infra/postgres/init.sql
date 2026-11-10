-- PostgreSQL initialization for AAG PoC
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Sessions metadata
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    src_ip INET,
    user_agent TEXT,
    decoy_persona VARCHAR(32),
    start_ts TIMESTAMPTZ,
    end_ts TIMESTAMPTZ,
    final_score REAL,
    threat_tier VARCHAR(16),
    if_score REAL,
    transformer_score REAL,
    llm_score REAL,
    honeytoken_boost REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- FSM session states (Deception Orchestrator)
CREATE TABLE IF NOT EXISTS fsm_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    current_state VARCHAR(32) NOT NULL,
    context JSONB DEFAULT '{}',
    score REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- FSM events log
CREATE TABLE IF NOT EXISTS fsm_events (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) REFERENCES fsm_sessions(session_id),
    event_type VARCHAR(64) NOT NULL,
    payload JSONB,
    from_state VARCHAR(32),
    to_state VARCHAR(32),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent families
CREATE TABLE IF NOT EXISTS agent_families (
    id SERIAL PRIMARY KEY,
    name VARCHAR(64) UNIQUE NOT NULL,
    description TEXT,
    caas_type VARCHAR(32),
    mitre_techniques TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Session embeddings (pgvector)
CREATE TABLE IF NOT EXISTS session_embeddings (
    session_id VARCHAR(64) PRIMARY KEY REFERENCES sessions(session_id),
    embedding VECTOR(384),
    family_id INTEGER REFERENCES agent_families(id),
    confidence REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Deception actions
CREATE TABLE IF NOT EXISTS deception_actions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) REFERENCES fsm_sessions(session_id),
    action VARCHAR(64) NOT NULL,
    fsm_state VARCHAR(32),
    trigger_score REAL,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sessions_threat_tier ON sessions(threat_tier);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_fsm_sessions_state ON fsm_sessions(current_state);
CREATE INDEX IF NOT EXISTS idx_fsm_events_session ON fsm_events(session_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_family ON session_embeddings(family_id);

-- Vector similarity index (IVFFlat for pgvector)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON session_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Default agent families
INSERT INTO agent_families (name, description, caas_type, mitre_techniques) VALUES
('AUTOGPT_RAAS', 'AutoGPT-based ransomware affiliates', 'RAAS', ARRAY['T1486', 'T1059', 'T1566']),
('BABYAGI_CAAS', 'BabyAGI for credential stuffing', 'ACCESS_BROKER', ARRAY['T1110', 'T1555', 'T1078']),
('CUSTOM_LLM_AGENT', 'Custom LLM agents for APT-style ops', 'C2_FRAMEWORK', ARRAY['T1584', 'T1190', 'T1059']),
('FRAUDGPT', 'Uncensored LLM for BEC/fraud', 'PHISHING_KIT', ARRAY['T1566', 'T1598']),
('WORMGPT', 'Uncensored LLM for malware/exploits', 'EXPLOIT_KIT', ARRAY['T1587', 'T1190', 'T1059']),
('COBALT_STRIKE_BEACON', 'CS Beacon with AI-assisted lateral', 'C2_FRAMEWORK', ARRAY['T1059', 'T1021', 'T1003']),
('METASPLOIT_AI', 'Metasploit with LLM-assisted modules', 'EXPLOIT_KIT', ARRAY['T1190', 'T1059', 'T1068']),
('BENIGN', 'Legitimate admin/dev activity', 'NONE', ARRAY[]::TEXT[])
ON CONFLICT (name) DO NOTHING;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aag;