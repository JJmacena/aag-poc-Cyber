-- ClickHouse initialization for AAG PoC
CREATE DATABASE IF NOT EXISTS aag;

-- Sessions table for behavioral analytics
CREATE TABLE IF NOT EXISTS aag.sessions (
    session_id String,
    src_ip String,
    user_agent String,
    decoy_persona String,
    start_ts DateTime64(3),
    end_ts DateTime64(3),
    duration_seconds Float64,
    final_score Float32,
    threat_tier String,
    if_score Float32,
    transformer_score Float32,
    llm_score Float32,
    honeytoken_boost Float32,
    command_count UInt32,
    network_count UInt32,
    file_count UInt32,
    mitre_count UInt32,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(created_at)
ORDER BY (session_id, created_at)
TTL created_at + INTERVAL 30 DAY;

-- Events table for detailed telemetry
CREATE TABLE IF NOT EXISTS aag.events (
    session_id String,
    event_type Enum8('command' = 1, 'network' = 2, 'file' = 3, 'llm' = 4, 'mitre' = 5),
    timestamp DateTime64(3),
    payload String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(created_at)
ORDER BY (session_id, timestamp)
TTL created_at + INTERVAL 7 DAY;

-- Deception actions log
CREATE TABLE IF NOT EXISTS aag.deception_actions (
    session_id String,
    action String,
    fsm_state String,
    trigger_score Float32,
    details String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(created_at)
ORDER BY (session_id, created_at)
TTL created_at + INTERVAL 90 DAY;

-- Agent attributions
CREATE TABLE IF NOT EXISTS aag.attributions (
    session_id String,
    family String,
    confidence Float32,
    embedding Array(Float32),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(created_at)
ORDER BY (session_id, created_at)
TTL created_at + INTERVAL 90 DAY;

GRANT ALL ON aag.* TO aag;