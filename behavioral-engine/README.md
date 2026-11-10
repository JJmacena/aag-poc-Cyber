# AAG Behavioral Engine

Triple-layer detection engine for autonomous AI-driven offensive agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BEHAVIORAL ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: CVE Context Match (Threat Intel Feeds)            │
│  Layer 2: Behavioral Anomaly                                │
│    ├─ Isolation Forest (Online/Streaming)                   │
│    └─ Transformer (Command Sequence Classification)         │
│  Layer 3: Agent Fingerprint                                 │
│    ├─ LLM Judge (Semantic Reasoning via Ollama)             │
│    └─ Agent Profiler (Sentence-BERT + FAISS)                │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/score` | Score a session through full ensemble |
| POST | `/score/features` | Score pre-computed 47-feature vector |
| GET | `/models/info` | Model configuration info |
| GET | `/features/names` | List of 47 feature names |

## Swagger UI

Available at: `http://localhost:8000/docs`

## Models

### Isolation Forest (Online)
- Algorithm: River's IsolationForest
- Features: 47 behavioral features
- Streaming: Online learning with dynamic threshold
- Config: 100 trees, max_depth=10, window_size=256

### Transformer (Command Sequence)
- Architecture: Mini-BERT (4 layers, 4 heads, d_model=128)
- Input: Tokenized command sequences (seq_len=128, cmd_len=64)
- Dual heads: Classification + Language Modeling
- Classes: Benign (0), Scripted (1), Agentic (2)

### LLM Judge (Semantic Reasoning)
- Backend: Ollama/vLLM (Nemotron-3-Ultra)
- Output: Classification, Confidence, Reasoning, MITRE, Deception Action, Behavioral IOCs
- Modes: Standard / Adversarial

## Ensemble Scoring

```
Final Score = 0.30 × IF_norm + 0.30 × Transformer + 0.30 × LLM_Judge + 0.10 × Honeytoken_Boost
```

## Threat Tiers

| Tier | Threshold | Action |
|------|-----------|--------|
| TIER-1 | ≥ 0.80 | Tar pit, data poisoning, intel generation, block egress |
| TIER-2 | ≥ 0.65 | Honeytoken injection, decoy redirect, enhanced capture |
| TIER-3 | ≥ 0.45 | Honeytoken scheduling, alert SOC (low) |
| BENIGN | < 0.45 | Log, baseline update |

## Development

```bash
# Install dependencies
pip install -e .[dev]

# Run API server
python -m aag_be.api

# Train Isolation Forest
python -m aag_be.train_if --data data/benign_sessions.parquet --output models/if_model.pkl

# Train Transformer
python -m aag_be.train_transformer --benign data/benign_sessions.parquet --redteam data/redteam --output models/transformer
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URL | http://ollama:11434 | Ollama API endpoint |
| MODEL_DIR | /models | Model storage directory |
| TIER1_THRESHOLD | 0.80 | Tier-1 threshold |
| TIER2_THRESHOLD | 0.65 | Tier-2 threshold |
| TIER3_THRESHOLD | 0.45 | Tier-3 threshold |