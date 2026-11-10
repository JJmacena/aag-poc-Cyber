# AAG Agent Profiler

Behavioral fingerprinting & agent family attribution via Sentence-BERT + FAISS IVF-PQ.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT PROFILER                          │
├─────────────────────────────────────────────────────────────┤
│  Session → Fingerprint Text → Sentence-BERT → Embeddings    │
│                                              │              │
│                                              ▼              │
│                                    FAISS IVF-PQ Index       │
│                                              │              │
│                                              ▼              │
│                              k-NN Search → Family Attribution
└─────────────────────────────────────────────────────────────┘
```

## Agent Families

| Family | Description | CaaS Type | Typical MITRE |
|--------|-------------|-----------|---------------|
| AUTOGPT_RAAS | AutoGPT-based ransomware affiliates | RAAS | T1486, T1059, T1566 |
| BABYAGI_CAAS | BabyAGI for credential stuffing | ACCESS_BROKER | T1110, T1555, T1078 |
| CUSTOM_LLM_AGENT | Custom LLM agents for APT-style ops | C2_FRAMEWORK | T1584, T1190, T1059 |
| FRAUDGPT | Uncensored LLM for BEC/fraud | PHISHING_KIT | T1566, T1598 |
| WORMGPT | Uncensored LLM for malware/exploits | EXPLOIT_KIT | T1587, T1190, T1059 |
| COBALT_STRIKE_BEACON | CS Beacon with AI-assisted lateral | C2_FRAMEWORK | T1059, T1021, T1003 |
| METASPLOIT_AI | Metasploit with LLM-assisted modules | EXPLOIT_KIT | T1190, T1059, T1068 |
| BENIGN | Legitimate admin/dev activity | NONE | - |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/attribute` | Attribute session to agent family |
| POST | `/embed` | Generate embeddings for sessions |
| POST | `/families/add` | Add new family to index |
| GET | `/families` | List all known families |
| GET | `/models/info` | Model and index info |
| POST | `/families/save` | Save FAISS index to disk |
| POST | `/debug/fingerprint` | Debug: session to fingerprint text |

## Swagger UI

Available at: `http://localhost:8002/docs`

## Model Details

- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Fine-tuning**: Contrastive loss on benign vs malicious pairs
- **Index Types**: IVF-PQ (default), HNSW, Flat
- **IVF-PQ Config**: nlist=100, m=16, nbits=8

## Training

```bash
# Fine-tune Sentence-BERT
python -m aag_profiler.fine_tune \
    --benign data/benign_sessions.parquet \
    --redteam data/redteam \
    --output models/aag-agent-profiler-v1 \
    --epochs 4 --batch-size 16 --lr 2e-5
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_DIR | /models/aag-agent-profiler-v1 | Model directory |
| DEVICE | cuda | Device (cuda/cpu) |