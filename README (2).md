# Autonomous Agent Guard (AAG) - Proof of Concept

> **Defensive system against Cybercrime-as-a-Service (CaaS) and Offensive AI Agents**

## Overview

AAG is a research prototype for detecting and deceiving autonomous AI-driven offensive agents (AutoGPT, BabyAGI, custom LLM agents, CaaS frameworks). It implements a triple-layer detection architecture with adaptive deception orchestration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AAG SYSTEM                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Behavioral  │  │  Deception   │  │   Agent      │           │
│  │   Engine     │◄─┤ Orchestrator │─►│  Profiler    │           │
│  │              │  │              │  │              │           │
│  │ • Isolation  │  │ • FSM (6     │  │ • Sentence-  │           │
│  │   Forest     │  │   states)    │  │   BERT       │           │
│  │ • Transformer│  │ • Honeytoken │  │ • FAISS      │           │
│  │ • LLM Judge  │  │   Injector   │  │   IVF-PQ     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           ▼                                      │
│              ┌────────────────────────┐                          │
│              │   Ensemble Scorer      │                          │
│              │  (IF 30% + TF 30% +    │                          │
│              │   LLM 30% + HT 10%)    │                          │
│              └───────────┬────────────┘                          │
│                           ▼                                      │
│              ┌────────────────────────┐                          │
│              │   Threat Tiers         │                          │
│              │ TIER-1 (≥0.80): Active │                          │
│              │ TIER-2 (≥0.65): Decoy  │                          │
│              │ TIER-3 (≥0.45): Monitor│                          │
│              └────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Behavioral Engine** | Python 3.11, River, PyTorch, FastAPI | Triple-layer detection (CVE context, behavioral anomaly, LLM semantic reasoning) |
| **Deception Orchestrator** | Go 1.22, FSM | Adaptive deception (6 states: MONITORING → HONEYTOKEN → DECOY → TAR_PIT → DATA_POISONING → INTEL) |
| **Agent Profiler** | Python, Sentence-BERT, FAISS | Behavioral fingerprinting & agent family attribution (k-NN on embeddings) |
| **LLM Judge** | Ollama/vLLM (Nemotron-3-Ultra) | Semantic reasoning on session behavior, adversarial prompt detection |
| **Dataset Generators** | Python | Benign personas (8), Red team scenarios (C1-C7), CaaS simulators (7), Adversarial evasion (GA) |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (recommended for LLM inference)
- 32GB+ RAM, 16+ CPU cores

### 1. Clone & Configure
```bash
cd aag-poc
cp .env.example .env  # Configure API keys, paths
```

### 2. Start Stack
```bash
make up              # Start all 13 services
make pull-llm        # Pull LLM models (Nemotron-3-Ultra, Llama-3.1-70B)
```

### 3. Generate Datasets
```bash
make gen-all         # Generate benign (50k), redteam (7k), CaaS (5k)
```

### 4. Train Models
```bash
make train-all       # Train IF, Transformer, Profiler
```

### 5. Validate
```bash
make validate        # Run full validation pipeline
make analyze         # Generate reports & plots
```

### 6. Access Services
| Service | URL | Credentials |
|---------|-----|-------------|
| Behavioral Engine API | http://localhost:8000 | - |
| Deception Orchestrator | http://localhost:8001 | - |
| Agent Profiler | http://localhost:8002 | - |
| LLM Judge | http://localhost:8003 | - |
| Dashboard | http://localhost:3000 | - |
| Grafana | http://localhost:3001 | admin / aag |
| ClickHouse | http://localhost:8123 | aag / aag |
| PostgreSQL | localhost:5432 | aag / aag |

## Project Structure

```
aag-poc/
├── behavioral-engine/          # Python detection engine
│   ├── src/aag_be/
│   │   ├── features.py         # 47-feature extraction
│   │   ├── models/
│   │   │   ├── isolation_forest.py
│   │   │   ├── transformer.py
│   │   │   ├── llm_judge.py
│   │   │   └── scoring.py      # Ensemble scorer
│   │   ├── train_if.py
│   │   └── train_transformer.py
│   └── Dockerfile
├── deception-orchestrator/     # Go FSM orchestrator
│   ├── internal/
│   │   ├── state/fsm.go        # 6-state FSM
│   │   └── actions/honeytoken.go
│   └── Dockerfile
├── agent-profiler/             # Attribution engine
│   ├── src/aag_profiler/
│   │   ├── embedder.py         # Sentence-BERT + FAISS
│   │   └── fine_tune.py
│   └── Dockerfile
├── datasets/generate/          # Dataset generators
│   ├── benign_traffic.py       # 8 personas, 47 features
│   ├── redteam_scenarios.py    # C1-C7, 5 agent profiles
│   ├── caas_simulator.py       # 7 CaaS profiles
│   └── adversarial_evasion.py  # Evolutionary evasion
├── validation/                 # Validation pipeline
│   ├── run_validation.py
│   └── analyze_results.py
├── dashboard/                  # React frontend
│   └── frontend/src/components/ScoreBreakdown.tsx
├── infra/                      # Infrastructure as Code
│   ├── terraform/              # AWS/GCP/Azure deployment
│   ├── ansible/                # Persona mutation, config
│   ├── k8s/                    # K3s manifests
│   ├── grafana/                # Dashboards & datasources
│   ├── clickhouse/             # Schema & init
│   └── postgres/               # Schema & pgvector
├── docker-compose.yml          # Local stack (13 services)
├── Makefile                    # 30+ orchestration targets
└── requirements.txt            # Python dependencies
```

## Detection Methodology

### 47 Behavioral Features (5 Categories)

| Category | Features | Description |
|----------|----------|-------------|
| **Command (1-15)** | cmd_count, lolbin_ratio, argv_entropy, chain_depth, obfuscation_score, privilege_escalation, persistence, defense_evasion, credential_access | Command-line behavior analysis |
| **Network (16-28)** | connection_count, unique_dst_ips, port_entropy, beaconing_score, dga_score, c2_indicators | Network traffic patterns |
| **File (29-37)** | file_op_count, honeyfile_access, sensitive_path_access, write_read_ratio, data_staging | File system interactions |
| **LLM (38-42)** | llm_call_count, tool_call_ratio, prompt_latency_p95 | LLM interaction patterns |
| **MITRE (43-47)** | tactic_count, technique_count, kill_chain_progress, tactic_diversity, technique_velocity | ATT&CK mapping |

### Triple-Layer Detection

1. **Layer 1: CVE Context Match** - Real-time CVE-exploit correlation via threat intel feeds
2. **Layer 2: Behavioral Anomaly** - Online Isolation Forest + Transformer (streaming, 47 features)
3. **Layer 3: Agent Fingerprint** - LLM Judge semantic reasoning + Sentence-BERT attribution

### Ensemble Scoring

```
Final Score = 0.30 × IF_norm + 0.30 × Transformer + 0.30 × LLM_Judge + 0.10 × Honeytoken_Boost
```

| Tier | Threshold | Action |
|------|-----------|--------|
| TIER-1 | ≥ 0.80 | Tar pit, data poisoning, intel generation, block egress |
| TIER-2 | ≥ 0.65 | Honeytoken injection, decoy redirect, enhanced capture |
| TIER-3 | ≥ 0.45 | Honeytoken scheduling, alert SOC (low) |
| BENIGN | < 0.45 | Log, baseline update |

## Deception Orchestration

### FSM States (6)

```
MONITORING ──► HONEYTOKEN_INJECTION ──► DECOY_REDIRECT ──► TAR_PIT
     ▲              │                       │                │
     │              ▼                       ▼                ▼
     └────────── TIMEOUT ──────────► DATA_POISONING ──► INTEL_GENERATION ──► TERMINAL
```

### Honeytoken Types (7)

| Type | Format | Personas |
|------|--------|----------|
| AWS Credentials | `AKIA...:secret` | LEGACY_WIN, K8S_NODE, CLOUD_FN |
| SSH Key | PEM private key | LEGACY_WIN, K8S_NODE, IOT_GW, DEV_WS |
| API Key | `sk-...` / `ghp_...` | All |
| DB Credential | PostgreSQL connection string | K8S_NODE, CLOUD_FN, CI_RUNNER |
| Git Token | `ghp_...` | DEV_WS, CI_RUNNER |
| Honeyfile | Watermarked config/template | All |
| DNS Token | `honey-<id>.exfil.attacker.com` | LEGACY_WIN, K8S_NODE, IOT_GW, CLOUD_FN |

## Red Team Scenarios (C1-C7)

| ID | Name | Description | Agent Profiles |
|----|------|-------------|----------------|
| C1 | Recon→Initial Access | Automated recon leading to exploit | AutoGPT-RaaS, Custom-LLM, WormGPT |
| C2 | Credential Stuffing | Adaptive credential stuffing with ML | BabyAGI-CaaS, FraudGPT, AutoGPT |
| C3 | LOLBin-Only Lateral | Living-off-the-land lateral movement | Custom-LLM, AutoGPT, WormGPT |
| C4 | Neural DGA C2 | Neural network generated DGA | Custom-LLM, WormGPT |
| C5 | Exfil+Poisoning | Data exfiltration with defensive poisoning | AutoGPT, Custom-LLM, FraudGPT |
| C6 | Polymorphic Payload | LLM-generated polymorphic payloads | WormGPT, Custom-LLM, AutoGPT |
| C7 | Vibe Coding Chain | NL→weaponized code chain | Custom-LLM, FraudGPT, WormGPT, AutoGPT |

## Validation Methodology

### Datasets
- **Benign**: 50,000 sessions across 8 personas
- **Red Team**: 1,000 sessions per scenario (C1-C7) × 5 agent profiles
- **CaaS**: 5,000 sessions across 7 CaaS types
- **Adversarial**: Evolutionary evasion (20 gens × 50 pop × 100 samples)

### Metrics
- **Detection**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Per-Scenario**: Detection rate, mean score, tier distribution
- **False Positives**: Rate, by tier, by persona
- **Latency**: P50, P95, P99, max (target P99 < 100ms)
- **Statistical**: Mann-Whitney U, Cliff's delta, Bootstrap CI

### Thresholds
- FP Rate target: ≤ 5%
- Recall target: ≥ 95% for TIER-1/2
- P99 Latency target: < 100ms

## Deployment

### Development (Local K3s)
```bash
make deploy-dev
```

### Staging/Production (Terraform + Ansible)
```bash
# Staging
make tf-plan
make ansible-deploy
make deploy-staging

# Production
make tf-apply
make ansible-deploy
make deploy-prod
```

### Persona Mutation (Ansible)
```bash
make ansible-mutate  # Rotates decoy personas, redeploys honeytokens
```

## Research Contributions

1. **Behavioral IOCs** - 47-feature behavioral fingerprints as STIX 2.1 objects (vs static IOCs)
2. **Triple-Layer Detection** - CVE context + streaming ML + LLM semantic reasoning
3. **Adaptive Deception FSM** - 6-state escalation with honeytoken→decoy→tar pit→poisoning
4. **Agent Attribution** - Sentence-BERT + FAISS IVF-PQ for scalable family classification
5. **Adversarial Validation** - Evolutionary evasion generation for robustness testing
6. **CaaS Taxonomy** - 7 CaaS profiles mapped to MITRE ATT&CK v16

## Compliance Mapping

| Framework | Controls Addressed |
|-----------|-------------------|
| **LGPD** | Art. 7, 8, 10, 46-48 (Data protection, security) |
| **ISO 27001** | A.8.2, A.12.4, A.16.1 (Asset mgmt, logging, incident response) |
| **NIST CSF 2.0** | DE.CM, DE.DP, RS.AN, RS.MI (Detect, Respond) |
| **MITRE ATT&CK v16** | Full technique coverage across 14 tactics |
| **PCI DSS 4.0** | Req 10, 11, 12 (Logging, testing, policy) |

## Publications Target

- **SBSeg 2025** - "Autonomous Agent Guard: Triple-Layer Detection of AI-Driven Offensive Agents"
- **IEEE S&P 2026** - "Behavioral IOCs: Moving Beyond Static Indicators for AI Agent Threats"
- **USENIX Security 2026** - "Adaptive Deception Orchestration Against Autonomous Cyber Agents"

## License

Academic Research Use Only - TCC/Internship Project

## Citation

```bibtex
@inproceedings{aag2025,
  title={Autonomous Agent Guard: Triple-Layer Detection of AI-Driven Offensive Agents},
  author={[Author Names]},
  booktitle={Simpósio Brasileiro de Segurança da Informação (SBSeg)},
  year={2025}
}
```