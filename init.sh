#!/bin/bash
# AAG PoC - Automated Initialization Script
# Runs inside containers to generate data, train models, and validate

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="/data"
MODELS_DIR="/models"
REDTEAM_DIR="${DATA_DIR}/redteam"
ADVERSARIAL_DIR="${DATA_DIR}/adversarial"
VALIDATION_DIR="/validation_results"

# Flags
GENERATE_DATA="${GENERATE_DATA:-true}"
TRAIN_MODELS="${TRAIN_MODELS:-true}"
RUN_VALIDATION="${RUN_VALIDATION:-true}"
PULL_LLM="${PULL_LLM:-true}"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=60
    local attempt=1

    log_info "Waiting for $service at $host:$port..."
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            log_error "$service not available after $max_attempts attempts"
            return 1
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    log_success "$service is ready"
}

# Pull LLM models via Ollama
pull_llm_models() {
    if [ "$PULL_LLM" != "true" ]; then
        log_info "Skipping LLM model pull (PULL_LLM=false)"
        return 0
    fi

    log_info "Pulling LLM models via Ollama..."
    wait_for_service ollama 11434 "Ollama"

    models=(
        "nemotron3-ultra:latest"
        "llama3.1:70b"
        "wizardcoder:34b"
    )

    for model in "${models[@]}"; do
        log_info "Pulling $model..."
        if ollama pull "$model" 2>&1 | tail -5; then
            log_success "Pulled $model"
        else
            log_warn "Failed to pull $model (may already exist)"
        fi
    done
}

# Generate benign traffic dataset
generate_benign() {
    log_info "Generating benign traffic dataset..."
    python -m datasets.generate.benign_traffic \
        -n 50000 \
        -o "${DATA_DIR}/benign_sessions.parquet" \
        --seed 42
    log_success "Benign dataset generated"
}

# Generate red team scenarios
generate_redteam() {
    log_info "Generating red team scenarios C1-C7..."
    python -m datasets.generate.redteam_scenarios \
        -n 1000 \
        -o "${REDTEAM_DIR}" \
        --seed 1337
    log_success "Red team datasets generated"
}

# Generate CaaS simulator dataset
generate_caas() {
    log_info "Generating CaaS simulator dataset..."
    python -m datasets.generate.caas_simulator \
        -n 5000 \
        -o "${DATA_DIR}/caas_sessions.parquet" \
        --seed 42
    log_success "CaaS dataset generated"
}

# Generate all datasets
generate_all_data() {
    if [ "$GENERATE_DATA" != "true" ]; then
        log_info "Skipping data generation (GENERATE_DATA=false)"
        return 0
    fi

    log_info "=== Generating All Datasets ==="
    mkdir -p "$DATA_DIR" "$REDTEAM_DIR" "$ADVERSARIAL_DIR"

    generate_benign
    generate_redteam
    generate_caas

    log_success "All datasets generated"
}

# Train Isolation Forest
train_if() {
    log_info "Training Isolation Forest baseline..."
    python -m aag_be.train_if \
        --data "${DATA_DIR}/benign_sessions.parquet" \
        --output "${MODELS_DIR}/if_model.pkl" \
        --n-trees 100 \
        --max-depth 10 \
        --window-size 256
    log_success "Isolation Forest trained"
}

# Train Transformer
train_transformer() {
    log_info "Training Transformer model..."
    python -m aag_be.train_transformer \
        --benign "${DATA_DIR}/benign_sessions.parquet" \
        --redteam "${REDTEAM_DIR}" \
        --output "${MODELS_DIR}/transformer" \
        --epochs 10 \
        --batch-size 32 \
        --lr 2e-4
    log_success "Transformer trained"
}

# Train Agent Profiler
train_profiler() {
    log_info "Fine-tuning Sentence-BERT profiler..."
    python -m aag_profiler.fine_tune \
        --benign "${DATA_DIR}/benign_sessions.parquet" \
        --redteam "${REDTEAM_DIR}" \
        --output "${MODELS_DIR}/aag-agent-profiler-v1" \
        --epochs 4 \
        --batch-size 16 \
        --lr 2e-5
    log_success "Agent Profiler trained"
}

# Train all models
train_all_models() {
    if [ "$TRAIN_MODELS" != "true" ]; then
        log_info "Skipping model training (TRAIN_MODELS=false)"
        return 0
    fi

    log_info "=== Training All Models ==="
    mkdir -p "$MODELS_DIR"

    train_if
    train_transformer
    train_profiler

    log_success "All models trained"
}

# Run validation pipeline
run_validation() {
    if [ "$RUN_VALIDATION" != "true" ]; then
        log_info "Skipping validation (RUN_VALIDATION=false)"
        return 0
    fi

    log_info "=== Running Validation Pipeline ==="
    mkdir -p "$VALIDATION_DIR"

    python -m validation.run_validation \
        --benign "${DATA_DIR}/benign_sessions.parquet" \
        --redteam-dir "${REDTEAM_DIR}" \
        --caas "${DATA_DIR}/caas_sessions.parquet" \
        --adversarial-dir "${ADVERSARIAL_DIR}" \
        --models-dir "${MODELS_DIR}" \
        --output-dir "${VALIDATION_DIR}" \
        --tier1 0.80 \
        --tier2 0.65 \
        --tier3 0.45

    log_success "Validation complete"

    # Analyze results
    log_info "Analyzing validation results..."
    LATEST=$(ls -t "${VALIDATION_DIR}"/validation_results_*.json | head -1)
    if [ -n "$LATEST" ]; then
        python -m validation.analyze_results "$LATEST" "$DATA_DIR" --output "${VALIDATION_DIR}/analysis"
        log_success "Analysis complete"
    fi
}

# Main execution
main() {
    log_info "=== AAG PoC Initialization Started ==="
    log_info "Data dir: $DATA_DIR"
    log_info "Models dir: $MODELS_DIR"

    # Check if already initialized
    if [ -f "${MODELS_DIR}/.initialized" ]; then
        log_info "Already initialized, skipping..."
        exit 0
    fi

    # Pull LLM models first (needed for adversarial generation)
    pull_llm_models

    # Generate datasets
    generate_all_data

    # Train models
    train_all_models

    # Run validation
    run_validation

    # Mark as initialized
    touch "${MODELS_DIR}/.initialized"
    log_success "=== AAG PoC Initialization Complete ==="
}

# Run main
main "$@"