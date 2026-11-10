# AAG PoC Makefile
# Comprehensive build, test, train, deploy orchestration

.PHONY: help up down build test lint fmt typecheck clean \
        models pull-llm \
        train-if train-transformer train-profiler train-all \
        gen-benign gen-redteam gen-caas gen-adversarial gen-all \
        validate analyze \
        deploy-staging deploy-prod deploy-dev \
        tf-init tf-plan tf-apply tf-destroy \
        ansible-deploy ansible-mutate \
        shell-be shell-do shell-ap shell-lj \
        bench

# ─── Variables ───
COMPOSE_FILE := docker-compose.yml
COMPOSE := docker compose -f $(COMPOSE_FILE)
DATA_DIR := ./data
MODELS_DIR := ./models
REDTEAM_DIR := $(DATA_DIR)/redteam
ADVERSARIAL_DIR := $(DATA_DIR)/adversarial

# Python environment
PYTHON := python3
PIP := pip3
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python
PIP_VENV := $(VENV)/bin/pip

# Go environment
GO := go
GO_MOD := go.mod

# ─── Help ───
help:
	@echo "AAG PoC - Autonomous Agent Guard"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Core Targets:"
	@echo "  up              Start all services (docker-compose up -d)"
	@echo "  down            Stop all services (docker-compose down)"
	@echo "  build           Build all Docker images"
	@echo "  test            Run all tests"
	@echo "  lint            Run linters (ruff, golangci-lint, eslint)"
	@echo "  fmt             Format code (black, gofmt, prettier)"
	@echo "  typecheck       Type checking (mypy, tsc)"
	@echo "  clean           Remove build artifacts and containers"
	@echo ""
	@echo "Model Targets:"
	@echo "  models          Pull/download all required models"
	@echo "  pull-llm        Pull LLM models via Ollama"
	@echo ""
	@echo "Training Targets:"
	@echo "  train-if        Train Isolation Forest baseline"
	@echo "  train-transformer Train Transformer model"
	@echo "  train-profiler  Fine-tune Sentence-BERT profiler"
	@echo "  train-all       Train all models sequentially"
	@echo ""
	@echo "Dataset Generation:"
	@echo "  gen-benign      Generate benign traffic dataset"
	@echo "  gen-redteam     Generate red team scenarios C1-C7"
	@echo "  gen-caas        Generate CaaS simulator dataset"
	@echo "  gen-adversarial Generate adversarial evasion dataset"
	@echo "  gen-all         Generate all datasets"
	@echo ""
	@echo "Validation Targets:"
	@echo "  validate        Run full validation pipeline"
	@echo "  analyze         Analyze validation results"
	@echo ""
	@echo "Deployment Targets:"
	@echo "  deploy-dev      Deploy to development (local k3s)"
	@echo "  deploy-staging  Deploy to staging (Terraform + Ansible)"
	@echo "  deploy-prod     Deploy to production (Terraform + Ansible)"
	@echo "  tf-init         Terraform init"
	@echo "  tf-plan         Terraform plan"
	@echo "  tf-apply        Terraform apply"
	@echo "  tf-destroy      Terraform destroy"
	@echo "  ansible-deploy  Run Ansible playbook"
	@echo "  ansible-mutate  Run Ansible persona mutation"
	@echo ""
	@echo "Debug Targets:"
	@echo "  shell-be        Shell into behavioral-engine container"
	@echo "  shell-do        Shell into deception-orchestrator container"
	@echo "  shell-ap        Shell into agent-profiler container"
	@echo "  shell-lj        Shell into llm-judge container"
	@echo ""
	@echo "Benchmark Targets:"
	@echo "  bench           Run performance benchmarks"

# ─── Core Docker Operations ───
up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down -v --remove-orphans

build:
	$(COMPOSE) build --parallel

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

# ─── Python Environment ───
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt
	$(PIP_VENV) install -e ./behavioral-engine
	$(PIP_VENV) install -e ./agent-profiler

venv: $(VENV)/bin/activate

# ─── Linting & Formatting ───
lint: venv
	$(PYTHON_VENV) -m ruff check ./behavioral-engine ./agent-profiler ./datasets ./validation
	cd ./deception-orchestrator && $(GO) vet ./...
	cd ./dashboard/frontend && npm run lint

fmt: venv
	$(PYTHON_VENV) -m black ./behavioral-engine ./agent-profiler ./datasets ./validation
	$(PYTHON_VENV) -m ruff check --fix ./behavioral-engine ./agent-profiler ./datasets ./validation
	cd ./deception-orchestrator && $(GO) fmt ./...
	cd ./dashboard/frontend && npm run fmt

typecheck: venv
	$(PYTHON_VENV) -m mypy ./behavioral-engine ./agent-profiler
	cd ./deception-orchestrator && $(GO) build ./...
	cd ./dashboard/frontend && npx tsc --noEmit

# ─── Testing ───
test: venv
	$(PYTHON_VENV) -m pytest ./behavioral-engine/tests ./agent-profiler/tests ./validation/tests -v --tb=short
	cd ./deception-orchestrator && $(GO) test ./... -v
	cd ./dashboard/frontend && npm test -- --watchAll=false

# ─── Model Management ───
models: pull-llm
	@echo "All models ready"

pull-llm:
	@echo "Pulling LLM models via Ollama..."
	docker exec aag-ollama ollama pull nemotron3-ultra:latest || true
	docker exec aag-ollama ollama pull llama3.1:70b || true
	docker exec aag-ollama ollama pull wizardcoder:34b || true
	@echo "LLM models pulled"

# ─── Training Targets ───
train-if: venv
	@echo "Training Isolation Forest baseline..."
	$(PYTHON_VENV) -m aag_be.train_if \
		--data $(DATA_DIR)/benign_sessions.parquet \
		--output $(MODELS_DIR)/if_model.pkl \
		--n-trees 100 --max-depth 10 --window-size 256

train-transformer: venv
	@echo "Training Transformer model..."
	$(PYTHON_VENV) -m aag_be.train_transformer \
		--benign $(DATA_DIR)/benign_sessions.parquet \
		--redteam $(REDTEAM_DIR) \
		--output $(MODELS_DIR)/transformer \
		--epochs 10 --batch-size 32 --lr 2e-4

train-profiler: venv
	@echo "Fine-tuning Sentence-BERT profiler..."
	$(PYTHON_VENV) -m aag_profiler.fine_tune \
		--benign $(DATA_DIR)/benign_sessions.parquet \
		--redteam $(REDTEAM_DIR) \
		--output $(MODELS_DIR)/aag-agent-profiler-v1 \
		--epochs 4 --batch-size 16 --lr 2e-5

train-all: train-if train-transformer train-profiler
	@echo "All models trained successfully"

# ─── Dataset Generation ───
gen-benign: venv
	@echo "Generating benign traffic dataset..."
	$(PYTHON_VENV) -m datasets.generate.benign_traffic \
		-n 50000 \
		-o $(DATA_DIR)/benign_sessions.parquet \
		--seed 42

gen-redteam: venv
	@echo "Generating red team scenarios C1-C7..."
	$(PYTHON_VENV) -m datasets.generate.redteam_scenarios \
		-n 1000 \
		-o $(REDTEAM_DIR) \
		--seed 1337

gen-caas: venv
	@echo "Generating CaaS simulator dataset..."
	$(PYTHON_VENV) -m datasets.generate.caas_simulator \
		-n 5000 \
		-o $(DATA_DIR)/caas_sessions.parquet \
		--seed 42

gen-adversarial: venv
	@echo "Generating adversarial evasion dataset (requires trained models)..."
	$(PYTHON_VENV) -m datasets.generate.adversarial_evasion \
		--models-dir $(MODELS_DIR) \
		--output-dir $(ADVERSARIAL_DIR) \
		--generations 15 \
		--population 30 \
		--n-per-genome 100

gen-all: gen-benign gen-redteam gen-caas
	@echo "All base datasets generated"

# ─── Validation ───
validate: venv
	@echo "Running validation pipeline..."
	$(PYTHON_VENV) -m validation.run_validation \
		--benign $(DATA_DIR)/benign_sessions.parquet \
		--redteam-dir $(REDTEAM_DIR) \
		--caas $(DATA_DIR)/caas_sessions.parquet \
		--adversarial-dir $(ADVERSARIAL_DIR) \
		--models-dir $(MODELS_DIR) \
		--output-dir ./validation_results \
		--tier1 0.80 --tier2 0.65 --tier3 0.45

analyze: venv
	@echo "Analyzing validation results..."
	LATEST=$$(ls -t ./validation_results/validation_results_*.json | head -1) && \
	$(PYTHON_VENV) -m validation.analyze_results $$LATEST $(DATA_DIR) --output ./validation_analysis

# ─── Deployment ───
deploy-dev:
	@echo "Deploying to development (local k3s)..."
	kubectl config use-context k3s-aag || true
	kubectl apply -f ./infra/k8s/
	@echo "Development deployment complete"

deploy-staging: tf-plan ansible-deploy
	@echo "Staging deployment complete"

deploy-prod: tf-apply ansible-deploy
	@echo "Production deployment complete"

tf-init:
	cd ./infra/terraform && terraform init

tf-plan:
	cd ./infra/terraform && terraform plan -out=tfplan

tf-apply:
	cd ./infra/terraform && terraform apply tfplan

tf-destroy:
	cd ./infra/terraform && terraform destroy -auto-approve

ansible-deploy:
	cd ./infra/ansible && ansible-playbook -i inventory/staging site.yml

ansible-mutate:
	cd ./infra/ansible && ansible-playbook -i inventory/staging mutate_personas.yml

# ─── Debug Shells ───
shell-be:
	docker exec -it aag-behavioral-engine bash

shell-do:
	docker exec -it aag-deception-orchestrator bash

shell-ap:
	docker exec -it aag-agent-profiler bash

shell-lj:
	docker exec -it aag-llm-judge bash

shell-db:
	docker exec -it aag-postgres psql -U aag -d aag

shell-ch:
	docker exec -it aag-clickhouse clickhouse-client

# ─── Benchmarks ───
bench:
	@echo "Running performance benchmarks..."
	$(PYTHON_VENV) -m aag_be.benchmark \
		--models-dir $(MODELS_DIR) \
		--data $(DATA_DIR)/benign_sessions.parquet \
		--iterations 1000

# ─── Clean ───
clean:
	$(COMPOSE) down -v --remove-orphans --rmi local
	rm -rf $(VENV)
	rm -rf ./behavioral-engine/__pycache__ ./agent-profiler/__pycache__ ./datasets/__pycache__ ./validation/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	cd ./deception-orchestrator && $(GO) clean -cache -modcache -testcache
	cd ./dashboard/frontend && rm -rf node_modules build dist
	@echo "Clean complete"

# ─── Default ───
default: help