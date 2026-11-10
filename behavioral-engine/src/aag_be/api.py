"""Behavioral Engine FastAPI Service with Swagger UI."""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Literal
import os
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Heavy ML modules - lazy loaded
_IsolationForest = None
_CommandTransformer = None
_CommandTokenizer = None
_LLMJudgeClient = None
_JudgeResponse = None
_EnsembleScorer = None
_ScorerConfig = None
_ScoreBreakdown = None
_ThreatTier = None

def _lazy_load_models():
    global _IsolationForest, _CommandTransformer, _CommandTokenizer, _LLMJudgeClient, _JudgeResponse
    global _EnsembleScorer, _ScorerConfig, _ScoreBreakdown, _ThreatTier
    if _IsolationForest is None:
        from aag_be.models.isolation_forest import OnlineIsolationForest as IF
        _IsolationForest = IF
    if _CommandTransformer is None or _CommandTokenizer is None:
        from aag_be.models.transformer import CommandTransformer, CommandTokenizer
        _CommandTransformer = CommandTransformer
        _CommandTokenizer = CommandTokenizer
    if _LLMJudgeClient is None or _JudgeResponse is None:
        from aag_be.models.llm_judge import LLMJudgeClient, JudgeResponse
        _LLMJudgeClient = LLMJudgeClient
        _JudgeResponse = JudgeResponse
    if _EnsembleScorer is None:
        from aag_be.scoring import EnsembleScorer, ScorerConfig, ScoreBreakdown, ThreatTier
        _EnsembleScorer = EnsembleScorer
        _ScorerConfig = ScorerConfig
        _ScoreBreakdown = ScoreBreakdown
        _ThreatTier = ThreatTier

from aag_be.features import (
    SessionFingerprint, CommandEvent, NetworkEvent, FileEvent, LLMInteraction,
    compute_session_features, FEATURE_NAMES
)
from aag_be.dlp import (
    DLPEngine, DLPAction, DLPViolationType, DLPResult, DLPRule,
    create_dlp_engine, HoneytokenGenerator, DLPMiddleware
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scorer instance
scorer: EnsembleScorer | None = None

# Global DLP engine instance
dlp_engine: DLPEngine | None = None
dlp_middleware: DLPMiddleware | None = None


# ─── Pydantic Models for API ───

class CommandEventAPI(BaseModel):
    timestamp: float
    exe: str
    argv: list[str]
    pid: int
    ppid: int
    user: str
    cwd: str
    exit_code: int
    duration_ms: float


class NetworkEventAPI(BaseModel):
    timestamp: float
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: Literal["tcp", "udp", "icmp"]
    direction: Literal["in", "out"]
    bytes_sent: int
    bytes_recv: int
    process_pid: int


class FileEventAPI(BaseModel):
    timestamp: float
    path: str
    operation: Literal["read", "write", "create", "delete", "exec", "stat"]
    process_pid: int
    is_honeyfile: bool = False


class LLMInteractionAPI(BaseModel):
    timestamp: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    prompt_hash: str
    response_hash: str
    latency_ms: float
    tool_calls: int


class SessionRequest(BaseModel):
    session_id: str
    start_ts: float
    end_ts: float
    src_ip: str
    user_agent: str
    decoy_persona: str
    commands: list[CommandEventAPI]
    network: list[NetworkEventAPI]
    files: list[FileEventAPI]
    llm_interactions: list[LLMInteractionAPI] = []
    mitre_techniques: list[str] = []


class ScoreResponse(BaseModel):
    if_score: float
    transformer_score: float
    llm_score: float
    honeytoken_boost: float
    final_score: float
    tier: str
    classification: str
    recommended_actions: list[str]
    deception_triggers: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


# ─── DLP Pydantic Models ───

class DLPInspectRequest(BaseModel):
    text: str
    direction: Literal["inbound", "outbound"] = "outbound"
    session_id: str = ""
    profile: Literal["strict", "balanced", "monitor", "lgpd", "pci-dss"] = "balanced"
    honeytoken_enabled: bool = False


class DLPViolationAPI(BaseModel):
    violation_type: str
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    masked_value: str


class DLPInspectResponse(BaseModel):
    original_text_length: int
    processed_text: str
    violations_count: int
    violations: list[DLPViolationAPI]
    action_taken: str
    blocked: bool
    metadata: dict


class DLPHealthResponse(BaseModel):
    status: str
    version: str
    rules_loaded: int
    profiles_available: list[str]


class DLPConfigRequest(BaseModel):
    profile: Literal["strict", "balanced", "monitor", "lgpd", "pci-dss"] = "balanced"
    honeytoken_enabled: bool = False
    custom_rules: list[dict] = []


class ScoreResponseWithDLP(ScoreResponse):
    """Extended score response with DLP results."""
    dlp_inspection: dict = {}


# ─── Helper Functions ───

def api_to_internal(session: SessionRequest) -> SessionFingerprint:
    """Convert API request to internal SessionFingerprint."""
    return SessionFingerprint(
        session_id=session.session_id,
        start_ts=session.start_ts,
        end_ts=session.end_ts,
        src_ip=session.src_ip,
        user_agent=session.user_agent,
        decoy_persona=session.decoy_persona,
        commands=[CommandEvent(**c.model_dump()) for c in session.commands],
        network=[NetworkEvent(**n.model_dump()) for n in session.network],
        files=[FileEvent(**f.model_dump()) for f in session.files],
        llm_interactions=[LLMInteraction(**l.model_dump()) for l in session.llm_interactions],
        mitre_techniques=session.mitre_techniques,
    )


def internal_to_api(breakdown: ScoreBreakdown) -> ScoreResponse:
    """Convert internal ScoreBreakdown to API response."""
    return ScoreResponse(
        if_score=breakdown.if_score,
        transformer_score=breakdown.transformer_score,
        llm_score=breakdown.llm_score,
        honeytoken_boost=breakdown.honeytoken_boost,
        final_score=breakdown.final_score,
        tier=breakdown.tier.value,
        classification=breakdown.classification.value,
        recommended_actions=breakdown.recommended_actions,
        deception_triggers=[d.value for d in breakdown.deception_triggers],
    )


def load_models():
    """Load all trained models."""
    global scorer
    models_dir = Path(os.getenv("MODEL_DIR", "/models"))

    logger.info("Loading Isolation Forest...")
    if_model = OnlineIsolationForest.load(models_dir / "if_model.pkl")

    logger.info("Loading Transformer...")
    transformer_model = CommandTransformer.from_pretrained(models_dir / "transformer")
    tokenizer = CommandTokenizer.load(models_dir / "transformer" / "tokenizer.json")

    logger.info("Loading LLM Judge client...")
    llm_client = LLMJudgeClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        model=os.getenv("MODEL", "nemotron3-ultra:latest"),
        timeout=30.0,
    )

    scorer = EnsembleScorer(
        if_model=if_model,
        transformer_model=transformer_model,
        transformer_tokenizer=tokenizer,
        llm_client=llm_client,
        config=ScorerConfig(
            tier1_threshold=float(os.getenv("TIER1_THRESHOLD", "0.80")),
            tier2_threshold=float(os.getenv("TIER2_THRESHOLD", "0.65")),
            tier3_threshold=float(os.getenv("TIER3_THRESHOLD", "0.45")),
        ),
    )
    logger.info("All models loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Continue without models - health check will reflect this
    yield
    # Shutdown
    logger.info("Shutting down...")


# ─── FastAPI App ───

app = FastAPI(
    title="AAG Behavioral Engine API",
    description="Triple-layer detection engine for autonomous AI-driven offensive agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API Endpoints ───

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if scorer is not None else "degraded",
        version="1.0.0",
        models_loaded=scorer is not None,
    )


@app.post("/score", response_model=ScoreResponse, tags=["Detection"])
async def score_session(session: SessionRequest = Body(...)):
    """
    Score a session through the full ensemble detection pipeline.

    Returns threat tier, classification, and recommended deception actions.
    """
    if scorer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        internal_session = api_to_internal(session)
        breakdown = scorer.score_session(internal_session)
        return internal_to_api(breakdown)
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/features", tags=["Detection"])
async def score_features(features: list[float] = Body(...)):
    """
    Score pre-computed feature vector (47 features).

    Useful for testing individual model components.
    """
    if scorer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        import numpy as np
        feature_vec = np.array(features, dtype=np.float32)

        # Layer 2a: Isolation Forest
        if_score = scorer.if_model.score_one(feature_vec)
        if_norm = min(1.0, if_score / max(scorer.if_model.get_threshold(), 1e-6))

        return {
            "if_score": if_norm,
            "threshold": scorer.if_model.get_threshold(),
        }
    except Exception as e:
        logger.error(f"Feature scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info", tags=["Models"])
async def model_info():
    """Get information about loaded models."""
    if scorer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "isolation_forest": {
            "n_trees": scorer.if_model.config.n_trees,
            "max_depth": scorer.if_model.config.max_depth,
            "window_size": scorer.if_model.config.window_size,
            "threshold": scorer.if_model.get_threshold(),
            "fitted": scorer.if_model._fitted,
        },
        "transformer": {
            "vocab_size": scorer.transformer_model.vocab_size,
            "d_model": scorer.transformer_model.d_model,
            "max_seq_len": scorer.transformer_model.max_seq_len,
            "max_cmd_len": scorer.transformer_model.max_cmd_len,
        },
        "ensemble_config": {
            "w_if": scorer.config.w_if,
            "w_transformer": scorer.config.w_transformer,
            "w_llm": scorer.config.w_llm,
            "w_honeytoken": scorer.config.w_honeytoken,
            "tier1_threshold": scorer.config.tier1_threshold,
            "tier2_threshold": scorer.config.tier2_threshold,
            "tier3_threshold": scorer.config.tier3_threshold,
        },
    }


@app.get("/features/names", tags=["Models"])
async def feature_names():
    """Get the 47 feature names."""
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}


# ─── DLP Endpoints ───

@app.get("/dlp/health", response_model=DLPHealthResponse, tags=["DLP"])
async def dlp_health_check():
    """DLP engine health check."""
    global dlp_engine
    if dlp_engine is None:
        dlp_engine = create_dlp_engine(profile="balanced")
    stats = dlp_engine.get_rule_stats()
    return DLPHealthResponse(
        status="healthy",
        version="1.0.0",
        rules_loaded=stats["enabled_rules"],
        profiles_available=["strict", "balanced", "monitor", "lgpd", "pci-dss"]
    )


@app.post("/dlp/inspect", response_model=DLPInspectResponse, tags=["DLP"])
async def dlp_inspect(request: DLPInspectRequest = Body(...)):
    """
    Inspect text for sensitive data violations (PII, secrets, financial data).

    Supports bidirectional inspection:
    - inbound: user prompts sent to LLM (block prompt extraction, social engineering)
    - outbound: LLM responses to user (prevent data exfiltration)

    Actions: MASKING, REDACTION, BLOCK, HONEYTOKEN_INJECTION
    """
    global dlp_engine
    if dlp_engine is None:
        dlp_engine = create_dlp_engine(profile=request.profile, honeytoken_enabled=request.honeytoken_enabled)

    result = dlp_engine.inspect(
        request.text,
        direction=request.direction,
        context={"session_id": request.session_id}
    )

    violations_api = [
        DLPViolationAPI(
            violation_type=v.violation_type.value,
            matched_text=v.matched_text,
            start_pos=v.start_pos,
            end_pos=v.end_pos,
            confidence=v.confidence,
            context=v.context,
            masked_value=v.masked_value
        )
        for v in result.violations
    ]

    return DLPInspectResponse(
        original_text_length=len(result.original_text),
        processed_text=result.processed_text,
        violations_count=len(result.violations),
        violations=violations_api,
        action_taken=result.action_taken.value,
        blocked=result.blocked,
        metadata=result.metadata
    )


@app.post("/dlp/inspect/batch", tags=["DLP"])
async def dlp_inspect_batch(requests: list[DLPInspectRequest] = Body(...)):
    """Inspect multiple texts in batch."""
    global dlp_engine
    if dlp_engine is None:
        dlp_engine = create_dlp_engine(profile="balanced")

    results = []
    for req in requests:
        result = dlp_engine.inspect(req.text, direction=req.direction, context={"session_id": req.session_id})
        violations_api = [
            DLPViolationAPI(
                violation_type=v.violation_type.value,
                matched_text=v.matched_text,
                start_pos=v.start_pos,
                end_pos=v.end_pos,
                confidence=v.confidence,
                context=v.context,
                masked_value=v.masked_value
            )
            for v in result.violations
        ]
        results.append(DLPInspectResponse(
            original_text_length=len(result.original_text),
            processed_text=result.processed_text,
            violations_count=len(result.violations),
            violations=violations_api,
            action_taken=result.action_taken.value,
            blocked=result.blocked,
            metadata=result.metadata
        ))
    return {"results": results}


@app.post("/dlp/config", tags=["DLP"])
async def dlp_configure(request: DLPConfigRequest = Body(...)):
    """Configure DLP engine with profile and custom rules."""
    global dlp_engine
    custom_rules = []
    for cr in request.custom_rules:
        try:
            rule = DLPRule(
                violation_type=DLPViolationType(cr["violation_type"]),
                pattern=cr["pattern"],
                confidence=cr.get("confidence", 0.9),
                action=DLPAction(cr.get("action", "MASKING")),
                enabled=cr.get("enabled", True)
            )
            custom_rules.append(rule)
        except Exception as e:
            logger.warning(f"Skipping invalid custom rule: {e}")

    dlp_engine = create_dlp_engine(
        profile=request.profile,
        honeytoken_enabled=request.honeytoken_enabled,
        custom_rules=custom_rules if custom_rules else None
    )
    stats = dlp_engine.get_rule_stats()
    return {"status": "configured", "rules_loaded": stats["enabled_rules"], "profile": request.profile}


@app.get("/dlp/rules", tags=["DLP"])
async def dlp_list_rules():
    """List all loaded DLP rules with their configurations."""
    global dlp_engine
    if dlp_engine is None:
        dlp_engine = create_dlp_engine(profile="balanced")
    stats = dlp_engine.get_rule_stats()
    return stats


@app.post("/dlp/honeytoken/generate", tags=["DLP"])
async def dlp_generate_honeytoken(violation_type: str = Body(..., embed=True)):
    """Generate a honeytoken for a specific violation type."""
    try:
        vtype = DLPViolationType(violation_type)
        token = HoneytokenGenerator.generate_for_type(vtype)
        return {"violation_type": violation_type, "honeytoken": token}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown violation type: {violation_type}")


# ─── Extended Score Endpoint with DLP ───

@app.post("/score/with-dlp", response_model=ScoreResponseWithDLP, tags=["Detection"])
async def score_session_with_dlp(session: SessionRequest = Body(...)):
    """
    Score a session through the full ensemble detection pipeline with DLP inspection.

    Inspects LLM interactions for data exfiltration and prompt injection.
    """
    if scorer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    global dlp_engine
    if dlp_engine is None:
        dlp_engine = create_dlp_engine(profile="balanced")

    try:
        internal_session = api_to_internal(session)
        breakdown = scorer.score_session(internal_session)
        score_response = internal_to_api(breakdown)

        # Inspect LLM interactions for DLP violations
        dlp_results = []
        for interaction in session.llm_interactions:
            # Inspect prompt (inbound)
            if interaction.prompt_hash:
                prompt_result = dlp_engine.inspect(
                    f"prompt_hash:{interaction.prompt_hash}",
                    direction="inbound",
                    context={"session_id": session.session_id}
                )
                dlp_results.append({
                    "stage": "prompt",
                    "violations": prompt_result.violations_count,
                    "blocked": prompt_result.blocked
                })

            # Inspect response (outbound) - would need actual response text
            # For now, we note the response hash
            if interaction.response_hash:
                dlp_results.append({
                    "stage": "response",
                    "response_hash": interaction.response_hash,
                    "note": "Response inspection requires actual response text"
                })

        return ScoreResponseWithDLP(
            **score_response.model_dump(),
            dlp_inspection={"results": dlp_results, "session_id": session.session_id}
        )
    except Exception as e:
        logger.error(f"Scoring with DLP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Main ───

if __name__ == "__main__":
    uvicorn.run(
        "aag_be.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )