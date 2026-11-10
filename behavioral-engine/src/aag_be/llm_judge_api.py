"""LLM Judge FastAPI Service with Swagger UI."""

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

from aag_be.models.llm_judge import (
    LLMJudgeClient, JudgeResponse, Classification, DeceptionAction,
    SYSTEM_PROMPT_V1, ADVERSARIAL_SYSTEM_PROMPT, _build_prompt, _parse_response
)
from aag_be.features import FEATURE_NAMES
from aag_be.dlp import (
    DLPEngine, DLPAction, DLPViolationType, DLPResult, DLPRule,
    create_dlp_engine, HoneytokenGenerator, DLPMiddleware
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client instance
llm_client: LLMJudgeClient | None = None

# Global DLP engine instance
dlp_engine: DLPEngine | None = None
dlp_middleware: DLPMiddleware | None = None


# ─── Pydantic Models for API ───

class JudgeRequest(BaseModel):
    features: list[float]
    commands: list[dict]
    network: list[dict]
    files: list[dict]
    llm_interactions: list[dict] = []
    mitre_techniques: list[str] = []
    adversarial_mode: bool = False


class JudgeResponseAPI(BaseModel):
    classification: str
    confidence: float
    reasoning: str
    mitre_techniques: list[str]
    deception_action: str
    behavioral_iocs: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    ollama_connected: bool


class ModelInfoResponse(BaseModel):
    model: str
    base_url: str
    adversarial_mode: bool
    system_prompt_length: int


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


# ─── Helper Functions ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global llm_client
    try:
        llm_client = LLMJudgeClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            model=os.getenv("MODEL", "nemotron3-ultra:latest"),
            timeout=30.0,
        )
        # Test connection
        async with llm_client:
            pass
        logger.info("LLM Judge client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Judge: {e}")
    yield
    # Shutdown
    logger.info("Shutting down LLM Judge...")


# ─── FastAPI App ───

app = FastAPI(
    title="AAG LLM Judge API",
    description="Semantic reasoning on agent behavior using LLM (Ollama/vLLM)",
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
        status="healthy" if llm_client is not None else "degraded",
        version="1.0.0",
        model=os.getenv("MODEL", "nemotron3-ultra:latest"),
        ollama_connected=llm_client is not None,
    )


@app.post("/judge", response_model=JudgeResponseAPI, tags=["Judge"])
async def judge_session(request: JudgeRequest = Body(...)):
    """
    Classify a session using LLM semantic reasoning.

    Returns classification (BENIGN/SCRIPTED/AGENTIC), confidence, reasoning,
    MITRE techniques, recommended deception action, and behavioral IOCs.
    """
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    try:
        # Build prompt
        prompt = _build_prompt(
            features=request.features,
            feature_names=FEATURE_NAMES,
            commands=request.commands,
            network=request.network,
            files=request.files,
            llm_interactions=request.llm_interactions,
            mitre_techniques=request.mitre_techniques,
            adversarial=request.adversarial_mode,
        )

        system_prompt = ADVERSARIAL_SYSTEM_PROMPT if request.adversarial_mode else SYSTEM_PROMPT_V1

        payload = {
            "model": llm_client.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "options": {"temperature": 0.1, "top_p": 0.9},
            "stream": False,
        }

        async with llm_client:
            resp = await llm_client._client.post(f"{llm_client.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("message", {}).get("content", "")

        judge_response = _parse_response(response_text)

        return JudgeResponseAPI(
            classification=judge_response.classification.value,
            confidence=judge_response.confidence,
            reasoning=judge_response.reasoning,
            mitre_techniques=judge_response.mitre_techniques,
            deception_action=judge_response.deception_action.value,
            behavioral_iocs=judge_response.behavioral_iocs,
        )
    except Exception as e:
        logger.error(f"Judge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info", response_model=ModelInfoResponse, tags=["Models"])
async def model_info():
    """Get information about the LLM model configuration."""
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    return ModelInfoResponse(
        model=llm_client.model,
        base_url=llm_client.base_url,
        adversarial_mode=llm_client.adversarial_mode,
        system_prompt_length=len(ADVERSARIAL_SYSTEM_PROMPT),
    )


@app.post("/judge/batch", tags=["Judge"])
async def judge_batch(requests: list[JudgeRequest] = Body(...)):
    """
    Classify multiple sessions in batch.

    Returns list of classifications.
    """
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    results = []
    for req in requests:
        try:
            result = await judge_session(req)
            results.append({"success": True, "result": result})
        except Exception as e:
            results.append({"success": False, "error": str(e)})

    return {"results": results}


@app.get("/prompt/template", tags=["Debug"])
async def prompt_template():
    """Get the system prompt template for debugging."""
    return {
        "system_prompt_v1": SYSTEM_PROMPT_V1,
        "adversarial_system_prompt": ADVERSARIAL_SYSTEM_PROMPT,
        "feature_names": FEATURE_NAMES,
    }


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

    # Recreate engine if profile changed
    stats = dlp_engine.get_rule_stats()
    if stats.get("_profile") != request.profile or stats.get("_honeytoken") != request.honeytoken_enabled:
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


# ─── Main ───

if __name__ == "__main__":
    uvicorn.run(
        "aag_be.llm_judge_api:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info",
    )