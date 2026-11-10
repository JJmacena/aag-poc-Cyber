"""Agent Profiler FastAPI Service with Swagger UI."""

from __future__ import annotations
from contextlib import asynccontextmanager
import os
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

from aag_profiler.embedder import AgentProfiler, AgentFamily, DEFAULT_FAMILIES, fingerprint_to_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global profiler instance
profiler: AgentProfiler | None = None


# ─── Pydantic Models for API ───

class SessionRequest(BaseModel):
    session_id: str
    start_ts: float
    end_ts: float
    src_ip: str
    user_agent: str
    decoy_persona: str
    command_sequence: list[dict]
    network: list[dict]
    files: list[dict]
    llm_interactions: list[dict] = []
    mitre_techniques: list[str] = []


class AttributeResponse(BaseModel):
    family: str
    confidence: float
    ranked_families: list[dict]


class AddFamilyRequest(BaseModel):
    family_name: str
    sessions: list[SessionRequest]
    family_meta: dict | None = None
    rebuild: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    index_loaded: bool
    families_count: int


class ModelInfoResponse(BaseModel):
    model_path: str
    dimension: int
    index_type: str
    families: list[str]


# ─── Helper Functions ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global profiler
    try:
        model_dir = Path(os.getenv("MODEL_DIR", "/models/aag-agent-profiler-v1"))
        index_path = model_dir / "agent_families.index"
        families_path = model_dir / "family_metadata.json"

        if index_path.exists() and families_path.exists():
            profiler = AgentProfiler(
                model_path=str(model_dir),
                index_path=str(index_path),
                families_path=str(families_path),
                device=os.getenv("DEVICE", "cuda"),
            )
            logger.info("Agent Profiler loaded successfully")
        else:
            logger.warning("Model or index not found, profiler will be unavailable")
    except Exception as e:
        logger.error(f"Failed to load profiler: {e}")
    yield
    # Shutdown
    logger.info("Shutting down Agent Profiler...")


# ─── FastAPI App ───

app = FastAPI(
    title="AAG Agent Profiler API",
    description="Behavioral fingerprinting & agent family attribution via Sentence-BERT + FAISS",
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
        status="healthy" if profiler is not None else "degraded",
        version="1.0.0",
        model_loaded=profiler is not None and profiler.model is not None,
        index_loaded=profiler is not None and profiler.index is not None,
        families_count=len(profiler.families) if profiler else 0,
    )


@app.post("/attribute", response_model=AttributeResponse, tags=["Attribution"])
async def attribute_session(session: SessionRequest = Body(...)):
    """
    Attribute a session to an agent family via k-NN on behavioral embeddings.

    Returns top family, confidence, and ranked alternatives.
    """
    if profiler is None or profiler.index is None:
        raise HTTPException(status_code=503, detail="Profiler not loaded")

    try:
        # Convert to dict format expected by profiler
        session_dict = session.model_dump()

        family, confidence, ranked = profiler.attribute(
            session_dict,
            k=5,
            threshold=0.7,
        )

        return AttributeResponse(
            family=family,
            confidence=confidence,
            ranked_families=[{"family": f, "score": s} for f, s in ranked],
        )
    except Exception as e:
        logger.error(f"Attribution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", tags=["Embeddings"])
async def embed_sessions(sessions: list[SessionRequest] = Body(...)):
    """
    Generate embeddings for multiple sessions.

    Returns list of embedding vectors.
    """
    if profiler is None:
        raise HTTPException(status_code=503, detail="Profiler not loaded")

    try:
        session_dicts = [s.model_dump() for s in sessions]
        embeddings = profiler.embed(session_dicts)

        return {
            "embeddings": embeddings.tolist(),
            "dimension": embeddings.shape[1],
            "count": embeddings.shape[0],
        }
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/families/add", tags=["Families"])
async def add_family(request: AddFamilyRequest = Body(...)):
    """
    Add a new agent family to the profiler index.

    Requires example sessions and optional family metadata.
    """
    if profiler is None:
        raise HTTPException(status_code=503, detail="Profiler not loaded")

    try:
        family_meta = None
        if request.family_meta:
            family_meta = AgentFamily(**request.family_meta)

        session_dicts = [s.model_dump() for s in request.sessions]

        profiler.add_family(
            family_name=request.family_name,
            sessions=session_dicts,
            family_meta=family_meta,
            rebuild=request.rebuild,
        )

        return {
            "message": f"Family {request.family_name} added successfully",
            "sessions_added": len(session_dicts),
            "total_families": len(profiler.families),
        }
    except Exception as e:
        logger.error(f"Add family error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/families", tags=["Families"])
async def list_families():
    """List all known agent families."""
    if profiler is None:
        raise HTTPException(status_code=503, detail="Profiler not loaded")

    families_info = {}
    for name, fam in profiler.families.items():
        families_info[name] = {
            "name": fam.name,
            "description": fam.description,
            "typical_mitre": fam.typical_mitre,
            "typical_lolbins": fam.typical_lolbins,
            "caas_type": fam.caas_type,
            "first_seen": fam.first_seen,
            "aliases": fam.aliases,
        }

    return {"families": families_info, "count": len(families_info)}


@app.get("/models/info", response_model=ModelInfoResponse, tags=["Models"])
async def model_info():
    """Get information about the loaded model and index."""
    if profiler is None:
        raise HTTPException(status_code=503, detail="Profiler not loaded")

    return ModelInfoResponse(
        model_path=os.getenv("MODEL_DIR", "/models/aag-agent-profiler-v1"),
        dimension=profiler.dimension,
        index_type=type(profiler.index).__name__,
        families=list(profiler.families.keys()),
    )


@app.post("/families/save", tags=["Families"])
async def save_index(index_path: str = Body(..., embed=True)):
    """Save the FAISS index to disk."""
    if profiler is None:
        raise HTTPException(status_code=503, detail="Profiler not loaded")

    try:
        profiler.save_index(index_path)
        return {"message": f"Index saved to {index_path}"}
    except Exception as e:
        logger.error(f"Save index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/fingerprint", tags=["Debug"])
async def debug_fingerprint(session: SessionRequest = Body(...)):
    """Convert session to fingerprint text for debugging."""
    session_dict = session.model_dump()
    text = fingerprint_to_text(session_dict)
    return {"fingerprint_text": text, "length": len(text)}


# ─── Main ───

if __name__ == "__main__":
    uvicorn.run(
        "aag_profiler.api:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info",
    )