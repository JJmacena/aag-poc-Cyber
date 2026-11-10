"""LLM Judge FastAPI Server - exposes /judge endpoint for behavioral engine."""

from __future__ import annotations
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from aag_be.models.llm_judge import (
    LLMJudgeClient,
    JudgeResponse,
    Classification,
    DeceptionAction,
)


# Global client instance
judge_client: LLMJudgeClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global judge_client
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    model = os.getenv("MODEL", "nemotron3-ultra:latest")
    judge_client = LLMJudgeClient(base_url=base_url, model=model)
    await judge_client.__aenter__()
    yield
    if judge_client:
        await judge_client.__aexit__(None, None, None)


app = FastAPI(title="AAG LLM Judge", version="0.1.0", lifespan=lifespan)


class SessionTelemetry(BaseModel):
    features: List[float]
    feature_names: List[str]
    commands: List[Dict[str, Any]]
    network: List[Dict[str, Any]]
    files: List[Dict[str, Any]]
    llm_interactions: List[Dict[str, Any]]
    mitre_techniques: List[str]


class JudgeResponseModel(BaseModel):
    classification: str
    confidence: float
    reasoning: str
    mitre_techniques: List[str]
    deception_action: str
    behavioral_iocs: List[Dict[str, Any]]


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "aag-llm-judge"}


@app.post("/judge", response_model=JudgeResponseModel)
async def judge_session(telemetry: SessionTelemetry):
    if judge_client is None:
        raise HTTPException(status_code=503, detail="LLM Judge client not initialized")
    
    try:
        response: JudgeResponse = await judge_client.judge(
            features=telemetry.features,
            feature_names=telemetry.feature_names,
            commands=telemetry.commands,
            network=telemetry.network,
            files=telemetry.files,
            llm_interactions=telemetry.llm_interactions,
            mitre_techniques=telemetry.mitre_techniques,
        )
        return JudgeResponseModel(
            classification=response.classification.value,
            confidence=response.confidence,
            reasoning=response.reasoning,
            mitre_techniques=response.mitre_techniques,
            deception_action=response.deception_action.value,
            behavioral_iocs=response.behavioral_iocs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Judge error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
