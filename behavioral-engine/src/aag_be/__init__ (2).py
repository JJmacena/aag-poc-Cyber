"""
AAG Behavioral Engine Package

Triple-layer detection engine for autonomous AI-driven offensive agents.
"""

from aag_be.features import (
    SessionFingerprint,
    CommandEvent,
    NetworkEvent,
    FileEvent,
    LLMInteraction,
    compute_session_features,
    FEATURE_NAMES,
)

# Core DLP - no heavy dependencies
from aag_be.dlp import (
    DLPEngine,
    DLPAction,
    DLPViolationType,
    DLPMatch,
    DLPResult,
    DLPRule,
    DLPValidator,
    DLP_PATTERNS,
    DLPMiddleware,
    HoneytokenGenerator,
    create_dlp_engine,
)

# DLP Client - only needs httpx
from aag_be.dlp_client import (
    DLPClient,
    SyncDLPClient,
    DLPInspectionResult,
    DLPViolation,
    DLPAction as ClientDLPAction,
    DLPViolationType as ClientDLPViolationType,
    quick_inspect,
    sync_quick_inspect,
)

# Lazy imports for heavy modules (require torch, river, etc.)
_IsolationForest = None
_CommandTransformer = None
_LLMJudgeClient = None
_EnsembleScorer = None


def _lazy_load_models():
    """Lazy load heavy model dependencies."""
    global _IsolationForest, _CommandTransformer, _LLMJudgeClient, _EnsembleScorer
    if _IsolationForest is None:
        from aag_be.models.isolation_forest import OnlineIsolationForest, IFConfig as IsolationForestConfig
        _IsolationForest = (OnlineIsolationForest, IsolationForestConfig)
    if _CommandTransformer is None:
        from aag_be.models.transformer import CommandTransformer, CommandTokenizer, TransformerConfig
        _CommandTransformer = (CommandTransformer, CommandTokenizer, TransformerConfig)
    if _LLMJudgeClient is None:
        from aag_be.models.llm_judge import (
            LLMJudgeClient,
            JudgeResponse,
            Classification,
            DeceptionAction,
            SYSTEM_PROMPT_V1,
            ADVERSARIAL_SYSTEM_PROMPT,
        )
        _LLMJudgeClient = (LLMJudgeClient, JudgeResponse, Classification, DeceptionAction, SYSTEM_PROMPT_V1, ADVERSARIAL_SYSTEM_PROMPT)
    if _EnsembleScorer is None:
        from aag_be.scoring import (
            EnsembleScorer,
            ScorerConfig,
            ScoreBreakdown,
            ThreatTier,
            Classification as ScoringClassification,
        )
        _EnsembleScorer = (EnsembleScorer, ScorerConfig, ScoreBreakdown, ThreatTier, ScoringClassification)


# Expose lazy-loaded symbols
def __getattr__(name: str):
    _lazy_load_models()
    if name in ("OnlineIsolationForest", "IsolationForestConfig"):
        return _IsolationForest[0] if name == "OnlineIsolationForest" else _IsolationForest[1]
    if name in ("CommandTransformer", "CommandTokenizer", "TransformerConfig"):
        idx = {"CommandTransformer": 0, "CommandTokenizer": 1, "TransformerConfig": 2}[name]
        return _CommandTransformer[idx]
    if name in ("LLMJudgeClient", "JudgeResponse", "Classification", "DeceptionAction", "SYSTEM_PROMPT_V1", "ADVERSARIAL_SYSTEM_PROMPT"):
        idx = {"LLMJudgeClient": 0, "JudgeResponse": 1, "Classification": 2, "DeceptionAction": 3, "SYSTEM_PROMPT_V1": 4, "ADVERSARIAL_SYSTEM_PROMPT": 5}[name]
        return _LLMJudgeClient[idx]
    if name in ("EnsembleScorer", "ScorerConfig", "ScoreBreakdown", "ThreatTier", "ScoringClassification"):
        idx = {"EnsembleScorer": 0, "ScorerConfig": 1, "ScoreBreakdown": 2, "ThreatTier": 3, "ScoringClassification": 4}[name]
        return _EnsembleScorer[idx]
    raise AttributeError(f"module 'aag_be' has no attribute '{name}'")


__version__ = "1.0.0"

__all__ = [
    # Features
    "SessionFingerprint",
    "CommandEvent",
    "NetworkEvent",
    "FileEvent",
    "LLMInteraction",
    "compute_session_features",
    "FEATURE_NAMES",
    # Models (lazy)
    "OnlineIsolationForest",
    "IsolationForestConfig",
    "CommandTransformer",
    "CommandTokenizer",
    "TransformerConfig",
    "LLMJudgeClient",
    "JudgeResponse",
    "Classification",
    "DeceptionAction",
    "SYSTEM_PROMPT_V1",
    "ADVERSARIAL_SYSTEM_PROMPT",
    # Scoring (lazy)
    "EnsembleScorer",
    "ScorerConfig",
    "ScoreBreakdown",
    "ThreatTier",
    "ScoringClassification",
    # DLP
    "DLPEngine",
    "DLPAction",
    "DLPViolationType",
    "DLPMatch",
    "DLPResult",
    "DLPRule",
    "DLPValidator",
    "DLP_PATTERNS",
    "DLPMiddleware",
    "HoneytokenGenerator",
    "create_dlp_engine",
    # DLP Client
    "DLPClient",
    "SyncDLPClient",
    "DLPInspectionResult",
    "DLPViolation",
    "ClientDLPAction",
    "ClientDLPViolationType",
    "quick_inspect",
    "sync_quick_inspect",
]