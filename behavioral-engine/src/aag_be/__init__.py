"""AAG Behavioral Engine - Anomaly detection for AI-driven agents."""

from .features import (
    SessionFingerprint,
    CommandEvent,
    NetworkEvent,
    FileEvent,
    LLMInteraction,
    compute_session_features,
    FEATURE_NAMES,
)

from .scoring import (
    EnsembleScorer,
    ScorerConfig,
    ScoreBreakdown,
    ThreatTier,
)

from .models.isolation_forest import (
    OnlineIsolationForest,
    IFConfig,
    train_baseline_model,
)

from .models.transformer import (
    CommandTransformer,
    CommandTokenizer,
    TransformerConfig,
    CommandSequenceDataset,
    train_transformer,
)

from .models.llm_judge import (
    LLMJudgeClient,
    JudgeResponse,
    Classification,
    DeceptionAction,
    judge_session,
)

__version__ = "0.1.0"
