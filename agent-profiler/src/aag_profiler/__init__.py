"""AAG Agent Profiler - Behavioral attribution via Sentence-BERT + FAISS."""

from .embedder import (
    AgentFamily,
    DEFAULT_FAMILIES,
    AgentProfiler,
    fingerprint_to_text,
    prepare_training_data,
    fine_tune_model,
    build_faiss_index,
)

__version__ = "0.1.0"
