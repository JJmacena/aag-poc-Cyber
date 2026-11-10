"""Ensemble scoring combining IF, Transformer, LLM Judge, and Honeytoken boost."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Literal
import numpy as np

from aag_be.models.isolation_forest import OnlineIsolationForest
from aag_be.models.transformer import CommandTransformer, CommandTokenizer
from aag_be.models.llm_judge import LLMJudgeClient, JudgeResponse, Classification, DeceptionAction
from aag_be.features import SessionFingerprint, compute_session_features, FEATURE_NAMES


class ThreatTier(str, Enum):
    BENIGN = "BENIGN"
    TIER_3 = "TIER-3"      # Low confidence, monitor
    TIER_2 = "TIER-2"      # Medium confidence, deploy honeytokens
    TIER_1 = "TIER-1"      # High confidence, active deception


@dataclass
class ScorerConfig:
    # Weights for ensemble (must sum to 1.0)
    w_if: float = 0.30
    w_transformer: float = 0.30
    w_llm: float = 0.30
    w_honeytoken: float = 0.10

    # Tier thresholds
    tier1_threshold: float = 0.80
    tier2_threshold: float = 0.65
    tier3_threshold: float = 0.45

    # Honeytoken boost
    honeytoken_boost: float = 0.25
    max_honeytoken_boost: float = 0.40


@dataclass
class ScoreBreakdown:
    if_score: float
    transformer_score: float
    llm_score: float
    honeytoken_boost: float
    final_score: float
    tier: ThreatTier
    classification: Classification
    recommended_actions: list[str]
    deception_triggers: list[DeceptionAction]


class EnsembleScorer:
    """
    Triple-layer detection ensemble:
    Layer 1: CVE Context Match (handled externally via threat intel)
    Layer 2: Behavioral Anomaly (Isolation Forest + Transformer)
    Layer 3: Agent Fingerprint Attribution (LLM Judge + Profiler)
    """

    def __init__(
        self,
        if_model: OnlineIsolationForest,
        transformer_model: CommandTransformer,
        transformer_tokenizer: CommandTokenizer,
        llm_client: LLMJudgeClient,
        config: ScorerConfig | None = None,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    ):
        self.if_model = if_model
        self.transformer_model = transformer_model.to(device)
        self.transformer_tokenizer = transformer_tokenizer
        self.llm_client = llm_client
        self.config = config or ScorerConfig()
        self.device = device
        self.transformer_model.eval()

    def score_session(self, session: SessionFingerprint) -> ScoreBreakdown:
        """Score a session through the full ensemble."""
        # Extract features
        features = compute_session_features(session)
        feature_vec = np.array(features, dtype=np.float32)

        # Layer 2a: Isolation Forest
        if_score = self.if_model.score_one(feature_vec)
        # Normalize to 0-1 using threshold
        if_norm = min(1.0, if_score / max(self.if_model.get_threshold(), 1e-6))

        # Layer 2b: Transformer
        transformer_score = self._score_transformer(session)

        # Layer 3: LLM Judge
        llm_response = self._score_llm(session, features)
        llm_score = self._llm_to_score(llm_response)

        # Honeytoken boost
        honeytoken_boost = self._compute_honeytoken_boost(session)

        # Weighted ensemble
        final_score = (
            self.config.w_if * if_norm +
            self.config.w_transformer * transformer_score +
            self.config.w_llm * llm_score +
            self.config.w_honeytoken * honeytoken_boost
        )
        final_score = min(1.0, final_score)

        # Determine tier
        tier = self._score_to_tier(final_score)

        # Recommended actions
        actions = self._get_recommended_actions(tier, llm_response.deception_action)
        deception_triggers = self._get_deception_triggers(tier, llm_response.deception_action, honeytoken_boost)

        return ScoreBreakdown(
            if_score=if_norm,
            transformer_score=transformer_score,
            llm_score=llm_score,
            honeytoken_boost=honeytoken_boost,
            final_score=final_score,
            tier=tier,
            classification=llm_response.classification,
            recommended_actions=actions,
            deception_triggers=deception_triggers,
        )

    def _score_transformer(self, session: SessionFingerprint) -> float:
        """Score using Transformer model."""
        import torch

        # Tokenize
        encoded = self.transformer_tokenizer.encode_session(
            session,
            self.transformer_model.max_seq_len,
            self.transformer_model.max_cmd_len,
        )
        input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

        # Attention mask
        attention_mask = torch.tensor(
            [[1 if any(t != self.transformer_tokenizer.vocab["[PAD]"] for t in cmd) else 0 for cmd in encoded]],
            dtype=torch.long,
        ).to(self.device)

        with torch.no_grad():
            cls_logits, _ = self.transformer_model(input_ids, attention_mask)
            probs = torch.softmax(cls_logits, dim=1)
            # Class 2 = agentic (index 2)
            agentic_prob = probs[0, 2].item() if probs.shape[1] > 2 else 0.0

        return agentic_prob

    def _score_llm(self, session: SessionFingerprint, features: list[float]) -> JudgeResponse:
        """Get LLM Judge classification."""
        # Convert session to dict format
        commands = [
            {"exe": c.exe, "argv": c.argv, "pid": c.pid, "timestamp": c.timestamp}
            for c in session.commands
        ]
        network = [
            {
                "src_ip": n.src_ip, "dst_ip": n.dst_ip, "dst_port": n.dst_port,
                "protocol": n.protocol, "direction": n.direction,
                "bytes_sent": n.bytes_sent, "bytes_recv": n.bytes_recv,
                "timestamp": n.timestamp,
            }
            for n in session.network
        ]
        files = [
            {"path": f.path, "operation": f.operation, "timestamp": f.timestamp,
             "is_honeyfile": f.is_honeyfile}
            for f in session.files
        ]
        llm_interactions = [
            {
                "model": l.model, "prompt_tokens": l.prompt_tokens,
                "completion_tokens": l.completion_tokens, "latency_ms": l.latency_ms,
                "tool_calls": l.tool_calls,
            }
            for l in session.llm_interactions
        ]

        # Run async judge (simplified sync version for now)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.llm_client.judge(
                features, FEATURE_NAMES, commands, network, files,
                llm_interactions, session.mitre_techniques
            )
        )

    def _llm_to_score(self, response: JudgeResponse) -> float:
        """Convert LLM classification to numeric score."""
        base_scores = {
            Classification.BENIGN: 0.1,
            Classification.SCRIPTED: 0.4,
            Classification.AGENTIC: 0.9,
        }
        base = base_scores.get(response.classification, 0.1)
        # Weight by confidence
        return base * response.confidence + 0.1 * (1 - response.confidence)

    def _compute_honeytoken_boost(self, session: SessionFingerprint) -> float:
        """Compute boost from honeytoken interactions."""
        honeytoken_accesses = sum(1 for f in session.files if f.is_honeyfile)
        if honeytoken_accesses == 0:
            return 0.0

        # Boost increases with each access, capped
        boost = min(
            self.config.honeytoken_boost * honeytoken_accesses,
            self.config.max_honeytoken_boost
        )
        return boost

    def _score_to_tier(self, score: float) -> ThreatTier:
        if score >= self.config.tier1_threshold:
            return ThreatTier.TIER_1
        elif score >= self.config.tier2_threshold:
            return ThreatTier.TIER_2
        elif score >= self.config.tier3_threshold:
            return ThreatTier.TIER_3
        return ThreatTier.BENIGN

    def _get_recommended_actions(self, tier: ThreatTier, deception_action: DeceptionAction) -> list[str]:
        actions = {
            ThreatTier.BENIGN: ["Log session", "Update baseline"],
            ThreatTier.TIER_3: ["Enhanced logging", "Schedule honeytoken injection", "Alert SOC (low)"],
            ThreatTier.TIER_2: ["Deploy honeytokens", "Redirect to decoy", "Alert SOC (medium)", "Start packet capture"],
            ThreatTier.TIER_1: ["Activate tar pit", "Data poisoning", "Full forensic capture", "Alert SOC (high)", "Block egress"],
        }
        result = actions.get(tier, [])
        if deception_action != DeceptionAction.NONE:
            result.append(f"Deception: {deception_action.value}")
        return result

    def _get_deception_triggers(
        self, tier: ThreatTier, deception_action: DeceptionAction, honeytoken_boost: float
    ) -> list[DeceptionAction]:
        triggers = []
        if deception_action != DeceptionAction.NONE:
            triggers.append(deception_action)

        if tier == ThreatTier.TIER_1:
            triggers.extend([DeceptionAction.TAR_PIT, DeceptionAction.DATA_POISONING, DeceptionAction.INTEL_GENERATION])
        elif tier == ThreatTier.TIER_2:
            triggers.extend([DeceptionAction.HONEYTOKEN_INJECTION, DeceptionAction.DECOY_REDIRECT])
        elif tier == ThreatTier.TIER_3 and honeytoken_boost > 0:
            triggers.append(DeceptionAction.HONEYTOKEN_INJECTION)

        return list(dict.fromkeys(triggers))  # Deduplicate preserving order