"""Validation pipeline for AAG PoC."""

from __future__ import annotations
import argparse
import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

from aag_be.features import SessionFingerprint, CommandEvent, NetworkEvent, FileEvent, LLMInteraction, compute_session_features
from aag_be.models.isolation_forest import OnlineIsolationForest
from aag_be.models.transformer import CommandTransformer, CommandTokenizer
from aag_be.scoring import EnsembleScorer, ScorerConfig, ScoreBreakdown, ThreatTier
from aag_be.models.llm_judge import LLMJudgeClient


@dataclass
class ValidationConfig:
    tier1_threshold: float = 0.80
    tier2_threshold: float = 0.65
    tier3_threshold: float = 0.45
    n_bootstrap: int = 1000
    confidence_level: float = 0.95


@dataclass
class ValidationResults:
    timestamp: str
    config: dict
    overall_metrics: dict
    per_scenario_metrics: dict
    per_tier_metrics: dict
    false_positive_analysis: dict
    latency_metrics: dict
    attribution_metrics: dict
    statistical_tests: dict


def load_models(models_dir: Path):
    """Load all trained models."""
    print("Loading Isolation Forest...")
    if_model = OnlineIsolationForest.load(models_dir / "if_model.pkl")

    print("Loading Transformer...")
    transformer_model = CommandTransformer.from_pretrained(models_dir / "transformer")
    tokenizer = CommandTokenizer.load(models_dir / "transformer" / "tokenizer.json")

    print("Loading LLM Judge client...")
    llm_client = LLMJudgeClient(
        base_url="http://localhost:11434",
        model="nemotron3-ultra:latest",
        timeout=30.0,
    )

    scorer = EnsembleScorer(
        if_model=if_model,
        transformer_model=transformer_model,
        transformer_tokenizer=tokenizer,
        llm_client=llm_client,
        config=ScorerConfig(
            tier1_threshold=0.80,
            tier2_threshold=0.65,
            tier3_threshold=0.45,
        ),
    )

    return scorer


def load_sessions(path: Path) -> list[SessionFingerprint]:
    """Load sessions from parquet."""
    table = pq.read_table(path)
    df = table.to_pandas()

    sessions = []
    for _, row in df.iterrows():
        import json as json_lib
        commands = [CommandEvent(**c) for c in json_lib.loads(row["commands_json"])]
        network = [NetworkEvent(**n) for n in json_lib.loads(row["network_json"])]
        files = [FileEvent(**f) for f in json_lib.loads(row["files_json"])]
        llm = [LLMInteraction(**l) for l in json_lib.loads(row.get("llm_json", "[]"))]

        session = SessionFingerprint(
            session_id=row["session_id"],
            start_ts=row["start_ts"],
            end_ts=row["end_ts"],
            src_ip=row["src_ip"],
            user_agent=row["user_agent"],
            decoy_persona=row["decoy_persona"],
            commands=commands,
            network=network,
            files=files,
            llm_interactions=llm,
            mitre_techniques=row["mitre_techniques"].split(",") if row["mitre_techniques"] else [],
        )
        sessions.append(session)

    return sessions


def load_redteam_sessions(redteam_dir: Path) -> dict[str, list[SessionFingerprint]]:
    """Load red team sessions grouped by scenario."""
    scenarios = {}
    for scenario_file in sorted(redteam_dir.glob("C*.parquet")):
        scenario_id = scenario_file.stem  # C1, C2, etc.
        print(f"  Loading {scenario_id}...")
        sessions = load_sessions(scenario_file)
        scenarios[scenario_id] = sessions
    return scenarios


def load_adversarial_sessions(adversarial_dir: Path) -> list[SessionFingerprint]:
    """Load adversarial evasion sessions."""
    adv_file = adversarial_dir / "adversarial_sessions.parquet"
    if adv_file.exists():
        return load_sessions(adv_file)
    return []


def run_validation(
    scorer: EnsembleScorer,
    benign_sessions: list[SessionFingerprint],
    redteam_sessions: dict[str, list[SessionFingerprint]],
    caas_sessions: list[SessionFingerprint],
    adversarial_sessions: list[SessionFingerprint],
    config: ValidationConfig,
) -> ValidationResults:
    """Run full validation pipeline."""
    all_results = []
    latencies = []

    # Benign sessions (should be BENIGN/TIER-3)
    print(f"Validating {len(benign_sessions)} benign sessions...")
    benign_results = []
    for i, session in enumerate(benign_sessions):
        start = time.perf_counter()
        breakdown = scorer.score_session(session)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        benign_results.append({
            "session_id": session.session_id,
            "true_label": "BENIGN",
            "pred_tier": breakdown.tier.value,
            "pred_classification": breakdown.classification.value,
            "score": breakdown.final_score,
            "latency_ms": latency,
        })

        if i % 1000 == 0:
            print(f"  Processed {i}/{len(benign_sessions)}")

    all_results.extend(benign_results)

    # Red team scenarios
    for scenario_id, sessions in redteam_sessions.items():
        print(f"Validating {len(sessions)} {scenario_id} sessions...")
        for i, session in enumerate(sessions):
            start = time.perf_counter()
            breakdown = scorer.score_session(session)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            all_results.append({
                "session_id": session.session_id,
                "true_label": scenario_id,
                "pred_tier": breakdown.tier.value,
                "pred_classification": breakdown.classification.value,
                "score": breakdown.final_score,
                "latency_ms": latency,
            })

    # CaaS sessions
    print(f"Validating {len(caas_sessions)} CaaS sessions...")
    for i, session in enumerate(caas_sessions):
        start = time.perf_counter()
        breakdown = scorer.score_session(session)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        all_results.append({
            "session_id": session.session_id,
            "true_label": "CAAS",
            "pred_tier": breakdown.tier.value,
            "pred_classification": breakdown.classification.value,
            "score": breakdown.final_score,
            "latency_ms": latency,
        })

    # Adversarial sessions
    print(f"Validating {len(adversarial_sessions)} adversarial sessions...")
    for i, session in enumerate(adversarial_sessions):
        start = time.perf_counter()
        breakdown = scorer.score_session(session)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        all_results.append({
            "session_id": session.session_id,
            "true_label": "ADVERSARIAL",
            "pred_tier": breakdown.tier.value,
            "pred_classification": breakdown.classification.value,
            "score": breakdown.final_score,
            "latency_ms": latency,
        })

    # Compute metrics
    metrics = _compute_metrics(all_results, config)
    statistical_tests = _run_statistical_tests(all_results, config)

    return ValidationResults(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        config=asdict(config),
        overall_metrics=metrics["overall"],
        per_scenario_metrics=metrics["per_scenario"],
        per_tier_metrics=metrics["per_tier"],
        false_positive_analysis=metrics["fp_analysis"],
        latency_metrics=metrics["latency"],
        attribution_metrics=metrics["attribution"],
        statistical_tests=statistical_tests,
    )


def _compute_metrics(results: list[dict], config: ValidationConfig) -> dict:
    """Compute comprehensive metrics."""
    y_true = [r["true_label"] for r in results]
    y_pred_tier = [r["pred_tier"] for r in results]
    y_pred_class = [r["pred_classification"] for r in results]
    scores = [r["score"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    # Binary: benign vs malicious
    y_true_bin = ["BENIGN" if t == "BENIGN" else "MALICIOUS" for t in y_true]
    y_pred_bin = ["BENIGN" if p in ("BENIGN", "TIER-3") else "MALICIOUS" for p in y_pred_tier]

    overall = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, pos_label="MALICIOUS"),
        "recall": recall_score(y_true_bin, y_pred_bin, pos_label="MALICIOUS"),
        "f1": f1_score(y_true_bin, y_pred_bin, pos_label="MALICIOUS"),
        "roc_auc": roc_auc_score(
            [1 if t == "MALICIOUS" else 0 for t in y_true_bin],
            scores,
        ),
        "pr_auc": average_precision_score(
            [1 if t == "MALICIOUS" else 0 for t in y_true_bin],
            scores,
        ),
    }

    # Per-scenario
    per_scenario = {}
    for scenario in set(y_true):
        if scenario == "BENIGN":
            continue
        scenario_results = [r for r in results if r["true_label"] == scenario]
        y_t = ["MALICIOUS"] * len(scenario_results)
        y_p = ["MALICIOUS" if r["pred_tier"] != "BENIGN" else "BENIGN" for r in scenario_results]
        s = [r["score"] for r in scenario_results]

        per_scenario[scenario] = {
            "n_samples": len(scenario_results),
            "detection_rate": sum(1 for p in y_p if p == "MALICIOUS") / len(y_p),
            "mean_score": np.mean(s),
            "median_score": np.median(s),
            "tier_distribution": {
                tier: sum(1 for r in scenario_results if r["pred_tier"] == tier)
                for tier in ["BENIGN", "TIER-3", "TIER-2", "TIER-1"]
            },
        }

    # Per-tier
    per_tier = {}
    for tier in ["BENIGN", "TIER-3", "TIER-2", "TIER-1"]:
        tier_results = [r for r in results if r["pred_tier"] == tier]
        if tier_results:
            per_tier[tier] = {
                "n_samples": len(tier_results),
                "benign_rate": sum(1 for r in tier_results if r["true_label"] == "BENIGN") / len(tier_results),
                "mean_score": np.mean([r["score"] for r in tier_results]),
            }

    # False positive analysis
    fp_results = [r for r in results if r["true_label"] == "BENIGN" and r["pred_tier"] != "BENIGN"]
    fp_analysis = {
        "count": len(fp_results),
        "rate": len(fp_results) / sum(1 for r in results if r["true_label"] == "BENIGN"),
        "by_tier": {
            tier: sum(1 for r in fp_results if r["pred_tier"] == tier)
            for tier in ["TIER-3", "TIER-2", "TIER-1"]
        },
        "top_fp_personas": _top_fp_personas(results),
    }

    # Latency
    latency_metrics = {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "max_ms": np.max(latencies),
    }

    # Attribution (simplified - would need profiler integration)
    attribution_metrics = {
        "note": "Requires AgentProfiler integration for full attribution metrics",
    }

    return {
        "overall": overall,
        "per_scenario": per_scenario,
        "per_tier": per_tier,
        "fp_analysis": fp_analysis,
        "latency": latency_metrics,
        "attribution": attribution_metrics,
    }


def _top_fp_personas(results: list[dict]) -> dict:
    """Analyze false positives by decoy persona."""
    fp_by_persona = {}
    total_by_persona = {}

    for r in results:
        # Would need persona info in results
        pass

    return fp_by_persona


def _run_statistical_tests(results: list[dict], config: ValidationConfig) -> dict:
    """Run statistical significance tests."""
    benign_scores = [r["score"] for r in results if r["true_label"] == "BENIGN"]
    malicious_scores = [r["score"] for r in results if r["true_label"] != "BENIGN"]

    tests = {}

    # Mann-Whitney U test
    if benign_scores and malicious_scores:
        stat, p = stats.mannwhitneyu(malicious_scores, benign_scores, alternative="greater")
        tests["mann_whitney_u"] = {
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < 0.05,
        }

    # Cliff's delta (effect size)
    if benign_scores and malicious_scores:
        delta = _cliffs_delta(malicious_scores, benign_scores)
        tests["cliffs_delta"] = {
            "delta": float(delta),
            "interpretation": _interpret_cliffs_delta(delta),
        }

    # Bootstrap confidence intervals
    if malicious_scores:
        ci = _bootstrap_ci(malicious_scores, config.n_bootstrap, config.confidence_level)
        tests["bootstrap_ci_mean_score"] = {
            "mean": float(np.mean(malicious_scores)),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
        }

    return tests


def _cliffs_delta(x: list[float], y: list[float]) -> float:
    """Compute Cliff's delta effect size."""
    x = np.array(x)
    y = np.array(y)
    n_x, n_y = len(x), len(y)

    # Compare all pairs
    greater = np.sum(x[:, None] > y)
    less = np.sum(x[:, None] < y)
    return (greater - less) / (n_x * n_y)


def _interpret_cliffs_delta(delta: float) -> str:
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def _bootstrap_ci(data: list[float], n_bootstrap: int, confidence: float) -> tuple[float, float]:
    """Bootstrap confidence interval for mean."""
    data = np.array(data)
    n = len(data)
    means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, n, replace=True)
        means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)

    return float(lower), float(upper)


def _save_results(results: ValidationResults, output_dir: Path):
    """Save validation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"validation_results_{timestamp}.json"

    # Convert to serializable
    serializable = asdict(results)

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"Results saved to {json_path}")
    return json_path


def _generate_report(results: ValidationResults, output_dir: Path):
    """Generate markdown report."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"validation_report_{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# AAG PoC Validation Report\n\n")
        f.write(f"**Date:** {results.timestamp}\n\n")

        f.write("## Configuration\n")
        for k, v in results.config.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Overall Metrics\n")
        for k, v in results.overall_metrics.items():
            f.write(f"- **{k}**: {v:.4f}\n")
        f.write("\n")

        f.write("## Per-Scenario Detection Rates\n")
        f.write("| Scenario | Samples | Detection Rate | Mean Score |\n")
        f.write("|----------|---------|----------------|------------|\n")
        for scenario, metrics in results.per_scenario_metrics.items():
            f.write(f"| {scenario} | {metrics['n_samples']} | {metrics['detection_rate']:.2%} | {metrics['mean_score']:.4f} |\n")
        f.write("\n")

        f.write("## Per-Tier Analysis\n")
        f.write("| Tier | Samples | Benign Rate | Mean Score |\n")
        f.write("|------|---------|-------------|------------|\n")
        for tier, metrics in results.per_tier_metrics.items():
            f.write(f"| {tier} | {metrics['n_samples']} | {metrics['benign_rate']:.2%} | {metrics['mean_score']:.4f} |\n")
        f.write("\n")

        f.write("## False Positive Analysis\n")
        fp = results.false_positive_analysis
        f.write(f"- **Total FPs**: {fp['count']}\n")
        f.write(f"- **FP Rate**: {fp['rate']:.2%}\n")
        f.write("- **By Tier**:\n")
        for tier, count in fp["by_tier"].items():
            f.write(f"  - {tier}: {count}\n")
        f.write("\n")

        f.write("## Latency Metrics\n")
        for k, v in results.latency_metrics.items():
            f.write(f"- **{k}**: {v:.2f} ms\n")
        f.write("\n")

        f.write("## Statistical Tests\n")
        for test, result in results.statistical_tests.items():
            f.write(f"### {test}\n")
            for k, v in result.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

    print(f"Report saved to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run AAG validation pipeline")
    parser.add_argument("--benign", type=Path, required=True)
    parser.add_argument("--redteam-dir", type=Path, required=True)
    parser.add_argument("--caas", type=Path, required=True)
    parser.add_argument("--adversarial-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./validation_results"))
    parser.add_argument("--tier1", type=float, default=0.80)
    parser.add_argument("--tier2", type=float, default=0.65)
    parser.add_argument("--tier3", type=float, default=0.45)
    args = parser.parse_args()

    config = ValidationConfig(
        tier1_threshold=args.tier1,
        tier2_threshold=args.tier2,
        tier3_threshold=args.tier3,
    )

    print("Loading models...")
    scorer = load_models(args.models_dir)

    print("Loading datasets...")
    benign = load_sessions(args.benign)
    redteam = load_redteam_sessions(args.redteam_dir)
    caas = load_sessions(args.caas)
    adversarial = load_adversarial_sessions(args.adversarial_dir)

    print(f"  Benign: {len(benign)}")
    print(f"  Redteam: {sum(len(s) for s in redteam.values())} across {len(redteam)} scenarios")
    print(f"  CaaS: {len(caas)}")
    print(f"  Adversarial: {len(adversarial)}")

    print("Running validation...")
    results = run_validation(scorer, benign, redteam, caas, adversarial, config)

    print("Saving results...")
    _save_results(results, args.output_dir)
    _generate_report(results, args.output_dir)

    print("Validation complete!")


if __name__ == "__main__":
    main()