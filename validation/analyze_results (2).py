"""Analysis and visualization for validation results."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Literal
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(path: Path) -> dict:
    """Load validation results JSON."""
    with open(path) as f:
        return json.load(f)


def plot_score_distributions(results: dict, output_dir: Path):
    """Plot score distributions by class."""
    # This would need the raw results with scores per session
    # For now, create a summary plot from per-scenario metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Per-scenario mean scores
    ax = axes[0, 0]
    scenarios = []
    means = []
    stds = []
    for scenario, metrics in results["per_scenario_metrics"].items():
        scenarios.append(scenario)
        means.append(metrics["mean_score"])
        # Approximate std from tier distribution
        tier_dist = metrics["tier_distribution"]
        total = sum(tier_dist.values())
        if total > 0:
            # Weighted variance approximation
            scores = []
            for tier, count in tier_dist.items():
                if tier == "BENIGN":
                    scores.extend([0.1] * count)
                elif tier == "TIER-3":
                    scores.extend([0.3] * count)
                elif tier == "TIER-2":
                    scores.extend([0.5] * count)
                elif tier == "TIER-1":
                    scores.extend([0.9] * count)
            stds.append(np.std(scores) if scores else 0)
        else:
            stds.append(0)

    ax.bar(scenarios, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    ax.set_ylabel('Mean Anomaly Score')
    ax.set_title('Mean Score by Scenario')
    ax.tick_params(axis='x', rotation=45)

    # Per-tier benign rates
    ax = axes[0, 1]
    tiers = list(results["per_tier_metrics"].keys())
    benign_rates = [results["per_tier_metrics"][t]["benign_rate"] for t in tiers]
    colors = ['green', 'yellow', 'orange', 'red']
    ax.bar(tiers, benign_rates, color=colors, edgecolor='black')
    ax.set_ylabel('Benign Rate')
    ax.set_title('Benign Rate by Predicted Tier')
    ax.axhline(y=0.05, color='red', linestyle='--', label='5% target')
    ax.legend()

    # Overall metrics radar
    ax = axes[1, 0]
    metrics = results["overall_metrics"]
    categories = list(metrics.keys())
    values = list(metrics.values())
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    ax = plt.subplot(2, 2, 3, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Metrics', pad=20)

    # FP by tier
    ax = axes[1, 1]
    fp_tiers = list(results["false_positive_analysis"]["by_tier"].keys())
    fp_counts = list(results["false_positive_analysis"]["by_tier"].values())
    ax.bar(fp_tiers, fp_counts, color=['yellow', 'orange', 'red'], edgecolor='black')
    ax.set_ylabel('False Positive Count')
    ax.set_title('False Positives by Tier')

    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_latency(results: dict, output_dir: Path):
    """Plot latency metrics."""
    latency = results["latency_metrics"]

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'max_ms']
    values = [latency.get(m, 0) for m in metrics]
    labels = ['Mean', 'Median', 'P95', 'P99', 'Max']

    bars = ax.bar(labels, values, color='steelblue', edgecolor='black')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency Distribution')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "latency.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(results: dict, output_dir: Path):
    """Plot confusion matrix (simplified from tier distribution)."""
    # Build approximate confusion matrix from per-tier benign rates
    tiers = ['BENIGN', 'TIER-3', 'TIER-2', 'TIER-1']
    n_tiers = len(tiers)

    # This is approximate - would need raw predictions for true CM
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create matrix from per-tier data
    matrix = np.zeros((2, n_tiers))  # [Benign, Malicious] x [Predicted tiers]
    for i, tier in enumerate(tiers):
        if tier in results["per_tier_metrics"]:
            m = results["per_tier_metrics"][tier]
            matrix[0, i] = m["benign_rate"] * m["n_samples"]  # True benign
            matrix[1, i] = (1 - m["benign_rate"]) * m["n_samples"]  # True malicious

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / (row_sums + 1e-10)

    sns.heatmap(matrix_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=tiers, yticklabels=['True Benign', 'True Malicious'],
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_title('Normalized Confusion Matrix (Approximate)')
    ax.set_xlabel('Predicted Tier')
    ax.set_ylabel('True Class')

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_detailed_report(results: dict, output_dir: Path):
    """Generate detailed markdown report with analysis."""
    md_path = output_dir / "detailed_analysis.md"

    with open(md_path, "w") as f:
        f.write("# Detailed Validation Analysis\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")

        f.write("## Executive Summary\n\n")
        overall = results["overall_metrics"]
        f.write(f"- **Accuracy**: {overall['accuracy']:.2%}\n")
        f.write(f"- **Precision**: {overall['precision']:.2%}\n")
        f.write(f"- **Recall**: {overall['recall']:.2%}\n")
        f.write(f"- **F1-Score**: {overall['f1']:.2%}\n")
        f.write(f"- **ROC-AUC**: {overall['roc_auc']:.4f}\n")
        f.write(f"- **PR-AUC**: {overall['pr_auc']:.4f}\n\n")

        f.write("## Scenario-Level Detection Performance\n\n")
        f.write("| Scenario | Samples | Detection Rate | Mean Score | Tier-1 | Tier-2 | Tier-3 | Benign |\n")
        f.write("|----------|---------|----------------|------------|--------|--------|--------|--------|\n")
        for scenario, metrics in results["per_scenario_metrics"].items():
            td = metrics["tier_distribution"]
            f.write(f"| {scenario} | {metrics['n_samples']} | {metrics['detection_rate']:.2%} | {metrics['mean_score']:.4f} | ")
            f.write(f"{td.get('TIER-1', 0)} | {td.get('TIER-2', 0)} | {td.get('TIER-3', 0)} | {td.get('BENIGN', 0)} |\n")
        f.write("\n")

        f.write("## Tier-Level Analysis\n\n")
        f.write("| Predicted Tier | Samples | Benign Rate | Malicious Rate | Mean Score |\n")
        f.write("|----------------|---------|-------------|----------------|------------|\n")
        for tier, metrics in results["per_tier_metrics"].items():
            f.write(f"| {tier} | {metrics['n_samples']} | {metrics['benign_rate']:.2%} | {1-metrics['benign_rate']:.2%} | {metrics['mean_score']:.4f} |\n")
        f.write("\n")

        f.write("## False Positive Analysis\n\n")
        fp = results["false_positive_analysis"]
        f.write(f"- **Total False Positives**: {fp['count']}\n")
        f.write(f"- **False Positive Rate**: {fp['rate']:.2%}\n")
        f.write("- **Breakdown by Tier**:\n")
        for tier, count in fp["by_tier"].items():
            f.write(f"  - {tier}: {count}\n")
        f.write("\n")

        # FP Rate target assessment
        target_fpr = 0.05  # 5%
        if fp['rate'] <= target_fpr:
            f.write(f"✅ **PASS**: FP rate ({fp['rate']:.2%}) is below target ({target_fpr:.0%})\n")
        else:
            f.write(f"❌ **FAIL**: FP rate ({fp['rate']:.2%}) exceeds target ({target_fpr:.0%})\n")
        f.write("\n")

        f.write("## Latency Analysis\n\n")
        latency = results["latency_metrics"]
        f.write(f"- **Mean**: {latency['mean_ms']:.2f} ms\n")
        f.write(f"- **Median**: {latency['median_ms']:.2f} ms\n")
        f.write(f"- **P95**: {latency['p95_ms']:.2f} ms\n")
        f.write(f"- **P99**: {latency['p99_ms']:.2f} ms\n")
        f.write(f"- **Max**: {latency['max_ms']:.2f} ms\n\n")

        # Latency target assessment
        target_p99 = 100  # 100ms
        if latency['p99_ms'] <= target_p99:
            f.write(f"✅ **PASS**: P99 latency ({latency['p99_ms']:.1f}ms) meets target ({target_p99}ms)\n")
        else:
            f.write(f"❌ **FAIL**: P99 latency ({latency['p99_ms']:.1f}ms) exceeds target ({target_p99}ms)\n")
        f.write("\n")

        f.write("## Statistical Significance\n\n")
        for test, result in results["statistical_tests"].items():
            f.write(f"### {test.replace('_', ' ').title()}\n")
            for k, v in result.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        f.write("## Recommendations\n\n")
        # Generate recommendations based on results
        if overall['recall'] < 0.95:
            f.write("- **Improve Recall**: Current recall is below 95%. Consider lowering Tier-3 threshold or adding more training data for missed scenarios.\n")
        if fp['rate'] > 0.05:
            f.write("- **Reduce False Positives**: FP rate exceeds 5%. Consider raising thresholds or improving benign baseline training.\n")
        if latency['p99_ms'] > 100:
            f.write("- **Optimize Latency**: P99 latency exceeds 100ms. Consider model quantization, batching, or hardware acceleration.\n")

        # Scenario-specific
        for scenario, metrics in results["per_scenario_metrics"].items():
            if metrics['detection_rate'] < 0.90:
                f.write(f"- **{scenario}**: Low detection rate ({metrics['detection_rate']:.2%}). Review feature coverage for this attack pattern.\n")

        f.write("\n---\n")
        f.write("*Generated by AAG Validation Analyzer*\n")

    print(f"Detailed report saved to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze validation results")
    parser.add_argument("results_file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output", type=Path, default=Path("./validation_analysis"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)

    print("Generating plots...")
    plot_score_distributions(results, args.output)
    plot_latency(results, args.output)
    plot_confusion_matrix(results, args.output)

    print("Generating detailed report...")
    generate_detailed_report(results, args.output)

    print(f"Analysis complete. Output in {args.output}")


if __name__ == "__main__":
    main()