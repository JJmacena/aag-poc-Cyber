"""Train Isolation Forest baseline model on benign sessions."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from aag_be.features import SessionFingerprint
from aag_be.models.isolation_forest import OnlineIsolationForest, IFConfig, train_baseline_model


def load_sessions(path: Path) -> list[SessionFingerprint]:
    """Load sessions from parquet or pickle."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        import json
        table = pq.read_table(path)
        df = table.to_pandas()
    else:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    sessions = []
    for _, row in df.iterrows():
        commands = [SessionFingerprint.CommandEvent(**c) for c in json.loads(row["commands_json"])]
        network = [SessionFingerprint.NetworkEvent(**n) for n in json.loads(row["network_json"])]
        files = [SessionFingerprint.FileEvent(**f) for f in json.loads(row["files_json"])]

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
            mitre_techniques=row["mitre_techniques"].split(",") if row["mitre_techniques"] else [],
        )
        sessions.append(session)

    return sessions


def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest baseline")
    parser.add_argument("--data", type=Path, required=True, help="Benign sessions parquet/pickle")
    parser.add_argument("--output", type=Path, required=True, help="Output model path")
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()

    print(f"Loading sessions from {args.data}...")
    sessions = load_sessions(args.data)
    print(f"Loaded {len(sessions)} sessions")

    config = IFConfig(
        n_trees=args.n_trees,
        max_depth=args.max_depth,
        window_size=args.window_size,
        seed=args.seed,
        threshold_percentile=args.threshold_percentile,
    )

    train_baseline_model(sessions, args.output, config)

    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()