"""Train Transformer model on benign + red team sessions."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from aag_be.features import SessionFingerprint
from aag_be.models.transformer import (
    CommandTransformer, CommandTokenizer, CommandSequenceDataset,
    train_transformer, TransformerConfig
)


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


def load_redteam_sessions(redteam_dir: Path) -> list[SessionFingerprint]:
    """Load all red team scenario sessions."""
    sessions = []
    for scenario_file in sorted(redteam_dir.glob("C*.parquet")):
        print(f"  Loading {scenario_file.name}...")
        sessions.extend(load_sessions(scenario_file))
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Train Transformer model")
    parser.add_argument("--benign", type=Path, required=True, help="Benign sessions parquet/pickle")
    parser.add_argument("--redteam", type=Path, required=True, help="Red team scenarios directory")
    parser.add_argument("--output", type=Path, required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-cmd-len", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading benign sessions from {args.benign}...")
    benign_sessions = load_sessions(args.benign)
    print(f"Loaded {len(benign_sessions)} benign sessions")

    print(f"Loading red team sessions from {args.redteam}...")
    redteam_sessions = load_redteam_sessions(args.redteam)
    print(f"Loaded {len(redteam_sessions)} red team sessions")

    # Combine and label
    all_sessions = benign_sessions + redteam_sessions
    labels = [0] * len(benign_sessions) + [2] * len(redteam_sessions)  # 0=benign, 2=agentic

    # Shuffle
    indices = np.random.permutation(len(all_sessions))
    all_sessions = [all_sessions[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Split train/val
    split_idx = int(len(all_sessions) * (1 - args.val_split))
    train_sessions = all_sessions[:split_idx]
    train_labels = labels[:split_idx]
    val_sessions = all_sessions[split_idx:]
    val_labels = labels[split_idx:]

    print(f"Train: {len(train_sessions)}, Val: {len(val_sessions)}")

    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = CommandTokenizer(max_vocab=8000)
    tokenizer.fit(all_sessions)

    # Create datasets
    train_ds = CommandSequenceDataset(
        train_sessions, tokenizer,
        max_seq_len=args.max_seq_len,
        max_cmd_len=args.max_cmd_len,
        labels=train_labels,
    )
    val_ds = CommandSequenceDataset(
        val_sessions, tokenizer,
        max_seq_len=args.max_seq_len,
        max_cmd_len=args.max_cmd_len,
        labels=val_labels,
    )

    # Initialize model
    model = CommandTransformer(vocab_size=len(tokenizer.vocab))

    # Training config
    config = TransformerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        max_cmd_len=args.max_cmd_len,
    )

    # Train
    print("Starting training...")
    history = train_transformer(model, train_ds, val_ds, config)

    # Save model and tokenizer
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save(args.output / "tokenizer.json")

    # Save training history
    import json
    with open(args.output / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()