"""Fine-tune Sentence-BERT for agent family attribution."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import json

from aag_be.features import SessionFingerprint
from aag_profiler.embedder import (
    prepare_training_data, fine_tune_model, build_faiss_index,
    AgentFamily, DEFAULT_FAMILIES, fingerprint_to_text
)


def load_sessions(path: Path) -> list[SessionFingerprint]:
    """Load sessions from parquet or pickle."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        import json as json_lib
        table = pq.read_table(path)
        df = table.to_pandas()
    else:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    sessions = []
    for _, row in df.iterrows():
        commands = [SessionFingerprint.CommandEvent(**c) for c in json_lib.loads(row["commands_json"])]
        network = [SessionFingerprint.NetworkEvent(**n) for n in json_lib.loads(row["network_json"])]
        files = [SessionFingerprint.FileEvent(**f) for f in json_lib.loads(row["files_json"])]

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


def load_redteam_sessions(redteam_dir: Path) -> list[dict]:
    """Load red team sessions with family labels."""
    sessions = []
    for scenario_file in sorted(redteam_dir.glob("C*.parquet")):
        print(f"  Loading {scenario_file.name}...")
        if scenario_file.suffix == ".parquet":
            import pyarrow.parquet as pq
            import json as json_lib
            table = pq.read_table(scenario_file)
            df = table.to_pandas()
        else:
            import pickle
            with open(scenario_file, "rb") as f:
                sessions.extend(pickle.load(f))
                continue

        for _, row in df.iterrows():
            commands = [SessionFingerprint.CommandEvent(**c) for c in json_lib.loads(row["commands_json"])]
            network = [SessionFingerprint.NetworkEvent(**n) for n in json_lib.loads(row["network_json"])]
            files = [SessionFingerprint.FileEvent(**f) for f in json_lib.loads(row["files_json"])]

            session = {
                "session_id": row["session_id"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "src_ip": row["src_ip"],
                "user_agent": row["user_agent"],
                "decoy_persona": row["decoy_persona"],
                "command_sequence": [{
                    "exe": c.exe, "argv": c.argv
                } for c in commands],
                "network": [{
                    "timestamp": n.timestamp, "src_ip": n.src_ip, "dst_ip": n.dst_ip,
                    "dst_port": n.dst_port, "protocol": n.protocol, "direction": n.direction,
                    "bytes_sent": n.bytes_sent, "bytes_recv": n.bytes_recv, "process_pid": n.process_pid
                } for n in network],
                "files": [{
                    "timestamp": f.timestamp, "path": f.path, "operation": f.operation,
                    "process_pid": f.process_pid, "is_honeyfile": f.is_honeyfile
                } for f in files],
                "mitre_techniques": row["mitre_techniques"].split(",") if row["mitre_techniques"] else [],
                "agent_family": row.get("agent_family", "UNKNOWN"),
                "agent_name": row.get("agent_name", "UNKNOWN"),
            }
            sessions.append(session)

    return sessions


def load_benign_sessions_dict(path: Path) -> list[dict]:
    """Load benign sessions as dicts."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        import json as json_lib
        table = pq.read_table(path)
        df = table.to_pandas()
    else:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    sessions = []
    for _, row in df.iterrows():
        commands = [SessionFingerprint.CommandEvent(**c) for c in json_lib.loads(row["commands_json"])]
        network = [SessionFingerprint.NetworkEvent(**n) for n in json_lib.loads(row["network_json"])]
        files = [SessionFingerprint.FileEvent(**f) for f in json_lib.loads(row["files_json"])]

        session = {
            "session_id": row["session_id"],
            "start_ts": row["start_ts"],
            "end_ts": row["end_ts"],
            "src_ip": row["src_ip"],
            "user_agent": row["user_agent"],
            "decoy_persona": row["decoy_persona"],
            "command_sequence": [{
                "exe": c.exe, "argv": c.argv
            } for c in commands],
            "network": [{
                "timestamp": n.timestamp, "src_ip": n.src_ip, "dst_ip": n.dst_ip,
                "dst_port": n.dst_port, "protocol": n.protocol, "direction": n.direction,
                "bytes_sent": n.bytes_sent, "bytes_recv": n.bytes_recv, "process_pid": n.process_pid
            } for n in network],
            "files": [{
                "timestamp": f.timestamp, "path": f.path, "operation": f.operation,
                "process_pid": f.process_pid, "is_honeyfile": f.is_honeyfile
            } for f in files],
            "mitre_techniques": row["mitre_techniques"].split(",") if row["mitre_techniques"] else [],
            "agent_family": "BENIGN",
            "agent_name": "BENIGN",
        }
        sessions.append(session)

    return sessions


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sentence-BERT profiler")
    parser.add_argument("--benign", type=Path, required=True, help="Benign sessions parquet/pickle")
    parser.add_argument("--redteam", type=Path, required=True, help="Red team scenarios directory")
    parser.add_argument("--output", type=Path, required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--index-type", type=str, default="IVF-PQ", choices=["IVF-PQ", "HNSW", "Flat"])
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading benign sessions from {args.benign}...")
    benign_sessions = load_benign_sessions_dict(args.benign)
    print(f"Loaded {len(benign_sessions)} benign sessions")

    print(f"Loading red team sessions from {args.redteam}...")
    malicious_sessions = load_redteam_sessions(args.redteam)
    print(f"Loaded {len(malicious_sessions)} malicious sessions")

    # Prepare training data
    print("Preparing training pairs...")
    train_examples = prepare_training_data(
        benign_sessions, malicious_sessions, DEFAULT_FAMILIES, max_pairs=10000
    )

    # Fine-tune model
    print("Fine-tuning model...")
    args.output.mkdir(parents=True, exist_ok=True)
    model = fine_tune_model(
        train_examples=train_examples,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
    )

    # Build FAISS index
    print("Building FAISS index...")
    all_sessions = benign_sessions + malicious_sessions
    index, labels = build_faiss_index(
        model, all_sessions, DEFAULT_FAMILIES,
        index_type=args.index_type,
        nlist=args.nlist, m=args.m, nbits=args.nbits,
    )

    # Save index and metadata
    import faiss
    import pickle
    faiss.write_index(index, str(args.output / "agent_families.index"))
    with open(args.output / "families.pkl", "wb") as f:
        pickle.dump(labels, f)

    # Save family metadata
    with open(args.output / "family_metadata.json", "w") as f:
        json.dump({name: {
            "name": fam.name,
            "description": fam.description,
            "typical_mitre": fam.typical_mitre,
            "typical_lolbins": fam.typical_lolbins,
            "caas_type": fam.caas_type,
            "first_seen": fam.first_seen,
            "aliases": fam.aliases,
        } for name, fam in DEFAULT_FAMILIES.items()}, f, indent=2)

    print(f"Profiler saved to {args.output}")
    print(f"  Model: {args.output}")
    print(f"  Index: {args.output}/agent_families.index")
    print(f"  Metadata: {args.output}/families.pkl")


if __name__ == "__main__":
    main()