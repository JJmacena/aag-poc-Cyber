"""Agent Profiler: Sentence-BERT embeddings + FAISS IVF-PQ for agent family attribution."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


@dataclass
class AgentFamily:
    name: str
    description: str
    typical_mitre: list[str]
    typical_lolbins: list[str]
    caas_type: str
    first_seen: str
    aliases: list[str] = field(default_factory=list)


DEFAULT_FAMILIES = {
    "AUTOGPT_RAAS": AgentFamily(
        name="AutoGPT-RaaS",
        description="AutoGPT-based ransomware affiliates with autonomous encryption",
        typical_mitre=["T1486", "T1059", "T1566", "T1021", "T1005", "T1041"],
        typical_lolbins=["powershell", "wmic", "certutil", "vssadmin", "bcdedit"],
        caas_type="RAAS",
        first_seen="2023-Q2",
        aliases=["AutoGPT-Locker", "GPT-Ransom"],
    ),
    "BABYAGI_CAAS": AgentFamily(
        name="BabyAGI-CaaS",
        description="BabyAGI framework adapted for credential stuffing and access brokering",
        typical_mitre=["T1110", "T1555", "T1078", "T1585", "T1583"],
        typical_lolbins=["curl", "python", "sqlmap", "hydra", "medusa"],
        caas_type="ACCESS_BROKER",
        first_seen="2023-Q3",
        aliases=["BabyAGI-Creds", "AGI-Stuffer"],
    ),
    "CUSTOM_LLM_AGENT": AgentFamily(
        name="Custom-LLM-Agent",
        description="Custom LLM agent frameworks (LangChain, custom) for APT-style operations",
        typical_mitre=["T1584", "T1583", "T1190", "T1059", "T1021", "T1005", "T1567"],
        typical_lolbins=["bash", "python", "go", "kubectl", "terraform", "ansible"],
        caas_type="C2_FRAMEWORK",
        first_seen="2023-Q4",
        aliases=["LLM-C2", "AgentAPT"],
    ),
    "FRAUDGPT": AgentFamily(
        name="FraudGPT",
        description="Uncensored LLM for BEC, fraud, and social engineering",
        typical_mitre=["T1566", "T1598", "T1585", "T1583", "T1656"],
        typical_lolbins=["curl", "python", "mshta", "regsvr32", "powershell"],
        caas_type="PHISHING_KIT",
        first_seen="2023-Q3",
        aliases=["WormGPT-Fraud", "DarkGPT"],
    ),
    "WORMGPT": AgentFamily(
        name="WormGPT",
        description="Uncensored LLM for malware generation and exploit development",
        typical_mitre=["T1587", "T1588", "T1190", "T1059", "T1027", "T1562"],
        typical_lolbins=["python", "gcc", "nasm", "metasploit", "msfvenom", "powershell"],
        caas_type="EXPLOIT_KIT",
        first_seen="2023-Q2",
        aliases=["WormGPT-Malware", "DarkBERT-Mal"],
    ),
    "COBALT_STRIKE_BEACON": AgentFamily(
        name="CobaltStrike-Beacon",
        description="Cobalt Strike Beacon with AI-assisted lateral movement",
        typical_mitre=["T1059", "T1021", "T1055", "T1003", "T1082", "T1018", "T1573"],
        typical_lolbins=["powershell", "rundll32", "regsvr32", "wmic", "certutil"],
        caas_type="C2_FRAMEWORK",
        first_seen="2020-Q1",
        aliases=["CS-Beacon", "Beacon-AI"],
    ),
    "METASPLOIT_AI": AgentFamily(
        name="Metasploit-AI",
        description="Metasploit Framework with LLM-assisted module selection and evasion",
        typical_mitre=["T1190", "T1059", "T1068", "T1003", "T1021", "T1055"],
        typical_lolbins=["msfconsole", "msfvenom", "ruby", "powershell", "bash"],
        caas_type="EXPLOIT_KIT",
        first_seen="2021-Q1",
        aliases=["MSF-AI", "Metasploit-GPT"],
    ),
    "BENIGN": AgentFamily(
        name="Benign",
        description="Legitimate administrative and development activity",
        typical_mitre=[],
        typical_lolbins=["bash", "python", "kubectl", "terraform", "ansible", "ssh", "git"],
        caas_type="NONE",
        first_seen="N/A",
        aliases=[],
    ),
}


def fingerprint_to_text(session: dict) -> str:
    """Convert session fingerprint to structured text for embedding."""
    parts = []

    # Command sequence
    cmds = session.get("command_sequence", [])
    if cmds:
        cmd_strs = []
        for c in cmds[:50]:  # Limit length
            exe = c.get("exe", "").split("\\")[-1].split("/")[-1]
            argv = " ".join(c.get("argv", [])[:5])
            cmd_strs.append(f"{exe} {argv}")
        parts.append("COMMANDS: " + " | ".join(cmd_strs))

    # Network
    net = session.get("network", [])
    if net:
        net_strs = []
        for n in net[:20]:
            net_strs.append(f"{n.get('direction','?')} {n.get('dst_ip','?')}:{n.get('dst_port','?')}")
        parts.append("NETWORK: " + " | ".join(net_strs))

    # Files
    files = session.get("files", [])
    if files:
        file_strs = []
        for f in files[:20]:
            honey = " [HONEY]" if f.get("is_honeyfile") else ""
            file_strs.append(f"{f.get('operation','?')} {f.get('path','?')}{honey}")
        parts.append("FILES: " + " | ".join(file_strs))

    # MITRE
    mitre = session.get("mitre_techniques", [])
    if mitre:
        parts.append("MITRE: " + ", ".join(mitre))

    # LLM interactions
    llm = session.get("llm_interactions", [])
    if llm:
        llm_strs = []
        for l in llm[:10]:
            llm_strs.append(f"{l.get('model','?')}:{l.get('prompt_tokens',0)}t")
        parts.append("LLM: " + " | ".join(llm_strs))

    return " [SEP] ".join(parts)


class AgentProfiler:
    """
    Agent attribution via behavioral embeddings.
    Uses fine-tuned Sentence-BERT + FAISS IVF-PQ for scalable k-NN search.
    """

    def __init__(
        self,
        model_path: str | Path,
        index_path: str | Path | None = None,
        families_path: str | Path | None = None,
        device: str = "cuda",
    ):
        self.model = SentenceTransformer(model_path, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.families: dict[str, AgentFamily] = {}

        # Load family metadata
        if families_path:
            self._load_families(families_path)
        else:
            self.families = DEFAULT_FAMILIES

        # Load or create FAISS index
        self.index: faiss.Index | None = None
        self.index_labels: list[str] = []
        if index_path:
            self._load_index(index_path)

    def _load_families(self, path: str | Path):
        with open(path) as f:
            data = json.load(f)
        self.families = {k: AgentFamily(**v) for k, v in data.items()}

    def _load_index(self, path: str | Path):
        self.index = faiss.read_index(str(path))
        labels_path = Path(path).with_suffix(".pkl")
        if labels_path.exists():
            with open(labels_path, "rb") as f:
                self.index_labels = pickle.load(f)

    def embed(self, sessions: list[dict]) -> np.ndarray:
        """Embed sessions to vectors."""
        texts = [fingerprint_to_text(s) for s in sessions]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def attribute(
        self,
        session: dict,
        k: int = 5,
        threshold: float = 0.7,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """
        Attribute session to agent family via k-NN.
        Returns: (top_family, confidence, [(family, score), ...])
        """
        if self.index is None:
            raise ValueError("FAISS index not loaded")

        embedding = self.embed([session])[0:1]
        scores, indices = self.index.search(embedding, k)

        # Weighted voting
        family_scores: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.index_labels):
                family = self.index_labels[idx]
                family_scores[family] = family_scores.get(family, 0) + float(score)

        if not family_scores:
            return "UNKNOWN", 0.0, []

        # Normalize
        total = sum(family_scores.values())
        ranked = sorted(
            [(f, s / total) for f, s in family_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        top_family, confidence = ranked[0]
        if confidence < threshold:
            top_family = "UNKNOWN"

        return top_family, confidence, ranked

    def add_family(
        self,
        family_name: str,
        sessions: list[dict],
        family_meta: AgentFamily | None = None,
        rebuild: bool = True,
    ):
        """Add new family to index incrementally."""
        if family_meta:
            self.families[family_name] = family_meta

        embeddings = self.embed(sessions)
        labels = [family_name] * len(sessions)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index_labels = []

        self.index.add(embeddings)
        self.index_labels.extend(labels)

        if rebuild and isinstance(self.index, faiss.IndexIVFPQ):
            self.index.train(self.index.reconstruct_n(0, self.index.ntotal))
            # Note: IVF-PQ requires full rebuild; consider HNSW for incremental

    def save_index(self, path: str | Path):
        """Save FAISS index and labels."""
        faiss.write_index(self.index, str(path))
        labels_path = Path(path).with_suffix(".pkl")
        with open(labels_path, "wb") as f:
            pickle.dump(self.index_labels, f)


def prepare_training_data(
    benign_sessions: list[dict],
    malicious_sessions: list[dict],
    families: dict[str, AgentFamily],
    max_pairs: int = 10000,
) -> list[InputExample]:
    """
    Prepare contrastive training pairs for Sentence-BERT fine-tuning.
    Positive pairs: same family. Negative pairs: different families or benign vs malicious.
    """
    examples = []

    # Group malicious by family
    by_family: dict[str, list[dict]] = {}
    for s in malicious_sessions:
        fam = s.get("agent_family", "UNKNOWN")
        if fam not in by_family:
            by_family[fam] = []
        by_family[fam].append(s)

    # Positive pairs within family
    for fam, sessions in by_family.items():
        if len(sessions) < 2:
            continue
        for i in range(min(len(sessions), 100)):
            for j in range(i + 1, min(len(sessions), 100)):
                if len(examples) >= max_pairs // 2:
                    break
                text_a = fingerprint_to_text(sessions[i])
                text_b = fingerprint_to_text(sessions[j])
                examples.append(InputExample(texts=[text_a, text_b], label=1.0))
            if len(examples) >= max_pairs // 2:
                break

    # Negative pairs: benign vs malicious, cross-family
    benign_sample = np.random.choice(benign_sessions, min(len(benign_sessions), 500), replace=False)
    malicious_sample = np.random.choice(malicious_sessions, min(len(malicious_sessions), 500), replace=False)

    for b in benign_sample:
        for m in malicious_sample:
            if len(examples) >= max_pairs:
                break
            text_a = fingerprint_to_text(b)
            text_b = fingerprint_to_text(m)
            examples.append(InputExample(texts=[text_a, text_b], label=0.0))
        if len(examples) >= max_pairs:
            break

    # Cross-family negative pairs
    family_names = list(by_family.keys())
    for i, fam_a in enumerate(family_names):
        for fam_b in family_names[i + 1:]:
            sessions_a = by_family[fam_a][:50]
            sessions_b = by_family[fam_b][:50]
            for s_a in sessions_a:
                for s_b in sessions_b:
                    if len(examples) >= max_pairs:
                        break
                    text_a = fingerprint_to_text(s_a)
                    text_b = fingerprint_to_text(s_b)
                    examples.append(InputExample(texts=[text_a, text_b], label=0.0))
                if len(examples) >= max_pairs:
                    break
            if len(examples) >= max_pairs:
                break

    return examples


def fine_tune_model(
    train_examples: list[InputExample],
    output_path: str | Path,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_seq_length: int = 256,
) -> SentenceTransformer:
    """Fine-tune Sentence-BERT with contrastive loss."""
    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.ContrastiveLoss(model)

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_path),
        show_progress_bar=True,
    )

    return model


def build_faiss_index(
    model: SentenceTransformer,
    sessions: list[dict],
    families: dict[str, AgentFamily],
    index_type: str = "IVF-PQ",
    nlist: int = 100,
    m: int = 16,
    nbits: int = 8,
) -> tuple[faiss.Index, list[str]]:
    """Build FAISS index from session embeddings."""
    dimension = model.get_sentence_embedding_dimension()

    # Embed all sessions
    embeddings = model.encode(
        [fingerprint_to_text(s) for s in sessions],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    labels = [s.get("agent_family", "BENIGN") for s in sessions]

    if index_type == "Flat":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "HNSW":
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 80
    elif index_type == "IVF-PQ":
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(embeddings)

    return index, labels