"""Benign traffic generator with realistic admin/dev personas."""

from __future__ import annotations
import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aag_be.features import SessionFingerprint, CommandEvent, NetworkEvent, FileEvent


# ─── Persona Profiles ───

@dataclass
class PersonaProfile:
    name: str
    user_agents: list[str]
    common_commands: list[tuple[str, list[str], float]]  # (exe, argv, weight)
    network_patterns: list[dict]
    file_patterns: list[dict]
    think_time_shape: float  # Weibull shape
    think_time_scale: float  # Weibull scale
    typo_rate: float
    session_duration_range: tuple[float, float]  # minutes


PERSONAS = {
    "LINUX_ADMIN": PersonaProfile(
        name="LINUX_ADMIN",
        user_agents=[
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "curl/7.88.1",
            "Wget/1.21.3",
        ],
        common_commands=[
            ("ssh", ["user@host"], 0.3),
            ("sudo", ["systemctl", "status", "nginx"], 0.15),
            ("tail", ["-f", "/var/log/nginx/access.log"], 0.1),
            ("grep", ["-r", "error", "/var/log"], 0.1),
            ("df", ["-h"], 0.05),
            ("free", ["-h"], 0.05),
            ("htop", [], 0.05),
            ("journalctl", ["-u", "docker", "-n", "50"], 0.05),
            ("kubectl", ["get", "pods", "-A"], 0.08),
            ("docker", ["ps"], 0.05),
            ("ansible", ["-m", "ping", "all"], 0.02),
            ("terraform", ["plan"], 0.02),
        ],
        network_patterns=[
            {"dst_port": 22, "protocol": "tcp", "bytes_range": (1000, 50000)},
            {"dst_port": 443, "protocol": "tcp", "bytes_range": (5000, 200000)},
            {"dst_port": 6443, "protocol": "tcp", "bytes_range": (10000, 100000)},
        ],
        file_patterns=[
            {"path": "/var/log/", "ops": ["read", "stat"], "weight": 0.4},
            {"path": "/etc/", "ops": ["read", "stat"], "weight": 0.2},
            {"path": "/home/", "ops": ["read", "write", "stat"], "weight": 0.2},
            {"path": "/opt/", "ops": ["read", "exec"], "weight": 0.1},
        ],
        think_time_shape=1.5,
        think_time_scale=30.0,
        typo_rate=0.02,
        session_duration_range=(5, 60),
    ),
    "WINDOWS_ADMIN": PersonaProfile(
        name="WINDOWS_ADMIN",
        user_agents=[
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "PowerShell/7.3.0",
        ],
        common_commands=[
            ("powershell", ["Get-Service"], 0.2),
            ("powershell", ["Get-EventLog", "-LogName", "System", "-Newest", "20"], 0.15),
            ("cmd", ["/c", "ipconfig", "/all"], 0.1),
            ("netstat", ["-ano"], 0.1),
            ("tasklist", [], 0.1),
            ("whoami", ["/all"], 0.05),
            ("systeminfo", [], 0.05),
            ("wmic", ["process", "get", "name,processid"], 0.05),
            ("certutil", ["-store", "my"], 0.03),
            ("bitsadmin", ["/list", "/allusers"], 0.02),
        ],
        network_patterns=[
            {"dst_port": 443, "protocol": "tcp", "bytes_range": (10000, 500000)},
            {"dst_port": 3389, "protocol": "tcp", "bytes_range": (100000, 1000000)},
            {"dst_port": 5985, "protocol": "tcp", "bytes_range": (5000, 50000)},
        ],
        file_patterns=[
            {"path": "C:\\Windows\\Logs\\", "ops": ["read", "stat"], "weight": 0.3},
            {"path": "C:\\ProgramData\\", "ops": ["read", "stat"], "weight": 0.2},
            {"path": "C:\\Users\\", "ops": ["read", "write", "stat"], "weight": 0.3},
        ],
        think_time_shape=1.3,
        think_time_scale=45.0,
        typo_rate=0.03,
        session_duration_range=(10, 120),
    ),
    "DEVOPS_ENGINEER": PersonaProfile(
        name="DEVOPS_ENGINEER",
        user_agents=["kubectl/v1.28.0", "terraform/1.6.0", "ansible/2.15.0", "docker/24.0.0"],
        common_commands=[
            ("kubectl", ["get", "pods", "-n", "production"], 0.15),
            ("kubectl", ["logs", "-n", "production", "app-xyz"], 0.1),
            ("terraform", ["apply", "-auto-approve"], 0.1),
            ("docker", ["build", "-t", "app:latest", "."], 0.1),
            ("docker", ["push", "registry/app:latest"], 0.08),
            ("ansible-playbook", ["deploy.yml", "-i", "inventory/prod"], 0.08),
            ("helm", ["upgrade", "--install", "app", "./chart"], 0.07),
            ("argocd", ["app", "sync", "production-app"], 0.05),
            ("git", ["push", "origin", "main"], 0.05),
            ("ctr", ["images", "ls"], 0.03),
            ("crictl", ["ps"], 0.02),
            ("skaffold", ["dev"], 0.02),
        ],
        network_patterns=[
            {"dst_port": 443, "protocol": "tcp", "bytes_range": (100000, 5000000)},
            {"dst_port": 6443, "protocol": "tcp", "bytes_range": (50000, 500000)},
            {"dst_port": 2376, "protocol": "tcp", "bytes_range": (10000, 100000)},
        ],
        file_patterns=[
            {"path": "/workspace/", "ops": ["read", "write", "exec"], "weight": 0.4},
            {"path": "/root/.kube/", "ops": ["read", "stat"], "weight": 0.2},
            {"path": "/var/lib/docker/", "ops": ["read", "stat"], "weight": 0.1},
        ],
        think_time_shape=1.8,
        think_time_scale=20.0,
        typo_rate=0.01,
        session_duration_range=(15, 180),
    ),
    "DEVELOPER": PersonaProfile(
        name="DEVELOPER",
        user_agents=["git/2.42.0", "npm/9.8.0", "python/3.11.0", "go/1.21.0"],
        common_commands=[
            ("git", ["status"], 0.15),
            ("git", ["diff"], 0.1),
            ("git", ["commit", "-m", "fix: update logic"], 0.1),
            ("git", ["push"], 0.08),
            ("python", ["-m", "pytest", "tests/"], 0.1),
            ("python", ["main.py"], 0.08),
            ("npm", ["test"], 0.08),
            ("npm", ["run", "build"], 0.07),
            ("go", ["test", "./..."], 0.07),
            ("go", ["build", "-o", "bin/app"], 0.05),
            ("docker", ["compose", "up", "-d"], 0.05),
            ("code", ["."], 0.03),
            ("vim", ["main.py"], 0.02),
        ],
        network_patterns=[
            {"dst_port": 443, "protocol": "tcp", "bytes_range": (5000, 500000)},
            {"dst_port": 8080, "protocol": "tcp", "bytes_range": (10000, 200000)},
            {"dst_port": 5432, "protocol": "tcp", "bytes_range": (5000, 100000)},
        ],
        file_patterns=[
            {"path": "/home/dev/project/", "ops": ["read", "write", "exec"], "weight": 0.5},
            {"path": "/tmp/", "ops": ["read", "write", "create"], "weight": 0.2},
            {"path": "/home/dev/.cache/", "ops": ["read", "write"], "weight": 0.1},
        ],
        think_time_shape=1.2,
        think_time_scale=60.0,
        typo_rate=0.05,
        session_duration_range=(30, 480),
    ),
    "CI_CD_PIPELINE": PersonaProfile(
        name="CI_CD_PIPELINE",
        user_agents=["github-actions/2.0", "gitlab-runner/16.0", "jenkins/2.400"],
        common_commands=[
            ("git", ["clone", "https://github.com/org/repo.git"], 0.2),
            ("npm", ["ci"], 0.15),
            ("npm", ["test"], 0.15),
            ("npm", ["run", "build"], 0.1),
            ("docker", ["build", "-t", "app:${{GITHUB_SHA}}", "."], 0.1),
            ("docker", ["push", "registry/app:${{GITHUB_SHA}}"], 0.08),
            ("kubectl", ["set", "image", "deployment/app", "app=registry/app:${{GITHUB_SHA}}"], 0.07),
            ("helm", ["upgrade", "--install", "app", "./chart", "--namespace", "prod"], 0.05),
            ("terraform", ["init"], 0.03),
            ("terraform", ["apply", "-auto-approve"], 0.02),
            ("sonar-scanner", [], 0.03),
            ("trivy", ["image", "registry/app:latest"], 0.02),
        ],
        network_patterns=[
            {"dst_port": 443, "protocol": "tcp", "bytes_range": (1000000, 50000000)},
            {"dst_port": 6443, "protocol": "tcp", "bytes_range": (100000, 1000000)},
        ],
        file_patterns=[
            {"path": "/workspace/", "ops": ["read", "write", "exec", "create", "delete"], "weight": 0.6},
            {"path": "/tmp/", "ops": ["read", "write", "create", "delete"], "weight": 0.3},
        ],
        think_time_shape=2.0,
        think_time_scale=5.0,
        typo_rate=0.0,
        session_duration_range=(5, 60),
    ),
    "MONITORING_AGENT": PersonaProfile(
        name="MONITORING_AGENT",
        user_agents=["prometheus/2.45.0", "telegraf/1.28.0", "datadog-agent/7.45.0"],
        common_commands=[
            ("curl", ["-s", "http://localhost:9090/metrics"], 0.3),
            ("curl", ["-s", "http://localhost:9100/metrics"], 0.2),
            ("curl", ["-s", "http://localhost:8080/health"], 0.15),
            ("psql", ["-c", "SELECT * FROM pg_stat_activity"], 0.1),
            ("redis-cli", ["INFO", "memory"], 0.1),
            ("nginx", ["-t"], 0.05),
            ("certbot", ["renew", "--dry-run"], 0.02),
            ("smartctl", ["-a", "/dev/sda"], 0.03),
        ],
        network_patterns=[
            {"dst_port": 9090, "protocol": "tcp", "bytes_range": (1000, 10000)},
            {"dst_port": 9100, "protocol": "tcp", "bytes_range": (1000, 10000)},
            {"dst_port": 8080, "protocol": "tcp", "bytes_range": (500, 5000)},
        ],
        file_patterns=[
            {"path": "/var/log/", "ops": ["read", "stat"], "weight": 0.4},
            {"path": "/etc/", "ops": ["read", "stat"], "weight": 0.3},
            {"path": "/proc/", "ops": ["read"], "weight": 0.2},
        ],
        think_time_shape=3.0,
        think_time_scale=30.0,
        typo_rate=0.0,
        session_duration_range=(1, 1440),
    ),
    "BACKUP_SCRIPT": PersonaProfile(
        name="BACKUP_SCRIPT",
        user_agents=["rsync/3.2.7", "borg/1.2.0", "restic/0.16.0"],
        common_commands=[
            ("rsync", ["-avz", "/data/", "backup@storage:/backups/data/"], 0.4),
            ("borg", ["create", "--compression", "lz4", "repo::{now}", "/data"], 0.3),
            ("restic", ["backup", "/data", "--repo", "s3:bucket/path"], 0.2),
            ("tar", ["-czf", "/tmp/backup.tar.gz", "/data"], 0.05),
            ("gpg", ["--encrypt", "--recipient", "backup@org", "/tmp/backup.tar.gz"], 0.03),
            ("aws", ["s3", "cp", "/tmp/backup.tar.gz.gpg", "s3://bucket/backups/"], 0.02),
        ],
        network_patterns=[
            {"dst_port": 22, "protocol": "tcp", "bytes_range": (10000000, 1000000000)},
            {"dst_port": 443, "protocol": "tcp", "bytes_range": (10000000, 500000000)},
        ],
        file_patterns=[
            {"path": "/data/", "ops": ["read"], "weight": 0.5},
            {"path": "/tmp/", "ops": ["write", "create", "delete"], "weight": 0.3},
            {"path": "/backup/", "ops": ["write", "create"], "weight": 0.2},
        ],
        think_time_shape=2.5,
        think_time_scale=10.0,
        typo_rate=0.0,
        session_duration_range=(30, 360),
    ),
    "SECURITY_SCANNER": PersonaProfile(
        name="SECURITY_SCANNER",
        user_agents=["nmap/7.94", "nessus/10.5.0", "openvas/22.4.0", "trivy/0.45.0"],
        common_commands=[
            ("nmap", ["-sS", "-T4", "10.0.0.0/24"], 0.3),
            ("nmap", ["-sV", "--script", "vuln", "target"], 0.2),
            ("trivy", ["image", "--severity", "HIGH,CRITICAL", "app:latest"], 0.2),
            ("nessus", ["-q", "-x", "-T", "html", "-i", "targets.txt"], 0.15),
            ("lynis", ["audit", "system"], 0.1),
            ("chkrootkit", [], 0.05),
        ],
        network_patterns=[
            {"dst_port": 22, "protocol": "tcp", "bytes_range": (1000, 50000)},
            {"dst_port": 80, "protocol": "tcp", "bytes_range": (5000, 200000)},
            {"dst_port": 443, "protocol": "tcp", "bytes_range": (5000, 200000)},
            {"dst_port": 3306, "protocol": "tcp", "bytes_range": (1000, 10000)},
        ],
        file_patterns=[
            {"path": "/tmp/scan_results/", "ops": ["write", "create"], "weight": 0.4},
            {"path": "/var/log/", "ops": ["read", "stat"], "weight": 0.3},
            {"path": "/etc/", "ops": ["read", "stat"], "weight": 0.2},
        ],
        think_time_shape=1.0,
        think_time_scale=120.0,
        typo_rate=0.0,
        session_duration_range=(60, 720),
    ),
}


def inject_typo(cmd: str, argv: list[str], rate: float) -> tuple[str, list[str]]:
    """Randomly inject typos into command."""
    if random.random() > rate:
        return cmd, argv

    typo_type = random.choice(["char_swap", "char_drop", "char_add", "wrong_flag"])
    all_args = [cmd] + argv
    idx = random.randrange(len(all_args))
    arg = all_args[idx]

    if typo_type == "char_swap" and len(arg) > 2:
        i = random.randrange(len(arg) - 1)
        arg = arg[:i] + arg[i+1] + arg[i] + arg[i+2:]
    elif typo_type == "char_drop" and len(arg) > 1:
        i = random.randrange(len(arg))
        arg = arg[:i] + arg[i+1:]
    elif typo_type == "char_add":
        i = random.randrange(len(arg) + 1)
        arg = arg[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + arg[i:]
    elif typo_type == "wrong_flag" and arg.startswith("-"):
        arg = "-" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=len(arg)-1))

    all_args[idx] = arg
    return all_args[0], all_args[1:]


def generate_session(persona: PersonaProfile, session_id: str, start_ts: float) -> SessionFingerprint:
    """Generate a single session for a persona."""
    duration_min = random.uniform(*persona.session_duration_range)
    end_ts = start_ts + duration_min * 60

    # Determine number of commands based on session duration
    n_commands = max(1, int(np.random.poisson(duration_min * 2)))

    commands = []
    network = []
    files = []
    mitre_techniques = []

    current_time = start_ts
    pid_counter = 1000

    for i in range(n_commands):
        # Think time
        think_time = np.random.weibull(persona.think_time_shape) * persona.think_time_scale
        current_time += think_time
        if current_time >= end_ts:
            break

        # Select command
        cmd_choices = [c[0] for c in persona.common_commands]
        cmd_weights = [c[2] for c in persona.common_commands]
        exe, base_argv, _ = random.choices(persona.common_commands, weights=cmd_weights)[0]

        # Inject typos
        exe, argv = inject_typo(exe, base_argv, persona.typo_rate)

        pid = pid_counter + i
        ppid = pid_counter if i > 0 else 1

        cmd_event = CommandEvent(
            timestamp=current_time,
            exe=exe,
            argv=argv,
            pid=pid,
            ppid=ppid,
            user="admin" if "ADMIN" in persona.name else "dev",
            cwd="/home/admin" if "ADMIN" in persona.name else "/home/dev",
            exit_code=0 if random.random() > 0.05 else random.choice([1, 2, 127]),
            duration_ms=random.uniform(10, 5000),
        )
        commands.append(cmd_event)

        # Network events (some commands generate traffic)
        if random.random() < 0.3:
            net_pattern = random.choice(persona.network_patterns)
            n_connections = random.randint(1, 3)
            for _ in range(n_connections):
                dst_ip = f"10.0.{random.randint(1,255)}.{random.randint(1,255)}"
                net_event = NetworkEvent(
                    timestamp=current_time + random.uniform(0, 1),
                    src_ip="10.0.0.50",
                    dst_ip=dst_ip,
                    dst_port=net_pattern["dst_port"],
                    protocol=net_pattern["protocol"],
                    direction="out",
                    bytes_sent=random.randint(*net_pattern["bytes_range"]),
                    bytes_recv=random.randint(*net_pattern["bytes_range"]),
                    process_pid=pid,
                )
                network.append(net_event)

        # File events
        if random.random() < 0.4:
            file_pattern = random.choice(persona.file_patterns)
            op = random.choice(file_pattern["ops"])
            base_path = file_pattern["path"]
            if op in ("write", "create"):
                filename = f"file_{random.randint(1000,9999)}.log"
            else:
                filename = f"file_{random.randint(1000,9999)}.log"
            file_path = base_path.rstrip("/") + "/" + filename

            file_event = FileEvent(
                timestamp=current_time + random.uniform(0, 0.5),
                path=file_path,
                operation=op,
                process_pid=pid,
                is_honeyfile=False,
            )
            files.append(file_event)

    # Session fingerprint
    return SessionFingerprint(
        session_id=session_id,
        start_ts=start_ts,
        end_ts=min(current_time, end_ts),
        src_ip="10.0.0.50",
        user_agent=random.choice(persona.user_agents),
        decoy_persona=persona.name,
        commands=commands,
        network=network,
        files=files,
        llm_interactions=[],
        mitre_techniques=mitre_techniques,
    )


def sessions_to_table(sessions: list[SessionFingerprint]) -> pa.Table:
    """Convert sessions to PyArrow table for parquet."""
    data = {
        "session_id": [],
        "start_ts": [],
        "end_ts": [],
        "src_ip": [],
        "user_agent": [],
        "decoy_persona": [],
        "commands_json": [],
        "network_json": [],
        "files_json": [],
        "mitre_techniques": [],
    }

    for s in sessions:
        data["session_id"].append(s.session_id)
        data["start_ts"].append(s.start_ts)
        data["end_ts"].append(s.end_ts)
        data["src_ip"].append(s.src_ip)
        data["user_agent"].append(s.user_agent)
        data["decoy_persona"].append(s.decoy_persona)
        data["commands_json"].append(json.dumps([asdict(c) for c in s.commands]))
        data["network_json"].append(json.dumps([asdict(n) for n in s.network]))
        data["files_json"].append(json.dumps([asdict(f) for f in s.files]))
        data["mitre_techniques"].append(",".join(s.mitre_techniques))

    return pa.Table.from_pydict(data)


def main():
    parser = argparse.ArgumentParser(description="Generate benign traffic dataset")
    parser.add_argument("-n", "--num-sessions", type=int, default=50000)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sessions = []
    start_time = time.time()

    persona_names = list(PERSONAS.keys())
    # Weighted distribution
    persona_weights = [0.2, 0.15, 0.2, 0.2, 0.1, 0.05, 0.05, 0.05]

    for i in range(args.num_sessions):
        if i % 5000 == 0:
            print(f"Generated {i}/{args.num_sessions} sessions...")

        persona_name = random.choices(persona_names, weights=persona_weights)[0]
        persona = PERSONAS[persona_name]

        session_id = f"benign-{persona_name.lower()}-{i:06d}"
        session = generate_session(persona, session_id, start_time + i * 10)
        sessions.append(session)

    print(f"Writing {len(sessions)} sessions to {args.output}...")
    table = sessions_to_table(sessions)
    pq.write_table(table, args.output, compression="snappy")
    print("Done!")


if __name__ == "__main__":
    main()