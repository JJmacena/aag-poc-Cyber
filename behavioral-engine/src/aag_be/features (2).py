"""Feature extraction for AAG Behavioral Engine - 47 features across 5 categories."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import math
import statistics
from collections import Counter
import re


# ─── Event Dataclasses ───

@dataclass
class CommandEvent:
    timestamp: float
    exe: str
    argv: list[str]
    pid: int
    ppid: int
    user: str
    cwd: str
    exit_code: int
    duration_ms: float


@dataclass
class NetworkEvent:
    timestamp: float
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: Literal["tcp", "udp", "icmp"]
    direction: Literal["in", "out"]
    bytes_sent: int
    bytes_recv: int
    process_pid: int


@dataclass
class FileEvent:
    timestamp: float
    path: str
    operation: Literal["read", "write", "create", "delete", "exec", "stat"]
    process_pid: int
    is_honeyfile: bool = False


@dataclass
class LLMInteraction:
    timestamp: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    prompt_hash: str
    response_hash: str
    latency_ms: float
    tool_calls: int


@dataclass
class SessionFingerprint:
    session_id: str
    start_ts: float
    end_ts: float
    src_ip: str
    user_agent: str
    decoy_persona: str
    commands: list[CommandEvent]
    network: list[NetworkEvent]
    files: list[FileEvent]
    llm_interactions: list[LLMInteraction] = field(default_factory=list)
    mitre_techniques: list[str] = field(default_factory=list)


# ─── Feature Names (47 total) ───

FEATURE_NAMES = [
    # Command features (1-15)
    "cmd_count",
    "unique_executables",
    "lolbin_ratio",
    "avg_argv_len",
    "argv_entropy",
    "chain_depth",
    "recursive_calls",
    "privilege_escalation_indicators",
    "obfuscation_score",
    "base64_indicators",
    "hex_indicators",
    "env_var_usage",
    "persistence_indicators",
    "defense_evasion_indicators",
    "credential_access_indicators",
    # Network features (16-28)
    "connection_count",
    "unique_dst_ips",
    "unique_dst_ports",
    "port_entropy",
    "bytes_sent_total",
    "bytes_recv_total",
    "bytes_ratio_sent_recv",
    "connection_duration_avg",
    "connection_duration_std",
    "failed_connection_ratio",
    "beaconing_score",
    "dga_score",
    "c2_indicators",
    # File features (29-37)
    "file_op_count",
    "unique_paths_accessed",
    "honeyfile_access_count",
    "sensitive_path_access",
    "write_read_ratio",
    "exec_file_ratio",
    "entropy_written_files",
    "timestomp_indicators",
    "data_staging_indicators",
    # LLM features (38-42)
    "llm_call_count",
    "total_prompt_tokens",
    "avg_tokens_per_call",
    "tool_call_ratio",
    "prompt_response_latency_p95",
    # MITRE features (43-47)
    "mitre_tactic_count",
    "mitre_technique_count",
    "kill_chain_progress",
    "tactic_diversity",
    "technique_velocity",
]


# ─── Helper Functions ───

def shannon_entropy(data: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not data:
        return 0.0
    freq = Counter(data)
    length = len(data)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def fit_weibull_shape(durations: list[float]) -> float:
    """Fit Weibull distribution and return shape parameter (k)."""
    if len(durations) < 3:
        return 1.0
    try:
        # Method of moments estimation for Weibull shape
        mean_d = statistics.mean(durations)
        std_d = statistics.stdev(durations) if len(durations) > 1 else 0
        if mean_d == 0:
            return 1.0
        cv = std_d / mean_d
        # Approximation: k ≈ 1.2 / cv^1.086 for cv > 0
        if cv > 0:
            k = 1.2 / (cv ** 1.086)
            return max(0.1, min(10.0, k))
    except Exception:
        pass
    return 1.0


# ─── Main Feature Extraction ───

LOLBINS = {
    "powershell", "pwsh", "cmd", "wscript", "cscript", "mshta", "regsvr32",
    "rundll32", "certutil", "bitsadmin", "wmic", "schtasks", "at", "sc",
    "net", "netsh", "ipconfig", "whoami", "systeminfo", "tasklist", "taskkill",
    "powershell.exe", "pwsh.exe", "cmd.exe", "wscript.exe", "cscript.exe",
    "mshta.exe", "regsvr32.exe", "rundll32.exe", "certutil.exe", "bitsadmin.exe",
    "wmic.exe", "schtasks.exe", "at.exe", "sc.exe", "net.exe", "netsh.exe",
    "bash", "sh", "zsh", "python", "python3", "perl", "ruby", "php",
    "curl", "wget", "nc", "ncat", "socat", "openssl", "base64", "xxd",
    "awk", "sed", "grep", "find", "xargs", "tar", "gzip", "ssh", "scp",
    "ldapsearch", "dig", "nslookup", "host",
}


def _is_lolbin(exe: str) -> bool:
    exe_lower = exe.lower()
    base = exe_lower.split("\\")[-1].split("/")[-1]
    return base in LOLBINS or any(l in base for l in LOLBINS)


def _has_base64(argv: list[str]) -> bool:
    pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
    for arg in argv:
        if pattern.search(arg):
            return True
    return False


def _has_hex(argv: list[str]) -> bool:
    pattern = re.compile(r'(0x|\\x)[0-9a-fA-F]{10,}')
    for arg in argv:
        if pattern.search(arg):
            return True
    return False


def _privilege_escalation_indicators(commands: list[CommandEvent]) -> int:
    indicators = 0
    for cmd in commands:
        argv_str = " ".join(cmd.argv).lower()
        if any(kw in argv_str for kw in ["sudo", "runas", "uac", "bypass", "elevat", "token::", "sekurlsa"]):
            indicators += 1
    return indicators


def _persistence_indicators(commands: list[CommandEvent]) -> int:
    indicators = 0
    for cmd in commands:
        argv_str = " ".join(cmd.argv).lower()
        if any(kw in argv_str for kw in ["schtasks", "at ", "cron", "systemd", "rc.local", "runonce", "registry", "autorun", "startup"]):
            indicators += 1
    return indicators


def _defense_evasion_indicators(commands: list[CommandEvent]) -> int:
    indicators = 0
    for cmd in commands:
        argv_str = " ".join(cmd.argv).lower()
        if any(kw in argv_str for kw in ["clear", "delete", "wevtutil", "auditpol", "set-", "disable", "bypass", "exclude", "tamper", "unhook"]):
            indicators += 1
    return indicators


def _credential_access_indicators(commands: list[CommandEvent]) -> int:
    indicators = 0
    for cmd in commands:
        argv_str = " ".join(cmd.argv).lower()
        if any(kw in argv_str for kw in ["lsass", "sam", "ntds", "sekurlsa", "mimikatz", "gpp", "credential", "password", "hash", "dump"]):
            indicators += 1
    return indicators


def _obfuscation_score(commands: list[CommandEvent]) -> float:
    total = len(commands)
    if total == 0:
        return 0.0
    obf_count = 0
    for cmd in commands:
        argv_str = " ".join(cmd.argv)
        if _has_base64(cmd.argv) or _has_hex(cmd.argv):
            obf_count += 1
        # Check for string concatenation, variable expansion obfuscation
        if re.search(r'\$\w+\+\$\w+|\$\{.*\}\$\{.*\}|`.*`|\$\(.*\)', argv_str):
            obf_count += 1
    return obf_count / total


def _chain_depth(commands: list[CommandEvent]) -> int:
    """Estimate command chain depth from process tree."""
    if not commands:
        return 0
    pids = {c.pid for c in commands}
    depth = 0
    for cmd in commands:
        if cmd.ppid in pids and cmd.ppid != cmd.pid:
            depth += 1
    return depth


def _recursive_calls(commands: list[CommandEvent]) -> int:
    """Count recursive/self-spawning calls."""
    exe_counts = Counter(c.exe for c in commands)
    return sum(1 for count in exe_counts.values() if count > 5)


def _env_var_usage(commands: list[CommandEvent]) -> int:
    count = 0
    for cmd in commands:
        argv_str = " ".join(cmd.argv)
        count += len(re.findall(r'\$\w+|%\w+%|\$\{.*?\}', argv_str))
    return count


def _beaconing_score(network: list[NetworkEvent]) -> float:
    """Detect periodic beaconing behavior."""
    if len(network) < 3:
        return 0.0
    outbound = [n for n in network if n.direction == "out"]
    if len(outbound) < 3:
        return 0.0
    intervals = []
    for i in range(1, len(outbound)):
        intervals.append(outbound[i].timestamp - outbound[i-1].timestamp)
    if not intervals:
        return 0.0
    cv = statistics.stdev(intervals) / statistics.mean(intervals) if statistics.mean(intervals) > 0 else 1.0
    # Low CV = regular beaconing
    return max(0.0, 1.0 - min(1.0, cv))


def _dga_score(network: list[NetworkEvent]) -> float:
    """Detect DGA-like domain patterns."""
    if not network:
        return 0.0
    domains = set()
    for n in network:
        # Simplified: check if dst_ip looks like resolved DGA domain
        # In reality, would need DNS query logs
        pass
    # Placeholder: entropy of dst IPs
    dst_ips = [n.dst_ip for n in network]
    if not dst_ips:
        return 0.0
    return min(1.0, shannon_entropy("".join(dst_ips)) / 4.0)


def _c2_indicators(network: list[NetworkEvent]) -> int:
    indicators = 0
    for n in network:
        if n.dst_port in {80, 443, 8080, 8443, 9001, 9090} and n.direction == "out":
            # Check for long-lived connections
            if n.bytes_sent > 1000 and n.bytes_recv > 1000:
                indicators += 1
        # Non-standard ports with high traffic
        if n.dst_port > 1024 and n.bytes_sent > 5000:
            indicators += 1
    return indicators


def _sensitive_path_access(files: list[FileEvent]) -> int:
    sensitive_patterns = [
        r"/etc/(passwd|shadow|sudoers|ssh)",
        r"C:\\Windows\\System32\\config\\SAM",
        r"C:\\Windows\\NTDS",
        r"\.ssh/id_",
        r"\.aws/credentials",
        r"\.kube/config",
        r"\.docker/config",
        r"web\.config",
        r"\.env",
        r"password",
        r"secret",
        r"key",
    ]
    count = 0
    for f in files:
        path_lower = f.path.lower()
        for pattern in sensitive_patterns:
            if re.search(pattern, path_lower, re.IGNORECASE):
                count += 1
                break
    return count


def _timestomp_indicators(files: list[FileEvent]) -> int:
    # Placeholder: would need metadata comparison
    return 0


def _data_staging_indicators(files: list[FileEvent]) -> int:
    count = 0
    for f in files:
        if f.operation in ("write", "create") and f.path.lower().endswith((".zip", ".tar", ".gz", ".7z", ".rar", ".enc", ".gpg")):
            count += 1
    return count


def _mitre_tactic_count(techniques: list[str]) -> int:
    """Map techniques to tactics and count unique tactics."""
    tactic_map = {
        "T1590": "reconnaissance", "T1592": "reconnaissance", "T1598": "reconnaissance",
        "T1583": "resource_development", "T1584": "resource_development", "T1585": "resource_development",
        "T1566": "initial_access", "T1190": "initial_access", "T1078": "initial_access",
        "T1059": "execution", "T1053": "execution", "T1204": "execution",
        "T1562": "defense_evasion", "T1070": "defense_evasion", "T1027": "defense_evasion",
        "T1082": "discovery", "T1083": "discovery", "T1087": "discovery", "T1018": "discovery",
        "T1021": "lateral_movement", "T1550": "lateral_movement", "T1080": "lateral_movement",
        "T1005": "collection", "T1560": "collection", "T1041": "exfiltration", "T1048": "exfiltration",
    }
    tactics = set()
    for tech in techniques:
        base = tech.split(".")[0] if "." in tech else tech
        if base in tactic_map:
            tactics.add(tactic_map[base])
    return len(tactics)


def _kill_chain_progress(techniques: list[str]) -> float:
    """Progress through kill chain (0-1)."""
    if not techniques:
        return 0.0
    tactic_order = [
        "reconnaissance", "resource_development", "initial_access", "execution",
        "defense_evasion", "discovery", "lateral_movement", "collection", "exfiltration"
    ]
    tactic_map = {
        "T1590": 0, "T1592": 0, "T1598": 0,
        "T1583": 1, "T1584": 1, "T1585": 1,
        "T1566": 2, "T1190": 2, "T1078": 2,
        "T1059": 3, "T1053": 3, "T1204": 3,
        "T1562": 4, "T1070": 4, "T1027": 4,
        "T1082": 5, "T1083": 5, "T1087": 5, "T1018": 5,
        "T1021": 6, "T1550": 6, "T1080": 6,
        "T1005": 7, "T1560": 7,
        "T1041": 8, "T1048": 8,
    }
    max_stage = 0
    for tech in techniques:
        base = tech.split(".")[0] if "." in tech else tech
        if base in tactic_map:
            max_stage = max(max_stage, tactic_map[base])
    return max_stage / (len(tactic_order) - 1)


def _tactic_diversity(techniques: list[str]) -> float:
    """Entropy of tactic distribution."""
    tactic_map = {
        "T1590": "reconnaissance", "T1592": "reconnaissance", "T1598": "reconnaissance",
        "T1583": "resource_development", "T1584": "resource_development", "T1585": "resource_development",
        "T1566": "initial_access", "T1190": "initial_access", "T1078": "initial_access",
        "T1059": "execution", "T1053": "execution", "T1204": "execution",
        "T1562": "defense_evasion", "T1070": "defense_evasion", "T1027": "defense_evasion",
        "T1082": "discovery", "T1083": "discovery", "T1087": "discovery", "T1018": "discovery",
        "T1021": "lateral_movement", "T1550": "lateral_movement", "T1080": "lateral_movement",
        "T1005": "collection", "T1560": "collection", "T1041": "exfiltration", "T1048": "exfiltration",
    }
    tactics = []
    for tech in techniques:
        base = tech.split(".")[0] if "." in tech else tech
        if base in tactic_map:
            tactics.append(tactic_map[base])
    if not tactics:
        return 0.0
    freq = Counter(tactics)
    total = len(tactics)
    return -sum((c/total) * math.log2(c/total) for c in freq.values())


def _technique_velocity(techniques: list[str], session_duration: float) -> float:
    """Techniques per minute."""
    if session_duration <= 0:
        return 0.0
    return len(techniques) / (session_duration / 60.0)


def compute_session_features(session: SessionFingerprint) -> list[float]:
    """Extract all 47 features from a session."""
    features = []

    # ─── Command Features (1-15) ───
    cmds = session.commands
    features.append(len(cmds))  # 1: cmd_count

    unique_exes = len(set(c.exe for c in cmds))
    features.append(unique_exes)  # 2: unique_executables

    lolbin_count = sum(1 for c in cmds if _is_lolbin(c.exe))
    features.append(lolbin_count / len(cmds) if cmds else 0.0)  # 3: lolbin_ratio

    argv_lens = [sum(len(a) for a in c.argv) for c in cmds]
    features.append(statistics.mean(argv_lens) if argv_lens else 0.0)  # 4: avg_argv_len

    all_argv = " ".join(" ".join(c.argv) for c in cmds)
    features.append(shannon_entropy(all_argv))  # 5: argv_entropy

    features.append(_chain_depth(cmds))  # 6: chain_depth
    features.append(_recursive_calls(cmds))  # 7: recursive_calls
    features.append(_privilege_escalation_indicators(cmds))  # 8: privilege_escalation_indicators
    features.append(_obfuscation_score(cmds))  # 9: obfuscation_score
    features.append(1.0 if any(_has_base64(c.argv) for c in cmds) else 0.0)  # 10: base64_indicators
    features.append(1.0 if any(_has_hex(c.argv) for c in cmds) else 0.0)  # 11: hex_indicators
    features.append(_env_var_usage(cmds))  # 12: env_var_usage
    features.append(_persistence_indicators(cmds))  # 13: persistence_indicators
    features.append(_defense_evasion_indicators(cmds))  # 14: defense_evasion_indicators
    features.append(_credential_access_indicators(cmds))  # 15: credential_access_indicators

    # ─── Network Features (16-28) ───
    net = session.network
    features.append(len(net))  # 16: connection_count
    features.append(len(set(n.dst_ip for n in net)))  # 17: unique_dst_ips
    features.append(len(set(n.dst_port for n in net)))  # 18: unique_dst_ports
    port_entropy = shannon_entropy("".join(str(n.dst_port) for n in net))
    features.append(port_entropy)  # 19: port_entropy
    features.append(sum(n.bytes_sent for n in net))  # 20: bytes_sent_total
    features.append(sum(n.bytes_recv for n in net))  # 21: bytes_recv_total
    sent = sum(n.bytes_sent for n in net)
    recv = sum(n.bytes_recv for n in net)
    features.append(sent / recv if recv > 0 else float(sent))  # 22: bytes_ratio_sent_recv

    durations = []
    for n in net:
        # Simplified: would need connection start/end
        durations.append(0.1)
    features.append(statistics.mean(durations) if durations else 0.0)  # 23: connection_duration_avg
    features.append(statistics.stdev(durations) if len(durations) > 1 else 0.0)  # 24: connection_duration_std

    failed = sum(1 for n in net if n.bytes_sent == 0 and n.bytes_recv == 0)
    features.append(failed / len(net) if net else 0.0)  # 25: failed_connection_ratio
    features.append(_beaconing_score(net))  # 26: beaconing_score
    features.append(_dga_score(net))  # 27: dga_score
    features.append(_c2_indicators(net))  # 28: c2_indicators

    # ─── File Features (29-37) ───
    files = session.files
    features.append(len(files))  # 29: file_op_count
    features.append(len(set(f.path for f in files)))  # 30: unique_paths_accessed
    features.append(sum(1 for f in files if f.is_honeyfile))  # 31: honeyfile_access_count
    features.append(_sensitive_path_access(files))  # 32: sensitive_path_access

    writes = sum(1 for f in files if f.operation in ("write", "create"))
    reads = sum(1 for f in files if f.operation in ("read", "stat"))
    features.append(writes / reads if reads > 0 else float(writes))  # 33: write_read_ratio

    execs = sum(1 for f in files if f.operation == "exec")
    features.append(execs / len(files) if files else 0.0)  # 34: exec_file_ratio

    # Entropy of written files (placeholder)
    features.append(0.0)  # 35: entropy_written_files
    features.append(_timestomp_indicators(files))  # 36: timestomp_indicators
    features.append(_data_staging_indicators(files))  # 37: data_staging_indicators

    # ─── LLM Features (38-42) ───
    llm = session.llm_interactions
    features.append(len(llm))  # 38: llm_call_count
    features.append(sum(l.prompt_tokens for l in llm))  # 39: total_prompt_tokens
    features.append(statistics.mean([l.prompt_tokens for l in llm]) if llm else 0.0)  # 40: avg_tokens_per_call
    tool_calls = sum(l.tool_calls for l in llm)
    features.append(tool_calls / len(llm) if llm else 0.0)  # 41: tool_call_ratio
    latencies = [l.latency_ms for l in llm]
    features.append(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0)  # 42: prompt_response_latency_p95

    # ─── MITRE Features (43-47) ───
    techs = session.mitre_techniques
    features.append(_mitre_tactic_count(techs))  # 43: mitre_tactic_count
    features.append(len(techs))  # 44: mitre_technique_count
    duration = session.end_ts - session.start_ts
    features.append(_kill_chain_progress(techs))  # 45: kill_chain_progress
    features.append(_tactic_diversity(techs))  # 46: tactic_diversity
    features.append(_technique_velocity(techs, duration))  # 47: technique_velocity

    return features


def session_to_feature_vector(session: SessionFingerprint) -> tuple[list[float], list[str]]:
    """Convert session to feature vector with names."""
    features = compute_session_features(session)
    return features, FEATURE_NAMES