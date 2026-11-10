"""Tests for feature extraction."""

import pytest
from aag_be.features import (
    SessionFingerprint, CommandEvent, NetworkEvent, FileEvent,
    compute_session_features, FEATURE_NAMES
)


def test_feature_names_count():
    """Verify we have exactly 47 features."""
    assert len(FEATURE_NAMES) == 47


def test_compute_session_features():
    """Test feature computation on minimal session."""
    session = SessionFingerprint(
        session_id="test-001",
        start_ts=1000.0,
        end_ts=1060.0,
        src_ip="10.0.0.1",
        user_agent="test-agent",
        decoy_persona="LEGACY_WIN",
        commands=[
            CommandEvent(
                timestamp=1000.0,
                exe="powershell",
                argv=["Get-Process"],
                pid=100,
                ppid=1,
                user="admin",
                cwd="C:\\",
                exit_code=0,
                duration_ms=100.0,
            )
        ],
        network=[],
        files=[],
        llm_interactions=[],
        mitre_techniques=[],
    )

    features = compute_session_features(session)
    assert len(features) == 47
    assert all(isinstance(f, float) for f in features)
    # cmd_count should be 1
    assert features[0] == 1.0


def test_lolbin_ratio():
    """Test LOLBin ratio calculation."""
    session = SessionFingerprint(
        session_id="test-002",
        start_ts=1000.0,
        end_ts=1060.0,
        src_ip="10.0.0.1",
        user_agent="test-agent",
        decoy_persona="LEGACY_WIN",
        commands=[
            CommandEvent(1000.0, "powershell", [], 100, 1, "admin", "C:\\", 0, 100.0),
            CommandEvent(1001.0, "cmd", [], 101, 1, "admin", "C:\\", 0, 100.0),
            CommandEvent(1002.0, "notepad", [], 102, 1, "admin", "C:\\", 0, 100.0),
        ],
        network=[],
        files=[],
        llm_interactions=[],
        mitre_techniques=[],
    )

    features = compute_session_features(session)
    # powershell and cmd are LOLBins, notepad is not
    # lolbin_ratio = 2/3
    assert abs(features[2] - 2/3) < 0.01


def test_network_features():
    """Test network feature extraction."""
    session = SessionFingerprint(
        session_id="test-003",
        start_ts=1000.0,
        end_ts=1060.0,
        src_ip="10.0.0.1",
        user_agent="test-agent",
        decoy_persona="LEGACY_WIN",
        commands=[],
        network=[
            NetworkEvent(1000.0, "10.0.0.1", "10.0.0.2", 443, "tcp", "out", 1000, 2000, 100),
            NetworkEvent(1001.0, "10.0.0.1", "10.0.0.3", 80, "tcp", "out", 500, 1500, 100),
        ],
        files=[],
        llm_interactions=[],
        mitre_techniques=[],
    )

    features = compute_session_features(session)
    # connection_count
    assert features[15] == 2.0
    # unique_dst_ips
    assert features[16] == 2.0
    # unique_dst_ports
    assert features[17] == 2.0


def test_file_features():
    """Test file feature extraction."""
    session = SessionFingerprint(
        session_id="test-004",
        start_ts=1000.0,
        end_ts=1060.0,
        src_ip="10.0.0.1",
        user_agent="test-agent",
        decoy_persona="LEGACY_WIN",
        commands=[],
        network=[],
        files=[
            FileEvent(1000.0, "C:\\temp\\file.txt", "read", 100, False),
            FileEvent(1001.0, "C:\\temp\\secret.txt", "write", 100, True),  # honeyfile
        ],
        llm_interactions=[],
        mitre_techniques=[],
    )

    features = compute_session_features(session)
    # file_op_count
    assert features[28] == 2.0
    # honeyfile_access_count
    assert features[30] == 1.0


def test_mitre_features():
    """Test MITRE feature extraction."""
    session = SessionFingerprint(
        session_id="test-005",
        start_ts=1000.0,
        end_ts=1060.0,
        src_ip="10.0.0.1",
        user_agent="test-agent",
        decoy_persona="LEGACY_WIN",
        commands=[],
        network=[],
        files=[],
        llm_interactions=[],
        mitre_techniques=["T1059", "T1021", "T1003"],
    )

    features = compute_session_features(session)
    # mitre_technique_count
    assert features[43] == 3.0
    # mitre_tactic_count (execution, lateral_movement, credential_access)
    assert features[42] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])