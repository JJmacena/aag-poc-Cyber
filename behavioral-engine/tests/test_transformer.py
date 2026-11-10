"""Tests for Transformer model."""

import pytest
import torch
from aag_be.models.transformer import (
    CommandTransformer, CommandTokenizer, CommandSequenceDataset, TransformerConfig
)
from aag_be.features import SessionFingerprint, CommandEvent


def test_transformer_config():
    """Test transformer configuration."""
    config = TransformerConfig()
    assert config.epochs == 10
    assert config.batch_size == 32
    assert config.lr == 2e-4
    assert config.max_seq_len == 128
    assert config.max_cmd_len == 64


def test_command_tokenizer():
    """Test command tokenizer."""
    tokenizer = CommandTokenizer(max_vocab=1000)

    # Create dummy sessions
    sessions = [
        SessionFingerprint(
            session_id=f"test-{i}",
            start_ts=1000.0,
            end_ts=1060.0,
            src_ip="10.0.0.1",
            user_agent="test",
            decoy_persona="LEGACY_WIN",
            commands=[
                CommandEvent(1000.0, "powershell", ["Get-Process"], 100, 1, "admin", "C:\\", 0, 100.0),
                CommandEvent(1001.0, "cmd", ["/c", "dir"], 101, 1, "admin", "C:\\", 0, 100.0),
            ],
            network=[],
            files=[],
            llm_interactions=[],
            mitre_techniques=[],
        )
        for i in range(5)
    ]

    tokenizer.fit(sessions)
    assert tokenizer._fitted
    assert len(tokenizer.vocab) > len(tokenizer.SPECIAL_TOKENS)

    # Test encoding
    encoded = tokenizer.encode_command("powershell", ["Get-Process"], 64)
    assert len(encoded) == 64
    assert encoded[0] == tokenizer.vocab.get("powershell", tokenizer.vocab["[UNK]"])


def test_transformer_forward():
    """Test transformer forward pass."""
    model = CommandTransformer(vocab_size=1000)

    batch_size = 2
    seq_len = 10
    cmd_len = 64

    input_ids = torch.randint(0, 1000, (batch_size, seq_len, cmd_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    cls_logits, lm_logits = model(input_ids, attention_mask)

    assert cls_logits.shape == (batch_size, 3)  # 3 classes
    assert lm_logits.shape == (batch_size, seq_len, cmd_len, 1000)


def test_transformer_save_load(tmp_path):
    """Test model save/load."""
    model = CommandTransformer(vocab_size=1000)
    tokenizer = CommandTokenizer(max_vocab=1000)

    # Fit tokenizer
    sessions = [
        SessionFingerprint(
            session_id="test",
            start_ts=1000.0,
            end_ts=1060.0,
            src_ip="10.0.0.1",
            user_agent="test",
            decoy_persona="LEGACY_WIN",
            commands=[CommandEvent(1000.0, "powershell", ["Get-Process"], 100, 1, "admin", "C:\\", 0, 100.0)],
            network=[],
            files=[],
            llm_interactions=[],
            mitre_techniques=[],
        )
    ]
    tokenizer.fit(sessions)

    model.save_pretrained(tmp_path / "transformer")
    tokenizer.save(tmp_path / "transformer" / "tokenizer.json")

    # Load
    loaded_model = CommandTransformer.from_pretrained(tmp_path / "transformer")
    loaded_tokenizer = CommandTokenizer.load(tmp_path / "transformer" / "tokenizer.json")

    assert loaded_model.vocab_size == model.vocab_size
    assert loaded_tokenizer.vocab == tokenizer.vocab


def test_dataset():
    """Test CommandSequenceDataset."""
    tokenizer = CommandTokenizer(max_vocab=1000)

    sessions = [
        SessionFingerprint(
            session_id=f"test-{i}",
            start_ts=1000.0,
            end_ts=1060.0,
            src_ip="10.0.0.1",
            user_agent="test",
            decoy_persona="LEGACY_WIN",
            commands=[CommandEvent(1000.0, "powershell", ["Get-Process"], 100, 1, "admin", "C:\\", 0, 100.0)],
            network=[],
            files=[],
            llm_interactions=[],
            mitre_techniques=[],
        )
        for i in range(10)
    ]

    tokenizer.fit(sessions)
    labels = [0] * 5 + [2] * 5  # 5 benign, 5 agentic

    dataset = CommandSequenceDataset(sessions, tokenizer, max_seq_len=10, max_cmd_len=32, labels=labels)

    assert len(dataset) == 10

    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape == (10, 32)
    assert item["labels"].item() in [0, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])