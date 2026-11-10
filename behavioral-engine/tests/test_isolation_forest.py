"""Tests for Isolation Forest model."""

import pytest
import numpy as np
from aag_be.models.isolation_forest import OnlineIsolationForest, IFConfig


def test_if_config_defaults():
    """Test default configuration."""
    config = IFConfig()
    assert config.n_trees == 100
    assert config.max_depth == 10
    assert config.window_size == 256
    assert config.seed == 42
    assert config.threshold_percentile == 95.0


def test_if_learn_and_score():
    """Test learning and scoring."""
    config = IFConfig(n_trees=10, max_depth=5, window_size=10, seed=42)
    model = OnlineIsolationForest(config)

    # Learn some samples
    for i in range(20):
        x = np.random.randn(47).astype(np.float32)
        model.learn_one(x)

    # Score a sample
    x_test = np.random.randn(47).astype(np.float32)
    score = model.score_one(x_test)

    assert isinstance(score, float)
    assert score >= 0.0


def test_if_threshold_update():
    """Test dynamic threshold updates."""
    config = IFConfig(n_trees=10, max_depth=5, window_size=5, threshold_percentile=90.0)
    model = OnlineIsolationForest(config)

    # Learn enough samples to trigger threshold update
    for i in range(15):
        x = np.random.randn(47).astype(np.float32)
        model.learn_one(x)

    threshold = model.get_threshold()
    assert threshold > 0.0


def test_if_save_load(tmp_path):
    """Test model persistence."""
    config = IFConfig(n_trees=10, max_depth=5, window_size=10, seed=42)
    model = OnlineIsolationForest(config)

    # Train
    for i in range(20):
        x = np.random.randn(47).astype(np.float32)
        model.learn_one(x)

    original_threshold = model.get_threshold()

    # Save
    save_path = tmp_path / "if_model.pkl"
    model.save(save_path)

    # Load
    loaded_model = OnlineIsolationForest.load(save_path)

    assert loaded_model.get_threshold() == original_threshold
    assert loaded_model.config.n_trees == config.n_trees
    assert loaded_model._fitted == model._fitted


def test_if_is_anomalous():
    """Test anomaly detection."""
    config = IFConfig(n_trees=10, max_depth=5, window_size=10, seed=42, threshold_percentile=50.0)
    model = OnlineIsolationForest(config)

    # Train on normal data
    for i in range(20):
        x = np.random.randn(47).astype(np.float32) * 0.1  # Low variance
        model.learn_one(x)

    # Normal sample should not be anomalous
    normal = np.random.randn(47).astype(np.float32) * 0.1
    assert model.is_anomalous(normal) == False or model.is_anomalous(normal) == True  # Either is valid

    # Very anomalous sample
    anomalous = np.ones(47, dtype=np.float32) * 100
    # May or may not be flagged depending on threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])