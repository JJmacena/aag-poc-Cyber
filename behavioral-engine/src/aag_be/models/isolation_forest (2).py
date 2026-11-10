"""Online Isolation Forest for streaming anomaly detection."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
from river import anomaly, preprocessing, compose


@dataclass
class IFConfig:
    n_trees: int = 100
    max_depth: int = 10
    window_size: int = 256
    seed: int = 42
    threshold_percentile: float = 95.0


class OnlineIsolationForest:
    """
    Wrapper around River's Isolation Forest for streaming anomaly detection.
    Includes RobustScaler pipeline and dynamic threshold management.
    """

    def __init__(self, config: IFConfig | None = None):
        self.config = config or IFConfig()
        self.model: anomaly.IsolationForest | None = None
        self.scaler = preprocessing.RobustScaler()
        self.pipeline: compose.Pipeline | None = None
        self._threshold: float = 0.0
        self._scores_buffer: list[float] = []
        self._fitted: bool = False

    def _build_pipeline(self, n_features: int) -> compose.Pipeline:
        """Build the preprocessing + model pipeline."""
        self.model = anomaly.IsolationForest(
            n_trees=self.config.n_trees,
            max_depth=self.config.max_depth,
            window_size=self.config.window_size,
            seed=self.config.seed,
        )
        return self.scaler | self.model

    def learn_one(self, x: np.ndarray) -> OnlineIsolationForest:
        """Learn from a single feature vector (streaming)."""
        if self.pipeline is None:
            self.pipeline = self._build_pipeline(len(x))

        # Convert to dict for River
        x_dict = {f"f{i}": float(val) for i, val in enumerate(x)}
        self.pipeline.learn_one(x_dict)

        # Score for threshold calibration
        score = self.score_one(x)
        self._scores_buffer.append(score)

        # Update threshold periodically
        if len(self._scores_buffer) >= self.config.window_size:
            self._update_threshold()

        self._fitted = True
        return self

    def score_one(self, x: np.ndarray) -> float:
        """Score a single feature vector (anomaly score)."""
        if self.pipeline is None:
            return 0.0

        x_dict = {f"f{i}": float(val) for i, val in enumerate(x)}
        try:
            # River's IsolationForest returns anomaly score (higher = more anomalous)
            score = self.pipeline.score_one(x_dict)
            return float(score) if score is not None else 0.0
        except Exception:
            return 0.0

    def _update_threshold(self):
        """Update dynamic threshold based on score distribution."""
        if self._scores_buffer:
            self._threshold = float(np.percentile(self._scores_buffer, self.config.threshold_percentile))
            # Keep buffer bounded
            if len(self._scores_buffer) > self.config.window_size * 2:
                self._scores_buffer = self._scores_buffer[-self.config.window_size:]

    def is_anomalous(self, x: np.ndarray) -> bool:
        """Check if a sample is anomalous based on current threshold."""
        return self.score_one(x) >= self._threshold

    def get_threshold(self) -> float:
        return self._threshold

    def save(self, path: Path):
        """Save model, scaler, and threshold to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pipeline": self.pipeline,
            "threshold": self._threshold,
            "config": self.config,
            "fitted": self._fitted,
            "scores_buffer": self._scores_buffer[-self.config.window_size:],
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> OnlineIsolationForest:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls(data["config"])
        obj.pipeline = data["pipeline"]
        obj._threshold = data["threshold"]
        obj._fitted = data["fitted"]
        obj._scores_buffer = data.get("scores_buffer", [])
        return obj


def train_baseline_model(
    sessions: list,
    output_path: Path,
    config: IFConfig | None = None,
) -> OnlineIsolationForest:
    """
    Train Isolation Forest baseline on benign sessions.
    Expects sessions with compute_session_features method.
    """
    from aag_be.features import compute_session_features

    config = config or IFConfig()
    model = OnlineIsolationForest(config)

    print(f"Training Isolation Forest on {len(sessions)} sessions...")
    for i, session in enumerate(sessions):
        features = compute_session_features(session)
        model.learn_one(np.array(features, dtype=np.float32))
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} sessions...")

    # Final threshold update
    model._update_threshold()
    print(f"Final threshold: {model.get_threshold():.4f}")

    model.save(output_path)
    return model