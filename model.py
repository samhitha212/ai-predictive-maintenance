"""Reproducible synthetic telemetry and held-out Random Forest evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "cpu_utilization",
    "core_temperature",
    "disk_health_score",
    "uptime_hours",
]
BOUNDS = {
    "cpu_utilization": (0, 100),
    "core_temperature": (30, 100),
    "disk_health_score": (0, 100),
    "uptime_hours": (0, 20000),
}


def validate_telemetry(telemetry):
    """Return a canonical numeric snapshot; reject missing or invalid features."""
    if not isinstance(telemetry, dict) or set(telemetry) != set(FEATURES):
        raise ValueError(f"Telemetry must contain exactly these features: {FEATURES}")
    cleaned = {}
    for feature, (minimum, maximum) in BOUNDS.items():
        value = telemetry[feature]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{feature} must be numeric.")
        value = float(value)
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{feature} must be between {minimum} and {maximum}.")
        cleaned[feature] = value
    return cleaned


def generate_synthetic_telemetry(n_samples=1200):
    """Simulate correlated telemetry, minority stress spikes, and noisy labels."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 100:
        raise ValueError("n_samples must be an integer of at least 100.")
    rng = np.random.default_rng(42)
    cpu = np.clip(rng.normal(48, 20, n_samples), 0, 100)
    temperature = np.clip(35 + 0.48 * cpu + rng.normal(0, 6, n_samples), 30, 100)
    uptime = np.clip(rng.gamma(2, 2200, n_samples), 0, 20000)
    disk = np.clip(98 - uptime / 650 + rng.normal(0, 10, n_samples), 0, 100)

    stressed = rng.choice(n_samples, size=max(1, int(n_samples * 0.18)), replace=False)
    cpu[stressed] = rng.uniform(86, 100, len(stressed))
    temperature[stressed] = rng.uniform(81, 100, len(stressed))
    disk[stressed] = rng.uniform(5, 39, len(stressed))
    uptime[stressed] = rng.uniform(4000, 20000, len(stressed))

    failure = (
        ((cpu > 85) & (temperature > 80) & (disk < 40))
        | ((disk < 25) & (uptime > 8000))
        | ((temperature > 95) & (cpu > 75))
    ).astype(int)
    noisy = rng.random(n_samples) < 0.015
    failure[noisy] = 1 - failure[noisy]
    return pd.DataFrame(
        {
            "cpu_utilization": cpu,
            "core_temperature": temperature,
            "disk_health_score": disk,
            "uptime_hours": uptime,
            "failure": failure,
        }
    )


class PredictiveMaintenanceModel:
    """Estimate synthetic failure-class probability, expressed as a percentage."""

    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        self._test_features = None
        self._test_labels = None
        self.training_summary = {}

    def train(self, n_samples=1200):
        data = generate_synthetic_telemetry(n_samples)
        x_train, x_test, y_train, y_test = train_test_split(
            data[FEATURES],
            data["failure"],
            test_size=0.25,
            stratify=data["failure"],
            random_state=42,
        )
        self.classifier.fit(x_train, y_train)
        self._test_features = x_test
        self._test_labels = y_test
        self.training_summary = {
            "samples": len(data),
            "training_samples": len(x_train),
            "test_samples": len(x_test),
            "failure_prevalence": float(data["failure"].mean()),
        }
        return self.evaluate()

    def _require_trained(self):
        if self._test_features is None:
            raise RuntimeError("Call train() before inference or evaluation.")

    def predict_proba(self, telemetry):
        """Return the unrounded failure-class probability multiplied by 100."""
        self._require_trained()
        snapshot = validate_telemetry(telemetry)
        frame = pd.DataFrame([snapshot], columns=FEATURES)
        failure_column = list(self.classifier.classes_).index(1)
        return float(self.classifier.predict_proba(frame)[0, failure_column] * 100)

    def evaluate(self):
        """Return held-out positive-class metrics at the default 0.5 threshold."""
        self._require_trained()
        predicted = self.classifier.predict(self._test_features)
        return {
            "f1_score": float(f1_score(self._test_labels, predicted, zero_division=0)),
            "precision": float(precision_score(self._test_labels, predicted, zero_division=0)),
            "recall": float(recall_score(self._test_labels, predicted, zero_division=0)),
        }


if __name__ == "__main__":
    model = PredictiveMaintenanceModel()
    print("Held-out evaluation:", model.train())
    print("Dataset summary:", model.training_summary)
