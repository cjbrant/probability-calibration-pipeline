from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from scipy.special import betaln
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:  # pragma: no cover - optional dependency
    from betacal import BetaCalibration
except ImportError:  # pragma: no cover
    BetaCalibration = None


def _ensure_col_vector(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional sequence.")
    return arr


class BetaCalibratorLib:
    """Thin wrapper around betacal's BetaCalibration model."""

    model_type = "beta"

    def __init__(self) -> None:
        if BetaCalibration is None:
            raise ImportError(
                "betacal is required for the beta calibrator. Install via `pip install betacal`."
            )
        self._model = BetaCalibration(parameters="abm")
        self._fitted = False

    def fit(self, p_prior: Iterable[float], y: Iterable[int]) -> "BetaCalibratorLib":
        probs = _ensure_col_vector(p_prior)
        targets = _ensure_col_vector(y)
        if probs.shape[0] != targets.shape[0]:
            raise ValueError("Probabilities and labels must have the same length.")
        self._model.fit(probs.reshape(-1, 1), targets)
        self._fitted = True
        return self

    def predict(self, p_prior: Iterable[float]) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Calibrator is not fitted.")
        probs = _ensure_col_vector(p_prior)
        preds = self._model.predict(probs)
        return preds

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BetaCalibratorLib":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a BetaCalibratorLib instance.")
        return obj


@dataclass
class _BBQBinModel:
    edges: np.ndarray
    posterior_means: np.ndarray
    log_evidence: float


class BBQCalibrator:
    model_type = "bbq"

    def __init__(
        self,
        bins_list: Sequence[int] = (5, 10, 20),
        alpha0: float = 0.5,
        beta0: float = 0.5,
    ) -> None:
        self.bins_list = tuple(bins_list)
        self.alpha0 = alpha0
        self.beta0 = beta0
        self._models: list[_BBQBinModel] = []
        self._weights: np.ndarray | None = None

    def fit(self, p_prior: Iterable[float], y: Iterable[int]) -> "BBQCalibrator":
        probs = _ensure_col_vector(p_prior)
        targets = _ensure_col_vector(y)
        if probs.shape[0] != targets.shape[0]:
            raise ValueError("Probabilities and labels must have the same length.")

        models: list[_BBQBinModel] = []
        for bins in self.bins_list:
            if bins <= 0:
                continue
            quantiles = np.linspace(0.0, 1.0, bins + 1)
            edges = np.quantile(probs, quantiles)
            edges = np.unique(edges)
            if edges.size < 2:
                continue
            bin_ids = np.clip(
                np.digitize(probs, edges[1:-1], right=True), 0, edges.size - 2
            )
            posterior_means = np.zeros(edges.size - 1, dtype=float)
            log_evidence = 0.0

            for idx in range(edges.size - 1):
                mask = bin_ids == idx
                successes = targets[mask].sum()
                trials = mask.sum()
                failures = trials - successes
                posterior_means[idx] = (self.alpha0 + successes) / (
                    self.alpha0 + self.beta0 + trials
                )
                log_evidence += betaln(self.alpha0 + successes, self.beta0 + failures) - betaln(
                    self.alpha0, self.beta0
                )

            models.append(
                _BBQBinModel(
                    edges=edges,
                    posterior_means=posterior_means,
                    log_evidence=log_evidence,
                )
            )

        if not models:
            raise ValueError("Unable to fit BBQ calibrator with the provided bins.")

        log_evidences = np.array([model.log_evidence for model in models])
        log_evidences -= np.max(log_evidences)
        weights = np.exp(log_evidences)
        weights /= weights.sum()

        self._models = models
        self._weights = weights
        return self

    def _predict_single(self, model: _BBQBinModel, probs: np.ndarray) -> np.ndarray:
        bin_ids = np.clip(
            np.digitize(probs, model.edges[1:-1], right=True), 0, model.edges.size - 2
        )
        return model.posterior_means[bin_ids]

    def predict(self, p_prior: Iterable[float]) -> np.ndarray:
        if not self._models or self._weights is None:
            raise ValueError("Calibrator is not fitted.")
        probs = _ensure_col_vector(p_prior)
        blended = np.zeros_like(probs, dtype=float)
        for weight, model in zip(self._weights, self._models):
            blended += weight * self._predict_single(model, probs)
        return blended

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BBQCalibrator":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a BBQCalibrator instance.")
        return obj


def ece(probs: Iterable[float], y: Iterable[int], n_bins: int = 10) -> float:
    predictions = _ensure_col_vector(probs)
    labels = _ensure_col_vector(y)
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError("Probabilities and labels must have the same length.")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(predictions)
    expected_calibration_error = 0.0
    for i in range(n_bins):
        start, end = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (predictions >= start) & (predictions <= end)
        else:
            mask = (predictions >= start) & (predictions < end)
        count = np.count_nonzero(mask)
        if count == 0:
            continue
        bin_conf = predictions[mask].mean()
        bin_acc = labels[mask].mean()
        expected_calibration_error += (count / total) * abs(bin_conf - bin_acc)
    return float(expected_calibration_error)


def evaluate_probs(probs: Iterable[float], y: Iterable[int]) -> dict:
    predictions = _ensure_col_vector(probs)
    labels = _ensure_col_vector(y)
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError("Probabilities and labels must have the same length.")

    clipped = np.clip(predictions, 1e-12, 1 - 1e-12)
    report: dict[str, float | None] = {}
    report["brier"] = float(brier_score_loss(labels, clipped))
    report["log_loss"] = float(log_loss(labels, clipped))
    try:
        report["roc_auc"] = float(roc_auc_score(labels, predictions))
    except ValueError:
        report["roc_auc"] = None
    report["ece"] = ece(predictions, labels)

    hard_preds = (predictions >= 0.5).astype(int)
    report["accuracy"] = float(accuracy_score(labels, hard_preds))
    report["precision"] = float(precision_score(labels, hard_preds, zero_division=0))
    report["recall"] = float(recall_score(labels, hard_preds, zero_division=0))
    report["n"] = int(labels.size)
    return report


def load_calibrator(path: str | Path):
    try:
        calibrator = joblib.load(path)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "Failed to load calibrator due to a missing dependency. "
            "Install betacal if loading a beta model."
        ) from exc
    for attr in ("predict", "save"):
        if not hasattr(calibrator, attr):
            raise TypeError("Loaded object does not implement the calibrator interface.")
    return calibrator


__all__ = [
    "BBQCalibrator",
    "BetaCalibratorLib",
    "evaluate_probs",
    "ece",
    "load_calibrator",
]
