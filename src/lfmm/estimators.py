"""Performance estimators: guess a deployed model's metric on a new batch.

Each estimator is fit once on a labeled reference window (model scores plus
true labels) and then, for each deployment batch, sees:

- the model's scores on the batch (no labels), and
- a ``History`` of earlier rows with only the labels that have arrived so far.

Label-free estimators (CBPE, ATC, DoC) use only the scores. Label-based
baselines (RecentArrivals, LatestCompleteCohort) use what has arrived, which
is how monitoring is usually done in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

METRICS = ("roc_auc", "accuracy", "prevalence")


def metric_value(metric: str, y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.nan
    if metric == "roc_auc":
        return roc_auc_score(y, proba) if 0 < y.mean() < 1 else np.nan
    if metric == "accuracy":
        return float(np.mean((proba >= threshold) == (y == 1)))
    if metric == "prevalence":
        return float(y.mean())
    raise ValueError(f"unknown metric {metric}")


def degradation(metric: str, ref: float, value: float) -> float:
    """How much worse ``value`` is than the reference (positive = worse).

    For prevalence any change counts, since the model's calibration breaks either way.
    """
    if metric == "prevalence":
        return abs(value - ref)
    return ref - value


def expected_auc(scores: np.ndarray, p: np.ndarray) -> float:
    """AUC expected when row i is positive with probability p[i], ranked by score.

    E[AUC] ~= sum_{i,j} p_i (1 - p_j) [s_i > s_j] / (sum_i p_i * sum_j (1 - p_j) - sum_i p_i (1 - p_i)),
    with ties counted as 1/2. O(n log n) via grouping equal scores.
    """
    uniq, inv = np.unique(scores, return_inverse=True)
    pos = np.bincount(inv, weights=p, minlength=len(uniq))
    neg = np.bincount(inv, weights=1 - p, minlength=len(uniq))
    neg_below = np.cumsum(neg) - neg
    # Pairs within a tie group (excluding i == j) count 1/2.
    tie_pairs = pos * neg - np.bincount(inv, weights=p * (1 - p), minlength=len(uniq))
    num = np.sum(pos * neg_below) + 0.5 * np.sum(tie_pairs)
    den = p.sum() * (1 - p).sum() - np.sum(p * (1 - p))
    return float(num / den) if den > 0 else np.nan


@dataclass
class History:
    """Rows scored before the current batch, with labels that have arrived by ``now``."""
    proba: np.ndarray
    y_known: np.ndarray       # NaN where the label has not arrived (or never will)
    event_time: np.ndarray
    label_time: np.ndarray    # NaT where not arrived
    now: np.datetime64


class Estimator:
    name = "base"
    supports: tuple[str, ...] = METRICS

    def fit(self, proba_ref: np.ndarray, y_ref: np.ndarray, threshold: float = 0.5) -> "Estimator":
        self.threshold = threshold
        self.ref = {m: metric_value(m, y_ref, proba_ref, threshold) for m in METRICS}
        return self

    def estimate(self, metric: str, proba: np.ndarray, history: History) -> float:
        raise NotImplementedError


class ReferencePerformance(Estimator):
    """Assume nothing changed since the reference window."""
    name = "reference"

    def estimate(self, metric, proba, history):
        return self.ref[metric]


class RecentArrivals(Estimator):
    """Metric on labels that arrived in the last ``window_days``.

    Common in practice and biased under label delay: labels that arrive early
    (e.g. defaults) are over-represented.
    """
    name = "recent_arrivals"

    def __init__(self, window_days: int = 90, min_rows: int = 200):
        self.window = np.timedelta64(window_days, "D")
        self.min_rows = min_rows

    def estimate(self, metric, proba, history):
        arrived = ~np.isnat(history.label_time)
        recent = arrived & (history.label_time > history.now - self.window)
        if recent.sum() < self.min_rows:
            return np.nan
        return metric_value(metric, history.y_known[recent], history.proba[recent], self.threshold)


class LatestCompleteCohort(Estimator):
    """Metric on the most recent event cohort whose labels have (almost) all arrived."""
    name = "latest_complete_cohort"

    def __init__(self, freq: str = "M", min_coverage: float = 0.95, min_rows: int = 200):
        self.freq = freq
        self.min_coverage = min_coverage
        self.min_rows = min_rows

    def estimate(self, metric, proba, history):
        import pandas as pd

        if len(history.proba) == 0:
            return np.nan
        periods = pd.PeriodIndex(pd.DatetimeIndex(history.event_time).to_period(self.freq))
        known = ~np.isnan(history.y_known)
        stats = pd.DataFrame({"p": periods, "known": known}).groupby("p")["known"].agg(["mean", "size"])
        ok = stats[(stats["mean"] >= self.min_coverage) & (stats["size"] >= self.min_rows)]
        if ok.empty:
            return np.nan
        mask = (periods == ok.index.max()) & known
        return metric_value(metric, history.y_known[mask], history.proba[mask], self.threshold)


class CBPE(Estimator):
    """Confidence-based performance estimation (as in NannyML).

    Calibrates scores on the reference window, then treats calibrated scores as
    the probability each row is positive. Unbiased under covariate shift when
    calibration holds (P(y|x) unchanged); fails under concept shift.
    """
    name = "cbpe"

    def fit(self, proba_ref, y_ref, threshold=0.5):
        super().fit(proba_ref, y_ref, threshold)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(proba_ref, y_ref)
        return self

    def estimate(self, metric, proba, history):
        c = self.calibrator.predict(proba)
        if metric == "prevalence":
            return float(c.mean())
        if metric == "accuracy":
            pred = proba >= self.threshold
            return float(np.mean(np.where(pred, c, 1 - c)))
        if metric == "roc_auc":
            return expected_auc(proba, c)
        raise ValueError(metric)


def _confidence(proba: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(proba >= threshold, proba, 1 - proba)


class ATC(Estimator):
    """Average Thresholded Confidence (Garg et al., ICLR 2022). Accuracy only."""
    name = "atc"
    supports = ("accuracy",)

    def fit(self, proba_ref, y_ref, threshold=0.5):
        super().fit(proba_ref, y_ref, threshold)
        conf = _confidence(proba_ref, threshold)
        # Pick tau so that the share of reference rows above tau equals reference accuracy.
        self.tau = np.quantile(conf, 1 - self.ref["accuracy"])
        return self

    def estimate(self, metric, proba, history):
        if metric != "accuracy":
            return np.nan
        return float(np.mean(_confidence(proba, self.threshold) > self.tau))


class DoC(Estimator):
    """Difference of Confidences (Guillory et al., ICCV 2021). Accuracy only."""
    name = "doc"
    supports = ("accuracy",)

    def fit(self, proba_ref, y_ref, threshold=0.5):
        super().fit(proba_ref, y_ref, threshold)
        self.ref_conf = _confidence(proba_ref, threshold).mean()
        return self

    def estimate(self, metric, proba, history):
        if metric != "accuracy":
            return np.nan
        return float(self.ref["accuracy"] + _confidence(proba, self.threshold).mean() - self.ref_conf)


def default_estimators(freq: str = "M") -> list[Estimator]:
    return [ReferencePerformance(), RecentArrivals(), LatestCompleteCohort(freq=freq),
            CBPE(), ATC(), DoC()]
