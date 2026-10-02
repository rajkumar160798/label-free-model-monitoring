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

    def estimate_inflight(self, metric: str, history: History, target: np.ndarray) -> float:
        """Metric of the in-flight book: history rows ``target``, some already labeled.

        Default: treat the rows like a new batch (ignore their arrived labels).
        """
        return self.estimate(metric, history.proba[target], history)


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


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


class DelayAdjustedCBPE(CBPE):
    """CBPE corrected with early, partial labels (chain-ladder style).

    1. Arrival curves: from mature rows (older than the longest observed label
       delay), estimate F(a) = P(a positive's label has arrived by age a) and
       G(a) for negatives.
    2. Offset: assume the true probability is sigmoid(logit(c) + delta), with c
       the reference-calibrated score. Fit delta by maximum likelihood on the
       most recent ``window_months`` of informative history rows:
         known positive:  p
         known negative:  1 - p
         not yet known:   1 - p F(age) - (1 - p) G(age)
       So early-arriving positives move delta immediately, without waiting for
       their cohort to complete, and are not over-counted.
    3. New batch: CBPE on the adjusted probabilities.
       In-flight book: rows already labeled count as their label; the rest use
       the posterior P(y=1 | label not arrived yet) = p(1-F) / (1 - pF - (1-p)G).

    With constant label delay (F = G = step) this reduces to recalibrating on
    the latest fully labeled data. Falls back to plain CBPE when no mature rows exist.
    """
    name = "da_cbpe"

    def __init__(self, window_months: int = 12, curve_months: int = 36, max_rows: int = 50_000,
                 seed: int = 0):
        self.window = np.timedelta64(window_months * 30, "D")
        self.curve = np.timedelta64(curve_months * 30, "D")
        self.max_rows = max_rows
        self.rng = np.random.default_rng(seed)
        self.last_delta = 0.0

    def _fit(self, history: History) -> None:
        """Arrival curves at each history row's age (F, G) and the offset delta."""
        from scipy.optimize import minimize_scalar

        n = len(history.proba)
        self.F, self.G, self.last_delta = np.zeros(n), np.zeros(n), 0.0
        known = ~np.isnan(history.y_known)
        if not known.any():
            return
        delay = (history.label_time[known] - history.event_time[known]).astype("timedelta64[D]")
        horizon = delay.max()
        age = (history.now - history.event_time).astype("timedelta64[D]")

        mature = known & (age >= horizon) & (age < horizon + self.curve)
        d_m = (history.label_time[mature] - history.event_time[mature]).astype("timedelta64[D]")
        pos_m = history.y_known[mature] == 1
        if pos_m.sum() < 10 or (~pos_m).sum() < 10:
            return
        f_delays, g_delays = np.sort(d_m[pos_m]), np.sort(d_m[~pos_m])
        F = np.searchsorted(f_delays, age, side="right") / len(f_delays)
        G = np.searchsorted(g_delays, age, side="right") / len(g_delays)
        self.F, self.G = F, G

        informative = known | (F > 0.05)
        never = ~known & (F > 0.999) & (G > 0.999)  # should have resolved but never will
        informative &= ~never
        if informative.sum() < 100:
            return
        newest = history.event_time[informative].max()
        rows = np.flatnonzero(informative & (history.event_time > newest - self.window))
        if len(rows) > self.max_rows:
            rows = self.rng.choice(rows, self.max_rows, replace=False)

        z = _logit(self.calibrator.predict(history.proba[rows]))
        y = history.y_known[rows]
        F_r, G_r = F[rows], G[rows]
        is_pos, is_neg, is_unk = y == 1, y == 0, np.isnan(y)

        def nll(delta: float) -> float:
            p = _sigmoid(z + delta)
            ll = np.log(np.clip(p[is_pos], 1e-12, None)).sum()
            ll += np.log(np.clip(1 - p[is_neg], 1e-12, None)).sum()
            unk = 1 - p[is_unk] * F_r[is_unk] - (1 - p[is_unk]) * G_r[is_unk]
            ll += np.log(np.clip(unk, 1e-12, None)).sum()
            return -ll

        self.last_delta = float(minimize_scalar(nll, bounds=(-6, 6), method="bounded").x)

    def _ensure_fit(self, history: History) -> None:
        key = (history.now, len(history.proba))
        if getattr(self, "_key", None) != key:  # one fit per decision time, shared across metrics
            self._key = key
            self._fit(history)

    def _adjusted(self, proba: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(self.calibrator.predict(proba)) + self.last_delta)

    def _metric(self, metric: str, proba: np.ndarray, c: np.ndarray) -> float:
        if metric == "prevalence":
            return float(c.mean())
        if metric == "accuracy":
            return float(np.mean(np.where(proba >= self.threshold, c, 1 - c)))
        if metric == "roc_auc":
            return expected_auc(proba, c)
        raise ValueError(metric)

    def estimate(self, metric, proba, history):
        self._ensure_fit(history)
        return self._metric(metric, proba, self._adjusted(proba))

    def estimate_inflight(self, metric, history, target):
        self._ensure_fit(history)
        proba = history.proba[target]
        p = self._adjusted(proba)
        F, G = self.F[target], self.G[target]
        y = history.y_known[target]
        posterior = p * (1 - F) / np.clip(1 - p * F - (1 - p) * G, 1e-12, None)
        c = np.where(np.isnan(y), np.clip(posterior, 0, 1), y)
        return self._metric(metric, proba, c)


class LabelsToDate(Estimator):
    """In-flight book only: metric on the labels arrived so far, counting the rest as negative.

    The common "default rate to date" figure. Biased low until cohorts complete.
    """
    name = "labels_to_date"

    def estimate(self, metric, proba, history):
        return np.nan

    def estimate_inflight(self, metric, history, target):
        y = np.nan_to_num(history.y_known[target], nan=0.0)
        return metric_value(metric, y, history.proba[target], self.threshold)


def default_estimators(freq: str = "M") -> list[Estimator]:
    return [ReferencePerformance(), RecentArrivals(), LatestCompleteCohort(freq=freq),
            CBPE(), ATC(), DoC(), DelayAdjustedCBPE()]
