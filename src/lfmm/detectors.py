"""Degradation detectors: raise an alarm when a deployed model has likely got worse.

Each detector is fit on the reference window, then for each batch returns
``(score, alarm)``: a drift score (higher = more drift) and a yes/no alarm at
the detector's conventional threshold. The harness scores both against the
true degradation, which the detector never sees.

Two families:

- input/output drift tests (KS/chi-squared, PSI, score KS, domain classifier):
  ask "has the data changed?", which is not the same as "has the model got worse?";
- estimator-based alarms: alarm when a performance estimate falls below the
  reference by more than ``delta``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from .estimators import Estimator, History, degradation


class Detector:
    name = "base"

    def fit(self, X_ref: pd.DataFrame, proba_ref: np.ndarray, y_ref: np.ndarray,
            threshold: float = 0.5) -> "Detector":
        self.X_ref = X_ref
        self.proba_ref = proba_ref
        return self

    def score(self, X: pd.DataFrame, proba: np.ndarray, history: History) -> tuple[float, bool]:
        raise NotImplementedError


def _is_cat(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype)


def _cat_counts(ref: pd.Series, cur: pd.Series) -> np.ndarray:
    """2 x k table of category counts, missing values as their own category."""
    a = ref.astype(object).fillna("<NA>").value_counts()
    b = cur.astype(object).fillna("<NA>").value_counts()
    table = pd.concat([a, b], axis=1).fillna(0).to_numpy().T
    return table[:, table.sum(axis=0) > 0]


class UnivariateTests(Detector):
    """Per-feature KS (numeric) or chi-squared (categorical) tests, Bonferroni-corrected.

    Alarms when the smallest corrected p-value is below ``alpha``. With thousands
    of rows per batch, tiny harmless shifts become significant, so this tends to
    alarm constantly. P-values also underflow to 0 at that size, so the score is
    the largest effect size instead: KS statistic (numeric) or Cramér's V
    (categorical), both in [0, 1].
    """
    name = "univariate_tests"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def score(self, X, proba, history):
        pvals, effects = [], []
        for c in self.X_ref.columns:
            if _is_cat(self.X_ref[c]):
                table = _cat_counts(self.X_ref[c], X[c])
                if table.shape[1] < 2:
                    pvals.append(1.0)
                    effects.append(0.0)
                    continue
                chi2, p = chi2_contingency(table)[:2]
                pvals.append(p)
                effects.append(np.sqrt(chi2 / table.sum()))  # Cramér's V for a 2 x k table
            else:
                a, b = self.X_ref[c].dropna(), X[c].dropna()
                if len(a) == 0 or len(b) == 0:
                    pvals.append(1.0)
                    effects.append(0.0)
                    continue
                res = ks_2samp(a, b)
                pvals.append(res.pvalue)
                effects.append(res.statistic)
        p_min = min(1.0, min(pvals) * len(pvals))
        return float(max(effects)), p_min < self.alpha


def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = expected / expected.sum() + eps
    a = actual / actual.sum() + eps
    return float(np.sum((a - e) * np.log(a / e)))


class PSI(Detector):
    """Population Stability Index, max over features (industry practice).

    Numeric features use 10 reference-quantile bins plus a missing bin. A PSI
    above 0.25 is the usual "significant shift" rule of thumb.
    """
    name = "psi_max"

    def __init__(self, alarm_at: float = 0.25, bins: int = 10):
        self.alarm_at = alarm_at
        self.bins = bins

    def fit(self, X_ref, proba_ref, y_ref, threshold=0.5):
        super().fit(X_ref, proba_ref, y_ref, threshold)
        self.edges = {}
        for c in X_ref.columns:
            if not _is_cat(X_ref[c]):
                q = np.nanquantile(X_ref[c].to_numpy(dtype=float), np.linspace(0, 1, self.bins + 1)[1:-1])
                self.edges[c] = np.unique(q)
        return self

    def _hist(self, c: str, s: pd.Series) -> np.ndarray:
        v = s.to_numpy(dtype=float)
        nan = np.isnan(v)
        counts = np.bincount(np.searchsorted(self.edges[c], v[~nan], side="right"),
                             minlength=len(self.edges[c]) + 1)
        return np.append(counts, nan.sum()).astype(float)

    def score(self, X, proba, history):
        worst = 0.0
        for c in self.X_ref.columns:
            if _is_cat(self.X_ref[c]):
                table = _cat_counts(self.X_ref[c], X[c])
                value = _psi(table[0], table[1])
            else:
                value = _psi(self._hist(c, self.X_ref[c]), self._hist(c, X[c]))
            worst = max(worst, value)
        return worst, worst > self.alarm_at


class ScoreKS(Detector):
    """KS test on the model's output scores. Score is the KS statistic."""
    name = "score_ks"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def score(self, X, proba, history):
        res = ks_2samp(self.proba_ref, proba)
        return float(res.statistic), res.pvalue < self.alpha


class DomainClassifier(Detector):
    """Classifier two-sample test: can a model tell reference rows from batch rows?

    Score is the cross-validated AUC (0.5 = indistinguishable). Alarms above ``alarm_at``.
    """
    name = "domain_classifier"

    def __init__(self, alarm_at: float = 0.6, max_rows: int = 5000, seed: int = 0):
        self.alarm_at = alarm_at
        self.max_rows = max_rows
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def _sample(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X) <= self.max_rows:
            return X
        return X.iloc[np.sort(self.rng.choice(len(X), self.max_rows, replace=False))]

    def score(self, X, proba, history):
        from .harness import _model_features

        both = pd.concat([self._sample(self.X_ref), self._sample(X)], ignore_index=True)
        label = np.r_[np.zeros(min(len(self.X_ref), self.max_rows)), np.ones(min(len(X), self.max_rows))]
        clf = HistGradientBoostingClassifier(max_iter=100, categorical_features="from_dtype",
                                             random_state=self.seed)
        p = cross_val_predict(clf, _model_features(both), label, cv=3, method="predict_proba")[:, 1]
        auc = roc_auc_score(label, p)
        return float(auc), auc > self.alarm_at


class EstimatorAlarm(Detector):
    """Alarm when an estimator's guess is worse than the reference by more than ``delta``."""

    def __init__(self, estimator: Estimator, metric: str, delta: float):
        self.estimator = estimator
        self.metric = metric
        self.delta = delta
        self.name = f"{estimator.name}:{metric}"

    def fit(self, X_ref, proba_ref, y_ref, threshold=0.5):
        super().fit(X_ref, proba_ref, y_ref, threshold)
        self.estimator.fit(proba_ref, y_ref, threshold)
        return self

    def score(self, X, proba, history):
        est = self.estimator.estimate(self.metric, proba, history)
        if np.isnan(est):
            return np.nan, False
        deg = degradation(self.metric, self.estimator.ref[self.metric], est)
        return float(deg), deg > self.delta


def default_detectors(metric: str, delta: float, freq: str = "M") -> list[Detector]:
    from .estimators import CBPE, LatestCompleteCohort, RecentArrivals

    return [
        UnivariateTests(), PSI(), ScoreKS(), DomainClassifier(),
        EstimatorAlarm(CBPE(), metric, delta),
        EstimatorAlarm(LatestCompleteCohort(freq=freq), metric, delta),
        EstimatorAlarm(RecentArrivals(), metric, delta),
    ]
