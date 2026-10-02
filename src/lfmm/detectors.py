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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from .estimators import Estimator, History, degradation


def _adwin(delta: float):
    try:
        from river.drift import ADWIN
    except ImportError as exc:  # optional dependency
        raise ImportError("ADWIN detectors need the 'river' package: pip install 'lfmm[streaming]'") from exc
    return ADWIN(delta=delta)


def _has_river() -> bool:
    import importlib.util

    return importlib.util.find_spec("river") is not None


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
        from .harness import _model_features, usable_columns

        both = pd.concat([self._sample(self.X_ref), self._sample(X)], ignore_index=True)
        both = _model_features(both)
        # Each CV fold must see some values of every column (all-missing breaks the binning).
        both = both[[c for c in usable_columns(both) if both[c].notna().sum() >= 10]]
        label = np.r_[np.zeros(min(len(self.X_ref), self.max_rows)), np.ones(min(len(X), self.max_rows))]
        clf = HistGradientBoostingClassifier(max_iter=100, categorical_features="from_dtype",
                                             random_state=self.seed)
        folds = StratifiedKFold(3, shuffle=True, random_state=self.seed)
        p = cross_val_predict(clf, both, label, cv=folds, method="predict_proba")[:, 1]
        auc = roc_auc_score(label, p)
        return float(auc), auc > self.alarm_at


class MMD(Detector):
    """Kernel two-sample test (Gretton et al., 2012) on numeric features plus the model score.

    Features are standardized with reference statistics (missing -> reference
    median); RBF bandwidth by the median heuristic on the reference. Score is the
    unbiased MMD^2; alarm when the permutation p-value is below ``alpha``.
    """
    name = "mmd"

    def __init__(self, alpha: float = 0.05, n: int = 1000, permutations: int = 200, seed: int = 0):
        self.alpha, self.n, self.permutations = alpha, n, permutations
        self.rng = np.random.default_rng(seed)

    def _matrix(self, X: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
        num = X[self.cols].to_numpy(dtype=float)
        num = np.where(np.isnan(num), self.median, num)
        return np.column_stack([(num - self.mean) / self.std, (proba - self.p_mean) / self.p_std])

    def fit(self, X_ref, proba_ref, y_ref, threshold=0.5):
        super().fit(X_ref, proba_ref, y_ref, threshold)
        self.cols = [c for c in X_ref.columns if not _is_cat(X_ref[c]) and X_ref[c].notna().any()]
        num = X_ref[self.cols].to_numpy(dtype=float)
        self.median = np.nanmedian(num, axis=0)
        self.mean = np.nanmean(num, axis=0)
        self.std = np.nanstd(num, axis=0) + 1e-9
        self.p_mean, self.p_std = proba_ref.mean(), proba_ref.std() + 1e-9
        self.Z_ref = self._sample(self._matrix(X_ref, proba_ref))
        d = np.sum((self.Z_ref[:300, None, :] - self.Z_ref[None, :300, :]) ** 2, axis=-1)
        self.gamma = 1.0 / np.median(d[d > 0])
        return self

    def _sample(self, Z):
        return Z if len(Z) <= self.n else Z[self.rng.choice(len(Z), self.n, replace=False)]

    @staticmethod
    def _mmd2(K, idx_a, idx_b):
        kaa = K[np.ix_(idx_a, idx_a)]
        kbb = K[np.ix_(idx_b, idx_b)]
        m, n = len(idx_a), len(idx_b)
        return ((kaa.sum() - np.trace(kaa)) / (m * (m - 1)) + (kbb.sum() - np.trace(kbb)) / (n * (n - 1))
                - 2 * K[np.ix_(idx_a, idx_b)].mean())

    def score(self, X, proba, history):
        Z = np.vstack([self.Z_ref, self._sample(self._matrix(X, proba))])
        sq = np.sum(Z ** 2, axis=1)
        K = np.exp(-self.gamma * np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))
        m = len(self.Z_ref)
        idx = np.arange(len(Z))
        stat = self._mmd2(K, idx[:m], idx[m:])
        null = []
        for _ in range(self.permutations):
            perm = self.rng.permutation(len(Z))
            null.append(self._mmd2(K, perm[:m], perm[m:]))
        p = (1 + np.sum(np.array(null) >= stat)) / (1 + self.permutations)
        return float(stat), p < self.alpha


class ADWINScores(Detector):
    """ADWIN (Bifet & Gavalda, 2007) on the stream of model scores. Label-free.

    Stateful: batches must arrive in time order. Up to ``per_batch`` scores per
    batch are fed in order. Score is how far ADWIN's current window mean is from
    the reference mean; alarm if ADWIN signalled a change during the batch.
    """
    name = "adwin_scores"

    def __init__(self, delta: float = 0.002, per_batch: int = 2000):
        self.delta, self.per_batch = delta, per_batch

    def fit(self, X_ref, proba_ref, y_ref, threshold=0.5):
        super().fit(X_ref, proba_ref, y_ref, threshold)
        self.adwin = _adwin(self.delta)
        for v in proba_ref[:: max(1, len(proba_ref) // self.per_batch)]:
            self.adwin.update(float(v))
        self.ref_mean = float(proba_ref.mean())
        return self

    def score(self, X, proba, history):
        detected = False
        for v in proba[:: max(1, len(proba) // self.per_batch)]:
            self.adwin.update(float(v))
            detected |= self.adwin.drift_detected
        return abs(self.adwin.estimation - self.ref_mean), detected


class ADWINErrors(Detector):
    """ADWIN on the squared error of predictions whose labels have just arrived.

    The usual way ADWIN is deployed: it only sees labels as they arrive, so under
    label delay it reacts late, and to early-arriving labels first.
    """
    name = "adwin_errors"

    def __init__(self, delta: float = 0.002, per_batch: int = 2000):
        self.delta, self.per_batch = delta, per_batch

    def fit(self, X_ref, proba_ref, y_ref, threshold=0.5):
        super().fit(X_ref, proba_ref, y_ref, threshold)
        self.adwin = _adwin(self.delta)
        err = (y_ref - proba_ref) ** 2
        for v in err[:: max(1, len(err) // self.per_batch)]:
            self.adwin.update(float(v))
        self.ref_mean = float(err.mean())
        self.last_now = None
        return self

    def score(self, X, proba, history):
        known = ~np.isnan(history.y_known)
        new = known if self.last_now is None else known & (history.label_time > self.last_now)
        self.last_now = history.now
        idx = np.flatnonzero(new)
        idx = idx[np.argsort(history.label_time[idx], kind="stable")]
        detected = False
        for i in idx[:: max(1, len(idx) // self.per_batch)]:
            self.adwin.update(float((history.y_known[i] - history.proba[i]) ** 2))
            detected |= self.adwin.drift_detected
        return abs(self.adwin.estimation - self.ref_mean), detected


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
    from .estimators import CBPE, DelayAdjustedCBPE, LatestCompleteCohort, RecentArrivals

    return [
        UnivariateTests(), PSI(), ScoreKS(), DomainClassifier(), MMD(),
        *([ADWINScores(), ADWINErrors()] if _has_river() else []),
        EstimatorAlarm(CBPE(), metric, delta),
        EstimatorAlarm(DelayAdjustedCBPE(), metric, delta),
        EstimatorAlarm(LatestCompleteCohort(freq=freq), metric, delta),
        EstimatorAlarm(RecentArrivals(), metric, delta),
    ]


def detectors_for_events(events: dict, freq: str = "M") -> list[Detector]:
    """Input/output drift detectors once, plus estimator alarms for each event metric."""
    from .estimators import CBPE, DelayAdjustedCBPE, LatestCompleteCohort, RecentArrivals

    dets: list[Detector] = [UnivariateTests(), PSI(), ScoreKS(), DomainClassifier(), MMD(permutations=100),
                            *([ADWINScores(), ADWINErrors()] if _has_river() else [])]
    for metric, (_, alarm_delta) in events.items():
        dets += [EstimatorAlarm(est, metric, alarm_delta) for est in
                 (CBPE(), DelayAdjustedCBPE(), LatestCompleteCohort(freq=freq), RecentArrivals())]
    return dets
