"""Walk-forward evaluation on a Stream.

1. At ``train_end``, train on event cohorts whose labels have (almost) all
   arrived by then; using partly resolved cohorts would over-sample the labels
   that arrive early. The last ``ref_months`` of those cohorts are held out as
   the reference window that monitors are fit on.
2. Score every row with the frozen model.
3. For each deployment period, give each monitor only the batch's features and
   scores, plus the labels that have arrived by the period's end. Compare its
   output to the true metric, computed from labels the monitor has not seen.

Two tasks share this loop: ``run_estimation`` (guess the metric) and
``run_detection`` (raise an alarm when performance has degraded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .estimators import METRICS, Estimator, History, default_estimators, degradation, metric_value
from .streams.base import Stream


def default_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.1, random_state=seed,
    )


def _model_features(X: pd.DataFrame, max_categories: int = 250) -> pd.DataFrame:
    """Categoricals with too many levels for HistGradientBoosting become integer codes."""
    X = X.copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype) and len(X[c].cat.categories) > max_categories:
            X[c] = X[c].cat.codes.astype(float).replace(-1, np.nan)
    return X


def usable_columns(X: pd.DataFrame) -> list[str]:
    """Columns a model can be fit on: numeric ones need at least two distinct values
    (HistGradientBoosting's binning fails otherwise), categoricals at least one."""
    keep = []
    for c in X.columns:
        n = X[c].nunique(dropna=True)
        if n >= 2 or (isinstance(X[c].dtype, pd.CategoricalDtype) and n >= 1):
            keep.append(c)
    return keep


def trainable_mask(stream: Stream, train_end, freq: str, min_coverage: float = 0.95) -> np.ndarray:
    """Rows in cohorts before train_end whose labels had (almost) all arrived by train_end."""
    now = np.datetime64(pd.Timestamp(train_end), "ns")
    before = stream.event_time < now
    known = stream.known_by(now)
    periods = stream.periods(freq)
    coverage = pd.Series(known[before]).groupby(periods[before]).mean()
    good = set(coverage[coverage >= min_coverage].index)
    return before & known & np.isin(periods, list(good))


@dataclass
class Batch:
    period: pd.Period
    mask: np.ndarray        # rows of the stream in this batch
    history: History        # earlier rows, labels arrived by the period's end
    label_coverage: float   # share of the batch whose labels ever resolve


@dataclass
class Deployment:
    """A model trained at ``train_end`` and frozen, with its reference window."""
    stream: Stream
    train_end: pd.Timestamp
    freq: str
    proba: np.ndarray
    fit_idx: np.ndarray
    ref_idx: np.ndarray
    threshold: float = 0.5

    def ref_metric(self, metric: str) -> float:
        return metric_value(metric, self.stream.y[self.ref_idx], self.proba[self.ref_idx], self.threshold)

    def truth(self, batch: Batch, metric: str, min_coverage: float = 0.9) -> float:
        if batch.label_coverage < min_coverage:
            return np.nan
        y = self.stream.y[batch.mask]
        ok = ~np.isnan(y)
        return metric_value(metric, y[ok], self.proba[batch.mask][ok], self.threshold)

    def periods(self) -> list[pd.Period]:
        """Deployment periods: those starting at or after train_end."""
        s = self.stream
        return list(s.periods(self.freq)[s.event_time >= np.datetime64(self.train_end, "ns")]
                    .unique().sort_values())

    def batch(self, p: pd.Period) -> Batch:
        s = self.stream
        in_batch = np.asarray(s.periods(self.freq) == p)
        now = np.datetime64(p.end_time.floor("D"), "ns")
        past = s.event_time < np.datetime64(p.start_time, "ns")
        known = s.known_by(now)[past]
        history = History(
            proba=self.proba[past],
            y_known=np.where(known, s.y[past], np.nan),
            event_time=s.event_time[past],
            label_time=np.where(known, s.label_time[past], np.datetime64("NaT", "ns")),
            now=now,
        )
        return Batch(p, in_batch, history, float((~np.isnan(s.y[in_batch])).mean()))

    def batches(self) -> Iterator[Batch]:
        for p in self.periods():
            yield self.batch(p)

    def info(self, metrics) -> dict:
        t = self.stream.event_time
        return {"n_fit": len(self.fit_idx), "n_ref": len(self.ref_idx),
                "fit_range": (str(t[self.fit_idx].min())[:10], str(t[self.fit_idx].max())[:10]),
                "ref_range": (str(t[self.ref_idx].min())[:10], str(t[self.ref_idx].max())[:10]),
                "ref_metrics": {m: self.ref_metric(m) for m in metrics}}


def deploy(stream: Stream, train_end, freq: str = "Q", ref_months: int = 12, model=None,
           train_start=None, train_window_months: int | None = None,
           max_train_rows: int | None = 500_000, threshold: float = 0.5,
           seed: int = 0) -> Deployment:
    """Train at ``train_end`` on fully labeled cohorts, optionally only those
    after ``train_start`` or within the last ``train_window_months`` of them."""
    train_end = pd.Timestamp(train_end)
    model = model if model is not None else default_model(seed)
    rng = np.random.default_rng(seed)

    mask = trainable_mask(stream, train_end, freq)
    if train_start is not None:
        mask &= stream.event_time >= np.datetime64(pd.Timestamp(train_start), "ns")
    if train_window_months is not None and mask.any():
        newest = stream.event_time[mask].max()
        mask &= stream.event_time > newest - np.timedelta64(train_window_months * 30, "D")
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("no fully labeled cohorts before train_end")
    ref_start = stream.event_time[idx].max() - np.timedelta64(ref_months * 30, "D")
    is_ref = stream.event_time[idx] > ref_start
    if is_ref.mean() > 0.7:  # too little history for a time split: split randomly
        is_ref = rng.random(len(idx)) < 0.3
    fit_idx, ref_idx = idx[~is_ref], idx[is_ref]
    if max_train_rows is not None and len(fit_idx) > max_train_rows:
        fit_idx = np.sort(rng.choice(fit_idx, max_train_rows, replace=False))

    X = _model_features(stream.X)
    # A feature that is missing or constant at training time (e.g. a field added
    # later) cannot be learned from; drop it rather than crash the binning.
    X = X[usable_columns(X.iloc[fit_idx])]
    model.fit(X.iloc[fit_idx], stream.y[fit_idx])
    proba = model.predict_proba(X)[:, 1]
    return Deployment(stream, train_end, freq, proba, fit_idx, ref_idx, threshold)


# --- Task 1: performance estimation -------------------------------------------

def _block_bootstrap(e: pd.DataFrame, n_boot: int, block: int, seed: int) -> np.ndarray:
    """Means of ``e``'s columns over moving-block resamples of its rows."""
    rng = np.random.default_rng(seed)
    x = e.to_numpy(dtype=float)
    n = len(x)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return np.nanmean(x[idx], axis=1)

@dataclass
class Result:
    stream: str
    train_end: pd.Timestamp
    records: pd.DataFrame  # one row per (period, metric, estimator)
    info: dict

    def summary(self) -> pd.DataFrame:
        """MAE and bias of each estimator per metric, over periods with complete truth."""
        r = self.records.dropna(subset=["truth", "estimate"])
        r = r.assign(err=r["estimate"] - r["truth"])
        out = r.groupby(["metric", "estimator"]).agg(
            mae=("err", lambda e: e.abs().mean()),
            bias=("err", "mean"),
            periods=("err", "size"),
        )
        return out.sort_values(["metric", "mae"])

    def _errors(self, metric: str) -> pd.DataFrame:
        """Absolute error per period (rows) and estimator (columns), complete periods only."""
        r = self.records[self.records["metric"] == metric].dropna(subset=["truth"])
        err = (r["estimate"] - r["truth"]).abs()
        return r.assign(err=err).pivot(index="period", columns="estimator", values="err").sort_index()

    def bootstrap_mae(self, metric: str, n_boot: int = 2000, block: int = 4, seed: int = 0) -> pd.DataFrame:
        """MAE with a 95% moving-block bootstrap interval over periods.

        Blocks of ``block`` consecutive periods keep the autocorrelation of
        errors (a shock affects several periods in a row).
        """
        e = self._errors(metric).dropna(axis=1, how="all")
        boots = _block_bootstrap(e, n_boot, block, seed)
        return pd.DataFrame({"mae": e.mean(), "lo": np.nanpercentile(boots, 2.5, axis=0),
                             "hi": np.nanpercentile(boots, 97.5, axis=0)}).sort_values("mae")

    def compare(self, metric: str, a: str, b: str, n_boot: int = 2000, block: int = 4,
                seed: int = 0) -> dict:
        """MAE(a) - MAE(b) with a 95% block-bootstrap interval (negative = a is better)."""
        e = self._errors(metric)[[a, b]].dropna()
        d = (e[a] - e[b]).to_frame("d")
        boots = _block_bootstrap(d, n_boot, block, seed)[:, 0]
        return {"diff": float(d["d"].mean()), "lo": float(np.percentile(boots, 2.5)),
                "hi": float(np.percentile(boots, 97.5)), "periods": len(d)}

    def wide(self, metric: str) -> pd.DataFrame:
        r = self.records[self.records["metric"] == metric]
        est = r.pivot(index="period", columns="estimator", values="estimate")
        base = r.drop_duplicates("period").set_index("period")[["n", "truth", "label_coverage"]]
        return base.join(est)


def run_estimation(
    stream: Stream,
    train_end,
    freq: str = "Q",
    metrics: tuple[str, ...] = ("roc_auc",),
    estimators: list[Estimator] | None = None,
    min_truth_coverage: float = 0.9,
    deployment: Deployment | None = None,
    **deploy_kwargs,
) -> Result:
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"unknown metric {m}")
    dep = deployment or deploy(stream, train_end, freq, **deploy_kwargs)
    estimators = estimators if estimators is not None else default_estimators(freq="M")
    for est in estimators:
        est.fit(dep.proba[dep.ref_idx], stream.y[dep.ref_idx], dep.threshold)

    rows = []
    for b in dep.batches():
        b_proba = dep.proba[b.mask]
        for m in metrics:
            truth = dep.truth(b, m, min_truth_coverage)
            for est in estimators:
                value = est.estimate(m, b_proba, b.history) if m in est.supports else np.nan
                rows.append({"period": str(b.period), "metric": m, "estimator": est.name,
                             "estimate": value, "truth": truth, "n": int(b.mask.sum()),
                             "label_coverage": b.label_coverage})
    return Result(stream.name, dep.train_end, pd.DataFrame(rows), dep.info(metrics))


def run_inflight(
    stream: Stream,
    train_end,
    freq: str = "Q",
    book_months: int = 24,
    metrics: tuple[str, ...] = ("prevalence", "roc_auc"),
    estimators: list[Estimator] | None = None,
    min_truth_coverage: float = 0.9,
    deployment: Deployment | None = None,
    **deploy_kwargs,
) -> Result:
    """Task 1b: estimate the metric of the in-flight book at each period end.

    The book is every row scored in the last ``book_months`` before the current
    period. Some of its labels have arrived (early ones, e.g. defaults); the rest
    are pending. For long-horizon labels this is the question monitoring can
    actually answer: a new batch's eventual outcome depends on events that have
    not happened yet.
    """
    dep = deployment or deploy(stream, train_end, freq, **deploy_kwargs)
    estimators = estimators if estimators is not None else default_estimators(freq="M")
    for est in estimators:
        est.fit(dep.proba[dep.ref_idx], stream.y[dep.ref_idx], dep.threshold)
    window = np.timedelta64(book_months * 30, "D")

    rows = []
    for b in dep.batches():
        past_idx = np.flatnonzero(stream.event_time < np.datetime64(b.period.start_time, "ns"))
        target = b.history.event_time > b.history.now - window
        idx = past_idx[target]
        y = stream.y[idx]
        coverage = float((~np.isnan(y)).mean()) if len(y) else 0.0
        arrived = float((~np.isnan(b.history.y_known[target])).mean()) if len(y) else 0.0
        for m in metrics:
            ok = ~np.isnan(y)
            truth = (metric_value(m, y[ok], dep.proba[idx][ok], dep.threshold)
                     if coverage >= min_truth_coverage else np.nan)
            for est in estimators:
                value = est.estimate_inflight(m, b.history, target) if m in est.supports else np.nan
                rows.append({"period": str(b.period), "metric": m, "estimator": est.name,
                             "estimate": value, "truth": truth, "n": int(target.sum()),
                             "label_coverage": coverage, "arrived_share": arrived})
    return Result(stream.name, dep.train_end, pd.DataFrame(rows), dep.info(metrics))


# --- Task 2: degradation detection --------------------------------------------

@dataclass
class DetectionResult:
    stream: str
    train_end: pd.Timestamp
    records: pd.DataFrame  # one row per (period, detector)
    event_metric: str
    deltas: tuple[float, ...]
    alarm_delta: float
    info: dict

    def summary(self) -> pd.DataFrame:
        """Per detector, over periods with complete truth:

        - spearman: rank correlation of drift score with actual degradation,
        - auroc@d: how well the score separates periods degraded by more than d,
        - precision / recall / false_alarm_rate of alarms at ``alarm_delta``.
        """
        r = self.records.dropna(subset=["degradation", "score"])
        out = []
        for name, g in r.groupby("detector", sort=False):
            row = {"detector": name, "periods": len(g),
                   "spearman": spearmanr(g["score"], g["degradation"]).statistic
                   if g["score"].nunique() > 1 else np.nan}
            for d in self.deltas:
                ev = g["degradation"] > d
                row[f"auroc@{d:g}"] = (roc_auc_score(ev, g["score"])
                                       if 0 < ev.mean() < 1 else np.nan)
            ev = g["degradation"] > self.alarm_delta
            al = g["alarm"].astype(bool)
            row["event_rate"] = ev.mean()
            row["alarm_rate"] = al.mean()
            row["precision"] = (al & ev).sum() / al.sum() if al.any() else np.nan
            row["recall"] = (al & ev).sum() / ev.sum() if ev.any() else np.nan
            row["false_alarm_rate"] = (al & ~ev).sum() / (~ev).sum() if (~ev).any() else np.nan
            out.append(row)
        return pd.DataFrame(out).set_index("detector").sort_values("spearman", ascending=False)


def run_detection_events(
    stream: Stream,
    train_end,
    detectors: list,
    events: dict[str, tuple[tuple[float, ...], float]],
    freq: str = "Q",
    min_truth_coverage: float = 0.9,
    deployment: Deployment | None = None,
    **deploy_kwargs,
) -> dict[str, DetectionResult]:
    """Score every detector once per period, then against each event definition.

    ``events`` maps a metric to (deltas scored by AUROC, delta used for alarms).
    Running the detectors once and reusing their output for every event is much
    cheaper than one run per metric (the slow detectors don't depend on the metric).
    """
    dep = deployment or deploy(stream, train_end, freq, **deploy_kwargs)
    ref_rows = dep.ref_idx
    X_ref = stream.X.iloc[ref_rows].reset_index(drop=True)
    for det in detectors:
        det.fit(X_ref, dep.proba[ref_rows], stream.y[ref_rows], dep.threshold)
    ref_values = {m: dep.ref_metric(m) for m in events}

    rows = []
    for b in dep.batches():
        X_b = stream.X.loc[b.mask].reset_index(drop=True)
        truths = {m: dep.truth(b, m, min_truth_coverage) for m in events}
        for det in detectors:
            score, alarm = det.score(X_b, dep.proba[b.mask], b.history)
            for m, truth in truths.items():
                deg = degradation(m, ref_values[m], truth) if not np.isnan(truth) else np.nan
                rows.append({"metric": m, "period": str(b.period), "detector": det.name,
                             "score": score, "alarm": bool(alarm), "truth": truth,
                             "degradation": deg, "n": int(b.mask.sum()),
                             "label_coverage": b.label_coverage})
    records = pd.DataFrame(rows)
    return {
        m: DetectionResult(stream.name, dep.train_end,
                           records[records["metric"] == m].drop(columns="metric").reset_index(drop=True),
                           m, tuple(deltas), alarm_delta, dep.info((m,)))
        for m, (deltas, alarm_delta) in events.items()
    }


def run_detection(
    stream: Stream,
    train_end,
    detectors: list,
    freq: str = "Q",
    event_metric: str = "roc_auc",
    deltas: tuple[float, ...] = (0.02, 0.05, 0.08),
    alarm_delta: float | None = None,
    min_truth_coverage: float = 0.9,
    deployment: Deployment | None = None,
    **deploy_kwargs,
) -> DetectionResult:
    """Single-event version of ``run_detection_events``."""
    alarm_delta = alarm_delta if alarm_delta is not None else deltas[1]
    return run_detection_events(stream, train_end, detectors, {event_metric: (tuple(deltas), alarm_delta)},
                                freq, min_truth_coverage, deployment, **deploy_kwargs)[event_metric]
