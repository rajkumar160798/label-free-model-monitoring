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

    def _prepare(self, history: History) -> dict | None:
        """Arrival curves at each history row's age (sets self.F, self.G) and the rows to
        fit the offset on. None when there is not enough labeled history (plain CBPE)."""
        n = len(history.proba)
        self.F, self.G, self.last_delta = np.zeros(n), np.zeros(n), 0.0
        known = ~np.isnan(history.y_known)
        if not known.any():
            return None
        delay = (history.label_time[known] - history.event_time[known]).astype("timedelta64[D]")
        horizon = delay.max()
        age = (history.now - history.event_time).astype("timedelta64[D]")

        mature = known & (age >= horizon) & (age < horizon + self.curve)
        d_m = (history.label_time[mature] - history.event_time[mature]).astype("timedelta64[D]")
        pos_m = history.y_known[mature] == 1
        if pos_m.sum() < 10 or (~pos_m).sum() < 10:
            return None
        f_delays, g_delays = np.sort(d_m[pos_m]), np.sort(d_m[~pos_m])
        self.F = np.searchsorted(f_delays, age, side="right") / len(f_delays)
        self.G = np.searchsorted(g_delays, age, side="right") / len(g_delays)

        informative = known | (self.F > 0.05)
        never = ~known & (self.F > 0.999) & (self.G > 0.999)  # should have resolved but never will
        informative &= ~never
        if informative.sum() < 100:
            return None
        newest = history.event_time[informative].max()
        rows = np.flatnonzero(informative & (history.event_time > newest - self.window))
        if len(rows) > self.max_rows:
            rows = self.rng.choice(rows, self.max_rows, replace=False)
        return {"rows": rows, "age": age, "f_delays": f_delays,
                "z": _logit(self.calibrator.predict(history.proba[rows])),
                "y": history.y_known[rows]}

    def _fit(self, history: History) -> None:
        """Fit the offset delta by maximum likelihood (see class docstring)."""
        from scipy.optimize import minimize_scalar

        prep = self._prepare(history)
        if prep is None:
            return
        rows, z, y = prep["rows"], prep["z"], prep["y"]
        F_r, G_r = self.F[rows], self.G[rows]
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


# Fitted hazard parameters by (decision time, history, reference): the estimation
# and in-flight tasks fit the same model at the same time; fit it once.
_HAZARD_FITS: dict = {}


class HazardAdjustedCBPE(DelayAdjustedCBPE):
    """Delay-adjusted CBPE as a discrete-time hazard model with calendar-month shocks.

    Fixes da_cbpe's weakness with shocks that hit every row at once (e.g. COVID
    forbearance), which it reads as a lasting shift. An age-period-cohort model on
    a monthly grid, over rows from the last ``window_months``:

    - age: positives' arrival curve F (from mature rows) gives each row a monthly
      hazard increment L(k) = log(1 - p F(k-1)) - log(1 - p F(k)), where
      p = sigmoid(logit(c) + delta) is its eventual probability;
    - cohort: delta_new for rows from the newest ``new_cohort_months``, delta_old
      for older ones. Only delta_new is projected to new rows;
    - period: in each of the last ``calendar_months`` calendar months, every row
      alive in that month gets the same extra hazard lambda_t >= 0 (a burst adds
      defaults, the same for safe and risky rows, and never removes them).

    Likelihood (G = negatives' arrival curve, S = exp(-cumulative hazard)):
      positive arriving in month k*:  S(k*-1) - S(k*)
      known negative:                 S(H)
      pending at age a:               S(a) - S(H) G(age)
    Older rows anchor lambda (a burst hits them too; a new-cohort shift does not),
    which is what separates a burst from a lasting shift. New rows get
    p = sigmoid(logit(c) + delta_new); pending in-flight rows use the posterior
    (S(a) - S(H)) / (S(a) - S(H) G) with the fitted bursts. Analytic gradients.
    """
    name = "da_cbpe_hz"
    MONTH_DAYS = 30.44

    def __init__(self, calendar_months: int = 24, ridge: float = 1.0, window_months: int = 24,
                 new_cohort_months: int = 12, max_rows: int = 20_000, **kwargs):
        super().__init__(window_months=window_months, max_rows=max_rows, **kwargs)
        self.K = calendar_months
        self.ridge = ridge
        self.new_cohort = np.timedelta64(int(new_cohort_months * 30.44), "D")
        self.last_kappa = np.zeros(calendar_months)
        self.last_delta_old = 0.0

    # -- grid helpers ------------------------------------------------------------

    def _increments(self, p: np.ndarray) -> np.ndarray:
        """Monthly hazard increments L(k) = log(1 - p F(k-1)) - log(1 - p F(k)), n x H."""
        Fk = np.clip(p[:, None] * self._Fm[None, :], 0, 1 - 1e-9)
        return np.log1p(-Fk[:, :-1]) - np.log1p(-Fk[:, 1:])

    def _grid(self, history: History, idx: np.ndarray):
        """Per row: calendar slot of each age month (n x H; -1 = no kappa), completed
        age months a, fraction of the current month already lived, label arrival month k*."""
        H = self._H
        age_days = (history.now - history.event_time[idx]) / np.timedelta64(1, "D")
        a = np.clip(np.floor(age_days / self.MONTH_DAYS), 0, H).astype(int)
        frac = np.where(a < H, age_days / self.MONTH_DAYS - a, 0.0)
        # months ago (from now) at which age month k (1..H) ended
        k = np.arange(1, H + 1)
        months_ago = np.floor((age_days[:, None] - k[None, :] * self.MONTH_DAYS) / self.MONTH_DAYS)
        slot = np.where((months_ago >= 0) & (months_ago < self.K), self.K - 1 - months_ago, -1).astype(int)
        current = k[None, :] == a[:, None] + 1                         # the month being lived now
        slot = np.where(current & (self.K > 0), self.K - 1, slot)
        slot = np.where(k[None, :] <= a[:, None] + 1, slot, -1)       # future months: no kappa
        lab_days = (history.label_time[idx] - history.event_time[idx]) / np.timedelta64(1, "D")
        kstar = np.clip(np.ceil(np.nan_to_num(lab_days) / self.MONTH_DAYS), 1, H).astype(int)
        return slot, a, frac, kstar

    @staticmethod
    def _cum_now(cum, a, frac):
        """Cumulative hazard at each row's exact current age (linear within the month)."""
        n = np.arange(len(a))
        nxt = np.minimum(a + 1, cum.shape[1] - 1)
        return cum[n, a] + frac * (cum[n, nxt] - cum[n, a])

    def _cum(self, L, slot, lam):
        shock = np.where(slot >= 0, lam[np.maximum(slot, 0)], 0.0) if len(lam) else 0.0
        return np.concatenate([np.zeros((len(L), 1)), np.cumsum(L + shock, axis=1)], axis=1)  # n x (H+1)

    # -- fitting -----------------------------------------------------------------

    def _fit(self, history: History) -> None:
        from scipy.optimize import minimize

        self.last_kappa = np.zeros(self.K)
        self._H = None
        prep = self._prepare(history)
        if prep is None:
            return
        # Grid length from the 99.5th percentile of positives' delays: a few very late
        # arrivals (e.g. modified loans whose age was reset) would otherwise stretch the
        # grid to years; they fall in the last month instead.
        f_days = prep["f_delays"] / np.timedelta64(1, "D")
        horizon_days = float(np.quantile(f_days, 0.995))
        self._H = H = max(1, int(np.ceil(horizon_days / self.MONTH_DAYS)))
        edges = (np.arange(H + 1) * self.MONTH_DAYS).astype("timedelta64[D]")
        Fm = np.searchsorted(prep["f_delays"], edges, side="right") / len(prep["f_delays"])
        Fm[-1] = 1.0
        # Blend in 1% of a uniform arrival curve: an empirical curve can have months with
        # no mass (e.g. defaults almost never arrive in month 1), which would give an
        # arrival in that month probability 0 and an infinite log-likelihood.
        self._Fm = 0.99 * Fm + 0.01 * np.arange(H + 1) / H

        rows, y = prep["rows"], prep["y"]
        z = prep["z"]
        t = history.event_time[rows]
        new = t > t.max() - self.new_cohort  # newest cohorts: their shift is projected forward
        slot, a, frac, kstar = self._grid(history, rows)
        G = self.G[rows]
        n = np.arange(len(rows))
        pos, neg, unk = y == 1, y == 0, np.isnan(y)
        # Pending rows past the horizon (or whose negatives have all arrived) should have
        # resolved: under the model their probability is ~0 and they carry no
        # information, while clipping it would break the gradient. Leave them out.
        unk &= (a < H) & (G < 0.999)
        in_current = kstar == a + 1  # arrived during the month being lived now

        nll = self._objective(z, new, slot, a, frac, kstar, G, pos, neg, unk, in_current)

        bounds = [(-5, 5), (-5, 5)] + [(0, 0.5)] * self.K
        key = (history.now, len(history.proba), float(history.proba[:1000].sum()), self.ref.get("prevalence"))
        if key in _HAZARD_FITS:  # same deployment and decision time, fitted by another task
            self.last_delta, self.last_delta_old, self.last_kappa = _HAZARD_FITS[key]
            return
        # warm start from the previous decision time: consecutive periods are similar
        x0 = getattr(self, "_x0", None)
        if x0 is None or len(x0) != self.K + 2:
            x0 = np.zeros(self.K + 2)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(nll, x0=x0, jac=True, method="L-BFGS-B", bounds=bounds)
        self._x0 = res.x
        self.last_delta, self.last_delta_old, self.last_kappa = float(res.x[0]), float(res.x[1]), res.x[2:]
        _HAZARD_FITS[key] = (self.last_delta, self.last_delta_old, self.last_kappa)

    def _objective(self, z, new, slot, a, frac, kstar, G, pos, neg, unk, in_current):
        """Negative log-likelihood (+ ridge) and its analytic gradient in
        (delta_new, delta_old, kappa_1..K)."""
        H, K, Fm = self._H, self.K, self._Fm
        n = np.arange(len(z))
        nxt = np.minimum(a + 1, H)
        zero = np.zeros((len(z), 1))
        # d cum / d lambda_m: months lived in calendar slot m (fixed)
        dcum_k = [np.concatenate([zero, np.cumsum(slot == m, axis=1)], axis=1).astype(float)
                  for m in range(K)]

        def at(arr, cols):  # arr: n x (H+1) [x extra], pick column per row
            return arr[n, cols]

        def f(theta):
            kappa = theta[2:]
            # Bound the logit so p < 1: at p = 1 the hazard is infinite and the
            # line search fails. The derivative is 0 where the bound is active.
            u_raw = z + np.where(new, theta[0], theta[1])
            u = np.clip(u_raw, -15.0, 9.0)
            p = _sigmoid(u)
            pF = p[:, None] * Fm[None, :]                               # n x (H+1), < 1
            L = np.log1p(-pF[:, :-1]) - np.log1p(-pF[:, 1:])            # n x H
            # dL/dp, then chain rule through dp/ddelta = p(1-p)
            dLdp = -Fm[None, :-1] / (1 - pF[:, :-1]) + Fm[None, 1:] / (1 - pF[:, 1:])
            shock = np.where(slot >= 0, kappa[np.maximum(slot, 0)], 0.0) if K else 0.0
            zero = np.zeros((len(z), 1))
            cum = np.concatenate([zero, np.cumsum(L + shock, axis=1)], axis=1)
            dpdu = p * (1 - p) * (u == u_raw)
            dcum_d = np.concatenate([zero, np.cumsum(dLdp, axis=1)], axis=1) * dpdu[:, None]

            def now(arr):
                return at(arr, a) + frac * (at(arr, nxt) - at(arr, a))

            S = np.exp(-cum)
            c_now = now(cum)
            S_now = np.exp(-c_now)
            grads = [dcum_d * new[:, None], dcum_d * ~new[:, None]] + dcum_k  # each n x (H+1)
            g_now = [now(g) for g in grads]
            g_H = [g[:, H] for g in grads]
            S_H = S[:, H]

            # positives: log(S(k*-1) - U), U = S(k*) or S_now for the current month
            lo = at(S, kstar - 1)
            hi = np.where(in_current, S_now, at(S, kstar))
            den_p = np.clip(lo - hi, 1e-300, None)
            ll = np.log(den_p[pos]).sum()
            g = np.zeros(K + 2)
            for j, gr in enumerate(grads):
                d_lo = -lo * at(gr, kstar - 1)
                d_hi = -hi * np.where(in_current, g_now[j], at(gr, kstar))
                g[j] += ((d_lo - d_hi)[pos] / den_p[pos]).sum()
            # negatives: log S(H) = -cum(H)
            ll += -cum[neg, H].sum()
            for j in range(K + 2):
                g[j] += -g_H[j][neg].sum()
            # pending: log(S_now - S(H) G)
            den_u = np.clip(S_now - S_H * G, 1e-300, None)
            ll += np.log(den_u[unk]).sum()
            for j in range(K + 2):
                d = -S_now * g_now[j] + S_H * G * g_H[j]
                g[j] += (d[unk] / den_u[unk]).sum()

            obj = -ll + self.ridge * np.sum(kappa ** 2)
            grad = -g
            grad[2:] += 2 * self.ridge * kappa
            return obj, grad

        return f

    # -- estimates ---------------------------------------------------------------

    def estimate_inflight(self, metric, history, target):
        self._ensure_fit(history)
        idx = np.flatnonzero(target)
        proba = history.proba[idx]
        y = history.y_known[idx]
        if self._H is None:  # no curves: plain CBPE on pending rows
            c = np.where(np.isnan(y), self.calibrator.predict(proba), y)
            return self._metric(metric, proba, c)
        slot, a, frac, _ = self._grid(history, idx)
        cum = self._cum(self._increments(self._adjusted(proba)), slot, self.last_kappa)
        Sa, SH = np.exp(-self._cum_now(cum, a, frac)), np.exp(-cum[:, -1])
        G = self.G[idx]
        posterior = (Sa - SH) / np.clip(Sa - SH * G, 1e-12, None)
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
            CBPE(), ATC(), DoC(), DelayAdjustedCBPE(), HazardAdjustedCBPE()]
