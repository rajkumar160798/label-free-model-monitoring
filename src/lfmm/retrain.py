"""Task 3: retrain or not.

At the end of each deployment period a policy decides whether to retrain. A
retrain at the end of period t can only use cohorts whose labels have (almost)
all arrived by then, and serves from period t+1. Cost of a policy:

    sum over periods of the serving model's loss  +  lambda * number of retrains

All candidate models are precomputed in a ``ModelBank``: model k is trained at
the start of period k (model 0 is the initial model), and ``loss[k, t]`` is its
loss on period t. Any policy, and the hindsight-optimal schedule (``oracle``,
by dynamic programming), is then evaluated from lookups.

Retrain cost is reported relative to the gain per retrain of retraining every
period: ``rho = lambda / gain``. At rho = 1, "always retrain" and "never
retrain" cost the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .detectors import Detector
from .estimators import metric_value
from .harness import Deployment, deploy
from .streams.base import Stream


def period_loss(name: str, y: np.ndarray, p: np.ndarray) -> float:
    if name == "logloss":
        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    if name == "auc":
        return 1.0 - metric_value("roc_auc", y, p)
    raise ValueError(name)


@dataclass
class ModelBank:
    stream: Stream
    freq: str
    periods: list[pd.Period]
    models: list[Deployment]            # models[k] serves from periods[k]
    loss: np.ndarray                    # [k, t], NaN where t < k
    loss_name: str
    masks: list[np.ndarray] = field(repr=False, default_factory=list)


def build_bank(stream: Stream, train_end, freq: str, loss: str = "logloss",
               min_truth_coverage: float = 0.9, retrain_window_months: int | None = 60,
               max_train_rows: int = 300_000, seed: int = 0, verbose: bool = True,
               **initial_deploy_kwargs) -> ModelBank:
    """Train one candidate model per decision point and score each on every later period."""
    m0 = deploy(stream, train_end, freq, max_train_rows=max_train_rows, seed=seed, **initial_deploy_kwargs)

    # Deployment periods up to the label frontier (where truth is still complete).
    all_periods = stream.periods(freq)
    periods, masks = [], []
    for p in m0.periods():
        mask = np.asarray(all_periods == p)
        if (~np.isnan(stream.y[mask])).mean() < min_truth_coverage:
            break
        periods.append(p)
        masks.append(mask)
    T = len(periods)

    models = [m0]
    for k in range(1, T):
        start = periods[k].start_time
        try:
            m = deploy(stream, start, freq, train_window_months=retrain_window_months,
                       max_train_rows=max_train_rows, seed=seed)
        except ValueError:  # no fully labeled cohorts yet
            m = models[-1]
        # No new labeled cohorts since the last model: retraining changes nothing.
        prev = models[-1]
        if stream.event_time[m.fit_idx].max() <= stream.event_time[prev.fit_idx].max():
            m = prev
        models.append(m)
        if verbose and k % 10 == 0:
            print(f"    trained {k}/{T - 1} candidate models")

    L = np.full((T, T), np.nan)
    for k, m in enumerate(models):
        for t in range(k, T):
            y = stream.y[masks[t]]
            ok = ~np.isnan(y)
            L[k, t] = period_loss(loss, y[ok], m.proba[masks[t]][ok])
    return ModelBank(stream, freq, periods, models, L, loss, masks)


# --- policies -----------------------------------------------------------------

class Policy:
    name = "policy"

    def reset(self, bank: ModelBank) -> None:
        pass

    def retrain(self, bank: ModelBank, t: int, k: int) -> bool:
        """After serving period t with model k: retrain now (model t+1 serves next)?"""
        raise NotImplementedError


class Never(Policy):
    name = "never"

    def retrain(self, bank, t, k):
        return False


class Always(Policy):
    name = "always"

    def retrain(self, bank, t, k):
        return True


class Every(Policy):
    def __init__(self, n: int):
        self.n = n
        self.name = f"every_{n}"

    def retrain(self, bank, t, k):
        return t - k + 1 >= self.n


class OnAlarm(Policy):
    """Retrain when a detector, fit on the serving model's reference window, alarms."""

    def __init__(self, make_detector: Callable[[], Detector], name: str | None = None):
        self.make_detector = make_detector
        self.name = f"alarm:{name or make_detector().name}"

    def reset(self, bank):
        self.detectors: dict[int, Detector] = {}

    def retrain(self, bank, t, k):
        m = bank.models[k]
        if k not in self.detectors:
            s = bank.stream
            det = self.make_detector()
            det.fit(s.X.iloc[m.ref_idx].reset_index(drop=True), m.proba[m.ref_idx], s.y[m.ref_idx], m.threshold)
            self.detectors[k] = det
        batch = m.batch(bank.periods[t])
        X_b = bank.stream.X.loc[batch.mask].reset_index(drop=True)
        _, alarm = self.detectors[k].score(X_b, m.proba[batch.mask], batch.history)
        return bool(alarm)


# --- evaluation -----------------------------------------------------------------

def simulate(bank: ModelBank, policy: Policy) -> tuple[float, list[int]]:
    """Total loss over all periods and the periods at which new models start serving."""
    policy.reset(bank)
    k, total, starts = 0, 0.0, []
    T = len(bank.periods)
    for t in range(T):
        total += bank.loss[k, t]
        if t < T - 1 and policy.retrain(bank, t, k):
            k = t + 1
            starts.append(k)
    return total, starts


def oracle(bank: ModelBank, lam: float) -> tuple[float, list[int]]:
    """Hindsight-optimal schedule by dynamic programming over the serving model."""
    L, T = bank.loss, len(bank.periods)
    best = np.full(T, np.inf)          # best[k]: min cost so far with model k serving
    back = [dict() for _ in range(T)]  # back[t][k] = model serving at t-1
    best[0] = L[0, 0]
    for t in range(1, T):
        new = np.full(T, np.inf)
        for k in range(t):             # keep serving model k
            if np.isfinite(best[k]):
                new[k] = best[k] + L[k, t]
                back[t][k] = k
        j = int(np.argmin(best))       # switch to model t
        new[t] = best[j] + lam + L[t, t]
        back[t][t] = j
        best = new
    k = int(np.argmin(best))
    cost = float(best[k])
    starts = []
    for t in range(T - 1, 0, -1):
        prev = back[t][k]
        if prev != k:
            starts.append(t)
        k = prev
    retrains = len(starts)
    return cost - lam * retrains, sorted(starts)  # loss part only; caller adds lam


def evaluate_policies(bank: ModelBank, policies: list[Policy], rhos=(0.0, 0.25, 1.0, 4.0)) -> pd.DataFrame:
    T = len(bank.periods)
    runs = {p.name: simulate(bank, p) for p in policies}
    never_loss, _ = runs.get("never", simulate(bank, Never()))
    always_loss, always_starts = runs.get("always", simulate(bank, Always()))
    gain = (never_loss - always_loss) / max(1, len(always_starts))
    retraining_helps = gain > 0
    if not retraining_helps:  # fall back to 1% of a period's loss as the cost unit
        gain = 0.01 * never_loss / T

    rows = []
    for rho in rhos:
        lam = rho * gain
        o_loss, o_starts = oracle(bank, lam)
        o_cost = o_loss + lam * len(o_starts)
        rows.append({"rho": rho, "policy": "oracle", "retrains": len(o_starts),
                     "mean_loss": o_loss / T, "cost": o_cost, "regret": 0.0})
        for name, (loss, starts) in runs.items():
            cost = loss + lam * len(starts)
            rows.append({"rho": rho, "policy": name, "retrains": len(starts),
                         "mean_loss": loss / T, "cost": cost, "regret": cost - o_cost})
    df = pd.DataFrame(rows)
    df.attrs.update(periods=T, gain_per_retrain=gain, retraining_helps=retraining_helps, loss=bank.loss_name)
    return df
