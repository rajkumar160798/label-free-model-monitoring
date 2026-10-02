import itertools

import numpy as np
import pandas as pd
import pytest

from lfmm.detectors import ScoreKS
from lfmm.retrain import (Always, Every, ModelBank, Never, OnAlarm, build_bank, evaluate_policies,
                          oracle, simulate)

from .test_core import synthetic_stream


def toy_bank(L):
    T = L.shape[0]
    return ModelBank(stream=None, freq="M", periods=list(range(T)), models=[None] * T,
                     loss=L, loss_name="toy")


def brute_force(L, lam):
    T = L.shape[0]
    best = np.inf
    for switches in itertools.product([0, 1], repeat=T - 1):
        k, cost = 0, L[0, 0]
        for t in range(1, T):
            if switches[t - 1]:
                k, cost = t, cost + lam
            cost += L[k, t]
        best = min(best, cost)
    return best


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [0.0, 0.3, 2.0])
def test_oracle_matches_brute_force(seed, lam):
    rng = np.random.default_rng(seed)
    T = 7
    L = np.triu(rng.random((T, T)) * 2)
    L[np.tril_indices(T, -1)] = np.nan
    loss, starts = oracle(toy_bank(L), lam)
    assert loss + lam * len(starts) == pytest.approx(brute_force(L, lam))


def test_simulate_fixed_policies():
    T = 6
    L = np.triu(np.ones((T, T)))
    L[np.tril_indices(T, -1)] = np.nan
    bank = toy_bank(L)
    assert simulate(bank, Never()) == (6.0, [])
    assert simulate(bank, Always())[1] == [1, 2, 3, 4, 5]
    assert simulate(bank, Every(2))[1] == [2, 4]


def test_retraining_helps_under_drift_end_to_end():
    s = synthetic_stream(n_months=36, per_month=1500, shift=3.0, delay_days=40)
    bank = build_bank(s, "2020-07-01", "M", verbose=False)
    T = len(bank.periods)
    assert T >= 20
    # a model's loss on a period cannot be computed before it exists
    assert np.isnan(bank.loss[5, 4]) and not np.isnan(bank.loss[4, 5])
    res = evaluate_policies(bank, [Never(), Always(), Every(6), OnAlarm(ScoreKS)])
    free = res[res.rho == 0].set_index("policy")
    # with free retraining under drift, always beats never and nothing beats the oracle
    assert free.loc["always", "mean_loss"] < free.loc["never", "mean_loss"]
    assert (res["regret"] >= -1e-9).all()
