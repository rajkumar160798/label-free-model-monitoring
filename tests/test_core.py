import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from lfmm.estimators import ATC, CBPE, LatestCompleteCohort, RecentArrivals, expected_auc
from lfmm.harness import run_estimation, trainable_mask
from lfmm.streams.base import Stream, with_label_delay


def synthetic_stream(n_months=48, per_month=2000, shift=1.5, seed=0, delay_days=180):
    """Covariate shift only: x drifts over time, P(y|x) is fixed."""
    rng = np.random.default_rng(seed)
    month = np.repeat(np.arange(n_months), per_month)
    x1 = rng.normal(shift * month / n_months, 1.0)
    x2 = rng.normal(0, 1, len(month))
    cat = pd.Categorical(rng.choice(["a", "b", "c"], len(month)))
    logit = -1.0 + 1.5 * x1 - 0.8 * x2 + 0.5 * (cat == "a")
    y = (rng.random(len(month)) < 1 / (1 + np.exp(-logit))).astype(float)
    event_time = (np.datetime64("2020-01-01", "M") + month).astype("datetime64[ns]")
    s = Stream("synthetic", pd.DataFrame({"x1": x1, "x2": x2, "cat": cat}), y, event_time, event_time)
    return with_label_delay(s, np.timedelta64(delay_days, "D"))


def test_expected_auc_matches_monte_carlo():
    rng = np.random.default_rng(1)
    scores = rng.random(3000)
    p = np.clip(scores + rng.normal(0, 0.1, 3000), 0.01, 0.99)
    draws = [roc_auc_score(rng.random(3000) < p, scores) for _ in range(200)]
    assert expected_auc(scores, p) == pytest.approx(np.mean(draws), abs=0.003)


def test_expected_auc_handles_ties():
    scores = np.array([0.1, 0.1, 0.9, 0.9])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    # one positive and one negative per tie group: AUC 0.5
    assert expected_auc(scores, p) == pytest.approx(0.5)


def test_stream_rejects_label_before_event():
    t = np.array(["2020-02-01"], dtype="datetime64[ns]")
    with pytest.raises(ValueError):
        Stream("bad", pd.DataFrame({"x": [1.0]}), [1.0], t, t - np.timedelta64(1, "D"))


def test_trainable_mask_excludes_partly_labeled_cohorts():
    s = synthetic_stream(n_months=12, per_month=100, delay_days=100)
    mask = trainable_mask(s, "2020-07-01", freq="M")
    # labels arrive 100 days later, so only cohorts up to ~March are complete by July
    assert s.event_time[mask].max() < np.datetime64("2020-04-01")
    assert s.known_by(np.datetime64("2020-07-01"))[mask].all()


def test_cbpe_tracks_truth_under_covariate_shift():
    s = synthetic_stream()
    res = run_estimation(s, "2021-01-01", freq="Q", metrics=("roc_auc", "accuracy", "prevalence"),
                         estimators=[CBPE(), ATC(), RecentArrivals(),
                                     LatestCompleteCohort(freq="M")])
    summ = res.summary()
    for m in ("roc_auc", "accuracy", "prevalence"):
        assert summ.loc[(m, "cbpe"), "mae"] < 0.02, summ
    # truth actually moves, so a constant guess would be wrong
    truth = res.wide("prevalence")["truth"]
    assert truth.max() - truth.min() > 0.1


def test_no_label_leaks_into_history():
    s = synthetic_stream(n_months=24, per_month=200, delay_days=365)
    seen = []

    class Spy(CBPE):
        name = "spy"

        def estimate(self, metric, proba, history):
            known = ~np.isnan(history.y_known)
            if known.any():
                seen.append(history.label_time[known].max() <= history.now)
            return super().estimate(metric, proba, history)

    run_estimation(s, "2021-03-01", freq="M", estimators=[Spy()])
    assert seen and all(seen)
