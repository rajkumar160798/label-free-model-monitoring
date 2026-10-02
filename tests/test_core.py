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


def natural_delay_stream(n_months=60, per_month=3000, shock_month=36, shock=1.5, horizon_days=720, seed=0):
    """Concept shift with credit-style label arrival.

    At ``shock_month`` the intercept jumps (same features, many more positives).
    Positives become known when the event happens (uniform over the horizon);
    negatives only once the horizon has passed.
    """
    rng = np.random.default_rng(seed)
    month = np.repeat(np.arange(n_months), per_month)
    x = rng.normal(0, 1, len(month))
    logit = -3.0 + 1.2 * x + shock * (month >= shock_month)
    y = (rng.random(len(month)) < 1 / (1 + np.exp(-logit))).astype(float)
    event_time = (np.datetime64("2010-01-01", "M") + month).astype("datetime64[ns]")
    delay_days = np.where(y == 1, rng.integers(30, horizon_days, len(month)), horizon_days)
    label_time = event_time + delay_days.astype("timedelta64[D]")
    return Stream("natural", pd.DataFrame({"x": x}), y, event_time, label_time)


@pytest.mark.slow
def test_delay_adjusted_cbpe_sees_concept_shift_before_cohorts_complete():
    from lfmm.estimators import CBPE, DelayAdjustedCBPE, LatestCompleteCohort

    s = natural_delay_stream()
    res = run_estimation(s, "2012-07-01", freq="Q", metrics=("prevalence",),
                         estimators=[CBPE(), DelayAdjustedCBPE(), LatestCompleteCohort(freq="M")])
    w = res.wide("prevalence")
    after = w.loc[[p for p in w.index if p >= "2013Q2"]]  # shock hit at 2013-01; ~1 quarter later
    err = (after[["cbpe", "da_cbpe", "latest_complete_cohort"]].sub(after["truth"], axis=0)).abs().mean()
    # CBPE cannot see a pure concept shift; the complete-cohort baseline is ~2 years late
    assert err["da_cbpe"] < 0.5 * err["cbpe"], err
    assert err["da_cbpe"] < 0.5 * err["latest_complete_cohort"], err


@pytest.mark.slow
def test_inflight_posterior_beats_ignoring_or_trusting_early_labels():
    from lfmm.estimators import CBPE, DelayAdjustedCBPE, LabelsToDate
    from lfmm.harness import run_inflight

    s = natural_delay_stream()
    res = run_inflight(s, "2012-07-01", freq="Q", book_months=24, metrics=("prevalence",),
                       estimators=[CBPE(), DelayAdjustedCBPE(), LabelsToDate()])
    summ = res.summary().loc["prevalence"]
    assert summ.loc["da_cbpe", "mae"] < 0.5 * summ.loc["cbpe", "mae"], summ
    assert summ.loc["da_cbpe", "mae"] < 0.5 * summ.loc["labels_to_date", "mae"], summ
    assert summ.loc["labels_to_date", "bias"] < 0  # counting pending labels as negative


def calendar_shock_stream(n_months=72, per_month=3000, burst=(40, 44), burst_hazard=0.03, horizon=24, seed=0):
    """Defaults by month with a calendar-time burst that hits every active loan.

    Base: each loan has an eventual-default probability p (logistic in x) and, if it
    defaults, does so at a uniform month of age 1..horizon. During calendar months in
    ``burst`` every loan still active and not yet defaulted also defaults with extra
    probability ``burst_hazard`` per month. Labels: positive known at the default
    month, negative at the horizon.
    """
    rng = np.random.default_rng(seed)
    start = np.repeat(np.arange(n_months), per_month)
    x = rng.normal(0, 1, len(start))
    p = 1 / (1 + np.exp(-(-3.5 + 1.2 * x)))
    will = rng.random(len(start)) < p
    t_def = np.where(will, start + rng.integers(1, horizon + 1, len(start)), 10**6)
    for m in range(*burst):  # burst month m: active, not-yet-defaulted loans may default now
        active = (start < m) & (start + horizon >= m) & (t_def > m)
        hit = active & (rng.random(len(start)) < burst_hazard)
        t_def[hit] = m
    y = (t_def <= start + horizon).astype(float)
    event_time = (np.datetime64("2010-01-01", "M") + start).astype("datetime64[ns]")
    label_month = np.where(y == 1, t_def, start + horizon)
    label_time = (np.datetime64("2010-01-01", "M") + label_month).astype("datetime64[ns]")
    return Stream("calendar", pd.DataFrame({"x": x}), y, event_time, label_time)


@pytest.mark.slow
@pytest.mark.xfail(reason="known limitation: a calendar-time burst of extra defaults (e.g. COVID "
                          "forbearance) is read as a lasting shift; needs an age-period hazard model",
                   strict=True)
def test_delay_adjusted_cbpe_does_not_overshoot_after_calendar_burst():
    from lfmm.estimators import DelayAdjustedCBPE, LatestCompleteCohort

    s = calendar_shock_stream()
    res = run_estimation(s, "2012-07-01", freq="Q", metrics=("prevalence",),
                         estimators=[DelayAdjustedCBPE(), LatestCompleteCohort(freq="M")])
    err = res.summary().loc["prevalence", "mae"]
    assert err["da_cbpe"] <= err["latest_complete_cohort"], err


@pytest.mark.slow
def test_hazard_cbpe_handles_calendar_burst():
    from lfmm.estimators import HazardAdjustedCBPE, LatestCompleteCohort

    s = calendar_shock_stream()
    res = run_estimation(s, "2012-07-01", freq="Q", metrics=("prevalence",),
                         estimators=[HazardAdjustedCBPE(), LatestCompleteCohort(freq="M")])
    err = res.summary().loc["prevalence", "mae"]
    assert err["da_cbpe_hz"] <= err["latest_complete_cohort"], err


@pytest.mark.slow
def test_hazard_cbpe_keeps_lasting_shift_accuracy():
    from lfmm.estimators import DelayAdjustedCBPE, HazardAdjustedCBPE

    s = natural_delay_stream()
    res = run_estimation(s, "2012-07-01", freq="Q", metrics=("prevalence",),
                         estimators=[DelayAdjustedCBPE(), HazardAdjustedCBPE()])
    err = res.summary().loc["prevalence", "mae"]
    assert err["da_cbpe_hz"] <= 1.3 * err["da_cbpe"], err


@pytest.mark.slow
def test_hazard_cbpe_without_calendar_terms_matches_delay_adjusted():
    """With no kappa the hazard likelihood is the delay-adjusted one: same offsets."""
    from lfmm.estimators import DelayAdjustedCBPE, HazardAdjustedCBPE

    s = natural_delay_stream(n_months=48)
    deltas = {"da": [], "hz": []}

    class DA(DelayAdjustedCBPE):
        def _fit(self, h):
            super()._fit(h)
            deltas["da"].append(self.last_delta)

    class HZ(HazardAdjustedCBPE):
        def __init__(self):
            # same window as da_cbpe, one cohort group, no calendar shocks
            super().__init__(calendar_months=0, window_months=12, new_cohort_months=12, max_rows=50_000)

        def _fit(self, h):
            super()._fit(h)
            deltas["hz"].append(self.last_delta)

    run_estimation(s, "2012-07-01", freq="Q", metrics=("prevalence",), estimators=[DA(), HZ()])
    assert np.allclose(deltas["da"], deltas["hz"], atol=0.05), deltas
