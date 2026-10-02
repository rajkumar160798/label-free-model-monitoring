import numpy as np
import pandas as pd
import pytest

from lfmm.detectors import PSI, DomainClassifier, EstimatorAlarm, ScoreKS, UnivariateTests, default_detectors
from lfmm.estimators import CBPE, History
from lfmm.harness import run_detection

from .test_core import synthetic_stream

EMPTY = History(np.array([]), np.array([]), np.array([], "datetime64[ns]"),
                np.array([], "datetime64[ns]"), np.datetime64("2020-01-01", "ns"))


def frame(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(shift, 1, n),
        "c": pd.Categorical(rng.choice(["a", "b"], n, p=[0.5 + shift / 4, 0.5 - shift / 4]),
                            categories=["a", "b"]),
    })


@pytest.mark.parametrize("det", [UnivariateTests(), PSI(), DomainClassifier()])
def test_input_detectors_quiet_without_drift_and_loud_with_it(det):
    ref = frame(4000, seed=1)
    det.fit(ref, np.full(len(ref), 0.5), np.zeros(len(ref)))
    s_same, a_same = det.score(frame(4000, seed=2), None, EMPTY)
    s_drift, a_drift = det.score(frame(4000, shift=1.0, seed=3), None, EMPTY)
    assert not a_same and a_drift
    assert s_drift > s_same


def test_psi_handles_missing_values():
    ref = frame(2000, seed=1)
    cur = frame(2000, seed=2)
    cur.loc[:999, "x"] = np.nan  # half missing is a big shift
    det = PSI().fit(ref, None, None)
    score, alarm = det.score(cur, None, EMPTY)
    assert alarm and score > 0.25


def test_score_ks():
    rng = np.random.default_rng(0)
    det = ScoreKS().fit(None, rng.random(3000), None)
    assert not det.score(None, rng.random(3000), EMPTY)[1]
    assert det.score(None, rng.random(3000) ** 3, EMPTY)[1]


def test_estimator_alarm_uses_reference_gap():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(float)
    det = EstimatorAlarm(CBPE(), "prevalence", delta=0.1).fit(None, p, y)
    assert not det.score(None, p, EMPTY)[1]
    score, alarm = det.score(None, p ** 4, EMPTY)  # far fewer likely positives
    assert alarm and score > 0.1


def test_run_detection_end_to_end():
    s = synthetic_stream(shift=3.0)
    res = run_detection(s, "2021-01-01", default_detectors("prevalence", 0.05), freq="Q",
                        event_metric="prevalence", deltas=(0.02, 0.05))
    summ = res.summary()
    assert {"univariate_tests", "psi_max", "cbpe:prevalence"} <= set(summ.index)
    # pure covariate shift: CBPE's estimated change should track the real one
    assert summ.loc["cbpe:prevalence", "spearman"] > 0.8
