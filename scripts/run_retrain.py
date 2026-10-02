"""Run the retrain-or-not benchmark on the real streams.

    uv run python scripts/run_retrain.py            # all experiments (slow: trains one model per period)
    uv run python scripts/run_retrain.py tlc_2019

For each experiment (src/lfmm/experiments.py), builds the model bank, evaluates
fixed schedules, alarm-triggered retraining and the hindsight oracle, and writes
results/retrain_<name>.csv. ``rho`` is the retrain cost relative to the average
gain per retrain of retraining every period.
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd

from lfmm.detectors import PSI, DomainClassifier, EstimatorAlarm, ScoreKS
from lfmm.estimators import CBPE, LatestCompleteCohort, RecentArrivals
from lfmm.experiments import EXPERIMENTS
from lfmm.retrain import Always, Every, Never, OnAlarm, build_bank, evaluate_policies

RESULTS = Path(__file__).resolve().parents[1] / "results"
EVERY = {"W": (2, 4), "M": (3, 6, 12), "Q": (2, 4, 8), "Y": (2, 3)}


def policies_for(exp) -> list:
    metric = "roc_auc" if "roc_auc" in exp.events else next(iter(exp.events))
    delta = exp.events[metric][1]
    alarm = lambda est: partial(EstimatorAlarm, est(), metric, delta)  # noqa: E731
    return [
        Never(), Always(), *[Every(n) for n in EVERY[exp.freq]],
        OnAlarm(PSI), OnAlarm(ScoreKS), OnAlarm(DomainClassifier),
        OnAlarm(alarm(CBPE), f"cbpe:{metric}"),
        OnAlarm(alarm(lambda: LatestCompleteCohort(freq="M")), f"latest_complete_cohort:{metric}"),
        OnAlarm(alarm(RecentArrivals), f"recent_arrivals:{metric}"),
    ]


def main(names: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    for name in names or list(EXPERIMENTS):
        exp = EXPERIMENTS[name]
        t0 = time.time()
        stream = exp.load()
        print(f"\n=== {name}: building model bank ===")
        bank = build_bank(stream, **exp.deploy_kwargs())
        res = evaluate_policies(bank, policies_for(exp))
        unit = (f"gain per retrain={res.attrs['gain_per_retrain']:.5f}" if res.attrs["retraining_helps"]
                else "retraining every period is WORSE than never; rho unit = 1% of a period's loss")
        print(f"{res.attrs['periods']} periods, loss={res.attrs['loss']}, {unit}, {time.time() - t0:.0f}s")
        table = res.pivot(index="policy", columns="rho", values="regret")
        table.insert(0, "retrains", res[res.rho == 0].set_index("policy")["retrains"])
        table.insert(1, "mean_loss", res[res.rho == 0].set_index("policy")["mean_loss"])
        print("regret vs hindsight oracle (lower is better), by relative retrain cost rho:")
        print(table.sort_values(1.0).round(4).to_string())
        res.to_csv(RESULTS / f"retrain_{name}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
