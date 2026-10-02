"""Run the degradation-detection benchmark on the real streams.

    uv run python scripts/run_detection.py            # all experiments
    uv run python scripts/run_detection.py freddie_2007

For each deployment, every detector is scored against each event definition
(metric + size of drop). Writes results/detection_<name>_<metric>_{records,summary}.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from lfmm.detectors import default_detectors
from lfmm.harness import deploy, run_detection
from lfmm.streams.acs import load_acs_income
from lfmm.streams.freddie import load_freddie

RESULTS = Path(__file__).resolve().parents[1] / "results"

# event metric -> (deltas scored by AUROC, delta used for alarms)
FREDDIE_EVENTS = {"roc_auc": ((0.02, 0.05, 0.08), 0.05),
                  "prevalence": ((0.005, 0.01, 0.02), 0.01)}

EXPERIMENTS = {
    "freddie_2007": dict(load=load_freddie, train_end="2007-01-01", freq="Q", events=FREDDIE_EVENTS),
    "freddie_2016": dict(load=load_freddie, train_end="2016-01-01", train_start="2010-01-01",
                         freq="Q", events=FREDDIE_EVENTS),
    "acs_income_2016": dict(load=load_acs_income, train_end="2016-01-01", freq="Y",
                            events={"accuracy": ((0.01, 0.03, 0.05), 0.03),
                                    "prevalence": ((0.02, 0.05, 0.1), 0.05)}),
}


def main(names: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    for name in names or list(EXPERIMENTS):
        cfg = dict(EXPERIMENTS[name])
        stream = cfg.pop("load")()
        events = cfg.pop("events")
        dep = deploy(stream, **cfg)
        for metric, (deltas, alarm_delta) in events.items():
            res = run_detection(stream, cfg["train_end"], default_detectors(metric, alarm_delta),
                                freq=cfg["freq"], event_metric=metric, deltas=deltas,
                                alarm_delta=alarm_delta, deployment=dep)
            ref = res.info["ref_metrics"][metric]
            print(f"\n=== {name} | event: {metric} worse than reference ({ref:.4f}) "
                  f"by > delta; alarms at {alarm_delta} ===")
            summary = res.summary()
            print(summary.round(3).to_string())
            res.records.to_csv(RESULTS / f"detection_{name}_{metric}_records.csv", index=False)
            summary.to_csv(RESULTS / f"detection_{name}_{metric}_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
