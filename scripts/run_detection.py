"""Run the degradation-detection benchmark on the real streams.

    uv run python scripts/run_detection.py            # all experiments
    uv run python scripts/run_detection.py freddie_2007

For each experiment (src/lfmm/experiments.py), every detector is scored against
each event definition (metric + size of drop). Writes
results/detection_<name>_<metric>_{records,summary}.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from lfmm.detectors import detectors_for_events
from lfmm.experiments import EXPERIMENTS
from lfmm.harness import run_detection_events

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(names: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    for name in names or list(EXPERIMENTS):
        exp = EXPERIMENTS[name]
        stream = exp.load()
        results = run_detection_events(stream, detectors=detectors_for_events(exp.events),
                                       events=exp.events, **exp.deploy_kwargs())
        for metric, res in results.items():
            ref = res.info["ref_metrics"][metric]
            print(f"\n=== {name} | event: {metric} worse than reference ({ref:.4f}) "
                  f"by > delta; alarms at {res.alarm_delta} ===")
            summary = res.summary()
            print(summary.round(3).to_string())
            res.records.to_csv(RESULTS / f"detection_{name}_{metric}_records.csv", index=False)
            summary.to_csv(RESULTS / f"detection_{name}_{metric}_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
