"""Run the performance-estimation benchmark on the real streams.

    uv run python scripts/run_estimation.py            # all experiments
    uv run python scripts/run_estimation.py freddie_2007 tlc_2019

Experiments are defined in src/lfmm/experiments.py. Writes per-period records
and a summary per experiment to results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from lfmm.experiments import EXPERIMENTS
from lfmm.harness import run_estimation

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(names: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    for name in names or list(EXPERIMENTS):
        exp = EXPERIMENTS[name]
        stream = exp.load()
        print(f"\n=== {name}: {len(stream):,} rows ===")
        res = run_estimation(stream, metrics=exp.metrics, **exp.deploy_kwargs())
        print("fit:", res.info["fit_range"], f"n={res.info['n_fit']:,}",
              "| ref:", res.info["ref_range"], f"n={res.info['n_ref']:,}",
              "| ref metrics:", {k: round(v, 4) for k, v in res.info["ref_metrics"].items()})
        summary = res.summary()
        print(summary.round(4).to_string())
        res.records.to_csv(RESULTS / f"{name}_records.csv", index=False)
        summary.to_csv(RESULTS / f"{name}_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
