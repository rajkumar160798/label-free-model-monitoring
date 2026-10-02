"""Run the performance-estimation benchmark on the real streams.

    uv run python scripts/run_estimation.py            # all experiments
    uv run python scripts/run_estimation.py freddie_2007

Writes per-period records and a summary per experiment to results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from lfmm.harness import run_estimation
from lfmm.streams.acs import load_acs_income
from lfmm.streams.freddie import load_freddie

RESULTS = Path(__file__).resolve().parents[1] / "results"

EXPERIMENTS = {
    # Model built at the start of 2007 from 1999-2004 loans, then monitored
    # through the crisis, the recovery, COVID and the 2020s rate rise.
    "freddie_2007": dict(load=load_freddie, train_end="2007-01-01", freq="Q",
                         metrics=("roc_auc", "prevalence")),
    # Model built in 2016 from 2010-2013 loans; COVID is the main shock.
    "freddie_2016": dict(load=load_freddie, train_end="2016-01-01", train_start="2010-01-01",
                         freq="Q", metrics=("roc_auc", "prevalence")),
    # Trained on 2014 survey (labels a year late), monitored yearly to 2024.
    "acs_income_2016": dict(load=load_acs_income, train_end="2016-01-01", freq="Y",
                            metrics=("roc_auc", "accuracy", "prevalence")),
}


def main(names: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    for name in names or list(EXPERIMENTS):
        cfg = dict(EXPERIMENTS[name])
        stream = cfg.pop("load")()
        print(f"\n=== {name}: {len(stream):,} rows ===")
        res = run_estimation(stream, **cfg)
        print("fit:", res.info["fit_range"], f"n={res.info['n_fit']:,}",
              "| ref:", res.info["ref_range"], f"n={res.info['n_ref']:,}",
              "| ref metrics:", {k: round(v, 4) for k, v in res.info["ref_metrics"].items()})
        summary = res.summary()
        print(summary.round(4).to_string())
        res.records.to_csv(RESULTS / f"{name}_records.csv", index=False)
        summary.to_csv(RESULTS / f"{name}_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
