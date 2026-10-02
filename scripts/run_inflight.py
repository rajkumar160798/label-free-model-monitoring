"""Run the in-flight book estimation task (natural label delay streams only).

    uv run python scripts/run_inflight.py            # freddie_2007, freddie_2016
    uv run python scripts/run_inflight.py freddie_2007

At each quarter end, estimate the default rate and AUC of all loans started in
the previous 24 months, some of whose labels have already arrived. Writes
results/inflight_<name>_{records,summary}.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from lfmm.estimators import (CBPE, DelayAdjustedCBPE, HazardAdjustedCBPE, LabelsToDate,
                             LatestCompleteCohort, ReferencePerformance)
from lfmm.experiments import EXPERIMENTS
from lfmm.harness import run_inflight

RESULTS = Path(__file__).resolve().parents[1] / "results"
NATURAL_DELAY = ["freddie_2007", "freddie_2016"]


def main(names: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    for name in names or NATURAL_DELAY:
        exp = EXPERIMENTS[name]
        stream = exp.load()
        res = run_inflight(stream, metrics=("prevalence", "roc_auc"), book_months=24,
                           estimators=[ReferencePerformance(), LabelsToDate(), LatestCompleteCohort(freq="M"),
                                       CBPE(), DelayAdjustedCBPE(), HazardAdjustedCBPE()],
                           **exp.deploy_kwargs())
        print(f"\n=== {name}: in-flight book (last 24 months of loans) ===")
        summary = res.summary()
        print(summary.round(4).to_string())
        res.records.to_csv(RESULTS / f"inflight_{name}_records.csv", index=False)
        summary.to_csv(RESULTS / f"inflight_{name}_summary.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
