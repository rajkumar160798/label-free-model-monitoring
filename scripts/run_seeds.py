"""Repeat the estimation tasks over several seeds.

    uv run python scripts/run_seeds.py                       # all experiments, seeds 0-4
    uv run python scripts/run_seeds.py freddie_2007 --seeds 3       # seeds 0-2
    uv run python scripts/run_seeds.py freddie_2007 --seeds 3,4     # just seeds 3 and 4

A seed changes the model's training randomness and training subsample, and for
sampled streams (ACS, BRFSS) which rows are drawn per year. For each experiment
and seed, runs next-batch estimation (and the in-flight book for natural-delay
streams). Writes results/seeds_<task>_<experiments>_s<seeds>.csv after every
seed (one row per experiment, metric, estimator, seed) and prints mean MAE and its standard deviation across seeds.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd

from lfmm.estimators import (CBPE, DelayAdjustedCBPE, HazardAdjustedCBPE, LabelsToDate,
                             LatestCompleteCohort, ReferencePerformance, default_estimators)
from lfmm.experiments import EXPERIMENTS
from lfmm.harness import deploy, run_estimation, run_inflight

RESULTS = Path(__file__).resolve().parents[1] / "results"
NATURAL_DELAY = {"freddie_2007", "freddie_2016"}


def load(exp, seed: int):
    fn = exp.load
    target = getattr(fn, "func", fn)
    return fn(seed=seed) if "seed" in inspect.signature(target).parameters else fn()


def _save(rows: list, task: str, tag: str) -> pd.DataFrame | None:
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(RESULTS / f"seeds_{task}_{tag}.csv", index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["experiment", "metric", "estimator"])["mae"].agg(["mean", "std", "count"])
    # how often each estimator is the best within an (experiment, metric, seed)
    rank = df.groupby(["experiment", "metric", "seed"])["mae"].rank(method="min")
    agg["win_rate"] = df.assign(win=rank == 1).groupby(["experiment", "metric", "estimator"])["win"].mean()
    return agg


def main(args: list[str]) -> None:
    seeds = list(range(5))
    if "--seeds" in args:  # --seeds 3 (seeds 0-2) or --seeds 2,3,4
        i = args.index("--seeds")
        val = args[i + 1]
        seeds = [int(x) for x in val.split(",")] if "," in val else list(range(int(val)))
        args = args[:i] + args[i + 2:]
    names = args or list(EXPERIMENTS)
    tag = "_".join(names) + "_s" + "-".join(map(str, seeds))
    RESULTS.mkdir(exist_ok=True)
    est_rows, book_rows = [], []
    for name in names:
        exp = EXPERIMENTS[name]
        for seed in seeds:
            stream = load(exp, seed)
            dep = deploy(stream, seed=seed, **exp.deploy_kwargs())
            ests = default_estimators(freq="M")
            res = run_estimation(stream, exp.train_end, exp.freq, metrics=exp.metrics,
                                 estimators=ests, deployment=dep)
            est_rows.append(res.summary().reset_index().assign(experiment=name, seed=seed))
            if name in NATURAL_DELAY:
                book = run_inflight(stream, exp.train_end, exp.freq, metrics=("prevalence", "roc_auc"),
                                    estimators=[ReferencePerformance(), LabelsToDate(),
                                                LatestCompleteCohort(freq="M"), CBPE(),
                                                DelayAdjustedCBPE(), HazardAdjustedCBPE()],
                                    deployment=dep)
                book_rows.append(book.summary().reset_index().assign(experiment=name, seed=seed))
            # save after every seed so a stopped job keeps what it finished
            _save(est_rows, "estimation", tag)
            _save(book_rows, "inflight", tag)
            print(f"{name} seed {seed} done", flush=True)

    pd.set_option("display.width", 200)
    for task, rows in [("estimation", est_rows), ("inflight", book_rows)]:
        if rows:
            print(f"\n=== {task}: MAE mean (std) over seeds ===")
            print(summarize(pd.concat(rows, ignore_index=True)).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
