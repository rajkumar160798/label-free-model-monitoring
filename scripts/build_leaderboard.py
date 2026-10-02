"""Compile results/ into one leaderboard per task.

    uv run python scripts/build_leaderboard.py

Within each experiment (and metric or event), methods are ranked; the
leaderboard reports each method's mean rank across experiments, how often it
was best, and how many experiments it took part in. Writes
leaderboard/LEADERBOARD.md and one CSV per task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lfmm.experiments import EXPERIMENTS

ROOT = Path(__file__).resolve().parents[1]
RESULTS, OUT = ROOT / "results", ROOT / "leaderboard"


def _family(name: str) -> str:
    """Merge per-metric variants of one method: 'cbpe:roc_auc' and 'cbpe:accuracy' -> 'cbpe alarm'."""
    if name.startswith("alarm:"):
        parts = name.split(":")
        return f"retrain on {parts[1]} alarm" if len(parts) > 2 else f"retrain on {parts[1]}"
    if ":" in name:
        return f"{name.split(':')[0]} alarm"
    return name


def _aggregate(scores: pd.DataFrame, method: str, lower_is_better: bool) -> pd.DataFrame:
    """scores: one row per (case, method) with a 'value' column."""
    scores = scores.dropna(subset=["value"]).copy()
    scores[method] = scores[method].map(_family)
    # one row per (case, method) after merging variants: keep the method's best variant
    best = (scores.groupby(["case", method])["value"].min() if lower_is_better
            else scores.groupby(["case", method])["value"].max())
    scores = best.reset_index()
    scores["rank"] = scores.groupby("case")["value"].rank(ascending=lower_is_better, method="min")
    scores["best"] = scores["rank"] == 1
    out = scores.groupby(method).agg(mean_rank=("rank", "mean"), wins=("best", "sum"),
                                     cases=("case", "nunique"), mean_value=("value", "mean"))
    # Methods evaluated on every case are ranked first: a method should not top the board by
    # being scored on fewer, easier cases. Partial ones (e.g. accuracy-only estimators,
    # submissions still being run) follow, ordered the same way.
    out["full_coverage"] = out["cases"] == scores["case"].nunique()
    return out.sort_values(["full_coverage", "mean_rank", "wins"], ascending=[False, True, False])


def submissions() -> list[dict]:
    """meta.json of every submitted method under results/submissions/."""
    out = []
    for meta in sorted((RESULTS / "submissions").glob("*/meta.json")):
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["dir"] = meta.parent
        out.append(m)
    return out


def _estimation_files(kind: str):
    """(experiment, summary csv) for baselines and submissions; kind = estimation | inflight."""
    for name in EXPERIMENTS:
        base = RESULTS / (f"{name}_summary.csv" if kind == "estimation" else f"inflight_{name}_summary.csv")
        if base.exists():
            yield name, base
        for sub in submissions():
            f = sub["dir"] / f"{kind}_{name}_summary.csv"
            if f.exists():
                yield name, f


def case_rows(task: str) -> pd.DataFrame:
    """One row per (case, method) with the score used for ranking."""
    rows = []
    if task in ("estimation", "inflight"):
        for name, f in _estimation_files(task):
            d = pd.read_csv(f)
            rows += [{"case": f"{name}/{r.metric}", "method": r.estimator, "value": r.mae}
                     for r in d.itertuples()]
    elif task == "detection":
        dirs = [RESULTS] + [sub["dir"] for sub in submissions()]
        for name, exp in EXPERIMENTS.items():
            for metric in exp.events:
                for base in dirs:
                    f = base / f"detection_{name}_{metric}_summary.csv"
                    if f.exists():
                        d = pd.read_csv(f)
                        # A constant drift score has an undefined correlation; it carries no
                        # information about degradation, so it counts as 0 rather than a gap.
                        rows += [{"case": f"{name}/{metric}", "method": r.detector,
                                  "value": 0.0 if pd.isna(r.spearman) else r.spearman}
                                 for r in d.itertuples()]
    return pd.DataFrame(rows).drop_duplicates(["case", "method"], keep="last")


def estimation() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (_aggregate(case_rows("estimation"), "method", True),
            _aggregate(case_rows("inflight"), "method", True))


def detection() -> pd.DataFrame:
    return _aggregate(case_rows("detection"), "method", False)


def retrain_cases(rho: float = 1.0) -> pd.DataFrame:
    rows = []
    for name in EXPERIMENTS:
        f = RESULTS / f"retrain_{name}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d["rho"] == rho) & (d["policy"] != "oracle")]
        d = d.assign(policy=d["policy"].where(~d["policy"].str.startswith("every_"), "every_k (best k)"))
        d = d.groupby("policy", as_index=False)["regret"].min()
        rows += [{"case": name, "method": r.policy, "value": r.regret} for r in d.itertuples()]
    return pd.DataFrame(rows)


def retrain(rho: float = 1.0) -> pd.DataFrame:
    rows = []
    for name in EXPERIMENTS:
        f = RESULTS / f"retrain_{name}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d["rho"] == rho) & (d["policy"] != "oracle")]
        # Fixed schedules differ in period length across experiments; group them.
        d = d.assign(policy=d["policy"].where(~d["policy"].str.startswith("every_"), "every_k (best k)"))
        d = d.groupby("policy", as_index=False)["regret"].min()
        rows += [{"case": name, "method": r.policy, "value": r.regret} for r in d.itertuples()]
    return _aggregate(pd.DataFrame(rows), "method", True)


def _md(df: pd.DataFrame, value_label: str) -> str:
    df = df.rename(columns={"mean_value": value_label}).round(4).reset_index()
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in df.itertuples(index=False))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    est, inflight = estimation()
    det, ret = detection(), retrain()
    for name, df in [("estimation", est), ("inflight", inflight), ("detection", det), ("retrain", ret)]:
        df.to_csv(OUT / f"{name}.csv")
    # per-case scores, for the website's detail views
    for task in ("estimation", "inflight", "detection"):
        case_rows(task).to_csv(OUT / f"cases_{task}.csv", index=False)
    retrain_cases().to_csv(OUT / "cases_retrain.csv", index=False)
    subs = [{k: v for k, v in m.items() if k != "dir"} for m in submissions()]
    (OUT / "submissions.json").write_text(json.dumps(subs, indent=2), encoding="utf-8")
    md = [
        "# Leaderboard",
        "",
        "Generated by `scripts/build_leaderboard.py` from `results/`. Within each experiment "
        "(and metric or event) methods are ranked; `mean_rank` averages those ranks, "
        "`wins` counts first places, `cases` is how many rankings the method took part in. "
        "Single runs: see PROGRESS.md for bootstrap intervals before reading small differences.",
        "",
        "## Performance estimation: next batch (rank by MAE)", "", _md(est, "mean_mae"),
        "## Performance estimation: in-flight book (rank by MAE)", "", _md(inflight, "mean_mae"),
        "## Degradation detection (rank by Spearman correlation with true degradation)", "",
        _md(det, "mean_spearman"),
        "## Retrain or not (rank by regret vs hindsight oracle, retrain cost rho = 1)", "",
        "Fixed schedules are grouped as `every_k (best k)`, which flatters them: k is picked in hindsight.",
        "", _md(ret, "mean_regret"),
    ]
    (OUT / "LEADERBOARD.md").write_text("\n".join(md), encoding="utf-8")
    print((OUT / "LEADERBOARD.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
