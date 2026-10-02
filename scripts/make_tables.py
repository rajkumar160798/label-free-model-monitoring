"""LaTeX tables for the paper from results/seeds_*.csv.

    uv run python scripts/make_tables.py

Writes paper/tables/estimation.tex and paper/tables/inflight.tex: MAE mean (std)
over seeds per experiment and metric; the best mean per row in bold.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"

COLUMNS = {
    "estimation": ["reference", "recent_arrivals", "latest_complete_cohort", "cbpe", "da_cbpe", "da_cbpe_hz"],
    "inflight": ["reference", "labels_to_date", "latest_complete_cohort", "cbpe", "da_cbpe", "da_cbpe_hz"],
}
HEADERS = {
    "reference": "Ref.", "recent_arrivals": "Recent", "labels_to_date": "To date",
    "latest_complete_cohort": "Complete", "cbpe": "CBPE", "da_cbpe": "DA-CBPE", "da_cbpe_hz": "DA-CBPE-HZ",
}
METRIC_NAMES = {"roc_auc": "AUC", "prevalence": "pos. rate", "accuracy": "accuracy"}


def load(task: str) -> pd.DataFrame:
    files = glob.glob(str(ROOT / "results" / f"seeds_{task}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no seed results for {task}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(["experiment", "metric", "estimator", "seed"], keep="last")


def table(task: str, caption: str, label: str) -> str:
    df = load(task)
    cols = COLUMNS[task]
    agg = df.groupby(["experiment", "metric", "estimator"])["mae"].agg(["mean", "std", "count"])
    n_seeds = int(agg["count"].max())
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption} MAE, mean (std) over {n_seeds} seeds; lower is better, best in bold.}}",
        rf"\label{{{label}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
        "Experiment & Metric & " + " & ".join(HEADERS[c] for c in cols) + r" \\", r"\midrule",
    ]
    for (exp, metric), g in agg.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        means = {c: g.loc[c, "mean"] for c in cols if c in g.index and pd.notna(g.loc[c, "mean"])}
        best = min(means.values()) if means else None
        cells = []
        for c in cols:
            if c not in means:
                cells.append("--")
                continue
            m, sd = means[c], g.loc[c, "std"]
            txt = f"{m:.4f} ({sd:.4f})" if pd.notna(sd) else f"{m:.4f}"
            cells.append(rf"\textbf{{{txt}}}" if m == best else txt)
        exp_tex = exp.replace("_", r"\_")
        lines.append(f"{exp_tex} & {METRIC_NAMES.get(metric, metric)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "estimation.tex").write_text(
        table("estimation", "Next-batch estimation.", "tab:estimation"), encoding="utf-8")
    try:
        (OUT / "inflight.tex").write_text(
            table("inflight", "In-flight book estimation (Freddie Mac, natural label delay).", "tab:inflight"),
            encoding="utf-8")
    except FileNotFoundError:
        print("no in-flight seed results yet")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
