"""Paper figures from results/*.csv.

    uv run python scripts/make_figures.py

Writes figures/*.png (and .pdf) for the Freddie Mac story: estimate vs truth
for the next batch and for the in-flight book.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"

# Reference palette (validated light-mode categorical slots 1-3) and chart ink.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {
    "truth": dict(label="Truth (hidden labels)", color=INK, lw=2.4, ls="-"),
    "cbpe": dict(label="CBPE (label-free)", color="#2a78d6", lw=2, ls=(0, (6, 2))),
    "latest_complete_cohort": dict(label="Latest complete cohort", color="#eb6834", lw=2, ls=(0, (2, 2))),
    "da_cbpe": dict(label="Delay-adjusted CBPE (ours)", color="#1baf7a", lw=2, ls="-"),
}
EVENTS = [("2007Q3", "housing crisis"), ("2020Q1", "COVID")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def timeline(ax, records: pd.DataFrame, metric: str, title: str, pct: bool):
    r = records[records["metric"] == metric]
    w = r.pivot_table(index="period", columns="estimator", values="estimate")
    w["truth"] = r.drop_duplicates("period").set_index("period")["truth"]
    w = w.dropna(subset=["truth"])
    x = pd.PeriodIndex(w.index, freq="Q").to_timestamp()
    scale = 100 if pct else 1
    for key, spec in SERIES.items():
        if key not in w:
            continue
        ax.plot(x, w[key] * scale, color=spec["color"], lw=spec["lw"], ls=spec["ls"],
                label=spec["label"], solid_capstyle="round", zorder=3 if key == "truth" else 2)
    for q, name in EVENTS:
        t = pd.Period(q, "Q").to_timestamp()
        if x.min() <= t <= x.max():
            ax.axvline(t, color=AXIS, lw=1, zorder=1)
            ax.text(t, 1.0, f" {name}", transform=ax.get_xaxis_transform(), color=MUTED,
                    fontsize=8.5, va="top")
    _style(ax)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_ylabel("default rate (%)" if pct else metric.replace("_", " ").upper(), color=INK_2, fontsize=9)
    ax.set_ylim(bottom=0 if pct else None)


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})

    batch = pd.read_csv(RESULTS / "freddie_2007_records.csv")
    book = pd.read_csv(RESULTS / "inflight_freddie_2007_records.csv")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    timeline(axes[0], batch, "prevalence",
             "New batch: eventual 24-month default rate of loans starting this quarter", pct=True)
    timeline(axes[1], book, "prevalence",
             "In-flight book: default rate of all loans from the last 24 months", pct=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Freddie Mac, model built Jan 2007: estimated vs true default rate",
                 x=0.01, ha="left", y=1.045, color=INK, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"freddie_2007_default_rate.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    timeline(ax, batch, "roc_auc", "New batch: AUC of the frozen model", pct=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=9,
              labelcolor=INK_2, ncol=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"freddie_2007_auc.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", sorted(p.name for p in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
