"""Appendix tables for the paper from results/.

    uv run python scripts/make_appendix.py

Writes paper/tables/{detection_spearman,detection_alarms,retrain,inflight_ci,accuracy}.tex.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from lfmm.experiments import EXPERIMENTS
from lfmm.harness import Result

ROOT = Path(__file__).resolve().parents[1]
RESULTS, OUT = ROOT / "results", ROOT / "paper" / "tables"

SHORT = {
    "freddie_2007": "FM07", "freddie_2016": "FM16", "acs_income_2016": "ACS", "brfss_2014": "BRFSS",
    "tlc_2019": "Taxi", "tabred_homecredit": "HCred", "tabred_homesite": "HSite", "tabred_ecom": "Ecom",
}
DETECTORS = [
    ("da_cbpe", "DA-CBPE alarm (ours)"), ("cbpe", "CBPE alarm"),
    ("latest_complete_cohort", "Complete-cohort alarm"), ("recent_arrivals", "Recent-arrivals alarm"),
    ("adwin_scores", "ADWIN (scores)"), ("adwin_errors", "ADWIN (arrived errors)"),
    ("score_ks", "KS (scores)"), ("univariate_tests", "Univariate KS/$\\chi^2$"),
    ("psi_max", "PSI (max)"), ("mmd", "MMD"), ("domain_classifier", "Domain classifier"),
]


def tex(s: str) -> str:
    return s.replace("_", r"\_")


def wrap(body: list[str], caption: str, label: str, colspec: str) -> str:
    return "\n".join([
        r"\begin{table}[h]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\scriptsize", r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        *body, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def detection_frame() -> pd.DataFrame:
    rows = []
    for name, exp in EXPERIMENTS.items():
        for metric, (_, alarm_delta) in exp.events.items():
            f = RESULTS / f"detection_{name}_{metric}_summary.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d["family"] = d["detector"].str.split(":").str[0]
            for r in d.itertuples():
                rows.append({"exp": name, "event": "perf" if metric != "prevalence" else "rate",
                             "family": r.family, "spearman": r.spearman, "precision": r.precision,
                             "recall": r.recall, "alarm_rate": r.alarm_rate, "event_rate": r.event_rate})
    return pd.DataFrame(rows)


def detection_tables() -> None:
    df = detection_frame()
    exps = list(EXPERIMENTS)
    # Spearman matrix: rows detectors, columns experiment x event
    head = "Detector & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{SHORT[e]}}}" for e in exps) + r" \\"
    sub = " & " + " & ".join(["perf & rate"] * len(exps)) + r" \\"
    body = [head, sub, r"\midrule"]
    for fam, label in DETECTORS:
        cells = []
        for e in exps:
            for ev in ("perf", "rate"):
                v = df[(df.exp == e) & (df.event == ev) & (df.family == fam)]["spearman"]
                cells.append("--" if v.empty or pd.isna(v.iloc[0]) else f"{v.iloc[0]:.2f}")
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    (OUT / "detection_spearman.tex").write_text(wrap(
        body, "Degradation detection, per experiment: Spearman correlation between the detector's drift "
              "score and the true degradation (higher is better). \\emph{perf}: AUC drop (accuracy for ACS); "
              "\\emph{rate}: change in positive rate. -- = score constant or undefined.",
        "tab:det-spearman", "l" + "r" * (2 * len(exps))), encoding="utf-8")

    # alarm behaviour: median over the 16 cases plus share of cases alarming every period
    agg = df.groupby("family").agg(
        alarm=("alarm_rate", "median"), always=("alarm_rate", lambda a: (a >= 0.999).mean()),
        precision=("precision", "median"), recall=("recall", "median"))
    body = [r"Detector & Median alarm rate & Cases alarming every period & Median precision & Median recall \\",
            r"\midrule"]
    for fam, label in DETECTORS:
        r = agg.loc[fam]
        body.append(f"{label} & {r.alarm:.2f} & {r.always:.0%} & {r.precision:.2f} & {r.recall:.2f} \\\\"
                    .replace("%", r"\%"))
    (OUT / "detection_alarms.tex").write_text(wrap(
        body, "Alarm behaviour of each detector over the 16 detection cases.", "tab:det-alarms", "lrrrr"),
        encoding="utf-8")


def retrain_table() -> None:
    policies = [("oracle", "Hindsight oracle"), ("never", "Never"), ("always", "Always"),
                ("every_k", "Every $k$ (best $k$)"), ("alarm:latest_complete_cohort", "Alarm: complete cohort"),
                ("alarm:recent_arrivals", "Alarm: recent arrivals"), ("alarm:cbpe", "Alarm: CBPE"),
                ("alarm:psi_max", "Alarm: PSI"), ("alarm:score_ks", "Alarm: KS (scores)"),
                ("alarm:domain_classifier", "Alarm: domain classifier")]
    exps = list(EXPERIMENTS)
    frames = {}
    for e in exps:
        f = RESULTS / f"retrain_{e}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d[d["rho"] == 1.0].copy()
            d["key"] = np.where(d["policy"].str.startswith("every_"), "every_k",
                                d["policy"].str.rsplit(":", n=1).str[0].where(d["policy"].str.count(":") == 2,
                                                                               d["policy"]))
            # for grouped schedules, report the one with the lowest regret (and its retrains)
            frames[e] = d.loc[d.groupby("key")["regret"].idxmin()].set_index("key")[["regret", "retrains"]]
    body = ["Policy & " + " & ".join(SHORT[e] for e in exps) + r" \\", r"\midrule"]
    for key, label in policies:
        cells = []
        for e in exps:
            fr = frames.get(e)
            if fr is None or key not in fr.index:
                cells.append("--")
                continue
            r = fr.loc[key]
            best = fr.drop(index="oracle", errors="ignore")["regret"].min()
            txt = f"{r.regret:.3f} ({int(r.retrains)})"
            cells.append(rf"\textbf{{{txt}}}" if key != "oracle" and np.isclose(r.regret, best) else txt)
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    (OUT / "retrain.tex").write_text(wrap(
        body, "Retrain or not: regret against the hindsight-optimal schedule at retrain cost $\\rho = 1$ "
              "(number of retrains in parentheses). For \\emph{every $k$}, the best $k$ is chosen in "
              "hindsight. Lower is better; best non-oracle policy in bold.",
        "tab:retrain", "l" + "r" * len(exps)), encoding="utf-8")


def inflight_ci_table() -> None:
    body = [r"Experiment & Metric & vs.\ complete cohort & vs.\ labels to date & vs.\ CBPE & vs.\ DA-CBPE \\",
            r"\midrule"]
    for name in ("freddie_2007", "freddie_2016"):
        r = Result(name, None, pd.read_csv(RESULTS / f"inflight_{name}_records.csv"), {})
        for metric, mname in (("prevalence", "default rate"), ("roc_auc", "AUC")):
            cells = []
            for base in ("latest_complete_cohort", "labels_to_date", "cbpe", "da_cbpe"):
                c = r.compare(metric, "da_cbpe_hz", base)
                txt = f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
                cells.append(rf"\textbf{{{txt}}}" if c["hi"] < 0 else txt)
            body.append(f"{tex(name)} & {mname} & " + " & ".join(cells) + r" \\")
    (OUT / "inflight_ci.tex").write_text(wrap(
        body, "In-flight book: MAE of the hazard variant minus MAE of each baseline (negative = ours better), "
              "with 95\\% moving-block bootstrap intervals over quarters (block length 4, 2000 resamples). "
              "Bold: interval excludes zero.", "tab:inflight-ci", "llrrrr"), encoding="utf-8")


def accuracy_table() -> None:
    files = glob.glob(str(RESULTS / "seeds_estimation_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["experiment", "metric", "estimator", "seed"])
    d = d[d["metric"] == "accuracy"]
    cols = ["reference", "latest_complete_cohort", "cbpe", "atc", "doc", "da_cbpe", "da_cbpe_hz"]
    names = ["Ref.", "Complete", "CBPE", "ATC", "DoC", "DA-CBPE", "DA-CBPE-HZ"]
    agg = d.groupby(["experiment", "estimator"])["mae"].agg(["mean", "std"])
    body = ["Experiment & " + " & ".join(names) + r" \\", r"\midrule"]
    for e in sorted(d["experiment"].unique()):
        g = agg.loc[e]
        best = g.loc[[c for c in cols if c in g.index], "mean"].min()
        cells = []
        for c in cols:
            if c not in g.index:
                cells.append("--")
                continue
            txt = f"{g.loc[c, 'mean']:.4f} ({g.loc[c, 'std']:.4f})"
            cells.append(rf"\textbf{{{txt}}}" if np.isclose(g.loc[c, "mean"], best) else txt)
        body.append(f"{tex(e)} & " + " & ".join(cells) + r" \\")
    (OUT / "accuracy.tex").write_text(wrap(
        body, "Accuracy estimation, including the accuracy-only estimators ATC and DoC (balanced-class "
              "streams). MAE, mean (std) over 5 seeds.", "tab:accuracy", "l" + "r" * len(cols)),
        encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detection_tables()
    retrain_table()
    inflight_ci_table()
    accuracy_table()
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
