# lfmm: monitoring ML models before the labels arrive

[![PyPI](https://img.shields.io/pypi/v/lfmm)](https://pypi.org/project/lfmm/)
[![Python](https://img.shields.io/pypi/pyversions/lfmm)](https://pypi.org/project/lfmm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/rajkumar160798/label-free-model-monitoring/blob/master/LICENSE)

A benchmark and toolkit for **performance estimation, drift detection and retraining decisions when true labels arrive late**: a loan default is known up to two years after the loan is made, a medical outcome weeks later, a fraud chargeback months later.

Most monitoring methods are evaluated on artificially injected drift with every label available immediately. `lfmm` replays **real, timestamped data** as streams in which a monitor sees each label only when it would have arrived in practice, including the *natural* label delay of mortgage defaults, and scores monitors on four tasks.

- Paper draft: [paper/main.pdf](https://github.com/rajkumar160798/label-free-model-monitoring/blob/master/paper/main.pdf)
- **Leaderboard:** https://rajkumar160798.github.io/label-free-model-monitoring/ ([submit a method](https://rajkumar160798.github.io/label-free-model-monitoring/submit.html), [CONTRIBUTING.md](https://github.com/rajkumar160798/label-free-model-monitoring/blob/master/CONTRIBUTING.md))

## Install

```bash
pip install lfmm                 # core: streams, tasks, estimators, detectors, retraining
pip install "lfmm[streaming]"    # + ADWIN detectors (river)
pip install "lfmm[data]"         # + downloaders and loaders for the benchmark datasets
pip install "lfmm[all]"          # everything needed to reproduce the paper
```

Python 3.10 or newer.

## Quickstart

Any table of scored events becomes a stream: features, label, when each row was scored, and when its label becomes known.

```python
import numpy as np
import pandas as pd
from lfmm.estimators import CBPE, DelayAdjustedCBPE, LatestCompleteCohort
from lfmm.harness import run_estimation
from lfmm.streams.base import Stream, with_label_delay

rng = np.random.default_rng(0)
n = 24_000
month = np.repeat(np.arange(24), n // 24)
X = pd.DataFrame({"x1": rng.normal(month / 12, 1), "x2": rng.normal(0, 1, n)})
y = (rng.random(n) < 1 / (1 + np.exp(-(X.x1 - X.x2 - 1)))).astype(float)
t = (np.datetime64("2022-01", "M") + month).astype("datetime64[ns]")

stream = Stream("demo", X, y, event_time=t, label_time=t)
stream = with_label_delay(stream, np.timedelta64(60, "D"))   # labels arrive 60 days late

# Train on fully labeled cohorts before 2023, freeze the model, walk forward monthly.
result = run_estimation(
    stream, train_end="2023-01-01", freq="M", metrics=("roc_auc", "prevalence"),
    estimators=[CBPE(), DelayAdjustedCBPE(), LatestCompleteCohort(freq="M")],
)
print(result.summary())                     # MAE and bias of each estimator
print(result.bootstrap_mae("prevalence"))   # with 95% block-bootstrap intervals
```

Monitors only ever see labels whose arrival time has passed, and the model is trained only on cohorts whose labels have (almost) all arrived.

## Tasks

| Task | Question | Scored by |
|---|---|---|
| Estimation (next batch) | What will this batch's metric turn out to be? | MAE vs the true metric, block-bootstrap intervals |
| Estimation (in-flight book) | What is the metric of everything scored in the last N months, part of it already labeled? | same |
| Detection | Has the model got worse than at deployment? | rank correlation with true degradation, AUROC, alarm precision/recall |
| Retrain or not | Retrain now, given the labels that have arrived? | total loss + λ × retrains, regret vs the hindsight-optimal schedule |

```python
from lfmm.harness import run_inflight, run_detection_events
from lfmm.retrain import build_bank, evaluate_policies
```

## Methods included

- **Estimators:** reference performance, recent arrivals, latest complete cohort, labels to date, CBPE, ATC, DoC, and **delay-adjusted CBPE** (ours) with its **hazard variant** for calendar-time shocks such as COVID forbearance.
- **Detectors:** univariate KS/χ², PSI, KS on scores, domain classifier, MMD, ADWIN on scores and on arrived errors, and alarms from any estimator.
- **Retraining policies:** never, always, every *k* periods, retrain on any detector's alarm, and the hindsight-optimal schedule (dynamic programming).

## Add your own method

Subclass `Estimator`, `Detector` or `Policy`. An estimator is fit on a labeled reference window and then sees each batch's model scores plus a `History` holding only the labels that have arrived:

```python
import numpy as np
from lfmm.estimators import Estimator, History

class MeanScore(Estimator):
    name = "mean_score"
    supports = ("prevalence",)

    def estimate(self, metric: str, proba: np.ndarray, history: History) -> float:
        return float(proba.mean())
```

## The benchmark datasets

| Stream | Source | Label | Label delay |
|---|---|---|---|
| Freddie Mac | Single-Family Loan-Level Dataset, 1999–2026 | default within 24 months | **natural** |
| ACS income | US Census PUMS (via Folktables), 2014–2024 | income > $50k | simulated, 365 days |
| BRFSS | CDC survey, 2011–2024 | diagnosed diabetes | simulated, 180 days |
| NYC taxi | TLC trip records, 2019–2021 | tip ≥ 25% of fare | simulated, 30 days |
| TabReD | homecredit, homesite, ecom (Rubachev et al.) | default / conversion / repeat purchase | simulated, 14–90 days |

```bash
export LFMM_DATA_DIR=/path/to/lfmm-data
lfmm-download acs          # also: tlc, brfss, all
lfmm-download sflld        # register hand-downloaded Freddie Mac files
lfmm-download tabred       # register hand-downloaded TabReD archives
```

Downloads resume and write each file's source URL and SHA-256 to `$LFMM_DATA_DIR/manifest.csv`. Freddie Mac and TabReD need an account or accepted terms, so they are downloaded by hand; no raw data is redistributed. The eight experiments of the paper are defined in `lfmm.experiments.EXPERIMENTS`.

## Reproducing the paper

From a clone of the repository:

```bash
uv sync --all-extras
uv run python scripts/run_estimation.py     # next-batch estimation -> results/
uv run python scripts/run_inflight.py       # in-flight book (Freddie Mac)
uv run python scripts/run_detection.py      # degradation detection
uv run python scripts/run_retrain.py        # retrain or not (slow: one model per period)
uv run python scripts/run_seeds.py          # estimation over 5 seeds
uv run python scripts/make_figures.py       # figures/
uv run python scripts/make_tables.py        # paper/tables/
uv run python scripts/make_appendix.py      # appendix tables
uv run python scripts/build_leaderboard.py  # leaderboard/
uv run pytest                               # fast tests; `uv run pytest -m slow` for the rest
```

Project notes: [PROGRESS.md](https://github.com/rajkumar160798/label-free-model-monitoring/blob/master/PROGRESS.md) (status, design decisions, results) and [DATA.md](https://github.com/rajkumar160798/label-free-model-monitoring/blob/master/DATA.md) (data sources and construction).

## License

MIT; see [LICENSE](https://github.com/rajkumar160798/label-free-model-monitoring/blob/master/LICENSE). The datasets keep their own terms.
