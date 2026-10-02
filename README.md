# lfmm: label-free model monitoring under real temporal drift

A benchmark and toolkit for drift detection, label-free performance estimation and retraining decisions on naturally drifting, timestamped data with realistic label delay.

Work in progress. Paper draft: [paper/main.pdf](paper/main.pdf). Data plan and download status: [DATA.md](DATA.md). Project status, design decisions and first results: [PROGRESS.md](PROGRESS.md).

## Setup

```bash
uv sync
# Raw data lives outside the repo; point the package at it:
export LFMM_DATA_DIR=/path/to/lfmm-data      # Windows: setx LFMM_DATA_DIR C:\data\lfmm
```

If the repo sits in a synced folder (OneDrive, Dropbox), keep the virtual environment elsewhere: `UV_PROJECT_ENVIRONMENT=C:\venvs\lfmm` and `UV_LINK_MODE=copy`.

## Download data

```bash
uv run python -m lfmm.download acs      # US Census ACS PUMS via folktables layout
uv run python -m lfmm.download tlc --start 2019-01 --end 2021-12   # NYC yellow taxi
uv run python -m lfmm.download brfss    # CDC BRFSS annual surveys
uv run python -m lfmm.download all
```

Downloads resume after an interruption and skip files already present. Each file's source URL, size and sha256 are written to `$LFMM_DATA_DIR/manifest.csv`.

## Run the benchmark

```bash
uv run python scripts/run_estimation.py     # performance-estimation experiments -> results/
uv run python scripts/run_detection.py      # degradation-detection experiments -> results/detection_*
uv run python scripts/run_inflight.py       # in-flight book estimation (natural label delay) -> results/inflight_*
uv run python scripts/run_retrain.py        # retrain-or-not task (slow: one model per period) -> results/retrain_*
uv run python scripts/run_seeds.py          # estimation over 5 seeds -> results/seeds_*
uv run python scripts/make_figures.py       # figures/ from results/
uv run python scripts/make_tables.py        # paper/tables/ from seed results
uv run python scripts/build_leaderboard.py  # leaderboard/ from results/
uv run pytest -q
```

## Tasks

| Task | Question | Scored by |
|---|---|---|
| Estimation (next batch) | What will this batch's metric turn out to be? | MAE vs the true metric, with block-bootstrap 95% intervals |
| Estimation (in-flight book) | What is the metric of everything scored in the last N months, some of it already labeled? | same |
| Detection | Has the model got worse than at deployment? | rank correlation with true degradation, AUROC, alarm precision/recall |
| Retrain or not | Retrain now, given the labels that have arrived? | total loss + λ × retrains, regret vs the hindsight-optimal schedule |

Monitors only ever see labels whose `label_time` has passed. Training uses only cohorts whose labels have (almost) all arrived.

## Add your own method

An estimator is fit on a labeled reference window, then asked for a metric on each new batch. It sees the batch's model scores and a `History` of earlier rows, holding only the labels that have arrived so far:

```python
import numpy as np
from lfmm.estimators import Estimator, History, metric_value
from lfmm.experiments import EXPERIMENTS
from lfmm.harness import run_estimation

class MeanScore(Estimator):
    """Toy example: the average model score as the positive rate."""
    name = "mean_score"
    supports = ("prevalence",)

    def estimate(self, metric: str, proba: np.ndarray, history: History) -> float:
        return float(proba.mean())

exp = EXPERIMENTS["freddie_2007"]
result = run_estimation(exp.load(), metrics=("prevalence",), estimators=[MeanScore()],
                        **exp.deploy_kwargs())
print(result.bootstrap_mae("prevalence"))
```

Detectors follow the same pattern (`lfmm.detectors.Detector`: `fit` on the reference window, `score` returns `(drift_score, alarm)`), as do retraining policies (`lfmm.retrain.Policy`).

## License

MIT; see [LICENSE](LICENSE). Data is not redistributed: Freddie Mac and TabReD have their own terms; use the loaders with your own downloads.
