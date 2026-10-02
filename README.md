# lfmm: label-free model monitoring under real temporal drift

A benchmark and toolkit for drift detection, label-free performance estimation and retraining decisions on naturally drifting, timestamped data with realistic label delay.

Work in progress. Data plan and download status: [DATA.md](DATA.md). Project status, design decisions and first results: [PROGRESS.md](PROGRESS.md).

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
uv run pytest -q
```
