# Contributing

Contributions of all kinds are welcome: new methods for the leaderboard, new data streams, bug fixes and documentation.

## Submitting a method to the leaderboard

The leaderboard ranks **performance estimators** (they estimate the model's metric before labels arrive) and **degradation detectors** (they raise alarms). Live results: https://rajkumar160798.github.io/label-free-model-monitoring/

1. **Set up**
   ```bash
   git clone https://github.com/rajkumar160798/label-free-model-monitoring.git
   cd label-free-model-monitoring
   uv sync --all-extras            # or: pip install -e ".[all]"
   ```
2. **Write your method** in `submissions/<name>.py`, starting from [`submissions/example_mean_score.py`](submissions/example_mean_score.py).
   - An estimator subclasses `lfmm.estimators.Estimator`, sets `name` and `supports` (the metrics it can estimate: `roc_auc`, `prevalence`, `accuracy`), and implements `estimate(metric, proba, history)`. Optional: `fit(proba_ref, y_ref, threshold)` to calibrate on the labeled reference window (call `super().fit(...)`), and `estimate_inflight(metric, history, target)` for the in-flight task.
   - A detector subclasses `lfmm.detectors.Detector`, implements `fit(X_ref, proba_ref, y_ref, threshold)` and `score(X, proba, history)`, and returns `(drift_score, alarm)`, where a higher score means more drift.
   - The constructor must work without arguments.
3. **Get the data** you can (see [DATA.md](DATA.md)). `lfmm-download all` fetches ACS, BRFSS and NYC taxi. Freddie Mac and TabReD need a free account; register their files with `lfmm-download sflld` and `lfmm-download tabred`. Experiments without data are skipped.
4. **Evaluate**
   ```bash
   uv run python scripts/evaluate_submission.py submissions/<name>.py:<ClassName> \
       --author "Your Name" --description "One sentence" --url https://link-to-paper-or-code
   uv run python scripts/build_leaderboard.py      # see where you rank
   ```
5. **Open a pull request** with `submissions/<name>.py` and `results/submissions/<name>/`. If you cannot open a pull request, [open a submission issue](https://github.com/rajkumar160798/label-free-model-monitoring/issues/new?template=method_submission.yml) instead.

### Rules

- A method may use only what the harness passes it: the model's scores (and features, for detectors) on the batch, and the `History` of earlier rows with labels that have already arrived. No outside data, no reading files, no access to future labels.
- Every experiment uses the same training date, model and seed as the baselines; do not change `lfmm/experiments.py` in a submission.
- Before merging, the maintainers re-run the submission on all experiments with the current release, so the leaderboard compares like for like. If the re-run differs materially from what you submitted, we will discuss it in the pull request.
- Submitted code is included under the MIT license.

### How ranking works

Within each case (an experiment and metric, or an experiment and degradation event), methods are ranked by mean absolute error (estimation), Spearman correlation with the true degradation (detection) or regret against the hindsight-optimal schedule (retraining). A detector whose score never changes in a case gets a correlation of 0 there. The leaderboard reports each method's mean rank, first places and mean score; methods evaluated on every case are ranked before methods evaluated on fewer.

## Other contributions

- **Bugs and ideas:** open an issue.
- **Code:** run `uv run pytest` (fast tests) before opening a pull request; `uv run pytest -m slow` runs the long statistical checks.
- **New data streams:** a loader that returns `lfmm.streams.base.Stream` with honest label arrival times is the most valuable addition. Please describe in the pull request how the label and its arrival time are constructed.
