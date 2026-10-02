# Progress

Where the project stands and what comes next. Read this together with [DATA.md](DATA.md) at the start of each session.

## Environment

- Python env lives **outside OneDrive** at `C:\venvs\lfmm` (OneDrive locked files in an in-project `.venv`). Set as user env vars: `UV_PROJECT_ENVIRONMENT=C:\venvs\lfmm`, `UV_LINK_MODE=copy`, `LFMM_DATA_DIR=C:\data\lfmm`.
- Run anything with `uv run ...`; tests with `uv run pytest -q`.

## Code layout

| Path | What it does |
|---|---|
| `src/lfmm/download/` | Downloaders and manual-file registration; writes `manifest.csv` |
| `src/lfmm/streams/base.py` | `Stream`: features, label, `event_time`, `label_time` (when the label arrives); `with_label_delay` for simulated delay |
| `src/lfmm/streams/freddie.py` | Freddie Mac: 90+ days late within 24 months; **natural** label arrival |
| `src/lfmm/streams/acs.py` | ACS income > $50k, 2014–2024; RELP/RELSHIPP and ST/STATE harmonized; 365-day delay |
| `src/lfmm/streams/brfss.py` | BRFSS diagnosed diabetes, 2011–2024, monthly by interview date; 28 variables harmonized; 180-day delay |
| `src/lfmm/streams/nyc_tlc.py` | NYC yellow taxi, card trips, tip ≥ 25% of fare, 2019–2021, 50k trips/month; 30-day delay |
| `src/lfmm/streams/tabred.py` | TabReD binary tasks: homecredit-default, homesite-insurance, ecom-offers; 90/30/14-day delays |
| `src/lfmm/experiments.py` | Registry of the 8 experiments (stream, training date, batch frequency, metrics, events) |
| `src/lfmm/estimators.py` | Estimators: reference, recent_arrivals, latest_complete_cohort, CBPE, ATC, DoC, **da_cbpe (ours)**, labels_to_date |
| `src/lfmm/detectors.py` | Detectors: univariate tests, PSI, score KS, domain classifier, MMD, ADWIN on scores, ADWIN on arrived errors, estimator alarms |
| `src/lfmm/harness.py` | `deploy`, `run_estimation`, `run_inflight`, `run_detection`; block-bootstrap intervals (`Result.bootstrap_mae`, `Result.compare`) |
| `src/lfmm/retrain.py` | Retrain-or-not: model bank, policies, hindsight oracle (dynamic programming), regret |
| `scripts/run_*.py` | One script per task, writing to `results/`; `make_figures.py` writes `figures/` |
| `tests/` | 35 tests: estimator maths, leakage, detectors, oracle vs brute force, da_cbpe on synthetic concept shift |

## Design decisions

- **Training uses only cohorts whose labels have ≥95% arrived** by the training date. Partly labeled cohorts over-represent early-arriving labels (defaults), which biases the model.
- **Freddie Mac label timing:** a positive is known in the month of the first credit event; a negative is known at loan age 24 or at earlier prepayment. Loans with neither (data ends, rare terminations) never resolve (`y = NaN`). 4 modified loans with reset first-payment dates were dropped.
- **Truth for a period** is computed only when ≥90% of its labels ever resolve, so 2024Q2 onward has estimates but no truth: the real "labels haven't arrived" frontier.
- `accuracy` is not used for Freddie Mac: with ~1% defaults it just tracks the default rate.

## First results (2026-10-01)

Mean absolute error of each estimate against the true metric, per quarter (Freddie Mac) or year (ACS). Lower is better.

| Experiment | Metric | Best | Notes |
|---|---|---|---|
| freddie_2007 (69 quarters) | AUC | CBPE 0.038 | latest complete cohort 0.048, reference 0.056 |
| freddie_2007 | default rate | reference 0.012 | CBPE 0.015, misses the crisis entirely |
| freddie_2016 (33 quarters) | AUC | latest complete cohort 0.044 | CBPE 0.068, reference 0.102 |
| acs_income_2016 (8 years) | AUC | latest complete cohort 0.002 | CBPE 0.006 |
| acs_income_2016 | accuracy | latest complete cohort 0.009 | ATC 0.025, CBPE/DoC 0.029 |

**The headline pattern so far: the two kinds of estimator fail in opposite ways.**

- *Label-free* (CBPE) misses shocks that change outcomes but not features. 2008Q1 loans: true default rate 7.7%, CBPE 1.0%. COVID, 2019Q1 loans: true 4.4%, CBPE 0.4%.
- *Label-based* (latest complete cohort) sees the shock about 2 years late, then keeps reporting it after it ends. It reported 4.7% in 2010Q1 when the true rate was 0.8%, and 4.6% in 2021Q1 against a true 0.8%.
- Neither family handles both. That gap is the benchmark's core motivation.

## Degradation detection task (2026-10-01)

Code: `src/lfmm/detectors.py`, `run_detection` in `src/lfmm/harness.py`, `scripts/run_detection.py` (about 4.5 min, mostly the domain classifier). Tests: `tests/test_detectors.py`.

**Detectors:**
- Input and output drift tests:
  - `univariate_tests`: KS / chi-squared with Bonferroni correction; scored by the largest effect size (KS statistic or Cramér's V).
  - `psi_max`: largest PSI over features; alarms above 0.25.
  - `score_ks`: KS test on the model's scores.
  - `domain_classifier`: cross-validated AUC of telling reference rows from batch rows; alarms above 0.6.
- Estimator-based alarms (`<estimator>:<metric>`): alarm when the estimated metric is worse than the reference by more than delta.

**Event:** a period whose true metric is worse than the reference by more than delta. For default rate, a change in either direction counts. The baseline is the reference-window metric, which is in-time and therefore optimistic. In freddie_2016, 85% of quarters lose more than 0.05 AUC, so the event rate is reported next to precision and recall.

**Scores:**
- Spearman correlation between the detector's score and the true degradation.
- AUROC at several delta values.
- Alarm precision, recall and false-alarm rate at one delta.

| Experiment / event | Best at ranking | Alarms |
|---|---|---|
| freddie_2007, AUC drop > 0.05 | CBPE (ρ 0.56, AUROC 0.82) | CBPE almost never fires (recall 3%). Latest complete cohort: precision 0.70, recall 0.68. All drift tests fire every quarter. |
| freddie_2007, default rate change > 0.01 | nothing beats chance (best AUROC 0.54) | CBPE is *anti*-correlated (ρ −0.56): the crisis is invisible in the inputs |
| freddie_2016, AUC drop > 0.05 | PSI (ρ 0.50); domain classifier AUROC 0.91 | Latest complete cohort is the only usable alarm (precision 0.96, recall 0.86); drift tests fire every quarter |
| freddie_2016, default rate change > 0.01 | drift tests (ρ ≈ 0.3, AUROC ≈ 0.8) | still fire every quarter |
| acs_income_2016 (accuracy, prevalence) | every detector ranks perfectly | uninformative: only 8 periods, and drift grows steadily with time |

**Findings:**
- **Input-drift tests fire constantly on real data.** Alarm rate is 100% on Freddie Mac. The largest PSI comes from changes that aren't necessarily harmful: lender churn (`seller_name`), interest-rate levels (`orig_rate`), the "TPO" channel code disappearing after 2007, and `valuation_method` only existing from 2017. Their scores rank severity somewhat, but their alarm thresholds are useless.
- **CBPE ranks AUC drops well but its alarms almost never fire**, because its estimates move too little. Its threshold would need calibrating separately from its estimate.
- **Shifts in outcome rate with no change in inputs (2007–08, COVID) are invisible to every label-free method.** That is the opening for methods that use early-arriving labels.
- **ACS needs a harder setup** (states as shifts, monthly or state-level batches) before its detection results mean anything.

## More streams (2026-10-01)

| Experiment | Stream | Rows | Train | Monitor | Notes |
|---|---|---|---|---|---|
| brfss_2014 | BRFSS diabetes | 1.68M (120k/yr sample of 6.3M) | 2011–mid 2013 | quarterly to 2025 | diabetes rate 12.4% → 14.6%; employment missing 2011–12, insurance missing 2021–22 |
| tlc_2019 | NYC taxi tips | 1.8M | Feb–May 2019 | monthly to Dec 2021 | tip ≥ 25% rate jumps 25% → 58% in Feb 2019, likely the congestion surcharge (unconfirmed); `airport_fee` only from 2021 |
| tabred_homecredit | TabReD | 382k, 696 features | Jan–May 2019 | monthly, 14 periods | |
| tabred_homesite | TabReD | 261k | Jan–Aug 2013 | monthly, 20 periods | |
| tabred_ecom | TabReD | 160k | Mar 1–17 2013 | weekly, 5 periods | hardest: AUC falls ~0.22 after deployment; too few periods for firm conclusions |

The five TabReD regression datasets are not loaded yet; the harness is binary-only.

## Our method: delay-adjusted CBPE (`da_cbpe`)

Chain-ladder idea from insurance reserving, in `DelayAdjustedCBPE`:
1. From mature cohorts, learn the label arrival curves: F(age) for positives, G(age) for negatives.
2. Fit one logit offset δ on the reference-calibrated scores by maximum likelihood on the last 12 months of history. Pending labels contribute 1 − pF − (1−p)G, so early defaults move δ at once without being over-counted.
3. New batch: CBPE with the adjusted probabilities. In-flight book: arrived labels count as themselves; pending rows use the posterior p(1−F) / (1 − pF − (1−p)G).

With constant delay it reduces to recalibrating on the latest complete labels.

**Two estimation targets.** For long-horizon labels, a new batch's eventual outcome depends on events that haven't happened yet. 2019Q1 Freddie loans defaulted at 4.4% because of COVID a year later, which no method could know in 2019. So there is a second task, the **in-flight book** (`run_inflight`): the metric of everything scored in the last 24 months, part of it already labeled. That is what lenders actually monitor.

**Results** (MAE difference, 95% block-bootstrap interval; negative = da_cbpe better):

| Experiment | Metric | vs latest complete cohort | vs CBPE |
|---|---|---|---|
| in-flight, freddie_2007 | AUC | **−0.018 [−0.031, −0.007]** | **−0.011 [−0.019, −0.005]** |
| in-flight, freddie_2007 | default rate | −0.003 [−0.007, +0.002] | **−0.007 [−0.012, −0.003]** |
| in-flight, freddie_2016 | AUC | +0.001 [−0.007, +0.011] | **−0.026 [−0.040, −0.014]** |
| in-flight, freddie_2016 | default rate | +0.002 [−0.003, +0.009] | **−0.002 [−0.004, −0.000]** |
| tabred_homesite | positive rate | **−0.012 [−0.017, −0.008]** | −0.000 [−0.002, +0.001] |
| brfss_2014 | positive rate | **−0.001 [−0.002, −0.000]** | −0.001 [−0.002, +0.001] |
| freddie_2007 (next batch) | default rate | −0.003 [−0.009, +0.003] | +0.001 [−0.004, +0.007] |

In plain terms: on the in-flight book, da_cbpe beats label-free CBPE in every case, and it ties or beats the label-based baseline while not having to wait for complete cohorts. On next-batch estimation the advantage isn't statistically clear.

**Known limitation: calendar-time shocks.** In COVID, forbearance pushed loans of every age into 90-days-late status within a few months. The method assumes defaults arrive over a loan's life at the usual pace, so it read the burst as the early edge of a much bigger wave: in-flight estimate about 10% against a true 4.3% (figures/freddie_2007_default_rate.png). Chain-ladder reserving has the same weakness with "calendar-year effects". The obvious fix is to model a calendar-time hazard alongside the age pattern.

Figures: `figures/freddie_2007_default_rate.png` (next batch and in-flight book) and `figures/freddie_2007_auc.png`.

## Next steps

1. Detection and retrain results for all 8 experiments (runs in progress; see below once filled in).
2. Fix the calendar-time weakness in da_cbpe (time-varying hazard), then recheck COVID.
3. Speed up detection: compute metric-independent detectors once per deployment, not once per event metric.
4. Calibrated alarm thresholds (from held-out reference periods) instead of rules of thumb.
5. Repeated model seeds (the intervals so far cover period-to-period variation only).
6. Regression support, to add the five TabReD regression datasets.
7. Release: license choice, PyPI package, leaderboard site, workshop paper draft.
