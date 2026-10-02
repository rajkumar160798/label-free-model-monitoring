# Data Plan: Label-Free Model Monitoring Benchmark

The data plan for the benchmark. Claude reads this file at the start of each session to learn where the data is and what has been downloaded.

> Licenses, year ranges and file sizes below were written from memory and have not been checked against the sources. Confirm them on each source page before relying on them.

---

## Data location

**Data folder:** `C:\data\lfmm` (raw files in `C:\data\lfmm\raw`). `LFMM_DATA_DIR` is set to this as a user environment variable.

- Keep raw data **outside OneDrive**. The datasets add up to tens of GB, which OneDrive would try to sync.
- This project folder holds only code, configs and small processed samples.
- The package reads the data location from the environment variable `LFMM_DATA_DIR`.

Suggested layout inside the data folder:

```
<LFMM_DATA_DIR>/
  raw/            # files exactly as downloaded, never modified
    folktables/
    nyc_tlc/
    brfss/
    freddie_mac/
    tabred/
    lending_club/
    bts_airline/
  processed/      # cleaned, harmonized Parquet files
  manifest.csv    # one row per raw file: source URL, download date, size, sha256
```

---

## Download status

Mark each item as it is downloaded, and note any deviations under "Notes".

- [x] Folktables (ACS PUMS), 2014–2019 + 2021–2024, 50 states + PR (folktables omits DC): 510 state files, 31 GB with extracted CSVs (2026-10-01)
- [x] NYC TLC Yellow Taxi, 2019–2021 (first pass): 36 months, 140.2M trips, 2.0 GB (2026-10-01)
- [ ] NYC TLC Yellow Taxi, 2015–present (full)
- [x] CDC BRFSS, 2011–2024: 14 years, 1.3 GB (2026-10-01)
- [x] Freddie Mac SFLLD sample files, origination years 1999–2026: 28 zips, 1.2 GB, 50k loans per full year, performance through 2026-03 (manual download, 2026-10-01)
- [x] TabReD, all 8 preprocessed datasets: 1.5 GB (manual download, 2026-10-01)
- [ ] Lending Club (optional)
- [ ] BTS Airline On-Time (optional)
- [ ] MIMIC-IV credentialing started (separate case study only)

**Notes:**

- Download code: `src/lfmm/download/`. Run `uv run python -m lfmm.download {acs,tlc,brfss,all}`. Rerunning skips files already present; `manifest.csv` records URL, size and sha256 for each file.
- Progress log for the latest run: `C:\data\lfmm\download.log`.
- ACS: 2020 1-year is skipped (experimental release, 404 on the standard census.gov path). ACS 2024 and BRFSS 2024 were available, so they are included beyond the original plan.
- ACS layout follows folktables: `raw/folktables/{year}/1-Year/csv_p{st}.zip`, with the CSV extracted alongside. Load offline with `ACSDataSource(survey_year=..., horizon='1-Year', survey='person', root_dir=r'C:\data\lfmm\raw\folktables')` and `get_data(download=False)`.
- ACS CSV data dictionaries (`definition.csv`) exist only for 2017 onward.
- **ACS harmonization needed:** census renamed `RELP` to `RELSHIPP` (with new codes) from 2019 on, so folktables' built-in ACSIncome task fails on 2019+ data. Map the two before building cross-year streams.
- BRFSS: XPT files read with `pd.read_sas(f, format='xport')` straight from the zip. Column count varies by year (454 in 2011, 279 in 2020, 301 in 2024). `IDATE` (interview date) is present in every year checked.
- All files verified readable on 2026-10-01.
- **Freddie Mac SFLLD:** downloaded by hand from the Clarity Data Intelligence portal (`claritydownload.fmapps.freddiemac.com/CRT/` → SFLLD Data Download; it is the official SFLLD location despite the "CRT" in the URL). Register new hand-downloaded zips with `uv run python -m lfmm.download sflld`.
  - Each `sample_YYYY.zip` holds `sample_orig_YYYY.txt` (one row per loan) and `sample_perf_YYYY.txt` (one row per loan-month). Pipe-delimited, no header; the column layout is in Freddie Mac's SFLLD user guide.
  - Share of loans ever 90+ days late or REO-acquired: 2003 4.1%, 2007 16.5%, 2019 5.6%, 2025 0.1%. Recent vintages have not had time to default, which is the real label delay.
  - **Label fix (2026-10-02):** the 24-month window is measured in calendar months from the first payment. Freddie Mac *resets* the reported LOAN AGE when a loan is modified, so using it let defaults years later count as early defaults (17.7% of positives before the fix). After the fix: 18,777 positives (was 22,812), no positive arrives later than 23 months, ≥99.6% of labels resolve for every full year; 24-month default rate e.g. 2006 1.7%, 2007 4.2%, 2008 4.8%, 2019 4.6%. Every Freddie Mac result produced before 2026-10-02 used the faulty labels and has been rerun.
  - Free for non-commercial research; redistribution needs a license. Ship a loader only, never the data.
  - Full standard (`historical_data_*`), Non-Standard and RPL files were not downloaded.
- **TabReD:** preprocessed archives in `raw/tabred/preprocessed/*.tabred`, from the TabReD authors' Kaggle upload, https://www.kaggle.com/datasets/irubachev/tabred (Ivan Rubachev, first author of the TabReD paper). Register with `uv run python -m lfmm.download tabred`.
  - Each `.tabred` is a gzipped tar holding `<name>/info.json, x_num.npy, x_cat.npy, x_bin.npy` (when present), `x_meta.npy, y.npy, splits/` (default, random-0..2, sliding-window-0..2).
  - Rows: cooking-time 320k, delivery-eta 416k, ecom-offers 160k, homecredit-default 382k, homesite-insurance 261k, maps-routing 341k, sberbank-housing 28k, weather 424k. Tasks: 3 binary classification (roc-auc), 5 regression (rmse).
  - Timestamp is `x_meta[:, 0]`, but units differ: **microseconds since epoch** for cooking-time, delivery-eta, homesite-insurance, maps-routing, weather; **days since epoch** for ecom-offers, homecredit-default, sberbank-housing.
  - Only the preprocessed arrays are here; the raw Yandex parquet files were not downloaded.
  - Licenses: Yandex-origin data is non-commercial research only; Kaggle-derived datasets (homecredit, homesite, ecom-offers, sberbank) follow their competition rules. Ship a loader only.
- `raw/freddie_mac/sflld/` also contains two resumes and a ScreenPal installer copied there by mistake. They are not in the manifest, and the loader only reads `sample_*.zip` / `historical_data_*.zip`.
- Freddie Mac, TabReD (Kaggle parts) and Lending Club need an account or accepted terms, so they are manual downloads.

---

## Tier 1: start here (free, no sign-up, timestamped)

### 1. Folktables (US Census ACS PUMS)
- **Access:** `pip install folktables`. It downloads from census.gov by itself.
- **What to get:** person files for all states, 2014–2018 (the officially supported years). Add later years straight from the census.gov PUMS site if needed.
- **Time unit:** year. State gives a spatial shift as well.
- **Tasks:** ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime.
- **Rough size:** ~1–2 GB per year.
- **Caveat:** ACS 2020 1-year data was released only as "experimental" estimates.

### 2. NYC TLC Trip Records
- **Access:** monthly Parquet files from the NYC TLC Trip Record Data page.
- **What to get:** Yellow Taxi, Jan 2015 to now. Start with 2019–2021 to cover the COVID break. Skip the high-volume FHV files at first (~20M rows a month).
- **Time unit:** second-level timestamps.
- **Tasks:** tip given (card payments only), trip duration, fare.
- **Rough size:** ~50–150 MB per month.

### 3. CDC BRFSS (Behavioral Risk Factor Surveillance System)
- **Access:** annual SAS XPT files from the CDC BRFSS site.
- **What to get:** 2011 onward. The method changed in 2011, so earlier years aren't comparable.
- **Time unit:** year, with monthly order from the interview date fields.
- **Tasks:** diabetes, high blood pressure, general health.
- **Rough size:** ~1 GB per year unzipped.
- **Caveat:** variable names change between years, so a mapping table is needed to line them up.

**Headline experiment:** all three show the 2020 COVID shock as natural drift, across census, mobility and health data.

---

## Tier 2: add once the evaluation harness works

### 4. TabReD
- **Access:** download scripts in the `yandex-research/tabred` GitHub repo. Some datasets need the Kaggle API, and each competition's rules must be accepted first.
- **Why:** industrial datasets with timestamps, built for temporal splits. Reviewers will expect them.
- **Caveat:** check redistribution terms per dataset. Kaggle competition data usually can't be redistributed.

### 5. Freddie Mac Single-Family Loan-Level Dataset (recommended replacement for Lending Club)
- **Access:** free registration at freddiemac.com. Fannie Mae offers similar data.
- **Why:** the monthly performance files show defaults appearing months after the loan starts, so the **label delay is real**, not simulated. This makes it the core dataset for the "labels arrive late" argument.
- **Use:** check the simulated label delay against this real delay.

---

## Tier 3: optional

### 6. Lending Club
- The official downloads were taken down around 2020. Kaggle copies (2007–2020Q3) exist, but their licensing is unclear.
- If used, the loader should point users to Kaggle. **Do not redistribute.**
- Freddie Mac covers the same credit-risk ground with cleaner licensing.

### 7. BTS Airline On-Time Performance
- Public domain, monthly back to 1987. Task: predicting flight delays.
- Adds an extra domain with long-run drift.

### 8. MIMIC-IV (ClinicDrift case study only)
- Needs PhysioNet credentialing (CITI training plus approval, which takes weeks). Start early if wanted.
- Its terms don't allow redistribution, so keep it **out of the benchmark core**.

---

## Label delay
- Only Freddie Mac and Fannie Mae have label delay built into the data.
- For every other dataset, the package simulates it as a configurable delay.

---

## Suggested download order
1. Folktables: one `pip install` and a few lines of code
2. NYC TLC for 2019–2021, around the COVID break
3. BRFSS for 2011–2023
4. Freddie Mac (register early, since approval can take a moment)
5. TabReD

## Next step after data
Set up the repo: package layout, `LFMM_DATA_DIR` config, download scripts for the Tier 1 sources that write sha256 checksums to `manifest.csv`.
