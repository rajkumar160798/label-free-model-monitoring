"""CDC BRFSS (2011-2024) as a monthly stream: predict diagnosed diabetes.

Task: has a doctor ever told the respondent they have diabetes (excluding
pregnancy-only), from demographics, health status and other conditions. This
is the target of the widely used CDC Diabetes Health Indicators dataset.

The survey has no natural label delay, so labels arrive ``label_delay_days``
after the interview (as if confirmed later from records). Interviews are
ordered by interview month (IYEAR, IMONTH), so a survey year's last interviews
can fall early in the next calendar year.

Harmonization (variable names change across years):

- renamed: DIABETE3/4, SEX/SEX1/SEXVAR, CHCKDNY1/2, ADDEPEV2/3, HAVARTH3/4/5,
  MEDCOST/MEDCOST1, PERSDOC2/3, _RACEGR2/3/4;
- _INCOMG1 (2021+) splits the top band ($50k+) into three; capped back to 5;
- EMPLOY1 does not exist in 2011-2012 and health insurance is not asked of all
  ages in 2021-2022; those values are left missing, as a pipeline would see them.
"""

from __future__ import annotations

import os
import tempfile
import zipfile

import numpy as np
import polars as pl

from ..config import processed_dir, raw_dir
from .base import Stream, with_label_delay

YEARS = list(range(2011, 2025))

# harmonized name -> source names, first match wins
SOURCES = {
    "state": ["_STATE"], "iyear": ["IYEAR"], "imonth": ["IMONTH"],
    "diabetes": ["DIABETE4", "DIABETE3"],
    "sex": ["SEXVAR", "SEX1", "SEX"],
    "age_group": ["_AGEG5YR"], "bmi": ["_BMI5"], "education": ["_EDUCAG"],
    "income": ["_INCOMG1", "_INCOMG"], "race": ["_RACEGR3", "_RACEGR4", "_RACEGR2"],
    "smoker": ["_SMOKER3"], "active": ["_TOTINDA"], "gen_health": ["GENHLTH"],
    "phys_days": ["PHYSHLTH"], "ment_days": ["MENTHLTH"], "checkup": ["CHECKUP1"],
    "marital": ["MARITAL"], "employ": ["EMPLOY1"],
    "insured": ["HLTHPLN1", "_HLTHPL2", "_HLTHPL1"], "cost_barrier": ["MEDCOST1", "MEDCOST"],
    "personal_doc": ["PERSDOC3", "PERSDOC2"],
    "heart_attack": ["CVDINFR4"], "chd": ["CVDCRHD4"], "stroke": ["CVDSTRK3"],
    "asthma": ["ASTHMA3"], "kidney": ["CHCKDNY2", "CHCKDNY1"],
    "depression": ["ADDEPEV3", "ADDEPEV2"], "arthritis": ["HAVARTH5", "HAVARTH4", "HAVARTH3"],
}
# "don't know / refused / not asked" codes per feature
MISSING = {
    "sex": [7, 9], "age_group": [14], "education": [9], "income": [9], "race": [9],
    "smoker": [9], "active": [9], "gen_health": [7, 9], "checkup": [7, 9], "marital": [9],
    "employ": [9], "insured": [7, 9], "cost_barrier": [7, 9], "personal_doc": [7, 9],
    "heart_attack": [7, 9], "chd": [7, 9], "stroke": [7, 9], "asthma": [7, 9],
    "kidney": [7, 9], "depression": [7, 9], "arthritis": [7, 9],
}
NUMERIC = ["age_group", "bmi", "education", "income", "gen_health", "phys_days", "ment_days", "checkup"]
CATEGORICAL = ["state", "sex", "race", "smoker", "active", "marital", "employ", "insured",
               "cost_barrier", "personal_doc", "heart_attack", "chd", "stroke", "asthma",
               "kidney", "depression", "arthritis"]


def _read_year(year: int, data_dir=None) -> pl.DataFrame:
    try:
        import pyreadstat
    except ImportError as exc:  # optional dependency
        raise ImportError("Reading BRFSS needs 'pyreadstat': pip install 'lfmm[data]'") from exc

    zpath = raw_dir(data_dir) / "brfss" / f"LLCP{year}XPT.zip"
    with zipfile.ZipFile(zpath) as z, tempfile.TemporaryDirectory() as tmp:
        member = z.namelist()[0]
        path = z.extract(member, tmp)
        _, meta = pyreadstat.read_xport(path, metadataonly=True, encoding="cp1252")
        present = {c.strip(): c for c in meta.column_names}
        pick = {}
        for name, candidates in SOURCES.items():
            src = next((c for c in candidates if c in present), None)
            if src is not None:
                pick[name] = present[src]
        df, _ = pyreadstat.read_xport(path, usecols=list(pick.values()), encoding="cp1252")
        os.remove(path)
    df = pl.from_pandas(df).rename({v: k for k, v in pick.items()})
    for name in SOURCES:  # features absent this year
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    return df.with_columns(
        *[pl.col(c).cast(pl.Float64, strict=False) for c in SOURCES],
        pl.lit(year).alias("survey_year"),
    ).select(["survey_year", *SOURCES])


def build_brfss_table(data_dir=None, years: list[int] = YEARS) -> pl.DataFrame:
    frames = []
    for y in years:
        print(f"  reading BRFSS {y}...")
        frames.append(_read_year(y, data_dir))
    df = pl.concat(frames)

    df = df.with_columns(
        [pl.when(pl.col(c).is_in(codes)).then(None).otherwise(pl.col(c)).alias(c)
         for c, codes in MISSING.items()]
    ).with_columns(
        (pl.col("bmi") / 100).alias("bmi"),
        pl.col("income").clip(upper_bound=5),
        # PHYSHLTH / MENTHLTH: 88 = none, 77/99 = unknown
        *[pl.when(pl.col(c) == 88).then(0.0).when(pl.col(c) > 30).then(None).otherwise(pl.col(c)).alias(c)
          for c in ("phys_days", "ment_days")],
        # 1 = yes; 2 (pregnancy only), 3 (no), 4 (prediabetes) = no; 7/9/blank dropped
        pl.when(pl.col("diabetes") == 1).then(1.0)
        .when(pl.col("diabetes").is_in([2, 3, 4])).then(0.0).otherwise(None).alias("y"),
        pl.date(pl.col("iyear").cast(pl.Int32), pl.col("imonth").cast(pl.Int32), 15).alias("event_time"),
    )
    return df.filter(pl.col("y").is_not_null() & pl.col("event_time").is_not_null()).select(
        ["survey_year", "event_time", "y", *NUMERIC, *CATEGORICAL])


def load_brfss_diabetes(data_dir=None, years: list[int] | None = None, sample_per_year: int | None = 120_000,
                        label_delay_days: int = 180, seed: int = 0, rebuild: bool = False) -> Stream:
    """BRFSS diabetes as a monthly Stream. ``sample_per_year`` keeps runs fast (None = all)."""
    cache = processed_dir(data_dir) / "brfss_diabetes.parquet"
    if rebuild or not cache.exists():
        build_brfss_table(data_dir).write_parquet(cache)
    df = pl.read_parquet(cache)
    if years is not None:
        df = df.filter(pl.col("survey_year").is_in(years))
    if sample_per_year is not None:
        df = df.group_by("survey_year", maintain_order=True).map_groups(
            lambda g: g.sample(min(sample_per_year, g.height), seed=seed))
    df = df.sort("event_time")

    X = df.select(NUMERIC + CATEGORICAL).to_pandas()
    for c in CATEGORICAL:
        X[c] = X[c].astype("Int64").astype("category")
    event_time = df["event_time"].to_numpy().astype("datetime64[ns]")
    stream = Stream("brfss_diabetes", X, df["y"].to_numpy(), event_time, event_time,
                    meta={"label_delay": f"simulated {label_delay_days} days",
                          "target": "ever told diabetes (excl. pregnancy-only)"})
    return with_label_delay(stream, np.timedelta64(label_delay_days, "D"))
