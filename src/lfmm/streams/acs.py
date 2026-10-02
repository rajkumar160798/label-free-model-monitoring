"""US Census ACS PUMS (person files) as a yearly stream: the folktables ACSIncome task.

Task: predict whether personal income (PINCP) exceeds $50,000, for employed
adults (same filter as folktables ACSIncome). ACS has no natural label delay,
so labels arrive ``label_delay_days`` after the survey year's midpoint.

Harmonization across years:

- RELP (to 2018) and RELSHIPP (2019 on) are mapped to one relationship code.
- The state column is ST up to 2022 and STATE from 2023.
- OCCP switched to 2018 SOC-based codes in 2018. It is kept as-is, so that
  year carries a real coding shift, as a production pipeline would see.
- Income is nominal dollars (not inflation-adjusted), as in folktables.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import processed_dir, raw_dir
from .base import Stream, with_label_delay

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
FEATURES_NUM = ["AGEP", "SCHL", "WKHP", "OCCP", "POBP"]
FEATURES_CAT = ["COW", "MAR", "REL", "SEX", "RAC1P", "ST"]

# RELSHIPP (2019+) -> RELP (2010-2018) coding.
RELSHIPP_TO_RELP = {
    20: 0, 21: 1, 23: 1, 22: 13, 24: 13, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6,
    30: 7, 31: 8, 32: 9, 33: 10, 34: 12, 35: 14, 36: 15, 37: 16, 38: 17,
}


def _read_year(year: int, data_dir=None) -> pl.DataFrame:
    base = raw_dir(data_dir) / "folktables" / str(year) / "1-Year"
    pattern = "psam_p*.csv" if year >= 2017 else f"ss{str(year)[-2:]}p*.csv"
    files = sorted(base.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no ACS person CSVs for {year} in {base}")
    header = files[0].open().readline().strip().replace('"', "").split(",")
    rel = "RELSHIPP" if "RELSHIPP" in header else "RELP"
    st = "STATE" if "STATE" in header else "ST"
    cols = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", rel, "WKHP", "SEX", "RAC1P",
            "PINCP", "PWGTP", st]
    frames = [
        pl.read_csv(f, columns=cols, infer_schema=False).rename({rel: "REL", st: "ST"})
        for f in files
    ]
    df = pl.concat(frames).with_columns(pl.all().cast(pl.Float64, strict=False))
    if rel == "RELSHIPP":
        df = df.with_columns(pl.col("REL").replace_strict(RELSHIPP_TO_RELP, default=None, return_dtype=pl.Float64))
    # folktables ACSIncome filter
    df = df.filter((pl.col("AGEP") > 16) & (pl.col("PINCP") > 100)
                   & (pl.col("WKHP") > 0) & (pl.col("PWGTP") >= 1))
    return df.with_columns(pl.lit(year).alias("year"),
                           (pl.col("PINCP") > 50_000).cast(pl.Float64).alias("y"))


def build_acs_table(data_dir=None, years: list[int] = YEARS) -> pl.DataFrame:
    frames = []
    for year in years:
        print(f"  reading ACS {year}...")
        frames.append(_read_year(year, data_dir))
    return pl.concat(frames, how="diagonal")


def load_acs_income(data_dir=None, years: list[int] | None = None, states: list[str] | None = None,
                    sample_per_year: int | None = 100_000, label_delay_days: int = 365,
                    seed: int = 0, rebuild: bool = False) -> Stream:
    """ACSIncome as a Stream. ``sample_per_year`` keeps runs fast (None = all rows)."""
    cache = processed_dir(data_dir) / "acs_income.parquet"
    if rebuild or not cache.exists():
        build_acs_table(data_dir).write_parquet(cache)
    df = pl.read_parquet(cache)
    if years is not None:
        df = df.filter(pl.col("year").is_in(years))
    if states is not None:
        from folktables.load_acs import _STATE_CODES
        codes = [float(_STATE_CODES[s.upper()]) for s in states]
        df = df.filter(pl.col("ST").is_in(codes))
    if sample_per_year is not None:
        df = df.group_by("year", maintain_order=True).map_groups(
            lambda g: g.sample(min(sample_per_year, g.height), seed=seed))
    df = df.sort("year")

    X = df.select(FEATURES_NUM + FEATURES_CAT).to_pandas()
    for c in FEATURES_CAT:
        X[c] = X[c].astype("Int64").astype("category")
    event_time = ((df["year"].to_numpy() - 1970).astype("datetime64[Y]").astype("datetime64[D]")
                  + np.timedelta64(181, "D"))  # 1 July
    stream = Stream(
        name="acs_income",
        X=X,
        y=df["y"].to_numpy(),
        event_time=event_time,
        label_time=event_time,
        meta={"label_delay": f"simulated {label_delay_days} days", "target": "PINCP > 50000"},
    )
    return with_label_delay(stream, np.timedelta64(label_delay_days, "D"))
