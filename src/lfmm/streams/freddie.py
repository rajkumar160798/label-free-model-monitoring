"""Freddie Mac Single-Family Loan-Level Dataset as a stream with natural label delay.

Task: predict at origination whether a loan becomes seriously delinquent within
``horizon`` months of its first payment. "Seriously delinquent" means 90+ days
late, REO acquisition, or a credit-loss termination (third-party sale, short
sale, REO disposition, whole-loan sale).

Label arrival is real, not simulated:

- positive: known in the reporting month of the first credit event;
- negative: known once the loan reaches age ``horizon`` without an event, or
  when it prepays earlier;
- otherwise (data ends first, or rare terminations such as defects) the label
  never arrives and ``y`` is NaN.

Column positions follow the SFLLD General User Guide (January 2026). The sample
files have 31 origination columns: Servicer Name is not included.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..config import processed_dir, raw_dir
from .base import Stream

ORIG_COLUMNS = [
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date", "msa",
    "mi_pct", "num_units", "occupancy", "cltv", "dti", "orig_upb", "ltv", "orig_rate",
    "channel", "ppm_flag", "amortization_type", "state", "property_type", "postal3",
    "loan_id", "loan_purpose", "orig_term", "num_borrowers", "seller_name",
    "super_conforming", "pre_relief_loan_id", "program", "relief_refi",
    "valuation_method", "interest_only", "mi_cancellation",
]
PERF_COLUMNS = {0: "loan_id", 1: "period", 3: "dlq", 4: "loan_age", 8: "zb_code"}

# Sentinel "not available" codes from the user guide.
MISSING_CODES = {
    "credit_score": 9999, "mi_pct": 999, "num_units": 99, "cltv": 999, "dti": 999,
    "ltv": 999, "num_borrowers": 99,
}
NUMERIC = ["credit_score", "mi_pct", "num_units", "cltv", "dti", "orig_upb", "ltv",
           "orig_rate", "orig_term", "num_borrowers"]
CATEGORICAL = ["first_time_homebuyer", "occupancy", "channel", "state", "property_type",
               "loan_purpose", "seller_name", "super_conforming", "program", "relief_refi",
               "valuation_method", "interest_only"]
# Left out: identifiers, maturity date (redundant with term), MSA and postal code
# (high cardinality), MI cancellation (happens after origination, so it leaks).

CREDIT_LOSS_ZB = ["02", "03", "09", "15"]
PREPAID_ZB = "01"


def _month(col: str) -> pl.Expr:
    return pl.col(col).str.strptime(pl.Date, "%Y%m", strict=False)


def _read_year(zpath: Path, year: int, horizon: int) -> pl.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        orig = pl.read_csv(z.read(f"sample_orig_{year}.txt"), separator="|", has_header=False,
                           new_columns=ORIG_COLUMNS, infer_schema=False)
        perf = pl.read_csv(z.read(f"sample_perf_{year}.txt"), separator="|", has_header=False,
                           columns=list(PERF_COLUMNS), new_columns=list(PERF_COLUMNS.values()),
                           infer_schema=False)

    perf = perf.with_columns(
        _month("period").alias("period"),
        pl.col("loan_age").cast(pl.Int32),
        pl.col("dlq").cast(pl.Int32, strict=False).alias("dlq_months"),
    )
    in_window = pl.col("loan_age") <= horizon
    event = (
        (pl.col("dlq_months") >= 3) | (pl.col("dlq") == "RA") | pl.col("zb_code").is_in(CREDIT_LOSS_ZB)
    ) & in_window
    prepaid = (pl.col("zb_code") == PREPAID_ZB) & in_window

    outcomes = perf.group_by("loan_id").agg(
        pl.col("period").filter(event).min().alias("event_month"),
        pl.col("period").filter(prepaid).min().alias("prepaid_month"),
        pl.col("period").filter(pl.col("loan_age") == horizon).min().alias("horizon_month"),
    )
    return orig.join(outcomes, on="loan_id", how="left")


def build_freddie_table(data_dir=None, horizon: int = 24) -> pl.DataFrame:
    """Parse every sample zip into one loan-level table. Slow; use load_freddie()."""
    folder = raw_dir(data_dir) / "freddie_mac" / "sflld"
    zips = sorted(folder.glob("sample_*.zip"))
    if not zips:
        raise FileNotFoundError(f"no sample_*.zip files in {folder}")
    frames = []
    for zpath in zips:
        year = int(zpath.stem.split("_")[1])
        print(f"  parsing SFLLD sample {year}...")
        frames.append(_read_year(zpath, year, horizon))
    df = pl.concat(frames)

    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC]
    ).with_columns(
        [pl.when(pl.col(c) == code).then(None).otherwise(pl.col(c)).alias(c)
         for c, code in MISSING_CODES.items()]
    ).with_columns(
        # Before 2018Q2 the field was only 1 vs "more than 1": cap so years agree.
        pl.col("num_borrowers").clip(upper_bound=2),
        _month("first_payment_date").alias("event_time"),
    )

    # Label and when it becomes known.
    df = df.with_columns(
        pl.when(pl.col("event_month").is_not_null()).then(1.0)
        .when(pl.col("prepaid_month").is_not_null() | pl.col("horizon_month").is_not_null()).then(0.0)
        .otherwise(None).alias("y"),
        pl.coalesce("event_month", pl.min_horizontal("prepaid_month", "horizon_month")).alias("label_time"),
    )
    # Reporting starts the month before the first payment, so a few prepayments
    # land up to two months "early"; treat them as known at the event. Larger gaps
    # are seller-owned modified loans whose first payment date was reset years
    # after origination (a handful per dataset); drop them.
    df = df.filter(
        pl.col("label_time").is_null()
        | (pl.col("label_time") >= pl.col("event_time").dt.offset_by("-2mo"))
    ).with_columns(pl.max_horizontal("label_time", "event_time").alias("label_time"))
    df = df.with_columns(
        pl.when(pl.col("y").is_null()).then(None).otherwise(pl.col("label_time")).alias("label_time")
    )
    return df.select(["loan_id", "event_time", "label_time", "y", *NUMERIC, *CATEGORICAL])


def load_freddie(data_dir=None, horizon: int = 24, years: list[int] | None = None,
                 rebuild: bool = False) -> Stream:
    """Freddie Mac SFLLD sample as a Stream ordered by first payment date."""
    cache = processed_dir(data_dir) / f"freddie_sflld_h{horizon}.parquet"
    if rebuild or not cache.exists():
        build_freddie_table(data_dir, horizon).write_parquet(cache)
    df = pl.read_parquet(cache)
    if years is not None:
        df = df.filter(pl.col("event_time").dt.year().is_in(years))
    df = df.sort("event_time", "loan_id")

    X = df.select([*NUMERIC, *CATEGORICAL]).to_pandas()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return Stream(
        name=f"freddie_sflld_h{horizon}",
        X=X,
        y=df["y"].to_numpy().astype(float),
        event_time=df["event_time"].to_numpy().astype("datetime64[ns]"),
        label_time=df["label_time"].to_numpy().astype("datetime64[ns]"),
        meta={"horizon_months": horizon, "label_delay": "natural",
              "target": f"90+ days delinquent or credit loss within {horizon} months"},
    )
