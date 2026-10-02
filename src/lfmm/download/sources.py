"""Download job lists for each dataset.

Each function returns the files to fetch; `_http.download_all` does the work.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from folktables.load_acs import _STATE_CODES, state_list

from ._http import Job

# --- Folktables (ACS PUMS) ---------------------------------------------------
# ACS 2020 1-year PUMS was only released as experimental estimates and is not on
# the standard census.gov path, so it is skipped by default.
ACS_DEFAULT_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
ACS_BASE = "https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year"


def acs_jobs(raw: Path, years: list[int], states: list[str] | None = None) -> list[Job]:
    """Per-state person files, laid out as folktables expects:
    raw/folktables/{year}/1-Year/csv_p{state}.zip"""
    states = states or state_list
    jobs = []
    for year in years:
        base = raw / "folktables" / str(year) / "1-Year"
        for st in states:
            name = f"csv_p{st.lower()}.zip"
            jobs.append(Job("folktables_acs", f"{ACS_BASE.format(year=year)}/{name}", base / name))
        if year >= 2017:  # CSV data dictionaries are only published from 2017 on
            dict_url = (
                "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/"
                f"PUMS_Data_Dictionary_{year}.csv"
            )
            jobs.append(Job("folktables_acs", dict_url, base / "definition.csv"))
    return jobs


def acs_extract(raw: Path, years: list[int]) -> None:
    """Unzip each state's CSV next to its zip so folktables can load it offline.
    The zips stay in place as the checksummed originals."""
    for year in years:
        base = raw / "folktables" / str(year) / "1-Year"
        for zpath in sorted(base.glob("csv_p*.zip")):
            st = zpath.stem.removeprefix("csv_p").upper()
            if year >= 2017:
                csv_name = f"psam_p{_STATE_CODES[st]}.csv"
            else:
                csv_name = f"ss{str(year)[-2:]}p{st.lower()}.csv"
            if (base / csv_name).exists():
                continue
            with zipfile.ZipFile(zpath) as z:
                z.extract(csv_name, path=base)


# --- NYC TLC trip records ----------------------------------------------------
TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def tlc_jobs(raw: Path, start: str, end: str, service: str = "yellow") -> list[Job]:
    """Monthly parquet files from start to end inclusive (YYYY-MM)."""
    y, m = map(int, start.split("-"))
    y_end, m_end = map(int, end.split("-"))
    jobs = []
    while (y, m) <= (y_end, m_end):
        name = f"{service}_tripdata_{y}-{m:02d}.parquet"
        jobs.append(Job("nyc_tlc", f"{TLC_BASE}/{name}", raw / "nyc_tlc" / service / name))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    # Zone lookup maps PULocationID/DOLocationID to borough and zone.
    jobs.append(Job(
        "nyc_tlc",
        "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
        raw / "nyc_tlc" / "taxi_zone_lookup.csv",
    ))
    return jobs


# --- Freddie Mac Single-Family Loan-Level Dataset (manual download) ----------
# Needs a Clarity Data Intelligence login, so files are downloaded by hand from
# the SFLLD Data Download page into raw/freddie_mac/sflld/ and then registered.
# Free for non-commercial research; redistribution needs a license.
SFLLD_PORTAL = "https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld"


def sflld_local_jobs(raw: Path) -> list[Job]:
    folder = raw / "freddie_mac" / "sflld"
    files = sorted(folder.glob("sample_*.zip")) + sorted(folder.glob("historical_data_*.zip"))
    return [Job("freddie_mac_sflld", f"{SFLLD_PORTAL} (manual: {f.name})", f) for f in files]


# --- TabReD (manual download) ------------------------------------------------
# Preprocessed .tabred archives (gzipped tar, one dataset each) placed by hand in
# raw/tabred/preprocessed/. The official route is `uvx tabred download all`.
# Yandex-origin data is non-commercial research only; Kaggle-derived datasets
# follow their competition rules. Ship a loader only, never the data.
TABRED_SOURCE = "https://www.kaggle.com/datasets/irubachev/tabred"


def tabred_local_jobs(raw: Path) -> list[Job]:
    files = sorted((raw / "tabred" / "preprocessed").glob("*.tabred"))
    return [Job("tabred", f"{TABRED_SOURCE} (manual: {f.name})", f) for f in files]


# --- CDC BRFSS ---------------------------------------------------------------
# 2011 is the first year with the current weighting and cell-phone sample.
BRFSS_DEFAULT_YEARS = list(range(2011, 2025))
BRFSS_BASE = "https://www.cdc.gov/brfss/annual_data/{year}/files"


def brfss_jobs(raw: Path, years: list[int]) -> list[Job]:
    jobs = []
    for year in years:
        name = f"LLCP{year}XPT.zip"
        jobs.append(Job("brfss", f"{BRFSS_BASE.format(year=year)}/{name}", raw / "brfss" / name))
    return jobs
