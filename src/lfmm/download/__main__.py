"""Download raw benchmark data.

    python -m lfmm.download acs
    python -m lfmm.download tlc --start 2019-01 --end 2021-12
    python -m lfmm.download brfss
    python -m lfmm.download all
    python -m lfmm.download sflld   # register hand-downloaded Freddie Mac files
    python -m lfmm.download tabred  # register hand-downloaded TabReD archives

Files go to $LFMM_DATA_DIR/raw and are recorded in $LFMM_DATA_DIR/manifest.csv.
Rerunning skips files already present, so it is safe to resume.
"""

from __future__ import annotations

import argparse
import sys

from ..config import data_dir, raw_dir
from . import sources
from ._http import Manifest, download_all, register_local


def _years(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return default
    years: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            years.extend(range(a, b + 1))
        else:
            years.append(int(part))
    return years


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lfmm.download", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["acs", "tlc", "brfss", "sflld", "tabred", "all"])
    p.add_argument("--data-dir", help="overrides $LFMM_DATA_DIR")
    p.add_argument("--years", help="e.g. 2014-2019,2021 (acs and brfss)")
    p.add_argument("--states", help="comma-separated state codes for acs, e.g. CA,TX")
    p.add_argument("--start", default="2019-01", help="first TLC month (YYYY-MM)")
    p.add_argument("--end", default="2021-12", help="last TLC month (YYYY-MM)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    root = data_dir(args.data_dir)
    raw = raw_dir(args.data_dir)
    manifest = Manifest(root)
    which = ["acs", "tlc", "brfss", "sflld", "tabred"] if args.dataset == "all" else [args.dataset]

    failures = []
    for name in which:
        print(f"\n=== {name} ===")
        if name == "acs":
            years = _years(args.years if args.dataset == "acs" else None, sources.ACS_DEFAULT_YEARS)
            states = args.states.upper().split(",") if args.states else None
            failures += download_all(sources.acs_jobs(raw, years, states), manifest, args.workers)
            print("extracting state CSVs for folktables...")
            sources.acs_extract(raw, years)
        elif name == "tlc":
            failures += download_all(sources.tlc_jobs(raw, args.start, args.end), manifest, args.workers)
        elif name == "brfss":
            years = _years(args.years if args.dataset == "brfss" else None, sources.BRFSS_DEFAULT_YEARS)
            failures += download_all(sources.brfss_jobs(raw, years), manifest, args.workers)
        elif name == "sflld":
            jobs = sources.sflld_local_jobs(raw)
            print(f"{len(jobs)} SFLLD zip(s) found")
            failures += register_local(jobs, manifest)
        elif name == "tabred":
            jobs = sources.tabred_local_jobs(raw)
            print(f"{len(jobs)} TabReD archive(s) found")
            failures += register_local(jobs, manifest)

    if failures:
        print(f"\n{len(failures)} file(s) failed; rerun the same command to retry:")
        for job, err in failures:
            print(f"  {job.url}: {err}")
        return 1
    print("\nAll files downloaded and recorded in", manifest.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
