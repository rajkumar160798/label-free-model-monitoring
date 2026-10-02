"""Resumable, checksummed HTTP downloads recorded in a manifest."""

from __future__ import annotations

import csv
import hashlib
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

CHUNK = 1 << 20
MANIFEST_FIELDS = ["dataset", "path", "url", "downloaded_at", "bytes", "sha256"]
USER_AGENT = "lfmm-benchmark-downloader/0.1 (+research use)"


@dataclass(frozen=True)
class Job:
    dataset: str
    url: str
    dest: Path  # absolute path under the data dir


class Manifest:
    """CSV of every raw file: where it came from and its sha256."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.path = data_dir / "manifest.csv"
        self._lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.rows[row["path"]] = row

    def rel(self, dest: Path) -> str:
        return dest.relative_to(self.data_dir).as_posix()

    def has(self, dest: Path) -> bool:
        row = self.rows.get(self.rel(dest))
        return row is not None and dest.exists() and dest.stat().st_size == int(row["bytes"])

    def record(self, job: Job, size: int, sha256: str) -> None:
        row = {
            "dataset": job.dataset,
            "path": self.rel(job.dest),
            "url": job.url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": size,
            "sha256": sha256,
        }
        with self._lock:
            self.rows[row["path"]] = row
            tmp = self.path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(sorted(self.rows.values(), key=lambda r: r["path"]))
            tmp.replace(self.path)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _validate(path: Path) -> None:
    """Reject HTML error pages and truncated files saved under a data name."""
    suffix = path.suffix.lower()
    if suffix == ".zip" and not zipfile.is_zipfile(path):
        raise ValueError(f"{path.name} is not a valid zip file")
    if suffix == ".parquet":
        with path.open("rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            tail = f.read(4)
        if head != b"PAR1" or tail != b"PAR1":
            raise ValueError(f"{path.name} is not a valid parquet file")


def _fetch(job: Job, session: requests.Session, retries: int = 4) -> int:
    """Download job.url to job.dest via a .part file, resuming if possible."""
    job.dest.parent.mkdir(parents=True, exist_ok=True)
    part = job.dest.with_name(job.dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with session.get(job.url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # .part already complete
                    break
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored Range
                    have = 0
                expected = r.headers.get("Content-Length")
                expected = int(expected) + have if expected else None
                with part.open("ab" if have else "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
            size = part.stat().st_size
            if expected is not None and size != expected:
                raise IOError(f"incomplete: {size} of {expected} bytes")
            break
        except (requests.RequestException, IOError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == retries or (status is not None and 400 <= status < 500):
                raise
            wait = 2**attempt
            print(f"  retry {attempt}/{retries - 1} for {job.dest.name} in {wait}s: {exc}")
            time.sleep(wait)
    _validate(part)
    part.replace(job.dest)
    return job.dest.stat().st_size


def url_exists(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30, headers={"User-Agent": USER_AGENT})
        return r.status_code == 200
    except requests.RequestException:
        return False


def register_local(jobs: list[Job], manifest: Manifest) -> list[tuple[Job, str]]:
    """Validate and record files that were downloaded by hand (behind a login)."""
    failures: list[tuple[Job, str]] = []
    for job in jobs:
        if manifest.has(job.dest):
            continue
        try:
            _validate(job.dest)
            if job.dest.suffix.lower() == ".zip":
                with zipfile.ZipFile(job.dest) as z:
                    if (bad := z.testzip()) is not None:
                        raise ValueError(f"corrupt member {bad}")
            elif job.dest.suffix.lower() == ".tabred":  # gzipped tar
                with tarfile.open(job.dest, "r:gz") as t:
                    for member in t:  # reading every member checks the gzip CRC
                        if member.isfile():
                            t.extractfile(member).read()
            manifest.record(job, job.dest.stat().st_size, sha256_of(job.dest))
            print(f"recorded {manifest.rel(job.dest)}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append((job, str(exc)))
            print(f"FAILED {job.dest.name}: {exc}")
    return failures


def download_all(jobs: list[Job], manifest: Manifest, workers: int = 4) -> list[tuple[Job, str]]:
    """Run jobs in parallel. Returns (job, error message) for each failure."""
    todo = []
    for job in jobs:
        if manifest.has(job.dest):
            continue
        if job.dest.exists():  # downloaded before the manifest knew about it
            manifest.record(job, job.dest.stat().st_size, sha256_of(job.dest))
            continue
        todo.append(job)

    print(f"{len(jobs) - len(todo)} of {len(jobs)} files already present; downloading {len(todo)}")
    failures: list[tuple[Job, str]] = []
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, job, session): job for job in todo}
        for fut in as_completed(futures):
            job = futures[fut]
            done += 1
            try:
                size = fut.result()
                manifest.record(job, size, sha256_of(job.dest))
                print(f"[{done}/{len(todo)}] {manifest.rel(job.dest)} ({size / 1e6:.1f} MB)")
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failures.append((job, str(exc)))
                print(f"[{done}/{len(todo)}] FAILED {job.url}: {exc}")
    return failures
