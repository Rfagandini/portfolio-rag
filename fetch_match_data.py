"""
Download ATP match data from the TML-Database.

Source: https://github.com/Tennismylife/TML-Database
Licence: CC BY-NC-SA 4.0 (attribution required, non-commercial).

This replaces the older JeffSackmann/tennis_atp repository, which no longer
exists. TML uses the same column format and is updated through 2026.

One CSV per season, saved to data/matches/{year}.csv. Each row is a single
match with winner/loser names, ranks, score, surface, round and tournament.

The script is idempotent: already-downloaded seasons are skipped, so it is
safe to re-run after a partial failure. The current season is always
re-downloaded because it grows as tournaments are played.

Usage:
    python fetch_match_data.py
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --- CONFIG ---

BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
DATA_DIR = Path("data") / "matches"

START_YEAR = 1990
END_YEAR = 2026

# The in-progress season is re-fetched every run; completed seasons never change.
CURRENT_SEASON = 2026

REQUEST_DELAY = 0.2  # seconds between downloads
MIN_CSV_BYTES = 1_000  # anything smaller is a 404 page or an empty season


# --- HELPERS ---

def download_season(year: int, dest: Path) -> int:
    """
    Download one season's CSV and write it to `dest`.

    Returns the number of bytes written. Raises urllib.error.HTTPError if the
    season is not available, or ValueError if the response is too small to be
    a real CSV (GitHub serves a short "404: Not Found" body for missing files).
    """
    url = f"{BASE_URL}/{year}.csv"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()

    if len(payload) < MIN_CSV_BYTES:
        raise ValueError(f"response too small ({len(payload)} bytes)")

    # Write to a temporary file first so an interrupted download never leaves a
    # truncated CSV behind that the skip-if-exists check would then trust.
    tmp = dest.with_suffix(".csv.part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return len(payload)


# --- MAIN ---

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    years = list(range(START_YEAR, END_YEAR + 1))
    downloaded, skipped, failed = 0, 0, []
    total_bytes = 0

    print(f"Downloading ATP match data {START_YEAR}-{END_YEAR} -> {DATA_DIR}\n")

    for i, year in enumerate(years, start=1):
        dest = DATA_DIR / f"{year}.csv"

        if dest.exists() and year != CURRENT_SEASON:
            print(f"  [{i}/{len(years)}] SKIP (cached): {year}")
            skipped += 1
            total_bytes += dest.stat().st_size
            continue

        label = "Refreshing" if dest.exists() else "Fetching"
        print(f"  [{i}/{len(years)}] {label}: {year}...", end=" ", flush=True)

        try:
            size = download_season(year, dest)
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            print(f"FAILED ({e})")
            failed.append(year)
            continue

        print(f"OK ({size:,} bytes)")
        downloaded += 1
        total_bytes += size
        time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {downloaded} downloaded, {skipped} cached, {len(failed)} failed")
    print(f"Total on disk: {total_bytes / 1e6:.1f} MB")
    print("=" * 60)

    if failed:
        print("\nFailed seasons (re-run to retry):")
        for year in failed:
            print(f"  - {year}")
        sys.exit(1)

    print("\nNext: python build_player_list.py")


if __name__ == "__main__":
    main()
