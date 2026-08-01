"""
Derive the list of players to fetch Wikipedia biographies for.

The TML match CSVs carry each player's ATP ranking at the time of every match
(`winner_rank` / `loser_rank`). There is no separate rankings file, so we
reconstruct "who mattered" by tracking each player's BEST (lowest-numbered)
rank observed in each season, then keeping everyone who broke the cutoff.

This is a slight undercount versus true year-end rankings: a player only
appears if they actually played a tour-level match while ranked that high.
In practice that is the right filter anyway — a top-50 player who never
appears in a match is not someone the RAG corpus needs.

Outputs data/top50_players.json:
    {
      "cutoff": 50,
      "seasons": [1990, ..., 2026],
      "players": ["Aaron Krickstein", ...],       # alphabetical
      "best_rank": {"Pete Sampras": 1, ...},      # career best in range
      "seasons_in_top": {"Pete Sampras": 15, ...} # seasons at or above cutoff
    }

Usage:
    python build_player_list.py
    python build_player_list.py --cutoff 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# --- CONFIG ---

DATA_DIR = Path("data") / "matches"
OUTPUT_PATH = Path("data") / "top50_players.json"

DEFAULT_CUTOFF = 50


# --- CORE ---

def collect_best_ranks(data_dir: Path) -> dict[int, dict[str, int]]:
    """
    Scan every season CSV and return {year: {player_name: best_rank_that_year}}.

    Both sides of every match are inspected. Ranks arrive as strings and may be
    empty (unranked players, or missing historical data), so non-numeric values
    are skipped rather than coerced.
    """
    best: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(lambda: sys.maxsize))

    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSVs in {data_dir}. Run `python fetch_match_data.py` first."
        )

    for path in csv_paths:
        year = int(path.stem)
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                for side in ("winner", "loser"):
                    name = (row.get(f"{side}_name") or "").strip()
                    rank = (row.get(f"{side}_rank") or "").strip()
                    if not name or not rank.isdigit():
                        continue
                    value = int(rank)
                    if value < best[year][name]:
                        best[year][name] = value

    return best


def select_players(
    best_by_year: dict[int, dict[str, int]], cutoff: int
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """
    Reduce the per-year table to the set of players who ever hit `cutoff`.

    Returns (sorted_names, career_best_rank, seasons_in_top).
    """
    career_best: dict[str, int] = {}
    seasons_in_top: dict[str, int] = defaultdict(int)

    for year, players in best_by_year.items():
        for name, rank in players.items():
            if rank <= cutoff:
                seasons_in_top[name] += 1
            if name not in career_best or rank < career_best[name]:
                career_best[name] = rank

    selected = sorted(name for name, rank in career_best.items() if rank <= cutoff)
    return (
        selected,
        {name: career_best[name] for name in selected},
        {name: seasons_in_top[name] for name in selected},
    )


# --- MAIN ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff", type=int, default=DEFAULT_CUTOFF,
        help=f"Keep players who reached this rank or better (default {DEFAULT_CUTOFF})",
    )
    args = parser.parse_args()

    best_by_year = collect_best_ranks(DATA_DIR)
    players, career_best, seasons_in_top = select_players(best_by_year, args.cutoff)
    seasons = sorted(best_by_year)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cutoff": args.cutoff,
                "seasons": seasons,
                "players": players,
                "best_rank": career_best,
                "seasons_in_top": seasons_in_top,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 60)
    print(f"Seasons scanned : {seasons[0]}-{seasons[-1]} ({len(seasons)})")
    print(f"Cutoff          : top {args.cutoff}")
    print(f"Players selected: {len(players)}")
    print(f"Written to      : {OUTPUT_PATH}")
    print("=" * 60)

    # Sanity signal: the most durable players should be recognisable names.
    longest = sorted(players, key=lambda n: -seasons_in_top[n])[:10]
    print(f"\nMost seasons in the top {args.cutoff}:")
    for name in longest:
        print(f"  {name:<28} {seasons_in_top[name]:>2} seasons, best rank {career_best[name]}")

    print("\nNext: python fetch_corpus.py")


if __name__ == "__main__":
    main()
