"""
Generate corpus documents from the TML match CSVs.

WHY THIS EXISTS
---------------
Wikipedia cannot supply tournament results. `wikipedia-api` strips wiki
tables, and tournament draws *are* tables, so scraping gives you stubs:

    Pete Sampras                      -> 37,897 chars
    2023 Wimbledon - Men's singles    ->  2,392 chars
    1995 Wimbledon - Men's singles    ->  1,116 chars

The match CSVs, by contrast, hold all 115,243 matches from 1990-2026 with
score, round, surface and seeding. We turn that structured data into prose
the retriever can actually match against.

THE ONE RULE: EVERY LINE MUST STAND ALONE
-----------------------------------------
Chunks get split at ~800 characters, and a chunk carries no memory of the
document it came from. So this does NOT work:

    # 1995 Wimbledon
    Final: Sampras beat Becker 6-7(5) 6-2 6-4 6-2      <- BAD
    SF: Becker beat Martin 7-6 6-4 6-4                 <- BAD

A chunk starting at line 2 has lost the year and the tournament forever. A
user asking "who won Wimbledon in 1995" cannot retrieve it, and worse, the
line is indistinguishable from every other "SF: X beat Y" in the corpus.

Instead, every line repeats its own context:

    At the 1995 Wimbledon Championships (Grass, Grand Slam), in the Final,
    Pete Sampras beat Boris Becker 6-7(5) 6-2 6-4 6-2.

Now any chunk boundary is harmless. This costs disk space and buys retrieval
correctness -- the trade is heavily worth it.

OUTPUT
------
    docs/results__{year}_{tournament}.txt   one per tournament edition
    docs/career__{Player}.txt               one per top-50 player
    docs/h2h__{PlayerA}_vs_{PlayerB}.txt    pairs with >= MIN_H2H_MEETINGS

Filenames follow the existing `{category}__{key}.txt` convention so that
`article_from_source()` in evaluate_retrieval.py keeps working unchanged.

Usage:
    python generate_match_docs.py
    python generate_match_docs.py --levels G M F     # slams/masters/finals only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Reuse the corpus naming rule rather than re-implementing it. If these two
# ever disagree, evaluate_retrieval.article_from_source() silently stops
# matching generated docs against expected article titles.
from fetch_corpus import sanitize_filename

# --- CONFIG ---

DATA_DIR = Path("data") / "matches"
DOCS_DIR = Path("docs")
TOP50_PATH = Path("data") / "top50_players.json"

# Which tournament tiers to write results documents for.
#   G = Grand Slam    M = Masters 1000    F = Tour Finals
#   500 / 250 = ATP 500 / 250            D = Davis Cup    O = Olympics
#
# All tour-level events are included by default. If evaluate_retrieval.py
# shows precision dropping, the ATP 250s are the first thing to cut: they add
# ~48,000 matches whose text is near-identical in shape, which is exactly the
# kind of lexical near-duplication that confuses BM25 and dense retrieval
# alike. Changing this one constant is the whole rollback.
INCLUDE_LEVELS = {"G", "M", "F", "500", "250", "A", "D", "O"}

MIN_H2H_MEETINGS = 5  # below this, a head-to-head doc is noise

# Filename prefixes this script owns. Everything matching them is deleted and
# rewritten on each run, so nothing else may use these prefixes.
# NOTE: ingest.py keeps a matching copy to pick its chunking strategy —
# change both together.
GENERATED_PREFIXES = ("results__", "career__", "h2h__")

# Human-readable expansions used when writing sentences.
LEVEL_NAMES = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "F": "Tour Finals",
    "500": "ATP 500",
    "250": "ATP 250",
    "A": "ATP Tour",
    "D": "Davis Cup",
    "O": "Olympics",
}

ROUND_NAMES = {
    "F": "Final",
    "SF": "Semifinal",
    "QF": "Quarterfinal",
    "R16": "Round of 16",
    "R32": "Round of 32",
    "R64": "Round of 64",
    "R128": "Round of 128",
    "RR": "Round Robin",
    "BR": "Bronze Medal Match",
}

# Draw order, biggest stage first. Used to sort matches within a document so
# the final -- the thing people actually ask about -- lands in the first chunk.
ROUND_ORDER = ["F", "BR", "SF", "QF", "R16", "R32", "R64", "R128", "RR"]

# Team competitions. Winning the deciding rubber in one of these is NOT a
# singles title, and the ATP does not count it as such. Verified against
# Pete Sampras: including the 1993 World Team Cup gives 65 titles, excluding
# it gives the official 64.
TEAM_EVENTS = {"Davis Cup", "World Team Cup", "ATP Cup", "United Cup", "Laver Cup"}


def is_walkover(match: dict) -> bool:
    """
    True if the match was never played (opponent withdrew).

    The ATP excludes walkovers from head-to-head and win/loss records because
    no tennis was played. Verified against Federer-Nadal: counting the 2019
    Indian Wells walkover gives 24-17, excluding it gives the official 24-16.
    """
    return "W/O" in (match.get("score") or "").upper()


def is_team_event(match: dict) -> bool:
    """True if this match belongs to a team competition, not a singles draw."""
    return (
        match.get("tourney_level") == "D"
        or (match.get("tourney_name") or "").strip() in TEAM_EVENTS
    )


# --- LOADING ---

def load_matches(data_dir: Path, levels: set[str]) -> list[dict]:
    """
    Read every season CSV and return the matches worth writing about.

    Keeps the whole filtered set in memory (~115k dicts) because the career
    and head-to-head passes each need a full traversal.
    """
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSVs in {data_dir}. Run `python fetch_match_data.py` first."
        )

    matches: list[dict] = []
    for path in paths:
        year = int(path.stem)
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("tourney_level") or "").strip() not in levels:
                    continue
                if not row.get("winner_name") or not row.get("loser_name"):
                    continue
                row["year"] = year
                matches.append(row)

    return matches


# --- FORMATTING (worked example -- use this as the pattern) ---

def describe_match(match: dict) -> str:
    """
    Render one match as a single self-contained sentence.

    This one is written out in full as the reference for the rest: note that
    year, tournament, surface and level all appear in EVERY line, which is
    what makes the output chunk-safe.

    Example output:
        At the 1995 Wimbledon (Grass, Grand Slam), in the Final, Pete Sampras
        beat Boris Becker 6-7(5) 6-2 6-4 6-2.
    """
    level = LEVEL_NAMES.get(match["tourney_level"], match["tourney_level"])
    round_name = ROUND_NAMES.get(match["round"], match["round"])
    surface = match.get("surface") or "Unknown surface"
    score = (match.get("score") or "").strip() or "score unavailable"
    prefix = f"At the {match['year']} {match['tourney_name']} ({surface}, {level}), "

    # A walkover was never played, so "beat" would be actively wrong.
    if is_walkover(match):
        return (
            f"{prefix}in the {round_name}, {match['winner_name']} advanced when "
            f"{match['loser_name']} withdrew (walkover, no match played)."
        )

    return (
        f"{prefix}in the {round_name}, {match['winner_name']} beat "
        f"{match['loser_name']} {score}."
    )


# --- DOCUMENT BUILDERS ---

def round_sort_key(match: dict) -> int:
    """Position of a match's round in the draw, final first. Unknown -> last."""
    try:
        return ROUND_ORDER.index(match["round"])
    except ValueError:
        return len(ROUND_ORDER)


def build_results_docs(matches: list[dict]) -> dict[str, str]:
    """
    One document per tournament edition. Returns {filename: contents}.

    Matches are ordered final-first so the champion -- the thing people
    actually ask about -- lands in the document's first chunk.
    """
    editions: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for m in matches:
        name = (m.get("tourney_name") or "").strip()
        if name:
            editions[(m["year"], name)].append(m)

    docs: dict[str, str] = {}
    for (year, name), group in editions.items():
        group.sort(key=round_sort_key)

        level = LEVEL_NAMES.get(group[0]["tourney_level"], group[0]["tourney_level"])
        surface = group[0].get("surface") or "Unknown surface"

        lines = [f"# {year} {name}", ""]

        # Headline sentence. Round-robin events (Tour Finals) and team events
        # do not always have an "F" row, so this is best-effort.
        final = next((m for m in group if m["round"] == "F"), None)
        if final is not None:
            lines.append(
                f"{final['winner_name']} won the {year} {name} "
                f"({surface}, {level}), beating {final['loser_name']} "
                f"in the final {(final.get('score') or '').strip()}."
            )
            lines.append(
                f"The runner-up at the {year} {name} was {final['loser_name']}."
            )
        lines.append(
            f"The {year} {name} was a {level} event played on {surface.lower()}, "
            f"with {len(group)} matches recorded."
        )
        lines.append("")

        lines.extend(describe_match(m) for m in group)

        filename = f"results__{year}_{sanitize_filename(name)}.txt"
        docs[filename] = "\n".join(lines) + "\n"

    return docs


def build_career_docs(matches: list[dict], players: list[str]) -> dict[str, str]:
    """
    One career-summary document per top-50 player. Returns {filename: contents}.

    This exists because Wikipedia prose answers "who is X" but not "how did X
    do on clay in 2003". Aggregates answer a class of question the biographies
    structurally cannot.

    Every line repeats the player's name -- never a pronoun. A chunk whose
    subject is "He" has no retrievable subject at all.
    """
    wanted = set(players)

    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    by_surface: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    by_season: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    titles: dict[str, list[dict]] = defaultdict(list)
    finals_lost: dict[str, list[dict]] = defaultdict(list)
    best_rank: dict[str, tuple[int, int]] = {}  # player -> (rank, year)

    for m in matches:
        surface = m.get("surface") or "Unknown"
        walkover = is_walkover(m)
        team = is_team_event(m)

        for side, table in (("winner", wins), ("loser", losses)):
            name = m[f"{side}_name"].strip()
            if name not in wanted:
                continue

            # Rankings are observable even when no match was played.
            rank = (m.get(f"{side}_rank") or "").strip()
            if rank.isdigit():
                r = int(rank)
                if name not in best_rank or r < best_rank[name][0]:
                    best_rank[name] = (r, m["year"])

            # A walkover is not a win or a loss — nobody played.
            if walkover:
                continue

            table[name] += 1
            idx = 0 if side == "winner" else 1
            by_surface[name][surface][idx] += 1
            by_season[name][m["year"]][idx] += 1

            # Winning a team-competition rubber is not a singles title.
            if m["round"] == "F" and not team:
                (titles if side == "winner" else finals_lost)[name].append(m)

    docs: dict[str, str] = {}
    for player in players:
        w, l = wins[player], losses[player]
        if w + l == 0:
            continue

        seasons = sorted(by_season[player])
        lines = [f"# {player} — career record", ""]
        lines.append(
            f"{player} won {w} and lost {l} tour-level matches between "
            f"{seasons[0]} and {seasons[-1]}."
        )
        if player in best_rank:
            rank, year = best_rank[player]
            # The window qualifier is essential, not padding. This dataset
            # starts in 1990, so for players who peaked earlier (Wilander,
            # Lendl, Edberg, McEnroe) the observed best rank is NOT their
            # career best — Wilander shows 10 here but was actually world 1.
            # Stating the window keeps the sentence true instead of misleading.
            lines.append(
                f"Between {seasons[0]} and {seasons[-1]}, {player} reached a "
                f"best ATP ranking of number {rank}, in {year}. "
                f"(This dataset covers 1990 onwards, so players who peaked "
                f"before 1990 may have ranked higher earlier in their careers.)"
            )
        lines.append(
            f"{player} won {len(titles[player])} tour-level titles and lost "
            f"{len(finals_lost[player])} finals."
        )
        lines.append("")

        lines.append(f"## {player} — record by surface")
        for surface, (sw, sl) in sorted(by_surface[player].items()):
            lines.append(
                f"On {surface.lower()} courts, {player} won {sw} matches "
                f"and lost {sl}."
            )
        lines.append("")

        if titles[player]:
            lines.append(f"## {player} — titles")
            for m in sorted(titles[player], key=lambda x: x["year"]):
                level = LEVEL_NAMES.get(m["tourney_level"], m["tourney_level"])
                lines.append(
                    f"{player} won the {m['year']} {m['tourney_name']} "
                    f"({m.get('surface') or 'Unknown surface'}, {level}), "
                    f"beating {m['loser_name']} in the final "
                    f"{(m.get('score') or '').strip()}."
                )
            lines.append("")

        if finals_lost[player]:
            lines.append(f"## {player} — runner-up finishes")
            for m in sorted(finals_lost[player], key=lambda x: x["year"]):
                lines.append(
                    f"{player} lost the final of the {m['year']} "
                    f"{m['tourney_name']} to {m['winner_name']}."
                )
            lines.append("")

        lines.append(f"## {player} — season by season")
        for year in seasons:
            sw, sl = by_season[player][year]
            lines.append(f"In {year}, {player} won {sw} matches and lost {sl}.")

        docs[f"career__{sanitize_filename(player)}.txt"] = "\n".join(lines) + "\n"

    return docs


def build_h2h_docs(matches: list[dict], min_meetings: int) -> dict[str, str]:
    """
    One document per notable rivalry. Returns {filename: contents}.

    Pairs are keyed on a sorted name tuple so (Federer, Nadal) and
    (Nadal, Federer) land in the same bucket, and the filename is stable
    across runs. The min_meetings floor is what stops this from producing
    O(485^2) near-empty documents.
    """
    pairs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in matches:
        key = tuple(sorted((m["winner_name"].strip(), m["loser_name"].strip())))
        pairs[key].append(m)

    docs: dict[str, str] = {}
    for (a, b), group in pairs.items():
        if len(group) < min_meetings:
            continue

        # Walkovers are listed in the document but excluded from the record,
        # matching how the ATP reports head-to-heads.
        played = [m for m in group if not is_walkover(m)]
        a_wins = sum(1 for m in played if m["winner_name"].strip() == a)
        b_wins = len(played) - a_wins

        if a_wins > b_wins:
            lead = f"{a} leads {b} {a_wins}-{b_wins} in their head-to-head."
        elif b_wins > a_wins:
            lead = f"{b} leads {a} {b_wins}-{a_wins} in their head-to-head."
        else:
            lead = f"{a} and {b} are level {a_wins}-{b_wins} in their head-to-head."

        group.sort(key=lambda m: (m["year"], round_sort_key(m)))

        lines = [f"# {a} vs {b} — head-to-head", ""]
        lines.append(lead)
        lines.append(
            f"{a} and {b} played {len(played)} completed matches at tour level, "
            f"between {group[0]['year']} and {group[-1]['year']}."
        )
        if len(played) != len(group):
            lines.append(
                f"{a} and {b} also met {len(group) - len(played)} further "
                f"time(s) that ended in a walkover, which does not count "
                f"towards their head-to-head record."
            )
        lines.append("")
        lines.extend(describe_match(m) for m in group)

        filename = f"h2h__{sanitize_filename(a)}_vs_{sanitize_filename(b)}.txt"
        docs[filename] = "\n".join(lines) + "\n"

    return docs


# --- MAIN ---

def write_docs(docs: dict[str, str], docs_dir: Path) -> int:
    """Write {filename: contents} to disk as UTF-8. Returns the count."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in docs.items():
        with open(docs_dir / filename, "w", encoding="utf-8") as f:
            f.write(contents)
    return len(docs)


def clear_generated(docs_dir: Path) -> int:
    """
    Delete previously generated docs before regenerating.

    Without this, narrowing --levels leaves the old wider run's files behind
    and the corpus silently keeps documents the current config excludes.
    """
    removed = 0
    for prefix in GENERATED_PREFIXES:
        for path in docs_dir.glob(f"{prefix}*.txt"):
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels", nargs="+", default=sorted(INCLUDE_LEVELS),
        help="Tournament levels to include (default: all tour-level)",
    )
    parser.add_argument(
        "--min-h2h", type=int, default=MIN_H2H_MEETINGS,
        help=f"Minimum meetings for a head-to-head doc (default {MIN_H2H_MEETINGS})",
    )
    args = parser.parse_args()

    levels = set(args.levels)
    print(f"Levels included: {', '.join(sorted(levels))}")

    matches = load_matches(DATA_DIR, levels)
    print(f"Loaded {len(matches):,} matches")

    with open(TOP50_PATH, encoding="utf-8") as f:
        players = json.load(f)["players"]
    print(f"Loaded {len(players)} players\n")

    removed = clear_generated(DOCS_DIR)
    if removed:
        print(f"Removed {removed:,} previously generated docs\n")

    results = build_results_docs(matches)
    print(f"  results__ : {len(results):,} documents")

    careers = build_career_docs(matches, players)
    print(f"  career__  : {len(careers):,} documents")

    h2h = build_h2h_docs(matches, args.min_h2h)
    print(f"  h2h__     : {len(h2h):,} documents")

    all_docs = {**results, **careers, **h2h}
    written = write_docs(all_docs, DOCS_DIR)

    total_chars = sum(len(v) for v in all_docs.values())
    print("\n" + "=" * 60)
    print(f"Wrote {written:,} documents ({total_chars / 1e6:.1f} MB of text)")
    print(f"Corpus now holds {len(list(DOCS_DIR.glob('*.txt'))):,} documents")
    print("=" * 60)
    print("\nNext: python ingest.py")


if __name__ == "__main__":
    main()
