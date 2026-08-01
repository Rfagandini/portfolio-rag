"""
Generate a retrieval evaluation set from the match data itself.

WHY GENERATE INSTEAD OF HAND-WRITING
------------------------------------
The 80 questions in test_questions.py were written by hand against the
original 98-document corpus. They cannot measure the expansion: none of them
touch the 1990s, head-to-heads, career aggregates or tournament results.

Hand-labelling 150 more is slow and error-prone, and asking an LLM to invent
questions produces plausible-sounding items with unverifiable gold answers.
But the generated corpus was itself built from structured data — so we
already KNOW the ground truth. Deriving questions from the same CSVs gives
exact gold answers for free, and the expected document is known by
construction rather than guessed.

Each question records:
    id       — stable integer
    category — grouping tag for per-category reporting
    input    — the user-facing question
    expected — ground-truth answer string (for a later LLM-as-judge run)
    articles — expected document stems, INCLUDING the category prefix
               (e.g. "results__1998_Rome_Masters"). The prefix matters:
               career__Pete_Sampras and players_top50__Pete_Sampras share a
               key, and a career-stats question is only truly answered by the
               career document.

Only questions whose target document actually exists in docs/ are emitted,
so the set can never reference something the corpus does not contain.

Usage:
    python build_eval_questions.py
    python build_eval_questions.py --per-category 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fetch_corpus import sanitize_filename
from generate_match_docs import (
    DATA_DIR,
    DOCS_DIR,
    INCLUDE_LEVELS,
    LEVEL_NAMES,
    TOP50_PATH,
    is_team_event,
    is_walkover,
    load_matches,
)

OUTPUT_PATH = Path("eval_questions_generated.json")
DEFAULT_PER_CATEGORY = 30
SEED = 20260731  # fixed so the set is reproducible across runs


def existing_stems() -> set[str]:
    """Every document stem currently in docs/, e.g. 'results__1995_Wimbledon'."""
    return {p.stem for p in DOCS_DIR.glob("*.txt")}


def build(matches: list[dict], players: list[str], per_category: int) -> list[dict]:
    rng = random.Random(SEED)
    stems = existing_stems()
    out: list[dict] = []

    def emit(category: str, question: str, answer: str, stem: str) -> bool:
        """Add a question only if its target document exists."""
        if stem not in stems:
            return False
        out.append({
            "id": len(out) + 1,
            "category": category,
            "input": question,
            "expected": answer,
            "articles": [stem],
        })
        return True

    # --- 1. Tournament champions (spread across eras and tiers) -----------
    finals = [
        m for m in matches
        if m["round"] == "F" and not is_team_event(m) and not is_walkover(m)
    ]
    rng.shuffle(finals)
    n = 0
    for m in finals:
        if n >= per_category:
            break
        stem = f"results__{sanitize_filename(str(m['year']) + ' ' + m['tourney_name'])}"
        if emit("tournament_winner",
                f"Who won the {m['year']} {m['tourney_name']}?",
                m["winner_name"], stem):
            n += 1

    # --- 2. Final scorelines (exact-string retrieval, BM25's speciality) --
    rng.shuffle(finals)
    n = 0
    for m in finals:
        if n >= per_category:
            break
        score = (m.get("score") or "").strip()
        if not score:
            continue
        stem = f"results__{sanitize_filename(str(m['year']) + ' ' + m['tourney_name'])}"
        if emit("tournament_score",
                f"What was the score of the {m['year']} {m['tourney_name']} final?",
                f"{m['winner_name']} beat {m['loser_name']} {score}", stem):
            n += 1

    # --- 3. Head-to-head records -----------------------------------------
    pairs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in matches:
        pairs[tuple(sorted((m["winner_name"].strip(), m["loser_name"].strip())))].append(m)

    rivalries = [(k, v) for k, v in pairs.items() if len(v) >= 8]
    rng.shuffle(rivalries)
    n = 0
    for (a, b), group in rivalries:
        if n >= per_category:
            break
        played = [m for m in group if not is_walkover(m)]
        a_wins = sum(1 for m in played if m["winner_name"].strip() == a)
        b_wins = len(played) - a_wins
        leader, hi, lo = (a, a_wins, b_wins) if a_wins >= b_wins else (b, b_wins, a_wins)
        stem = f"h2h__{sanitize_filename(a)}_vs_{sanitize_filename(b)}"
        if emit("head_to_head",
                f"What is the head-to-head record between {a} and {b}?",
                f"{leader} leads {hi}-{lo}", stem):
            n += 1

    # --- 4. Career titles and best ranking --------------------------------
    titles: dict[str, int] = defaultdict(int)
    best: dict[str, int] = {}
    for m in matches:
        if m["round"] == "F" and not is_team_event(m) and not is_walkover(m):
            titles[m["winner_name"].strip()] += 1
        for side in ("winner", "loser"):
            name = m[f"{side}_name"].strip()
            r = (m.get(f"{side}_rank") or "").strip()
            if r.isdigit() and (name not in best or int(r) < best[name]):
                best[name] = int(r)

    sampled = players[:]
    rng.shuffle(sampled)
    n = 0
    for p in sampled:
        if n >= per_category:
            break
        if titles.get(p, 0) < 1:
            continue
        stem = f"career__{sanitize_filename(p)}"
        if emit("career_titles",
                f"How many tour-level singles titles did {p} win?",
                f"{titles[p]} titles", stem):
            n += 1

    rng.shuffle(sampled)
    n = 0
    for p in sampled:
        if n >= per_category:
            break
        if p not in best:
            continue
        stem = f"career__{sanitize_filename(p)}"
        # "since 1990" is load-bearing: the dataset starts in 1990, so for
        # players who peaked earlier the unqualified question has a different
        # correct answer than the data can support (Wilander was world 1, but
        # only ranks 10 within this window).
        if emit("career_rank",
                f"What was the best ATP singles ranking reached by {p} since 1990?",
                f"world number {best[p]} (within 1990 onwards)", stem):
            n += 1

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY)
    args = parser.parse_args()

    matches = load_matches(DATA_DIR, set(INCLUDE_LEVELS))
    with open(TOP50_PATH, encoding="utf-8") as f:
        players = json.load(f)["players"]

    questions = build(matches, players, args.per_category)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    counts: dict[str, int] = defaultdict(int)
    for q in questions:
        counts[q["category"]] += 1

    print("=" * 60)
    print(f"Generated {len(questions)} questions -> {OUTPUT_PATH}")
    for cat, c in sorted(counts.items()):
        print(f"  {cat:<20} {c}")
    print("=" * 60)
    print("\nSamples:")
    for q in questions[:3] + questions[-2:]:
        print(f"  [{q['category']}] {q['input']}")
        print(f"      -> {q['expected']}   (doc: {q['articles'][0]})")

    print("\nNext: python evaluate_generated.py")


if __name__ == "__main__":
    main()
