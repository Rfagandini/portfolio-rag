"""
Fetch Wikipedia articles for the tennis RAG corpus.

Saves each article as a .txt file in docs/. Skips already-fetched articles,
so the script is idempotent and safe to re-run if it fails partway.

Two sources of titles:
  1. The hand-curated lists below (the original Phase 4 corpus).
  2. data/top50_players.json — the 485 players who reached the ATP top 50
     between 1990 and 2026, derived from match data by build_player_list.py.

Usage:
    pip install wikipedia-api
    python fetch_match_data.py && python build_player_list.py
    python fetch_corpus.py
"""

import json
import re
import sys
import time
from pathlib import Path

import wikipediaapi

# Player names carry accents (Čilić, Ríos, Söderling). On Windows the console
# and any redirected stdout default to cp1252, which cannot encode them, so
# printing progress would crash the run. Force UTF-8 and never fail on output.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --- CONFIG ---

USER_AGENT = "portfolio-rag-tennis/1.0 (contact: rfagandini@gmail.com)"
DOCS_DIR = Path("docs")
MIN_ARTICLE_LENGTH = 500  # skip disambiguation
REQUEST_DELAY = 0.3  # seconds between requests

TOP50_PATH = Path("data") / "top50_players.json"


# --- CORPUS DEFINITION ---

PLAYERS_HISTORICAL = [
    "Rod Laver", "Björn Borg", "John McEnroe", "Jimmy Connors",
    "Ivan Lendl", "Andre Agassi", "Pete Sampras", "Stefan Edberg",
    "Boris Becker", "Mats Wilander",
]

PLAYERS_2000S = [
    "Marat Safin", "Marcelo Ríos", "Gustavo Kuerten", "Lleyton Hewitt",
    "Juan Carlos Ferrero", "Andy Roddick", "David Nalbandian",
    "James Blake (tennis)", "Fernando González", "Nikolay Davydenko",
    "Tim Henman", "Tommy Haas",
]

PLAYERS_2010S = [
    "Andy Murray", "Stan Wawrinka", "Juan Martín del Potro",
    "Tomáš Berdych", "Jo-Wilfried Tsonga", "Richard Gasquet",
    "David Ferrer", "Kei Nishikori", "Milos Raonic", "Marin Čilić",
    "Grigor Dimitrov", "Gaël Monfils",
]

PLAYERS_BIG3 = ["Roger Federer", "Rafael Nadal", "Novak Djokovic"]

PLAYERS_CURRENT = [
    "Daniil Medvedev", "Alexander Zverev", "Stefanos Tsitsipas",
    "Dominic Thiem", "Andrey Rublev", "Casper Ruud",
    "Matteo Berrettini", "Hubert Hurkacz", "Félix Auger-Aliassime",
    "Taylor Fritz", "Holger Rune", "Carlos Alcaraz", "Jannik Sinner",
    "Frances Tiafoe", "Ben Shelton", "Lorenzo Musetti", "Fabio Fognini",
]

PLAYERS_WOMEN = [
    "Serena Williams", "Steffi Graf", "Martina Navratilova",
    "Iga Świątek", "Aryna Sabalenka",
]

# Grand Slam men's singles for 2020-2024 (5 years × 4 slams = 20)
_SLAMS = ["Australian Open", "French Open", "Wimbledon Championships", "US Open"]
TOURNAMENTS = [
    f"{year} {slam} – Men's singles"
    for year in range(2020, 2025)
    for slam in _SLAMS
] + [
    f"{year} ATP Finals" for year in range(2020, 2025)
]

GENERAL = [
    "Tennis", "History of tennis", "Open Era", "Grand Slam (tennis)",
    "Career Grand Slam", "ATP Tour", "ATP rankings", "ATP Finals",
    "Davis Cup", "Laver Cup", "Tennis scoring system", "Tennis court",
    "Clay court", "Grass court", "Hard court",
]

# Names where Wikipedia's title cannot be derived from the ATP name.
# Populated by hand from the "unresolved" list this script prints.
#
# The pattern in all three: the ATP feed writes names ASCII-folded, and while
# Wikipedia redirects the bare folded name to a disambiguation page, there is
# no redirect for the folded name PLUS the "(tennis)" suffix. So the automatic
# probe for "Alex Calatrava (tennis)" misses "Álex Calatrava (tennis)".
PLAYER_ALIASES: dict[str, str] = {
    "Alex Calatrava": "Álex Calatrava (tennis)",
    "Tomas Carbonell": "Tomás Carbonell (tennis)",
    "Miloslav Mecir Sr.": "Miloslav Mečíř",
}


def load_top50_players() -> list[str]:
    """
    Load the data-derived player list written by build_player_list.py.

    Returns [] with a warning if the file is missing, so the hand-curated
    corpus can still be fetched on its own.
    """
    if not TOP50_PATH.exists():
        print(f"NOTE: {TOP50_PATH} not found — run build_player_list.py first.")
        print("      Continuing with the hand-curated lists only.\n")
        return []
    with open(TOP50_PATH, encoding="utf-8") as f:
        return json.load(f)["players"]


# Tag each title with a category so filenames preserve grouping.
CATEGORIES = [
    ("players_historical", PLAYERS_HISTORICAL),
    ("players_2000s", PLAYERS_2000S),
    ("players_2010s", PLAYERS_2010S),
    ("players_big3", PLAYERS_BIG3),
    ("players_current", PLAYERS_CURRENT),
    ("players_women", PLAYERS_WOMEN),
    ("tournaments", TOURNAMENTS),
    ("general", GENERAL),
    ("players_top50", load_top50_players()),
]


# --- HELPERS ---

def sanitize_filename(title: str) -> str:
    """Make a safe filename from a Wikipedia title."""
    safe = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE)
    safe = re.sub(r"\s+", "_", safe).strip("_")
    return safe[:120]


def is_tennis_page(page: wikipediaapi.WikipediaPage) -> bool:
    """
    True if the page's categories mention tennis.

    Guards against name collisions: "Alexander Volkov" is a cosmonaut, a
    footballer AND a tennis player, and the bare title resolves to the
    disambiguation page. Costs one extra API call per check.
    """
    return "tennis" in " ".join(page.categories.keys()).lower()


def with_retry(fn, attempts: int = 4, base_delay: float = 1.0):
    """
    Call `fn`, retrying with exponential backoff.

    Wikipedia rate-limits bursts by returning an HTML error page where the
    client expects JSON, which surfaces as "Expecting value: line 1 column 1".
    That is transient, so back off and try again rather than recording the
    player as missing.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def resolve_page(
    wiki: wikipediaapi.Wikipedia, title: str
) -> tuple[wikipediaapi.WikipediaPage, str] | None:
    """
    Find the Wikipedia page for a player, handling ambiguous names.

    Tries "{title} (tennis)" first: when that disambiguated form exists it is
    unambiguously the right person, so it is accepted without a categories
    lookup. Only the bare-title fallback needs the tennis-category check,
    since that is where collisions live ("Alexander Volkov" is also a
    cosmonaut and a footballer, and resolves to a disambiguation page).

    Keeping the categories call off the common path matters: it is a second
    network round trip per candidate, and doing it for all 485 players is
    what triggers rate limiting.

    Returns (page, resolved_title) or None if nothing matched.
    """
    time.sleep(REQUEST_DELAY)
    disambiguated = f"{title} (tennis)"
    page = with_retry(lambda: wiki.page(disambiguated))
    if page.exists() and len(page.text) >= MIN_ARTICLE_LENGTH:
        return page, disambiguated

    time.sleep(REQUEST_DELAY)
    page = with_retry(lambda: wiki.page(title))
    if not page.exists() or len(page.text) < MIN_ARTICLE_LENGTH:
        return None
    if not with_retry(lambda: is_tennis_page(page)):
        return None
    return page, title


def fetch_article(wiki: wikipediaapi.Wikipedia, title: str) -> str | None:
    """Return article text, or None if not found / too short."""
    page = wiki.page(title)
    if not page.exists():
        return None
    text = page.text
    if len(text) < MIN_ARTICLE_LENGTH:
        return None
    return text


def save_article(filepath: Path, title: str, text: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{text}")


# --- MAIN ---

def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)

    # Warn about legacy PDFs left over from the previous corpus.
    legacy_pdfs = list(DOCS_DIR.glob("*.pdf"))
    if legacy_pdfs:
        print(f"NOTE: {len(legacy_pdfs)} old .pdf file(s) still in docs/")
        for p in legacy_pdfs:
            print(f"  - {p.name}")
        print("Delete them manually after this script finishes successfully.\n")

    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")

    # A player can appear in both a hand-curated list and the top-50 list
    # (Federer is in players_big3, Agassi in players_historical, ...). Fetching
    # them twice would put duplicate chunks in the corpus, which inflates the
    # index and lets one player crowd out others at retrieval time. Key on the
    # sanitized title, ignoring the category prefix, to catch those.
    existing_keys = {
        p.stem.split("__", 1)[1] if "__" in p.stem else p.stem
        for p in DOCS_DIR.glob("*.txt")
    }

    total = sum(len(titles) for _, titles in CATEGORIES)
    success, skipped, deduped, failed = 0, 0, 0, []
    counter = 0

    for category, titles in CATEGORIES:
        print(f"\n=== {category.upper()} ({len(titles)} articles) ===")
        for title in titles:
            counter += 1
            key = sanitize_filename(title)
            filename = f"{category}__{key}.txt"
            filepath = DOCS_DIR / filename

            if filepath.exists():
                print(f"  [{counter}/{total}] SKIP (cached): {title}")
                skipped += 1
                continue

            if key in existing_keys:
                print(f"  [{counter}/{total}] SKIP (already in corpus): {title}")
                deduped += 1
                continue

            print(f"  [{counter}/{total}] Fetching: {title}...", end=" ", flush=True)

            # Player names are ambiguous; the curated lists are already exact
            # Wikipedia titles, so only the derived list needs resolution.
            resolve = category == "players_top50"
            lookup = PLAYER_ALIASES.get(title, title)

            try:
                if resolve:
                    found = resolve_page(wiki, lookup)
                    if found is None:
                        print("UNRESOLVED")
                        failed.append(title)
                        continue
                    page, resolved_title = found
                    text = page.text
                else:
                    resolved_title = lookup
                    text = with_retry(lambda: fetch_article(wiki, lookup))
            except Exception as e:
                print(f"ERROR: {e}")
                failed.append(title)
                continue

            if text is None:
                print("NOT FOUND")
                failed.append(title)
                continue

            # Store under the ATP name so build_player_list.py, the generated
            # match docs and the eval harness all agree on one key per player.
            save_article(filepath, title, text)
            existing_keys.add(key)
            note = "" if resolved_title == title else f" via '{resolved_title}'"
            print(f"OK ({len(text):,} chars){note}")
            success += 1
            time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {success} fetched, {skipped} cached, "
          f"{deduped} deduped, {len(failed)} failed")
    print(f"Corpus now holds {len(list(DOCS_DIR.glob('*.txt')))} documents")
    print("=" * 60)
    if failed:
        print("\nUnresolved titles — add a Wikipedia title for each to")
        print("PLAYER_ALIASES at the top of this script, then re-run:")
        for t in failed:
            print(f'    "{t}": "",')


if __name__ == "__main__":
    main()
