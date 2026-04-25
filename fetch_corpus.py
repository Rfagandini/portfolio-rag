"""
Fetch Wikipedia articles for the tennis RAG corpus.

Saves each article as a .txt file in docs/. Skips already-fetched articles,
so the script is idempotent and safe to re-run if it fails partway.

Usage:
    pip install wikipedia-api
    python fetch_corpus.py

After successful run, delete the old PDFs in docs/ manually.
"""

import re
import time
from pathlib import Path

import wikipediaapi


# --- CONFIG ---

USER_AGENT = "portfolio-rag-tennis/1.0 (contact: rfagandini@gmail.com)"
DOCS_DIR = Path("docs")
MIN_ARTICLE_LENGTH = 500  # skip disambiguation
REQUEST_DELAY = 0.3  # seconds between requests


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
]


# --- HELPERS ---

def sanitize_filename(title: str) -> str:
    """Make a safe filename from a Wikipedia title."""
    safe = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE)
    safe = re.sub(r"\s+", "_", safe).strip("_")
    return safe[:120]


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

    total = sum(len(titles) for _, titles in CATEGORIES)
    success, skipped, failed = 0, 0, []
    counter = 0

    for category, titles in CATEGORIES:
        print(f"\n=== {category.upper()} ({len(titles)} articles) ===")
        for title in titles:
            counter += 1
            filename = f"{category}__{sanitize_filename(title)}.txt"
            filepath = DOCS_DIR / filename

            if filepath.exists():
                print(f"  [{counter}/{total}] SKIP (cached): {title}")
                skipped += 1
                continue

            print(f"  [{counter}/{total}] Fetching: {title}...", end=" ", flush=True)
            try:
                text = fetch_article(wiki, title)
            except Exception as e:
                print(f"ERROR: {e}")
                failed.append(title)
                continue

            if text is None:
                print("NOT FOUND")
                failed.append(title)
                continue

            save_article(filepath, title, text)
            print(f"OK ({len(text):,} chars)")
            success += 1
            time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {success} fetched, {skipped} skipped, {len(failed)} failed")
    print("=" * 60)
    if failed:
        print("\nFailed titles (edit the script and re-run to retry):")
        for t in failed:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
