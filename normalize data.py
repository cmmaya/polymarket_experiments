"""
clean_normalize.py

Reads all CSVs in ./fetched_data and writes cleaned/renamed CSVs to ./data.

Input CSV schema (expected):
  timestamp, price_yes, price_no, volume

Operations:
1) Cut rows until the first non-null price_yes is found (keep from that row onward).
2) Round price_yes and price_no to 3 decimals.
3) Rename output files: replace leading "politics-" with "<category>-"
   where category is determined from the provided slug→category lists.

Notes:
- If a slug appears in multiple categories (it does: will-hezbollah-to-disarm-before-in-2025),
  this script resolves conflicts by priority order (configurable below).
- If a file's slug is unknown (not in your lists), it is kept under "unknown-<slug>.csv"
  and reported in the summary.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------
INPUT_DIR = Path("fetched_data")
OUTPUT_DIR = Path("data")

# Priority order used when a slug is listed in multiple categories
CATEGORY_PRIORITY = [
    "geopolitics",
    "world",
    "politics",
    "tech",
    "culture",
    "science",
]

CATEGORIES: Dict[str, List[str]] = {
    "politics": [
        "will-venezuela-give-the-us-oil-by-january-31",
        "will-saudi-arabia-strike-yemen-by-january-31",
        "israel-strike-on-the-west-bank-by-december-31-586",
        "nicols-maduro-seen-in-public-by-january-5",
        "us-anti-cartel-operation-on-foreign-soil-by-december-31",
        "will-alberta-vote-for-independence-in-2025",
        "will-hezbollah-to-disarm-before-in-2025",
        "us-strike-on-syria-by-529",
        "will-trump-try-to-fire-powell-in-2025",
        "us-forces-in-iran-in-2025",
        "israel-strikes-iran-by-january-23-2026",
    ],
    "tech": [
        "will-apple-release-airtag-2-by-june-30",
        "x-relaunches-vine-in-2025",
        "nothing-ever-happens-ai-edition",
        "us-enacts-ai-safety-bill-in-2025",
        "will-apple-release-a-new-product-line-in-2025",
        "will-a-chinese-ai-model-become-1-in-2025",
        "will-apple-release-homepod-mini-successor-by-december-31",
        "musk-no-longer-richest-person-in-the-world-by-end-of-year",
        "openai-announces-it-has-achieved-agi-in-2025",
        "musk-out-as-tesla-ceo-in-2025",
        "fanduel-launches-prediction-markets-with-cme-by-end-of-2025",
        "will-polymarket-us-go-live-in-2025",
    ],
    "culture": [
        "will-zelenskyy-wear-a-suit-and-tie-at-the-world-economic-forum",
        "will-an-fbi-top-ten-most-wanted-fugitive-be-captured-by-june-2026",
        "jd-vance-baby-boy-or-girl",
        "will-trump-and-machado-share-the-nobel-peace-prize",
        "will-cz-return-to-binance-by-december-31",
        "will-trump-remove-his-rob-reiner-post",
        "epstein-or-maxwell-confirmed-mossad-opperatives-in-2025",
        "trump-confirmed-to-be-satoshi-by-december-31",
        "gta-vi-released-in-2025",
        "kamala-harris-divorce-in-2025",
        "major-meteor-strike-10kt-in-2025",
        "trump-divorce-in-2025",
    ],
    "world": [
        "north-korea-missile-launch-by-january-31-661",
        "will-russia-announce-a-christmas-truce",
        "maduro-travels-outside-venezuela-by-march-31",
        "us-agrees-to-give-ukraine-security-guarantee-in-2025",
        "will-north-and-south-korea-engage-in-direct-talks-in-2025",
        "will-china-unban-bitcoin-in-2025",
        "german-government-files-appeal-to-ban-afd-in-2025",
        "will-the-iranian-regime-fall-in-2025",
        "iran-nuclear-test-in-2025",
        "will-ukraine-agree-to-cede-territory-to-russia",
    ],
    "science": [
        "will-spacex-rescue-the-stranded-chinese-astronauts",
    ],
    "geopolitics": [
        "will-hezbollah-to-disarm-before-in-2025",
        "russia-strike-on-kyiv-municipality-by-december-31-684",
        "foreign-intervention-in-gaza-in-2025",
        "will-japanese-pm-apologize-for-china-comments-in-2025",
        "will-iran-withdraw-from-the-npt-in-2025",
        "maduro-in-us-custody-by-december-31",
        "will-the-us-invade-afghanistan-in-2025",
        "will-russia-capture-slovainsk-in-2025",
        "china-x-japan-sever-diplomatic-relations-in-2025",
        "will-trump-pardon-maduro-by-december-31",
        "will-netanyahu-be-pardoned-in-2025",
    ],
}


# -----------------------------
# Helpers
# -----------------------------
def build_slug_to_category(
    categories: Dict[str, List[str]],
    priority: List[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
      - slug_to_category: chosen category per slug (resolves duplicates via priority)
      - slug_to_all_categories: all categories each slug appears in (for reporting)
    """
    slug_to_all: Dict[str, List[str]] = {}
    for cat, slugs in categories.items():
        for s in slugs:
            slug_to_all.setdefault(s, []).append(cat)

    prio_rank = {c: i for i, c in enumerate(priority)}
    slug_to_category: Dict[str, str] = {}

    for slug, cats in slug_to_all.items():
        # choose category with lowest priority index
        chosen = sorted(cats, key=lambda c: prio_rank.get(c, 10**9))[0]
        slug_to_category[slug] = chosen

    return slug_to_category, slug_to_all


def parse_slug_from_filename(filename: str) -> str:
    """
    Expects filenames like:
      politics-<slug>.csv
    Returns <slug>. If it doesn't match, returns filename stem without extension.
    """
    stem = Path(filename).stem
    m = re.match(r"^[^-]+-(.+)$", stem)
    return m.group(1) if m else stem


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "price_yes", "price_no", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Ensure numeric columns are numeric (coerce bad values to NaN)
    for col in ["price_yes", "price_no", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) Cut rows until first non-null price_yes
    first_valid_idx = df["price_yes"].first_valid_index()
    if first_valid_idx is None:
        # All price_yes are null -> return empty df (still with columns)
        return df.iloc[0:0].copy()

    df = df.loc[first_valid_idx:].copy()

    # 2) Round prices to 3 decimals
    df["price_yes"] = df["price_yes"].round(3)
    df["price_no"] = df["price_no"].round(3)

    return df


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_dirs()

    slug_to_category, slug_to_all_categories = build_slug_to_category(
        CATEGORIES, CATEGORY_PRIORITY
    )

    # Reporting structures
    unknown_files: List[str] = []
    failed_files: List[Tuple[str, str]] = []
    written_files: List[str] = []
    duplicate_slugs = {s: cats for s, cats in slug_to_all_categories.items() if len(cats) > 1}

    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {INPUT_DIR.resolve()}")
        return

    print(f"Found {len(csv_files)} CSV files in {INPUT_DIR}/")

    for fp in csv_files:
        try:
            slug = parse_slug_from_filename(fp.name)
            category = slug_to_category.get(slug, "unknown")
            if category == "unknown":
                unknown_files.append(fp.name)

            df = pd.read_csv(fp)

            df_clean = clean_df(df)

            out_name = f"{category}-{slug}.csv"
            out_path = OUTPUT_DIR / out_name
            df_clean.to_csv(out_path, index=False)

            written_files.append(out_name)
            print(f"OK  {fp.name}  ->  {out_name}  (rows: {len(df_clean)})")

        except Exception as e:
            failed_files.append((fp.name, str(e)))
            print(f"ERR {fp.name}: {e}")

    # Summary
    print("\n==================== SUMMARY ====================")
    print(f"Written: {len(written_files)} file(s) to {OUTPUT_DIR}/")
    if unknown_files:
        print(f"Unknown slug files (written as unknown-*.csv): {len(unknown_files)}")
        for f in unknown_files:
            print(f"  - {f}")

    if duplicate_slugs:
        print(f"\nSlugs listed in multiple categories: {len(duplicate_slugs)}")
        for slug, cats in sorted(duplicate_slugs.items()):
            chosen = slug_to_category.get(slug)
            print(f"  - {slug}: {cats} -> chosen='{chosen}' (priority={CATEGORY_PRIORITY})")

    if failed_files:
        print(f"\nFailures: {len(failed_files)}")
        for name, err in failed_files:
            print(f"  - {name}: {err}")
    else:
        print("\nFailures: 0")


if __name__ == "__main__":
    main()
