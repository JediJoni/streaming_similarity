# src/data/build_catalog.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import pandas as pd


PLATFORM_FILES = {
    "Amazon Prime": "amazon_prime_titles.csv",
    "Disney+": "disney_plus_titles.csv",
    "Netflix": "netflix_titles.csv",
}

REQUIRED_RAW_COLS = [
    "title", "type", "release_year", "duration", "listed_in", "country", "cast"
]


def load_raw_csvs(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load platform CSVs from data/raw into a dict[platform] -> DataFrame."""
    raw_dir = Path(raw_dir)
    dfs: dict[str, pd.DataFrame] = {}

    for platform, filename in PLATFORM_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file for {platform}: {path}")

        df = pd.read_csv(path)
        df["platform"] = platform
        dfs[platform] = df

    return dfs


# ---------- parsing helpers ----------

def _split_multi(value: object) -> list[str]:
    """Split comma-separated strings to normalized list[str]."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    # normalize spacing/casing lightly (keep original words, just trim)
    return [p for p in parts if p]


_DURATION_RE = re.compile(r"^\s*(\d+)\s*(min|mins|minute|minutes|season|seasons)\s*$", re.I)

def _parse_duration(duration: object) -> tuple[int | None, int | None]:
    """
    Return (duration_mins, duration_seasons) from raw duration field.
    Examples:
      "90 min" -> (90, None)
      "2 Seasons" -> (None, 2)
    """
    if duration is None or (isinstance(duration, float) and pd.isna(duration)):
        return None, None
    s = str(duration).strip()
    if not s:
        return None, None

    m = _DURATION_RE.match(s)
    if not m:
        return None, None

    n = int(m.group(1))
    unit = m.group(2).lower()
    if "season" in unit:
        return None, n
    return n, None


def _stable_title_id(platform: str, title: str, release_year: int | None, type_: str) -> str:
    """Deterministic ID from key fields; stable across runs."""
    key = f"{platform}|{title}|{release_year}|{type_}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def build_catalog(platform_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build canonical catalog with:
      required raw cols + processed cols (genres, countries, cast_list, duration_mins, duration_seasons, title_id)
    """
    frames = []
    for platform, df in platform_dfs.items():
        df = df.copy()

        # Standardize column names if needed (datasets usually already match)
        # Ensure required columns exist
        missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{platform} missing columns: {missing}")

        # Enforce types
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

        # Parse duration
        parsed = df["duration"].apply(_parse_duration)
        df["duration_mins"] = [x[0] for x in parsed]
        df["duration_seasons"] = [x[1] for x in parsed]

        # Process multi-label text fields
        df["genres"] = df["listed_in"].apply(_split_multi)
        df["countries"] = df["country"].apply(_split_multi)
        df["cast_list"] = df["cast"].apply(_split_multi)

        # Simple derived stats (useful for interpretability)
        df["n_genres"] = df["genres"].apply(len)
        df["n_countries"] = df["countries"].apply(len)
        df["n_cast"] = df["cast_list"].apply(len)

        # Title ID
        df["title_id"] = [
            _stable_title_id(platform, t, int(y) if pd.notna(y) else None, ty)
            for t, y, ty in zip(df["title"], df["release_year"], df["type"])
        ]

        # Keep canonical columns only (and in a stable order)
        keep = [
            "title_id", "platform", "title", "type", "release_year",
            "duration_mins", "duration_seasons",
            "listed_in", "country", "cast",
            "genres", "countries", "cast_list",
            "n_genres", "n_countries", "n_cast",
        ]
        frames.append(df[keep])

    catalog = pd.concat(frames, ignore_index=True)

    # Defensive: remove exact duplicates by title_id if any (rare but possible)
    catalog = catalog.drop_duplicates(subset=["title_id"]).reset_index(drop=True)

    return catalog


def save_catalog_parquet(catalog: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_path, index=False)