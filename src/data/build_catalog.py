from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------- helpers ----------
_SPLIT_RE = re.compile(r"\s*,\s*")


def _split_list(s: object) -> list[str]:
    """Split comma-separated strings into a clean list; return [] for missing."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in _SPLIT_RE.split(s) if x.strip()]


def _norm_token(x: str) -> str:
    """Light normalization for tokens."""
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x


def _make_title_id(platform: str, title: str, release_year: int | float, type_: str) -> str:
    key = f"{platform}|{title}|{int(release_year) if pd.notna(release_year) else 'NA'}|{type_}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _parse_duration(type_series: pd.Series, duration_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Returns (duration_mins, duration_seasons)
    - For Movies: minutes parsed from '90 min'
    - For TV Shows: seasons parsed from '2 Seasons'
    """
    duration_mins = pd.Series([np.nan] * len(duration_series), index=duration_series.index, dtype="float64")
    duration_seasons = pd.Series([np.nan] * len(duration_series), index=duration_series.index, dtype="float64")

    s = duration_series.fillna("").astype(str).str.lower().str.strip()
    is_movie = type_series.fillna("").astype(str).str.lower().eq("movie")
    is_tv = type_series.fillna("").astype(str).str.lower().str.contains("tv")

    mins = s.str.extract(r"(\d+)\s*min")[0]
    seasons = s.str.extract(r"(\d+)\s*season")[0]

    duration_mins.loc[is_movie] = pd.to_numeric(mins.loc[is_movie], errors="coerce")
    duration_seasons.loc[is_tv] = pd.to_numeric(seasons.loc[is_tv], errors="coerce")

    return duration_mins, duration_seasons


# ---------- main build ----------
REQUIRED_COLS = {
    "type",
    "title",
    "release_year",
    "duration",
    "listed_in",
    "country",
    "cast",
}

CANON_COLS = [
    "title_id",
    "platform",
    "title",
    "type",
    "release_year",
    "duration",
    "duration_mins",
    "duration_seasons",
    "listed_in",
    "country",
    "cast",
    # processed
    "genres",
    "countries",
    "cast_list",
    "n_genres",
    "n_countries",
    "n_cast",
]


def standardize_platform_df(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """Standardize a raw platform df into canonical schema + parsed lists."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{platform}: missing required columns: {sorted(missing)}")

    out = df.copy()

    # basic fields
    out["platform"] = platform
    out["title"] = out["title"].astype(str).str.strip()
    out["type"] = out["type"].astype(str).str.strip()
    out["release_year"] = pd.to_numeric(out["release_year"], errors="coerce")
    out["duration"] = out["duration"]

    # parse duration
    dur_mins, dur_seasons = _parse_duration(out["type"], out["duration"])
    out["duration_mins"] = dur_mins
    out["duration_seasons"] = dur_seasons

    # parsed lists
    out["genres"] = out["listed_in"].apply(_split_list).apply(lambda xs: [_norm_token(x) for x in xs])
    out["countries"] = out["country"].apply(_split_list).apply(lambda xs: [_norm_token(x) for x in xs])
    out["cast_list"] = out["cast"].apply(_split_list).apply(lambda xs: [_norm_token(x) for x in xs])

    out["n_genres"] = out["genres"].apply(len)
    out["n_countries"] = out["countries"].apply(len)
    out["n_cast"] = out["cast_list"].apply(len)

    # title_id
    out["title_id"] = [
        _make_title_id(platform, t, y, ty) for t, y, ty in zip(out["title"], out["release_year"], out["type"])
    ]

    # keep only canonical columns
    out = out[CANON_COLS]

    # drop obvious junk rows
    out = out.dropna(subset=["title", "type"]).reset_index(drop=True)
    return out


def build_catalog(platform_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine standardized dfs into a single canonical catalog."""
    parts = []
    for platform, df in platform_dfs.items():
        parts.append(standardize_platform_df(df, platform))
    catalog = pd.concat(parts, ignore_index=True)

    # optional: dedupe within platform on title_id (should already be stable)
    catalog = catalog.drop_duplicates(subset=["platform", "title_id"]).reset_index(drop=True)
    return catalog


def load_raw_csvs(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    mapping = {
        "Netflix": raw_dir / "netflix_titles.csv",
        "Amazon Prime": raw_dir / "amazon_prime_titles.csv",
        "Disney+": raw_dir / "disney_plus_titles.csv",
    }
    dfs = {}
    for platform, path in mapping.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file for {platform}: {path}")
        dfs[platform] = pd.read_csv(path)
    return dfs