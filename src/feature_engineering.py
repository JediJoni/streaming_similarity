# src/feature_engineering.py
from __future__ import annotations

from dataclasses import dataclass, is_dataclass, asdict
from typing import Iterable
from pathlib import Path
import pandas as pd
import numpy as np
import json

@dataclass(frozen=True)
class CoreFeatureConfig:
    top_genres: int = 30
    top_countries: int = 25
    year_bins: tuple[int, ...] = (0, 1980, 1990, 2000, 2010, 2020, 2030)  # inclusive-ish
    movie_mins_bins: tuple[int, ...] = (0, 60, 80, 100, 120, 150, 10_000)
    tv_seasons_bins: tuple[int, ...] = (0, 1, 2, 3, 5, 10, 100)
    include_n_cast: bool = True


def _top_k_from_lists(series: pd.Series, k: int) -> list[str]:
    """series contains lists. Return top-k most frequent tokens."""
    counts = {}
    for items in series:
        if items is None:
            continue
        try:
            if len(items) == 0:
                continue
        except TypeError:
            continue
        for x in items:
            counts[x] = counts.get(x, 0) + 1
    top = sorted(counts.items(), key=lambda t: t[1], reverse=True)[:k]
    return [t[0] for t in top]


def _as_list(x) -> list:
    # Handles list, tuple, numpy array, pyarrow list scalars, NaN/None
    if x is None:
        return []
    # pandas NA / numpy nan
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()

    # pyarrow sometimes returns objects that behave like sequences
    try:
        return list(x)
    except Exception:
        return []


def _multi_hot(series: pd.Series, vocab: list[str], prefix: str, other_label: str | None = "Other") -> pd.DataFrame:
    """
    Build multi-hot columns for list-valued series.
    Unknown tokens can be optionally bucketed into prefix+Other.
    """
    vocab_set = set(vocab)
    data = {f"{prefix}{v}": np.zeros(len(series), dtype=np.int8) for v in vocab}
    if other_label is not None:
        data[f"{prefix}{other_label}"] = np.zeros(len(series), dtype=np.int8)

    for i, items in enumerate(series):
        items = _as_list(items)
        if len(items) == 0:
            continue
        hit_other = False
        for x in items:
            if x in vocab_set:
                data[f"{prefix}{x}"][i] = 1
            else:
                hit_other = True
        if other_label is not None and hit_other:
            data[f"{prefix}{other_label}"][i] = 1

    return pd.DataFrame(data, index=series.index)


def _bin_series(values: pd.Series, bins: tuple[int, ...], prefix: str) -> pd.DataFrame:
    labels = [f"{prefix}{bins[i]}_{bins[i+1]}" for i in range(len(bins) - 1)]
    cat = pd.cut(values, bins=list(bins), labels=labels, right=False, include_lowest=True)
    return pd.get_dummies(cat).astype(np.int8)


def build_core_feature_matrix(catalog: pd.DataFrame, cfg: CoreFeatureConfig = CoreFeatureConfig()) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      X_core: DataFrame indexed by title_id
      meta: dict with vocabularies used (genres/countries) for interpretability + reproducibility
    """
    df = catalog.copy()

    # Index by title_id for stable alignment
    df = df.set_index("title_id", drop=False)

    df["genres"] = df["genres"].apply(_as_list)
    df["countries"] = df["countries"].apply(_as_list)
    
    # if you use cast_list in core later, coerce it too
    if "cast_list" in df.columns:
        df["cast_list"] = df["cast_list"].apply(_as_list)

    # Top vocab selection across *all platforms*
    top_genres = _top_k_from_lists(df["genres"], cfg.top_genres)
    top_countries = _top_k_from_lists(df["countries"], cfg.top_countries)

    X_genres = _multi_hot(df["genres"], top_genres, prefix="Genre:")
    X_countries = _multi_hot(df["countries"], top_countries, prefix="Country:", other_label="Other")

    # Type one-hot
    X_type = pd.get_dummies(df["type"], prefix="Type").astype(np.int8)

    # Year bins
    year_num = pd.to_numeric(df["release_year"], errors="coerce")
    X_year = _bin_series(year_num.fillna(0), cfg.year_bins, prefix="Year:")

    # Duration bins: separate signals for movies vs TV
    mins = pd.to_numeric(df["duration_mins"], errors="coerce")
    seasons = pd.to_numeric(df["duration_seasons"], errors="coerce")

    X_mins = _bin_series(mins.fillna(0), cfg.movie_mins_bins, prefix="MovieMins:")
    X_seasons = _bin_series(seasons.fillna(0), cfg.tv_seasons_bins, prefix="TVSeasons:")

    # Mask bins so they don't pollute the wrong type (optional but helps interpretability)
    is_movie = (df["type"].str.lower() == "movie").astype(np.int8).values
    is_tv = (df["type"].str.lower().str.contains("tv")).astype(np.int8).values

    X_mins = X_mins.mul(is_movie, axis=0)
    X_seasons = X_seasons.mul(is_tv, axis=0)

    parts = [X_genres, X_countries, X_type, X_year, X_mins, X_seasons]

    if cfg.include_n_cast:
        # simple numeric, binned is often cleaner than raw
        n_cast = pd.to_numeric(df["n_cast"], errors="coerce").fillna(0)
        X_n_cast = _bin_series(n_cast, (0, 1, 3, 6, 10, 20, 10_000), prefix="NCast:")
        parts.append(X_n_cast)

    X_core = pd.concat(parts, axis=1).astype(np.int8)

    meta = {
        "top_genres": top_genres,
        "top_countries": top_countries,
        "config": asdict(cfg),
    }
    return X_core, meta

###

@dataclass(frozen=True)
class CastFeatureConfig:
    top_actors: int = 200
    include_n_cast: bool = True


def build_cast_feature_matrix(catalog: pd.DataFrame, cfg: CastFeatureConfig = CastFeatureConfig()) -> tuple[pd.DataFrame, dict]:
    """
    Build sparse actor multi-hot features.
    Use this as a *secondary* analysis (can be noisy / fame-driven).
    """
    df = catalog.copy().set_index("title_id", drop=False)

    df["cast_list"] = df["cast_list"].apply(_as_list)

    top_actors = _top_k_from_lists(df["cast_list"], cfg.top_actors)
    X_cast = _multi_hot(df["cast_list"], top_actors, prefix="Actor:", other_label=None)

    parts = [X_cast]

    if cfg.include_n_cast:
        n_cast = pd.to_numeric(df["n_cast"], errors="coerce").fillna(0)
        X_n_cast = _bin_series(n_cast, (0, 1, 3, 6, 10, 20, 10_000), prefix="NCast:")
        parts.append(X_n_cast)

    X_cast = pd.concat(parts, axis=1).astype(np.int8)

    meta = {
        "top_actors": top_actors,
        "config": asdict(cfg),
    }
    return X_cast, meta

# Helper save/load utilities functions for feature bundles (matrix + meta) 
# to enable reproducibility & interpretability of features used in downstream analyses
def _jsonable(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (list, dict, str, int, float, bool)) or x is None:
        return x
    return str(x)

def save_feature_bundle(X: pd.DataFrame, meta: dict, out_dir: Path, name: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = out_dir / f"{name}.parquet"
    meta_path = out_dir / f"{name}_meta.json"

    X.to_parquet(X_path, index=True)

    meta_clean = {k: _jsonable(v) for k, v in meta.items()}
    with open(meta_path, "w") as f:
        json.dump(meta_clean, f, indent=2)