# src/similarity.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.data_loader import repo_root_from_cwd, load_catalog, load_features


@dataclass(frozen=True)
class SimilarityConfig:
    # Cosine similarity on row-normalized features
    metric: str = "cosine"
    n_neighbors: int = 11  # includes self by default
    include_self: bool = False


def _ensure_title_id_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame index is title_id. Works whether title_id is already index or a column.
    """
    if df.index.name == "title_id":
        return df
    if "title_id" in df.columns:
        return df.set_index("title_id", drop=False)
    # If parquet was saved with index=True, title_id might be the unnamed index already
    # In that case, we keep it but name it for clarity.
    out = df.copy()
    out.index.name = out.index.name or "title_id"
    return out


def build_similarity_index(
    X: pd.DataFrame,
    cfg: SimilarityConfig = SimilarityConfig(),
) -> tuple[NearestNeighbors, np.ndarray, pd.Index]:
    """
    Build a nearest-neighbour index over rows of X.
    Returns (knn_model, X_mat, title_ids).
    """
    X = _ensure_title_id_index(X)

    # float for sklearn
    X_mat = X.values.astype(np.float32)

    # For cosine, normalizing makes distances stable and matches your clustering choice.
    if cfg.metric == "cosine":
        X_mat = normalize(X_mat, norm="l2")

    knn = NearestNeighbors(metric=cfg.metric, algorithm="brute", n_neighbors=cfg.n_neighbors)
    knn.fit(X_mat)

    return knn, X_mat, X.index


def find_similar_titles(
    repo_root: Optional[Path] = None,
    *,
    title_id: str,
    k: int = 10,
    include_catalog_cols: tuple[str, ...] = ("title", "platform", "type", "release_year"),
    cfg: SimilarityConfig = SimilarityConfig(),
) -> pd.DataFrame:
    """
    Return the k most similar titles to `title_id` using cosine similarity over X_core.

    Output columns:
      - neighbor_title_id
      - cosine_similarity
      - (optional) catalog fields: title/platform/type/release_year
    """
    repo_root = repo_root or repo_root_from_cwd()

    X = load_features(repo_root, "X_core")
    X = _ensure_title_id_index(X)

    if title_id not in X.index:
        raise KeyError(f"title_id not found in X_core index: {title_id}")

    # n_neighbors: k + 1 to allow dropping self robustly
    local_cfg = SimilarityConfig(metric=cfg.metric, n_neighbors=max(k + 1, 2), include_self=cfg.include_self)
    knn, X_mat, ids = build_similarity_index(X, local_cfg)

    row_idx = ids.get_loc(title_id)
    distances, indices = knn.kneighbors(X_mat[row_idx].reshape(1, -1), n_neighbors=local_cfg.n_neighbors)

    distances = distances.ravel()
    indices = indices.ravel()

    # Convert cosine distance -> cosine similarity
    if local_cfg.metric == "cosine":
        sims = 1.0 - distances
    else:
        # For other metrics, interpret as "closer is better"; keep as negative distance score.
        sims = -distances

    neighbors = pd.DataFrame(
        {
            "neighbor_title_id": ids[indices].astype(str),
            "score": sims,
        }
    )

    # Drop self if present
    if not local_cfg.include_self:
        neighbors = neighbors[neighbors["neighbor_title_id"] != str(title_id)]

    neighbors = neighbors.head(k).reset_index(drop=True)

    # Rename score to something explicit for cosine
    if local_cfg.metric == "cosine":
        neighbors = neighbors.rename(columns={"score": "cosine_similarity"})
    else:
        neighbors = neighbors.rename(columns={"score": f"{local_cfg.metric}_score"})

    # Optional enrichment from catalog
    if include_catalog_cols:
        catalog = load_catalog(repo_root)
        catalog = _ensure_title_id_index(catalog)

        cols = [c for c in include_catalog_cols if c in catalog.columns]
        if cols:
            enrich = catalog.loc[neighbors["neighbor_title_id"], cols].reset_index().rename(columns={"title_id": "neighbor_title_id"})
            neighbors = neighbors.merge(enrich, on="neighbor_title_id", how="left")

    return neighbors


def main() -> None:
    """
    Minimal CLI usage:
      python -m src.similarity <title_id> [k]
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.similarity <title_id> [k]")
        raise SystemExit(2)

    tid = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 10

    out = find_similar_titles(title_id=tid, k=k)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()