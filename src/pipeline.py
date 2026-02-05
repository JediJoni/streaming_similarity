# src/pipeline.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd

from src.data_loader import repo_root_from_cwd
from src.data.build_catalog import load_raw_csvs, build_catalog, save_catalog_parquet
from src.feature_engineering import (
    CoreFeatureConfig,
    CastFeatureConfig,
    build_core_feature_matrix,
    build_cast_feature_matrix,
    save_feature_bundle,
)
from src.clustering import ClusterConfig, fit_clusters, cluster_profiles


def run(
    k: int = 12,
    use_cosine: bool = True,
    top_genres: int = 30,
    top_countries: int = 25,
    top_actors: int = 200,
    build_cast: bool = False,
    random_state: int = 42,
) -> Path:
    repo = repo_root_from_cwd()
    processed = repo / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # -------- 0) Run config (auditability) --------
    run_cfg = {
        "k": k,
        "use_cosine": use_cosine,
        "top_genres": top_genres,
        "top_countries": top_countries,
        "top_actors": top_actors,
        "build_cast": build_cast,
        "random_state": random_state,
    }
    (processed / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    # -------- 1) Build catalog --------
    raw_dir = repo / "data" / "raw"
    catalog_path = processed / "catalog.parquet"

    platform_dfs = load_raw_csvs(raw_dir)
    catalog = build_catalog(platform_dfs)

    # Optional: enforce stable ordering by title_id for reproducibility
    if "title_id" in catalog.columns:
        catalog = catalog.sort_values("title_id").reset_index(drop=True)

    save_catalog_parquet(catalog, catalog_path)

    # -------- 2) Features (Core) --------
    core_cfg = CoreFeatureConfig(top_genres=top_genres, top_countries=top_countries)
    X_core, core_meta = build_core_feature_matrix(catalog, core_cfg)

    # enforce deterministic ordering
    X_core = X_core.sort_index()
    X_core = X_core.reindex(sorted(X_core.columns), axis=1)

    save_feature_bundle(X_core, core_meta, processed, name="X_core")

    # -------- 2b) Features (Cast) optional --------
    if build_cast:
        cast_cfg = CastFeatureConfig(top_actors=top_actors)
        X_cast, cast_meta = build_cast_feature_matrix(catalog, cast_cfg)
        X_cast = X_cast.sort_index()
        X_cast = X_cast.reindex(sorted(X_cast.columns), axis=1)
        save_feature_bundle(X_cast, cast_meta, processed, name="X_cast")

    # -------- 3) Clustering --------
    clus_cfg = ClusterConfig(method="kmeans", k=k, use_cosine=use_cosine, random_state=random_state)
    result = fit_clusters(X_core, clus_cfg)
    labels = result["labels"]

    # alignment safety
    if not labels.index.equals(X_core.index):
        labels = labels.reindex(X_core.index)
        if labels.isna().any():
            raise ValueError("Cluster labels misaligned with X_core index after reindexing.")

    labels_path = processed / f"clusters_kmeans_k{k}.parquet"
    labels.to_frame("cluster").to_parquet(labels_path, index=True)

    # -------- 4) Summaries --------
    summary = cluster_profiles(X_core, labels, top_n=12)
    summary_path = processed / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)

    return processed


def main() -> None:
    out_dir = run()
    print(f"✅ Pipeline complete. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
