from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

@dataclass(frozen=True)
class ClusterConfig:
    method: str = "kmeans"  # "kmeans" or "agglom"
    k: int = 12
    random_state: int = 42
    use_cosine: bool = True  # normalize rows then euclidean ~= cosine-ish

def fit_clusters(X: pd.DataFrame, cfg: ClusterConfig) -> dict:
    """
    Returns dict with:
      labels: pd.Series indexed like X
      model: fitted model
      silhouette: float (if computable)
    """
    X_mat = X.values.astype(float)

    if cfg.use_cosine:
        X_mat = normalize(X_mat, norm="l2")

    if cfg.method == "kmeans":
        model = KMeans(n_clusters=cfg.k, random_state=cfg.random_state, n_init="auto")
        labels = model.fit_predict(X_mat)
    elif cfg.method == "agglom":
        model = AgglomerativeClustering(n_clusters=cfg.k, linkage="ward")
        labels = model.fit_predict(X_mat)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    # silhouette can be slow on 20k rows; sample for speed
    sil = None
    try:
        n = len(X_mat)
        if n > 5000:
            idx = np.random.RandomState(cfg.random_state).choice(n, 5000, replace=False)
            sil = float(silhouette_score(X_mat[idx], labels[idx]))
        else:
            sil = float(silhouette_score(X_mat, labels))
    except Exception:
        sil = None

    return {
        "labels": pd.Series(labels, index=X.index, name="cluster"),
        "model": model,
        "silhouette": sil,
        "config": cfg,
    }

def cluster_profiles(X: pd.DataFrame, labels: pd.Series, top_n: int = 12) -> pd.DataFrame:
    """
    Mean feature activation per cluster, returns top features per cluster.
    """
    df = X.copy()
    df["cluster"] = labels.values
    means = df.groupby("cluster").mean(numeric_only=True)

    rows = []
    for c in means.index:
        top = means.loc[c].sort_values(ascending=False).head(top_n)
        rows.append(pd.DataFrame({"cluster": c, "feature": top.index, "mean": top.values}))
    return pd.concat(rows, ignore_index=True)
