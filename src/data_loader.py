# src/data_loader.py
from pathlib import Path
import pandas as pd
import json

def repo_root_from_cwd() -> Path:
    p = Path.cwd().resolve()
    try:
        return next(x for x in [p, *p.parents] if (x / "src").exists())
    except StopIteration:
        raise RuntimeError(f"Could not find repo root from cwd={p} (expected a 'src/' directory).")

def load_catalog(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data" / "processed" / "catalog.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Catalog parquet not found at: {path}")
    return pd.read_parquet(path)

def load_features(repo_root: Path, name: str) -> pd.DataFrame:
    path = repo_root / "data" / "processed" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature parquet not found at: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Feature parquet is empty (0 bytes): {path}. Rebuild it from 03_feature_engineering.ipynb.")
    return pd.read_parquet(path)

def load_feature_meta(repo_root: Path, name: str) -> dict:
    path = repo_root / "data" / "processed" / f"{name}_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Feature meta JSON not found at: {path}")
    return json.loads(path.read_text())

def save_features(repo_root: Path, df: pd.DataFrame, name: str) -> Path:
    out = repo_root / "data" / "processed" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=True)
    return out

def save_feature_meta(repo_root: Path, meta: dict, name: str) -> Path:
    out = repo_root / "data" / "processed" / f"{name}_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    return out