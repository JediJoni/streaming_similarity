# Streaming Content Similarity & Genre Clustering

Build an interpretable, metadata-driven view of **content similarity** across streaming platforms.
This repo turns a cleaned title catalog into feature matrices (genres, countries, type, year, duration, cast size)
and runs clustering to reveal coarse “content archetypes”.

## What this does

1. **Build a unified catalog** from raw platform exports  
2. **Engineer interpretable features** (multi-hot + binned numeric signals)
3. **Cluster titles** (k-means on cosine distance)
4. **Save artifacts** so results are reproducible

## Quickstart

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline
