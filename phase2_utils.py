"""
Phase 2 helpers: query category prediction and category-aware reranking.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def merge_fields(row: pd.Series | dict) -> str:
    if isinstance(row, pd.Series):
        title = str(row.get("title", "") or "")
        text = str(row.get("text", "") or "")
        tags_list = row.get("tags", [])
    else:
        title = str(row.get("title", "") or "")
        text = str(row.get("text", "") or "")
        tags_list = row.get("tags", [])
    tags = " ".join(tags_list) if isinstance(tags_list, list) else ""
    combined = f"{title} {text} {tags}"
    return " ".join(combined.split())


def load_classifier_artifacts(
    base_dir: Path | str | None = None,
) -> tuple[Any, Any]:
    base = Path(base_dir) if base_dir else Path(".")
    with open(base / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(base / "tfidf_clf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return clf, vectorizer


def doc_id_to_category(df_docs: pd.DataFrame) -> dict[str, Any]:
    return dict(zip(df_docs["id"].astype(str), df_docs["category"]))


def predict_query_categories(
    df_queries: pd.DataFrame,
    clf: Any,
    vectorizer: Any,
    text_column: str = "content",
) -> np.ndarray:
    if text_column not in df_queries.columns:
        df_queries = df_queries.copy()
        df_queries["content"] = df_queries.apply(merge_fields, axis=1)
    X = vectorizer.transform(df_queries[text_column].tolist())
    return clf.predict(X)


def rerank_by_category(
    retrieved_ids: Sequence[str],
    scores: Sequence[float] | None,
    pred_query_cat: Any,
    id_to_cat: dict[str, Any],
    alpha: float = 1.0,
) -> list[str]:
    """
    Reorder retrieved_ids by boosting scores when document category matches
    predicted query category. Same document set; improves MRR when ranking is wrong.
    """
    ids = list(retrieved_ids)
    if not ids:
        return []
    if scores is None or len(scores) != len(ids):
        scores = [float(len(ids) - i) for i in range(len(ids))]
    s = np.array(scores, dtype=np.float64)
    mx = float(np.max(np.abs(s)) + 1e-9)
    bonus = np.array(
        [
            alpha * mx if id_to_cat.get(str(did)) == pred_query_cat else 0.0
            for did in ids
        ]
    )
    order = np.argsort(s + bonus)[::-1]
    return [ids[i] for i in order]
