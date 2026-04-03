"""
Phase 2 evaluation: query category accuracy + retrieval metrics with category reranking.
Requires: pipeline_results.pkl (from run_pipeline.py / run_embeddings.py), classifier.pkl, tfidf_clf.pkl
"""
import json
import pickle

import numpy as np

from ir_data_paths import DATA_DIR
from phase2_utils import (
    doc_id_to_category,
    merge_fields,
    predict_query_categories,
    rerank_by_category,
    load_classifier_artifacts,
)

K = 10


def _parse_ground_truth() -> dict[str, list]:
    with open(DATA_DIR / "qgts_train.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        qid: [item["doc_id"] for item in data["relevant_doc_ids"]]
        for qid, data in raw.items()
    }


def compute_recall(retrieved_ids: list, relevant_ids: list) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)


def compute_precision(retrieved_ids: list, relevant_ids: list) -> float:
    if not retrieved_ids:
        return 0.0
    return len(set(retrieved_ids) & set(relevant_ids)) / len(retrieved_ids)


def compute_mrr(retrieved_ids: list, relevant_ids: list) -> float:
    rel = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def evaluate_method(
    topk_lists: list[list],
    score_lists: list[list[float]] | None,
    ground_truth: dict,
    query_ids: list,
    pred_cats: np.ndarray | list,
    id_to_cat: dict,
    alpha: float,
) -> tuple[dict, dict]:
    recalls, precs, mrrs = [], [], []
    recalls_r, precs_r, mrrs_r = [], [], []

    for i, qid in enumerate(query_ids):
        rel = ground_truth.get(qid, [])
        retrieved = topk_lists[i]
        scores = score_lists[i] if score_lists is not None else None

        recalls.append(compute_recall(retrieved, rel))
        precs.append(compute_precision(retrieved, rel))
        mrrs.append(compute_mrr(retrieved, rel))

        reranked = rerank_by_category(
            retrieved, scores, pred_cats[i], id_to_cat, alpha=alpha
        )
        recalls_r.append(compute_recall(reranked, rel))
        precs_r.append(compute_precision(reranked, rel))
        mrrs_r.append(compute_mrr(reranked, rel))

    base = {
        "Recall@k": float(np.mean(recalls)),
        "Precision@k": float(np.mean(precs)),
        "MRR": float(np.mean(mrrs)),
    }
    rr = {
        "Recall@k": float(np.mean(recalls_r)),
        "Precision@k": float(np.mean(precs_r)),
        "MRR": float(np.mean(mrrs_r)),
    }
    return base, rr


def main() -> None:
    ground_truth = _parse_ground_truth()

    clf, vectorizer = load_classifier_artifacts()
    with open("pipeline_results.pkl", "rb") as f:
        results = pickle.load(f)

    df_queries_train = results["df_queries_train"]
    if "content" not in df_queries_train.columns:
        df_queries_train = df_queries_train.copy()
        df_queries_train["content"] = df_queries_train.apply(merge_fields, axis=1)

    true_cats = df_queries_train["category"].tolist()
    query_ids_train = list(results["query_ids_train"])
    pred_cats = predict_query_categories(df_queries_train, clf, vectorizer, "content")

    cat_acc = float(np.mean(np.array(pred_cats) == np.array(true_cats)))
    print("=" * 60)
    print("PHASE 2 - QUERY CATEGORY PREDICTION (train queries)")
    print("=" * 60)
    print(f"Average accuracy (pred vs true category): {cat_acc:.4f}")

    df_docs = results["df_docs"]
    id_to_cat = doc_id_to_category(df_docs)

    alpha = 1.0
    print(f"\nCategory reranking: additive bonus = alpha * |max score| per query (alpha={alpha})")
    print("=" * 60)
    print(f"{'Method':<14} {'Metric':<12} {'Before':>10} {'After rerank':>14}")
    print("-" * 60)

    methods = [
        ("TF-IDF", "topk_indices_tfidf", "topk_scores_tfidf"),
        ("BM25+", "topk_indices_bm25", "topk_scores_bm25"),
    ]
    if "topk_indices_emb" in results:
        methods.append(("Embeddings", "topk_indices_emb", "topk_scores_emb"))

    for label, key_idx, key_sc in methods:
        if key_idx not in results:
            continue
        topk_lists = results[key_idx]
        score_lists = results.get(key_sc)
        base, rr = evaluate_method(
            topk_lists,
            score_lists,
            ground_truth,
            query_ids_train,
            pred_cats,
            id_to_cat,
            alpha,
        )
        for metric in ("Recall@k", "Precision@k", "MRR"):
            print(
                f"{label:<14} {metric:<12} {base[metric]:>10.4f} {rr[metric]:>14.4f}"
            )
        print("-" * 60)

    print("\nNote: category reranking only reorders the existing top-k list (same @k candidates).")
    print("Train classifier with: python train_classifier.py")


if __name__ == "__main__":
    main()
