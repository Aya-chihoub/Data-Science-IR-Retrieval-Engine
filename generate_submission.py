"""
Generate Kaggle submission file (Phase 2).
Format: query_id, relevant_doc_ids (JSON array), category (predicted query category).

Methods (--method):
  bm25        — BM25+ top-k (fast baseline)
  embeddings  — SentenceTransformer cosine similarity on full corpus
  hybrid      — RRF: BM25 + embeddings
  fusion      — RRF: BM25 + TF-IDF + embeddings (recommended; matches assignment “combine” methods)

Kaggle-style score weights retrieval (Recall, Precision, MRR, hit-accuracy); fusion usually beats bm25 alone.

Requires: classifier.pkl, tfidf_clf.pkl from train_classifier.py
          pipeline_results.pkl from run_pipeline.py (must include tfidf_* for fusion)

Examples:
  python generate_submission.py --method bm25
  python generate_submission.py --method fusion --model sentence-transformers/all-mpnet-base-v2
  python generate_submission.py --method fusion --fusion-mode weighted --rrf-pool 150
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Plus
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from ir_data_paths import DATA_DIR
from phase2_utils import (
    doc_id_to_category,
    merge_fields,
    load_classifier_artifacts,
    rerank_by_category,
)
DEFAULT_K = 10
DEFAULT_K_POOL = 80
DEFAULT_RERANK_ALPHA = 0.0
DEFAULT_RRF_K = 60
DEFAULT_RRF_POOL = 120
# Stronger retrieval model (slower, larger cache than MiniLM)
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BATCH = 32
DEFAULT_FUSION_WEIGHTS = (0.25, 0.45, 0.30)  # bm25, emb, tfidf for weighted mode


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
) -> list[str]:
    """
    Standard RRF: score(d) = sum_i 1 / (rrf_k + rank_i(d)).
    ranked_lists: each list is doc IDs from best to worst for one retriever.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[str(doc_id)] = scores.get(str(doc_id), 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.keys(), key=lambda d: -scores[d])


def load_or_compute_doc_embeddings(
    contents: list[str],
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    model=None,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "doc_embeddings_meta.json"
    arr_path = cache_dir / "doc_embeddings.npy"
    n = len(contents)

    if arr_path.is_file() and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("model") == model_name
            and meta.get("n_docs") == n
        ):
            emb = np.load(arr_path, mmap_mode="r")
            if emb.shape[0] == n:
                print(f"Loaded doc embeddings from {arr_path} ({emb.shape})")
                return np.asarray(emb, dtype=np.float32)

    from sentence_transformers import SentenceTransformer

    print(f"Encoding {n:,} documents with {model_name} (no valid cache)...")
    if model is None:
        model = SentenceTransformer(model_name)
    emb = model.encode(
        contents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    np.save(arr_path, emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "n_docs": n, "dim": int(emb.shape[1])}, f)
    print(f"Saved doc embeddings to {arr_path}")
    return emb


def topk_from_scores(
    scores: np.ndarray,
    doc_ids: list[str],
    k: int,
) -> list[str]:
    if k >= len(scores):
        top_idx = np.argsort(scores)[::-1]
    else:
        # argpartition for speed on large corpora
        part = np.argpartition(scores, -k)[-k:]
        top_idx = part[np.argsort(scores[part])[::-1]]
    return [doc_ids[i] for i in top_idx]


def _minmax_row(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def weighted_triple_fusion(
    bm25_scores: np.ndarray,
    emb_sims: np.ndarray,
    tfidf_sims: np.ndarray,
    doc_ids: list[str],
    k: int,
    weights: tuple[float, float, float],
) -> list[str]:
    w_bm25, w_emb, w_tfidf = weights
    s = w_bm25 + w_emb + w_tfidf
    if s > 0:
        w_bm25, w_emb, w_tfidf = w_bm25 / s, w_emb / s, w_tfidf / s
    combined = (
        w_bm25 * _minmax_row(bm25_scores)
        + w_emb * _minmax_row(emb_sims)
        + w_tfidf * _minmax_row(tfidf_sims)
    )
    return topk_from_scores(combined, doc_ids, k)


def load_sentence_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_texts_single_model(model, texts: list[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 Kaggle submission CSV")
    p.add_argument(
        "--method",
        choices=("bm25", "embeddings", "hybrid", "fusion"),
        default="fusion",
        help="Retrieval method (fusion = BM25 + TF-IDF + embeddings; best for leaderboard)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Final top-k documents")
    p.add_argument("--output", default="submission.csv", help="Output CSV path")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/embeddings"),
        help="Cache directory for full document embeddings",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument(
        "--rerank-alpha",
        type=float,
        default=DEFAULT_RERANK_ALPHA,
        help="Category rerank strength (0=off; >0 only if validated locally)",
    )
    p.add_argument("--k-pool", type=int, default=DEFAULT_K_POOL, help="Pool for rerank")
    p.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF smoothing constant k (typical 40–80)",
    )
    p.add_argument(
        "--rrf-pool",
        type=int,
        default=DEFAULT_RRF_POOL,
        help="Top candidates per retriever before RRF / fusion",
    )
    p.add_argument(
        "--fusion-mode",
        choices=("rrf", "weighted"),
        default="rrf",
        help="fusion only: RRF (default) or min-max weighted sum of three scores",
    )
    p.add_argument(
        "--w-bm25",
        type=float,
        default=DEFAULT_FUSION_WEIGHTS[0],
        help="weighted fusion: BM25 weight",
    )
    p.add_argument(
        "--w-emb",
        type=float,
        default=DEFAULT_FUSION_WEIGHTS[1],
        help="weighted fusion: embedding similarity weight",
    )
    p.add_argument(
        "--w-tfidf",
        type=float,
        default=DEFAULT_FUSION_WEIGHTS[2],
        help="weighted fusion: TF-IDF cosine weight",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    k = args.k
    rerank_alpha = args.rerank_alpha

    print("=" * 60)
    print("GENERATING KAGGLE SUBMISSION (Phase 2)")
    print("=" * 60)

    clf_path = Path("classifier.pkl")
    vec_path = Path("tfidf_clf.pkl")
    if not clf_path.is_file() or not vec_path.is_file():
        print(
            "\nMissing classifier.pkl or tfidf_clf.pkl.\n"
            "Run:  python train_classifier.py\n"
        )
        raise SystemExit(1)

    clf, tfidf_vec = load_classifier_artifacts()

    print("\nLoading test queries...")
    df_queries_test = pd.read_json(DATA_DIR / "queries_test.json")
    print(f"Test queries: {len(df_queries_test)}")

    print("Loading pipeline results...")
    with open("pipeline_results.pkl", "rb") as f:
        results = pickle.load(f)

    df_docs = results["df_docs"]
    doc_ids = results["doc_ids"]
    id_to_cat = doc_id_to_category(df_docs) if rerank_alpha > 0 else {}

    df_queries_test = df_queries_test.copy()
    df_queries_test["content"] = df_queries_test.apply(merge_fields, axis=1)
    df_queries_test["content_clean"] = df_queries_test["content"].apply(clean_text)

    print("Predicting query categories...")
    Xq = tfidf_vec.transform(df_queries_test["content"].tolist())
    pred_categories = clf.predict(Xq)

    tokenized_corpus = [doc.split() for doc in df_docs["content_clean"]]
    bm25_model = BM25Plus(tokenized_corpus)

    need_emb = args.method in ("embeddings", "hybrid", "fusion")
    need_tfidf = args.method == "fusion"

    if need_tfidf:
        if "tfidf_vectorizer" not in results or "tfidf_doc_matrix" not in results:
            print(
                "ERROR: fusion needs tfidf_vectorizer + tfidf_doc_matrix in pipeline_results.pkl.\n"
                "Run: python run_pipeline.py\n"
            )
            raise SystemExit(1)

    doc_emb_norm: np.ndarray | None = None
    emb_sims_all: np.ndarray | None = None
    st_model = None

    if need_emb:
        print("Loading sentence-transformers model (once for docs + queries)...")
        st_model = load_sentence_model(args.model)
        doc_texts = df_docs["content"].tolist()
        doc_emb = load_or_compute_doc_embeddings(
            doc_texts,
            args.model,
            args.cache_dir,
            args.batch_size,
            model=st_model,
        )
        doc_emb_norm = normalize(doc_emb, norm="l2", axis=1)
        q_texts = df_queries_test["content"].tolist()
        query_emb = encode_texts_single_model(st_model, q_texts, args.batch_size)
        query_emb_norm = normalize(query_emb, norm="l2", axis=1)
        emb_sims_all = query_emb_norm @ doc_emb_norm.T

    tfidf_sims_all: np.ndarray | None = None
    if need_tfidf:
        print("Computing TF-IDF similarities for test queries (one matrix multiply)...")
        tv = results["tfidf_vectorizer"]
        td = results["tfidf_doc_matrix"]
        tfidf_q = tv.transform(df_queries_test["content_clean"])
        tfidf_sims_all = cosine_similarity(tfidf_q, td)

    print(f"\nMethod: {args.method} | top-{k}", end="")
    if args.method in ("hybrid", "fusion"):
        print(
            f" | pool={args.rrf_pool} rrf_k={args.rrf_k}"
            + (f" fusion_mode={args.fusion_mode}" if args.method == "fusion" else "")
        )
    elif rerank_alpha > 0:
        print(f" | pool={args.k_pool} rerank_alpha={rerank_alpha}")
    else:
        print()

    test_rows: list[dict] = []
    pool = min(args.rrf_pool, len(doc_ids))
    fw = (args.w_bm25, args.w_emb, args.w_tfidf)

    for i, (_, row) in enumerate(
        tqdm(
            df_queries_test.iterrows(),
            total=len(df_queries_test),
            desc="Retrieval",
        )
    ):
        query_clean = row["content_clean"]
        qid = row["id"]
        pred_cat = pred_categories[i]

        tokenized_query = query_clean.split()
        bm25_scores = bm25_model.get_scores(tokenized_query)

        if args.method == "bm25":
            if rerank_alpha > 0:
                top_idx = np.argsort(bm25_scores)[-args.k_pool :][::-1]
                pool_ids = [doc_ids[j] for j in top_idx]
                pool_scores = bm25_scores[top_idx].tolist()
                topk_ids = rerank_by_category(
                    pool_ids, pool_scores, pred_cat, id_to_cat, alpha=rerank_alpha
                )[:k]
            else:
                top_idx = np.argsort(bm25_scores)[-k:][::-1]
                topk_ids = [doc_ids[j] for j in top_idx]

        elif args.method == "embeddings":
            assert emb_sims_all is not None
            topk_ids = topk_from_scores(emb_sims_all[i], doc_ids, k)

        elif args.method == "hybrid":
            assert emb_sims_all is not None
            bm25_ranked = topk_from_scores(bm25_scores, doc_ids, pool)
            emb_ranked = topk_from_scores(emb_sims_all[i], doc_ids, pool)
            fused = reciprocal_rank_fusion(
                [bm25_ranked, emb_ranked],
                rrf_k=args.rrf_k,
            )
            topk_ids = fused[:k]

        else:  # fusion: BM25 + TF-IDF + embeddings (Phase 2 “combine methods”)
            assert emb_sims_all is not None and tfidf_sims_all is not None
            if args.fusion_mode == "weighted":
                topk_ids = weighted_triple_fusion(
                    bm25_scores,
                    emb_sims_all[i],
                    tfidf_sims_all[i],
                    doc_ids,
                    k,
                    fw,
                )
            else:
                bm25_ranked = topk_from_scores(bm25_scores, doc_ids, pool)
                emb_ranked = topk_from_scores(emb_sims_all[i], doc_ids, pool)
                tfidf_ranked = topk_from_scores(tfidf_sims_all[i], doc_ids, pool)
                fused = reciprocal_rank_fusion(
                    [bm25_ranked, emb_ranked, tfidf_ranked],
                    rrf_k=args.rrf_k,
                )
                topk_ids = fused[:k]

        test_rows.append(
            {
                "query_id": qid,
                "doc_ids": topk_ids,
                "category": pred_cat,
            }
        )

    out_path = Path(args.output)
    print(f"\nWriting {out_path}...")
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["query_id", "relevant_doc_ids", "category"])
        for r in test_rows:
            writer.writerow(
                [r["query_id"], json.dumps(r["doc_ids"]), r["category"]]
            )

    print(f"Rows: {len(test_rows)}, top-k: {k}")
    print("\nSample:")
    for line in out_path.read_text(encoding="utf-8").splitlines()[:4]:
        print(line)
    print(f"\nDone. Upload {out_path} to Kaggle.")


if __name__ == "__main__":
    main()
