"""
Information Retrieval Pipeline - Phase 1
Runs all retrieval methods and evaluation
"""
import json
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Plus
from tqdm import tqdm
import time

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path('data/retrieval-engine-competition')
K = 10  # Top-k results

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df_docs = pd.read_json(DATA_DIR / 'docs.json')
df_queries_train = pd.read_json(DATA_DIR / 'queries_train.json')
df_queries_test = pd.read_json(DATA_DIR / 'queries_test.json')

with open(DATA_DIR / 'qgts_train.json', 'r') as f:
    ground_truth = json.load(f)

print(f"Documents: {len(df_docs):,}")
print(f"Training Queries: {len(df_queries_train):,}")
print(f"Test Queries: {len(df_queries_test):,}")

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

def merge_fields(row):
    """Combine title, text, and tags into content field."""
    title = str(row.get('title', '') or '')
    text = str(row.get('text', '') or '')
    tags_list = row.get('tags', [])
    tags = " ".join(tags_list) if isinstance(tags_list, list) else ""
    combined = f"{title} {text} {tags}"
    return " ".join(combined.split())

def clean_text(text):
    """Clean text: lowercase and remove punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

print("Creating content field for documents...")
df_docs['content'] = df_docs.apply(merge_fields, axis=1)
df_docs['content_clean'] = df_docs['content'].apply(clean_text)

print("Creating content field for training queries...")
df_queries_train['content'] = df_queries_train.apply(merge_fields, axis=1)
df_queries_train['content_clean'] = df_queries_train['content'].apply(clean_text)

print("Creating content field for test queries...")
df_queries_test['content'] = df_queries_test.apply(merge_fields, axis=1)
df_queries_test['content_clean'] = df_queries_test['content'].apply(clean_text)

doc_ids = df_docs['id'].tolist()
query_ids_train = df_queries_train['id'].tolist()

print("Preprocessing complete!")

# ============================================================
# 3. TF-IDF RETRIEVAL
# ============================================================
print("\n" + "=" * 60)
print("3. TF-IDF RETRIEVAL")
print("=" * 60)

start_time = time.time()

print("Building TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100000)
tfidf_doc_matrix = tfidf_vectorizer.fit_transform(df_docs['content_clean'])
print(f"TF-IDF Matrix Shape: {tfidf_doc_matrix.shape}")

print("Transforming queries...")
tfidf_query_matrix = tfidf_vectorizer.transform(df_queries_train['content_clean'])

print("Computing similarities and retrieving top-k...")
similarities = cosine_similarity(tfidf_query_matrix, tfidf_doc_matrix)

topk_indices_tfidf = []
topk_scores_tfidf = []

for i in tqdm(range(len(df_queries_train)), desc="TF-IDF Retrieval"):
    scores = similarities[i]
    top_idx = np.argsort(scores)[-K:][::-1]
    topk_indices_tfidf.append([doc_ids[j] for j in top_idx])
    topk_scores_tfidf.append(scores[top_idx].tolist())

tfidf_time = time.time() - start_time
print(f"TF-IDF complete in {tfidf_time:.2f}s")

# ============================================================
# 4. BM25+ RETRIEVAL
# ============================================================
print("\n" + "=" * 60)
print("4. BM25+ RETRIEVAL")
print("=" * 60)

start_time = time.time()

print("Tokenizing corpus...")
tokenized_corpus = [doc.split() for doc in df_docs['content_clean']]

print("Building BM25+ model...")
bm25_model = BM25Plus(tokenized_corpus)

print("Retrieving top-k for each query...")
topk_indices_bm25 = []
topk_scores_bm25 = []

for query in tqdm(df_queries_train['content_clean'], desc="BM25+ Retrieval"):
    tokenized_query = query.split()
    scores = bm25_model.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[-K:][::-1]
    topk_indices_bm25.append([doc_ids[j] for j in top_idx])
    topk_scores_bm25.append(scores[top_idx].tolist())

bm25_time = time.time() - start_time
print(f"BM25+ complete in {bm25_time:.2f}s")

# ============================================================
# 5. EVALUATION FUNCTIONS
# ============================================================
print("\n" + "=" * 60)
print("5. EVALUATION")
print("=" * 60)

def compute_recall(retrieved_ids, relevant_ids):
    if len(relevant_ids) == 0:
        return 0.0
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)

def compute_precision(retrieved_ids, relevant_ids):
    if len(retrieved_ids) == 0:
        return 0.0
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(retrieved_ids)

def compute_mrr(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(topk_indices, ground_truth, query_ids):
    recalls, precisions, mrrs = [], [], []
    for i, query_id in enumerate(query_ids):
        retrieved = topk_indices[i]
        relevant = ground_truth.get(query_id, [])
        recalls.append(compute_recall(retrieved, relevant))
        precisions.append(compute_precision(retrieved, relevant))
        mrrs.append(compute_mrr(retrieved, relevant))
    return {
        'Recall@k': np.mean(recalls),
        'Precision@k': np.mean(precisions),
        'MRR': np.mean(mrrs)
    }

# Evaluate TF-IDF
results_tfidf = evaluate_retrieval(topk_indices_tfidf, ground_truth, query_ids_train)
print(f"\nTF-IDF Results (k={K}):")
print(f"  Recall@{K}: {results_tfidf['Recall@k']:.4f}")
print(f"  Precision@{K}: {results_tfidf['Precision@k']:.4f}")
print(f"  MRR: {results_tfidf['MRR']:.4f}")

# Evaluate BM25+
results_bm25 = evaluate_retrieval(topk_indices_bm25, ground_truth, query_ids_train)
print(f"\nBM25+ Results (k={K}):")
print(f"  Recall@{K}: {results_bm25['Recall@k']:.4f}")
print(f"  Precision@{K}: {results_bm25['Precision@k']:.4f}")
print(f"  MRR: {results_bm25['MRR']:.4f}")

# ============================================================
# 6. SAVE INTERMEDIATE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("6. SAVING RESULTS")
print("=" * 60)

results = {
    'topk_indices_tfidf': topk_indices_tfidf,
    'topk_scores_tfidf': topk_scores_tfidf,
    'topk_indices_bm25': topk_indices_bm25,
    'topk_scores_bm25': topk_scores_bm25,
    'results_tfidf': results_tfidf,
    'results_bm25': results_bm25,
    'tfidf_time': tfidf_time,
    'bm25_time': bm25_time,
    'doc_ids': doc_ids,
    'query_ids_train': query_ids_train,
    'ground_truth': ground_truth,
    'tfidf_vectorizer': tfidf_vectorizer,
    'tfidf_doc_matrix': tfidf_doc_matrix,
    'df_docs': df_docs,
    'df_queries_train': df_queries_train,
    'df_queries_test': df_queries_test
}

with open('pipeline_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved to pipeline_results.pkl")
print("\nPipeline Part 1 complete! Run embeddings script next.")
