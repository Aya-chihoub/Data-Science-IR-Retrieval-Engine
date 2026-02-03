"""
Embeddings Pipeline - Phase 1
Using SentenceTransformers for semantic retrieval
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path('data/retrieval-engine-competition')
K = 10
# Set to None to process all documents, or a number for sampling
SAMPLE_SIZE = 10000  # Start with 10k docs for testing (set to None for full)

print("=" * 60)
print("EMBEDDINGS PIPELINE")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n1. Loading data...")

with open('pipeline_results.pkl', 'rb') as f:
    prev_results = pickle.load(f)

df_docs = prev_results['df_docs']
df_queries_train = prev_results['df_queries_train']
ground_truth = prev_results.get('ground_truth_parsed')

if ground_truth is None:
    # Parse ground truth if not already done
    with open(DATA_DIR / 'qgts_train.json', 'r') as f:
        raw_gt = json.load(f)
    ground_truth = {qid: [item['doc_id'] for item in data['relevant_doc_ids']] 
                    for qid, data in raw_gt.items()}

doc_ids = df_docs['id'].tolist()
query_ids_train = df_queries_train['id'].tolist()

print(f"Documents: {len(df_docs):,}")
print(f"Queries: {len(df_queries_train):,}")

# ============================================================
# 2. SAMPLE (for faster testing)
# ============================================================
if SAMPLE_SIZE and SAMPLE_SIZE < len(df_docs):
    print(f"\n2. Sampling {SAMPLE_SIZE:,} documents for faster testing...")
    np.random.seed(42)
    
    # Make sure we include documents that are relevant to queries
    relevant_doc_ids = set()
    for rel_docs in ground_truth.values():
        relevant_doc_ids.update(rel_docs)
    
    # Get indices of relevant docs
    relevant_indices = [i for i, doc_id in enumerate(doc_ids) if doc_id in relevant_doc_ids]
    print(f"  Including {len(relevant_indices)} relevant documents")
    
    # Sample remaining docs
    other_indices = [i for i in range(len(df_docs)) if i not in relevant_indices]
    remaining = SAMPLE_SIZE - len(relevant_indices)
    if remaining > 0:
        sampled_other = np.random.choice(other_indices, min(remaining, len(other_indices)), replace=False).tolist()
    else:
        sampled_other = []
    
    sample_indices = sorted(relevant_indices + sampled_other)
    
    df_docs_sample = df_docs.iloc[sample_indices].reset_index(drop=True)
    doc_ids_sample = [doc_ids[i] for i in sample_indices]
    
    print(f"  Final sample: {len(df_docs_sample):,} documents")
else:
    df_docs_sample = df_docs
    doc_ids_sample = doc_ids
    print("\n2. Using all documents (this will take a while)...")

# ============================================================
# 3. LOAD MODEL
# ============================================================
print("\n3. Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"  Model: all-MiniLM-L6-v2")
print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

# ============================================================
# 4. GENERATE EMBEDDINGS
# ============================================================
print("\n4. Generating embeddings...")

# Document embeddings
print("  Encoding documents...")
start_time = time.time()
doc_embeddings = model.encode(
    df_docs_sample['content'].tolist(),
    show_progress_bar=True,
    batch_size=32
)
doc_time = time.time() - start_time
print(f"  Document embeddings shape: {doc_embeddings.shape}")
print(f"  Time: {doc_time:.1f}s")

# Query embeddings
print("\n  Encoding queries...")
start_time = time.time()
query_embeddings = model.encode(
    df_queries_train['content'].tolist(),
    show_progress_bar=True,
    batch_size=32
)
query_time = time.time() - start_time
print(f"  Query embeddings shape: {query_embeddings.shape}")
print(f"  Time: {query_time:.1f}s")

# ============================================================
# 5. RETRIEVAL
# ============================================================
print("\n5. Retrieving top-k documents...")

start_time = time.time()
similarities = cosine_similarity(query_embeddings, doc_embeddings)

topk_indices_emb = []
topk_scores_emb = []

for i in tqdm(range(len(df_queries_train)), desc="Embedding Retrieval"):
    scores = similarities[i]
    top_idx = np.argsort(scores)[-K:][::-1]
    topk_indices_emb.append([doc_ids_sample[j] for j in top_idx])
    topk_scores_emb.append(scores[top_idx].tolist())

retrieval_time = time.time() - start_time
print(f"  Time: {retrieval_time:.1f}s")

# ============================================================
# 6. EVALUATION
# ============================================================
print("\n6. Evaluation...")

def compute_recall(retrieved, relevant):
    if not relevant: return 0.0
    return len(set(retrieved) & set(relevant)) / len(relevant)

def compute_precision(retrieved, relevant):
    if not retrieved: return 0.0
    return len(set(retrieved) & set(relevant)) / len(retrieved)

def compute_mrr(retrieved, relevant):
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0

recalls, precisions, mrrs = [], [], []
for i, query_id in enumerate(query_ids_train):
    retrieved = topk_indices_emb[i]
    relevant = ground_truth.get(query_id, [])
    recalls.append(compute_recall(retrieved, relevant))
    precisions.append(compute_precision(retrieved, relevant))
    mrrs.append(compute_mrr(retrieved, relevant))

results_emb = {
    'Recall@k': np.mean(recalls),
    'Precision@k': np.mean(precisions),
    'MRR': np.mean(mrrs)
}

print(f"\nEmbeddings Results (k={K}):")
print(f"  Recall@{K}:    {results_emb['Recall@k']:.4f}")
print(f"  Precision@{K}: {results_emb['Precision@k']:.4f}")
print(f"  MRR:           {results_emb['MRR']:.4f}")

# ============================================================
# 7. COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON (all methods)")
print("=" * 60)

# Load previous results
results_tfidf = prev_results.get('results_tfidf', {})
results_bm25 = prev_results.get('results_bm25', {})

print(f"\n{'Method':<15} {'Recall@10':<12} {'Precision@10':<14} {'MRR':<10}")
print("-" * 50)
if results_tfidf:
    print(f"{'TF-IDF':<15} {results_tfidf['Recall@k']:<12.4f} {results_tfidf['Precision@k']:<14.4f} {results_tfidf['MRR']:<10.4f}")
if results_bm25:
    print(f"{'BM25+':<15} {results_bm25['Recall@k']:<12.4f} {results_bm25['Precision@k']:<14.4f} {results_bm25['MRR']:<10.4f}")
print(f"{'Embeddings':<15} {results_emb['Recall@k']:<12.4f} {results_emb['Precision@k']:<14.4f} {results_emb['MRR']:<10.4f}")

if SAMPLE_SIZE:
    print(f"\n* Embeddings evaluated on {SAMPLE_SIZE:,} doc sample")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("Saving results...")

prev_results['topk_indices_emb'] = topk_indices_emb
prev_results['topk_scores_emb'] = topk_scores_emb
prev_results['results_emb'] = results_emb
prev_results['doc_embeddings'] = doc_embeddings
prev_results['query_embeddings'] = query_embeddings
prev_results['embedding_sample_size'] = SAMPLE_SIZE

with open('pipeline_results.pkl', 'wb') as f:
    pickle.dump(prev_results, f)

print("Results saved!")
print("\nDone!")
