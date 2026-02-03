"""
Evaluation with correct ground truth parsing
"""
import json
import pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path('data/retrieval-engine-competition')

# Load ground truth
with open(DATA_DIR / 'qgts_train.json', 'r') as f:
    raw_ground_truth = json.load(f)

# Parse ground truth - extract just the doc_ids
ground_truth = {}
for query_id, data in raw_ground_truth.items():
    relevant_docs = [item['doc_id'] for item in data['relevant_doc_ids']]
    ground_truth[query_id] = relevant_docs

print(f"Parsed ground truth for {len(ground_truth)} queries")
print(f"Example - Query: {list(ground_truth.keys())[0]}")
print(f"         Relevant docs: {ground_truth[list(ground_truth.keys())[0]]}")

# Load pipeline results
with open('pipeline_results.pkl', 'rb') as f:
    results = pickle.load(f)

topk_indices_tfidf = results['topk_indices_tfidf']
topk_indices_bm25 = results['topk_indices_bm25']
query_ids_train = results['query_ids_train']

# Evaluation functions
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

# Run evaluation
K = 10
print("\n" + "=" * 60)
print(f"EVALUATION RESULTS (k={K})")
print("=" * 60)

results_tfidf = evaluate_retrieval(topk_indices_tfidf, ground_truth, query_ids_train)
print(f"\nTF-IDF:")
print(f"  Recall@{K}:    {results_tfidf['Recall@k']:.4f}")
print(f"  Precision@{K}: {results_tfidf['Precision@k']:.4f}")
print(f"  MRR:           {results_tfidf['MRR']:.4f}")

results_bm25 = evaluate_retrieval(topk_indices_bm25, ground_truth, query_ids_train)
print(f"\nBM25+:")
print(f"  Recall@{K}:    {results_bm25['Recall@k']:.4f}")
print(f"  Precision@{K}: {results_bm25['Precision@k']:.4f}")
print(f"  MRR:           {results_bm25['MRR']:.4f}")

# Save corrected ground truth
with open('pipeline_results.pkl', 'rb') as f:
    all_results = pickle.load(f)
    
all_results['ground_truth_parsed'] = ground_truth
all_results['results_tfidf'] = results_tfidf
all_results['results_bm25'] = results_bm25

with open('pipeline_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\nResults updated and saved!")
