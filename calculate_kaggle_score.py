"""
Calculate Kaggle score with all 4 metrics
score = 0.25 * Recall + 0.25 * Precision + 0.25 * MRR + 0.25 * Accuracy
"""
import pickle
import numpy as np

# Load results
with open('pipeline_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Get parsed ground truth
ground_truth = results.get('ground_truth_parsed')
if ground_truth is None:
    # Parse it if not already done
    import json
    from pathlib import Path
    DATA_DIR = Path('data/retrieval-engine-competition')
    with open(DATA_DIR / 'qgts_train.json', 'r') as f:
        raw_gt = json.load(f)
    ground_truth = {}
    for query_id, data in raw_gt.items():
        relevant_docs = [item['doc_id'] for item in data['relevant_doc_ids']]
        ground_truth[query_id] = relevant_docs

topk_indices_tfidf = results['topk_indices_tfidf']
topk_indices_bm25 = results['topk_indices_bm25']
query_ids_train = results['query_ids_train']

def compute_all_metrics(topk_indices, ground_truth, query_ids):
    recalls, precisions, mrrs, accuracies = [], [], [], []
    
    for i, query_id in enumerate(query_ids):
        retrieved = topk_indices[i]
        relevant = ground_truth.get(query_id, [])
        
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        intersection = retrieved_set & relevant_set
        
        # Recall
        if len(relevant_set) > 0:
            recall = len(intersection) / len(relevant_set)
        else:
            recall = 0.0
        recalls.append(recall)
        
        # Precision
        if len(retrieved) > 0:
            precision = len(intersection) / len(retrieved)
        else:
            precision = 0.0
        precisions.append(precision)
        
        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)
        
        # Accuracy (1 if at least one relevant doc retrieved, 0 otherwise)
        accuracy = 1.0 if len(intersection) > 0 else 0.0
        accuracies.append(accuracy)
    
    return {
        'Recall': np.mean(recalls),
        'Precision': np.mean(precisions),
        'MRR': np.mean(mrrs),
        'Accuracy': np.mean(accuracies)
    }

def kaggle_score(metrics):
    return 0.25 * metrics['Recall'] + 0.25 * metrics['Precision'] + 0.25 * metrics['MRR'] + 0.25 * metrics['Accuracy']

# Calculate for TF-IDF
print("=" * 60)
print("KAGGLE SCORE CALCULATION")
print("=" * 60)

metrics_tfidf = compute_all_metrics(topk_indices_tfidf, ground_truth, query_ids_train)
score_tfidf = kaggle_score(metrics_tfidf)
print(f"\nTF-IDF:")
print(f"  Recall:    {metrics_tfidf['Recall']:.4f}")
print(f"  Precision: {metrics_tfidf['Precision']:.4f}")
print(f"  MRR:       {metrics_tfidf['MRR']:.4f}")
print(f"  Accuracy:  {metrics_tfidf['Accuracy']:.4f}")
print(f"  --> KAGGLE SCORE: {score_tfidf:.4f}")

# Calculate for BM25+
metrics_bm25 = compute_all_metrics(topk_indices_bm25, ground_truth, query_ids_train)
score_bm25 = kaggle_score(metrics_bm25)
print(f"\nBM25+:")
print(f"  Recall:    {metrics_bm25['Recall']:.4f}")
print(f"  Precision: {metrics_bm25['Precision']:.4f}")
print(f"  MRR:       {metrics_bm25['MRR']:.4f}")
print(f"  Accuracy:  {metrics_bm25['Accuracy']:.4f}")
print(f"  --> KAGGLE SCORE: {score_bm25:.4f}")

# Load embeddings results if available
try:
    with open('embeddings_results.pkl', 'rb') as f:
        emb_results = pickle.load(f)
    
    topk_indices_emb = emb_results['topk_indices_embeddings']
    
    metrics_emb = compute_all_metrics(topk_indices_emb, ground_truth, query_ids_train)
    score_emb = kaggle_score(metrics_emb)
    print(f"\nEmbeddings:")
    print(f"  Recall:    {metrics_emb['Recall']:.4f}")
    print(f"  Precision: {metrics_emb['Precision']:.4f}")
    print(f"  MRR:       {metrics_emb['MRR']:.4f}")
    print(f"  Accuracy:  {metrics_emb['Accuracy']:.4f}")
    print(f"  --> KAGGLE SCORE: {score_emb:.4f}")
except FileNotFoundError:
    print("\nEmbeddings results not found (embeddings_results.pkl)")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"TF-IDF Kaggle Score:  {score_tfidf:.4f}")
print(f"BM25+ Kaggle Score:   {score_bm25:.4f}")
try:
    print(f"Embeddings Score:     {score_emb:.4f}")
    print(f"\nBest method: Embeddings with score {score_emb:.4f}")
except:
    print(f"\nBest method: BM25+ with score {score_bm25:.4f}")
