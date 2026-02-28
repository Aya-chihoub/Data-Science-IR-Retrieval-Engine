"""
Generate Kaggle submission file
Format: queryID, relevantIDs (JSON array), category
"""
import json
import csv
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('data/retrieval-engine-competition')
K = 10  # Top-k results to submit

print("=" * 60)
print("GENERATING KAGGLE SUBMISSION")
print("=" * 60)

# Load test queries
print("\nLoading test queries...")
df_queries_test = pd.read_json(DATA_DIR / 'queries_test.json')
print(f"Test queries: {len(df_queries_test)}")

# Load pipeline results
print("Loading pipeline results...")
with open('pipeline_results.pkl', 'rb') as f:
    results = pickle.load(f)

df_docs = results['df_docs']
doc_ids = results['doc_ids']

# Preprocessing functions
import re

def merge_fields(row):
    title = str(row.get('title', '') or '')
    text = str(row.get('text', '') or '')
    tags_list = row.get('tags', [])
    tags = " ".join(tags_list) if isinstance(tags_list, list) else ""
    combined = f"{title} {text} {tags}"
    return " ".join(combined.split())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Preprocess test queries
print("Preprocessing test queries...")
df_queries_test['content'] = df_queries_test.apply(merge_fields, axis=1)
df_queries_test['content_clean'] = df_queries_test['content'].apply(clean_text)

# Choose method - use BM25+ (better than TF-IDF, faster than embeddings)
print("\nUsing BM25+ for submission...")
from rank_bm25 import BM25Plus

print("Tokenizing corpus...")
tokenized_corpus = [doc.split() for doc in df_docs['content_clean']]

print("Building BM25+ model...")
bm25_model = BM25Plus(tokenized_corpus)

print("Retrieving top-k for test queries...")
test_results = []

for idx, row in tqdm(df_queries_test.iterrows(), total=len(df_queries_test), desc="BM25+ Retrieval"):
    query = row['content_clean']
    query_id = row['id']
    
    tokenized_query = query.split()
    scores = bm25_model.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[-K:][::-1]
    
    # Get doc IDs (not indices!)
    retrieved_doc_ids = [doc_ids[j] for j in top_idx]
    
    test_results.append({
        'query_id': query_id,
        'doc_ids': retrieved_doc_ids
    })

# Generate submission CSV
print("\nWriting submission.csv...")
with open('submission.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['queryID', 'relevantIDs', 'category'])
    
    for result in test_results:
        query_id = result['query_id']
        doc_ids_json = json.dumps(result['doc_ids'])
        writer.writerow([query_id, doc_ids_json, "?"])

print(f"\nSubmission file created: submission.csv")
print(f"Total test queries: {len(test_results)}")
print(f"Top-k per query: {K}")

# Show sample
print("\nSample rows:")
with open('submission.csv', 'r') as f:
    for i, line in enumerate(f):
        if i < 4:
            print(line.strip())
        else:
            break

print("\nDone! Upload submission.csv to Kaggle.")
