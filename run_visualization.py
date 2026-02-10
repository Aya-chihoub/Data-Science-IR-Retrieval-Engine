"""
Visualization script for IR Retrieval Engine
Generates t-SNE and UMAP plots of document embeddings
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import pickle

# Try to import umap
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available, will only generate t-SNE plot")

print("Loading data...")

# Load documents
data_path = Path("data/retrieval-engine-competition")
with open(data_path / "docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Convert to list if dict
if isinstance(docs, dict):
    docs_list = list(docs.values())
else:
    docs_list = docs

print(f"Loaded {len(docs_list)} documents")

# Sample documents for visualization (too many to plot all)
np.random.seed(42)
sample_size = 2000
sample_indices = np.random.choice(len(docs_list), min(sample_size, len(docs_list)), replace=False)
sample_docs = [docs_list[i] for i in sample_indices]

# Get categories
categories = [doc.get('category', 'unknown') for doc in sample_docs]
unique_categories = list(set(categories))
print(f"Categories: {unique_categories}")

# Load or generate embeddings
print("Loading embedding model...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare text content
def merge_fields(doc):
    title = doc.get('title', '') or ''
    text = doc.get('text', '') or ''
    tags = doc.get('tags', []) or []
    if isinstance(tags, list):
        tags = ' '.join(tags)
    return f"{title} {text} {tags}".strip()

contents = [merge_fields(doc) for doc in sample_docs]

print(f"Generating embeddings for {len(contents)} documents...")
embeddings = model.encode(contents, show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)

# Create reports directory if it doesn't exist
Path("reports").mkdir(exist_ok=True)

# t-SNE visualization
print("Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings_2d_tsne = tsne.fit_transform(embeddings)

# Plot t-SNE
plt.figure(figsize=(12, 8))
category_to_num = {cat: i for i, cat in enumerate(unique_categories)}
colors = [category_to_num[cat] for cat in categories]

scatter = plt.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], 
                      c=colors, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(scatter, ticks=range(len(unique_categories)), label='Category')
plt.clim(-0.5, len(unique_categories) - 0.5)

# Add legend
handles = [plt.scatter([], [], c=[plt.cm.tab10(i/10)], label=cat, s=50) 
           for i, cat in enumerate(unique_categories)]
plt.legend(handles=handles, title='Categories', loc='upper right', fontsize=8)

plt.title('t-SNE Visualization of Document Embeddings', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.savefig('reports/tsne_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/tsne_visualization.png")

# UMAP visualization (if available)
if UMAP_AVAILABLE:
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d_umap = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], 
                          c=colors, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(len(unique_categories)), label='Category')
    plt.clim(-0.5, len(unique_categories) - 0.5)
    
    handles = [plt.scatter([], [], c=[plt.cm.tab10(i/10)], label=cat, s=50) 
               for i, cat in enumerate(unique_categories)]
    plt.legend(handles=handles, title='Categories', loc='upper right', fontsize=8)
    
    plt.title('UMAP Visualization of Document Embeddings', fontsize=14)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.tight_layout()
    plt.savefig('reports/umap_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/umap_visualization.png")

# Results comparison bar chart
print("Creating results comparison chart...")
methods = ['TF-IDF', 'BM25+', 'Embeddings']
recall = [0.1122, 0.1506, 0.4456]
precision = [0.0657, 0.0896, 0.2703]
mrr = [0.2021, 0.2452, 0.5658]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, recall, width, label='Recall@10', color='#2ecc71')
bars2 = ax.bar(x, precision, width, label='Precision@10', color='#3498db')
bars3 = ax.bar(x + width, mrr, width, label='MRR', color='#9b59b6')

ax.set_ylabel('Score')
ax.set_title('Retrieval Methods Comparison - Phase 1 Results')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0, 0.7)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('reports/results_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/results_comparison.png")

print("\n✅ All visualizations complete!")
print("Generated files in reports/:")
print("  - tsne_visualization.png")
if UMAP_AVAILABLE:
    print("  - umap_visualization.png")
print("  - results_comparison.png")
