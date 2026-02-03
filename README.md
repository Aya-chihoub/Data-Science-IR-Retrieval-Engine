# Information Retrieval Engine - Phase 1

A document retrieval system implementing TF-IDF, BM25+, and Semantic Embeddings for the Data Science course project.

## Team Members
- Nour (Data Prep, TF-IDF, BM25+)
- Aya (Embeddings, Visualization, Evaluation)

## Project Overview

This project implements an information retrieval pipeline that:
1. Loads and preprocesses a document collection (216k+ documents)
2. Implements three retrieval methods:
   - **TF-IDF** with cosine similarity
   - **BM25+** ranking algorithm
   - **Semantic Embeddings** using SentenceTransformers
3. Evaluates performance using Recall, Precision, and MRR metrics

## Results (Phase 1)

| Method | Recall@10 | Precision@10 | MRR |
|--------|-----------|--------------|-----|
| TF-IDF | 0.1122 | 0.0657 | 0.2021 |
| BM25+ | 0.1506 | 0.0896 | 0.2452 |
| **Embeddings** | **0.4456** | **0.2703** | **0.5658** |

Embeddings significantly outperform traditional methods!

## Project Structure

```
├── IR_Project_Phase1.ipynb    # Main notebook with all code
├── run_pipeline.py            # TF-IDF & BM25 pipeline script
├── run_embeddings.py          # Embeddings pipeline script
├── run_evaluation.py          # Evaluation metrics script
├── requirements.txt           # Python dependencies
└── data/                      # Dataset (not in repo - download from Kaggle)
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download dataset from Kaggle and place in `data/` folder
4. Run the notebook or scripts

## Dataset

Download from: https://www.kaggle.com/t/f383bc6c1f194226bb43d21ab3d65418

Required files:
- `docs.json` - Document collection
- `queries_train.json` - Training queries
- `queries_test.json` - Test queries
- `qgts_train.json` - Ground truth relevance judgments

## Technologies Used

- Python 3.10+
- scikit-learn (TF-IDF)
- rank_bm25 (BM25+)
- sentence-transformers (Embeddings)
- pandas, numpy, matplotlib

## Phase 1 Deadline
March 2, 2026, 11:59 PM
