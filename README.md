# Information Retrieval Engine - Phase 1

A document retrieval system implementing TF-IDF, BM25+, and Semantic Embeddings for the Data Science course project.

## Team Members
- Yanis Hemdane
- Rayane Khatim
- Nour el imene Khelassi
- Aya Chihoub

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
| **Embeddings** | **0.1959** | **0.1141** | **0.2745** |

**Kaggle Score:** 0.39344

Embeddings outperform traditional keyword-based methods across all metrics!

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

# Information Retrieval Engine - Phase 2

Extension of the Phase 1 retrieval pipeline with document classification, category-aware retrieval, and learning-to-rank reranking.

## Team Members

- Yanis Hemdane
- Rayane Khatim
- Nour el imene Khelassi
- Aya Chihoub

## Project Overview

Phase 2 builds on the Phase 1 pipeline by adding:

1. A **category classifier** (Logistic Regression + TF-IDF) to predict the domain of queries and documents across 5 categories: `gaming`, `tex`, `unix`, `android`, `programmers`
2. **Category-aware retrieval** using hard filtering and soft category bonuses to reduce cross-domain noise
3. A **cooperative fusion pipeline** combining BM25+, multiple embedding models, and Pseudo-Relevance Feedback via weighted 8-way RRF
4. A **LightGBM LambdaRank reranker** as a second-stage reranking step trained on 10 query-document features

## Results (Phase 2)

### Classifier Performance

| Metric   | Score  |
|----------|--------|
| Accuracy | 99.69% |
| F1 (avg) | ~1.00  |

### Retrieval — Ablation Study (k=10)

| Method                        | Recall@10 | Precision@10 | MRR    |
|-------------------------------|-----------|--------------|--------|
| BM25 only                     | 0.1215    | 0.0706       | 0.2061 |
| Embeddings only               | 0.1618    | 0.0976       | 0.2386 |
| BM25 + Embeddings (RRF)       | 0.1768    | 0.1064       | 0.2824 |
| + Soft category bonus         | 0.1849    | 0.1110       | 0.2815 |
| + LightGBM reranker           | —         | —            | 0.4827 |

### Optimization Journey

| Step                                  | Kaggle Score |
|---------------------------------------|--------------|
| Baseline + category filtering         | ~0.38–0.43   |
| Model selection (MiniLM-L12-v2)       | ~0.458       |
| Cooperative pipeline + PRF            | ~0.50        |
| **LightGBM LambdaRank reranking**     | **0.54292**  |

**Best Kaggle Score: 0.54292**

## Setup

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Download the dataset from Kaggle and place it in the `data/` folder
4. Run `IR_Project_Phase2.ipynb` for the full Phase 2 pipeline

## Dataset

Download from: https://www.kaggle.com/t/f383bc6c1f194226bb43d21ab3d65418

Required files:

- `docs.json` — Document collection (216k+ documents)
- `queries_train.json` — Training queries
- `queries_test.json` — Test queries
- `qgts_train.json` — Ground truth relevance judgments

## Technologies Used

- Python 3.10+
- scikit-learn (Logistic Regression, TF-IDF)
- rank_bm25 (BM25+)
- sentence-transformers (`all-MiniLM-L12-v2`, `mpnet-base-v2`)
- lightgbm (LGBMRanker)
- pandas, numpy

## Key Takeaways

- Category filtering was the single most impactful retrieval improvement
- Larger embedding models (bge-large, e5-large) did not improve results and were much slower
- RRF fusion consistently outperforms individual retrieval methods
- LightGBM reranking provided the biggest score jump: ~0.50 → 0.54

## Phase 2 Deadline

April 6, 2026, 11:59 PM
