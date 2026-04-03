"""
Train TF-IDF + LogisticRegression document-category classifier (Phase 2).
Matches IR_Project_Phase2 notebook settings; fits on the full document set for deployment.
Outputs: classifier.pkl, tfidf_clf.pkl
"""
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ir_data_paths import DATA_DIR
from phase2_utils import merge_fields


def main() -> None:
    print("=" * 60)
    print("TRAIN QUERY/DOC CATEGORY CLASSIFIER (Phase 2)")
    print("=" * 60)

    df_docs = pd.read_json(DATA_DIR / "docs.json")
    df_docs["content"] = df_docs.apply(merge_fields, axis=1)
    doc_labels = df_docs["category"].tolist()
    doc_contents = df_docs["content"].tolist()

    tfidf_vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X_docs = tfidf_vectorizer.fit_transform(doc_contents)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X_docs, doc_labels)

    with open("classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("tfidf_clf.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    print(f"Saved classifier.pkl ({len(doc_labels):,} documents, {len(clf.classes_)} classes)")
    print(f"Saved tfidf_clf.pkl (max_features=10000)")
    print(f"Classes: {list(clf.classes_)}")


if __name__ == "__main__":
    main()
