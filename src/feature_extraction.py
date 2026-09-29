import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import normalize, tokenize_and_filter

TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def build_vectorizer() -> TfidfVectorizer:
    # preprocessor -> normalize(); tokenizer -> tokenize + stopword removal (shared with the UI).
    return TfidfVectorizer(preprocessor=normalize, tokenizer=tokenize_and_filter,
                           token_pattern=None, lowercase=False, **TFIDF_PARAMS)


def describe(vec, X_train, X_test, n_top=15):
    terms = np.array(vec.get_feature_names_out())
    mean_w = np.asarray(X_train.mean(axis=0)).ravel()
    top = np.argsort(mean_w)[::-1][:n_top]
    return {"vocab_size": len(terms), "train_shape": X_train.shape, "test_shape": X_test.shape,
            "nnz_per_doc": float(X_train.nnz / X_train.shape[0]),
            "density_pct": float(100 * X_train.nnz / (X_train.shape[0] * X_train.shape[1])),
            "top_terms": [(terms[i], float(mean_w[i])) for i in top], "params": vec.get_params()
            | {"preprocessor": "preprocessing.normalize", "tokenizer": "preprocessing.tokenize_and_filter"}}


def sample_features(vec, text, n=12):
    """Non-zero TF-IDF weights of one document (readable sample of the sparse vector)."""
    row = vec.transform([text])
    terms = vec.get_feature_names_out()
    idx = row.indices[np.argsort(row.data)[::-1][:n]]
    return [(terms[i], float(row[0, i])) for i in idx], row.shape, row.nnz
