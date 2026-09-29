import hashlib
from pathlib import Path
import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src import evaluation, feature_extraction as fe
from src.preprocessing import STOPWORD_FILE

SEED, TEST_SIZE = 42, 0.20
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def make_classifiers():
    # Library-default C / alpha (no tuning on the test set). class_weight='balanced' is used where
    # supported because the dataset is strongly imbalanced; MultinomialNB has no such option.
    return {"Multinomial Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            "Linear SVM": LinearSVC(class_weight="balanced", random_state=SEED)}


def _fingerprint(df, text_col, label_col):
    h = hashlib.sha256()
    h.update(df[[text_col, label_col]].to_csv(index=False).encode("utf-8"))
    h.update(STOPWORD_FILE.read_bytes() if STOPWORD_FILE.exists() else b"")
    h.update(f"{SEED}{TEST_SIZE}{sklearn.__version__}".encode())
    return h.hexdigest()[:16]


def train_all(df, text_col, label_col):
    """One stratified split -> one TF-IDF fit on train only -> three classifiers -> same test set."""
    key = _fingerprint(df, text_col, label_col)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    path = ARTIFACT_DIR / f"bundle_{key}.joblib"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            path.unlink(missing_ok=True)
    X_tr, X_te, y_tr, y_te = train_test_split(df[text_col], df[label_col], test_size=TEST_SIZE,
                                              stratify=df[label_col], random_state=SEED)
    classes = sorted(df[label_col].unique())
    vec = fe.build_vectorizer()
    Xtr = vec.fit_transform(X_tr)          # fit on TRAIN only
    Xte = vec.transform(X_te)              # transform-only on TEST
    results, pipelines = {}, {}
    for name, clf in make_classifiers().items():
        clf.fit(Xtr, y_tr)
        results[name] = evaluation.evaluate(y_te, clf.predict(Xte), classes)
        pipelines[name] = Pipeline([("tfidf", vec), ("clf", clf)])   # already fitted; used for user prediction
    best = evaluation.select_best(results)
    bundle = {"key": key, "classes": classes, "pipelines": pipelines, "results": results, "best": best,
              "table": evaluation.comparison_table(results, best), "tfidf_info": fe.describe(vec, Xtr, Xte),
              "X_test": X_te.reset_index(drop=True), "y_test": y_te.reset_index(drop=True),
              "train_counts": y_tr.value_counts().reindex(classes), "test_counts": y_te.value_counts().reindex(classes),
              "n_train": len(X_tr), "n_test": len(X_te), "seed": SEED, "test_size": TEST_SIZE}
    joblib.dump(bundle, path)
    return bundle
