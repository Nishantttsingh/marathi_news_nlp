"""Run: python tests/verify.py   (checks the items in the project checklist)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src import models, prediction, preprocessing as pp
from utils import data_loader

d = data_loader.load_dataset(ROOT / "data" / "combined_train.csv")
df, T, L = d["df"], d["text_col"], d["label_col"]
B = models.train_all(df, T, L)
print("columns:", T, L, "| rows:", len(df), "| classes:", B["classes"])
print("split:", B["n_train"], B["n_test"])
# same test set for all models + independent metric recomputation from the pipelines
for n, p in B["pipelines"].items():
    yp = p.predict(B["X_test"])
    assert (yp == B["results"][n]["y_pred"]).all(), n
    assert abs(accuracy_score(B["y_test"], yp) - B["results"][n]["accuracy"]) < 1e-12
    assert abs(f1_score(B["y_test"], yp, average="macro") - B["results"][n]["f1"]) < 1e-12
    assert B["results"][n]["confusion"].sum() == B["n_test"]
    print(f"{n:26s} acc={B['results'][n]['accuracy']:.4f} macroF1={B['results'][n]['f1']:.4f}")
assert B["best"] == max(B["results"], key=lambda m: B["results"][m]["f1"]); print("selected:", B["best"])
# leakage: no train/test text overlap, vectorizer vocabulary comes from train only
assert not set(B["X_test"]) & set(df.loc[~df[T].isin(set(B["X_test"])), T]); 
vec = B["pipelines"][B["best"]].named_steps["tfidf"]
an = vec.build_analyzer()
test_only = set(t for x in B["X_test"] for t in an(x)) - set(t for x in df.loc[~df[T].isin(set(B["X_test"])), T] for t in an(x))
assert not (test_only & set(vec.vocabulary_)), "test-only terms found in vocabulary"
# preprocessing consistency: UI trace == vectorizer unigrams
for x in df[T].sample(500, random_state=0):
    assert [w for w in an(x) if " " not in w and w in vec.vocabulary_] == [w for w in pp.process(x)["tokens_after"] if w in vec.vocabulary_]
print("no leakage; UI preprocessing == model preprocessing (500 samples)")
tests = ["मुंबईमध्ये आज मुसळधार पावसामुळे अनेक भागांमध्ये वाहतूक विस्कळीत झाली.",
         "भारतीय संघाने ऑस्ट्रेलियाविरुद्धचा क्रिकेट सामना जिंकला.",
         "शेअर बाजारात आज सेन्सेक्समध्ये मोठी घसरण झाली.",
         "अभिनेत्याच्या नव्या चित्रपटाचा ट्रेलर प्रदर्शित झाला.",
         "दहावीच्या परीक्षेचा निकाल जाहीर झाला, विद्यार्थ्यांना गुणपत्रिका मिळणार.",
         "मधुमेह टाळण्यासाठी रोज व्यायाम करण्याचा डॉक्टरांचा सल्ला."]
for t in tests:
    for m in B["pipelines"]:
        r, e = prediction.predict(B, t, m); assert e is None, e
    r, _ = prediction.predict(B, t)
    print(f"{r['prediction']:14s} {r['score_type'][:11]} {r['top'][0][1]:.3f} <- {t[:45]}")
for bad in ["", "   ", None, "hello world", "आहे आणि ।"]:
    r, e = prediction.predict(B, bad); assert r is None and e; print("error ok:", repr(bad), "->", e)
r, e = prediction.predict(B, "पाऊस"); print("short input ok:", r["prediction"], r["short_input"])
print("ALL CHECKS PASSED")
