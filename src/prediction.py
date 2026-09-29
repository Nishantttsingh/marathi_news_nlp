import numpy as np
from src.preprocessing import process

DEVANAGARI = range(0x0900, 0x0980)


def predict(bundle, text, model_name=None, top_k=3):
    """Returns (result, error). `error` is a user-facing message or None."""
    if text is None or not str(text).strip():
        return None, "Input is empty. Enter a Marathi news headline or article."
    trace = process(text)
    if not any(ord(c) in DEVANAGARI for c in trace["original"]):
        return None, "The input contains no Devanagari (Marathi) characters."
    if not trace["tokens_after"]:
        return None, "No usable tokens remain after preprocessing (input is only stopwords/punctuation)."
    name = model_name or bundle["best"]
    pipe = bundle["pipelines"][name]
    clf = pipe.named_steps["clf"]
    pred = pipe.predict([str(text)])[0]
    res = {"model": name, "prediction": pred, "trace": trace, "warning": None,
           "short_input": len(trace["tokens_after"]) < 3}
    x = pipe.named_steps["tfidf"].transform([str(text)])
    if hasattr(clf, "predict_proba"):
        s, res["score_type"] = clf.predict_proba(x)[0], "Probability"
    else:
        s, res["score_type"] = clf.decision_function(x)[0], "Decision score (not a probability)"
    order = np.argsort(s)[::-1][:top_k]
    res["top"] = [(clf.classes_[i], float(s[i])) for i in order]
    res["oov_ratio"] = 1 - x.nnz / max(1, len(trace["tokens_after"]))
    res["known_features"] = int(x.nnz)
    if x.nnz == 0:
        res["warning"] = "None of the input's terms are in the training vocabulary; the prediction reflects class priors/bias only."
    return res, None
