"""
Pluggable classifier backends behind one interface.

Both backends expose the same two methods, so predict.py never needs to know
which model is running:

    backend.prob_fake(text)  -> float in [0, 1]
    backend.explain(text)    -> (fake_words, real_words)

Select with FAKE_NEWS_BACKEND=sklearn|distilbert. The transformer backend falls
back to sklearn automatically if torch/transformers or the fine-tuned weights are
missing — so the app never hard-fails just because the heavy option isn't set up.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


class Backend:
    name = "base"

    def prob_fake(self, text):
        raise NotImplementedError

    def explain(self, text, k=6):
        """Return (fake_words, real_words). May be empty for non-linear models."""
        return [], []


class SklearnBackend(Backend):
    name = "sklearn (TF-IDF + Logistic Regression)"

    def __init__(self):
        import joblib
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"No model at {config.MODEL_PATH}. Train first: python src/train.py"
            )
        self.model = joblib.load(config.MODEL_PATH)

    def prob_fake(self, text):
        return float(self.model.predict_proba([text])[0][1])

    def explain(self, text, k=6):
        features = self.model.named_steps["features"]
        clf = self.model.named_steps["clf"]
        x = features.transform([text]).toarray()[0]
        contrib = x * clf.coef_[0]
        names = features.get_feature_names_out()
        scored = [
            (names[i][len("word__"):], contrib[i])
            for i, n in enumerate(names)
            if n.startswith("word__") and contrib[i] != 0
        ]
        fake_words = [w for w, c in sorted(scored, key=lambda p: p[1], reverse=True) if c > 0][:k]
        real_words = [w for w, c in sorted(scored, key=lambda p: p[1]) if c < 0][:k]
        return fake_words, real_words


class TransformerBackend(Backend):
    name = "distilbert (fine-tuned transformer)"

    def __init__(self):
        # Imported lazily so the classic path never needs torch.
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
        )
        if not os.path.isdir(config.TRANSFORMER_DIR):
            raise FileNotFoundError(
                f"No fine-tuned model at {config.TRANSFORMER_DIR}. "
                "Train it: python src/train_transformer.py"
            )
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.TRANSFORMER_DIR)
        self.model.eval()

    def prob_fake(self, text):
        torch = self._torch
        enc = self.tokenizer(
            text, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
        # Convention: label 1 == fake (set at training time).
        return float(probs[1])

    # Explainability for a transformer would need attention/attribution; we return
    # empty and let the rule-based signals carry the "why" instead of faking it.
    def explain(self, text, k=6):
        return [], []


_backend = None
_active_name = None


def get_backend():
    """Return the configured backend, falling back to sklearn if needed."""
    global _backend, _active_name
    if _backend is not None:
        return _backend

    want = config.MODEL_BACKEND
    if want == "distilbert":
        try:
            _backend = TransformerBackend()
        except Exception as e:
            sys.stderr.write(
                f"[classifier] DistilBERT backend unavailable ({e}); "
                "falling back to sklearn.\n"
            )
            _backend = SklearnBackend()
    else:
        _backend = SklearnBackend()

    _active_name = _backend.name
    return _backend


def active_backend_name():
    get_backend()
    return _active_name


if __name__ == "__main__":
    b = get_backend()
    txt = " ".join(sys.argv[1:]) or "SHOCKING secret they are HIDING!!!"
    print("backend:", b.name)
    print("P(fake):", round(b.prob_fake(txt), 4))
    print("explain:", b.explain(txt))
