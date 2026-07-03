"""
Score an article: verdict, credibility score, red flags, explainability, and
(optional) source reputation.

Credibility score (0-100, higher = more trustworthy) blends three signals:
  1. the model's probability the article is REAL,
  2. rule-based red flags (clickbait, ALL CAPS, missing sources), and
  3. source reputation when a URL is provided.
Blending keeps the tool sensible even when the model is unsure, and every input
to the score is shown to the user so the verdict is explainable.

The underlying classifier is pluggable (see classifier.py): the default is the
TF-IDF + Logistic Regression model; set FAKE_NEWS_BACKEND=distilbert to use the
fine-tuned transformer. `analyze()`'s output shape is identical either way.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from text_utils import credibility_signals, clean  # noqa: E402
from source_reputation import reputation  # noqa: E402
from classifier import get_backend, active_backend_name  # noqa: E402

MODEL_PATH = config.MODEL_PATH


def analyze(text, url=None):
    """Analyze `text` (optionally attributed to `url`); return a result dict."""
    text = clean(text)
    backend = get_backend()
    prob_fake = backend.prob_fake(text)

    flags, penalty = credibility_signals(text)
    # Blend model probability with the rule-based penalty.
    fake_score = min(1.0, (1 - config.RULE_BLEND) * prob_fake + config.RULE_BLEND * penalty)
    credibility = (1.0 - fake_score) * 100.0

    rep = reputation(url) if url else reputation(None)
    credibility = max(0.0, min(100.0, credibility + rep["adjust"]))

    is_fake = credibility < 50.0
    verdict = "Likely FAKE" if is_fake else "Likely REAL"
    fake_words, real_words = backend.explain(text)

    return {
        "verdict": verdict,
        "is_fake": is_fake,
        "credibility_score": round(credibility, 1),
        "model_prob_fake": round(prob_fake, 4),
        "rule_penalty": round(penalty, 2),
        "red_flags": flags,
        "fake_indicators": fake_words,
        "real_indicators": real_words,
        "source": rep,
        "backend": active_backend_name(),
    }


if __name__ == "__main__":
    from pprint import pprint
    args = sys.argv[1:]
    url = None
    if args and args[0].startswith("--url="):
        url = args.pop(0)[len("--url="):]
    sample = " ".join(args) or "SHOCKING!! You won't BELIEVE this secret they are HIDING!!!"
    pprint(analyze(sample, url=url))
