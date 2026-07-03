"""Central configuration — every path and tunable in one place."""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

DATASET_CSV = os.path.join(DATA_DIR, "news.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Transformer (DistilBERT) backend.
TRANSFORMER_DIR = os.path.join(MODEL_DIR, "distilbert")
TRANSFORMER_BASE = os.environ.get("FAKE_NEWS_TRANSFORMER_BASE", "distilbert-base-uncased")

# Which classifier backend to use: "sklearn" (default) or "distilbert".
# Falls back to sklearn automatically if the transformer isn't available.
MODEL_BACKEND = os.environ.get("FAKE_NEWS_BACKEND", "sklearn").lower()

# Google Fact Check Tools API key (optional, legacy). Get one at:
# https://console.cloud.google.com/  (enable "Fact Check Tools API")
FACTCHECK_API_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")

# Ollama — free, local LLM used for keyless AI fact-assessment.
# Install: https://ollama.com  then:  ollama pull llama3.2
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Real Kaggle "Fake and Real News" files, used automatically if present.
KAGGLE_FAKE = os.path.join(DATA_DIR, "Fake.csv")
KAGGLE_TRUE = os.path.join(DATA_DIR, "True.csv")

# Claude model for summaries (override with FAKE_NEWS_MODEL).
CLAUDE_MODEL = os.environ.get("FAKE_NEWS_MODEL", "claude-opus-4-8")

# Free local (offline) abstractive summarizer — a Hugging Face model that runs on
# your own machine, no API key or cost. Downloaded once on first use.
# Lighter alternatives: "sshleifer/distilbart-cnn-6-6", "t5-small".
SUMMARIZER_MODEL = os.environ.get("FAKE_NEWS_SUMMARIZER", "sshleifer/distilbart-cnn-12-6")

# Summary preference order: "auto" tries Claude (if key) -> local model -> extractive.
# Set to "local" to skip Claude entirely, or "extractive" to force the lightweight one.
SUMMARY_MODE = os.environ.get("FAKE_NEWS_SUMMARY_MODE", "auto").lower()

RANDOM_STATE = 42

# Blend weight: final fake-score = ALPHA * model_prob + (1-ALPHA) * rule_penalty.
RULE_BLEND = 0.25
