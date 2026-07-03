"""
Test suite for the Fake News Detector.

Run:  pytest -q

Tests that need the trained model auto-skip if it hasn't been built yet, so the
pure-logic tests (signals, reputation, parsing, summary fallback) always run.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config  # noqa: E402
import text_utils  # noqa: E402
import source_reputation as rep  # noqa: E402
import summarize  # noqa: E402
import factcheck  # noqa: E402
import ocr  # noqa: E402
import ollama_check  # noqa: E402
from url_fetch import _ArticleParser  # noqa: E402

MODEL_EXISTS = os.path.exists(config.MODEL_PATH)
needs_model = pytest.mark.skipif(not MODEL_EXISTS, reason="model not trained yet")


# ---------------------------------------------------------------- text_utils
def test_caps_ratio_detects_shouting():
    assert text_utils.caps_ratio("THIS IS ALL SHOUTING TEXT HERE") > 0.5
    assert text_utils.caps_ratio("this is normal lower case text here") < 0.1


def test_signals_flag_clickbait():
    flags, penalty = text_utils.credibility_signals(
        "SHOCKING!! You won't BELIEVE this SECRET they are HIDING!!! Share before deleted!!!"
    )
    assert penalty > 0.5
    assert any("Clickbait" in f for f in flags)
    assert any("capital" in f.lower() for f in flags)


def test_signals_clean_on_sourced_text():
    flags, penalty = text_utils.credibility_signals(
        "According to a peer-reviewed study published by researchers, the data was "
        "reviewed by an independent panel before the conclusions were confirmed."
    )
    assert penalty == 0.0
    assert flags == []


def test_sentences_split():
    assert len(text_utils.sentences("One. Two! Three?")) == 3


# ---------------------------------------------------------------- reputation
def test_domain_extraction():
    assert rep.domain_of("https://www.reuters.com/world/india") == "reuters.com"
    assert rep.domain_of("news.bbc.co.uk/story") == "news.bbc.co.uk"
    assert rep.domain_of("not a url") == ""


def test_reputable_source_boosts():
    r = rep.reputation("https://www.reuters.com/article")
    assert r["tier"] == "reputable" and r["adjust"] > 0


def test_low_credibility_source_penalized():
    r = rep.reputation("https://worldnewsdailyreport.com/story")
    assert r["tier"] == "low" and r["adjust"] < 0


def test_subdomain_matches_parent():
    r = rep.reputation("https://feeds.bbc.co.uk/news")
    assert r["tier"] == "reputable"


def test_unknown_source_neutral():
    r = rep.reputation("https://some-random-blog.example/post")
    assert r["tier"] == "unknown" and r["adjust"] == 0


# ---------------------------------------------------------------- html parsing
def test_article_parser_extracts_paragraphs_and_skips_script():
    html = (
        "<html><head><title>My Title</title></head><body>"
        "<script>var x = 'ignore this script content entirely please';</script>"
        "<p>This is a real paragraph with enough words to be kept here.</p>"
        "<div>Short</div>"
        "<p>Another substantial paragraph of article text for the reader here.</p>"
        "</body></html>"
    )
    p = _ArticleParser()
    p.feed(html)
    text = p.text()
    assert p.title.strip() == "My Title"
    assert "real paragraph" in text
    assert "ignore this script" not in text
    assert "Short" not in text  # too few words, dropped


# ---------------------------------------------------------------- summarize
def test_extractive_summary_shortens_long_text():
    long_text = " ".join(
        f"Sentence number {i} discusses the ministry report and the reviewed data."
        for i in range(10)
    )
    summary = summarize.extractive_summary(long_text, max_sentences=3)
    assert 0 < len(summary) < len(long_text)


def test_summarize_extractive_mode_returns_source(monkeypatch):
    # Force the lightweight tier so the test never downloads the local AI model.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(summarize.config, "SUMMARY_MODE", "extractive")
    summary, source = summarize.summarize(
        "The ministry said the data was reviewed. Officials confirmed the findings. "
        "Experts urged caution. Monitoring will continue after the rollout period."
    )
    assert source == "extractive"
    assert isinstance(summary, str) and summary


def test_summarize_dispatcher_always_returns_tuple(monkeypatch):
    monkeypatch.setattr(summarize.config, "SUMMARY_MODE", "extractive")
    out = summarize.summarize("Short neutral text about a policy review process here today.")
    assert isinstance(out, tuple) and len(out) == 2


# ---------------------------------------------------------------- model-backed
@needs_model
def test_fake_text_scored_low():
    from predict import analyze
    r = analyze("SHOCKING!! You won't BELIEVE the SECRET they are HIDING!!! Share before DELETED!!!")
    assert r["is_fake"] is True
    assert r["credibility_score"] < 40


@needs_model
def test_real_text_scored_high():
    from predict import analyze
    r = analyze(
        "According to a report published by the ministry on Tuesday, officials reviewed "
        "the water supply data over six months before outlining the next steps."
    )
    assert r["is_fake"] is False
    assert r["credibility_score"] > 60


@needs_model
def test_reputable_source_raises_score():
    from predict import analyze
    text = "According to a report, officials reviewed the data before outlining next steps."
    base = analyze(text)["credibility_score"]
    boosted = analyze(text, url="https://www.reuters.com/x")["credibility_score"]
    assert boosted >= base


@needs_model
def test_result_schema_is_complete():
    from predict import analyze
    r = analyze("Some neutral article text about the budget review process here.")
    for key in ("verdict", "is_fake", "credibility_score", "model_prob_fake",
                "red_flags", "fake_indicators", "real_indicators", "source", "backend"):
        assert key in r


# ---------------------------------------------------------------- fact-check
def test_factcheck_query_builder_drops_stopwords():
    q = factcheck.build_query("The government announced a new education policy today.")
    assert "the" not in q.lower().split()
    assert "government" in q and "education" in q


def test_factcheck_disabled_without_key(monkeypatch):
    monkeypatch.setattr(config, "FACTCHECK_API_KEY", "")
    out = factcheck.check("The vaccine contains a secret microchip.")
    assert out["enabled"] is False
    assert out["claims"] == []
    assert "GOOGLE_FACTCHECK_API_KEY" in out["note"]


def test_factcheck_result_shape_is_stable(monkeypatch):
    monkeypatch.setattr(config, "FACTCHECK_API_KEY", "")
    out = factcheck.check("Anything here at all.")
    for key in ("enabled", "query", "claims", "note"):
        assert key in out


# ---------------------------------------------------------------- backends
def test_default_backend_is_sklearn():
    import importlib
    import classifier
    importlib.reload(classifier)
    monkey_backend = classifier.SklearnBackend if MODEL_EXISTS else None
    if MODEL_EXISTS:
        assert isinstance(classifier.get_backend(), monkey_backend)


@needs_model
def test_unknown_backend_falls_back_to_sklearn(monkeypatch):
    import importlib
    import classifier
    importlib.reload(classifier)
    monkeypatch.setattr(classifier.config, "MODEL_BACKEND", "distilbert")
    # No fine-tuned model guaranteed in CI -> must fall back, not raise.
    b = classifier.get_backend()
    assert b is not None and hasattr(b, "prob_fake")


# ---------------------------------------------------------------- OCR
def test_ocr_roundtrip_or_skip():
    """Render text to an image, OCR it back. Skips if no OCR engine installed."""
    if ocr.available_engine() is None:
        pytest.skip("no OCR engine installed")
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGB", (600, 120), "white")
    ImageDraw.Draw(img).text((20, 40), "BREAKING NEWS HEADLINE TEXT", fill="black")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    text = ocr.extract_text(buf.getvalue())
    assert "NEWS" in text.upper() or "BREAKING" in text.upper()


def test_ocr_missing_engine_raises(monkeypatch):
    monkeypatch.setattr(ocr, "available_engine", lambda: None)
    with pytest.raises(ocr.OCRError):
        ocr.extract_text(b"not-an-image")


# ---------------------------------------------------------------- Ollama
def test_ollama_graceful_when_unavailable(monkeypatch):
    # Point at a dead port so it can't connect -> disabled result, no exception.
    monkeypatch.setattr(ollama_check.config, "OLLAMA_URL", "http://127.0.0.1:1")
    out = ollama_check.check("The moon is made of cheese.")
    assert out["enabled"] is False
    assert "verdict" in out and out["suspicious_claims"] == []


def test_ollama_result_shape_is_stable(monkeypatch):
    monkeypatch.setattr(ollama_check.config, "OLLAMA_URL", "http://127.0.0.1:1")
    out = ollama_check.check("Any text.")
    for key in ("enabled", "model", "verdict", "confidence", "reasoning",
                "suspicious_claims", "note"):
        assert key in out
