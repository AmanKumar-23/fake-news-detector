"""
Produce a concise summary of an article, using the best available option.

Three tiers, tried in order (see config.SUMMARY_MODE):
  1. Claude API      — best quality, needs ANTHROPIC_API_KEY (paid).
  2. Local AI model  — FREE. A Hugging Face summarizer (distilbart) that runs on
                       your own machine; no key, no cost, offline after a one-time
                       model download. Requires `pip install transformers torch`.
  3. Extractive      — FREE, always works, stdlib only. Picks the most
                       representative sentences.

`summarize()` never raises for normal input — it degrades down the tiers.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

MODEL = config.CLAUDE_MODEL

SYSTEM_PROMPT = (
    "You are a media-literacy assistant for students. Given a news article, write "
    "a concise, strictly neutral summary. Do not add opinions or facts not present "
    "in the text. Then give one short line assessing how the article reads in terms "
    "of credibility signals (sourcing, tone, verifiability) — without claiming "
    "certainty. Keep the whole response under 120 words."
)

_local_tok = None
_local_model = None


def _has_api_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ----------------------------------------------------------------- Claude (paid)
def claude_summary(text):
    """Summarize via the Claude API. Raises if the SDK/key is unavailable."""
    import anthropic  # lazy import so the free paths never need the package

    client = anthropic.Anthropic()
    article = text.strip()[:12000]
    message = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Summarize this article and assess its credibility:\n\n{article}",
        }],
    )
    return "".join(b.text for b in message.content if b.type == "text").strip()


# --------------------------------------------------------- local AI model (free)
def _get_local_model():
    """
    Lazily load and cache a seq2seq summarizer (tokenizer + model).

    Loads the model directly rather than via pipeline("summarization"), because
    that pipeline task was removed in transformers 5.x. `.generate()` works across
    versions.
    """
    global _local_tok, _local_model
    if _local_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # lazy import
        _local_tok = AutoTokenizer.from_pretrained(config.SUMMARIZER_MODEL)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(config.SUMMARIZER_MODEL)
        _local_model.eval()
    return _local_tok, _local_model


def local_summary(text):
    """
    Free, offline abstractive summary via a local Hugging Face model.
    Raises if transformers/torch aren't installed (caller falls back).
    """
    import torch
    tok, model = _get_local_model()
    article = re.sub(r"\s+", " ", text.strip())
    n_words = len(article.split())
    if n_words < 40:
        return extractive_summary(text)  # too short to abstract meaningfully

    inputs = tok(article, truncation=True, max_length=1024, return_tensors="pt")
    max_len = min(140, max(30, n_words // 2))
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_length=max_len, min_length=20,
            num_beams=4, no_repeat_ngram_size=3, early_stopping=True,
        )
    return tok.decode(ids[0], skip_special_tokens=True).strip()


# -------------------------------------------------------------- extractive (free)
def extractive_summary(text, max_sentences=3):
    """Free, stdlib-only fallback: score sentences by word frequency, keep the top few."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if len(s.split()) >= 4]
    if len(sentences) <= max_sentences:
        return text.strip()

    words = re.findall(r"[a-z']+", text.lower())
    stop = set(
        "the a an and or but of to in on for with at by from is are was were be this "
        "that it as you your we they he she his her its their our".split()
    )
    freq = {}
    for w in words:
        if w not in stop and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    def score(sent):
        sw = re.findall(r"[a-z']+", sent.lower())
        return sum(freq.get(w, 0) for w in sw) / (len(sw) or 1)

    ranked = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    keep = sorted(ranked[:max_sentences])
    return " ".join(sentences[i] for i in keep)


# ------------------------------------------------------------------- dispatcher
def summarize(text):
    """
    Return (summary_text, source_label). Never raises for normal input.
    Order controlled by config.SUMMARY_MODE: auto | local | extractive.
    """
    mode = config.SUMMARY_MODE

    if mode == "extractive":
        return extractive_summary(text), "extractive"

    if mode != "local" and _has_api_key():
        try:
            return claude_summary(text), "Claude"
        except Exception as e:
            return f"{extractive_summary(text)}\n\n(Claude unavailable: {e})", "extractive"

    # Free local AI model (the default when there's no Claude key).
    try:
        return local_summary(text), "local-AI (distilbart)"
    except Exception:
        # transformers/torch not installed, or model download failed → always works.
        return extractive_summary(text), "extractive"


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or (
        "The health ministry said on Tuesday that a new vaccine had completed trials. "
        "According to the official statement, an independent panel reviewed six months "
        "of data before recommending approval. Experts cautioned that monitoring would "
        "continue after rollout, citing standard safety procedures. Officials added that "
        "the data would be published for independent review by outside scientists."
    )
    summary, src = summarize(sample)
    print(f"[source: {src}]\n{summary}")
