"""
AI fact-assessment using Ollama — a free, local LLM. No API key, no cost, runs on
your own machine. This replaces the paid/Google fact-check path: instead of looking
claims up in a database, it asks a local model to reason about the text's claims and
flag likely misinformation.

Setup (one time):
    1. Install Ollama:  https://ollama.com   (or: brew install ollama)
    2. Pull a model:    ollama pull llama3.2
    3. Ollama runs a local server at http://localhost:11434 automatically.

Everything degrades gracefully: if Ollama isn't running, check() returns a disabled
result with setup guidance rather than raising.
"""
import os
import sys
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

PROMPT = (
    "You are a careful media-literacy assistant helping a student judge whether a "
    "news item is trustworthy. Analyze the TEXT below. Do NOT use outside knowledge "
    "you are unsure about; reason from the text's claims, tone, and sourcing.\n\n"
    "Respond ONLY as JSON with these keys:\n"
    '  "verdict": one of "likely true", "likely false", "uncertain"\n'
    '  "confidence": integer 0-100\n'
    '  "reasoning": 2-3 sentence explanation\n'
    '  "suspicious_claims": list of short strings (specific claims that need verification)\n\n'
    "TEXT:\n"
)


def is_available(timeout=3):
    """Return (ok, note). Checks the Ollama server and whether the model is present."""
    try:
        req = Request(f"{config.OLLAMA_URL}/api/tags")
        with urlopen(req, timeout=timeout) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, ValueError):
        return False, ("Ollama not running. Install from https://ollama.com, then run "
                       f"`ollama pull {config.OLLAMA_MODEL}`.")
    models = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
    if config.OLLAMA_MODEL.split(":")[0] not in models:
        return False, (f"Ollama is running but model '{config.OLLAMA_MODEL}' isn't pulled. "
                       f"Run: ollama pull {config.OLLAMA_MODEL}")
    return True, "ok"


def check(text, model=None, timeout=120):
    """
    Ask the local LLM to assess the text. Returns a dict:
      enabled:  bool
      model:    str
      verdict:  str | ""
      confidence: int | None
      reasoning: str
      suspicious_claims: list[str]
      note:     str
    Never raises for normal use.
    """
    model = model or config.OLLAMA_MODEL
    ok, note = is_available()
    if not ok:
        return {"enabled": False, "model": model, "verdict": "", "confidence": None,
                "reasoning": "", "suspicious_claims": [], "note": note}

    body = json.dumps({
        "model": model,
        "prompt": PROMPT + text.strip()[:6000],
        "stream": False,
        "format": "json",          # ask Ollama to return valid JSON
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = Request(f"{config.OLLAMA_URL}/api/generate", data=body,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, ValueError) as e:
        return {"enabled": True, "model": model, "verdict": "", "confidence": None,
                "reasoning": "", "suspicious_claims": [],
                "note": f"Ollama request failed: {e}"}

    # Ollama returns the model's text in data["response"]; it should be JSON.
    try:
        parsed = json.loads(data.get("response", "{}"))
    except ValueError:
        parsed = {}

    claims = parsed.get("suspicious_claims") or []
    if isinstance(claims, str):
        claims = [claims]
    return {
        "enabled": True,
        "model": model,
        "verdict": str(parsed.get("verdict", "")).strip(),
        "confidence": parsed.get("confidence"),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "suspicious_claims": [str(c).strip() for c in claims if str(c).strip()][:6],
        "note": "AI assessment by local Ollama model.",
    }


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or (
        "SHOCKING: scientists confirm the moon is made of cheese and NASA has been "
        "hiding it for decades. Share before this gets deleted!"
    )
    from pprint import pprint
    pprint(check(sample))
