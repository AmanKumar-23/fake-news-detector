"""
Live fact-check lookup via the Google Fact Check Tools API.

Given article text, it derives a search query and asks Google's ClaimReview index
whether professional fact-checkers have already rated related claims. This surfaces
real, third-party verdicts (e.g. "False — PolitiFact") alongside the ML analysis —
directly serving the "trustworthy" goal of the problem statement.

Needs a free API key (GOOGLE_FACTCHECK_API_KEY). Without one, `check()` returns a
disabled result rather than raising, so the rest of the app keeps working.

Enable the API + create a key:
  https://console.cloud.google.com/  ->  "Fact Check Tools API"
"""
import os
import re
import sys
import json
import ssl
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
_STOP = set(
    "the a an and or but of to in on for with at by from is are was were be this that "
    "it as you your we they he she his her its their our will would can could not".split()
)


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def build_query(text, max_terms=6):
    """Pick the most salient content words as a search query."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
    seen, terms = set(), []
    for w in words:
        lw = w.lower()
        if lw in _STOP or lw in seen:
            continue
        seen.add(lw)
        terms.append(w)
        if len(terms) >= max_terms:
            break
    return " ".join(terms)


def check(text, api_key=None, language="en", max_results=5, timeout=12):
    """
    Return a dict:
      enabled: bool           — whether a key was available
      query:   str            — the query used
      claims:  list of {text, claimant, rating, publisher, url}
      note:    str            — human-readable status
    Never raises for normal use — network/API errors become a note.
    """
    api_key = api_key or config.FACTCHECK_API_KEY
    query = build_query(text)

    if not api_key:
        return {"enabled": False, "query": query, "claims": [],
                "note": "Fact-check API off — set GOOGLE_FACTCHECK_API_KEY to enable."}
    if not query:
        return {"enabled": True, "query": "", "claims": [],
                "note": "Not enough text to build a fact-check query."}

    params = urlencode({"query": query, "languageCode": language,
                        "pageSize": max_results, "key": api_key})
    req = Request(f"{API_URL}?{params}", headers={"User-Agent": "FakeNewsDetector/1.0"})
    try:
        with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        return {"enabled": True, "query": query, "claims": [],
                "note": f"Fact-check API error (HTTP {e.code}). Check the key/quota."}
    except (URLError, TimeoutError, ValueError) as e:
        return {"enabled": True, "query": query, "claims": [],
                "note": f"Could not reach the fact-check API: {e}"}

    claims = []
    for c in data.get("claims", [])[:max_results]:
        for review in c.get("claimReview", [])[:1]:
            claims.append({
                "text": c.get("text", "").strip(),
                "claimant": c.get("claimant", "").strip(),
                "rating": (review.get("textualRating") or "").strip(),
                "publisher": (review.get("publisher", {}) or {}).get("name", "").strip(),
                "url": review.get("url", ""),
            })

    note = (f"Found {len(claims)} related fact-check(s)." if claims
            else "No matching fact-checks found for this text.")
    return {"enabled": True, "query": query, "claims": claims, "note": note}


if __name__ == "__main__":
    sample = " ".join(sys.argv[1:]) or "The COVID vaccine contains a microchip tracking device."
    from pprint import pprint
    pprint(check(sample))
