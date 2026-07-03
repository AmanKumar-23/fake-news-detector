"""
Lightweight source-reputation lookup by domain.

This is an illustrative, editable list — not an authoritative ranking. It gives
the tool a real, explainable signal when the input is a URL: an article from a
wire service reads differently from one on a known fabrication site.

Extend REPUTABLE / LOW_CREDIBILITY for your own report or region.
"""
from urllib.parse import urlparse

REPUTABLE = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "economist.com", "nature.com", "science.org", "who.int", "nih.gov",
    "thehindu.com", "indianexpress.com", "ndtv.com", "pib.gov.in",
}

# Domains widely flagged by fact-checkers for fabricated or misleading content.
LOW_CREDIBILITY = {
    "yournewswire.com", "worldnewsdailyreport.com", "empirenews.net",
    "nationalreport.net", "theonion.com", "clickhole.com", "infowars.com",
    "beforeitsnews.com", "naturalnews.com",
}


def domain_of(url):
    """Return the registrable-ish host (drops leading www), or '' if not a URL."""
    if not url or "." not in url:
        return ""
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.netloc or "").lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def reputation(url):
    """
    Return a dict describing the source's reputation:
      tier:  'reputable' | 'low' | 'unknown'
      adjust: signed nudge to the credibility score (+/- points, 0 if unknown)
      note:  human-readable explanation
    """
    domain = domain_of(url)
    if not domain:
        return {"tier": "unknown", "domain": "", "adjust": 0,
                "note": "No source URL provided."}

    # Match domain or any parent (news.bbc.co.uk -> bbc.co.uk).
    parts = domain.split(".")
    candidates = {".".join(parts[i:]) for i in range(len(parts) - 1)}
    candidates.add(domain)

    if candidates & REPUTABLE:
        return {"tier": "reputable", "domain": domain, "adjust": +10,
                "note": f"{domain} is a generally reputable source."}
    if candidates & LOW_CREDIBILITY:
        return {"tier": "low", "domain": domain, "adjust": -25,
                "note": f"{domain} is frequently flagged for false or satirical content."}
    return {"tier": "unknown", "domain": domain, "adjust": 0,
            "note": f"{domain} is not in the reputation list — judge on content."}
