"""
Shared text processing: normalization, readability, and rule-based credibility
signals. Kept dependency-free (stdlib only) so it works everywhere.
"""
import re

# Emotional / clickbait phrases common in fabricated or sensational content.
CLICKBAIT_PHRASES = [
    "shocking", "you won't believe", "won't believe", "wake up", "share before",
    "they don't want you to know", "doctors hate", "miracle", "exposed", "secret",
    "hoax", "conspiracy", "urgent", "before it gets deleted", "100% guaranteed",
    "doctors are furious", "the truth about", "forward to everyone", "gone viral",
    "what happens next", "this one trick", "big pharma", "mainstream media won't",
    "banned", "they lied", "cover up", "cover-up",
]

# Words signalling responsible sourcing / attribution.
SOURCING_TERMS = [
    "according to", "reported", "study", "research", "official", "reuters",
    "associated press", "spokesperson", "data", "statement", "peer-reviewed",
    "published", "researchers", "analysis", "confirmed", "evidence", "survey",
]

_WORD_RE = re.compile(r"[a-z']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def clean(text):
    """Collapse whitespace; keep content otherwise intact."""
    return re.sub(r"\s+", " ", (text or "").strip())


def words(text):
    return _WORD_RE.findall((text or "").lower())


def sentences(text):
    return [s for s in _SENT_RE.split((text or "").strip()) if s.strip()]


def caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def credibility_signals(text):
    """
    Return (flags, penalty) where flags is a list of human-readable red flags and
    penalty is in [0, 1] (higher = less credible). Deterministic and explainable.
    """
    flags = []
    low = (text or "").lower()

    cr = caps_ratio(text)
    if cr > 0.30:
        flags.append(f"Excessive capital letters ({cr:.0%} of letters)")

    n_excl = text.count("!")
    if n_excl >= 3:
        flags.append(f"Many exclamation marks ({n_excl})")

    hits = sorted({p for p in CLICKBAIT_PHRASES if p in low})
    if hits:
        shown = ", ".join(f'"{h}"' for h in hits[:4])
        more = f" (+{len(hits) - 4} more)" if len(hits) > 4 else ""
        flags.append(f"Clickbait / emotional phrases: {shown}{more}")

    if not any(term in low for term in SOURCING_TERMS):
        flags.append("No attribution to sources, officials, or data")

    n_words = len(text.split())
    if n_words < 15:
        flags.append("Very short — too little context to verify")

    # Count of ALL-CAPS words (SHOUTING) beyond the caps ratio check.
    shouts = len(re.findall(r"\b[A-Z]{3,}\b", text))
    if shouts >= 4:
        flags.append(f"Many all-caps words ({shouts})")

    # Five independent checks map to the penalty scale.
    penalty = min(len(flags) / 5.0, 1.0)
    return flags, penalty
