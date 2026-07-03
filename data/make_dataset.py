"""
Build a labeled news dataset for training.

Priority:
  1. If the real Kaggle "Fake and Real News" files (Fake.csv / True.csv) are found
     in data/, use them (recommended for real, defensible accuracy).
  2. Otherwise generate a realistic synthetic dataset. It is intentionally *not*
     trivially separable: it includes borderline cases and a little label noise so
     the reported metrics and probability calibration are believable rather than a
     misleading 100%.

Output: data/news.csv  with columns [text, label]  (label: 1 = fake, 0 = real)

Get the real dataset (optional, better accuracy):
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  Drop Fake.csv and True.csv into data/, then re-run this script.
"""
import os
import sys
import random
import itertools

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config  # noqa: E402

TOPICS = [
    "the new education policy", "the city water supply", "a popular vaccine",
    "the upcoming elections", "the stock market", "a celebrity's health",
    "climate change", "the national budget", "5G technology", "electric vehicles",
    "a university admission process", "the local hospital", "artificial intelligence",
    "the new highway project", "online banking", "a food safety recall",
    "the monsoon forecast", "a cricket tournament", "the metro expansion",
    "a new smartphone", "the census results", "a mental-health campaign",
]

# Clearly FAKE: sensational, emotional, unsourced, urgent, conspiratorial.
FAKE_TEMPLATES = [
    "SHOCKING!! You won't BELIEVE what they are HIDING about {t}! Doctors are FURIOUS. Share before it gets DELETED!!!",
    "BREAKING: Secret documents PROVE {t} is a total HOAX. The government does NOT want you to know this SHOCKING truth!!",
    "They LIED to you about {t}! This ONE weird trick exposes the conspiracy. Mainstream media is SILENT. WAKE UP!!!",
    "EXPOSED: {t} is causing a deadly crisis that officials are COVERING UP. 100% guaranteed. Forward to everyone NOW!!",
    "URGENT WARNING about {t}!! Insiders reveal a terrifying secret plan. No sources needed, just TRUST me. Act fast!!!",
    "MIRACLE discovery about {t} that BIG PHARMA is desperate to ban! Anonymous experts confirm the unbelievable.",
    "Viral post: {t} will DESTROY your life unless you do THIS immediately. Everyone is talking about this scandal!!!",
    "This is what THEY don't want you to see about {t}. Banned everywhere. Wake up before it's too late!!!",
]

# Clearly REAL: measured, sourced, attributed, factual.
REAL_TEMPLATES = [
    "According to a report published by the ministry on Tuesday, officials reviewed {t} and outlined the next steps.",
    "Researchers at the university released a peer-reviewed study on {t}, noting the findings require further validation.",
    "The committee said in a statement that data on {t} was analyzed over six months before the conclusions were shared.",
    "A spokesperson confirmed that {t} would be examined by an independent panel, according to the official filing.",
    "Reuters reported that experts discussed {t} at a conference, citing figures from the national statistics office.",
    "The local authority announced updates on {t}, adding that public feedback will be collected before any decision.",
    "Analysts noted moderate changes related to {t}, referencing quarterly data and cautioning against overinterpretation.",
    "Officials briefed reporters on {t}, and the associated data was published on the department's website for review.",
]

# HARD cases: overlap the two styles so the model must learn nuance, not keywords.
# Fake but calm-sounding (no obvious clickbait):
HARD_FAKE = [
    "Sources close to the matter suggest {t} may be secretly manipulated, though no evidence has been made public.",
    "It is being widely shared that {t} is not what it seems, according to unnamed insiders on social media.",
    "Some claim {t} has hidden consequences the authorities refuse to discuss, but the details remain unverified.",
]
# Real but brief / a little emotional (harder to recognize as credible):
HARD_REAL = [
    "The minister said {t} is important and urged citizens to stay informed, according to the press briefing.",
    "Officials called {t} a serious concern and promised a detailed report, the agency confirmed on Monday.",
    "The report on {t} was welcomed by experts, who said the evidence, while limited, points in a clear direction.",
]


def load_real():
    if not (os.path.exists(config.KAGGLE_FAKE) and os.path.exists(config.KAGGLE_TRUE)):
        return None
    fake = pd.read_csv(config.KAGGLE_FAKE)
    true = pd.read_csv(config.KAGGLE_TRUE)

    def combine(df):
        title = df["title"] if "title" in df else ""
        body = df["text"] if "text" in df else ""
        return (title.fillna("") + ". " + body.fillna("")).str.strip()

    fake_txt, true_txt = combine(fake), combine(true)
    df = pd.DataFrame({
        "text": pd.concat([fake_txt, true_txt], ignore_index=True),
        "label": [1] * len(fake_txt) + [0] * len(true_txt),
    })
    return df[df["text"].str.len() > 20].reset_index(drop=True)


def make_synthetic(noise_rate=0.06):
    rng = random.Random(config.RANDOM_STATE)
    rows = []

    for t, tmpl in itertools.product(TOPICS, FAKE_TEMPLATES):
        rows.append([tmpl.format(t=t), 1])
    for t, tmpl in itertools.product(TOPICS, REAL_TEMPLATES):
        rows.append([tmpl.format(t=t), 0])
    for t, tmpl in itertools.product(TOPICS, HARD_FAKE):
        rows.append([tmpl.format(t=t), 1])
    for t, tmpl in itertools.product(TOPICS, HARD_REAL):
        rows.append([tmpl.format(t=t), 0])

    # Inject a little label noise so the task isn't perfectly separable —
    # this makes accuracy realistic and the calibration curve meaningful.
    n_flip = int(len(rows) * noise_rate)
    for idx in rng.sample(range(len(rows)), n_flip):
        rows[idx][1] = 1 - rows[idx][1]

    df = pd.DataFrame(rows, columns=["text", "label"])
    return df.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)


def main():
    df = load_real()
    source = "real Kaggle dataset"
    if df is None:
        df = make_synthetic()
        source = "synthetic demo dataset (with borderline cases + label noise)"

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(config.DATASET_CSV, index=False)
    n_fake = int((df.label == 1).sum())
    n_real = int((df.label == 0).sum())
    print(f"Wrote {len(df)} rows to {config.DATASET_CSV}  ({source})")
    print(f"  fake={n_fake}  real={n_real}")


if __name__ == "__main__":
    main()
