"""
Fake News Detector for Students — Streamlit web app.

Three modes: analyze pasted text, fetch & analyze a URL, or batch-check many
snippets. Each analysis shows a credibility gauge, red flags, the words that drove
the verdict, source reputation, and a concise trustworthy summary (Claude when an
API key is set, otherwise an offline extractive summary).

Run:  streamlit run app.py
"""
import os
import io
import sys
import json

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config  # noqa: E402
from predict import analyze  # noqa: E402
from summarize import summarize  # noqa: E402
from url_fetch import fetch_article, FetchError  # noqa: E402
from ollama_check import check as ollama_check, is_available as ollama_available  # noqa: E402
from ocr import extract_text as ocr_extract, available_engine as ocr_engine, OCRError  # noqa: E402
from classifier import active_backend_name  # noqa: E402

st.set_page_config(page_title="Fake News Detector for Students", page_icon="📰", layout="wide")

SAMPLE_FAKE = (
    "SHOCKING!! You won't BELIEVE what they are HIDING about the new vaccine! "
    "Doctors are FURIOUS and the government does NOT want you to know this secret truth. "
    "Share this before it gets DELETED!!!"
)
SAMPLE_REAL = (
    "According to a report published by the health ministry on Tuesday, an independent "
    "panel reviewed six months of trial data before recommending approval of the vaccine. "
    "Officials said monitoring would continue after rollout, citing standard procedures."
)


@st.cache_data(show_spinner=False)
def load_metrics():
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as f:
            return json.load(f)
    return None


def gauge_color(score):
    if score >= 66:
        return "#1a9850"
    if score >= 40:
        return "#f4a600"
    return "#d73027"


def credibility_gauge(score):
    color = gauge_color(score)
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;border-radius:10px;padding:14px 18px;">
          <div style="font-size:0.85rem;color:#666;">CREDIBILITY SCORE</div>
          <div style="font-size:2.6rem;font-weight:700;color:{color};line-height:1.1;">
            {score:.0f}<span style="font-size:1rem;color:#999;"> / 100</span>
          </div>
          <div style="background:#eee;border-radius:6px;height:12px;margin-top:8px;">
            <div style="width:{score}%;background:{color};height:12px;border-radius:6px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_check(text):
    st.markdown("**AI fact-check** (local LLM via Ollama)")
    with st.spinner("Asking the local model..."):
        fc = ollama_check(text)
    if not fc["enabled"]:
        st.caption("🔌 " + fc["note"])
        return
    verdict = fc["verdict"] or "uncertain"
    conf = f" · confidence {fc['confidence']}%" if fc.get("confidence") is not None else ""
    tone = {"likely false": "error", "likely true": "success"}.get(verdict.lower(), "info")
    getattr(st, tone)(f"**{verdict.title()}**{conf} — {fc['reasoning']}")
    if fc["suspicious_claims"]:
        st.markdown("Claims to verify:")
        for c in fc["suspicious_claims"]:
            st.markdown(f"- 🔎 {c}")
    st.caption(f"Model: {fc['model']} (runs locally, free)")


def render_result(text, result, want_summary=True, want_aicheck=False):
    left, right = st.columns([1, 1.4])

    with left:
        if result["is_fake"]:
            st.error(f"### ⚠️ {result['verdict']}")
        else:
            st.success(f"### ✅ {result['verdict']}")
        credibility_gauge(result["credibility_score"])
        st.caption(
            f"Model P(fake) = {result['model_prob_fake']:.2f} · "
            f"rule penalty = {result['rule_penalty']}"
        )
        src = result["source"]
        if src["tier"] == "reputable":
            st.success(f"🔗 {src['note']}", icon="🔗")
        elif src["tier"] == "low":
            st.error(f"🔗 {src['note']}", icon="🔗")
        elif src["domain"]:
            st.info(f"🔗 {src['note']}", icon="🔗")

    with right:
        st.markdown("**Credibility signals**")
        if result["red_flags"]:
            for flag in result["red_flags"]:
                st.markdown(f"- 🚩 {flag}")
        else:
            st.markdown("- ✅ No obvious red flags detected.")

        with st.expander("Why? Words that influenced the verdict"):
            c1, c2 = st.columns(2)
            c1.markdown("**→ toward FAKE**")
            c1.write(", ".join(result["fake_indicators"]) or "—")
            c2.markdown("**→ toward REAL**")
            c2.write(", ".join(result["real_indicators"]) or "—")

    if want_aicheck:
        render_ai_check(text)

    if want_summary:
        st.markdown("**Concise summary**")
        with st.spinner("Summarizing..."):
            summary, source = summarize(text)
        st.write(summary)
        st.caption(f"Summary source: {source} · classifier: {result.get('backend', 'sklearn')}")
        return summary
    return None


def downloadable_report(text, result, summary):
    lines = [
        "FAKE NEWS DETECTOR — ANALYSIS REPORT",
        "=" * 40,
        f"Verdict: {result['verdict']}",
        f"Credibility score: {result['credibility_score']}/100",
        f"Model P(fake): {result['model_prob_fake']}",
        f"Source: {result['source']['note']}",
        "",
        "Red flags:",
    ]
    lines += [f"  - {f}" for f in result["red_flags"]] or ["  - none"]
    lines += ["", f"Fake-leaning words: {', '.join(result['fake_indicators']) or '—'}"]
    lines += [f"Real-leaning words: {', '.join(result['real_indicators']) or '—'}"]
    if summary:
        lines += ["", "Summary:", summary]
    lines += ["", "Analyzed text:", text]
    return "\n".join(lines)


# ---------------------------------------------------------------- sidebar
metrics = load_metrics()
with st.sidebar:
    st.header("📰 About")
    st.markdown(
        "A hybrid tool that helps students spot misinformation:\n"
        "- **Classifier** (ML) estimates fake vs real\n"
        "- **Rule signals** flag clickbait, ALL CAPS, missing sources\n"
        "- **Source reputation** weighs the domain (for URLs)\n"
        "- **OCR** reads screenshots / WhatsApp forwards\n"
        "- **AI fact-check** via a free local Ollama LLM\n"
        "- **Summary** — free local AI model, or Claude"
    )
    st.markdown("---")
    st.markdown(f"**Classifier:** {active_backend_name()}")
    st.caption("Switch with FAKE_NEWS_BACKEND=distilbert")

    key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if key_set:
        st.markdown("**Summaries:** ✅ Claude (paid)")
    elif config.SUMMARY_MODE != "extractive":
        st.markdown("**Summaries:** 🆓 local AI model (free)")
        st.caption("Runs on your machine — no key needed.")
    else:
        st.markdown("**Summaries:** extractive (free)")

    ollama_ok, _ = ollama_available()
    st.markdown(f"**AI fact-check (Ollama):** {'✅ ready' if ollama_ok else '❌ not running'}")
    if not ollama_ok:
        st.caption(f"Free & local. Install ollama.com, then `ollama pull {config.OLLAMA_MODEL}`.")
    run_aicheck = st.checkbox("Run AI fact-check", value=ollama_ok,
                              help="Uses a local Ollama LLM — free, no API key")

    if metrics:
        st.markdown("---")
        st.subheader("Model performance")
        c1, c2 = st.columns(2)
        c1.metric("CV accuracy", f"{metrics['cv_accuracy_mean']*100:.1f}%",
                  help=f"±{metrics['cv_accuracy_std']*100:.1f}% over 5 folds")
        c2.metric("ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
        c1.metric("Test F1", f"{metrics['test_f1_fake']:.3f}")
        c2.metric("Brier", f"{metrics['test_brier']:.3f}",
                  help="Lower = better-calibrated probabilities")
        st.caption(f"Trained on {metrics['n_train']} articles, tested on {metrics['n_test']}.")

# ---------------------------------------------------------------- main
st.title("📰 Fake News Detector for Students")
st.caption("Analyze an article's credibility and get a trustworthy summary — before misinformation spreads.")

if not os.path.exists(config.MODEL_PATH):
    st.error(
        "No trained model found. From the project folder run:\n\n"
        "```\npython data/make_dataset.py\npython src/train.py\n```"
    )
    st.stop()

tab_text, tab_url, tab_image, tab_batch = st.tabs(
    ["📝 Paste text", "🔗 From URL", "🖼️ Image / Screenshot", "📚 Batch"]
)

with tab_text:
    c1, c2 = st.columns(2)
    if c1.button("Load fake example", use_container_width=True):
        st.session_state["text_input"] = SAMPLE_FAKE
    if c2.button("Load real example", use_container_width=True):
        st.session_state["text_input"] = SAMPLE_REAL

    text = st.text_area("Paste the news article or social-media post:", height=200,
                        key="text_input", placeholder="Paste article text here...")
    if st.button("Analyze", type="primary", key="analyze_text"):
        if len(text.strip()) < 10:
            st.warning("Please paste a longer piece of text.")
        else:
            result = analyze(text)
            summary = render_result(text, result, want_aicheck=run_aicheck)
            st.download_button("⬇️ Download report",
                               downloadable_report(text, result, summary),
                               file_name="analysis_report.txt")
            st.info("A decision aid, not a final verdict — always cross-check trusted sources.", icon="ℹ️")

with tab_url:
    url = st.text_input("Article URL:", placeholder="https://...")
    if st.button("Fetch & analyze", type="primary", key="analyze_url"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Fetching article..."):
                    title, body = fetch_article(url)
                st.success(f"Fetched: **{title or url}** ({len(body.split())} words)")
                with st.expander("Extracted text"):
                    st.write(body[:2000] + ("..." if len(body) > 2000 else ""))
                full = f"{title}. {body}" if title else body
                result = analyze(full, url=url)
                summary = render_result(full, result, want_aicheck=run_aicheck)
                st.download_button("⬇️ Download report",
                                   downloadable_report(full, result, summary),
                                   file_name="analysis_report.txt")
            except FetchError as e:
                st.error(f"Couldn't fetch that URL: {e}")

with tab_image:
    st.caption("Upload a screenshot or image (e.g. a WhatsApp forward). "
               "The text is extracted with OCR, then analyzed.")
    engine = ocr_engine()
    if engine is None:
        st.warning(
            "No OCR engine installed. Run:  `brew install tesseract && pip install pytesseract`  "
            "(or `pip install easyocr`), then reload."
        )
    else:
        st.caption(f"OCR engine: {engine}")
        up = st.file_uploader("Image file", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is not None:
            st.image(up, caption="Uploaded image", width=380)
            if st.button("Extract text & analyze", type="primary", key="analyze_image"):
                try:
                    with st.spinner("Reading text from image..."):
                        text = ocr_extract(up.getvalue())
                    st.markdown("**Extracted text**")
                    st.info(text)
                    result = analyze(text)
                    summary = render_result(text, result, want_aicheck=run_aicheck)
                    st.download_button("⬇️ Download report",
                                       downloadable_report(text, result, summary),
                                       file_name="analysis_report.txt")
                except OCRError as e:
                    st.error(str(e))

with tab_batch:
    st.caption("Paste multiple snippets (one per line) to score them all at once.")
    blob = st.text_area("One article/headline per line:", height=180,
                        placeholder="Headline 1\nHeadline 2\n...", key="batch_input")
    if st.button("Analyze all", type="primary", key="analyze_batch"):
        lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
        if not lines:
            st.warning("Enter at least one line.")
        else:
            rows = []
            for ln in lines:
                r = analyze(ln)
                rows.append({
                    "text": ln[:80] + ("..." if len(ln) > 80 else ""),
                    "verdict": r["verdict"],
                    "credibility": r["credibility_score"],
                    "P(fake)": r["model_prob_fake"],
                    "red_flags": len(r["red_flags"]),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            csv = io.StringIO(); df.to_csv(csv, index=False)
            st.download_button("⬇️ Download CSV", csv.getvalue(), file_name="batch_results.csv")
