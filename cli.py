#!/usr/bin/env python3
"""
Command-line interface for the Fake News Detector.

Examples:
  python cli.py --text "SHOCKING!! You won't believe this secret!!!"
  python cli.py --url https://en.wikipedia.org/wiki/Misinformation
  python cli.py --image screenshot.png           # OCR the image, then analyze
  python cli.py --url https://example.com/article --summary
  python cli.py --text "..." --aicheck           # local LLM (Ollama) fact-check
  python cli.py --file articles.txt --json       # batch, machine-readable output
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from predict import analyze  # noqa: E402


def _render(result, summary=None, aicheck=None):
    v = result["verdict"]
    icon = "⚠️ " if result["is_fake"] else "✅ "
    print(f"\n{icon}{v}   (credibility {result['credibility_score']}/100)")
    print(f"   backend: {result.get('backend', 'sklearn')}")
    print(f"   model P(fake)={result['model_prob_fake']:.2f}  rule-penalty={result['rule_penalty']}")
    src = result["source"]
    if src["tier"] != "unknown":
        print(f"   source: {src['note']}")
    if result["red_flags"]:
        print("   red flags:")
        for f in result["red_flags"]:
            print(f"     - {f}")
    if result["fake_indicators"] or result["real_indicators"]:
        print(f"   fake words: {', '.join(result['fake_indicators']) or '—'}")
        print(f"   real words: {', '.join(result['real_indicators']) or '—'}")
    if aicheck is not None:
        if not aicheck["enabled"]:
            print(f"\n   AI fact-check: {aicheck['note']}")
        else:
            conf = f" ({aicheck['confidence']}%)" if aicheck.get("confidence") is not None else ""
            print(f"\n   AI fact-check [{aicheck['model']}]: {aicheck['verdict']}{conf}")
            if aicheck["reasoning"]:
                print(f"     {aicheck['reasoning']}")
            for c in aicheck["suspicious_claims"]:
                print(f"     • verify: {c}")
    if summary is not None:
        print(f"\n   summary: {summary}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze news for credibility.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="article text to analyze")
    g.add_argument("--url", help="URL of an article to fetch and analyze")
    g.add_argument("--image", help="image/screenshot to OCR, then analyze")
    g.add_argument("--file", help="file with one article per line")
    p.add_argument("--summary", action="store_true", help="also produce a summary")
    p.add_argument("--aicheck", action="store_true",
                   help="also run a local-LLM (Ollama) AI fact-check — free, no key")
    p.add_argument("--json", action="store_true", help="output JSON")
    args = p.parse_args(argv)

    items = []  # list of (text, url)
    if args.text:
        items.append((args.text, None))
    elif args.url:
        from url_fetch import fetch_article, FetchError
        try:
            title, body = fetch_article(args.url)
            items.append((f"{title}. {body}" if title else body, args.url))
        except FetchError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif args.image:
        from ocr import extract_text, OCRError
        try:
            text = extract_text(args.image)
            print(f"[OCR extracted]\n{text}\n")
            items.append((text, None))
        except OCRError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif args.file:
        with open(args.file) as f:
            for line in f:
                if line.strip():
                    items.append((line.strip(), None))

    results = []
    for text, url in items:
        result = analyze(text, url=url)
        summary = None
        if args.summary:
            from summarize import summarize
            summary, _ = summarize(text)
        aicheck = None
        if args.aicheck:
            from ollama_check import check
            aicheck = check(text)
        results.append({"result": result, "summary": summary, "aicheck": aicheck})
        if not args.json:
            _render(result, summary, aicheck)

    if args.json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
