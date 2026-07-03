"""
Fine-tune DistilBERT for fake-news classification (optional transformer backend).

A compact, dependency-light training loop (torch + transformers only — no
accelerate/Trainer) so it stays readable and CPU-runnable for a student project.
It fine-tunes `distilbert-base-uncased` on data/news.csv and saves the result to
models/distilbert/, which classifier.py's TransformerBackend loads.

Label convention: 1 = fake, 0 = real (matches the sklearn model).

Usage:
    python src/train_transformer.py            # defaults (2 epochs)
    python src/train_transformer.py --epochs 3 --batch-size 16

Note: the first run downloads the base model (~250 MB) from the Hugging Face Hub,
so it needs internet once. On the tiny synthetic dataset this is a demonstration
of the pipeline; use the real Kaggle data for meaningful transformer accuracy.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    if not os.path.exists(config.DATASET_CSV):
        raise SystemExit(f"No dataset at {config.DATASET_CSV}. Run: python data/make_dataset.py")

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    torch.manual_seed(config.RANDOM_STATE)

    df = pd.read_csv(config.DATASET_CSV).dropna(subset=["text", "label"])
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   base model: {config.TRANSFORMER_BASE}")

    tok = AutoTokenizer.from_pretrained(config.TRANSFORMER_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TRANSFORMER_BASE, num_labels=2
    ).to(device)

    def encode(texts):
        return tok(texts, truncation=True, padding="max_length",
                   max_length=args.max_length, return_tensors="pt")

    enc_tr, enc_te = encode(X_tr), encode(X_te)
    ds_tr = TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], torch.tensor(y_tr))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for input_ids, attn, labels in dl_tr:
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            opt.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            total += out.loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  avg loss={total / len(dl_tr):.4f}")

    # Evaluate.
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=enc_te["input_ids"].to(device),
                       attention_mask=enc_te["attention_mask"].to(device)).logits
        proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    print("\n=== Held-out test set (DistilBERT) ===")
    print(f"accuracy={accuracy_score(y_te, pred):.3f}  "
          f"f1={f1_score(y_te, pred, zero_division=0):.3f}  "
          f"roc_auc={roc_auc_score(y_te, proba):.3f}")

    os.makedirs(config.TRANSFORMER_DIR, exist_ok=True)
    model.save_pretrained(config.TRANSFORMER_DIR)
    tok.save_pretrained(config.TRANSFORMER_DIR)
    print(f"\nSaved DistilBERT -> {config.TRANSFORMER_DIR}")
    print("Use it with:  FAKE_NEWS_BACKEND=distilbert streamlit run app.py")


if __name__ == "__main__":
    main()
