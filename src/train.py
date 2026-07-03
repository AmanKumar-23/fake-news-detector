"""
Train and rigorously evaluate the fake-news classifier.

Model: TF-IDF (word 1-2 grams + char 3-5 grams) -> Logistic Regression.
  - Word n-grams capture topic/phrasing; char n-grams catch obfuscation
    (sp4ced-out or mis-spelled sensational words).
  - Logistic Regression stays linear and interpretable, so predict.py can show
    *which words* drove each verdict.

Evaluation goes well beyond accuracy, because a credibility tool must not overstate
confidence:
  - Stratified 5-fold cross-validation (accuracy + F1, mean ± std)
  - Held-out ROC-AUC, precision/recall/F1
  - Brier score + log loss (probability quality)
  - Saved plots: confusion matrix, ROC curve, calibration curve

Usage:  python src/train.py
"""
import os
import sys
import json

import numpy as np
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve


def build_pipeline():
    word_vec = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=1, max_features=20000, sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1,
        max_features=20000, sublinear_tf=True,
    )
    return Pipeline([
        ("features", FeatureUnion([("word", word_vec), ("char", char_vec)])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])


def _save_plots(y_true, proba, pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "fake"]); ax.set_yticklabels(["real", "fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(config.REPORT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(config.REPORT_DIR, "roc_curve.png"), dpi=120)
    plt.close(fig)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(config.REPORT_DIR, "calibration_curve.png"), dpi=120)
    plt.close(fig)


def main():
    if not os.path.exists(config.DATASET_CSV):
        raise SystemExit(f"No dataset at {config.DATASET_CSV}. Run: python data/make_dataset.py")

    df = pd.read_csv(config.DATASET_CSV).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    X, y = df["text"].astype(str), df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline()

    # Cross-validation on the training split for a robust performance estimate.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1"])
    cv_acc = (cv_res["test_accuracy"].mean(), cv_res["test_accuracy"].std())
    cv_f1 = (cv_res["test_f1"].mean(), cv_res["test_f1"].std())

    # Fit on train, evaluate on the untouched held-out set.
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_te, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, pred, average="binary", pos_label=1, zero_division=0
    )
    metrics = {
        "n_total": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "cv_accuracy_mean": round(cv_acc[0], 4),
        "cv_accuracy_std": round(cv_acc[1], 4),
        "cv_f1_mean": round(cv_f1[0], 4),
        "cv_f1_std": round(cv_f1[1], 4),
        "test_accuracy": round(float(acc), 4),
        "test_precision_fake": round(float(prec), 4),
        "test_recall_fake": round(float(rec), 4),
        "test_f1_fake": round(float(f1), 4),
        "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "test_brier": round(float(brier_score_loss(y_te, proba)), 4),
        "test_log_loss": round(float(log_loss(y_te, proba, labels=[0, 1])), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        _save_plots(np.asarray(y_te), proba, pred)
        plots_note = f"Saved plots  -> {config.REPORT_DIR}/"
    except Exception as e:  # matplotlib missing shouldn't fail training
        plots_note = f"(plots skipped: {e})"

    print("=== 5-fold cross-validation (train split) ===")
    print(f"accuracy: {cv_acc[0]:.3f} ± {cv_acc[1]:.3f}    f1: {cv_f1[0]:.3f} ± {cv_f1[1]:.3f}")
    print("\n=== Held-out test set ===")
    print(classification_report(y_te, pred, target_names=["real", "fake"], zero_division=0))
    print(f"accuracy={acc:.3f}  ROC-AUC={metrics['test_roc_auc']:.3f}  "
          f"Brier={metrics['test_brier']:.3f}  log-loss={metrics['test_log_loss']:.3f}")
    print(f"\nSaved model  -> {config.MODEL_PATH}")
    print(f"Saved metrics-> {config.METRICS_PATH}")
    print(plots_note)


if __name__ == "__main__":
    main()
