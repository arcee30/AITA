"""Train and use a simple Reddit AITA text classifier. See README.md."""

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
LABELS = {0: "Not the asshole (NTA / NAH)", 1: "Asshole (YTA / ESH)"}
VERDICT_MAP = {"NTA": 0, "NAH": 0, "YTA": 1, "ESH": 1}


def clean_text(text):
    """Apply identical, deterministic cleaning at training and prediction time."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove obvious verdict acronyms, which can reveal the answer in edits.
    text = re.sub(r"\b(?:aita|wibta|yta|nta|esh|nah)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combine_text(title, body):
    return clean_text(f"{title}\n{body}")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    missing = {"title", "text", "is_asshole"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    audit = {"input_rows": len(df)}
    # Zero-based source row index, excluding the CSV header.
    df["source_row"] = np.arange(len(df))
    target = pd.to_numeric(df["is_asshole"], errors="coerce")
    if not target.isin([0, 1]).all():
        raise ValueError("is_asshole must contain only 0 or 1, with no missing labels.")
    df["is_asshole"] = target.astype(int)
    if "verdict" in df:
        expected = df["verdict"].astype(str).str.upper().str.strip().map(VERDICT_MAP)
        if expected.isna().any() or not expected.eq(df["is_asshole"]).all():
            raise ValueError("verdict and is_asshole disagree with NTA/NAH=0, YTA/ESH=1.")
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    body = df["text"].map(clean_text)
    usable = ~body.isin(["", "[deleted]", "[removed]"])
    audit["empty_or_removed_bodies_dropped"] = int((~usable).sum())
    df = df.loc[usable].copy()
    # Deduplicate by cleaned BODY, even if the titles differ.
    df["body_hash"] = body.loc[usable].map(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    )
    conflicts = df.groupby("body_hash")["is_asshole"].transform("nunique") > 1
    audit["conflicting_duplicate_rows_dropped"] = int(conflicts.sum())
    df = df.loc[~conflicts].copy()
    before = len(df)
    df = df.drop_duplicates("body_hash").reset_index(drop=True)
    audit["duplicate_rows_dropped"] = before - len(df)
    df["model_text"] = [combine_text(t, b) for t, b in zip(df.title, df.text)]
    audit["usable_rows"] = len(df)
    audit["class_counts"] = {str(k): int(v) for k, v in df.is_asshole.value_counts().items()}
    return df, audit


def split_data(df, seed=42):
    """70% training, 15% validation, 15% test; preserve class proportions."""
    train, holdout = train_test_split(
        df, test_size=0.30, stratify=df.is_asshole, random_state=seed
    )
    validation, test = train_test_split(
        holdout, test_size=0.50, stratify=holdout.is_asshole, random_state=seed
    )
    return train, validation, test


def build_model(seed=42):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.98,
            max_features=80000, sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            C=2.0, class_weight="balanced", solver="liblinear",
            max_iter=1000, random_state=seed,
        )),
    ])


def evaluate(y_true, y_pred, scores):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "asshole_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=[LABELS[0], LABELS[1]],
            output_dict=True, zero_division=0,
        ),
    }


def train_model(csv_path, output_dir, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, audit = load_data(csv_path)
    train, validation, test = split_data(df, seed)
    print(f"Loaded {len(df):,} usable posts. Class counts: {audit['class_counts']}", flush=True)
    print(f"Train: {len(train):,} | Validation: {len(validation):,} | Test: {len(test):,}", flush=True)

    # The vocabulary and IDF weights are learned from the training partition ONLY.
    model = build_model(seed)
    print("Fitting TF-IDF + logistic regression...", flush=True)
    model.fit(train.model_text, train.is_asshole)

    # Choose the classification threshold using validation data only.
    validation_scores = model.predict_proba(validation.model_text)[:, 1]
    threshold_results = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.02), 2):
        score = f1_score(
            validation.is_asshole, validation_scores >= threshold,
            average="macro", zero_division=0,
        )
        threshold_results.append({"threshold": float(threshold), "validation_macro_f1": float(score)})
    best = max(threshold_results, key=lambda r: (r["validation_macro_f1"], -abs(r["threshold"] - 0.5)))
    threshold = best["threshold"]
    pd.DataFrame(threshold_results).to_csv(output_dir / "validation_thresholds.csv", index=False)

    # Evaluate once on the untouched test partition. Do not tune using these results.
    test_scores = model.predict_proba(test.model_text)[:, 1]
    test_predictions = (test_scores >= threshold).astype(int)
    test_metrics = evaluate(test.is_asshole, test_predictions, test_scores)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train), 1)), train.is_asshole)
    baseline_input = np.zeros((len(test), 1))
    baseline_metrics = evaluate(
        test.is_asshole, baseline.predict(baseline_input),
        baseline.predict_proba(baseline_input)[:, 1],
    )
    split_counts = {
        name: {"rows": len(part), "positive_rows": int(part.is_asshole.sum())}
        for name, part in [("train", train), ("validation", validation), ("test", test)]
    }
    versions = {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}
    metrics = {
        "seed": seed, "data_audit": audit, "splits": split_counts,
        "selected_threshold": threshold, "best_validation_macro_f1": best["validation_macro_f1"],
        "label_mapping": LABELS, "versions": versions,
        "test_model": test_metrics, "test_majority_baseline": baseline_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save the fitted pipeline and threshold together. No fit is needed for prediction.
    joblib.dump({"pipeline": model, "threshold": threshold, "labels": LABELS,
                 "versions": versions, "seed": seed}, output_dir / "aita_model.joblib", compress=3)
    manifest = pd.concat([
        part[["source_row", "body_hash", "is_asshole"]].assign(split=name)
        for name, part in [("train", train), ("validation", validation), ("test", test)]
    ]).sort_values("source_row")
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    predictions = test[["source_row", "title", "is_asshole"]].copy()
    predictions["predicted_label"] = test_predictions
    predictions["asshole_score"] = test_scores
    predictions["correct"] = predictions.is_asshole.eq(predictions.predicted_label)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    report = classification_report(
        test.is_asshole, test_predictions, labels=[0, 1],
        target_names=[LABELS[0], LABELS[1]], digits=3, zero_division=0,
    )
    summary = (
        f"AITA text classifier — held-out test results\n\n"
        f"Training / validation / test rows: {len(train)} / {len(validation)} / {len(test)}\n"
        f"Validation-selected threshold: {threshold:.2f}\n"
        f"Validation macro F1: {best['validation_macro_f1']:.3f}\n\n"
        f"{'Metric':<24} {'Model':>10} {'Majority baseline':>20}\n"
    )
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "asshole_f1", "roc_auc", "average_precision"]:
        summary += f"{key:<24} {test_metrics[key]:>10.3f} {baseline_metrics[key]:>20.3f}\n"
    summary += f"\n{report}\nConfusion matrix: rows=true, columns=predicted; order=[0,1]\n"
    summary += str(np.array(test_metrics["confusion_matrix"])) + "\n"
    (output_dir / "evaluation.txt").write_text(summary, encoding="utf-8")
    print(summary)
    save_confusion_matrix(test_metrics["confusion_matrix"], output_dir)
    print(f"Saved model and results to: {output_dir.resolve()}")
    return metrics


def save_confusion_matrix(matrix, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.array(matrix), display_labels=["Not asshole", "Asshole"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title("AITA classifier — held-out test set")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def predict_post(model_path, title, text):
    if not str(text).strip():
        raise ValueError("Provide a nonempty post body.")
    bundle = joblib.load(model_path)
    score = float(bundle["pipeline"].predict_proba([combine_text(title, text)])[0, 1])
    label = int(score >= bundle["threshold"])
    return {"prediction": bundle["labels"][label], "label": label,
            "asshole_score": round(score, 4), "threshold": bundle["threshold"],
            "note": "Model score is not a calibrated probability or an objective moral judgment."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train, validate, evaluate, and save a model")
    training.add_argument("--data", type=Path, default=ROOT / "data/reddit_aita_moral_judgments.csv")
    training.add_argument("--output", type=Path, default=ROOT / "outputs")
    training.add_argument("--seed", type=int, default=42)
    prediction = commands.add_parser("predict", help="Predict the label of a new post")
    prediction.add_argument("--model", type=Path, default=ROOT / "outputs/aita_model.joblib")
    prediction.add_argument("--title", default="")
    body = prediction.add_mutually_exclusive_group(required=True)
    body.add_argument("--text", help="Post body as a quoted string")
    body.add_argument("--text-file", type=Path, help="UTF-8 .txt file containing a post body")
    args = parser.parse_args()
    try:
        if args.command == "train":
            train_model(args.data, args.output, args.seed)
        else:
            text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
            print(json.dumps(predict_post(args.model, args.title, text), indent=2))
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
