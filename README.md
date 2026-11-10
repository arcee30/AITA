# AITA: a simple machine-learning pipeline

Predict the binary label attached to a Reddit AITA post using its title and body.
Everything runs locally on a CPU. No API key, GPU, or model downloads are needed.
You will need to download the CSV on your own (linked in references)

## Run in VS Code

1. Unzip this project and open the **aita_ml** folder in VS Code.
2. Open a terminal inside that folder. Use **Python 3.11 or 3.12**.
3. Create and activate a virtual environment:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of
`python` in the commands below; activation is optional.

4. Install dependencies and train:

```bash
python -m pip install -r requirements.txt
python pipeline.py train
```

Training writes results into `outputs/`. It replaces the included reference
results; use `--output my_run` to keep a separate run. In VS Code, choose
**Python: Select Interpreter** and select `.venv` if you also run code in the editor.

## Predict a new post

You can use the included trained model immediately after installing dependencies.
Edit `example_post.txt`, then run:

```bash
python pipeline.py predict --title "AITA for locking up my groceries?" --text-file example_post.txt
```

Or pass the body directly:

```bash
python pipeline.py predict --title "AITA for refusing to lend my car?" --text "My friend keeps returning my car with an empty gas tank. I refused to lend it again."
```

Output includes the predicted label, an `asshole_score`, and the decision threshold.
The score is **not a calibrated probability**; class weighting and threshold
selection affect its interpretation.

## Data and labels

The provided CSV has **38,900 rows**:

| Binary target | Original verdicts | Count |
|---|---|---:|
| 0: not the asshole | NTA + NAH | 33,440 |
| 1: asshole | YTA + ESH | 5,460 |

This preserves the dataset's existing `is_asshole` target. ESH means everyone
sucks here, so the original poster is included in the positive class. NAH means
no assholes here, so it belongs to class 0.

**Features:** `title` + `text` only. `verdict` would directly reveal the answer,
so it is only checked for label consistency and never used as an input feature.
The precomputed length columns are not used.

## What the pipeline does

1. Checks labels, fills missing titles, and removes empty or deleted bodies.
2. Lowercases text, removes URLs and verdict acronyms, and collapses whitespace.
   It keeps stopwords, including negation words such as "not".
3. Removes duplicate cleaned bodies before splitting, even if titles differ.
   All copies of a body with conflicting labels are removed. None of these
   cleaning rules learn parameters from the data.
4. Creates reproducible, stratified **70% train / 15% validation / 15% test**
   partitions using seed 42. Both classes keep approximately the same proportions.
5. Fits a TF-IDF vectorizer on **training text only**, using words and two-word
   phrases with at most 80,000 features.
6. Trains logistic regression with `class_weight="balanced"`, so the minority
   class has more influence on the training loss. Its regularization is fixed at C=2.
7. Selects a score threshold on the validation set to maximize **macro F1**.
   The test set is not used for this choice.
8. Evaluates once on the test set against an always-majority-class baseline.
   The saved model remains the model fitted on the training partition; validation
   and test rows are not added back into training.
9. Saves the fitted vectorizer and classifier together, plus the selected threshold.

Accuracy is a weak headline here: predicting "not the asshole" for every post
already gets about 86% accuracy. **Macro F1**, balanced accuracy, and class-1
precision/recall show whether the model identifies the minority class. Average
precision summarizes its positive-class ranking quality. These are a simple
baseline's results, not a guarantee of strong performance on new stories.

## Included run: actual held-out results

All 38,900 rows passed the cleaning and duplicate checks. The split contains
27,230 training, 5,835 validation, and 5,835 test posts. Validation selected a
threshold of 0.50; validation macro F1 was 0.643.

| Test metric | Logistic regression | Always predict class 0 |
|---|---:|---:|
| Accuracy | 0.801 | 0.860 |
| Balanced accuracy | 0.653 | 0.500 |
| Macro F1 | 0.633 | 0.462 |
| Class-1 F1 | 0.386 | 0.000 |
| ROC AUC | 0.744 | 0.500 |
| Average precision | 0.326 | 0.140 |

Class-1 precision was 34.0% and recall was 44.7%: useful as a first baseline,
but there are still many false positives and missed positive posts. The model
has lower raw accuracy than the majority baseline because it predicts both
classes. The included confusion matrix shows exactly where it makes mistakes.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Loading, cleaning, splitting, training, evaluation, and prediction |
| `requirements.txt` | Dependency versions used for the included trained model |
| `data/reddit_aita_moral_judgments.csv` | Your original dataset, unchanged |
| `example_post.txt` | An editable example for prediction |
| `outputs/aita_model.joblib` | Trained TF-IDF pipeline, labels, and threshold |
| `outputs/evaluation.txt` | Human-readable test metrics and classification report |
| `outputs/metrics.json` | Complete metrics, data audit, split counts, and versions |
| `outputs/confusion_matrix.png` | Test confusion matrix |
| `outputs/test_predictions.csv` | Test titles, true labels, predictions, and scores |
| `outputs/split_manifest.csv` | Exact source row IDs and their train/validation/test assignment |
| `outputs/validation_thresholds.csv` | Validation macro F1 for each candidate threshold |

`source_row` is the zero-based data-row index in the original CSV, excluding its
header. It lets you reconstruct a partition without duplicating the large text CSV:

```python
import pandas as pd
data = pd.read_csv("data/reddit_aita_moral_judgments.csv")
splits = pd.read_csv("outputs/split_manifest.csv")
test = data.iloc[splits.loc[splits["split"] == "test", "source_row"].to_numpy()]
```

## Limits and next steps

This predicts the dataset's Reddit judgments, not objective morality. It learns
word patterns and does not reason through a situation like a person.
Verdict acronyms are stripped, but some posts may contain edits or paraphrased
reactions that reveal the outcome. Near-duplicates are not detected. There are
no author IDs or timestamps, so this is a random post split, not an author-held-out
or future-post test. A stronger follow-up would audit those issues and compare
predefined alternatives using validation data or cross-validation, preserving a
fresh test set for the final evaluation.

## References

- [scikit-learn TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn logistic regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Data leakage and preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html)

The package versions above are the tested environment; current online documentation
may describe a newer release.
