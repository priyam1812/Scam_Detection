# Explanation - How the Pipeline Works

A stage-by-stage walk-through of the four notebooks: what each one does, why it does it, and
how the four connect into a single pipeline.

---

## 1. The complete workflow

The project answers three research questions about credit-card fraud detection, and all three
share one feature matrix and one test set. Rather than rebuilding that shared work three
times, the pipeline computes it once and flows it forward:

```
       Kaggle dataset: fraudTrain.csv + fraudTest.csv
                          │
                          ▼
      ┌───────────────────────────────────────┐
      │  01_data_loading.ipynb                │
      │  load → validate → feature engineering│
      └───────────────────────────────────────┘
                          │
              output/data/*.parquet
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
  ┌─────────────────────┐   ┌─────────────────────────┐
  │  02_eda.ipynb       │   │  03_model_training.ipynb│
  │  explore + plot     │   │  fit 7 models           │
  └─────────────────────┘   └─────────────────────────┘
             │                         │
     output/plots/01–06        output/preds/*.parquet
     output/results/eda_*      output/models/*.joblib
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  04_evaluation.ipynb    │
                          │  metrics + graphs +     │
                          │  conclusions            │
                          └─────────────────────────┘
                                       │
                          output/plots/08–17
                          output/results/*.csv
```

Two design rules run through every stage:

**The test set is never touched.** Balancing, sub-sampling and label-free fitting are all
*training-side* choices. Every model - supervised or unsupervised, balanced or imbalanced - is
scored on the same untouched `fraudTest.csv` at its real 0.386% fraud rate. If the test set
were balanced too, the numbers would look far better and mean nothing.

**Accuracy is never the headline.** With 99.6% non-fraud, a model that always answers "not
fraud" scores 99.6% accuracy and catches zero fraud. The conclusions rest on Precision,
Recall, F1 and PR-AUC instead.

---

## 2. Stage 1 - Data Loading (`01_data_loading.ipynb`)

### What it does

**Loads the official split.** `fraudTrain.csv` (1,296,675 rows) and `fraudTest.csv`
(555,719 rows) are read separately rather than concatenated and re-split. Kaggle already
split these by time, and a time-based split is the honest one for fraud: you train on the
past and predict the future. Re-splitting randomly would leak future information backwards.

**Inspects the raw data.** `head()`, `columns`, `info()`, `isnull().sum()`,
`duplicated().sum()`, `value_counts()` on the target, and `describe()`. The findings: 23
columns, no missing values, no duplicate rows, and 7,506 fraud rows out of 1,296,675 -
**0.5789%**, an imbalance of 171.8 : 1.

**Validates the split.** The set intersection of `trans_num` between train and test is
computed and printed. It comes out empty, which rules out the most basic form of leakage - the
same transaction appearing on both sides.

**Engineers features.** Two functions do the work:

- `haversine_km(lat1, lon1, lat2, lon2)` - great-circle distance in kilometres between two
  lat/long points, used to measure how far a transaction happened from the cardholder's
  registered address.
- `engineer_features(df)` - turns the 23 raw columns into a modelling matrix.

What it builds and why:

| Feature | Derivation | Reasoning |
|---|---|---|
| `age` | `(trans_date_trans_time - dob).days / 365.25` | Fraud patterns differ by age group |
| `hour` | Hour of the transaction timestamp | Fraud clusters at particular times |
| `dayofweek` | Weekday of the timestamp | Weekday/weekend spending differs |
| `is_night` | `1` when `hour` is 0–5 | Night-time fraud flag |
| `distance_km` | Haversine of customer vs merchant coords | Card-present fraud happens far from home |
| `log_amt` | `log1p(amt)` | `amt` is heavily right-skewed; the log helps linear models |
| `gender` | `1` when `"M"` | Binary encoding |
| `city_pop` | Kept as-is | City-size context |
| `category_*` | `get_dummies(..., drop_first=True)` | 14 merchant categories → 13 indicator columns |

Dropped: `first`, `last`, `street`, `city`, `state`, `zip`, `job`, `cc_num`, `trans_num`,
`merchant`, `unix_time`, and the raw lat/long columns. Names and addresses are identifiers,
not signals; `merchant` has such high cardinality that one-hot encoding it would produce
hundreds of near-empty columns.

**Aligns train and test.** `train_fe.align(test_fe, join="outer", axis=1, fill_value=0)`
guarantees both matrices carry identical columns in identical order, so a model fitted on one
can predict on the other. The result is 22 features plus the target.

### What it writes

| File | Why |
|---|---|
| `output/data/train_fe.parquet` | Engineered training matrix - used by stages 2 and 3 |
| `output/data/test_fe.parquet` | Engineered test matrix - used by stage 3 |
| `output/data/train_eda.parquet` | The four raw columns (`is_fraud`, `amt`, `category`, `trans_date_trans_time`) the EDA plots need, so stage 2 never has to re-read 350 MB of CSV |
| `output/results/data_validation.csv` | Row counts, fraud counts, fraud rates, leakage check |

Parquet rather than CSV because it preserves dtypes exactly - in particular the boolean
one-hot columns and the datetime column survive the round trip, so stage 3 gets byte-identical
inputs to what stage 1 produced.

---

## 3. Stage 2 - EDA (`02_eda.ipynb`)

### What it does

Stage 2 is where the data is looked at rather than modelled. It reads the two parquet files
from stage 1 and produces the evidence that the modelling choices later rest on.

**Class balance.** A count plot and a bar-plus-pie pair make the imbalance visible: 1,289,169
non-fraud against 7,506 fraud, a 171.8 : 1 ratio. This is the single most important fact in
the project - it is why accuracy is unusable and why the balancing question is worth asking.

**Transaction amount.** A box plot by class, a density histogram capped at the 99th percentile
(so outliers do not flatten the plot), and `groupby("is_fraud")["amt"].describe()`. Fraudulent
transactions average **$531** against **$68** for legitimate ones, with medians of $397 and
$47. Amount is clearly a strong signal.

**Time of day.** Fraud rate per hour, plotted across all 24 hours. The rate is far from flat,
which is what motivates the `hour` and `is_night` features.

**Merchant category.** Fraud rate per category as a table and a horizontal bar chart.
`shopping_net` (1.76%), `misc_net` (1.45%) and `grocery_pos` (1.41%) lead; `health_fitness`
(0.15%) trails. Online categories being at the top matches how card-not-present fraud works.

**Engineered features against the target.** A box plot of `distance_km` by class and a bar
chart of fraud rate by the `is_night` flag - a sanity check that the features built in stage 1
actually separate the classes before any model sees them.

### What it writes

`output/plots/01`-`06` (six figures) and three tables under `output/results/`:
`eda_amount_by_class.csv`, `eda_fraud_rate_by_category.csv`, `eda_fraud_rate_by_hour.csv`.

Stage 2 is a **leaf** of the pipeline: stages 3 and 4 do not depend on it. It can be run at
any time after stage 1, or skipped entirely without breaking the modelling chain.

---

## 4. Stage 3 - Model Training (`03_model_training.ipynb`)

### What it does

**Builds X and y.** `train_fe` and `test_fe` are split into features and target. A
`StandardScaler` is then fitted **on the training features only** and applied to both sets.
Fitting the scaler on train alone is what keeps test statistics out of the training pipeline.

Scaled and unscaled matrices are both kept: Logistic Regression needs scaled inputs to
converge sensibly, Random Forest is scale-invariant and uses the raw matrix.

**Builds the balanced training set.** Random Under-Sampling: keep all 7,506 fraud rows, draw
7,506 non-fraud rows without replacement, shuffle. Result: 15,012 rows at exactly 1:1.

Under-sampling was chosen over SMOTE for four reasons stated in the original project: there
are already 7,506 *real* fraud cases to learn from; under-sampling keeps only real
transactions; the majority class is large enough (1.29M) to drop from safely; and SMOTE's
synthetic minority samples can invent fraud patterns that do not exist.

The balanced set is scaled with the **same scaler already fitted on the full training data**,
not a new one. A fresh scaler fitted on the balanced subset would give the two experiments
different feature scales and make them incomparable.

**Trains seven models.**

*Group A - imbalanced training (full 1,296,675 rows):*
- `LogisticRegression(max_iter=1000)` on the scaled matrix
- `RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=2)` on the raw matrix

*Group B - balanced training (15,012 rows):* the same two model configurations, built by the
same `make_models()` factory so that nothing but the training data differs between A and B.

*Group C - unsupervised (fitted on non-fraud rows only):*
- `IsolationForest(n_estimators=100, contamination=0.0058)` on all 1,289,169 normal rows
- `OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)` on a 40,000-row subsample
- `LocalOutlierFactor(n_neighbors=20, novelty=True)` on the same subsample

OCSVM and LOF get a subsample because both scale roughly quadratically with the number of
training points; 1.29M rows is computationally out of reach for them.

These three learn what "normal" looks like and flag deviations. Their `predict()` returns
`-1` for anomaly and `+1` for normal, so `anomaly_pred_score()` converts that to the same
0/1 convention as the classifiers and negates `decision_function()` so that **higher always
means more suspicious** - which is what the ROC and PR curves in stage 4 need.

**Reuses group A as the supervised arm.** The supervised-vs-unsupervised study uses Logistic
Regression and Random Forest trained on the full imbalanced data — the exact same
configuration and the exact same data as group A. So they are fitted once and appear under
both names in stage 4, which cuts the most expensive training step in the project from four
runs to two without changing a single number.

**Scores the training data too.** For the overfitting analysis, each supervised model also
predicts on the data it was fitted on. The imbalanced models use a 100,000-row sample that
keeps every fraud row, because predicting on all 1.29M rows is unnecessary work; the balanced
models are scored on their full 15,012-row training set.

### What it writes

| File | Contents |
|---|---|
| `output/preds/test_preds.parquet` | `y_true` plus `::pred` and `::score` for all 7 models, 555,719 rows |
| `output/preds/train_sample_preds.parquet` | Imbalanced LR/RF on their 100k training sample |
| `output/preds/train_bal_preds.parquet` | Balanced LR/RF on their 15,012-row training set |
| `output/models/*.joblib` | All 7 fitted estimators plus the scaler, joblib-compressed |
| `output/models/rf_feature_importance.csv` | `feature_importances_` from the imbalanced Random Forest |
| `output/plots/07_before_after_balancing.png` | Class counts before and after under-sampling |

**Why store predictions rather than just models.** Stage 4 needs `y_pred` and `y_score` for
seven models on 555,719 rows. If it loaded the models instead, it would also have to rebuild
the entire feature matrix and re-run the scaler — duplicating stage 1 and half of stage 3 —
and the slowest predictions in the project (One-Class SVM across half a million rows) would
run a second time. Storing the arrays makes stage 4 pure evaluation: it does arithmetic on
numbers, and nothing else. The models are still saved, for inspection and reuse.

---

## 5. Stage 4 - Evaluation (`04_evaluation.ipynb`)

Stage 4 reads the prediction arrays and produces all three analyses. It fits nothing.

Two scoring helpers do the work. `evaluate_model()` returns Accuracy, Precision, Recall, F1,
ROC-AUC and PR-AUC while printing a full `classification_report` and confusion matrix;
`evaluate()` does the same and additionally tags each row Supervised or Unsupervised for the
third analysis.

### Part 1 - Imbalanced vs balanced training

All four models from groups A and B, scored on the same imbalanced test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Imbalanced \| Logistic Regression | 0.9967 | 0.7778 | 0.2186 | 0.3413 | 0.8619 | 0.3463 |
| **Imbalanced \| Random Forest** | 0.9989 | **0.9426** | 0.7506 | **0.8357** | 0.9961 | **0.8819** |
| Balanced \| Logistic Regression | 0.8977 | 0.0303 | 0.8233 | 0.0585 | 0.9514 | 0.1633 |
| Balanced \| Random Forest | 0.9787 | 0.1506 | **0.9716** | 0.2608 | 0.9968 | 0.8014 |

Outputs: the metric table, a grouped bar chart, overlaid ROC and Precision–Recall curves for
all four models, confusion matrices for the two Random Forests side by side, and the top-12
Random Forest feature importances.

**Reading it.** Balancing does exactly what the theory predicts. Balanced Random Forest
catches **97.2%** of fraud against 75.1% for the imbalanced version — but its Precision
collapses from 0.94 to 0.15, meaning roughly six out of every seven fraud alerts it raises are
false alarms. Its confusion matrix shows 11,754 false positives against 98. On the combined
measures the imbalanced model wins clearly: F1 0.84 vs 0.26, PR-AUC 0.88 vs 0.80.

Note how little ROC-AUC separates the two (0.9961 vs 0.9968). ROC-AUC is dominated by the huge
non-fraud class and stays flattering even when precision has collapsed — which is exactly why
PR-AUC is the metric of record for rare-event problems.

### Part 2 — Overfitting analysis

Each supervised model is scored on the data it was fitted on and on the test set, and the
train-minus-test gap on F1 and PR-AUC is printed explicitly. Two figures accompany the table:
a per-model train-vs-test panel and a test-only comparison.

The conclusion distinguishes two different failure modes that look similar in a metrics table:

- **Overfitting** - the model memorised training rows, so train scores are near-perfect and
  test scores are much lower. The imbalanced Random Forest does not show this pattern.
- **Class-prior mismatch** - the balanced models score well on 1:1 training data and badly on
  0.4%-fraud test data. Nothing was memorised; the model simply learned that fraud is common,
  so its default 0.5 decision threshold sits in the wrong place for the real world.

The balanced Random Forest's Precision drop is prior mismatch, not memorisation.

### Part 3 - Supervised vs unsupervised

| Model | Type | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Supervised \| Logistic Regression | Supervised | 0.7778 | 0.2186 | 0.3413 | 0.3463 |
| **Supervised \| Random Forest** | Supervised | **0.9426** | **0.7506** | **0.8357** | **0.8819** |
| Unsupervised \| Isolation Forest | Unsupervised | 0.1890 | 0.3445 | 0.2441 | 0.1052 |
| Unsupervised \| One-Class SVM | Unsupervised | 0.0923 | 0.3193 | 0.1432 | 0.0567 |
| Unsupervised \| Local Outlier Factor | Unsupervised | 0.1606 | 0.2531 | 0.1965 | 0.1159 |

Averaged by approach: supervised PR-AUC **0.614** against unsupervised **0.093**.

Outputs: the metric table, a bar chart, ROC and PR curves for all five, confusion matrices for
the best supervised against the best unsupervised model, and the grouped means.

**Reading it.** The gap is not close. Anomaly detectors flag what is *unusual*, and unusual is
not the same as fraudulent - a rare but legitimate large purchase looks exactly like an
anomaly, which is why precision stays under 0.19 for all three. Their ROC-AUC values (0.83–0.89)
look respectable and are misleading for the same reason as in Part 1. Unsupervised methods
earn their place when labels are missing or arrive too late to train on; once labels exist,
they do not compete.

---

## 6. Outputs and results

Everything generated lands under `output/`, in five folders by kind: `data/` (engineered
matrices), `plots/` (17 PNG figures), `models/` (fitted estimators and feature importance),
`preds/` (prediction arrays), and `results/` (eight metric tables as CSV). The full file
listing is in [`README.md`](README.md).

### What the results say

**Random Forest on the original imbalanced data is the best model in the project.** Precision
0.94, Recall 0.75, F1 0.84, PR-AUC 0.88. It catches three-quarters of fraud while keeping
false alarms low enough to be operationally usable - 98 false positives across 555,719
transactions.

**Balancing is not automatically an improvement.** It reliably raises Recall and reliably
costs Precision. It is worth using when the business explicitly prefers catching fraud over
avoiding false alarms - and worth reporting as a sensitivity study in either case. It is not a
default step to apply without measuring.

**Labels are worth a great deal.** The best unsupervised PR-AUC (0.116) is roughly one eighth
of the best supervised one (0.882).

**Accuracy is the wrong metric here, and consistently so.** Every model in every table scores
above 0.89 accuracy, including ones that raise 56,454 false alarms.

---

## 7. How the four notebooks connect

Each stage writes its artifacts into `output/`, and the next stage reads them:

| Stage | Reads | Writes |
|---|---|---|
| **01 Data Loading** | `fraudTrain.csv`, `fraudTest.csv` | `output/data/*.parquet`, `output/results/data_validation.csv` |
| **02 EDA** | `output/data/train_eda.parquet`, `train_fe.parquet` | `output/plots/01-06`, `output/results/eda_*.csv` |
| **03 Model Training** | `output/data/train_fe.parquet`, `test_fe.parquet` | `output/preds/*.parquet`, `output/models/*`, `output/plots/07` |
| **04 Evaluation** | `output/preds/*.parquet`, `output/models/rf_feature_importance.csv` | `output/plots/08-17`, `output/results/*.csv` |

The dependency graph is `01 → {02, 03} → 04`, where only 03 feeds 04. Notebooks 02 and 03 can
run simultaneously; notebook 02 can be skipped without affecting the results.

### The path helper

Each notebook resolves its inputs the same way, which is what lets one file run unchanged both
locally and on Kaggle:

```python
def find_data_dir():
    """Folder holding fraudTrain.csv / fraudTest.csv (Kaggle input mount or local archive/)."""
    for c in [Path("/kaggle/input/fraud-detection"), Path("archive"), Path(".")]:
        if (c / "fraudTrain.csv").exists():
            return c
    for pattern in ["*/fraudTrain.csv", "*/*/fraudTrain.csv"]:
        for p in sorted(Path("/kaggle/input").glob(pattern)):
            return p.parent
    return Path("archive")
```

The search means the dataset is found regardless of what Kaggle names the mount folder, which
varies. Output goes to `/kaggle/working/output` on Kaggle and `output/` locally.

Reading an earlier stage's work uses `upstream()`:

```python
def upstream(rel):
    """Locate a file written by an earlier notebook (local run or Kaggle kernel input)."""
    local = OUT / rel
    if local.exists():
        return local
    for p in sorted(Path("/kaggle/input").glob("*/output/" + rel)):
        return p
    raise FileNotFoundError("Run the earlier notebook first - missing: " + rel)
```

Locally, all four notebooks share one `output/` folder, so the local copy is found first. On
Kaggle each notebook is a separate kernel, and an upstream kernel's output is mounted under
`/kaggle/input/<kernel-slug>/` - so the glob finds it there instead. The same line of code
works in both environments, and the error message names the missing file when a stage is run
out of order.

---

