# 03_model_training.ipynb - Explanation

**Stage 3 of 4 - build the training matrices and fit every model in the project.**

| | |
|---|---|
| **Kaggle kernel** | [fraud-03-model-training](https://www.kaggle.com/code/hiteshparakh00/fraud-03-model-training) - ✅ COMPLETE |
| **Reads** | `output/data/train_fe.parquet`, `output/data/test_fe.parquet` (from notebook 01) |
| **Writes** | `output/models/` (9 files), `output/preds/` (3 files), `output/plots/07` |
| **Feeds** | Notebook 04 (evaluation) |
| **Size** | 26 cells - 15 code, 11 markdown |

---

## 1. What this notebook is for

Seven models are fitted here and **nothing is scored**. Every metric, table and chart lives in
notebook 04. Keeping the split clean means the expensive work (fitting on 1.29 million rows,
predicting with a One-Class SVM across half a million) happens exactly once, and the
evaluation can be re-run and re-styled as often as needed without re-training anything.

The three experiment groups:

| Group | Training data | Models | Answers |
|---|---|---|---|
| **A. Imbalanced** | Full train, 1,296,675 rows, 0.58% fraud | Logistic Regression, Random Forest | RQ1 (baseline arm) |
| **B. Balanced** | Under-sampled to 1:1, 15,012 rows | Logistic Regression, Random Forest | RQ1 (treatment arm) |
| **C. Unsupervised** | Non-fraud rows only, no labels used | Isolation Forest, One-Class SVM, LOF | RQ2 |

**The test set is never touched by any of this.** Balancing, sub-sampling and label-free
fitting are all training-side choices. Every model is scored in notebook 04 on the same
untouched `fraudTest` at its real 0.386% fraud rate.

---

## 2. Preparing X and y

```python
X_train_full = train_fe.drop(columns=["is_fraud"])
y_train_full = train_fe["is_fraud"].astype(int)
X_test = test_fe.drop(columns=["is_fraud"])
y_test = test_fe["is_fraud"].astype(int)

scaler = StandardScaler()
X_train_full_scaled = pd.DataFrame(scaler.fit_transform(X_train_full), ...)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), ...)
```

**`fit_transform` on train, `transform` on test.** This is the single most important line in
the notebook for leakage. Fitting the scaler on the combined data - or on the test set - would
let the test distribution's mean and variance influence the training pipeline. Fitting on
train alone means the test set is genuinely unseen.

**Both scaled and unscaled matrices are kept.** Logistic Regression is a distance-based
optimiser and needs comparable feature ranges to converge sensibly; `amt` spans 1 to 28,948
while `is_night` is 0 or 1. Random Forest splits on thresholds one feature at a time and is
completely scale-invariant, so it uses the raw matrix. Feeding a tree scaled data would change
nothing except make the feature-importance output harder to read.

Resulting shapes: **X_train_full (1,296,675 × 22)** with 7,506 frauds, **X_test (555,719 × 22)**
with 2,145 frauds.

---

## 3. The balanced training set - `07_before_after_balancing.png`

```python
fraud = train_bal[train_bal["is_fraud"] == 1]
non_fraud = train_bal[train_bal["is_fraud"] == 0]
non_fraud_down = resample(non_fraud, replace=False, n_samples=len(fraud),
                          random_state=RANDOM_STATE)
train_balanced = pd.concat([fraud, non_fraud_down]).sample(frac=1, random_state=RANDOM_STATE)
```

Keep all 7,506 frauds, draw 7,506 legitimate rows without replacement, shuffle. Result:
**15,012 rows at exactly 1:1**. In doing so, **1,281,663 legitimate rows are discarded -
99.42% of the majority class.**

### Why Random Under-Sampling and not SMOTE

1. There are already 7,506 *real* fraud cases - enough to learn from without inventing more.
2. Under-sampling keeps only genuine transactions; nothing is synthesised.
3. The majority class is large enough (1.29M) to drop from heavily and still leave a clean set.
4. SMOTE interpolates between minority points to create new ones. For fraud, a synthetic point
   halfway between two real frauds may describe a pattern that no fraudster has ever used.

### One subtle but important detail

```python
# Scale using the SAME scaler fitted on full train (no data leakage from test)
X_train_bal_scaled = pd.DataFrame(scaler.transform(X_train_bal), ...)
```

The balanced set is scaled with the scaler **already fitted on the full training data**, not a
new one fitted on the 15,012 rows. If each experiment fitted its own scaler, groups A and B
would be working in different feature spaces and the comparison between them - which is the
entire point of RQ1 - would be confounded by the scaling as well as the balancing.

The figure shows class counts before and after, on the same axis pair.

---

## 4. The model factory

```python
def make_models():
    """Same two models for both experiments."""
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=2,
                                n_jobs=-1, random_state=RANDOM_STATE)
    return {"Logistic Regression": lr, "Random Forest": rf}
```

Called once for group A and once for group B. Because both groups are built by the same
function, **nothing differs between them except the training data** - same algorithms, same
hyperparameters, same seed. That is what makes RQ1 a controlled experiment rather than an
observation.

`max_depth=16` and `min_samples_leaf=2` cap the forest so it cannot grow a leaf per training
row; `n_estimators=100` is the usual default trade-off between variance and runtime.

---

## 5. Group A - imbalanced training

Logistic Regression on the scaled matrix, Random Forest on the raw one, both on all 1,296,675
rows. Predictions and probabilities on the test set are stored immediately:

```python
name = "Imbalanced | Random Forest"
test_pred[name + "::pred"]  = models_imb["Random Forest"].predict(X_test)
test_pred[name + "::score"] = models_imb["Random Forest"].predict_proba(X_test)[:, 1]
```

The `::pred` / `::score` naming keeps hard predictions and continuous scores side by side in
one table. Notebook 04 needs both: predictions for precision, recall, F1 and the confusion
matrix; scores for ROC-AUC, PR-AUC and the curves.

---

## 6. Group B - balanced training

The same two model configurations, fitted on the 15,012-row balanced set, predicting on the
same test set.

---

## 7. Group C - unsupervised detectors

```python
X_normal = X_train_full_scaled[y_train_full.values == 0]
sub_idx = rng.choice(len(X_normal), size=min(NORMAL_SUBSAMPLE, len(X_normal)), replace=False)
X_normal_sub = X_normal.iloc[sub_idx]
```

All three are fitted on **legitimate transactions only**. They never see a fraud label while
learning; the labels are used solely to score them afterwards. That is what makes the
comparison in RQ2 fair - these are genuinely label-free methods.

| Model | Configuration | Fitted on |
|---|---|---|
| Isolation Forest | `n_estimators=100`, `contamination=` train fraud rate | all 1,289,169 normals |
| One-Class SVM | RBF kernel, `gamma="scale"`, `nu=0.01` | 40,000-row subsample |
| Local Outlier Factor | `n_neighbors=20`, `novelty=True` | the same 40,000 rows |

```python
NORMAL_SUBSAMPLE = 40_000  # ponytail: OCSVM/LOF too slow on full 1.29M normals
```

One-Class SVM and LOF both scale roughly quadratically in the number of training points;
1.29 million rows is computationally out of reach for either. The subsample is documented as a
limitation rather than hidden - Isolation Forest, which does scale, uses all of them.

### Converting anomaly output to the project's convention

```python
def anomaly_pred_score(model, X):
    y_pred = (model.predict(X) == -1).astype(int)
    y_score = -model.decision_function(X)  # higher = more anomalous
    return y_pred, y_score
```

scikit-learn's outlier detectors return `-1` for anomaly and `+1` for normal, and a
`decision_function` where *lower* means more anomalous - the opposite of a classifier's
`predict_proba`. This helper flips both so that everything downstream follows one rule:
**1 means fraud, and a higher score means more suspicious**. Without the negation the ROC and
PR curves in notebook 04 would come out inverted.

---

## 8. Reusing group A as the supervised arm

The supervised half of the RQ2 comparison uses Logistic Regression and Random Forest trained
on the full imbalanced data - the identical configuration and identical data as group A. So
they are fitted **once** here and referenced under both names in notebook 04:

```python
ALIAS = {
    "Supervised | Logistic Regression": "Imbalanced | Logistic Regression",
    "Supervised | Random Forest": "Imbalanced | Random Forest",
}
```

This cuts the most expensive training step in the project from four runs to two and changes no
number anywhere - the two names refer to the same fitted object.

---

## 9. Training-set predictions for the overfitting check

Notebook 04 compares each supervised model on the data it was fitted on against the test set.
Those predictions are generated here:

```python
# ponytail: full 1.29M train predict can crash the kernel; 100k stratified sample is enough
n_sample = min(100_000, len(y_train_full))
fraud_idx = np.flatnonzero(y_train_full.values == 1)
nonfraud_idx = np.flatnonzero(y_train_full.values == 0)
sample_idx = np.concatenate([fraud_idx, rng.choice(nonfraud_idx, size=n_sample - len(fraud_idx),
                                                   replace=False)])
```

- **Imbalanced models** are scored on a 100,000-row sample that keeps every fraud row.
- **Balanced models** are scored on their full 15,012-row training set.

**Read the resulting gaps with care.** The 100k sample keeps all 7,506 frauds, so its fraud
rate is **7.51%** - about 19× the test set's 0.386%. Precision, F1 and PR-AUC all depend on
class prevalence, so part of any train-minus-test gap in notebook 04 is that prevalence
difference rather than memorisation. ROC-AUC is the prevalence-invariant metric of the six and
is the safer one to read a true generalisation gap from.

---

## 10. What gets saved

**Models** - `output/models/`

| File | Size |
|---|---|
| `Imbalanced__Random_Forest.joblib` | 5.09 MB |
| `Balanced__Random_Forest.joblib` | 1.44 MB |
| `Unsupervised__Local_Outlier_Factor.joblib` | 8.58 MB |
| `Unsupervised__Isolation_Forest.joblib` | 0.33 MB |
| `Unsupervised__OneClass_SVM.joblib` | 0.03 MB |
| `Imbalanced__Logistic_Regression.joblib`, `Balanced__Logistic_Regression.joblib` | ~1 KB each |
| `scaler.joblib` | ~1 KB |
| `rf_feature_importance.csv` | 22 features with their importance |

All are written with `joblib.dump(..., compress=3)`.

**Predictions** - `output/preds/`

| File | Rows | Fraud rate | Contents |
|---|---|---|---|
| `test_preds.parquet` | 555,719 | 0.386% | `y_true` + `pred`/`score` for all 7 models |
| `train_sample_preds.parquet` | 100,000 | 7.506% | Imbalanced LR/RF on their training sample |
| `train_bal_preds.parquet` | 15,012 | 50.000% | Balanced LR/RF on their training set |

### Why store predictions, not just models

Notebook 04 needs `y_pred` and `y_score` for seven models across 555,719 rows. If it loaded
the fitted models instead, it would have to rebuild the whole feature matrix and re-run the
scaler - duplicating notebook 01 and half of this one - and then re-run the slowest prediction
in the project, the One-Class SVM across half a million rows, a second time.

Storing the arrays makes notebook 04 pure evaluation: it does arithmetic on numbers and
nothing else, and re-runs in seconds. The models are still saved for inspection and reuse.

---

## 11. How to run it

**On Kaggle**
1. Import `03_model_training.ipynb`
2. *Add Input* → **Notebooks** → select your completed **01** kernel → Add
3. Save & Run All

Accelerator **None (CPU)** - none of these models use a GPU. This is the longest-running
notebook of the four; the Random Forest on 1.29 million rows and the One-Class SVM predicting
across the test set dominate the time.

**Locally**
Run notebook 01 first. Expect roughly 8 GB of free memory: the full matrix and its scaled copy
are each about 228 MB, and the Random Forest needs considerably more than that while fitting.

Notebooks 02 and 03 both depend only on 01, so they can be run at the same time.
