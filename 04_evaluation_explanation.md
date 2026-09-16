# 04_evaluation.ipynb - Explanation

**Stage 4 of 4 - score every model, produce every chart, and answer the research questions.**

| | |
|---|---|
| **Kaggle kernel** | [fraud-04-evaluation](https://www.kaggle.com/code/hiteshparakh00/fraud-04-evaluation) - ✅ COMPLETE |
| **Reads** | `output/preds/` (3 files), `output/models/rf_feature_importance.csv` (from notebook 03) |
| **Writes** | `output/plots/08`-`17`, four tables in `output/results/` |
| **Feeds** | Nothing - this is the end of the pipeline |
| **Size** | 35 cells - 21 code, 14 markdown |

---

## 1. What this notebook is for

Three analyses, all scored on the same untouched imbalanced test set (555,719 rows, 0.386%
fraud):

1. **Imbalanced vs balanced training** - does balancing the training set help? (RQ1)
2. **Overfitting analysis** - are the models memorising, and why do balanced models behave
   differently?
3. **Supervised vs unsupervised** - do labelled classifiers beat anomaly detectors? (RQ2)

**This notebook fits nothing.** It reads the prediction arrays notebook 03 saved and does
arithmetic on them, so it re-runs in seconds and can be restyled freely without re-training.

### Which metrics matter, and why

| Metric | What it answers | Weight here |
|---|---|---|
| **Recall** | Of all real frauds, how many did we catch? | High - a missed fraud is a direct loss |
| **Precision** | Of all alerts raised, how many were real? | High - false alarms cost review time and annoy customers |
| **F1** | Balance of the two | High |
| **PR-AUC** | Threshold-free quality on the rare class | **Highest** - the standard for rare-event problems |
| **ROC-AUC** | Ranking quality, independent of prevalence | Medium - useful, but flattering under imbalance |
| **Accuracy** | Overall correctness | **Lowest** - see below |

A model that predicts "legitimate" for every one of the 555,719 test transactions scores
**99.61% accuracy** and catches zero fraud. Accuracy is reported for completeness and then set
aside.

---

## 2. Loading and the alias table

```python
test_pred = pd.read_parquet(upstream("preds/test_preds.parquet"))
train_sample_pred = pd.read_parquet(upstream("preds/train_sample_preds.parquet"))
train_bal_pred = pd.read_parquet(upstream("preds/train_bal_preds.parquet"))
rf_importance = pd.read_csv(upstream("models/rf_feature_importance.csv"), index_col=0)["importance"]

ALIAS = {
    "Supervised | Logistic Regression": "Imbalanced | Logistic Regression",
    "Supervised | Random Forest": "Imbalanced | Random Forest",
}

def get(name, kind):
    """Prediction ('pred') or score ('score') column for a model on the test set."""
    return test_pred[ALIAS.get(name, name) + "::" + kind].values
```

The supervised arm of analysis 3 is the same pair of fitted models as the imbalanced arm of
analysis 1 - same parameters, same training data. `ALIAS` lets both analyses refer to them
under their own naming without training anything twice.

Two scoring helpers do all the work: `evaluate_model()` returns the six metrics and prints a
full `classification_report` plus confusion matrix; `evaluate()` does the same and additionally
tags each row Supervised or Unsupervised for analysis 3.

---

## 3. Analysis 1 - imbalanced vs balanced training

Four models, one test set, only the training distribution differs.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Imbalanced · Logistic Regression | 0.9967 | 0.7778 | 0.2186 | 0.3413 | 0.8619 | 0.3463 |
| **Imbalanced · Random Forest** | 0.9989 | **0.9426** | 0.7506 | **0.8357** | 0.9961 | **0.8819** |
| Balanced · Logistic Regression | 0.8977 | 0.0303 | 0.8233 | 0.0585 | 0.9514 | 0.1633 |
| Balanced · Random Forest | 0.9787 | 0.1506 | **0.9716** | 0.2608 | 0.9968 | 0.8014 |

### What the confusion matrices say in operational terms

| | Imbalanced RF | Balanced RF |
|---|---|---|
| Frauds caught (TP) | 1,610 | 2,084 |
| Frauds missed (FN) | 535 | 61 |
| False alarms (FP) | **98** | **11,754** |
| Total alerts raised | 1,708 | 13,838 |
| **Reviews per fraud found** | **1.1** | **6.6** |

Balancing works exactly as the theory predicts. Balanced Random Forest catches **97.2%** of
fraud against 75.1% - 474 more frauds found. But it raises **12,130 more alerts** to do it,
which is **25.6 extra manual reviews for every extra fraud caught**. Precision collapses from
0.94 to 0.15: roughly six out of every seven of its alerts are false.

On the combined measures the imbalanced model wins clearly - F1 0.84 vs 0.26, PR-AUC 0.88 vs
0.80.

### The ROC-AUC trap

Note how little ROC-AUC separates the two Random Forests: **0.9961 vs 0.9968**. Read alone, it
would suggest the two models are equivalent, or even that the balanced one is slightly better.
ROC-AUC's false-positive rate is measured against the 553,574 legitimate rows, so 11,754 false
alarms barely move it. PR-AUC, which measures precision against the 2,145 frauds, drops from
0.88 to 0.80. This is the concrete demonstration of why PR-AUC is the metric of record for
rare events.

**Figures:** `08_balancing_key_metrics.png`, `09_balancing_roc_pr_curves.png`,
`10_balancing_confusion_matrices.png`, `11_rf_feature_importance.png`

### Feature importance

Top of the Random Forest's 22 features:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `amt` | 0.2712 |
| 2 | `log_amt` | 0.2696 |
| 3 | `hour` | 0.1154 |
| 4 | `category_grocery_pos` | 0.0831 |
| 5 | `age` | 0.0720 |

Amount in its two forms carries **54%** of total importance, matching the EDA finding that
fraud averages $531 against $68. Time is confirmed as the second real signal. Impurity-based
importance is a plausibility check, not a causal account - it tells you what the trees split
on, not what causes fraud.

---

## 4. Analysis 2 - overfitting

Each supervised model scored on the data it was fitted on and on the test set, with the gap
printed explicitly.

| Model | Split | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Imbalanced · LR | train | 0.9881 | 0.2217 | 0.3621 | 0.7052 |
| Imbalanced · LR | test | 0.7778 | 0.2186 | 0.3413 | 0.3463 |
| Imbalanced · RF | train | 1.0000 | 0.8280 | 0.9059 | 0.9980 |
| Imbalanced · RF | test | 0.9426 | 0.7506 | 0.8357 | 0.8819 |
| Balanced · LR | train | 0.8954 | 0.8511 | 0.8727 | 0.9507 |
| Balanced · LR | test | 0.0303 | 0.8233 | 0.0585 | 0.1633 |
| Balanced · RF | train | 0.9936 | 0.9923 | 0.9929 | 0.9998 |
| Balanced · RF | test | 0.1506 | 0.9716 | 0.2608 | 0.8014 |

### Two different failure modes that look alike in a table

**Overfitting** - the model memorised training rows. Train scores near-perfect, test scores far
below, *and recall falls too* because the memorised patterns do not generalise.

**Class-prior mismatch** - the model learned a correct decision rule under the wrong base rate.
Trained on 50% fraud, it carries a threshold calibrated for a world where fraud is common; the
test world has 0.386% fraud, so the same rule fires far too often.

The balanced Random Forest is the second, not the first:

- **Recall barely moves**: 0.9923 on train → **0.9716** on test. A model that had memorised
  its 15,012 training rows could not possibly generalise this well to 555,719 unseen ones.
- **Precision is what collapses**: 0.9936 → 0.1506. The false-positive *rate* is roughly
  unchanged; what changed is the denominator, from 7,506 legitimate training rows to 553,574
  legitimate test rows.
- **ROC-AUC is nearly identical** to the imbalanced forest (0.9968 vs 0.9961) - both rank
  transactions almost equally well.

The fix for prior mismatch is a threshold adjustment, not retraining.

### An honest caveat on this table

The imbalanced models' "train" row is a 100,000-row sample that keeps every fraud, so its fraud
rate is **7.51%** - about 19× the test set's 0.386%. Precision, F1 and PR-AUC all move with
prevalence, so part of the gap shown for the imbalanced models is that difference in
prevalence rather than memorisation. ROC-AUC is the prevalence-invariant metric and is the
safer basis for a strict generalisation claim. A prevalence-matched train sample would be the
cleaner design and is listed as future work.

**Figures:** `12_train_vs_test.png`, `13_test_only_comparison.png`

---

## 5. Analysis 3 - supervised vs unsupervised

| Model | Type | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Supervised · Logistic Regression | Supervised | 0.7778 | 0.2186 | 0.3413 | 0.8619 | 0.3463 |
| **Supervised · Random Forest** | Supervised | **0.9426** | **0.7506** | **0.8357** | 0.9961 | **0.8819** |
| Unsupervised · Isolation Forest | Unsupervised | 0.1890 | 0.3445 | 0.2441 | 0.8612 | 0.1052 |
| Unsupervised · One-Class SVM | Unsupervised | 0.0923 | 0.3193 | 0.1432 | 0.8270 | 0.0567 |
| Unsupervised · Local Outlier Factor | Unsupervised | 0.1606 | 0.2531 | 0.1965 | 0.8878 | 0.1159 |

Averaged by approach: **supervised PR-AUC 0.614 against unsupervised 0.093**.

The best supervised model has **7.6× the PR-AUC** of the best unsupervised one (0.8819 vs
0.1159 for LOF). In operational terms: Random Forest finds 1,610 frauds from 1,708 alerts;
LOF finds 543 from 3,381 - **about three times the fraud from half the alerts**.

**Why the gap is so wide.** Anomaly detectors flag what is *unusual*, and unusual is not the
same as fraudulent. A rare but perfectly legitimate large purchase looks exactly like an
anomaly, which is why precision stays under 0.19 for all three. Their ROC-AUC values look
respectable (0.83-0.89) for the same reason as in analysis 1 - the huge legitimate class
absorbs the false positives.

This does not make unsupervised methods useless. They need no labels, so they are the right
tool when fraud labels do not exist yet, arrive months late, or when a new fraud pattern has no
labelled examples at all. But once labels exist, they do not compete.

**Figures:** `14_supervised_vs_unsupervised_metrics.png`,
`15_supervised_vs_unsupervised_curves.png`, `16_supervised_vs_unsupervised_confusion.png`,
`17_supervised_vs_unsupervised_grouped.png`

---

## 6. Answers to the research questions

**RQ1 - Is balancing the training data necessary or beneficial?**

No, and it helps only under a narrow, quantifiable condition. The imbalanced Random Forest
wins on precision (0.9426), F1 (0.8357) and PR-AUC (0.8819) while raising the fewest alerts.
Balancing buys 474 extra frauds at a cost of 25.6 extra reviews each. It is the right choice
only when the business explicitly values catch-rate over review cost - and the balanced model's
weakness is prior mismatch, correctable by moving the decision threshold rather than by
retraining.

**RQ2 - Do supervised classifiers beat unsupervised anomaly detectors?**

Yes, decisively - 7.6× the PR-AUC of the best unsupervised detector. Labels are worth a great
deal here.

**A confound worth naming.** The balanced arm trains on 15,012 rows and the imbalanced arm on
1,296,675 - **86× more data**. Strictly, this experiment varies balancing *and* training-set
size together. A cleaner design would add two control arms: `class_weight="balanced"` on the
full data (balancing without data loss) and a random 15,012-row imbalanced subsample (data loss
without balancing). This is stated as a limitation, not corrected, because correcting it would
change the methodology the project set out to test.

---

## 7. What gets saved

**Figures** - `output/plots/08`-`17` (ten files, listed by analysis above)

**Tables** - `output/results/`

| File | Contents |
|---|---|
| `imbalanced_vs_balanced_results.csv` | RQ1 - 4 models × 6 metrics |
| `overfitting_train_vs_test.csv` | Analysis 2 - 4 models × train/test × 5 metrics |
| `supervised_vs_unsupervised_results.csv` | RQ2 - 5 models × 6 metrics |
| `supervised_vs_unsupervised_grouped.csv` | Mean metrics per approach |
