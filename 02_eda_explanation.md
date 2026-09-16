# 02_eda.ipynb - Explanation

**Stage 2 of 4 - explore the data and produce the evidence the modelling choices rest on.**

| | |
|---|---|
| **Kaggle kernel** | [fraud-02-eda](https://www.kaggle.com/code/hiteshparakh00/fraud-02-eda) — ✅ COMPLETE |
| **Reads** | `output/data/train_eda.parquet`, `output/data/train_fe.parquet` (from notebook 01) |
| **Writes** | `output/plots/01`-`06`, three tables in `output/results/` |
| **Feeds** | Nothing - this is a leaf of the pipeline |
| **Size** | 22 cells - 13 code, 9 markdown |

---

## 1. What this notebook is for

This is where the data is looked at rather than modelled. Every claim the project later makes
about *why* it made a modelling choice traces back to a chart in here:

- The imbalance is severe → so accuracy is discarded as the headline metric, and the whole
  balanced-versus-imbalanced question becomes worth asking at all.
- Fraud looks different from legitimate spend on amount, time and category → so those three
  become the core engineered features.
- The engineered features do separate the classes → checked before any model sees them.

It is a **leaf**: notebooks 03 and 04 do not read anything it writes. It can be run at any
time after notebook 01, in parallel with 03, or skipped entirely without breaking the
modelling chain.

---

## 2. Loading

```python
train_raw = pd.read_parquet(upstream("data/train_eda.parquet"))
train_fe  = pd.read_parquet(upstream("data/train_fe.parquet"))
```

`upstream()` finds a file written by an earlier stage. Locally that is the shared `output/`
folder; on Kaggle, notebook 01's output is mounted somewhere under `/kaggle/input/`, so the
helper searches there instead:

```python
def upstream(rel):
    """Locate a file written by an earlier notebook (local run or Kaggle kernel input)."""
    local = OUT / rel
    if local.exists():
        return local
    if KAGGLE_IN.exists():
        for p in sorted(KAGGLE_IN.rglob(rel.split("/")[-1])):
            if p.as_posix().endswith("output/" + rel):
                return p
    raise FileNotFoundError("Run the earlier notebook first - missing: " + rel)
```

Two files are needed because the plots span both worlds: `train_eda.parquet` carries the raw
columns (`amt`, `category`, the timestamp), while `train_fe.parquet` carries the engineered
ones (`distance_km`, `is_night`).

Every figure is saved through one helper, so all 17 plots in the project land in the same
place with the same settings:

```python
def save_fig(name):
    """Save the current matplotlib figure into output/plots/."""
    plt.savefig(OUT / "plots" / (name + ".png"), dpi=150, bbox_inches="tight")
```

---

## 3. Class balance - `01_fraud_distribution.png`, `02_class_imbalance.png`

A seaborn count plot, then a bar-and-pie pair with the counts written on the bars.

**1,289,169 legitimate against 7,506 fraud - a ratio of 171.8 : 1, or 0.5789% fraud.**

This single fact drives the rest of the project. A model that answers "not fraud" for every
transaction scores 99.42% accuracy on this training set and catches nothing at all, which is
why the evaluation notebook reports six metrics and treats accuracy as the least informative
of them.

The pie chart exists because the bar chart alone under-sells the problem: at this ratio the
fraud bar is barely a line above the axis, and the "0.58%" label on the pie is what makes the
scale land.

---

## 4. Transaction amount - `03_amount_by_class.png`

Three views of the same variable: a box plot by class, a density histogram capped at the 99th
percentile, and the numeric summary.

```python
q99 = train_raw["amt"].quantile(0.99)
sns.histplot(data=train_raw[train_raw["amt"] <= q99], x="amt", hue="is_fraud",
             bins=40, element="step", stat="density", common_norm=False, ...)
```

Two choices worth explaining:

- **The 99th-percentile cap.** Amounts reach $28,948.90. Plotting the full range squashes
  every ordinary transaction into the leftmost pixel. The cap is for readability only - the
  outliers are still in the data and still in the models.
- **`common_norm=False`.** Each class is normalised to its own density. With 172× more
  legitimate rows, a shared normalisation would make the fraud curve invisible.

| Class | Count | Mean | Median | Std | Max |
|---|---|---|---|---|---|
| Legitimate | 1,289,169 | $67.67 | $47.28 | $154.01 | $28,948.90 |
| **Fraud** | 7,506 | **$531.32** | **$396.51** | $390.56 | $1,376.04 |

Fraud averages **7.9× the legitimate amount**. This is the strongest single signal in the
dataset, and it shows up later as the top feature in the Random Forest (`amt` 0.2712 and
`log_amt` 0.2696, together 54% of total importance).

Saved to `output/results/eda_amount_by_class.csv`.

---

## 5. Time of day - `04_fraud_rate_by_hour.png`

Fraud rate as a percentage, for each of the 24 hours.

```python
tmp["hour"] = pd.to_datetime(tmp["trans_date_trans_time"]).dt.hour
hourly = tmp.groupby("hour")["is_fraud"].mean() * 100
```

The result is far from flat:

| Hours | Fraud rate |
|---|---|
| **22:00, 23:00** | **2.88%, 2.84%** - the two highest hours in the day |
| 00:00-03:00 | 1.42% - 1.53% |
| 04:00-21:00 | roughly 0.09% - 0.14% |

So fraud concentrates in a late-night window running from about 22:00 to 03:00, at up to
**30× the daytime rate**. This is what justifies including `hour` as a feature at all.

It is also the chart that exposes a limitation in the `is_night` flag, which covers hours 0-5:
it excludes the two peak hours and includes two baseline ones. See section 10 of
[`01_data_loading_explanation.md`](01_data_loading_explanation.md) for the detail.

Saved to `output/results/eda_fraud_rate_by_hour.csv`.

---

## 6. Merchant category - `05_fraud_rate_by_category.png`

A table and a horizontal bar chart of fraud rate per category, sorted.

| Category | Transactions | Frauds | Fraud rate |
|---|---|---|---|
| `shopping_net` | 97,543 | 1,713 | **1.76%** |
| `misc_net` | 63,287 | 915 | **1.45%** |
| `grocery_pos` | 123,638 | 1,743 | **1.41%** |
| `shopping_pos` | 116,672 | 843 | 0.72% |
| `gas_transport` | 131,659 | 618 | 0.47% |
| … | | | |
| `health_fitness` | 85,879 | 133 | 0.15% |

The spread is **11× from top to bottom** (1.76% down to 0.15%), which is what makes one-hot
encoding the category worth the 13 extra columns.

The ordering is also a sanity check on the data itself: the two `_net` (online) categories
sit at the top, which is exactly how card-not-present fraud behaves in reality. A simulated
dataset that got this backwards would be suspect.

Saved to `output/results/eda_fraud_rate_by_category.csv`.

---

## 7. Engineered features against the target - `06_engineered_features.png`

The last check before modelling: do the features built in stage 1 actually separate the
classes?

- **`distance_km` by class** - a box plot from `train_fe`, comparing customer-to-merchant
  distance for fraud and non-fraud.
- **Fraud rate by `is_night`** - the night flag against the target, with the percentage
  printed on each bar.

This is deliberately done *after* feature engineering rather than before. A feature that looks
sensible in the abstract can still turn out to carry no signal, and it is cheaper to find that
out from a chart here than from a confusing feature-importance plot in notebook 04.

---

## 8. What gets saved

**Figures** - `output/plots/`

| File | Content |
|---|---|
| `01_fraud_distribution.png` | Count plot of the target |
| `02_class_imbalance.png` | Bar + pie, with counts and percentages |
| `03_amount_by_class.png` | Amount box plot + capped density histogram |
| `04_fraud_rate_by_hour.png` | Fraud rate across 24 hours |
| `05_fraud_rate_by_category.png` | Fraud rate per merchant category |
| `06_engineered_features.png` | `distance_km` box plot + `is_night` bar chart |

**Tables** - `output/results/`

| File | Content |
|---|---|
| `eda_amount_by_class.csv` | Full `describe()` of amount, split by class |
| `eda_fraud_rate_by_category.csv` | Count, fraud count and rate per category |
| `eda_fraud_rate_by_hour.csv` | Fraud rate per hour of day |

---

## 9. How to run it

**On Kaggle**
1. Import `02_eda.ipynb`
2. *Add Input* → **Notebooks** → select your completed **01** kernel → Add
3. Save & Run All

Notebook 01 must have finished successfully first - Kaggle can only mount a kernel's output
once that kernel has committed.

**Locally**
Run notebook 01 first so `output/data/` exists, then run this one. It is light: it reads two
small parquet files, not the source CSVs.

Because 02 and 03 both depend only on 01, they can be run at the same time.
