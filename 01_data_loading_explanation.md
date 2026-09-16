# 01_data_loading.ipynb - Explanation

**Stage 1 of 4 - load the raw data, validate it, and build the feature matrix.**

| | |
|---|---|
| **Kaggle kernel** | [fraud-01-data-loading](https://www.kaggle.com/code/hiteshparakh00/fraud-01-data-loading) - ✅ COMPLETE |
| **Reads** | `fraudTrain.csv`, `fraudTest.csv` (Kaggle dataset `kartik2112/fraud-detection`) |
| **Writes** | `output/data/` (3 parquet files), `output/results/data_validation.csv` |
| **Feeds** | Notebook 02 (EDA) and notebook 03 (model training) |
| **Size** | 22 cells - 15 code, 7 markdown |

---

## 1. What this notebook is for

Every later stage needs the same two things: a clean feature matrix for training and a
matching one for testing. Building those twice would risk them drifting apart, so this
notebook builds them once, saves them, and hands them on.

It also does the checks that have to happen *before* any model is trained. If the data has
leakage or missing values, no amount of good modelling later will fix it - and a reader
should be able to see that check was done, not just take it on trust.

---

## 2. Path setup

The notebook runs unchanged on Kaggle and on a local machine because it works out where it
is before touching any file:

```python
def find_data_dir():
    """Folder holding fraudTrain.csv / fraudTest.csv (Kaggle input mount or local archive/)."""
    for c in [KAGGLE_IN / "fraud-detection", Path("archive"), Path(".")]:
        if (c / "fraudTrain.csv").exists():
            return c
    for p in sorted(KAGGLE_IN.rglob("fraudTrain.csv")) if KAGGLE_IN.exists() else []:
        return p.parent
    return Path("archive")
```

The recursive search matters: Kaggle does not always mount a dataset at the folder name you
expect. On this project's kernel it appeared under `/kaggle/input/datasets/...`, not
`/kaggle/input/fraud-detection/`, so a hard-coded path fails. Searching for the file itself
works regardless of how Kaggle names the mount.

Output goes to `/kaggle/working/output` on Kaggle and `output/` locally, with five
sub-folders created up front: `data`, `plots`, `models`, `preds`, `results`.

---

## 3. Loading the data

```python
train_raw = pd.read_csv(DATA_DIR / "fraudTrain.csv")
test_raw = pd.read_csv(DATA_DIR / "fraudTest.csv")
```

The two files are read separately and never concatenated. Kaggle already split this dataset
by time, and for fraud a time-based split is the honest one - you train on the past and
predict the future. Shuffling the two files together and re-splitting at random would let
information from later transactions leak backwards into training, and the resulting scores
would be optimistic in a way that never survives deployment.

**What loads:**

| Split | Rows | Columns | Fraud rows | Fraud rate |
|---|---|---|---|---|
| `fraudTrain.csv` | 1,296,675 | 23 | 7,506 | 0.5789% |
| `fraudTest.csv` | 555,719 | 23 | 2,145 | 0.3860% |

Imbalance in training: **171.8 non-fraud for every 1 fraud**.

---

## 4. Dataset overview

Seven quick inspections, each answering one question a marker would ask:

| Cell | Question | Answer |
|---|---|---|
| `df.head()` | What does a row look like? | 23 columns: timestamp, card number, merchant, category, amount, customer details, coordinates, `is_fraud` |
| `df.columns` | What are the fields called? | Listed in full |
| `df.info()` | What types, how much memory? | 5 float, 6 int, 12 object; ~227 MB in memory |
| `df.isnull().sum()` | Any gaps? | **Zero** missing values in every column |
| `df.duplicated().sum()` | Any repeated rows? | **Zero** duplicates |
| `df["is_fraud"].value_counts()` | How many frauds? | 1,289,169 legitimate vs 7,506 fraud |
| `df.describe()` | What ranges? | Amounts run from $1.00 to $28,948.90 |

No missing values and no duplicates means no imputation and no de-duplication step is
needed - worth stating explicitly, because their absence is a finding, not an oversight.

---

## 5. Data validation

This is the section that rules out the most damaging silent error in a train/test project:

```python
overlap = set(train_raw["trans_num"]) & set(test_raw["trans_num"])
print("\nTrain/test overlapping trans_num:", len(overlap), "(expected: 0 - no row leakage)")
```

**Result: 0 overlapping transaction IDs.** No row appears on both sides, so no model can
score well simply by having memorised a test row during training.

The notebook also prints a few sample fraud and non-fraud rows side by side. Fraudulent
transactions are visibly larger and cluster in different categories - a first, informal sign
that the classes really are separable before any model is involved.

All of this is saved to `output/results/data_validation.csv`:

```
split,rows,fraud,fraud_rate_pct,trans_num_overlap
train,1296675,7506,0.5788651743883394,0
test,555719,2145,0.3859864427885316,0
```

---

## 6. Feature engineering

Two functions do the work.

### `haversine_km(lat1, lon1, lat2, lon2)`

Great-circle distance in kilometres between two points on the earth. The dataset gives the
cardholder's registered coordinates and the merchant's coordinates as four separate numbers,
which no model can use directly. The distance between them is the useful signal: a card
being used far from where its owner lives is a classic fraud pattern.

```python
def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two lat/long points in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))
```

The `6371` is the earth's mean radius in kilometres. The whole function is vectorised over
NumPy arrays, so it runs across 1.29 million rows in one call rather than row by row.

### `engineer_features(df)`

Turns 23 raw columns into 22 model-ready features plus the target.

| Feature | How it is derived | Why it should help |
|---|---|---|
| `amt` | Kept as-is | Fraud averages $531 against $68 for legitimate spend |
| `log_amt` | `np.log1p(amt)` | Amount is heavily right-skewed; the log makes it usable by a linear model |
| `age` | `(trans_date_trans_time - dob).dt.days / 365.25` | Fraud exposure differs by age group |
| `hour` | `.dt.hour` of the timestamp | Fraud is strongly concentrated at certain hours |
| `dayofweek` | `.dt.dayofweek` of the timestamp | Weekday and weekend spending differ |
| `is_night` | `1` when `hour` is between 0 and 5 | Night-time flag |
| `distance_km` | Haversine of customer vs merchant coordinates | Distance from home |
| `gender` | `1` when `"M"` | Binary encoding |
| `city_pop` | Kept as-is | City-size context |
| `category_*` | `pd.get_dummies(..., drop_first=True)` | 14 merchant categories become 13 indicator columns |

**What is dropped, and why.** `first`, `last`, `street`, `city`, `state`, `zip`, `job`,
`cc_num`, `trans_num`, `merchant`, `unix_time`, and the four raw coordinate columns. Names,
addresses and card numbers are identifiers - a flexible model would memorise specific people
rather than learn what fraud looks like. `merchant` has such high cardinality that one-hot
encoding it would add hundreds of near-empty columns. The raw coordinates are dropped because
`distance_km` already carries what they were needed for.

---

## 7. Aligning train and test

```python
train_fe, test_fe = train_fe.align(test_fe, join="outer", axis=1, fill_value=0)
feature_cols = [c for c in train_fe.columns if c != "is_fraud"]
train_fe = train_fe[feature_cols + ["is_fraud"]]
test_fe = test_fe[feature_cols + ["is_fraud"]]
```

A model fitted on one matrix can only predict on another if both have identical columns in
identical order. `align` guarantees that, and re-ordering with `feature_cols + ["is_fraud"]`
keeps the target last in both.

Final shapes: **train (1,296,675 × 23)**, **test (555,719 × 23)** - 22 features plus the
target in each.

---

## 8. What gets saved

| File | Contents | Read by |
|---|---|---|
| `output/data/train_fe.parquet` | Engineered training matrix | 02, 03 |
| `output/data/test_fe.parquet` | Engineered test matrix | 03 |
| `output/data/train_eda.parquet` | The four raw columns the EDA plots need (`is_fraud`, `amt`, `category`, `trans_date_trans_time`) | 02 |
| `output/results/data_validation.csv` | Row counts, fraud counts, rates, leakage check | reference |

**Why parquet and not CSV.** Parquet stores the dtype alongside the data. The 13 one-hot
columns are booleans and the timestamp is a datetime; through a CSV round-trip both would come
back as strings or objects and stage 3 would be training on subtly different inputs. Parquet
gives stage 3 exactly what stage 1 produced. It is also far smaller and faster to read than a
350 MB CSV.

**Why a separate `train_eda.parquet`.** Notebook 02 needs four raw columns that do not survive
feature engineering (`category` becomes dummies, the timestamp is consumed). Saving just those
four columns means the EDA notebook never has to re-read the 350 MB source CSV.

---

## 9. How to run it

**On Kaggle**
1. New Notebook → *File → Import Notebook* → upload `01_data_loading.ipynb`
2. *Add Input* → **Datasets** → search `kartik2112/fraud-detection` → Add
3. *Save Version → Save & Run All (Commit)*

Accelerator **None (CPU)**, Internet **off**. Nothing needs installing - every library is in
the Kaggle base image.

**Locally**
Put `fraudTrain.csv` and `fraudTest.csv` in `archive/`, then run the notebook top to bottom.
Reading and feature-engineering 1.85 million rows needs roughly 4 GB of free memory.

---

## 10. A limitation worth knowing

The `is_night` flag covers hours 0–5. Measured against the data (see
`output/results/eda_fraud_rate_by_hour.csv`), the highest-fraud hours are **22:00 (2.88%)** and
**23:00 (2.84%)**, which the flag excludes, while hours 4 and 5 sit at roughly the 0.10%
baseline. The flag therefore captures about 35% of all fraud; a 22:00-03:00 window would
capture about 85%.

This shows up in the results: `is_night` ranks 10th of 22 features in the Random Forest
importance (0.0159), while the raw `hour` feature ranks 3rd (0.1154) - the model recovers the
time signal from `hour` on its own rather than from the flag. The definition was kept as
originally written so that results stay comparable with the earlier version of this project;
it is a clear candidate for future work.
