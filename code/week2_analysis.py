# =============================================================================
# WEEK 2: DATA COLLECTION, CLEANING, AND PREPROCESSING
# Project: Logistics Delivery Time Analysis and Prediction
# Target Variable: lead_time_days
# =============================================================================

# ---- 1. IMPORT REQUIRED LIBRARIES ----
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1: DATASET INSPECTION
# =============================================================================
print("="*70)
print("STEP 1: DATASET INSPECTION")
print("="*70)

# Workspace location (this script lives in E:\Yuva Intern alongside the raw dataset)
WORKSPACE_DIR = r"E:\Yuva Intern"

# Load the dataset from the E:\Yuva Intern workspace
import os
file_path = os.path.join(WORKSPACE_DIR, "dynamic_supply_chain_logistics_dataset.csv")
df = pd.read_csv(file_path)

# 1. Shape
print("\n[1.1] Dataset Shape")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 2. Column names
print("\n[1.2] Complete List of Column Names")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# 3. Data types
print("\n[1.3] Data Types of All Columns")
print(df.dtypes.to_string())

# 4. First 5 rows
print("\n[1.4] First 5 Rows")
print(df.head().to_string())

# 5. Statistical summary of numerical columns
print("\n[1.5] Statistical Summary of Numerical Columns")
print(df.describe().round(3).to_string())

# 6. Missing values
print("\n[1.6] Number of Missing Values in Every Column")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0].to_string() if (missing > 0).any()
      else "  No missing values found in any column.")

# 7. Duplicate rows
print("\n[1.7] Total Number of Duplicate Rows")
num_dupes = df.duplicated().sum()
print(f"  Duplicate rows found: {num_dupes}")

# 8. Unique values in important categorical columns (object dtype columns)
print("\n[1.8] Unique Values in Categorical Columns")
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if len(cat_cols) == 0:
    print("  No object-type categorical columns detected.")
else:
    for col in cat_cols:
        print(f"\n  --- {col} ---")
        print(f"    Unique count: {df[col].nunique()}")
        uniq = df[col].unique()
        if len(uniq) <= 30:
            for v in uniq:
                print(f"      - {v}  (n = {(df[col]==v).sum()})")
        else:
            print(f"    (Showing first 20 of {len(uniq)} unique values)")
            for v in uniq[:20]:
                print(f"      - {v}")


# =============================================================================
# STEP 2: DATASET UNDERSTANDING
# =============================================================================
print("\n" + "="*70)
print("STEP 2: DATASET UNDERSTANDING")
print("="*70)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
datetime_cols = []

for col in df.columns:
    # Only consider object-string columns as candidates for real datetime parsing
    # (numeric columns with "time" in the name are durations, not timestamps)
    if df[col].dtype == "object" and ("date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower()):
        try:
            sample_parsed = pd.to_datetime(df[col].head(5))
            # Ensure year range is sensible for a logistics dataset (not epoch 1970 from garbage)
            if sample_parsed.dt.year.min() >= 2000:
                datetime_cols.append(col)
        except Exception:
            pass

print("\n[2.1] Useful Variables for Logistics Analysis")
print("  All columns appear to be supply-chain/logistics related.")
print(f"  Primary target for analysis: lead_time_days")

print("\n[2.2] Numerical Variables:")
for c in numeric_cols:
    print(f"    - {c}")

print("\n[2.3] Categorical Variables (object dtype):")
for c in categorical_cols:
    print(f"    - {c}")

print("\n[2.4] Timestamp / Date Columns Detected:")
if datetime_cols:
    for c in datetime_cols:
        print(f"    - {c}")
else:
    print("    No pure timestamp/datetime columns detected in current dtypes.")
    print("    Checking columns for date-like semantic names...")
    date_semantic = [c for c in df.columns if any(k in c.lower() for k in ["date","time_stamp","timestamp","ship","order"])]
    if date_semantic:
        print("    Candidate date-like columns (by name):")
        for c in date_semantic:
            print(f"      - {c}")

print("\n[2.5] Variables Suitable for EDA (Week 3)")
print("  - lead_time_days                 (target distribution)")
print("  - shipping_costs                 (cost vs lead time)")
print("  - fuel_consumption_rate          (cost efficiency)")
print("  - transportation_mode / carrier  (performance by operator)")
print("  - route_id / origin / destination (geographic patterns)")
print("  - weather_condition_severity     (environmental impact)")
print("  - port_congestion_level          (bottlenecks)")
print("  - warehouse_inventory_level      (stock & delays)")
print("  - supplier_reliability_score     (supplier performance)")

print("\n[2.6] Variables Suitable as Features for Predictive Modeling")
print("  Candidate predictors for lead_time_days regression:")
for c in numeric_cols:
    if c != "lead_time_days":
        print(f"    - {c}")
if categorical_cols:
    for c in categorical_cols:
        print(f"    - {c}  (after encoding)")


# =============================================================================
# STEP 3: DATA CLEANING
# =============================================================================
print("\n" + "="*70)
print("STEP 3: DATA CLEANING")
print("="*70)

# --- A. Missing Values ---
print("\n[3.A] MISSING VALUES")
missing_counts = df.isnull().sum()
cols_with_missing = missing_counts[missing_counts > 0]
if len(cols_with_missing) == 0:
    print("  OK - No missing values in any column. No imputation required.")
else:
    print("  Columns with missing values:")
    for col, cnt in cols_with_missing.items():
        pct = cnt / len(df) * 100
        print(f"    - {col}: {cnt} values missing ({pct:.2f}%)")
        # Strategy: numeric = median; categorical = mode
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"      → Imputed with MEDIAN = {df[col].median():.3f}")
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"      → Imputed with MODE = {df[col].mode()[0]}")

# --- B. Duplicate Records ---
print("\n[3.B] DUPLICATE RECORDS")
n_dupes_before = df.duplicated().sum()
if n_dupes_before == 0:
    print("  OK - No duplicate rows found. No removal required.")
else:
    print(f"  Found {n_dupes_before} duplicate rows. Removing them...")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  OK - Removed {n_dupes_before} duplicates.")

# --- C. Data Types ---
print("\n[3.C] DATA TYPE CHECKS & CONVERSIONS")
conversions = []
for col in df.columns:
    if any(k in col.lower() for k in ["date","timestamp","time_stamp","ship_date","order_date"]):
        try:
            before = df[col].dtype
            df[col] = pd.to_datetime(df[col], errors="coerce")
            after = df[col].dtype
            if str(before) != str(after):
                conversions.append(f"  - {col}: {before}  to  datetime64")
        except Exception as e:
            print(f"  Could not convert {col}: {e}")

if conversions:
    for line in conversions:
        print(line)
else:
    print("  No conversions required. All columns have appropriate dtypes.")
    datetime_cols_real = []

# Confirm any real datetime columns
datetime_cols_real = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
if datetime_cols_real:
    print("  Actual datetime columns after conversion:")
    for c in datetime_cols_real:
        print(f"    - {c}")

# --- D. Invalid or Inconsistent Values ---
print("\n[3.D] INVALID / INCONSISTENT VALUES")
issues_found = False
for col in numeric_cols:
    # Check for negative values that shouldn't be negative
    if any(k in col.lower() for k in ["days","hours","cost","rate","level","time","score","demand","probability","deviation","quantity","weight","volume","distance","duration"]):
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"  WARN - {col}: {neg_count} negative values detected.")
            issues_found = True
    # Check probability columns are in [0,1]
    if "probability" in col.lower():
        outside = ((df[col] < 0) | (df[col] > 1)).sum()
        if outside > 0:
            print(f"  WARN - {col}: {outside} values outside [0,1] range.")
            issues_found = True

# Categorical consistency
for col in categorical_cols:
    if col in df.columns:
        null_like = df[col].astype(str).str.strip().isin(["", "nan", "None", "NaN"]).sum()
        if null_like > 0:
            print(f"  WARN - {col}: {null_like} blank/null-like string values.")
            issues_found = True

if not issues_found:
    print("  OK - No invalid or suspicious values detected in numeric or categorical columns.")


# =============================================================================
# STEP 4: OUTLIER DETECTION (IQR METHOD)
# =============================================================================
print("\n" + "="*70)
print("STEP 4: OUTLIER DETECTION USING IQR METHOD")
print("="*70)

candidate_numerics = [
    "fuel_consumption_rate", "eta_variation_hours", "warehouse_inventory_level",
    "loading_unloading_time", "weather_condition_severity", "port_congestion_level",
    "shipping_costs", "supplier_reliability_score", "lead_time_days",
    "historical_demand", "customs_clearance_time", "delay_probability",
    "delivery_time_deviation"
]
# Only keep columns that actually exist in the dataset
cols_to_check = [c for c in candidate_numerics if c in df.columns]

outlier_summary = []
kept_all = True

for col in cols_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    pct_out = n_out / len(df) * 100
    outlier_summary.append({
        "Column": col,
        "Q1": round(Q1, 3),
        "Q3": round(Q3, 3),
        "IQR": round(IQR, 3),
        "Lower Bound": round(lower, 3),
        "Upper Bound": round(upper, 3),
        "Potential Outliers": n_out,
        "Outlier %": round(pct_out, 2)
    })
    print(f"  {col}:")
    print(f"    Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}, Bounds=[{lower:.3f}, {upper:.3f}]")
    print(f"    Potential outliers: {n_out} rows ({pct_out:.2f}%)")

outlier_df = pd.DataFrame(outlier_summary)
print("\n  --- IQR Outlier Summary Table ---")
print(outlier_df.to_string(index=False))

# Decision on outlier handling
total_outlier_cols = (outlier_df["Potential Outliers"] > 0).sum()
print(f"\n  Outlier Decision Summary:")
print(f"  - {total_outlier_cols} / {len(cols_to_check)} numeric columns have potential IQR outliers.")
print(f"  - Decision: RETAIN all potential outliers.")
print(f"    Rationale: Logistics datasets naturally contain real variability")
print(f"    (congestion surges, bad weather, premium routes, etc.). Removing tails")
print(f"    would distort the prediction of lead_time_days. No removal performed.")


# =============================================================================
# STEP 5: FEATURE PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print("STEP 5: FEATURE PREPROCESSING")
print("="*70)

# 1 & 2. Timestamp column already converted in 3.C; create time features
time_features_created = []
if datetime_cols_real:
    ts_col = datetime_cols_real[0]
    print(f"\n[5.1-2] Timestamp Feature Engineering from column: {ts_col}")
    df["year"]  = df[ts_col].dt.year
    df["month"] = df[ts_col].dt.month
    df["day"]   = df[ts_col].dt.day
    df["day_of_week"] = df[ts_col].dt.dayofweek   # 0 = Monday, 6 = Sunday
    time_features_created = ["year","month","day","day_of_week"]
    for f in time_features_created:
        print(f"  OK - Created feature: {f}  (sample value: {df[f].iloc[0]})")
else:
    print("\n[5.1-2] Timestamp column: No native datetime column available.")
    print("  No time-based (year/month/day/day_of_week) features created.")

# 3. Identify categorical variables
cat_cols_final = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\n[5.3] Categorical variables for encoding: {cat_cols_final}")

# 4. Categorical encoding - preserve original for A; create model-ready for B
from sklearn.preprocessing import StandardScaler

# ---- A. CLEANED DATASET (for EDA) - keep original categoricals ----
df_cleaned = df.copy()

# ---- B. MODEL-READY DATASET (encoded + scaled where useful) ----
df_model = df_cleaned.copy()

encoding_methods = []
if cat_cols_final:
    # Low-cardinality (<=10 categories per col): one-hot encoding
    # Higher-cardinality: skip one-hot to keep dimensionality manageable;
    # still drop string cols so all numerics remain
    one_hot_cols = []
    drop_cols    = []
    for c in cat_cols_final:
        nuniq = df_model[c].nunique()
        if nuniq <= 10:
            one_hot_cols.append(c)
        else:
            drop_cols.append(c)

    if one_hot_cols:
        print(f"\n[5.4.A] One-Hot Encoding applied to low-cardinality categoricals (<=10 categories):")
        for c in one_hot_cols:
            print(f"  - {c} ({df_model[c].nunique()} categories) → one-hot")
            encoding_methods.append(f"One-Hot: {c}")
        df_model = pd.get_dummies(df_model, columns=one_hot_cols, drop_first=False, dtype=int)
    if drop_cols:
        print(f"\n[5.4.B] Dropped high-cardinality categoricals (>{10} categories each, not one-hot):")
        for c in drop_cols:
            print(f"  - {c} (nunique={df_model[c].nunique()}) → dropped for ML matrix")
            encoding_methods.append(f"Dropped (high-cardinality): {c}")
        df_model.drop(columns=drop_cols, inplace=True, errors="ignore")
else:
    print("\n[5.4] No categorical columns present → no encoding required.")

# StandardScaler: apply only to non-binary, non-target numeric columns
print("\n[5.5] Feature Scaling (StandardScaler)")
scaler_info = []
if "lead_time_days" in df_model.columns:
    target = "lead_time_days"
    numeric_features = df_model.select_dtypes(include=[np.number]).columns.tolist()

    # Identify binary (0/1 only) columns and exclude target from scaling
    binary_cols = [c for c in numeric_features
                   if c != target and set(df_model[c].dropna().unique()).issubset({0, 1})]
    cols_to_scale = [c for c in numeric_features if c != target and c not in binary_cols]

    if cols_to_scale:
        print(f"  Scaling applied to {len(cols_to_scale)} continuous numeric features:")
        for c in cols_to_scale:
            print(f"    - {c}")
        scaler_info = cols_to_scale[:]
        scaler = StandardScaler()
        df_model[cols_to_scale] = scaler.fit_transform(df_model[cols_to_scale])
        print(f"  Rationale: Scaling equalises the magnitude of different units")
        print(f"  (costs in dollars, times in hours, rates in km/l, etc.) so distance-based")
        print(f"  or gradient-based algorithms (SVM, linear regression, NN) perform stably.")
        print(f"  Target 'lead_time_days' was intentionally NOT scaled (preserves interpretability).")
    else:
        print("  No suitable continuous numeric columns to scale. Scaling skipped.")
else:
    print("  lead_time_days not found in model df.")


# =============================================================================
# STEP 6: SAVE OUTPUT DATASETS
# =============================================================================
print("\n" + "="*70)
print("STEP 6: SAVE OUTPUT DATASETS")
print("="*70)

import os
out_dir = r"e:\Yuva Intern"

cleaned_path = os.path.join(out_dir, "cleaned_logistics_data.csv")
preproc_path = os.path.join(out_dir, "preprocessed_logistics_data.csv")

df_cleaned.to_csv(cleaned_path, index=False)
df_model.to_csv(preproc_path, index=False)

print(f"  A. Cleaned dataset (for EDA):    {cleaned_path}   ({df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} cols)")
print(f"  B. Preprocessed dataset (for ML): {preproc_path}   ({df_model.shape[0]} rows × {df_model.shape[1]} cols)")
print(f"  OK - Original raw CSV left untouched.")


# =============================================================================
# STEP 8: RESULTS SUMMARY (printed inline for capture; Step 7 and 9 are narrative)
# =============================================================================
print("\n" + "="*70)
print("STEP 8: FACTUAL RESULTS SUMMARY")
print("="*70)
# Placeholders to be filled with the exact numbers after running

print(f"1. Original dataset shape       : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"2. Missing values found         : {cols_with_missing.sum() if len(cols_with_missing) > 0 else 0} total")
print(f"3. Duplicate rows found         : {n_dupes_before}")
print(f"4. Data type changes made       : {len(conversions)} conversion(s)")
for ln in conversions:
    print(f"     {ln}")
print(f"5. Outlier analysis results     :")
print(f"     - Columns scanned for IQR outliers: {len(cols_to_check)}")
tot_outliers = sum(1 for r in outlier_summary if r["Potential Outliers"] > 0)
print(f"     - Columns with ≥1 potential outlier rows: {tot_outliers}")
print(f"     - Decision on outliers: RETAINED ALL (realistic logistics variation, no rows removed)")
print(f"6. Cleaning decisions made      :")
if len(cols_with_missing) > 0:
    print(f"     - Missing values handled via median (numeric) / mode (categorical)")
else:
    print(f"     - No missing values; no imputation.")
if n_dupes_before > 0:
    print(f"     - {n_dupes_before} duplicate rows removed.")
else:
    print(f"     - No duplicates; no rows removed.")
print(f"7. New features created         : {time_features_created if time_features_created else 'None (no datetime column present)'}")
print(f"8. Encoding methods used        : {encoding_methods if encoding_methods else 'No categorical columns required encoding'}")
print(f"9. Scaling methods used         :")
if scaler_info:
    print(f"     - StandardScaler applied to {len(scaler_info)} continuous numeric columns (target lead_time_days NOT scaled)")
else:
    print(f"     - No scaling required or performed.")
print(f"10. Final cleaned dataset shape : {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
print(f"11. Final preprocessed shape    : {df_model.shape[0]} rows × {df_model.shape[1]} columns")
print(f"    Target 'lead_time_days' present: {'lead_time_days' in df_model.columns}")
