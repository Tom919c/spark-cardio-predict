"""
Local data cleaning, dual-label construction, and training table generation.
Core module for local data cleaning and feature engineering.

Data flow:
  raw/datasets/ (6 sources) → load_all_datasets() → enforce_data_integrity()
  → run_mice_imputation() → staging/CVD_Standard_DWD.csv
  → CardioFeatureEngineering.build_training_dataframe() → training-ready DataFrame
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ============================================================
# DWD standard columns (11 fields, dual-label version)
# ============================================================
DWD_COLUMNS = [
    "age", "gender", "bmi", "cholesterol", "diabetes",
    "hypertension", "smoker", "alcohol", "exercise",
    "label_heart", "label_stroke",
]

FEATURE_COLUMNS = [
    "age", "gender", "bmi", "cholesterol", "diabetes",
    "hypertension", "smoker", "alcohol", "exercise",
]

LABEL_COLUMNS = ["label_heart", "label_stroke"]


# ============================================================
# Helper functions
# ============================================================

def _cholesterol_to_cat(val):
    """Total cholesterol (mg/dL) → category: 1=normal, 2=borderline, 3=high."""
    if pd.isna(val):
        return np.nan
    if val < 200:
        return 1.0
    elif val < 240:
        return 2.0
    else:
        return 3.0


def _agecat_to_numeric(cat):
    """BRFSS AgeCategory → numeric age (interval midpoint)."""
    mapping = {
        "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
        "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
        "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77,
        "80 or older": 83,
    }
    return mapping.get(cat, np.nan)


def _brfss_diabetes(val):
    """BRFSS Diabetic → 0/1. Yes / Yes (during pregnancy) → 1."""
    if pd.isna(val):
        return np.nan
    if val.startswith("Yes"):
        return 1.0
    return 0.0


# ============================================================
# Dataset loaders — one per source, all return DWD-format DataFrame
# ============================================================

def load_cardio_train(path):
    """
    Cardiovascular Disease Dataset (Kaggle: sulianova).
    Raw: id|age(days)|gender(1=f,2=m)|height|weight|ap_hi|ap_lo|cholesterol|gluc|smoke|alco|active|cardio
    Labels: cardio=1 → label_heart=1, label_stroke=0 (no stroke info in this dataset).
    """
    df = pd.read_csv(path, sep=";")
    out = pd.DataFrame()
    out["age"] = (df["age"] / 365).round().astype(int)
    out["gender"] = df["gender"].map({2: 1, 1: 0})
    out["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    out["cholesterol"] = df["cholesterol"]
    out["diabetes"] = df["gluc"].apply(lambda x: 1 if x == 3 else 0)
    out["hypertension"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)
    out["smoker"] = df["smoke"].map({0: 0, 1: 2})
    out["alcohol"] = df["alco"]
    out["exercise"] = df["active"]
    out["label_heart"] = df["cardio"].astype(int)
    out["label_stroke"] = 0
    return out


def load_framingham(path):
    """
    Framingham Heart Study (Kaggle: shreyjain601).
    Labels: ANYCHD/ANGINA/HOSPMI/MI_FCHD=1 → label_heart=1, STROKE=1 → label_stroke=1.
    """
    df = pd.read_csv(path)
    time_cols = ["TIME", "PERIOD", "TIMEAP", "TIMEMI", "TIMEMIFC",
                 "TIMECHD", "TIMESTRK", "TIMECVD", "TIMEDTH", "TIMEHYP"]
    drop_extra = ["RANDID", "CIGPDAY", "BPMEDS", "HEARTRTE",
                  "HDLC", "LDLC", "DEATH", "CVD", "HYPERTEN",
                  "PREVCHD", "PREVAP", "PREVMI", "PREVSTRK"]

    out = pd.DataFrame()
    out["age"] = df["AGE"].astype(int)
    out["gender"] = df["SEX"].map({1: 1, 2: 0})
    out["bmi"] = df["BMI"]
    out["cholesterol"] = df["TOTCHOL"].apply(_cholesterol_to_cat)

    out["diabetes"] = df["DIABETES"].copy()
    gluc_dm_mask = out["diabetes"].isna() & df["GLUCOSE"].notna()
    out.loc[gluc_dm_mask, "diabetes"] = (df.loc[gluc_dm_mask, "GLUCOSE"] >= 126).astype(int)

    out["hypertension"] = df["PREVHYP"].copy()
    bp_ht_mask = out["hypertension"].isna() & df["SYSBP"].notna() & df["DIABP"].notna()
    out.loc[bp_ht_mask, "hypertension"] = (
        (df.loc[bp_ht_mask, "SYSBP"] >= 140) | (df.loc[bp_ht_mask, "DIABP"] >= 90)
    ).astype(int)

    out["smoker"] = df["CURSMOKE"].map({0: 0, 1: 2})
    out["alcohol"] = np.nan
    out["exercise"] = np.nan

    out["label_heart"] = (
        (df["ANYCHD"] == 1) | (df["ANGINA"] == 1) |
        (df["HOSPMI"] == 1) | (df["MI_FCHD"] == 1)
    ).astype(int)
    out["label_stroke"] = df["STROKE"].fillna(0).astype(int)
    return out


def load_stroke(path):
    """
    Stroke Prediction Dataset (Kaggle: fedesoriano).
    Labels: stroke=1 → label_stroke=1, label_heart=0 (no heart info).
    """
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["age"] = df["age"].astype(int)
    out["gender"] = df["gender"].map({"Male": 1, "Female": 0, "Other": np.nan})
    out["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    out["cholesterol"] = np.nan
    out["diabetes"] = (df["avg_glucose_level"] >= 126).astype(int)
    out["hypertension"] = df["hypertension"]
    out["smoker"] = df["smoking_status"].map({
        "never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": np.nan,
    })
    out["alcohol"] = np.nan
    out["exercise"] = np.nan
    out["label_heart"] = 0
    out["label_stroke"] = df["stroke"].astype(int)
    return out


def load_heart_failure(path):
    """
    Heart Failure Clinical Records (Kaggle: andrewmvd).
    Labels: DEATH_EVENT=1 → label_heart=1, label_stroke=0.
    """
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["age"] = df["age"].astype(int)
    out["gender"] = df["sex"]
    out["bmi"] = np.nan
    out["cholesterol"] = np.nan
    out["diabetes"] = df["diabetes"]
    out["hypertension"] = df["high_blood_pressure"]
    out["smoker"] = df["smoking"].map({0: 0, 1: 2})
    out["alcohol"] = np.nan
    out["exercise"] = np.nan
    out["label_heart"] = df["DEATH_EVENT"].astype(int)
    out["label_stroke"] = 0
    return out


def load_china_heart_attack(path):
    """
    China Heart Attack Risk Dataset (Kaggle: ankushpanday2).
    Labels: Heart_Attack='Yes' → label_heart=1, label_stroke=0.
    """
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["age"] = df["Age"].astype(int)
    out["gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    out["bmi"] = np.nan
    out["cholesterol"] = df["Cholesterol_Level"].map({"Low": 1, "Normal": 1, "High": 3})
    out["diabetes"] = df["Diabetes"].map({"Yes": 1, "No": 0})
    out["hypertension"] = df["Hypertension"].map({"Yes": 1, "No": 0})
    bp_ht = pd.to_numeric(df["Blood_Pressure"], errors="coerce")
    bp_ht_mask = out["hypertension"].isna() & bp_ht.notna()
    out.loc[bp_ht_mask, "hypertension"] = (bp_ht[bp_ht_mask] >= 140).astype(int)
    out["smoker"] = df["Smoking_Status"].map({"Non-Smoker": 0, "Smoker": 2})
    out["alcohol"] = df["Alcohol_Consumption"].map({"Yes": 1, "No": 0})
    out["exercise"] = df["Physical_Activity"].map({"Low": 0, "Medium": 1, "High": 1})
    out["label_heart"] = df["Heart_Attack"].map({"Yes": 1, "No": 0})
    out["label_stroke"] = 0
    return out


def load_brfss(path):
    """
    BRFSS 2020 Heart Disease Dataset (Kaggle: kamilpytlak).
    Labels: HeartDisease='Yes' → label_heart=1, Stroke='Yes' → label_stroke=1.
    """
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["age"] = df["AgeCategory"].apply(_agecat_to_numeric)
    out["gender"] = df["Sex"].map({"Male": 1, "Female": 0})
    out["bmi"] = df["BMI"]
    out["cholesterol"] = np.nan
    out["diabetes"] = df["Diabetic"].apply(_brfss_diabetes)
    out["hypertension"] = np.nan
    out["smoker"] = df["Smoking"].map({"Yes": 2, "No": 0})
    out["alcohol"] = df["AlcoholDrinking"].map({"Yes": 1, "No": 0})
    out["exercise"] = df["PhysicalActivity"].map({"Yes": 1, "No": 0})
    out["label_heart"] = (df["HeartDisease"] == "Yes").astype(int)
    out["label_stroke"] = (df["Stroke"] == "Yes").astype(int)
    return out


# ============================================================
# Pipeline functions
# ============================================================

def load_all_datasets(raw_dir):
    """Load and merge all 6 source datasets into a single DWD DataFrame."""
    datasets = [
        load_cardio_train(os.path.join(raw_dir, "cardio_train.csv")),
        load_framingham(os.path.join(raw_dir, "Framingham Dataset.csv")),
        load_stroke(os.path.join(raw_dir, "healthcare-dataset-stroke-data.csv")),
        load_heart_failure(os.path.join(raw_dir, "heart_failure_clinical_records_dataset.csv")),
        load_china_heart_attack(os.path.join(raw_dir, "heart_attack_china.csv")),
        load_brfss(os.path.join(raw_dir, "heart_2020_cleaned.csv")),
    ]
    df = pd.concat(datasets, ignore_index=True)
    return df[DWD_COLUMNS]


def enforce_data_integrity(df):
    """
    Cross-dataset integrity cleaning:
      - Clip extreme BMI (<10 or >80 → NaN)
      - Clip extreme age (<1 or >120 → NaN)
      - Enforce valid ranges for categorical columns.
    """
    df = df.copy()
    df.loc[df["bmi"].notna() & ((df["bmi"] < 10) | (df["bmi"] > 80)), "bmi"] = np.nan
    df.loc[df["age"].notna() & ((df["age"] < 1) | (df["age"] > 120)), "age"] = np.nan

    df.loc[df["cholesterol"].notna() & ~df["cholesterol"].isin([1, 2, 3]), "cholesterol"] = np.nan
    df.loc[df["smoker"].notna() & ~df["smoker"].isin([0, 1, 2]), "smoker"] = np.nan
    df.loc[df["gender"].notna() & ~df["gender"].isin([0, 1]), "gender"] = np.nan

    for col in ["alcohol", "exercise", "diabetes", "hypertension", "label_heart", "label_stroke"]:
        df.loc[df[col].notna() & ~df[col].isin([0, 1]), col] = np.nan

    return df


def run_mice_imputation(df):
    """
    MICE imputation using IterativeImputer (RandomForest estimator).
    Imputes the 9 feature columns; label columns are filled with 0.
    """
    feature_cols = [c for c in DWD_COLUMNS if c not in LABEL_COLUMNS]
    X = df[feature_cols].copy()
    y_heart = df["label_heart"].copy()
    y_stroke = df["label_stroke"].copy()

    if X.isna().sum().sum() == 0:
        df_out = X.copy()
        df_out["label_heart"] = y_heart.fillna(0).astype(int).clip(0, 1).values
        df_out["label_stroke"] = y_stroke.fillna(0).astype(int).clip(0, 1).values
        return df_out[DWD_COLUMNS]

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1),
        max_iter=10, random_state=42, verbose=0,
    )
    X_imputed = imputer.fit_transform(X)
    df_imp = pd.DataFrame(X_imputed, columns=feature_cols, index=df.index)

    # Post-process: clip to valid ranges
    df_imp["age"] = df_imp["age"].round().clip(1, 120).astype(int)
    df_imp["bmi"] = df_imp["bmi"].clip(10.0, 80.0).round(1)
    df_imp["cholesterol"] = df_imp["cholesterol"].round().clip(1, 3).astype(int)
    df_imp["smoker"] = df_imp["smoker"].round().clip(0, 2).astype(int)
    for col in ["gender", "diabetes", "hypertension", "alcohol", "exercise"]:
        df_imp[col] = (df_imp[col] >= 0.5).astype(int)

    df_imp["label_heart"] = y_heart.fillna(0).astype(int).clip(0, 1).values
    df_imp["label_stroke"] = y_stroke.fillna(0).astype(int).clip(0, 1).values

    return df_imp[DWD_COLUMNS]


def build_dwd_dataset(raw_dir, verbose=True):
    """Full pipeline: load 6 sources → merge → integrity clean → MICE → clean DWD DataFrame."""
    names = [
        "cardio_train", "Framingham", "Stroke", "Heart Failure",
        "China Heart Attack", "BRFSS 2020",
    ]
    loaders = [
        load_cardio_train, load_framingham, load_stroke,
        load_heart_failure, load_china_heart_attack, load_brfss,
    ]
    files = [
        "cardio_train.csv", "Framingham Dataset.csv",
        "healthcare-dataset-stroke-data.csv",
        "heart_failure_clinical_records_dataset.csv",
        "heart_attack_china.csv", "heart_2020_cleaned.csv",
    ]

    datasets = []
    for i, (name, loader, fname) in enumerate(zip(names, loaders, files)):
        if verbose:
            print(f"  [{i + 1}/6] Loading {name} ...", end=" ")
        df_src = loader(os.path.join(raw_dir, fname))
        datasets.append(df_src)
        if verbose:
            print(f"{len(df_src):,} rows")

    if verbose:
        print("  Merging 6 datasets ...", end=" ")
    df = pd.concat(datasets, ignore_index=True)
    df = df[DWD_COLUMNS]
    if verbose:
        print(f"{len(df):,} total rows")

    if verbose:
        before = df.isna().sum().sum()
        print(f"  Integrity cleaning ... (NaN cells: {before:,})", end=" ")
    df = enforce_data_integrity(df)
    if verbose:
        after = df.isna().sum().sum()
        print(f"→ {after:,}")

    if verbose:
        print("  MICE imputation (IterativeImputer) ...", end=" ", flush=True)
    df = run_mice_imputation(df)
    if verbose:
        remaining = df.isna().sum().sum()
        print(f"remaining NaN: {remaining}")

    return df


# ============================================================
# CardioFeatureEngineering — training-oriented preprocessing
# ============================================================

class CardioFeatureEngineering:
    """Preprocessing for the standardized DWD dataset before model training."""

    def __init__(self, feature_columns, target_column, test_size, random_state,
                 label_columns=None):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.label_columns = label_columns or LABEL_COLUMNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, dataframe):
        """Generate pre-training summary: shapes, preview, train/test split info."""
        model_frame = self.build_training_dataframe(dataframe)
        split_index = int(len(model_frame) * (1 - self.test_size))
        train_frame = model_frame.iloc[:split_index]
        test_frame = model_frame.iloc[split_index:]

        return {
            "processed_shape": [int(model_frame.shape[0]), int(model_frame.shape[1])],
            "processed_columns": model_frame.columns.tolist(),
            "processed_preview": model_frame.head(5).to_dict(orient="records"),
            "train_shape": [int(train_frame.shape[0]), int(train_frame.shape[1])],
            "test_shape": [int(test_frame.shape[0]), int(test_frame.shape[1])],
            "test_size": self.test_size,
            "random_state": self.random_state,
        }

    def build_training_dataframe(self, dataframe):
        """Build training-ready DataFrame: clean → dual targets → select columns."""
        working_frame = dataframe.copy()
        working_frame = self._clean_basic_values(working_frame)
        working_frame = self._build_dual_targets(working_frame)

        available_features = [c for c in self.feature_columns if c in working_frame.columns]
        label_cols = [c for c in self.label_columns if c in working_frame.columns]

        model_frame = working_frame[
            available_features + label_cols + ["heart_risk", "stroke_risk"]
        ].copy()
        return model_frame.dropna()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_basic_values(self, dataframe):
        """Filter rows with out-of-range values in DWD-format data."""
        df = dataframe.copy()

        for col in ["age", "bmi", "cholesterol"]:
            if col in df.columns:
                df = df[df[col] > 0]

        for col in ["gender", "diabetes", "hypertension", "alcohol", "exercise"]:
            if col in df.columns:
                df = df[df[col].isin([0, 1])]

        if "smoker" in df.columns:
            df = df[df["smoker"].isin([0, 1, 2])]
        if "cholesterol" in df.columns:
            df = df[df["cholesterol"].isin([1, 2, 3])]

        return df

    def _build_dual_targets(self, dataframe):
        """
        Build heart_risk / stroke_risk columns.
        DWD format: use label_heart / label_stroke directly.
        Legacy format: derive from target_disease (0=healthy, 1=heart, 2=stroke).
        """
        df = dataframe.copy()

        if "label_heart" in df.columns and "label_stroke" in df.columns:
            df["heart_risk"] = df["label_heart"].astype(int)
            df["stroke_risk"] = df["label_stroke"].astype(int)
        elif self.target_column in df.columns:
            df["heart_risk"] = (df[self.target_column] == 1).astype(int)
            df["stroke_risk"] = (df[self.target_column] == 2).astype(int)

        return df
