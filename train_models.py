import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score
)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

# ----------------------------
# Config
# ----------------------------
CSV_PATH = os.environ.get("WNV_WEATHER_CSV", "west_nile_virus_data_with_weather.csv")

FEATURES = [
    "Latitude", "Longitude", "Year", "Week",
    "temp", "humidity", "rain", "wind_speed"
]

OUT_INF = "model_inf_ensemble.joblib"
OUT_SP  = "model_sp_calibrated.joblib"


def _as_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    df = pd.read_csv(CSV_PATH)

    # Basic column checks
    required = set(FEATURES + ["RESULT", "SPECIES"])
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Clean + types
    df = df.copy()
    df = _as_numeric(df, ["Latitude", "Longitude", "Year", "Week", "temp", "humidity", "rain", "wind_speed"])

    # Target: infection positive
    df["RESULT"] = df["RESULT"].astype(str)
    df["target_inf"] = df["RESULT"].str.contains("Positive", case=False, na=False).astype(int)

    # Drop unusable rows for infection model
    df_inf = df.dropna(subset=["Latitude", "Longitude", "Year", "Week"]).copy()
    df_inf[FEATURES] = df_inf[FEATURES].fillna(0.0)

    X = df_inf[FEATURES]          # pandas DF -> model gets feature_names_in_
    y = df_inf["target_inf"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ----------------------------
    # 1) Infection model (binary)
    # ----------------------------
    # HistGradientBoostingClassifier supports predict_proba and handles nonlinearities well.
    inf_model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=None,
        random_state=42
    )
    inf_model.fit(X_train, y_train)

    p_test = inf_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    ap  = average_precision_score(y_test, p_test)
    acc = accuracy_score(y_test, (p_test >= 0.5).astype(int))

    print("\n[Infection model]")
    print("Features:", list(getattr(inf_model, "feature_names_in_", [])))
    print(f"AUC: {auc:.4f} | AP: {ap:.4f} | Acc@0.5: {acc:.4f}")

    dump(inf_model, OUT_INF)
    print(f"Saved: {OUT_INF}")

    # ----------------------------
    # 2) Species model (multiclass)
    # ----------------------------
    # Goal: predict SPECIES from the same FEATURES so app can do species weighting per gridpoint.
    df_sp = df.dropna(subset=["SPECIES", "Latitude", "Longitude", "Year", "Week"]).copy()
    df_sp[FEATURES] = df_sp[FEATURES].fillna(0.0)

    # Normalize species labels a bit
    df_sp["SPECIES"] = df_sp["SPECIES"].astype(str).str.strip()
    df_sp = df_sp[df_sp["SPECIES"] != ""].copy()

    Xs = df_sp[FEATURES]
    ys = df_sp["SPECIES"]

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        Xs, ys, test_size=0.20, random_state=42, stratify=ys
    )

    # Base model: RF -> stable multiclass probabilities
    base_sp = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )

    # Calibrate probabilities (better behaved predict_proba)
    sp_model = CalibratedClassifierCV(
        estimator=base_sp,
        method="sigmoid",
        cv=3
    )
    sp_model.fit(Xs_train, ys_train)

    sp_pred = sp_model.predict(Xs_test)
    sp_acc = accuracy_score(ys_test, sp_pred)

    print("\n[Species model]")
    # CalibratedClassifierCV doesn't expose feature_names_in_ reliably; we care that we trained on FEATURES.
    print("Features used:", FEATURES)
    print(f"Accuracy: {sp_acc:.4f} | Classes: {len(sp_model.classes_)}")

    dump(sp_model, OUT_SP)
    print(f"Saved: {OUT_SP}")

    print("\nDone.")


if __name__ == "__main__":
    main()
