import os
import numpy as np
import pandas as pd

from joblib import dump

from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, mean_absolute_error
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


DATA_PATH = "west_nile_virus_data_with_weather.csv"

OUT_INF = "model_inf_ensemble.joblib"
OUT_SP  = "model_sp_calibrated.joblib"
OUT_AB  = "model_abundance_poisson.joblib"


# --------------------------
# Load + basic clean
# --------------------------
df = pd.read_csv(DATA_PATH)

# Standardize column names we will use
# (your file already has these, but we do this safely)
df["Year"] = pd.to_numeric(df.get("Year"), errors="coerce")
df["Week"] = pd.to_numeric(df.get("Week"), errors="coerce")
df["Latitude"] = pd.to_numeric(df.get("Latitude"), errors="coerce")
df["Longitude"] = pd.to_numeric(df.get("Longitude"), errors="coerce")

df["temp"] = pd.to_numeric(df.get("temp"), errors="coerce")
df["humidity"] = pd.to_numeric(df.get("humidity"), errors="coerce")
df["rain"] = pd.to_numeric(df.get("rain"), errors="coerce")
df["wind_speed"] = pd.to_numeric(df.get("wind_speed"), errors="coerce")

# Targets
df["RESULT"] = df.get("RESULT").astype(str)
df["is_positive"] = df["RESULT"].str.contains("positive", case=False, na=False).astype(int)

df["SPECIES"] = df.get("SPECIES").astype(str)

# Abundance target: total mosquitoes in the trap sample
df["num_mosq"] = pd.to_numeric(df.get("NUMBER OF MOSQUITOES"), errors="coerce")
df["num_mosq"] = df["num_mosq"].fillna(0.0)
df.loc[df["num_mosq"] < 0, "num_mosq"] = 0.0

# Drop rows missing core predictors
need = ["Year", "Week", "Latitude", "Longitude", "temp", "humidity", "rain", "wind_speed", "SPECIES"]
df = df.dropna(subset=need).copy()

df["Year"] = df["Year"].astype(int)
df["Week"] = df["Week"].astype(int)

# Optional: remove obviously bad coords
df = df[(df["Latitude"].between(-90, 90)) & (df["Longitude"].between(-180, 180))].copy()

# Features must match your app.py model usage
FEATURES = ["Latitude", "Longitude", "Year", "Week", "temp", "humidity", "rain", "wind_speed"]

X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Time split (better than random): last year as test
max_year = int(df["Year"].max())
train_mask = df["Year"] < max_year
test_mask = df["Year"] == max_year

X_train, X_test = X.loc[train_mask], X.loc[test_mask]

y_inf_train, y_inf_test = df.loc[train_mask, "is_positive"], df.loc[test_mask, "is_positive"]
y_sp_train, y_sp_test = df.loc[train_mask, "SPECIES"], df.loc[test_mask, "SPECIES"]
y_ab_train, y_ab_test = df.loc[train_mask, "num_mosq"], df.loc[test_mask, "num_mosq"]

print(f"Loaded: {len(df):,} rows | Train years < {max_year}: {train_mask.sum():,} | Test year = {max_year}: {test_mask.sum():,}")
print("Species classes (top 10):")
print(df["SPECIES"].value_counts().head(10))


# --------------------------
# 1) Infection probability model
# --------------------------
model_inf = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model_inf.fit(X_train, y_inf_train)

p_test = model_inf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_inf_test, p_test) if len(np.unique(y_inf_test)) > 1 else float("nan")
ll  = log_loss(y_inf_test, np.clip(p_test, 1e-6, 1 - 1e-6))

print("\n[INF] Test ROC-AUC:", auc)
print("[INF] Test LogLoss:", ll)

dump(model_inf, OUT_INF)
print(f"[INF] Saved -> {OUT_INF}")


# --------------------------
# 2) Species classifier + calibration
# --------------------------
# Base multiclass model
model_sp_base = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.07,
    max_depth=6,
    random_state=42
)
model_sp_base.fit(X_train, y_sp_train)

# Calibrate probability outputs (sigmoid is safer than isotonic with many classes)
model_sp = CalibratedClassifierCV(model_sp_base, method="sigmoid", cv=3)
model_sp.fit(X_train, y_sp_train)

sp_pred = model_sp.predict(X_test)
acc = accuracy_score(y_sp_test, sp_pred)

print("\n[SP] Test Accuracy:", acc)

dump(model_sp, OUT_SP)
print(f"[SP] Saved -> {OUT_SP}")


# --------------------------
# 3) Abundance model (expected total mosquitoes)
# --------------------------
# Poisson loss works well for counts (nonnegative); predictions are expected counts
model_ab = HistGradientBoostingRegressor(
    loss="poisson",
    max_iter=400,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
# Poisson requires nonnegative target
y_ab_train_nn = np.clip(y_ab_train.to_numpy(dtype=float), 0.0, None)
model_ab.fit(X_train, y_ab_train_nn)

ab_pred = model_ab.predict(X_test)
ab_pred = np.clip(ab_pred, 0.0, None)

mae = mean_absolute_error(y_ab_test, ab_pred)

print("\n[AB] Test MAE (mosquito count):", mae)
print(f"[AB] Example preds: min={ab_pred.min():.2f}, median={np.median(ab_pred):.2f}, max={ab_pred.max():.2f}")

dump(model_ab, OUT_AB)
print(f"[AB] Saved -> {OUT_AB}")


print("\nDone. Models created:")
print(" -", OUT_INF)
print(" -", OUT_SP)
print(" -", OUT_AB)
print("\nNOTE: These models use temp in Celsius because your CSV temp values are in Celsius.")
