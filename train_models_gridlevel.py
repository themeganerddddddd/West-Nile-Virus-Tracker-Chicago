import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import dump

FEATURE_CSV = "wnv_gridlevel_features.csv"

def main():
    df = pd.read_csv(FEATURE_CSV)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # ---- Infection model ----
    y = df["target"].astype(int)

    features_inf = [
        "Latitude","Longitude","Year","Week",
        "sin_doy","cos_doy",
        "near_pos_count_lag1","near_pos_rate_lag1","log_near_mosq_sum_lag1","near_n_samples_lag1",
        "near_pos_count_lag2","near_pos_rate_lag2","log_near_mosq_sum_lag2","near_n_samples_lag2",
        # NOTE: we intentionally do NOT include log_mosq here,
        # because grid points don't have "mosq_count" (trap-only).
        # The neighborhood mosquito history features *are* grid-computable.
    ]

    X = df[features_inf].apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_inf = HistGradientBoostingClassifier(max_iter=400, random_state=42)
    model_inf.fit(X_train, y_train)

    p = model_inf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)
    print(f"Infection model ROC-AUC: {auc:.4f}")

    dump(model_inf, "model_inf_ensemble.joblib")
    print("Saved model_inf_ensemble.joblib")

    # ---- Species model ----
    # We train species on the same grid-computable features.
    # (We could include trap-only features like log_mosq, but then grid can't use them.)
    df_sp = df[df["SPECIES"].astype(str).str.len() > 0].copy()
    y_sp = df_sp["SPECIES"].astype(str)

    X_sp = df_sp[features_inf].apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sp, y_sp, test_size=0.2, random_state=42, stratify=y_sp
    )

    base = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    model_sp = CalibratedClassifierCV(base, cv=5)
    model_sp.fit(X_train, y_train)

    pred = model_sp.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Species model accuracy: {acc:.4f}")

    dump(model_sp, "model_sp_calibrated.joblib")
    print("Saved model_sp_calibrated.joblib")

if __name__ == "__main__":
    main()
