# train_abundance_models.py
import argparse
import json
import numpy as np
import pandas as pd
from joblib import dump
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to west_nile_virus_data_with_weather.csv")
    ap.add_argument("--out_poisson", default="mosq_total_poisson.joblib")
    ap.add_argument("--out_nb", default="mosq_total_nb.joblib")
    ap.add_argument("--meta", default="mosq_total_models_meta.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Ensure required columns
    required = {
        "Latitude", "Longitude", "Year", "Week",
        "temp", "humidity", "rain", "wind_speed",
        "NUMBER OF MOSQUITOES", "SPECIES", "TRAP_TYPE"
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Parse date if present (not required)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Standardize target + features
    df = df.rename(columns={"NUMBER OF MOSQUITOES": "num_mosq"})
    _coerce_numeric(df, ["Latitude", "Longitude", "Year", "Week", "temp", "humidity", "rain", "wind_speed", "num_mosq"])

    # Clean
    df["SPECIES"] = df["SPECIES"].astype(str).str.upper().str.strip()
    df["TRAP_TYPE"] = df["TRAP_TYPE"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Latitude", "Longitude", "Year", "Week", "temp", "humidity", "rain", "wind_speed", "num_mosq", "SPECIES", "TRAP_TYPE"]).copy()

    # Count must be >= 0
    df["num_mosq"] = df["num_mosq"].clip(lower=0).astype(float)

    # Helpful seasonal features (smooth weekly cycle)
    df["sin_week"] = np.sin(2 * np.pi * df["Week"].astype(float) / 52.0)
    df["cos_week"] = np.cos(2 * np.pi * df["Week"].astype(float) / 52.0)

    # Model formula: expected total mosquitoes per trap sample
    # (You can add more terms later; keep this stable for wiring into app.py.)
    formula = (
        "num_mosq ~ Latitude + Longitude + Year + Week + sin_week + cos_week "
        "+ temp + humidity + rain + wind_speed "
        "+ C(SPECIES) + C(TRAP_TYPE)"
    )

    # ---------- Poisson GLM ----------
    poisson_res = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson()
    ).fit()

    # ---------- NB GLM (alpha via method-of-moments) ----------
    mu = poisson_res.fittedvalues.to_numpy(dtype=float)
    y = df["num_mosq"].to_numpy(dtype=float)
    v = float(np.nanvar(y))
    m = float(np.nanmean(y))
    # Var ≈ mu + alpha*mu^2  -> alpha ≈ max((Var - Mean)/Mean^2, eps)
    alpha_mom = max((v - m) / (m * m + 1e-12), 1e-6)

    nb_res = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(alpha=alpha_mom)
    ).fit()

    dump(poisson_res, args.out_poisson)
    dump(nb_res, args.out_nb)

    meta = {
        "formula": formula,
        "alpha_mom": alpha_mom,
        "columns_used": [
            "Latitude","Longitude","Year","Week","sin_week","cos_week",
            "temp","humidity","rain","wind_speed","SPECIES","TRAP_TYPE"
        ],
        "n_rows": int(len(df)),
        "aic_poisson": float(poisson_res.aic),
        "aic_nb": float(nb_res.aic),
        "species_levels": sorted(df["SPECIES"].unique().tolist()),
        "trap_type_levels": sorted(df["TRAP_TYPE"].unique().tolist()),
        "default_trap_type_mode": str(df["TRAP_TYPE"].mode().iloc[0]) if len(df["TRAP_TYPE"].mode()) else None,
    }

    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {args.out_poisson}")
    print(f"Saved: {args.out_nb}")
    print(f"Saved: {args.meta}")
    print("Rows used:", len(df))
    print("alpha_mom:", alpha_mom)
    print("AIC poisson:", poisson_res.aic)
    print("AIC NB:", nb_res.aic)


if __name__ == "__main__":
    main()
