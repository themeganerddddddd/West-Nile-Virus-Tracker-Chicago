# app.py
import os
import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
from alphashape import alphashape
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from joblib import Memory, load
from shapely.geometry import Point, shape
from sklearn.neighbors import BallTree

from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler


# ============================================================
# CONFIG
# ============================================================

ORIG_WNV_CSV = "west_nile_virus_data.csv"
WEATHER_WNV_CSV = "west_nile_virus_data_with_weather.csv"  # optional fallback if exists

OWM_KEY = os.environ.get("OWM_KEY", "79e121d53c70ecf6ebd8a573f98d702e")
LAT, LON = 41.8781, -87.6298  # Chicago coords

POP_DENSITY_CSV = "pop_density.csv"

# If you want to shrink compute, raise step. If you want fewer “grid artifacts”, LOWER step.
GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP = 41.6445, 42.0230, 0.003
GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP = -87.9409, -87.5237, 0.003


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = False

memory = Memory(location="./cache_dir", verbose=0)


# ============================================================
# CONSTANTS / UTILS
# ============================================================

EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(lat1, lon1, lat2_arr, lon2_arr):
    """
    Great-circle distance (km) from a single point to arrays.
    """
    lat2_arr = np.asarray(lat2_arr, dtype=float)
    lon2_arr = np.asarray(lon2_arr, dtype=float)

    lat1r = np.radians(float(lat1))
    lon1r = np.radians(float(lon1))
    lat2r = np.radians(lat2_arr)
    lon2r = np.radians(lon2_arr)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        v = float(x)
        return default if np.isnan(v) else v
    except Exception:
        return default


def _clip01(x):
    return float(np.clip(float(x), 0.0, 1.0))


def _normalize_robust(arr, lo_q=0.01, hi_q=0.99, eps=1e-12):
    """
    Robust 0..1 scaling so one extreme point doesn’t flatten your whole surface.
    """
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    lo = np.nanquantile(a, lo_q)
    hi = np.nanquantile(a, hi_q)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo + eps):
        mn = np.nanmin(a)
        mx = np.nanmax(a)
        if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx <= mn + eps):
            return np.zeros_like(a)
        return (a - mn) / (mx - mn + eps)
    a = np.clip(a, lo, hi)
    return (a - lo) / (hi - lo + eps)


# ============================================================
# LOAD MODELS (infection + species) + ABUNDANCE (Poisson/NB)
# ============================================================

model_inf = load("model_inf_ensemble.joblib")
model_sp = load("model_sp_calibrated.joblib")

FEATURES_INF = list(getattr(model_inf, "feature_names_in_", []))
FEATURES_SP = list(getattr(model_sp, "feature_names_in_", []))

# --- Abundance models (statsmodels GLMResults saved via joblib) ---
MOSQ_ABUNDANCE_KIND = os.environ.get("MOSQ_ABUNDANCE_KIND", "poisson").strip().lower()
MOSQ_POISSON_PATH = os.environ.get("MOSQ_POISSON_PATH", "mosq_total_poisson.joblib")
MOSQ_NB_PATH = os.environ.get("MOSQ_NB_PATH", "mosq_total_nb.joblib")

try:
    model_ab_poisson = load(MOSQ_POISSON_PATH)
except Exception as e:
    print(f"[WARN] Could not load Poisson abundance model at {MOSQ_POISSON_PATH}: {e}")
    model_ab_poisson = None

try:
    model_ab_nb = load(MOSQ_NB_PATH)
except Exception as e:
    print(f"[WARN] Could not load NB abundance model at {MOSQ_NB_PATH}: {e}")
    model_ab_nb = None


def _get_abundance_model():
    # Prefer requested; fall back safely
    if MOSQ_ABUNDANCE_KIND == "nb" and model_ab_nb is not None:
        return model_ab_nb
    if model_ab_poisson is not None:
        return model_ab_poisson
    return model_ab_nb  # may still be None


model_ab = _get_abundance_model()

# --- Abundance model meta (levels + defaults) ---
MOSQ_META_PATH = os.environ.get("MOSQ_META_PATH", "mosq_total_models_meta.json")
MOSQ_META = {}
if os.path.exists(MOSQ_META_PATH):
    try:
        with open(MOSQ_META_PATH, "r", encoding="utf-8") as f:
            MOSQ_META = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load abundance meta {MOSQ_META_PATH}: {e}")
        MOSQ_META = {}
else:
    print(f"[WARN] Abundance meta not found at {MOSQ_META_PATH}; unseen categories may break predict().")


# ============================================================
# LOAD ORIGINAL WNV DATA (PRIMARY SOURCE)
# ============================================================

wnv_raw = pd.read_csv(ORIG_WNV_CSV)

wnv_raw = wnv_raw.rename(
    columns={
        "SEASON YEAR": "Year",
        "WEEK": "Week",
        "TEST DATE": "date",
        "NUMBER OF MOSQUITOES": "num_mosq",
        "SPECIES": "Species",
    }
)

wnv_raw["date"] = pd.to_datetime(wnv_raw["date"], errors="coerce")
wnv_raw["Year"] = pd.to_numeric(wnv_raw["Year"], errors="coerce")
wnv_raw["Week"] = pd.to_numeric(wnv_raw["Week"], errors="coerce")
wnv_raw["Latitude"] = pd.to_numeric(wnv_raw["Latitude"], errors="coerce")
wnv_raw["Longitude"] = pd.to_numeric(wnv_raw["Longitude"], errors="coerce")
wnv_raw["num_mosq"] = pd.to_numeric(wnv_raw.get("num_mosq", np.nan), errors="coerce")

wnv_raw["is_positive"] = (
    wnv_raw["RESULT"].astype(str).str.contains("Positive", case=False, na=False).astype(int)
)

wnv_raw = wnv_raw.dropna(subset=["Latitude", "Longitude", "Year", "Week"]).copy()
wnv_raw["Year"] = wnv_raw["Year"].astype(int)
wnv_raw["Week"] = wnv_raw["Week"].astype(int)


# ============================================================
# OPTIONAL: LOAD WEATHER-ENRICHED DATA (FALLBACK FOR WEEKLY MEANS)
# ============================================================

df_ww = None
if os.path.exists(WEATHER_WNV_CSV):
    try:
        wnv_weather = pd.read_csv(WEATHER_WNV_CSV, parse_dates=["date"])
        if {"Year", "Week", "temp", "humidity", "rain", "wind_speed"}.issubset(set(wnv_weather.columns)):
            df_ww = (
                wnv_weather.groupby(["Year", "Week"])[["temp", "humidity", "rain", "wind_speed"]]
                .mean()
                .sort_index()
            )
    except Exception:
        df_ww = None


# ============================================================
# POPULATION DENSITY
# ============================================================

pop_df = pd.read_csv(POP_DENSITY_CSV)
pop_df["Latitude"] = pd.to_numeric(pop_df["Latitude"], errors="coerce")
pop_df["Longitude"] = pd.to_numeric(pop_df["Longitude"], errors="coerce")
pop_df["pop_density"] = pd.to_numeric(pop_df["pop_density"], errors="coerce")
pop_df = pop_df.dropna(subset=["Latitude", "Longitude", "pop_density"]).copy()

pop_tree = BallTree(np.radians(pop_df[["Latitude", "Longitude"]].values), metric="haversine")
GLOBAL_POP_MEAN = float(pop_df["pop_density"].mean())

SPECIES_WEIGHTS = {
    "CULEX PIPIENS": 1.0,
    "CULEX RESTUANS": 1.45,
    "CULEX PIPIENS/RESTUANS": 1.0,
    # other species default 1.0
}


@memory.cache
def cached_query_radius(coords_radians_tuple, radius_km):
    arr = np.array(coords_radians_tuple, dtype=float)
    return pop_tree.query_radius(arr, r=float(radius_km) / 6371.0)


# ============================================================
# CONCAVE HULL (from positives)
# ============================================================

pos_pts = wnv_raw.loc[wnv_raw["is_positive"] == 1, ["Longitude", "Latitude"]].dropna().values
if len(pos_pts) >= 10:
    concave_hull = alphashape(pos_pts, alpha=0.1)
else:
    concave_hull = None


# ============================================================
# WEATHER FETCH (LIVE) + FALLBACK TO df_ww
# ============================================================

def fetch_weekly_avg(dt: datetime):
    """
    dt should be a Monday. Pull 7 days of hourly via OWM timemachine,
    average it; if that fails, fall back to df_ww if available.
    Returns METRIC units (temp in Celsius) because units=metric.
    """
    temps, hums, rains, winds = [], [], [], []
    for day_offset in range(7):
        ts = int((dt + timedelta(days=day_offset)).timestamp())
        url = (
            f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
            f"?lat={LAT}&lon={LON}&dt={ts}&appid={OWM_KEY}&units=metric"
        )
        try:
            res = requests.get(url, timeout=20).json()
        except Exception:
            res = {}

        for h in res.get("hourly", []) or []:
            try:
                temps.append(float(h.get("temp", np.nan)))
                hums.append(float(h.get("humidity", np.nan)))
                winds.append(float(h.get("wind_speed", np.nan)))
                rains.append(float((h.get("rain", {}) or {}).get("1h", 0.0)))
            except Exception:
                continue

        time.sleep(1)  # throttle

    if len(temps) == 0:
        if df_ww is not None:
            try:
                y, w = dt.isocalendar()[0:2]
                row = df_ww.loc[(int(y), int(w))]
                return {
                    "temp": float(row["temp"]),
                    "humidity": float(row["humidity"]),
                    "rain": float(row["rain"]),
                    "wind_speed": float(row["wind_speed"]),
                }
            except Exception:
                pass

        return {"temp": 20.0, "humidity": 50.0, "rain": 0.0, "wind_speed": 2.0}

    return {
        "temp": float(np.nanmean(temps)),
        "humidity": float(np.nanmean(hums)),
        "rain": float(np.nanmean(rains)),
        "wind_speed": float(np.nanmean(winds)),
    }


def get_weather_for_week(year: int, week: int, cw_from_request: Optional[Dict[str, Any]]):
    """
    Ensure we always return a usable weather dict with keys:
    temp, humidity, rain, wind_speed, source

    Priority:
    1) values provided by request (if not NaN)
    2) df_ww weekly mean (if available)
    3) OWM live (timemachine)
    4) defaults
    """
    cw_from_request = cw_from_request or {}

    cw = {
        "temp": _safe_float(cw_from_request.get("temp")),
        "humidity": _safe_float(cw_from_request.get("humidity")),
        "rain": _safe_float(cw_from_request.get("rain")),
        "wind_speed": _safe_float(cw_from_request.get("wind_speed")),
    }

    # If all provided, "request". Otherwise we may upgrade.
    source = "request"

    # df_ww
    if any(np.isnan(cw[k]) for k in cw) and df_ww is not None:
        try:
            row = df_ww.loc[(int(year), int(week))]
            cw2 = {
                "temp": float(row["temp"]),
                "humidity": float(row["humidity"]),
                "rain": float(row["rain"]),
                "wind_speed": float(row["wind_speed"]),
            }
            for k in cw:
                if np.isnan(cw[k]):
                    cw[k] = cw2[k]
            source = "df_ww"
        except Exception:
            pass

    # OWM fallback
    if any(np.isnan(cw[k]) for k in cw):
        mon = datetime.fromisocalendar(int(year), int(week), 1)
        cw2 = fetch_weekly_avg(mon)
        for k in cw:
            if np.isnan(cw[k]):
                cw[k] = cw2[k]
        source = "owm"

    # final defaults
    used_default = False
    for k, default in [("temp", 20.0), ("humidity", 50.0), ("rain", 0.0), ("wind_speed", 2.0)]:
        if np.isnan(cw[k]):
            cw[k] = default
            used_default = True
    if used_default:
        source = "default"

    cw["source"] = source
    return cw


# ============================================================
# HISTORY KDE SURFACE + SMOOTHING (THE “ACCURACY” UPGRADES)
# ============================================================

def _collect_positive_points(
    year: int,
    week: int,
    lookbacks: Tuple[int, int] = (4, 8),
    cross_years: bool = True,
    cross_year_window: int = 1,
) -> np.ndarray:
    """
    Collect positive points:
    - same-year lookback up to max(lookbacks)
    - cross-year same week-of-year ± cross_year_window
    """
    y0, w0 = int(year), int(week)
    pts = []

    w_long = int(max(lookbacks))
    for dw in range(1, w_long + 1):
        ww = w0 - dw
        if ww < 1:
            break
        dfw = wnv_raw[(wnv_raw["Year"] == y0) & (wnv_raw["Week"] == ww) & (wnv_raw["is_positive"] == 1)]
        if len(dfw):
            pts.append(dfw[["Latitude", "Longitude"]].dropna().values)

    if cross_years:
        years = sorted(pd.unique(wnv_raw["Year"].dropna()))
        for yy in years:
            yy = int(yy)
            if yy == y0:
                continue
            for ww in range(w0 - int(cross_year_window), w0 + int(cross_year_window) + 1):
                if ww < 1 or ww > 53:
                    continue
                dfw = wnv_raw[(wnv_raw["Year"] == yy) & (wnv_raw["Week"] == int(ww)) & (wnv_raw["is_positive"] == 1)]
                if len(dfw):
                    pts.append(dfw[["Latitude", "Longitude"]].dropna().values)

    if not pts:
        return np.empty((0, 2), dtype=float)
    return np.vstack(pts).astype(float)


def add_history_features_to_grid_kde(
    grid_df: pd.DataFrame,
    year: int,
    week: int,
    hist_radius_km: float = 1.5,
    lookbacks: Tuple[int, int] = (4, 8),
    kernel_sigma_km: float = 0.9,
    cross_years: bool = True,
    cross_year_window: int = 1,
) -> pd.DataFrame:
    """
    Adds a smooth KDE-style historical score hist_score in [0,1].
    This eliminates hard “blocks/lines” caused by binary radius scoring.
    """
    pts = _collect_positive_points(
        year=year,
        week=week,
        lookbacks=lookbacks,
        cross_years=cross_years,
        cross_year_window=cross_year_window,
    )

    if pts.shape[0] == 0:
        grid_df["hist_score"] = 0.0
        return grid_df

    glat = grid_df["Latitude"].to_numpy(dtype=float)
    glon = grid_df["Longitude"].to_numpy(dtype=float)
    plat = pts[:, 0].astype(float)
    plon = pts[:, 1].astype(float)

    tree = BallTree(np.radians(np.c_[plat, plon]), metric="haversine")
    r_rad = float(hist_radius_km) / EARTH_RADIUS_KM
    idxs = tree.query_radius(np.radians(np.c_[glat, glon]), r=r_rad)

    sigma = max(1e-6, float(kernel_sigma_km))
    scores = np.zeros(len(grid_df), dtype=float)

    # Exact haversine distances to neighbors (stable)
    for i, neigh in enumerate(idxs):
        if len(neigh) == 0:
            continue
        p0 = np.radians([glat[i], glon[i]])
        pn = np.radians(np.c_[plat[neigh], plon[neigh]])

        dlat = pn[:, 0] - p0[0]
        dlon = pn[:, 1] - p0[1]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(p0[0]) * np.cos(pn[:, 0]) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        d_km = c * EARTH_RADIUS_KM

        w = np.exp(-(d_km ** 2) / (2.0 * sigma ** 2))
        scores[i] = float(w.sum())

    grid_df["hist_score"] = _normalize_robust(scores, 0.01, 0.99).astype(float)
    return grid_df


def smooth_grid_risk_knn(grid_df: pd.DataFrame, col: str, k: int = 60, sigma_km: float = 0.9) -> pd.Series:
    """
    kNN Gaussian smoothing on the grid to remove discretization striping.
    """
    coords = np.radians(grid_df[["Latitude", "Longitude"]].to_numpy(dtype=float))
    if len(coords) == 0:
        return pd.Series([], dtype=float, index=grid_df.index)

    tree = BallTree(coords, metric="haversine")
    k = int(max(5, min(int(k), len(grid_df))))
    dists, idxs = tree.query(coords, k=k)
    d_km = dists * EARTH_RADIUS_KM

    sigma = max(1e-6, float(sigma_km))
    w = np.exp(-(d_km ** 2) / (2.0 * sigma ** 2))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)

    v = grid_df[col].to_numpy(dtype=float)
    sm = np.sum(w * v[idxs], axis=1)
    return pd.Series(sm, index=grid_df.index)


# ============================================================
# GRID BUILD (CACHED) — UPGRADED
# ============================================================

@memory.cache
def build_grid(
    year: int,
    week: int,
    pop_strength: float,
    pop_radius_km: float,
    cw: dict,
    lw1: dict,
    lw2: dict,
    trap_effort_mean: float = None,
    hist_strength: float = 0.40,
    hist_radius_km: float = 1.5,
    hist_lookback_1: int = 4,
    hist_lookback_2: int = 8,
    # extra accuracy knobs (safe defaults)
    hist_kernel_sigma_km: float = 0.9,
    hist_cross_years: bool = True,
    hist_cross_year_window: int = 1,
    smooth_final: bool = True,
    smooth_k: int = 60,
    smooth_sigma_km: float = 0.9,
    use_population: bool = True,
):
    """
    Build grid points, predict infection + species, compute expected mosquito abundance,
    and compute final risk surfaces (raw and pop-weighted).

    NEW:
      - pred_mosq_total: expected total mosquitoes from abundance model
      - pred_mosq_infected: expected infected mosquitoes = pred_mosq_total * risk_inf
      - pred_prevalence: infection prevalence %
      - weather_source: tells whether weather came from request / df_ww / owm / default
    """

    year = int(year)
    week = int(week)
    pop_strength = float(pop_strength)
    pop_radius_km = float(pop_radius_km)

    hist_strength = _clip01(hist_strength)
    hist_radius_km = float(hist_radius_km)

    cw = get_weather_for_week(year, week, cw)

    # Temporal features (kept; not required by current models)
    grid_date = datetime.fromisocalendar(year, week, 1)
    doy = grid_date.timetuple().tm_yday
    sin_doy = np.sin(2 * np.pi * doy / 365.0)
    cos_doy = np.cos(2 * np.pi * doy / 365.0)

    # Build grid points (optionally inside hull)
    recs = []
    lat_vals = np.arange(GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP)
    lon_vals = np.arange(GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP)

    for lat in lat_vals:
        for lon in lon_vals:
            if concave_hull is not None:
                if not concave_hull.covers(Point(lon, lat)):
                    continue

            recs.append(
                {
                    "Latitude": float(lat),
                    "Longitude": float(lon),

                    # model-required
                    "Year": int(year),
                    "Week": int(week),
                    "temp": float(cw["temp"]),
                    "humidity": float(cw["humidity"]),
                    "rain": float(cw["rain"]),
                    "wind_speed": float(cw["wind_speed"]),

                    # extra (not used by your current models)
                    "day_of_year": int(doy),
                    "sin_doy": float(sin_doy),
                    "cos_doy": float(cos_doy),
                    "trap_effort_mean": _safe_float(trap_effort_mean),

                    # placeholders
                    "risk_inf": np.nan,
                    "Species": None,
                    "risk_model": np.nan,
                    "hist_score": np.nan,
                    "risk_blend": np.nan,
                    "species_weight": np.nan,
                    "pop_density": np.nan,
                    "pop_weight": np.nan,
                    "pop_multiplier": np.nan,
                    "risk_raw_value": np.nan,
                    "risk_pop_value": np.nan,
                    "risk_raw": np.nan,
                    "risk_pop": np.nan,
                    "risk_final": np.nan,

                    # abundance outputs
                    "pred_mosq_total": np.nan,
                    "pred_mosq_infected": np.nan,
                    "pred_prevalence": np.nan,

                    # weather source for debugging
                    "weather_source": str(cw.get("source", "unknown")),
                }
            )

    grid_df = pd.DataFrame(recs)
    if len(grid_df) == 0:
        return grid_df, 0.0, 0.0

    # Predict infection risk + species using named features
    X_inf = grid_df.reindex(columns=FEATURES_INF).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_sp = grid_df.reindex(columns=FEATURES_SP).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    grid_df["risk_inf"] = model_inf.predict_proba(X_inf)[:, 1]
    sp_probs = model_sp.predict_proba(X_sp)
    grid_df["Species"] = model_sp.classes_[np.argmax(sp_probs, axis=1)]

    # ------------------------------------------------------------
    # Abundance model (Poisson/NB GLM) -> expected TOTAL mosquitoes
    # IMPORTANT: statsmodels formula was trained on SPECIES + TRAP_TYPE
    # so we must provide those exact column names (uppercase).
    # Also clamp categories to training levels to avoid patsy errors.
    # ------------------------------------------------------------

    # Week cycle terms required by formula
    grid_df["sin_week"] = np.sin(2 * np.pi * grid_df["Week"].astype(float) / 52.0)
    grid_df["cos_week"] = np.cos(2 * np.pi * grid_df["Week"].astype(float) / 52.0)

    # Exact column names expected by the statsmodels formula
    grid_df["SPECIES"] = grid_df["Species"].astype(str).fillna("UNKNOWN").str.upper().str.strip()

    default_trap_type = (
        str(MOSQ_META.get("default_trap_type_mode", "GRAVID"))
        if isinstance(MOSQ_META, dict) else "GRAVID"
    )
    grid_df["TRAP_TYPE"] = default_trap_type

    # Clamp categoricals to training levels (prevents predict() from failing)
    species_levels = set(
        [s.upper().strip() for s in MOSQ_META.get("species_levels", []) if isinstance(s, str)]
    )
    trap_levels = set(
        [t.upper().strip() for t in MOSQ_META.get("trap_type_levels", []) if isinstance(t, str)]
    )

    if species_levels:
        safe_species = next(iter(species_levels))
        grid_df.loc[~grid_df["SPECIES"].isin(species_levels), "SPECIES"] = safe_species

    if trap_levels:
        safe_trap = default_trap_type.upper().strip()
        if safe_trap not in trap_levels:
            safe_trap = next(iter(trap_levels))
        grid_df.loc[~grid_df["TRAP_TYPE"].isin(trap_levels), "TRAP_TYPE"] = safe_trap

    # Coerce numeric columns used in formula
    for c in ["Latitude", "Longitude", "Year", "Week", "temp", "humidity", "rain", "wind_speed", "sin_week", "cos_week"]:
        grid_df[c] = pd.to_numeric(grid_df[c], errors="coerce").fillna(0.0)

    # ============================================================
    # Predict expected TOTAL mosquitoes (statsmodels GLM w/ Patsy)
    # ============================================================
    if model_ab is not None:
        try:
            design_info = getattr(model_ab.model.data, "design_info", None)

            if design_info is not None:
                from patsy import build_design_matrices

                exog = build_design_matrices(
                    [design_info],
                    grid_df,
                    return_type="dataframe"
                )[0]

                # CRITICAL: exog is already transformed -> prevent Patsy from re-running
                pred_total = model_ab.predict(exog, transform=False)

            else:
                # If no design_info, fall back to raw DataFrame prediction
                pred_total = model_ab.predict(grid_df)

            pred_total = np.asarray(pred_total, dtype=float)
            pred_total = np.where(np.isfinite(pred_total), pred_total, 0.0)
            pred_total = np.clip(pred_total, 0.0, None)

            grid_df["pred_mosq_total"] = pred_total

            # Optional sanity debug
            if float(np.nanmax(pred_total)) <= 0.0:
                print("[WARN] Abundance model returned all zeros after predict().")
                print("[WARN] SPECIES sample:", grid_df["SPECIES"].value_counts().head(5).to_dict())
                print("[WARN] TRAP_TYPE:", grid_df["TRAP_TYPE"].unique().tolist())

        except Exception as e:
            print("[ERROR] Abundance predict crashed:", repr(e))
            print("[ERROR] Columns available:", sorted(list(grid_df.columns)))
            print("[ERROR] SPECIES/TRAP_TYPE sample:",
                  grid_df[["SPECIES", "TRAP_TYPE"]].head(5).to_dict("records"))

            grid_df["pred_mosq_total"] = 0.0
    else:
        print("[WARN] model_ab is None — abundance disabled.")
        grid_df["pred_mosq_total"] = 0.0

    # Expected infected mosquitoes (use TRUE infection probability, not normalized heatmap)
    grid_df["pred_mosq_infected"] = (
        grid_df["pred_mosq_total"].astype(float) * grid_df["risk_inf"].astype(float)
    ).clip(lower=0.0)

    # Infection prevalence %
    denom = np.maximum(grid_df["pred_mosq_total"].astype(float), 1e-9)
    grid_df["pred_prevalence"] = (100.0 * grid_df["pred_mosq_infected"].astype(float) / denom).clip(lower=0.0)

    # Robust normalize model risk -> 0..1
    grid_df["risk_model"] = _normalize_robust(grid_df["risk_inf"].to_numpy(dtype=float), 0.01, 0.99)

    # Smooth history KDE -> hist_score
    grid_df = add_history_features_to_grid_kde(
        grid_df,
        year=year,
        week=week,
        hist_radius_km=hist_radius_km,
        lookbacks=(int(hist_lookback_1), int(hist_lookback_2)),
        kernel_sigma_km=float(hist_kernel_sigma_km),
        cross_years=bool(hist_cross_years),
        cross_year_window=int(hist_cross_year_window),
    )

    # Blend model risk + history
    grid_df["risk_blend"] = (1.0 - hist_strength) * grid_df["risk_model"] + hist_strength * grid_df["hist_score"]

    # Species weights
    grid_df["species_weight"] = (
        grid_df["Species"].astype(str).str.upper().map(SPECIES_WEIGHTS).fillna(1.0).astype(float)
    )

    # Population weighting
    coords_rad = tuple(map(tuple, np.radians(grid_df[["Latitude", "Longitude"]].values)))
    nbrs = cached_query_radius(coords_rad, pop_radius_km)

    pop_local = []
    for idx in nbrs:
        if len(idx) > 0:
            pop_local.append(float(pop_df.iloc[idx]["pop_density"].mean()))
        else:
            pop_local.append(float(GLOBAL_POP_MEAN))
    grid_df["pop_density"] = pop_local

    grid_df["pop_weight"] = _normalize_robust(grid_df["pop_density"].to_numpy(dtype=float), 0.01, 0.99)

    # ============================================================
    # FINAL RISK (produce BOTH raw and population-adjusted surfaces)
    # ============================================================

    # Base (no population): model+history * species weight
    grid_df["risk_raw_value"] = (grid_df["risk_blend"] * grid_df["species_weight"]).fillna(0.0)

    # Population multiplier (always computed so you can show it in popups)
    grid_df["pop_multiplier"] = (1.0 + pop_strength * grid_df["pop_weight"]).fillna(1.0)

    # Population-adjusted raw value
    grid_df["risk_pop_value"] = (grid_df["risk_raw_value"] * grid_df["pop_multiplier"]).fillna(0.0)

    # Optional smoothing — apply smoothing to BOTH surfaces so toggle is apples-to-apples
    if bool(smooth_final) and len(grid_df) > 10:
        grid_df["risk_raw_value"] = smooth_grid_risk_knn(
            grid_df, col="risk_raw_value", k=int(smooth_k), sigma_km=float(smooth_sigma_km)
        ).astype(float)

        grid_df["risk_pop_value"] = smooth_grid_risk_knn(
            grid_df, col="risk_pop_value", k=int(smooth_k), sigma_km=float(smooth_sigma_km)
        ).astype(float)

    # Normalize BOTH for frontend display (0..1)
    grid_df["risk_raw"] = _normalize_robust(
        grid_df["risk_raw_value"].to_numpy(dtype=float), 0.01, 0.99
    ).astype(float)

    grid_df["risk_pop"] = _normalize_robust(
        grid_df["risk_pop_value"].to_numpy(dtype=float), 0.01, 0.99
    ).astype(float)

    # Backwards compatibility: keep risk_final as whichever mode is requested
    grid_df["risk_final"] = grid_df["risk_pop"] if bool(use_population) else grid_df["risk_raw"]

    # Return min/max for the selected surface
    return grid_df, float(grid_df["risk_final"].min()), float(grid_df["risk_final"].max())


# ============================================================
# SPRAY SITE SELECTION (greedy for <=10, QUBO otherwise)
# ============================================================

def select_spray_sites(grid_df: pd.DataFrame, n_sites: int, spray_radius_km: float) -> pd.DataFrame:
    """
    Pick n_sites centers that maximize captured risk (risk_final),
    with no sites within 2*spray_radius_km of each other.
    Greedy for n_sites <= 10; QUBO for larger.
    """
    n_sites = int(n_sites)
    spray_radius_km = float(spray_radius_km)

    lats = grid_df["Latitude"].values.astype(float)
    lons = grid_df["Longitude"].values.astype(float)
    risks = np.clip(grid_df["risk_final"].values.astype(float), 0.0, 1.0)

    def _dist_from(i):
        return haversine_distance_km(lats[i], lons[i], lats, lons)

    # Fast greedy
    if n_sites <= 10:
        remaining = grid_df.copy()
        chosen_idx = []
        for _ in range(n_sites):
            if len(remaining) == 0:
                break
            idx = int(remaining["risk_final"].idxmax())
            chosen_idx.append(idx)
            lat0, lon0 = remaining.loc[idx, ["Latitude", "Longitude"]].values
            d = haversine_distance_km(lat0, lon0, remaining["Latitude"].values, remaining["Longitude"].values)
            remaining = remaining.loc[d > (2.0 * spray_radius_km)].copy()
        return grid_df.loc[chosen_idx].reset_index(drop=True)

    # QUBO for bigger K
    M = len(grid_df)
    dist = np.zeros((M, M), dtype=float)
    for i in range(M):
        dist[i, :] = _dist_from(i)

    gain = np.array([risks[dist[i] <= spray_radius_km].sum() for i in range(M)], dtype=float)
    if gain.max() > 0:
        gain = gain / gain.max()

    A = (gain.sum() + 1e-6) * 10.0
    B = A

    bqm = BinaryQuadraticModel(vartype="BINARY")

    # cardinality term (approx)
    for i in range(M):
        bqm.add_variable(i, -gain[i] + A * (1 - 2 * n_sites))

    for i in range(M):
        for j in range(i + 1, M):
            bqm.add_interaction(i, j, 2 * A)
            if dist[i, j] <= 2.0 * spray_radius_km:
                bqm.add_interaction(i, j, B)

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=25)
    best = sampleset.first.sample

    chosen = [i for i, v in best.items() if v == 1]
    if len(chosen) > n_sites:
        chosen = sorted(chosen, key=lambda i: gain[i], reverse=True)[:n_sites]

    return grid_df.iloc[chosen].reset_index(drop=True)


def simulate_spray(grid_df: pd.DataFrame, sites_df: pd.DataFrame, spray_radius_km: float) -> pd.DataFrame:
    """
    Reduce risk near spray sites, with species-specific efficacy.
    IMPORTANT: We reduce BOTH raw-value surfaces so toggles remain consistent.
    We ALSO reduce expected total + infected mosquitoes near spray sites.
    """
    df2 = grid_df.copy()
    spray_radius_km = float(spray_radius_km)

    efficacy = {
        "CULEX PIPIENS": 0.52,
        "CULEX TARSALIS": 0.30,
        "CULEX RESTUANS": 0.40,
        "CULEX PIPIENS/RESTUANS": 0.45,
    }

    # Ensure value columns exist
    if "risk_raw_value" not in df2.columns:
        df2["risk_raw_value"] = df2.get("risk_raw", df2.get("risk_final", np.nan))
    if "risk_pop_value" not in df2.columns:
        df2["risk_pop_value"] = df2.get("risk_pop", df2.get("risk_final", np.nan))

    for _, site in sites_df.iterrows():
        lat0 = float(site["Latitude"])
        lon0 = float(site["Longitude"])
        d = haversine_distance_km(lat0, lon0, df2["Latitude"].values, df2["Longitude"].values)
        mask = d <= spray_radius_km

        for sp, eff in efficacy.items():
            match_species = df2["Species"].astype(str).str.upper() == sp.upper()
            affected = mask & match_species

            mult = (1.0 - float(eff))

            # Reduce BOTH underlying value surfaces
            df2.loc[affected, "risk_raw_value"] = (df2.loc[affected, "risk_raw_value"].astype(float) * mult)
            df2.loc[affected, "risk_pop_value"] = (df2.loc[affected, "risk_pop_value"].astype(float) * mult)

            # Reduce expected TOTAL + INFECTED mosquitoes (post-spray)
            if "pred_mosq_total" in df2.columns:
                df2.loc[affected, "pred_mosq_total"] = (df2.loc[affected, "pred_mosq_total"].astype(float) * mult)

            if "pred_mosq_infected" in df2.columns:
                df2.loc[affected, "pred_mosq_infected"] = (df2.loc[affected, "pred_mosq_infected"].astype(float) * mult)

    # Re-normalize surfaces after reductions so map stays 0..1
    df2["risk_raw"] = _normalize_robust(df2["risk_raw_value"].to_numpy(dtype=float), 0.01, 0.99).astype(float)
    df2["risk_pop"] = _normalize_robust(df2["risk_pop_value"].to_numpy(dtype=float), 0.01, 0.99).astype(float)

    # Recompute prevalence %
    if "pred_mosq_total" in df2.columns and "pred_mosq_infected" in df2.columns:
        denom = np.maximum(df2["pred_mosq_total"].astype(float), 1e-9)
        df2["pred_prevalence"] = (100.0 * df2["pred_mosq_infected"].astype(float) / denom).clip(lower=0.0)

    # risk_final will be selected later based on use_population
    df2["risk_final"] = df2.get("risk_final", np.nan)
    return df2


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    y, w = datetime.now().isocalendar()[0:2]
    return render_template("index.html", sel_year=y, sel_week=w)


@app.route("/grid")
def grid_data():
    p = request.args

    y = p.get("year", type=int)
    w = p.get("week", type=int)
    if y is None or w is None:
        return jsonify({"error": "year and week are required"}), 400

    pop_strength = p.get("pop_strength", 1.0, type=float)
    pop_radius_km = p.get("pop_radius_km", 1.0, type=float)
    trap = p.get("trap_effort_mean", np.nan, type=float)

    hist_strength = p.get("hist_strength", 0.45, type=float)
    hist_radius_km = p.get("hist_radius_km", 1.5, type=float)
    hist_lb1 = p.get("hist_lb1", 4, type=int)
    hist_lb2 = p.get("hist_lb2", 10, type=int)

    hist_kernel_sigma_km = p.get("hist_kernel_sigma_km", 0.9, type=float)
    hist_cross_years = p.get("hist_cross_years", "true").lower() != "false"
    hist_cross_year_window = p.get("hist_cross_year_window", 1, type=int)

    smooth_final = p.get("smooth_final", "true").lower() != "false"
    smooth_k = p.get("smooth_k", 60, type=int)
    smooth_sigma_km = p.get("smooth_sigma_km", 0.9, type=float)

    # This controls which surface is returned in the legacy "risk" field
    use_population = p.get("use_population", "true").lower() != "false"

    cw_req = {
        "temp": p.get("temp", np.nan, type=float),
        "humidity": p.get("humidity", np.nan, type=float),
        "rain": p.get("rain", np.nan, type=float),
        "wind_speed": p.get("wind_speed", np.nan, type=float),
    }

    # Compute weather once (also gives source) for meta/debug
    cw_used = get_weather_for_week(int(y), int(w), cw_req)

    # (kept as-is, though these are expensive because they call OWM)
    this_mon = datetime.fromisocalendar(int(y), int(w), 1)
    lw1 = fetch_weekly_avg(this_mon - timedelta(weeks=1))
    lw2 = fetch_weekly_avg(this_mon - timedelta(weeks=2))

    grid, _, _ = build_grid(
        int(y), int(w),
        pop_strength,
        pop_radius_km,
        cw_req,
        lw1,
        lw2,
        trap_effort_mean=trap,
        hist_strength=hist_strength,
        hist_radius_km=hist_radius_km,
        hist_lookback_1=hist_lb1,
        hist_lookback_2=hist_lb2,
        hist_kernel_sigma_km=hist_kernel_sigma_km,
        hist_cross_years=hist_cross_years,
        hist_cross_year_window=hist_cross_year_window,
        smooth_final=smooth_final,
        smooth_k=smooth_k,
        smooth_sigma_km=smooth_sigma_km,
        use_population=use_population,
    )

    if grid is None or len(grid) == 0:
        return jsonify({"grid": [], "min_risk": 0.0, "max_risk": 1.0}), 200

    # Choose legacy risk field for the heatmap
    risk_field = "risk_pop" if use_population else "risk_raw"
    grid["risk"] = grid[risk_field]

    # Min/max for selected view
    mn = float(np.nanmin(grid["risk"].to_numpy(dtype=float)))
    mx = float(np.nanmax(grid["risk"].to_numpy(dtype=float)))
    if not np.isfinite(mn): mn = 0.0
    if not np.isfinite(mx): mx = 1.0

    # Also return min/max for BOTH surfaces
    mn_raw = float(np.nanmin(grid["risk_raw"].to_numpy(dtype=float))) if "risk_raw" in grid.columns else 0.0
    mx_raw = float(np.nanmax(grid["risk_raw"].to_numpy(dtype=float))) if "risk_raw" in grid.columns else 1.0
    mn_pop = float(np.nanmin(grid["risk_pop"].to_numpy(dtype=float))) if "risk_pop" in grid.columns else 0.0
    mx_pop = float(np.nanmax(grid["risk_pop"].to_numpy(dtype=float))) if "risk_pop" in grid.columns else 1.0

    out_cols = [
        "Latitude", "Longitude", "Species",
        "risk",          # selected risk (legacy)
        "risk_raw",      # no population (0..1)
        "risk_pop",      # with population (0..1)
        "pop_density", "pop_weight", "pop_multiplier",

        # abundance outputs
        "pred_mosq_total",
        "pred_mosq_infected",
        "pred_prevalence",

        # helpful debugging
        "weather_source",
    ]

    out_cols = [c for c in out_cols if c in grid.columns]
    out = grid[out_cols].copy()
    out = out.where(pd.notnull(out), None)

    records = out.to_dict("records")

    meta = {
        "use_population": bool(use_population),
        "pop_strength": float(pop_strength),
        "pop_radius_km": float(pop_radius_km),
        "hist_strength": float(hist_strength),
        "hist_radius_km": float(hist_radius_km),
        "weather_used": {
            "temp": float(cw_used["temp"]),
            "humidity": float(cw_used["humidity"]),
            "rain": float(cw_used["rain"]),
            "wind_speed": float(cw_used["wind_speed"]),
            "source": str(cw_used.get("source", "unknown")),
            "units": {"temp": "Celsius", "wind_speed": "m/s", "rain": "mm (hourly summed then averaged into week)"},
        },
        "mosquito_estimate": {
            "pred_mosq_total": "Expected TOTAL mosquitoes (E[num_mosq]) from abundance GLM.",
            "pred_mosq_infected": "Expected INFECTED mosquitoes ≈ pred_mosq_total * risk_inf.",
            "pred_prevalence": "Infection prevalence % = 100 * pred_mosq_infected / pred_mosq_total.",
        },
        "definitions": {
            "risk_raw": "Risk surface without population weighting (model+history blended, species-weighted).",
            "risk_pop": "Risk surface with population multiplier applied: risk_raw_value * (1 + pop_strength * pop_weight).",
            "pop_weight": "0..1 scaled local population density across the grid.",
            "hist_strength": "Blend factor between model risk and historical positives KDE (0=model only, 1=history only).",
            "weather": "Weather inputs (temp, humidity, rain, wind) are included in the model features.",
        },
    }

    return jsonify(
        {
            "grid": records,
            "min_risk": mn, "max_risk": mx,
            "min_risk_raw": mn_raw, "max_risk_raw": mx_raw,
            "min_risk_pop": mn_pop, "max_risk_pop": mx_pop,
            "meta": meta,
        }
    )


@app.route("/spray", methods=["POST"])
def spray():
    d = request.get_json() or {}

    y = d.get("year")
    w = d.get("week")
    if y is None or w is None:
        return jsonify({"error": "year and week are required"}), 400

    try:
        n_sites = int(d.get("num_trucks", 5))
    except ValueError:
        return jsonify({"error": "num_trucks must be an integer"}), 400

    use_population = bool(d.get("use_population", True))

    pop_strength = float(d.get("pop_strength", 1.0))
    pop_radius_km = float(d.get("pop_radius_km", 1.0))
    trap_mean = float(d.get("trap_effort_mean", np.nan))

    hist_strength = float(d.get("hist_strength", 0.45))
    hist_radius_km = float(d.get("hist_radius_km", 1.5))
    hist_lb1 = int(d.get("hist_lb1", 4))
    hist_lb2 = int(d.get("hist_lb2", 10))

    hist_kernel_sigma_km = float(d.get("hist_kernel_sigma_km", 0.9))
    hist_cross_years = bool(d.get("hist_cross_years", True))
    hist_cross_year_window = int(d.get("hist_cross_year_window", 1))

    smooth_final = bool(d.get("smooth_final", True))
    smooth_k = int(d.get("smooth_k", 60))
    smooth_sigma_km = float(d.get("smooth_sigma_km", 0.9))

    cw_req = {
        "temp": float(d.get("temp", np.nan)),
        "humidity": float(d.get("humidity", np.nan)),
        "rain": float(d.get("rain", np.nan)),
        "wind_speed": float(d.get("wind_speed", np.nan)),
    }

    mon_date = datetime.fromisocalendar(int(y), int(w), 1)
    lw1 = fetch_weekly_avg(mon_date - timedelta(weeks=1))
    lw2 = fetch_weekly_avg(mon_date - timedelta(weeks=2))

    grid_cur, mn, mx = build_grid(
        int(y), int(w),
        pop_strength,
        pop_radius_km,
        cw_req,
        lw1,
        lw2,
        trap_effort_mean=trap_mean,
        hist_strength=hist_strength,
        hist_radius_km=hist_radius_km,
        hist_lookback_1=hist_lb1,
        hist_lookback_2=hist_lb2,
        hist_kernel_sigma_km=hist_kernel_sigma_km,
        hist_cross_years=hist_cross_years,
        hist_cross_year_window=hist_cross_year_window,
        smooth_final=smooth_final,
        smooth_k=smooth_k,
        smooth_sigma_km=smooth_sigma_km,
        use_population=use_population,
    )

    spray_radius = float(d.get("spray_radius_km", 1.0))

    # Clip to polygon if provided
    if d.get("polygon"):
        poly = shape(d["polygon"])
        mask = grid_cur.apply(lambda r: poly.covers(Point(r.Longitude, r.Latitude)), axis=1)
        grid_cur = grid_cur.loc[mask].reset_index(drop=True)

    # Ensure risk_final is aligned with toggle
    grid_cur["risk_final"] = grid_cur["risk_pop"] if use_population else grid_cur["risk_raw"]

    sites_df = select_spray_sites(grid_cur, n_sites, spray_radius)

    # Apply spray reductions to BOTH surfaces + re-normalize
    post_grid_df = simulate_spray(grid_cur, sites_df, spray_radius)

    # Enforce selected legacy `risk` field
    def _apply_selected_risk(df):
        df["risk_final"] = df["risk_pop"] if use_population else df["risk_raw"]
        df["risk"] = df["risk_final"]
        return df

    grid_cur = _apply_selected_risk(grid_cur)
    post_grid_df = _apply_selected_risk(post_grid_df)
    sites_df = _apply_selected_risk(sites_df)

    def _prep(df):
        keep = [
            "Latitude", "Longitude", "Species",
            "risk", "risk_raw", "risk_pop",
            "pop_density", "pop_weight", "pop_multiplier",
            "pred_mosq_total", "pred_mosq_infected", "pred_prevalence",
        ]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()
        out = out.where(pd.notnull(out), None)
        return out.to_dict("records")

    return jsonify(
        {
            "use_population": use_population,
            "sites": _prep(sites_df),
            "grid_pre_spray": _prep(grid_cur),
            "grid_post_spray": _prep(post_grid_df),
            "min_risk_pre": float(mn),
            "max_risk_pre": float(mx),
        }
    )


@app.route("/simulate_spray", methods=["POST"])
def simulate_spray_endpoint():
    d = request.get_json() or {}

    y = d.get("year")
    w = d.get("week")
    if y is None or w is None:
        return jsonify({"error": "year and week are required"}), 400

    use_population = bool(d.get("use_population", True))

    pop_strength = float(d.get("pop_strength", 1.0))
    pop_radius_km = float(d.get("pop_radius_km", 1.0))
    trap_mean = float(d.get("trap_effort_mean", np.nan))

    hist_strength = float(d.get("hist_strength", 0.45))
    hist_radius_km = float(d.get("hist_radius_km", 1.5))
    hist_lb1 = int(d.get("hist_lb1", 4))
    hist_lb2 = int(d.get("hist_lb2", 10))

    hist_kernel_sigma_km = float(d.get("hist_kernel_sigma_km", 0.9))
    hist_cross_years = bool(d.get("hist_cross_years", True))
    hist_cross_year_window = int(d.get("hist_cross_year_window", 1))

    smooth_final = bool(d.get("smooth_final", True))
    smooth_k = int(d.get("smooth_k", 60))
    smooth_sigma_km = float(d.get("smooth_sigma_km", 0.9))

    cw_req = {
        "temp": float(d.get("temp", np.nan)),
        "humidity": float(d.get("humidity", np.nan)),
        "rain": float(d.get("rain", np.nan)),
        "wind_speed": float(d.get("wind_speed", np.nan)),
    }

    mon_date = datetime.fromisocalendar(int(y), int(w), 1)
    lw1 = fetch_weekly_avg(mon_date - timedelta(weeks=1))
    lw2 = fetch_weekly_avg(mon_date - timedelta(weeks=2))

    grid_cur, _, _ = build_grid(
        int(y), int(w),
        pop_strength,
        pop_radius_km,
        cw_req,
        lw1,
        lw2,
        trap_effort_mean=trap_mean,
        hist_strength=hist_strength,
        hist_radius_km=hist_radius_km,
        hist_lookback_1=hist_lb1,
        hist_lookback_2=hist_lb2,
        hist_kernel_sigma_km=hist_kernel_sigma_km,
        hist_cross_years=hist_cross_years,
        hist_cross_year_window=hist_cross_year_window,
        smooth_final=smooth_final,
        smooth_k=smooth_k,
        smooth_sigma_km=smooth_sigma_km,
        use_population=use_population,
    )

    if grid_cur is None or len(grid_cur) == 0:
        return jsonify(
            {
                "use_population": use_population,
                "sites": [],
                "grid_pre_spray": [],
                "grid_post_spray": [],
                "min_risk_pre": 0.0,
                "max_risk_pre": 1.0,
            }
        ), 200

    # optional polygon clip
    if d.get("polygon"):
        poly = shape(d["polygon"])
        mask = grid_cur.apply(lambda r: poly.covers(Point(r.Longitude, r.Latitude)), axis=1)
        grid_cur = grid_cur.loc[mask].reset_index(drop=True)

    n_sites = int(d.get("n_sites", d.get("num_trucks", 5)))
    spray_radius = float(d.get("spray_radius_km", 1.0))

    # Align risk_final with toggle
    grid_cur["risk_final"] = grid_cur["risk_pop"] if use_population else grid_cur["risk_raw"]

    sites_df = select_spray_sites(grid_cur, n_sites, spray_radius)
    post_grid_df = simulate_spray(grid_cur, sites_df, spray_radius)

    def _apply_selected_risk(df: pd.DataFrame) -> pd.DataFrame:
        df["risk_final"] = df["risk_pop"] if use_population else df["risk_raw"]
        df["risk"] = df["risk_final"]
        return df

    grid_cur = _apply_selected_risk(grid_cur)
    post_grid_df = _apply_selected_risk(post_grid_df)
    sites_df = _apply_selected_risk(sites_df)

    # Min/max for selected view
    mn_sel = float(np.nanmin(grid_cur["risk"].to_numpy(dtype=float)))
    mx_sel = float(np.nanmax(grid_cur["risk"].to_numpy(dtype=float)))
    if not np.isfinite(mn_sel): mn_sel = 0.0
    if not np.isfinite(mx_sel): mx_sel = 1.0

    def _prep(df: pd.DataFrame) -> list:
        keep = [
            "Latitude", "Longitude", "Species",
            "risk", "risk_raw", "risk_pop",
            "pop_density", "pop_weight", "pop_multiplier",
            "pred_mosq_total", "pred_mosq_infected", "pred_prevalence",
        ]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()
        out = out.where(pd.notnull(out), None)
        return out.to_dict("records")

    return jsonify(
        {
            "use_population": use_population,
            "sites": _prep(sites_df),
            "grid_pre_spray": _prep(grid_cur),
            "grid_post_spray": _prep(post_grid_df),
            "min_risk_pre": float(mn_sel),
            "max_risk_pre": float(mx_sel),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
