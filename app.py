# app.py
import os
import time
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List

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

# SciPy is used for sparse movement operator. If unavailable, we fall back to dense (slower).
try:
    from scipy import sparse
    _HAVE_SCIPY = True
except Exception:
    sparse = None
    _HAVE_SCIPY = False


# ============================================================
# CONFIG
# ============================================================

ORIG_WNV_CSV = "west_nile_virus_data.csv"
WEATHER_WNV_CSV = "west_nile_virus_data_with_weather.csv"  # optional fallback if exists

OWM_KEY = os.environ.get("OWM_KEY", "79e121d53c70ecf6ebd8a573f98d702e")
LAT, LON = 41.8781, -87.6298  # Chicago coords

POP_DENSITY_CSV = "pop_density.csv"

# If you want to shrink compute, raise step. If you want fewer grid artifacts, lower step.
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

# Mechanistic switch:
#   USE_MECH=1 -> mechanistic abundance + transmission + spatial movement (daily Euler)
#   USE_MECH=0 -> ML infection + GLM abundance + history blend
USE_MECH = os.environ.get("USE_MECH", "1").strip() == "1"

# Mechanistic scaling knobs
MECH_MOSQ_SCALE = float(os.environ.get("MECH_MOSQ_SCALE", "1.0"))
MECH_BIRD_PER_HUMAN = float(os.environ.get("MECH_BIRD_PER_HUMAN", "0.02"))
MECH_BIRD_BASE = float(os.environ.get("MECH_BIRD_BASE", "200.0"))
MECH_NH_SCALE = float(os.environ.get("MECH_NH_SCALE", "0.001"))
MECH_NH_BASE = float(os.environ.get("MECH_NH_BASE", "1.0"))

# Optional ULV pulse
MECH_ULV_APPLY_DOY = os.environ.get("MECH_ULV_APPLY_DOY", "").strip()
MECH_ULV_APPLY_DOY = int(MECH_ULV_APPLY_DOY) if MECH_ULV_APPLY_DOY.isdigit() else None
MECH_ULV_DURATION_DAYS = int(os.environ.get("MECH_ULV_DURATION_DAYS", "7"))
MECH_ZETA0 = float(os.environ.get("MECH_ZETA0", "0.5"))

# Optional bird introduction pulse
MECH_INTRO_A0 = float(os.environ.get("MECH_INTRO_A0", "0.0"))
MECH_INTRO_Am = float(os.environ.get("MECH_INTRO_Am", "200.0"))
MECH_INTRO_Aw = float(os.environ.get("MECH_INTRO_Aw", "30.0"))

CALIB_MIN_SCALE = float(os.environ.get("CALIB_MIN_SCALE", "0.10"))
CALIB_MAX_SCALE = float(os.environ.get("CALIB_MAX_SCALE", "25.0"))
CALIB_USE_GLM_BRIDGE = os.environ.get("CALIB_USE_GLM_BRIDGE", "1").strip() == "1"
CALIB_EMPIRICAL_WINDOW = int(os.environ.get("CALIB_EMPIRICAL_WINDOW", "2"))
CALIB_LOCAL_RADIUS_KM = float(os.environ.get("CALIB_LOCAL_RADIUS_KM", "2.0"))
CALIB_LOCAL_WEEK_WINDOW = int(os.environ.get("CALIB_LOCAL_WEEK_WINDOW", "2"))
CALIB_TRAP_BIAS_RADIUS_KM = float(os.environ.get("CALIB_TRAP_BIAS_RADIUS_KM", "2.5"))


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
    Robust 0..1 scaling so one extreme point doesn't flatten the whole surface.
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


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D arrays."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(np.sum(ex))
    if s <= 0:
        return np.ones_like(x) / max(1, x.size)
    return ex / s


# ============================================================
# LOAD MODELS (infection + species) + ABUNDANCE (Poisson/NB)
# ============================================================

model_inf = load("model_inf_ensemble.joblib")
model_sp = load("model_sp_calibrated.joblib")

FEATURES_INF = list(getattr(model_inf, "feature_names_in_", []))
FEATURES_SP = list(getattr(model_sp, "feature_names_in_", []))

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
    if MOSQ_ABUNDANCE_KIND == "nb" and model_ab_nb is not None:
        return model_ab_nb
    if model_ab_poisson is not None:
        return model_ab_poisson
    return model_ab_nb


model_ab = _get_abundance_model()

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
}

SPECIES_MECH = {
    "CULEX PIPIENS": {
        "dev_mult": 1.05,
        "mort_mult": 0.95,
        "bite_mult": 1.00,
        "move_mult": 1.00,
        "calib_mult": 1.00,
    },
    "CULEX RESTUANS": {
        "dev_mult": 0.92,
        "mort_mult": 1.08,
        "bite_mult": 0.88,
        "move_mult": 0.90,
        "calib_mult": 1.10,
    },
    "CULEX PIPIENS/RESTUANS": {
        "dev_mult": 1.00,
        "mort_mult": 1.00,
        "bite_mult": 0.95,
        "move_mult": 0.95,
        "calib_mult": 1.00,
    },
}


@memory.cache
def cached_query_radius(coords_radians_tuple, radius_km):
    arr = np.array(coords_radians_tuple, dtype=float)
    return pop_tree.query_radius(arr, r=float(radius_km) / 6371.0)


# ============================================================
# EMPIRICAL CALIBRATION TABLES
# ============================================================

def _clean_species_name(x: Any) -> str:
    s = str(x or "").upper().strip()
    if s in SPECIES_MECH:
        return s
    if "RESTUANS" in s and "PIPIENS" in s:
        return "CULEX PIPIENS/RESTUANS"
    if "RESTUANS" in s:
        return "CULEX RESTUANS"
    if "PIPIENS" in s:
        return "CULEX PIPIENS"
    return "CULEX PIPIENS/RESTUANS"


def _clean_trap_type(x: Any) -> str:
    s = str(x or "").upper().strip()
    if not s or s == "NAN":
        return "UNKNOWN"
    return s


wnv_raw["SpeciesClean"] = wnv_raw["Species"].apply(_clean_species_name)
wnv_raw["TrapTypeClean"] = wnv_raw.get("TRAP_TYPE", pd.Series(index=wnv_raw.index, dtype=object)).apply(_clean_trap_type)
wnv_raw["TrapIDClean"] = wnv_raw.get("TRAP", pd.Series(index=wnv_raw.index, dtype=object)).astype(str).str.upper().str.strip()

_emp_df = wnv_raw.copy()
_emp_df = _emp_df[np.isfinite(_emp_df["num_mosq"])].copy()
_emp_df = _emp_df[_emp_df["num_mosq"] >= 0].copy()

if len(_emp_df):
    EMP_GLOBAL_MEDIAN = float(np.nanmedian(_emp_df["num_mosq"].values))
else:
    EMP_GLOBAL_MEDIAN = 1.0

if not np.isfinite(EMP_GLOBAL_MEDIAN) or EMP_GLOBAL_MEDIAN <= 0:
    EMP_GLOBAL_MEDIAN = 1.0

EMP_WEEK_SPECIES = (
    _emp_df.groupby(["Week", "SpeciesClean"])["num_mosq"]
    .median()
    .reset_index()
)

EMP_WEEK_SPECIES_TRAPTYPE = (
    _emp_df.groupby(["Week", "SpeciesClean", "TrapTypeClean"])["num_mosq"]
    .median()
    .reset_index()
)

EMP_TRAP_BASELINE = (
    _emp_df.groupby(["TrapIDClean"])["num_mosq"]
    .median()
    .reset_index()
)

EMP_WEEK_SPECIES_MAP = {
    (int(r["Week"]), str(r["SpeciesClean"])): float(r["num_mosq"])
    for _, r in EMP_WEEK_SPECIES.iterrows()
}

EMP_WEEK_SPECIES_TRAPTYPE_MAP = {
    (int(r["Week"]), str(r["SpeciesClean"]), str(r["TrapTypeClean"])): float(r["num_mosq"])
    for _, r in EMP_WEEK_SPECIES_TRAPTYPE.iterrows()
}

EMP_TRAP_BASELINE_MAP = {
    str(r["TrapIDClean"]): float(r["num_mosq"])
    for _, r in EMP_TRAP_BASELINE.iterrows()
}

_emp_loc = _emp_df.dropna(subset=["Latitude", "Longitude"]).copy()
_emp_loc = _emp_loc[
    ["Year", "Week", "Latitude", "Longitude", "num_mosq", "SpeciesClean", "TrapTypeClean", "TrapIDClean"]
].copy()

if len(_emp_loc):
    EMP_LOC_TREE = BallTree(np.radians(_emp_loc[["Latitude", "Longitude"]].to_numpy(dtype=float)), metric="haversine")
else:
    EMP_LOC_TREE = None


def _empirical_species_week_factor(week: int, species: str, window: int = CALIB_EMPIRICAL_WINDOW) -> float:
    species = _clean_species_name(species)
    vals = []
    for ww in range(int(week) - int(window), int(week) + int(window) + 1):
        if ww < 1 or ww > 53:
            continue
        v = EMP_WEEK_SPECIES_MAP.get((ww, species))
        if v is not None and np.isfinite(v):
            vals.append(float(v))

    if not vals:
        return 1.0

    med = float(np.median(vals))
    factor = med / max(EMP_GLOBAL_MEDIAN, 1e-9)
    return float(np.clip(factor, 0.5, 2.5))


def _empirical_species_week_traptype_factor(
    week: int,
    species: str,
    trap_type: str,
    window: int = CALIB_EMPIRICAL_WINDOW,
) -> float:
    species = _clean_species_name(species)
    trap_type = _clean_trap_type(trap_type)

    vals = []
    for ww in range(int(week) - int(window), int(week) + int(window) + 1):
        if ww < 1 or ww > 53:
            continue
        v = EMP_WEEK_SPECIES_TRAPTYPE_MAP.get((ww, species, trap_type))
        if v is not None and np.isfinite(v):
            vals.append(float(v))

    if not vals:
        return _empirical_species_week_factor(week, species, window=window)

    med = float(np.median(vals))
    factor = med / max(EMP_GLOBAL_MEDIAN, 1e-9)
    return float(np.clip(factor, 0.4, 3.0))


def _species_param_arrays(species_series: pd.Series) -> Dict[str, np.ndarray]:
    sp = species_series.astype(str).str.upper().fillna("CULEX PIPIENS/RESTUANS")
    dev = sp.map(lambda s: SPECIES_MECH.get(_clean_species_name(s), SPECIES_MECH["CULEX PIPIENS/RESTUANS"])["dev_mult"]).astype(float).to_numpy()
    mort = sp.map(lambda s: SPECIES_MECH.get(_clean_species_name(s), SPECIES_MECH["CULEX PIPIENS/RESTUANS"])["mort_mult"]).astype(float).to_numpy()
    bite = sp.map(lambda s: SPECIES_MECH.get(_clean_species_name(s), SPECIES_MECH["CULEX PIPIENS/RESTUANS"])["bite_mult"]).astype(float).to_numpy()
    move = sp.map(lambda s: SPECIES_MECH.get(_clean_species_name(s), SPECIES_MECH["CULEX PIPIENS/RESTUANS"])["move_mult"]).astype(float).to_numpy()
    calib = sp.map(lambda s: SPECIES_MECH.get(_clean_species_name(s), SPECIES_MECH["CULEX PIPIENS/RESTUANS"])["calib_mult"]).astype(float).to_numpy()
    return {
        "dev_mult": dev,
        "mort_mult": mort,
        "bite_mult": bite,
        "move_mult": move,
        "calib_mult": calib,
    }


def _build_glm_target_on_grid(grid_df: pd.DataFrame) -> Optional[np.ndarray]:
    if model_ab is None:
        return None

    tmp = grid_df.copy()
    tmp["sin_week"] = np.sin(2 * np.pi * tmp["Week"].astype(float) / 52.0)
    tmp["cos_week"] = np.cos(2 * np.pi * tmp["Week"].astype(float) / 52.0)
    tmp["SPECIES"] = tmp["Species"].astype(str).fillna("UNKNOWN").str.upper().str.strip()

    default_trap_type = (
        str(MOSQ_META.get("default_trap_type_mode", "GRAVID"))
        if isinstance(MOSQ_META, dict) else "GRAVID"
    )
    if "TRAP_TYPE" not in tmp.columns:
        tmp["TRAP_TYPE"] = default_trap_type
    else:
        tmp["TRAP_TYPE"] = tmp["TRAP_TYPE"].fillna(default_trap_type)

    species_levels = set([s.upper().strip() for s in MOSQ_META.get("species_levels", []) if isinstance(s, str)])
    trap_levels = set([t.upper().strip() for t in MOSQ_META.get("trap_type_levels", []) if isinstance(t, str)])

    if species_levels:
        safe_species = next(iter(species_levels))
        tmp.loc[~tmp["SPECIES"].isin(species_levels), "SPECIES"] = safe_species

    if trap_levels:
        safe_trap = default_trap_type.upper().strip()
        if safe_trap not in trap_levels:
            safe_trap = next(iter(trap_levels))
        tmp.loc[~tmp["TRAP_TYPE"].astype(str).str.upper().isin(trap_levels), "TRAP_TYPE"] = safe_trap

    for c in ["Latitude", "Longitude", "Year", "Week", "temp", "humidity", "rain", "wind_speed", "sin_week", "cos_week"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)

    try:
        design_info = getattr(model_ab.model.data, "design_info", None)
        if design_info is not None:
            from patsy import build_design_matrices
            exog = build_design_matrices([design_info], tmp, return_type="dataframe")[0]
            pred = model_ab.predict(exog, transform=False)
        else:
            pred = model_ab.predict(tmp)

        pred = np.asarray(pred, dtype=float)
        pred = np.where(np.isfinite(pred), pred, 0.0)
        pred = np.clip(pred, 0.0, None)
        return pred
    except Exception as e:
        print("[WARN] GLM bridge calibration failed:", repr(e))
        return None


def _local_empirical_target_surface(
    grid_df: pd.DataFrame,
    year: int,
    week: int,
    species_col: str = "Species",
    radius_km: float = CALIB_LOCAL_RADIUS_KM,
    week_window: int = CALIB_LOCAL_WEEK_WINDOW,
) -> np.ndarray:
    if EMP_LOC_TREE is None or len(_emp_loc) == 0 or len(grid_df) == 0:
        return np.full(len(grid_df), EMP_GLOBAL_MEDIAN, dtype=float)

    use = _emp_loc[
        (_emp_loc["Week"] >= max(1, int(week) - int(week_window))) &
        (_emp_loc["Week"] <= min(53, int(week) + int(week_window)))
    ].copy()

    if len(use) == 0:
        return np.full(len(grid_df), EMP_GLOBAL_MEDIAN, dtype=float)

    use_tree = BallTree(np.radians(use[["Latitude", "Longitude"]].to_numpy(dtype=float)), metric="haversine")
    gcoords = np.radians(grid_df[["Latitude", "Longitude"]].to_numpy(dtype=float))

    idxs, dists = use_tree.query_radius(
        gcoords,
        r=float(radius_km) / EARTH_RADIUS_KM,
        return_distance=True,
        sort_results=True
    )

    out = np.full(len(grid_df), EMP_GLOBAL_MEDIAN, dtype=float)

    for i, (ii, dd) in enumerate(zip(idxs, dists)):
        if len(ii) == 0:
            continue

        neigh = use.iloc[ii].copy()
        dkm = np.asarray(dd, dtype=float) * EARTH_RADIUS_KM
        w = 1.0 / np.maximum(dkm, 0.25)

        target_species = _clean_species_name(grid_df.iloc[i][species_col])
        species_match = (neigh["SpeciesClean"].astype(str) == target_species).to_numpy(dtype=bool)

        if species_match.any():
            neigh = neigh.loc[species_match].copy()
            w = w[species_match]

        if len(neigh) == 0:
            continue

        vals = neigh["num_mosq"].to_numpy(dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.nan)
        good = np.isfinite(vals)
        if good.any():
            out[i] = float(np.average(vals[good], weights=w[good]))

    return np.clip(out, 0.0, None)


def _local_trap_bias_surface(
    grid_df: pd.DataFrame,
    week: int,
    radius_km: float = CALIB_TRAP_BIAS_RADIUS_KM,
    week_window: int = CALIB_LOCAL_WEEK_WINDOW,
) -> np.ndarray:
    if EMP_LOC_TREE is None or len(_emp_loc) == 0 or len(grid_df) == 0:
        return np.ones(len(grid_df), dtype=float)

    use = _emp_loc[
        (_emp_loc["Week"] >= max(1, int(week) - int(week_window))) &
        (_emp_loc["Week"] <= min(53, int(week) + int(week_window)))
    ].copy()

    if len(use) == 0:
        return np.ones(len(grid_df), dtype=float)

    use["trap_bias"] = use["TrapIDClean"].map(
        lambda t: float(EMP_TRAP_BASELINE_MAP.get(str(t), EMP_GLOBAL_MEDIAN)) / max(EMP_GLOBAL_MEDIAN, 1e-9)
    )

    use_tree = BallTree(np.radians(use[["Latitude", "Longitude"]].to_numpy(dtype=float)), metric="haversine")
    gcoords = np.radians(grid_df[["Latitude", "Longitude"]].to_numpy(dtype=float))

    idxs, dists = use_tree.query_radius(
        gcoords,
        r=float(radius_km) / EARTH_RADIUS_KM,
        return_distance=True,
        sort_results=True
    )

    out = np.ones(len(grid_df), dtype=float)

    for i, (ii, dd) in enumerate(zip(idxs, dists)):
        if len(ii) == 0:
            continue

        neigh = use.iloc[ii].copy()
        dkm = np.asarray(dd, dtype=float) * EARTH_RADIUS_KM
        w = 1.0 / np.maximum(dkm, 0.25)

        vals = neigh["trap_bias"].to_numpy(dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.nan)
        good = np.isfinite(vals)
        if good.any():
            out[i] = float(np.average(vals[good], weights=w[good]))

    return np.clip(out, 0.6, 1.8)


def _calibrate_mech_outputs(
    grid_df: pd.DataFrame,
    adult_raw: np.ndarray,
    infected_raw: np.ndarray,
    week: int,
    year: int,
    trap_type: str = "GRAVID",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    adult_raw = np.asarray(adult_raw, dtype=float)
    infected_raw = np.asarray(infected_raw, dtype=float)

    sp_arr = grid_df["Species"].astype(str).fillna("CULEX PIPIENS/RESTUANS").tolist()

    species_week_factor = np.array(
        [_empirical_species_week_factor(int(week), s) for s in sp_arr],
        dtype=float
    )

    species_week_traptype_factor = np.array(
        [_empirical_species_week_traptype_factor(int(week), s, trap_type) for s in sp_arr],
        dtype=float
    )

    species_mech = _species_param_arrays(grid_df["Species"])
    calib_mult = species_mech["calib_mult"]

    base_scale = np.ones(len(grid_df), dtype=float)

    if CALIB_USE_GLM_BRIDGE:
        glm_target = _build_glm_target_on_grid(grid_df)
        if glm_target is not None:
            ratio = glm_target / np.maximum(adult_raw, 1e-6)
            ratio = np.where(np.isfinite(ratio), ratio, np.nan)
            med_ratio = np.nanmedian(ratio)
            if np.isfinite(med_ratio):
                bridge = float(np.clip(med_ratio, CALIB_MIN_SCALE, CALIB_MAX_SCALE))
                base_scale *= bridge

    local_target = _local_empirical_target_surface(
        grid_df=grid_df,
        year=year,
        week=week,
        species_col="Species",
        radius_km=CALIB_LOCAL_RADIUS_KM,
        week_window=CALIB_LOCAL_WEEK_WINDOW,
    )

    local_ratio = local_target / np.maximum(adult_raw, 1e-6)
    local_ratio = np.where(np.isfinite(local_ratio), local_ratio, 1.0)
    local_ratio = np.clip(local_ratio, 0.25, 8.0)

    trap_bias_surface = _local_trap_bias_surface(
        grid_df=grid_df,
        week=week,
        radius_km=CALIB_TRAP_BIAS_RADIUS_KM,
        week_window=CALIB_LOCAL_WEEK_WINDOW,
    )

    total_scale = np.clip(
        base_scale *
        species_week_factor *
        species_week_traptype_factor *
        calib_mult *
        local_ratio *
        trap_bias_surface,
        CALIB_MIN_SCALE,
        CALIB_MAX_SCALE,
    )

    adult_cal = np.clip(adult_raw * total_scale, 0.0, None)
    infected_cal = np.clip(infected_raw * total_scale, 0.0, None)
    prevalence = 100.0 * infected_cal / np.maximum(adult_cal, 1e-9)

    return adult_cal, infected_cal, prevalence


# ============================================================
# CONCAVE HULL (from positives)
# ============================================================

pos_pts = wnv_raw.loc[wnv_raw["is_positive"] == 1, ["Longitude", "Latitude"]].dropna().values
if len(pos_pts) >= 10:
    concave_hull = alphashape(pos_pts, alpha=0.1)
else:
    concave_hull = None


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
# WEATHER FETCH (LIVE) + FALLBACK TO df_ww
# ============================================================

def fetch_weekly_avg(dt: datetime):
    """
    dt should be a Monday. Pull 7 days of hourly via OWM timemachine,
    average it; if that fails, fall back to df_ww if available.
    Returns metric units.
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

        time.sleep(1)

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
    """
    cw_from_request = cw_from_request or {}

    cw = {
        "temp": _safe_float(cw_from_request.get("temp")),
        "humidity": _safe_float(cw_from_request.get("humidity")),
        "rain": _safe_float(cw_from_request.get("rain")),
        "wind_speed": _safe_float(cw_from_request.get("wind_speed")),
    }

    source = "request"

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

    if any(np.isnan(cw[k]) for k in cw):
        mon = datetime.fromisocalendar(int(year), int(week), 1)
        cw2 = fetch_weekly_avg(mon)
        for k in cw:
            if np.isnan(cw[k]):
                cw[k] = cw2[k]
        source = "owm"

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
# PAPER-DRIVEN WEATHER / DIAPAUSE HELPERS
# ============================================================

def _day_of_year_from_iso(year: int, week: int) -> int:
    d = datetime.fromisocalendar(int(year), int(week), 1)
    return int(d.timetuple().tm_yday)


def _daylength_hours(lat_deg: float, doy: int) -> float:
    """
    Approx daylength from latitude + day-of-year.
    """
    lat = math.radians(float(lat_deg))
    decl = 0.409 * math.sin(2.0 * math.pi * (float(doy) - 81.0) / 368.0)
    cos_omega = -math.tan(lat) * math.tan(decl)
    cos_omega = max(-1.0, min(1.0, cos_omega))
    omega = math.acos(cos_omega)
    return float((24.0 / math.pi) * omega)


def _alphaM_from_daylength(D_hours: float) -> float:
    """
    Fraction of non-diapausing mosquitoes.
    """
    D = float(D_hours)
    return float(np.clip(1.0 / (1.0 + math.exp(-(D - 12.5) / 0.5)), 0.0, 1.0))


def _eta_biting_rate(T_c: float) -> float:
    """
    Temperature-dependent biting rate proxy (1/day).
    """
    T = float(T_c)
    if T < 10.0:
        return 0.01
    return float(0.05 + 0.35 * (1.0 / (1.0 + math.exp(-(T - 20.0) / 3.0))))


def _gammaM_EIP_rate(T_c: float, base: float) -> float:
    """
    Temperature-dependent mosquito incubation rate gamma_M(T) (1/day).
    """
    T = float(T_c)
    if T < 12.0:
        return 1.0 / 20.0
    if T > 30.0:
        return 1.0 / 6.0
    return float((1.0 / 20.0) + (T - 12.0) * ((1.0 / 6.0) - (1.0 / 20.0)) / (30.0 - 12.0))


def _gamma_L_from_temp(T: float) -> float:
    """
    Egg -> larva maturity.
    """
    T = float(T)
    val = 0.16 * (
        math.exp(0.105 * (T - 10.0)) -
        math.exp(0.105 * (35.0 - 10.0) - (35.0 - T) / 5.007)
    )
    return float(max(0.0, val))


def _gamma_A_from_temp(T: float) -> float:
    """
    Pupa -> emerging adult maturity.
    """
    T = float(T)
    val = 0.021 * (
        math.exp(0.162 * (T - 10.0)) -
        math.exp(0.162 * (35.0 - 10.0) - (35.0 - T) / 5.007)
    )
    return float(max(0.0, val))


def _beta_P_from_temp(T: float, mu_P: float = 0.0) -> float:
    """
    Pupal mortality.
    """
    T = float(T)
    return float(math.exp(-T / 2.0) + float(mu_P))


def _beta_L_from_temp(T: float) -> float:
    """
    Larval mortality.
    """
    T = float(T)
    val = 0.0025 * T * T - 0.094 * T + 1.0257
    return float(max(0.0, val))


def _gamma_A0_from_temp(T: float) -> float:
    """
    Egg-laying rate.
    """
    T = float(T)
    val = -15.837 + 1.289 * T - 0.0163 * (T ** 2)
    return float(max(0.0, val))


def _weekly_rain_to_daily_mm(rain_value: float) -> float:
    """
    Convert weekly-average rain proxy to rough daily mm.
    """
    r = max(0.0, float(rain_value))
    return float(r * 24.0)


def _rolling_precip_norm(mm_hist: List[float], window_days: int = 14) -> float:
    """
    P_norm(t): 2-week accumulated rainfall, normalized to [0,1].
    """
    if not mm_hist:
        return 0.0
    s = float(np.sum(mm_hist[-window_days:]))
    return float(np.clip(s / 140.0, 0.0, 1.0))


def _flush_mortality_from_rain(mm_hist: List[float]) -> float:
    """
    Additional larval mortality beta_W due to flushing.
    Rule: when total rain over 7 days exceeds 2.5 cm.
    """
    if not mm_hist:
        return 0.0
    mm7 = float(np.sum(mm_hist[-7:]))
    cm7 = mm7 / 10.0
    if cm7 >= 2.5:
        return 0.20
    return 0.0


# ============================================================
# MECHANISTIC ABUNDANCE + WNV TRANSMISSION (INLINE)
# ============================================================

@dataclass
class MechParams:
    alpha_A: float = 1.0
    gamma_A0: float = 0.08
    gamma_L: float = 0.20
    gamma_P: float = 0.14
    gamma_A: float = 0.12
    sigma_A: float = 1.0
    gamma_em: float = 0.2
    K_P: float = 2000.0

    beta_E: float = 0.05
    beta_L: float = 0.06
    beta_P: float = 0.04
    beta_A: float = 0.03

    beta_1: float = 0.00002
    K_L: float = 5000.0
    beta_W: float = 0.00

    psi: float = 1.0

    gamma_B: float = 0.10
    gamma_En: float = 0.10
    gamma_El: float = 0.10
    mu_B: float = 0.00

    bB: float = 0.00342
    mB: float = 0.0012
    gammaBird: float = 0.182
    deltaB: float = 0.26

    beta1: float = 0.25
    beta2: float = 0.25
    beta3: float = 0.10

    alphaF: float = 10.0
    phiB: float = 15.0
    phiH: float = 0.03

    gammaM_base: float = 1.0 / 10.0

    zeta0: float = MECH_ZETA0
    ulv_apply_doy: Optional[int] = MECH_ULV_APPLY_DOY
    ulv_duration_days: int = MECH_ULV_DURATION_DAYS

    intro_A0: float = MECH_INTRO_A0
    intro_Am: float = MECH_INTRO_Am
    intro_Aw: float = MECH_INTRO_Aw

    d_max_km: float = float(os.environ.get("MECH_DMAX_KM", "3.0"))
    m0: float = float(os.environ.get("MECH_M0", "0.15"))
    use_push_pull: bool = os.environ.get("MECH_PUSH_PULL", "1").strip() == "1"

    a_S: float = 1.5
    a_D: float = 0.4
    a_H: float = 0.4
    K_A: float = 2000.0

    w_H: float = 1.0
    w_S: float = 1.0
    lam: float = 2.0

    mosq_scale: float = MECH_MOSQ_SCALE
    birds_mode: str = "per_cell"


def _kernel_linear(d_km: np.ndarray, dmax: float) -> np.ndarray:
    k = (dmax - d_km) / max(1e-9, dmax)
    return np.clip(k, 0.0, 1.0)


@memory.cache
def _build_neighbors_cached(lat_vals: Tuple[float, ...], lon_vals: Tuple[float, ...], dmax_km: float):
    """
    Cached neighbor lists for a given set of coordinates and movement radius.
    """
    lat = np.array(lat_vals, dtype=float)
    lon = np.array(lon_vals, dtype=float)
    coords = np.radians(np.c_[lat, lon])
    tree = BallTree(coords, metric="haversine")
    r = float(dmax_km) / EARTH_RADIUS_KM

    ind, dist = tree.query_radius(coords, r=r, return_distance=True, sort_results=True)

    neigh: List[np.ndarray] = []
    kij: List[np.ndarray] = []

    for j in range(len(lat)):
        idx = ind[j]
        dist_rad = dist[j]
        if idx.size == 0:
            neigh.append(np.zeros((0,), dtype=int))
            kij.append(np.zeros((0,), dtype=float))
            continue
        mask = idx != j
        idx = idx[mask]
        dist_km = (dist_rad[mask] * EARTH_RADIUS_KM).astype(float)
        neigh.append(idx.astype(int))
        kij.append(_kernel_linear(dist_km, dmax_km))

    return neigh, kij


def _build_R(
    A_pool: np.ndarray,
    neigh: List[np.ndarray],
    kij: List[np.ndarray],
    params: MechParams,
    H: np.ndarray,
    Sspr: np.ndarray,
):
    """
    Build movement operator R (sparse if SciPy available, else dense).
    move_pool = R @ A_pool gives net inflow/outflow for each cell.
    """
    N = A_pool.size

    if _HAVE_SCIPY:
        rows, cols, data = [], [], []
        for j in range(N):
            nbr = neigh[j]
            if nbr.size == 0:
                continue
            K = kij[j]

            if not params.use_push_pull:
                r_out = params.m0 * K
            else:
                Aj = max(0.0, float(A_pool[j]))
                mj = params.m0 * np.exp(
                    params.a_S * float(Sspr[j]) +
                    params.a_D * (Aj / max(1e-9, params.K_A)) +
                    params.a_H * (1.0 - float(H[j]))
                )
                Ui = params.w_H * H[nbr] - params.w_S * Sspr[nbr]
                logits = params.lam * Ui + np.log(np.maximum(K, 1e-12))
                p = _safe_softmax(logits)
                r_out = mj * p

            rows.extend(nbr.tolist())
            cols.extend([j] * nbr.size)
            data.extend(r_out.tolist())

            rows.append(j)
            cols.append(j)
            data.append(-float(np.sum(r_out)))

        return sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    R = np.zeros((N, N), dtype=float)
    for j in range(N):
        nbr = neigh[j]
        if nbr.size == 0:
            continue
        K = kij[j]

        if not params.use_push_pull:
            r_out = params.m0 * K
        else:
            Aj = max(0.0, float(A_pool[j]))
            mj = params.m0 * np.exp(
                params.a_S * float(Sspr[j]) +
                params.a_D * (Aj / max(1e-9, params.K_A)) +
                params.a_H * (1.0 - float(H[j]))
            )
            Ui = params.w_H * H[nbr] - params.w_S * Sspr[nbr]
            logits = params.lam * Ui + np.log(np.maximum(K, 1e-12))
            p = _safe_softmax(logits)
            r_out = mj * p

        R[nbr, j] += r_out
        R[j, j] -= np.sum(r_out)

    return R


def _matvec(R, x):
    if _HAVE_SCIPY:
        return R.dot(x)
    return R @ x


def _zeta_ulv(day_of_year: int, params: MechParams) -> float:
    if params.ulv_apply_doy is None:
        return 0.0
    d0 = int(params.ulv_apply_doy)
    d1 = d0 + int(params.ulv_duration_days)
    return float(params.zeta0) if (d0 <= int(day_of_year) <= d1) else 0.0


def _psiB_introduction(t_day: float, params: MechParams) -> float:
    """
    Proxy for bird introduction pulse.
    """
    A0, Am, Aw = float(params.intro_A0), float(params.intro_Am), float(params.intro_Aw)
    if A0 <= 0.0:
        return 0.0
    x = (Am - float(t_day)) / max(1e-6, Aw)
    ex = math.exp(x)
    return float(A0 * (ex / ((1.0 + ex) ** 2)))


def _mech_step_day(
    st: Dict[str, np.ndarray],
    params: MechParams,
    neigh: List[np.ndarray],
    kij: List[np.ndarray],
    H: np.ndarray,
    Sspr: np.ndarray,
    eta_L: np.ndarray,
    zeta_adult: np.ndarray,
    dt: float,
    NH: np.ndarray,
    NB_target: np.ndarray,
    alphaM: float,
    eta: float,
    gammaM: float,
    day_of_year: int,
    dev_mult: np.ndarray,
    mort_mult: np.ndarray,
    bite_mult: np.ndarray,
) -> Dict[str, np.ndarray]:
    E = st["E"]; L = st["L"]; P = st["P"]
    A = st["A"]; A_B = st["A_B"]; A_En = st["A_En"]; A_El = st["A_El"]

    S_M = st["S_M"]; E_M = st["E_M"]; I_M = st["I_M"]
    S_B = st["S_B"]; I_B = st["I_B"]; R_B = st["R_B"]

    gamma_A0_eff = params.gamma_A0 * dev_mult
    gamma_L_eff = params.gamma_L * dev_mult
    gamma_P_eff = params.gamma_P * dev_mult
    gamma_A_eff = params.gamma_A * dev_mult

    beta_L_eff = params.beta_L * mort_mult
    beta_P_eff = params.beta_P * mort_mult
    beta_A_eff = params.beta_A * mort_mult

    births_E = gamma_A0_eff * params.alpha_A * A_El
    flow_E_to_L = params.psi * gamma_L_eff * E
    flow_L_to_P = gamma_P_eff * L
    flow_P_to_A = gamma_A_eff * params.sigma_A * P * np.exp(
        -params.gamma_em * (1.0 + (P / max(1e-9, params.K_P)))
    )

    flow_A_to_B = params.gamma_B * A
    flow_B_to_En = params.gamma_En * A_B
    flow_En_to_El = params.gamma_El * A_En

    dE = births_E - flow_E_to_L - params.beta_E * E
    dL = (
        flow_E_to_L
        - flow_L_to_P
        - beta_L_eff * L
        - params.beta_W * L
        - eta_L * L
        - (params.beta_1 / max(1e-9, params.K_L)) * (L * L)
    )
    dP = flow_L_to_P - beta_P_eff * P - (gamma_A_eff * P)

    zeta_pulse = _zeta_ulv(day_of_year, params)
    adult_mort = (beta_A_eff + zeta_adult + zeta_pulse)

    dA_local = flow_P_to_A - adult_mort * A - flow_A_to_B
    dA_B_local = flow_A_to_B - adult_mort * A_B - params.mu_B * A_B - flow_B_to_En
    dA_En_local = flow_B_to_En - adult_mort * A_En - flow_En_to_El
    dA_El_local = flow_En_to_El - adult_mort * A_El

    A_pool = A + A_B + A_En + A_El
    Rop = _build_R(A_pool, neigh, kij, params, H=H, Sspr=Sspr)
    move_pool = _matvec(Rop, A_pool)

    with np.errstate(divide="ignore", invalid="ignore"):
        shA = np.where(A_pool > 0, A / A_pool, 0.0)
        shAB = np.where(A_pool > 0, A_B / A_pool, 0.0)
        shAEn = np.where(A_pool > 0, A_En / A_pool, 0.0)
        shAEl = np.where(A_pool > 0, A_El / A_pool, 0.0)

    dA_move = move_pool * shA
    dAB_move = move_pool * shAB
    dAEn_move = move_pool * shAEn
    dAEl_move = move_pool * shAEl

    with np.errstate(divide="ignore", invalid="ignore"):
        fS = np.where(A_pool > 0, S_M / A_pool, 0.0)
        fE = np.where(A_pool > 0, E_M / A_pool, 0.0)
        fI = np.where(A_pool > 0, I_M / A_pool, 0.0)

    S_M_m = S_M + (move_pool * fS) * dt
    E_M_m = E_M + (move_pool * fE) * dt
    I_M_m = I_M + (move_pool * fI) * dt

    NB = np.maximum(S_B + I_B + R_B, 1e-9)
    NB_blend = 0.98 * NB + 0.02 * np.maximum(NB_target, 1.0)
    scaleB = np.where(NB > 0, NB_blend / NB, 1.0)
    S_B = S_B * scaleB
    I_B = I_B * scaleB
    R_B = R_B * scaleB
    NB = np.maximum(S_B + I_B + R_B, 1e-9)

    eta_eff = eta * bite_mult
    denom = np.maximum(params.alphaF * NB + np.maximum(NH, 1e-9), 1e-9)

    lambda_M = (alphaM * params.beta1 * eta_eff * params.alphaF * I_B) / denom
    lambda_B = (params.phiB * alphaM * params.beta2 * eta_eff * params.alphaF * I_M_m) / denom
    lambda_H = (params.phiH * alphaM * params.beta3 * eta_eff * I_M_m) / denom

    dS_M = -lambda_M * S_M_m
    dE_M = lambda_M * S_M_m - gammaM * E_M_m
    dI_M = gammaM * E_M_m

    S_M_new = S_M_m + dS_M * dt
    E_M_new = E_M_m + dE_M * dt
    I_M_new = I_M_m + dI_M * dt

    psiB = _psiB_introduction(float(day_of_year), params)

    dS_B = params.bB * NB - params.mB * S_B - lambda_B * S_B
    dI_B = lambda_B * S_B - params.gammaBird * I_B - params.mB * I_B - params.deltaB * I_B + (I_B * psiB)
    dR_B = params.gammaBird * I_B - params.mB * R_B

    S_B_new = S_B + dS_B * dt
    I_B_new = I_B + dI_B * dt
    R_B_new = R_B + dR_B * dt

    if params.birds_mode == "well_mixed":
        SBm = float(np.mean(S_B_new))
        IBm = float(np.mean(I_B_new))
        RBm = float(np.mean(R_B_new))
        S_B_new = np.full_like(S_B_new, SBm)
        I_B_new = np.full_like(I_B_new, IBm)
        R_B_new = np.full_like(R_B_new, RBm)

    E_new = np.maximum(E + dE * dt, 0.0)
    L_new = np.maximum(L + dL * dt, 0.0)
    P_new = np.maximum(P + dP * dt, 0.0)

    A_new = np.maximum(A + (dA_local + dA_move) * dt, 0.0)
    A_B_new = np.maximum(A_B + (dA_B_local + dAB_move) * dt, 0.0)
    A_En_new = np.maximum(A_En + (dA_En_local + dAEn_move) * dt, 0.0)
    A_El_new = np.maximum(A_El + (dA_El_local + dAEl_move) * dt, 0.0)

    A_pool_new = A_new + A_B_new + A_En_new + A_El_new
    total_sei = np.maximum(S_M_new + E_M_new + I_M_new, 1e-12)
    scale = np.clip(A_pool_new / total_sei, 0.0, 1.0)
    S_M_new = np.maximum(S_M_new * scale, 0.0)
    E_M_new = np.maximum(E_M_new * scale, 0.0)
    I_M_new = np.maximum(I_M_new * scale, 0.0)

    S_B_new = np.maximum(S_B_new, 0.0)
    I_B_new = np.maximum(I_B_new, 0.0)
    R_B_new = np.maximum(R_B_new, 0.0)

    return {
        "E": E_new, "L": L_new, "P": P_new,
        "A": A_new, "A_B": A_B_new, "A_En": A_En_new, "A_El": A_El_new,
        "S_M": S_M_new, "E_M": E_M_new, "I_M": I_M_new,
        "S_B": S_B_new, "I_B": I_B_new, "R_B": R_B_new,
        "lambda_H": lambda_H,
    }


def run_mechanistic_week(
    grid_df: pd.DataFrame,
    cw: Dict[str, float],
    year: int,
    week: int,
    days: int = 21,
    intervention: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    params = MechParams()

    lat = grid_df["Latitude"].to_numpy(dtype=float)
    lon = grid_df["Longitude"].to_numpy(dtype=float)
    N = len(grid_df)

    neigh, kij = _build_neighbors_cached(tuple(lat.tolist()), tuple(lon.tolist()), float(params.d_max_km))

    H = np.full(N, 0.5, dtype=float)
    Sspr = np.zeros(N, dtype=float)

    eta_L_base = np.zeros(N, dtype=float)
    zeta_adult_base = np.zeros(N, dtype=float)

    T = float(cw.get("temp", 20.0))
    rain_week = float(cw.get("rain", 0.0))
    rain_mm_day = _weekly_rain_to_daily_mm(rain_week)

    doy0 = _day_of_year_from_iso(year, week)
    Dhrs = float(np.mean([_daylength_hours(float(lat[i]), doy0) for i in range(min(N, 50))])) if N > 0 else 14.0
    alphaM = _alphaM_from_daylength(Dhrs)
    eta = _eta_biting_rate(T)
    gammaM = _gammaM_EIP_rate(T, params.gammaM_base)

    if "pop_density" in grid_df.columns:
        popd = grid_df["pop_density"].to_numpy(dtype=float)
        popd = np.where(np.isfinite(popd), popd, GLOBAL_POP_MEAN)
    else:
        popd = np.full(N, GLOBAL_POP_MEAN, dtype=float)

    NH = MECH_NH_BASE + MECH_NH_SCALE * np.maximum(popd, 0.0)
    NB_target = np.maximum(MECH_BIRD_BASE + MECH_BIRD_PER_HUMAN * NH * 1000.0, 50.0)

    sp_arrays = _species_param_arrays(grid_df["Species"])
    dev_mult = sp_arrays["dev_mult"]
    mort_mult = sp_arrays["mort_mult"]
    bite_mult = sp_arrays["bite_mult"]

    # initial life stages
    A0_raw = np.full(N, 80.0, dtype=float)
    L0_raw = np.full(N, 120.0, dtype=float)
    P0_raw = np.full(N, 40.0, dtype=float)
    E0_raw = np.full(N, 150.0, dtype=float)

    st = {
        "E": E0_raw.copy(),
        "L": L0_raw.copy(),
        "P": P0_raw.copy(),

        "A": A0_raw.copy(),
        "A_B": np.zeros(N, dtype=float),
        "A_En": np.zeros(N, dtype=float),
        "A_El": np.zeros(N, dtype=float),

        "S_M": A0_raw.copy(),
        "E_M": np.zeros(N, dtype=float),
        "I_M": np.zeros(N, dtype=float),

        "S_B": np.maximum(NB_target - 2.0, 10.0).astype(float),
        "I_B": np.full(N, 2.0, dtype=float),
        "R_B": np.zeros(N, dtype=float),

        "lambda_H": np.zeros(N, dtype=float),
    }

    spray_mask = np.zeros(N, dtype=bool)
    larv_mask = np.zeros(N, dtype=bool)

    iv = intervention or {}
    sites = iv.get("sites") or []
    spray_r_km = float(iv.get("spray_radius_km", 0.0) or 0.0)

    adult_kill = float(iv.get("adult_kill_frac", 0.45))
    extra_mort = float(iv.get("extra_mortality", 0.20))
    larv_kill = float(iv.get("larvicide_kill_rate", 0.15))

    adulticide_days = iv.get("adulticide_days", [0])
    larvicide_days_list = iv.get("larvicide_days_list", [0])

    if isinstance(adulticide_days, (int, float)):
        adulticide_days = [int(adulticide_days)]
    adulticide_days = set([int(x) for x in adulticide_days])

    if isinstance(larvicide_days_list, (int, float)):
        larvicide_days_list = [int(larvicide_days_list)]
    larvicide_days_list = set([int(x) for x in larvicide_days_list])

    residual_half_life = float(iv.get("residual_half_life_days", 2.0))
    larv_half_life = float(iv.get("larvicide_half_life_days", 5.0))

    if sites and spray_r_km > 0:
        for s in sites:
            lat0 = float(s["Latitude"])
            lon0 = float(s["Longitude"])
            d = haversine_distance_km(lat0, lon0, lat, lon)
            spray_mask |= (d <= spray_r_km)
            larv_mask |= (d <= spray_r_km)

    rain_hist = []
    lam_hist = []
    pool_hist = []
    I_hist = []
    prev_hist = []
    E_hist = []
    L_hist = []
    P_hist = []

    last_adult_day = None
    last_larv_day = None

    for d in range(int(days)):
        day_of_year = doy0 + d

        params.psi = alphaM
        params.gamma_L = _gamma_L_from_temp(T)
        params.gamma_A = _gamma_A_from_temp(T)
        params.gamma_P = params.gamma_A / 4.0
        params.beta_P = _beta_P_from_temp(T, mu_P=0.0)
        params.beta_L = _beta_L_from_temp(T)
        params.beta_A = params.beta_L / 10.0
        params.gamma_A0 = _gamma_A0_from_temp(T)

        rain_hist.append(rain_mm_day)
        pnorm = _rolling_precip_norm(rain_hist, window_days=14)
        params.K_L = float(5000.0 * (1.0 + pnorm))
        params.beta_W = float(_flush_mortality_from_rain(rain_hist))

        eta_L = eta_L_base.copy()
        zeta_adult = zeta_adult_base.copy()

        if spray_mask.any() and d in adulticide_days and adult_kill > 0:
            kill = float(np.clip(adult_kill, 0.0, 0.99))
            for key in ["A", "A_B", "A_En", "A_El", "S_M", "E_M", "I_M"]:
                st[key][spray_mask] *= (1.0 - kill)
            last_adult_day = d

        if larv_mask.any() and d in larvicide_days_list:
            last_larv_day = d

        if spray_mask.any() and last_adult_day is not None:
            dd = max(0, d - last_adult_day)
            zeta_now = extra_mort * math.exp(-math.log(2.0) * dd / max(residual_half_life, 1e-6))
            zeta_adult[spray_mask] += float(np.clip(zeta_now, 0.0, 5.0))

        if larv_mask.any() and last_larv_day is not None:
            dd = max(0, d - last_larv_day)
            larv_now = larv_kill * math.exp(-math.log(2.0) * dd / max(larv_half_life, 1e-6))
            eta_L[larv_mask] += float(np.clip(larv_now, 0.0, 5.0))

        st = _mech_step_day(
            st, params, neigh, kij,
            H=H, Sspr=Sspr,
            eta_L=eta_L,
            zeta_adult=zeta_adult,
            dt=1.0,
            NH=NH,
            NB_target=NB_target,
            alphaM=alphaM,
            eta=eta,
            gammaM=gammaM,
            day_of_year=day_of_year,
            dev_mult=dev_mult,
            mort_mult=mort_mult,
            bite_mult=bite_mult,
        )

        A_pool = st["A"] + st["A_B"] + st["A_En"] + st["A_El"]
        pool_hist.append(A_pool)
        I_hist.append(st["I_M"])
        lam_hist.append(st["lambda_H"])
        prev_hist.append(100.0 * st["I_M"] / np.maximum(A_pool, 1e-9))
        E_hist.append(st["E"])
        L_hist.append(st["L"])
        P_hist.append(st["P"])

    adult_raw = np.mean(np.vstack(pool_hist[-7:]), axis=0)
    infected_raw = np.mean(np.vstack(I_hist[-7:]), axis=0)
    lamH = np.mean(np.vstack(lam_hist[-7:]), axis=0)

    egg_mean = np.mean(np.vstack(E_hist[-7:]), axis=0)
    larva_mean = np.mean(np.vstack(L_hist[-7:]), axis=0)
    pupa_mean = np.mean(np.vstack(P_hist[-7:]), axis=0)

    trap_type_for_cal = "GRAVID"
    if "TRAP_TYPE" in grid_df.columns and len(grid_df["TRAP_TYPE"].dropna()) > 0:
        trap_type_for_cal = str(grid_df["TRAP_TYPE"].dropna().iloc[0])

    adult_cal, infected_cal, prevalence_cal = _calibrate_mech_outputs(
        grid_df=grid_df,
        adult_raw=adult_raw * params.mosq_scale,
        infected_raw=infected_raw * params.mosq_scale,
        week=week,
        year=year,
        trap_type=trap_type_for_cal,
    )

    return {
        "egg_pool": np.clip(egg_mean, 0.0, None),
        "larva_pool": np.clip(larva_mean, 0.0, None),
        "pupa_pool": np.clip(pupa_mean, 0.0, None),
        "adult_pool_raw": np.clip(adult_raw, 0.0, None),
        "adult_pool": np.clip(adult_cal, 0.0, None),
        "I_mosq_raw": np.clip(infected_raw, 0.0, None),
        "I_mosq": np.clip(infected_cal, 0.0, None),
        "lambda_H": np.clip(lamH, 0.0, None),
        "prevalence": np.clip(prevalence_cal, 0.0, None),
    }


# ============================================================
# HISTORY KDE SURFACE + SMOOTHING
# ============================================================

def _collect_positive_points(
    year: int,
    week: int,
    lookbacks: Tuple[int, int] = (4, 8),
    cross_years: bool = True,
    cross_year_window: int = 1,
) -> np.ndarray:
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

    for i, neigh_i in enumerate(idxs):
        if len(neigh_i) == 0:
            continue
        p0 = np.radians([glat[i], glon[i]])
        pn = np.radians(np.c_[plat[neigh_i], plon[neigh_i]])

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
# GRID BUILD (CACHED)
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
    hist_kernel_sigma_km: float = 0.9,
    hist_cross_years: bool = True,
    hist_cross_year_window: int = 1,
    smooth_final: bool = True,
    smooth_k: int = 60,
    smooth_sigma_km: float = 0.9,
    use_population: bool = True,
):
    """
    Build grid points and compute risk surfaces.
    """
    year = int(year)
    week = int(week)
    pop_strength = float(pop_strength)
    pop_radius_km = float(pop_radius_km)

    hist_strength = _clip01(hist_strength)
    hist_radius_km = float(hist_radius_km)

    cw = get_weather_for_week(year, week, cw)

    grid_date = datetime.fromisocalendar(year, week, 1)
    doy = grid_date.timetuple().tm_yday
    sin_doy = np.sin(2 * np.pi * doy / 365.0)
    cos_doy = np.cos(2 * np.pi * doy / 365.0)

    recs = []
    lat_vals = np.arange(GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP)
    lon_vals = np.arange(GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP)

    for lat in lat_vals:
        for lon in lon_vals:
            if concave_hull is not None and (not concave_hull.covers(Point(lon, lat))):
                continue

            recs.append(
                {
                    "Latitude": float(lat),
                    "Longitude": float(lon),

                    "Year": int(year),
                    "Week": int(week),
                    "temp": float(cw["temp"]),
                    "humidity": float(cw["humidity"]),
                    "rain": float(cw["rain"]),
                    "wind_speed": float(cw["wind_speed"]),

                    "day_of_year": int(doy),
                    "sin_doy": float(sin_doy),
                    "cos_doy": float(cos_doy),
                    "trap_effort_mean": _safe_float(trap_effort_mean),
                    "TRAP_TYPE": "GRAVID",

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

                    "stage_egg": np.nan,
                    "stage_larva": np.nan,
                    "stage_pupa": np.nan,
                    "stage_adult_raw": np.nan,

                    "pred_mosq_total": np.nan,
                    "pred_mosq_infected": np.nan,
                    "pred_prevalence": np.nan,

                    "weather_source": str(cw.get("source", "unknown")),
                }
            )

    grid_df = pd.DataFrame(recs)
    if len(grid_df) == 0:
        return grid_df, 0.0, 0.0

    X_sp = grid_df.reindex(columns=FEATURES_SP).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    sp_probs = model_sp.predict_proba(X_sp)
    grid_df["Species"] = model_sp.classes_[np.argmax(sp_probs, axis=1)]

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

    if not USE_MECH:
        X_inf = grid_df.reindex(columns=FEATURES_INF).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        grid_df["risk_inf"] = model_inf.predict_proba(X_inf)[:, 1]

        grid_df["sin_week"] = np.sin(2 * np.pi * grid_df["Week"].astype(float) / 52.0)
        grid_df["cos_week"] = np.cos(2 * np.pi * grid_df["Week"].astype(float) / 52.0)

        grid_df["SPECIES"] = grid_df["Species"].astype(str).fillna("UNKNOWN").str.upper().str.strip()
        default_trap_type = (
            str(MOSQ_META.get("default_trap_type_mode", "GRAVID"))
            if isinstance(MOSQ_META, dict) else "GRAVID"
        )
        grid_df["TRAP_TYPE"] = default_trap_type

        species_levels = set([s.upper().strip() for s in MOSQ_META.get("species_levels", []) if isinstance(s, str)])
        trap_levels = set([t.upper().strip() for t in MOSQ_META.get("trap_type_levels", []) if isinstance(t, str)])

        if species_levels:
            safe_species = next(iter(species_levels))
            grid_df.loc[~grid_df["SPECIES"].isin(species_levels), "SPECIES"] = safe_species

        if trap_levels:
            safe_trap = default_trap_type.upper().strip()
            if safe_trap not in trap_levels:
                safe_trap = next(iter(trap_levels))
            grid_df.loc[~grid_df["TRAP_TYPE"].isin(trap_levels), "TRAP_TYPE"] = safe_trap

        for c in ["Latitude", "Longitude", "Year", "Week", "temp", "humidity", "rain", "wind_speed", "sin_week", "cos_week"]:
            grid_df[c] = pd.to_numeric(grid_df[c], errors="coerce").fillna(0.0)

        if model_ab is not None:
            try:
                design_info = getattr(model_ab.model.data, "design_info", None)
                if design_info is not None:
                    from patsy import build_design_matrices
                    exog = build_design_matrices([design_info], grid_df, return_type="dataframe")[0]
                    pred_total = model_ab.predict(exog, transform=False)
                else:
                    pred_total = model_ab.predict(grid_df)

                pred_total = np.asarray(pred_total, dtype=float)
                pred_total = np.where(np.isfinite(pred_total), pred_total, 0.0)
                pred_total = np.clip(pred_total, 0.0, None)
                grid_df["pred_mosq_total"] = pred_total

            except Exception as e:
                print("[ERROR] Abundance predict crashed:", repr(e))
                grid_df["pred_mosq_total"] = 0.0
        else:
            grid_df["pred_mosq_total"] = 0.0

        grid_df["pred_mosq_infected"] = (
            grid_df["pred_mosq_total"].astype(float) * grid_df["risk_inf"].astype(float)
        ).clip(lower=0.0)

        denom = np.maximum(grid_df["pred_mosq_total"].astype(float), 1e-9)
        grid_df["pred_prevalence"] = (100.0 * grid_df["pred_mosq_infected"].astype(float) / denom).clip(lower=0.0)

        grid_df["risk_model"] = _normalize_robust(grid_df["risk_inf"].to_numpy(dtype=float), 0.01, 0.99)

    else:
        mech = run_mechanistic_week(
            grid_df,
            cw=cw,
            year=year,
            week=week,
            days=21,
            intervention=None,
        )

        grid_df["stage_egg"] = np.clip(mech["egg_pool"], 0.0, None)
        grid_df["stage_larva"] = np.clip(mech["larva_pool"], 0.0, None)
        grid_df["stage_pupa"] = np.clip(mech["pupa_pool"], 0.0, None)
        grid_df["stage_adult_raw"] = np.clip(mech["adult_pool_raw"], 0.0, None)

        grid_df["pred_mosq_total"] = np.clip(mech["adult_pool"], 0.0, None)
        grid_df["pred_mosq_infected"] = np.clip(mech["I_mosq"], 0.0, None)
        grid_df["pred_prevalence"] = np.clip(mech["prevalence"], 0.0, None)

        grid_df["risk_inf"] = np.clip(mech["lambda_H"], 0.0, None)
        grid_df["risk_model"] = _normalize_robust(grid_df["risk_inf"].to_numpy(dtype=float), 0.01, 0.99)

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

    grid_df["risk_blend"] = (1.0 - hist_strength) * grid_df["risk_model"] + hist_strength * grid_df["hist_score"]

    grid_df["species_weight"] = (
        grid_df["Species"].astype(str).str.upper().map(SPECIES_WEIGHTS).fillna(1.0).astype(float)
    )

    grid_df["pop_multiplier"] = (1.0 + pop_strength * grid_df["pop_weight"]).fillna(1.0)

    grid_df["risk_raw_value"] = (grid_df["risk_blend"] * grid_df["species_weight"]).fillna(0.0)
    grid_df["risk_pop_value"] = (grid_df["risk_raw_value"] * grid_df["pop_multiplier"]).fillna(0.0)

    if bool(smooth_final) and len(grid_df) > 10:
        grid_df["risk_raw_value"] = smooth_grid_risk_knn(
            grid_df, col="risk_raw_value", k=int(smooth_k), sigma_km=float(smooth_sigma_km)
        ).astype(float)
        grid_df["risk_pop_value"] = smooth_grid_risk_knn(
            grid_df, col="risk_pop_value", k=int(smooth_k), sigma_km=float(smooth_sigma_km)
        ).astype(float)

    grid_df["risk_raw"] = _normalize_robust(grid_df["risk_raw_value"].to_numpy(dtype=float), 0.01, 0.99).astype(float)
    grid_df["risk_pop"] = _normalize_robust(grid_df["risk_pop_value"].to_numpy(dtype=float), 0.01, 0.99).astype(float)
    grid_df["risk_final"] = grid_df["risk_pop"] if bool(use_population) else grid_df["risk_raw"]

    return grid_df, float(grid_df["risk_final"].min()), float(grid_df["risk_final"].max())


# ============================================================
# SPRAY SITE SELECTION
# ============================================================

def select_spray_sites(grid_df: pd.DataFrame, n_sites: int, spray_radius_km: float) -> pd.DataFrame:
    """
    Pick n_sites centers that maximize captured risk (risk_final),
    with no sites within 2*spray_radius_km of each other.
    """
    n_sites = int(n_sites)
    spray_radius_km = float(spray_radius_km)

    lats = grid_df["Latitude"].values.astype(float)
    lons = grid_df["Longitude"].values.astype(float)
    risks = np.clip(grid_df["risk_final"].values.astype(float), 0.0, 1.0)

    def _dist_from(i):
        return haversine_distance_km(lats[i], lons[i], lats, lons)

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
    Fallback non-mechanistic / post-hoc layer.
    """
    df2 = grid_df.copy()
    spray_radius_km = float(spray_radius_km)

    efficacy = {
        "CULEX PIPIENS": 0.52,
        "CULEX TARSALIS": 0.30,
        "CULEX RESTUANS": 0.40,
        "CULEX PIPIENS/RESTUANS": 0.45,
    }

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

            df2.loc[affected, "risk_raw_value"] = (df2.loc[affected, "risk_raw_value"].astype(float) * mult)
            df2.loc[affected, "risk_pop_value"] = (df2.loc[affected, "risk_pop_value"].astype(float) * mult)

            if "pred_mosq_total" in df2.columns:
                df2.loc[affected, "pred_mosq_total"] = (df2.loc[affected, "pred_mosq_total"].astype(float) * mult)
            if "pred_mosq_infected" in df2.columns:
                df2.loc[affected, "pred_mosq_infected"] = (df2.loc[affected, "pred_mosq_infected"].astype(float) * mult)

    df2["risk_raw"] = _normalize_robust(df2["risk_raw_value"].to_numpy(dtype=float), 0.01, 0.99).astype(float)
    df2["risk_pop"] = _normalize_robust(df2["risk_pop_value"].to_numpy(dtype=float), 0.01, 0.99).astype(float)

    if "pred_mosq_total" in df2.columns and "pred_mosq_infected" in df2.columns:
        denom = np.maximum(df2["pred_mosq_total"].astype(float), 1e-9)
        df2["pred_prevalence"] = (100.0 * df2["pred_mosq_infected"].astype(float) / denom).clip(lower=0.0)

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

    use_population = p.get("use_population", "true").lower() != "false"

    cw_req = {
        "temp": p.get("temp", np.nan, type=float),
        "humidity": p.get("humidity", np.nan, type=float),
        "rain": p.get("rain", np.nan, type=float),
        "wind_speed": p.get("wind_speed", np.nan, type=float),
    }

    cw_used = get_weather_for_week(int(y), int(w), cw_req)

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

    risk_field = "risk_pop" if use_population else "risk_raw"
    grid["risk"] = grid[risk_field]

    mn = float(np.nanmin(grid["risk"].to_numpy(dtype=float)))
    mx = float(np.nanmax(grid["risk"].to_numpy(dtype=float)))
    if not np.isfinite(mn):
        mn = 0.0
    if not np.isfinite(mx):
        mx = 1.0

    mn_raw = float(np.nanmin(grid["risk_raw"].to_numpy(dtype=float))) if "risk_raw" in grid.columns else 0.0
    mx_raw = float(np.nanmax(grid["risk_raw"].to_numpy(dtype=float))) if "risk_raw" in grid.columns else 1.0
    mn_pop = float(np.nanmin(grid["risk_pop"].to_numpy(dtype=float))) if "risk_pop" in grid.columns else 0.0
    mx_pop = float(np.nanmax(grid["risk_pop"].to_numpy(dtype=float))) if "risk_pop" in grid.columns else 1.0

    out_cols = [
        "Latitude", "Longitude", "Species",
        "risk",
        "risk_raw",
        "risk_pop",
        "pop_density", "pop_weight", "pop_multiplier",
        "stage_egg",
        "stage_larva",
        "stage_pupa",
        "stage_adult_raw",
        "pred_mosq_total",
        "pred_mosq_infected",
        "pred_prevalence",
        "weather_source",
    ]

    out_cols = [c for c in out_cols if c in grid.columns]
    out = grid[out_cols].copy()
    out = out.where(pd.notnull(out), None)

    records = out.to_dict("records")

    meta = {
        "mode": "mechanistic" if USE_MECH else "ml_glm",
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
            "units": {"temp": "Celsius", "wind_speed": "m/s", "rain": "mm"},
        },
        "mosquito_estimate": {
            "pred_mosq_total": "Expected abundance (ML/GLM) OR mechanistic adult pool (calibrated) (mechanistic mode).",
            "pred_mosq_infected": "Expected infected mosquitoes (ML) OR mechanistic infectious adults I_M (calibrated) (mechanistic mode).",
            "pred_prevalence": "Infection prevalence % = 100 * pred_mosq_infected / pred_mosq_total.",
        },
        "mechanistic_notes": {
            "NH_proxy": "NH uses pop_density: NH = NH_BASE + NH_SCALE * pop_density.",
            "NB_proxy": "NB is still a proxy surface derived from NH with MECH_BIRD_BASE and MECH_BIRD_PER_HUMAN until you supply bird density.",
            "weather_forcing": "Mechanistic abundance now uses paper-based functions for gamma_L(T), gamma_A(T), beta_P(T), beta_L(T), gamma_A0(T), plus rain-driven K_L and flushing mortality beta_W.",
            "diapause": "alphaM(daylength) now acts as a non-diapausing fraction / seasonal activity switch.",
            "ulv_pulse": f"ULV extra mortality applied if MECH_ULV_APPLY_DOY is set (current={MECH_ULV_APPLY_DOY}).",
            "scale": f"MECH_MOSQ_SCALE={MECH_MOSQ_SCALE} rescales model adult pool before calibration.",
            "calibration": "Mechanistic adult abundance is calibrated to trap-count-like scale using five layers: GLM bridge, species-week empirical medians, species-week-trap-type medians, a local nearby-trap target surface, and a local trap-ID neighborhood bias surface.",
            "species_specific": "Mechanistic development, mortality, and biting are now adjusted by dominant species at each cell.",
            "life_stages": "Grid now includes stage_egg, stage_larva, stage_pupa, and stage_adult_raw in addition to calibrated adult abundance.",
            "spray_dynamics": "Mechanistic spray supports repeated adulticide days, repeated larvicide days, and exponential residual decay.",
        },
        "definitions": {
            "risk_raw": "Risk surface without population weighting (model+history blended, species-weighted).",
            "risk_pop": "Risk surface with population multiplier applied: risk_raw_value * (1 + pop_strength * pop_weight).",
            "pop_weight": "0..1 scaled local population density across the grid.",
            "hist_strength": "Blend factor between model risk and historical positives KDE (0=model only, 1=history only).",
        },
        "notes": {
            "mechanistic_risk_inf": "In mechanistic mode, risk_inf is lambda_H (a force-of-infection proxy) before normalization.",
            "scipy_sparse": bool(_HAVE_SCIPY),
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

    if d.get("polygon"):
        poly = shape(d["polygon"])
        mask = grid_cur.apply(lambda r: poly.covers(Point(r.Longitude, r.Latitude)), axis=1)
        grid_cur = grid_cur.loc[mask].reset_index(drop=True)

    grid_cur["risk_final"] = grid_cur["risk_pop"] if use_population else grid_cur["risk_raw"]
    sites_df = select_spray_sites(grid_cur, n_sites, spray_radius)

    if not USE_MECH:
        post_grid_df = simulate_spray(grid_cur, sites_df, spray_radius)
    else:
        cw_used = get_weather_for_week(int(y), int(w), cw_req)

        iv = {
            "sites": sites_df[["Latitude", "Longitude"]].to_dict("records"),
            "spray_radius_km": float(spray_radius),
            "adult_kill_frac": float(d.get("adult_kill_frac", 0.45)),
            "extra_mortality": float(d.get("extra_mortality", 0.20)),
            "larvicide_kill_rate": float(d.get("larvicide_kill_rate", 0.15)),
            "adulticide_days": d.get("adulticide_days", [0]),
            "larvicide_days_list": d.get("larvicide_days_list", [0]),
            "residual_half_life_days": float(d.get("residual_half_life_days", 2.0)),
            "larvicide_half_life_days": float(d.get("larvicide_half_life_days", 5.0)),
        }

        mech_post = run_mechanistic_week(
            grid_cur,
            cw=cw_used,
            year=int(y),
            week=int(w),
            days=21,
            intervention=iv,
        )

        post_grid_df = grid_cur.copy()

        post_grid_df["stage_egg"] = np.clip(mech_post["egg_pool"], 0.0, None)
        post_grid_df["stage_larva"] = np.clip(mech_post["larva_pool"], 0.0, None)
        post_grid_df["stage_pupa"] = np.clip(mech_post["pupa_pool"], 0.0, None)
        post_grid_df["stage_adult_raw"] = np.clip(mech_post["adult_pool_raw"], 0.0, None)

        adult_pool_post = np.clip(mech_post["adult_pool"], 0.0, None)
        I_post = np.clip(mech_post["I_mosq"], 0.0, None)
        prev_post = np.clip(mech_post["prevalence"], 0.0, None)
        lam_post = np.clip(mech_post["lambda_H"], 0.0, None)

        post_grid_df["pred_mosq_total"] = adult_pool_post
        post_grid_df["pred_mosq_infected"] = I_post
        post_grid_df["pred_prevalence"] = prev_post

        post_grid_df["risk_inf"] = lam_post
        post_grid_df["risk_model"] = _normalize_robust(post_grid_df["risk_inf"].to_numpy(dtype=float), 0.01, 0.99)

        post_grid_df["risk_blend"] = (
            (1.0 - hist_strength) * post_grid_df["risk_model"] +
            hist_strength * post_grid_df["hist_score"]
        )

        post_grid_df["risk_raw_value"] = (
            post_grid_df["risk_blend"] * post_grid_df["species_weight"]
        ).fillna(0.0)

        post_grid_df["risk_pop_value"] = (
            post_grid_df["risk_raw_value"] * post_grid_df["pop_multiplier"]
        ).fillna(0.0)

        if bool(smooth_final) and len(post_grid_df) > 10:
            post_grid_df["risk_raw_value"] = smooth_grid_risk_knn(
                post_grid_df, col="risk_raw_value", k=int(smooth_k), sigma_km=float(smooth_sigma_km)
            ).astype(float)
            post_grid_df["risk_pop_value"] = smooth_grid_risk_knn(
                post_grid_df, col="risk_pop_value", k=int(smooth_k), sigma_km=float(smooth_sigma_km)
            ).astype(float)

        post_grid_df["risk_raw"] = _normalize_robust(
            post_grid_df["risk_raw_value"].to_numpy(dtype=float), 0.01, 0.99
        ).astype(float)
        post_grid_df["risk_pop"] = _normalize_robust(
            post_grid_df["risk_pop_value"].to_numpy(dtype=float), 0.01, 0.99
        ).astype(float)

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
            "stage_egg", "stage_larva", "stage_pupa", "stage_adult_raw",
            "pred_mosq_total", "pred_mosq_infected", "pred_prevalence",
        ]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()
        out = out.where(pd.notnull(out), None)
        return out.to_dict("records")

    return jsonify(
        {
            "mode": "mechanistic" if USE_MECH else "ml_glm",
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
                "mode": "mechanistic" if USE_MECH else "ml_glm",
                "use_population": use_population,
                "sites": [],
                "grid_pre_spray": [],
                "grid_post_spray": [],
                "min_risk_pre": 0.0,
                "max_risk_pre": 1.0,
            }
        ), 200

    if d.get("polygon"):
        poly = shape(d["polygon"])
        mask = grid_cur.apply(lambda r: poly.covers(Point(r.Longitude, r.Latitude)), axis=1)
        grid_cur = grid_cur.loc[mask].reset_index(drop=True)

    n_sites = int(d.get("n_sites", d.get("num_trucks", 5)))
    spray_radius = float(d.get("spray_radius_km", 1.0))

    grid_cur["risk_final"] = grid_cur["risk_pop"] if use_population else grid_cur["risk_raw"]
    sites_df = select_spray_sites(grid_cur, n_sites, spray_radius)

    if not USE_MECH:
        post_grid_df = simulate_spray(grid_cur, sites_df, spray_radius)
    else:
        cw_used = get_weather_for_week(int(y), int(w), cw_req)

        iv = {
            "sites": sites_df[["Latitude", "Longitude"]].to_dict("records"),
            "spray_radius_km": float(spray_radius),
            "adult_kill_frac": float(d.get("adult_kill_frac", 0.45)),
            "extra_mortality": float(d.get("extra_mortality", 0.20)),
            "larvicide_kill_rate": float(d.get("larvicide_kill_rate", 0.15)),
            "adulticide_days": d.get("adulticide_days", [0]),
            "larvicide_days_list": d.get("larvicide_days_list", [0]),
            "residual_half_life_days": float(d.get("residual_half_life_days", 2.0)),
            "larvicide_half_life_days": float(d.get("larvicide_half_life_days", 5.0)),
        }

        mech_post = run_mechanistic_week(
            grid_cur,
            cw=cw_used,
            year=int(y),
            week=int(w),
            days=21,
            intervention=iv,
        )

        post_grid_df = grid_cur.copy()

        post_grid_df["stage_egg"] = np.clip(mech_post["egg_pool"], 0.0, None)
        post_grid_df["stage_larva"] = np.clip(mech_post["larva_pool"], 0.0, None)
        post_grid_df["stage_pupa"] = np.clip(mech_post["pupa_pool"], 0.0, None)
        post_grid_df["stage_adult_raw"] = np.clip(mech_post["adult_pool_raw"], 0.0, None)

        adult_pool_post = np.clip(mech_post["adult_pool"], 0.0, None)
        I_post = np.clip(mech_post["I_mosq"], 0.0, None)
        prev_post = np.clip(mech_post["prevalence"], 0.0, None)
        lam_post = np.clip(mech_post["lambda_H"], 0.0, None)

        post_grid_df["pred_mosq_total"] = adult_pool_post
        post_grid_df["pred_mosq_infected"] = I_post
        post_grid_df["pred_prevalence"] = prev_post
        post_grid_df["risk_inf"] = lam_post
        post_grid_df["risk_model"] = _normalize_robust(post_grid_df["risk_inf"].to_numpy(dtype=float), 0.01, 0.99)

        post_grid_df["risk_blend"] = (
            (1.0 - hist_strength) * post_grid_df["risk_model"] +
            hist_strength * post_grid_df["hist_score"]
        )
        post_grid_df["risk_raw_value"] = (
            post_grid_df["risk_blend"] * post_grid_df["species_weight"]
        ).fillna(0.0)
        post_grid_df["risk_pop_value"] = (
            post_grid_df["risk_raw_value"] * post_grid_df["pop_multiplier"]
        ).fillna(0.0)

        if bool(smooth_final) and len(post_grid_df) > 10:
            post_grid_df["risk_raw_value"] = smooth_grid_risk_knn(
                post_grid_df, col="risk_raw_value", k=int(smooth_k), sigma_km=float(smooth_sigma_km)
            ).astype(float)
            post_grid_df["risk_pop_value"] = smooth_grid_risk_knn(
                post_grid_df, col="risk_pop_value", k=int(smooth_k), sigma_km=float(smooth_sigma_km)
            ).astype(float)

        post_grid_df["risk_raw"] = _normalize_robust(
            post_grid_df["risk_raw_value"].to_numpy(dtype=float), 0.01, 0.99
        ).astype(float)
        post_grid_df["risk_pop"] = _normalize_robust(
            post_grid_df["risk_pop_value"].to_numpy(dtype=float), 0.01, 0.99
        ).astype(float)

    def _apply_selected_risk(df: pd.DataFrame) -> pd.DataFrame:
        df["risk_final"] = df["risk_pop"] if use_population else df["risk_raw"]
        df["risk"] = df["risk_final"]
        return df

    grid_cur = _apply_selected_risk(grid_cur)
    post_grid_df = _apply_selected_risk(post_grid_df)
    sites_df = _apply_selected_risk(sites_df)

    mn_sel = float(np.nanmin(grid_cur["risk"].to_numpy(dtype=float)))
    mx_sel = float(np.nanmax(grid_cur["risk"].to_numpy(dtype=float)))
    if not np.isfinite(mn_sel):
        mn_sel = 0.0
    if not np.isfinite(mx_sel):
        mx_sel = 1.0

    def _prep(df: pd.DataFrame) -> list:
        keep = [
            "Latitude", "Longitude", "Species",
            "risk", "risk_raw", "risk_pop",
            "pop_density", "pop_weight", "pop_multiplier",
            "stage_egg", "stage_larva", "stage_pupa", "stage_adult_raw",
            "pred_mosq_total", "pred_mosq_infected", "pred_prevalence",
        ]
        keep = [c for c in keep if c in df.columns]
        out = df[keep].copy()
        out = out.where(pd.notnull(out), None)
        return out.to_dict("records")

    return jsonify(
        {
            "mode": "mechanistic" if USE_MECH else "ml_glm",
            "use_population": use_population,
            "sites": _prep(sites_df),
            "grid_pre_spray": _prep(grid_cur),
            "grid_post_spray": _prep(post_grid_df),
            "min_risk_pre": float(mn_sel),
            "max_risk_pre": float(mx_sel),
        }
    )


if __name__ == "__main__":
    print(f"[INFO] Mode: {'mechanistic' if USE_MECH else 'ml_glm'} | SciPy sparse: {_HAVE_SCIPY}")
    if USE_MECH:
        print(
            "[INFO] MECH knobs:",
            f"MECH_MOSQ_SCALE={MECH_MOSQ_SCALE}, MECH_NH_SCALE={MECH_NH_SCALE}, MECH_NH_BASE={MECH_NH_BASE},",
            f"MECH_BIRD_BASE={MECH_BIRD_BASE}, MECH_BIRD_PER_HUMAN={MECH_BIRD_PER_HUMAN},",
            f"MECH_ULV_APPLY_DOY={MECH_ULV_APPLY_DOY}, MECH_ZETA0={MECH_ZETA0}, MECH_INTRO_A0={MECH_INTRO_A0},",
            f"CALIB_MIN_SCALE={CALIB_MIN_SCALE}, CALIB_MAX_SCALE={CALIB_MAX_SCALE}, CALIB_USE_GLM_BRIDGE={CALIB_USE_GLM_BRIDGE},",
            f"CALIB_LOCAL_RADIUS_KM={CALIB_LOCAL_RADIUS_KM}, CALIB_TRAP_BIAS_RADIUS_KM={CALIB_TRAP_BIAS_RADIUS_KM}",
        )
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))