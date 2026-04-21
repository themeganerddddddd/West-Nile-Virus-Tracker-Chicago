import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

INPUT_CSV = "west_nile_virus_data.csv"
OUTPUT_CSV = "wnv_gridlevel_features.csv"

# ---- Tunable knobs ----
RADIUS_KM = 1.0         # neighborhood radius for "nearby trap history"
LAGS_WEEKS = [1, 2]     # use last 1 and last 2 weeks history

def to_iso_year_week(dt: pd.Timestamp):
    iso = dt.isocalendar()
    return int(iso.year), int(iso.week)

def haversine_radius_radians(km: float) -> float:
    return km / 6371.0

def safe_upper(s):
    return str(s).upper().strip() if pd.notnull(s) else ""

def main():
    df = pd.read_csv(INPUT_CSV)

    # --- standardize column names (your file uses spaces/case) ---
    # Expecting at least:
    # 'Date', 'Latitude', 'Longitude', 'Trap', 'Species', 'Result', 'Number of Mosquitoes'
    # Your file may use 'TEST DATE', 'TRAP', 'SPECIES', 'RESULT', 'NUMBER OF MOSQUITOES', etc.
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Try to locate important columns robustly
    def pick_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"Missing columns; tried: {candidates}. Available: {list(df.columns)}")

    col_date   = pick_col(["TEST DATE", "Date", "date"])
    col_lat    = pick_col(["Latitude", "LATITUDE"])
    col_lon    = pick_col(["Longitude", "LONGITUDE"])
    col_trap   = pick_col(["TRAP", "Trap"])
    col_species= pick_col(["SPECIES", "Species"])
    col_result = pick_col(["RESULT", "Result"])
    col_mosq   = pick_col(["NUMBER OF MOSQUITOES", "Number of Mosquitoes", "NUM MOSQUITOES"])

    # Parse date
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.dropna(subset=[col_date, col_lat, col_lon, col_result])

    # Clean
    df["Latitude"]  = pd.to_numeric(df[col_lat], errors="coerce")
    df["Longitude"] = pd.to_numeric(df[col_lon], errors="coerce")
    df["TRAP"]      = df[col_trap].astype(str)
    df["SPECIES"]   = df[col_species].astype(str)
    df["RESULT"]    = df[col_result].astype(str)
    df["mosq_count"]= pd.to_numeric(df[col_mosq], errors="coerce").fillna(0).clip(lower=0)

    # Target
    df["target"] = df["RESULT"].str.contains("Positive", case=False, na=False).astype(int)

    # ISO Year/Week from date (more robust than relying on provided columns)
    iso = df[col_date].dt.isocalendar()
    df["Year"] = iso["year"].astype(int)
    df["Week"] = iso["week"].astype(int)

    # Seasonality encoding
    df["day_of_year"] = df[col_date].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*df["day_of_year"]/365.0)
    df["cos_doy"] = np.cos(2*np.pi*df["day_of_year"]/365.0)

    # Log mosquito count (very strong signal; stable at grid-level via neighborhood aggregates)
    df["log_mosq"] = np.log1p(df["mosq_count"])

    # ---- Build per-week trap summary (so neighborhood lookups are cheaper and consistent) ----
    # For each trap-week, aggregate:
    # - mean lat/lon (trap is fixed, but just in case)
    # - positives count
    # - total samples (rows)
    # - mosquito count sum
    wk = (
        df.groupby(["Year","Week","TRAP"], as_index=False)
          .agg(
              Latitude=("Latitude","mean"),
              Longitude=("Longitude","mean"),
              pos_count=("target","sum"),
              n_samples=("target","size"),
              mosq_sum=("mosq_count","sum"),
          )
    )
    wk["pos_rate"] = wk["pos_count"] / wk["n_samples"].replace(0, np.nan)
    wk["pos_rate"] = wk["pos_rate"].fillna(0)

    # ---- Precompute BallTrees by week for fast neighbor queries ----
    # Build dict: (Year,Week) -> (BallTree, arrays)
    week_index = {(int(y),int(w)) for y,w in wk[["Year","Week"]].itertuples(index=False, name=None)}
    trees = {}

    for (y,w) in sorted(week_index):
        sub = wk[(wk["Year"]==y) & (wk["Week"]==w)].copy()
        if len(sub) == 0:
            continue
        coords_rad = np.radians(sub[["Latitude","Longitude"]].values)
        tree = BallTree(coords_rad, metric="haversine")
        trees[(y,w)] = {
            "tree": tree,
            "Latitude": sub["Latitude"].values,
            "Longitude": sub["Longitude"].values,
            "pos_count": sub["pos_count"].values.astype(float),
            "n_samples": sub["n_samples"].values.astype(float),
            "mosq_sum": sub["mosq_sum"].values.astype(float),
            "pos_rate": sub["pos_rate"].values.astype(float),
        }

    # ---- Attach neighborhood-history features to EACH ORIGINAL ROW ----
    # These are computable for arbitrary grid points later.
    radius_rad = haversine_radius_radians(RADIUS_KM)

    # Preallocate
    for L in LAGS_WEEKS:
        df[f"near_pos_count_lag{L}"] = 0.0
        df[f"near_pos_rate_lag{L}"]  = 0.0
        df[f"near_mosq_sum_lag{L}"]  = 0.0
        df[f"near_n_samples_lag{L}"] = 0.0

    # Map (Year,Week) -> rows indices for vectorized-ish handling
    # We'll loop over unique weeks, then process all rows in that week.
    for (y,w), idxs in df.groupby(["Year","Week"]).groups.items():
        # Coordinates of rows in this week
        rows = df.loc[idxs, ["Latitude","Longitude"]].values
        rows_rad = np.radians(rows)

        for L in LAGS_WEEKS:
            y_lag, w_lag = y, w - L
            # Handle ISO week rollover crudely: simplest approach is to use actual date-based shifting
            # But since we built Year/Week from date, we can instead compute lag week by date:
            # We'll do that more robustly:
            # Find the Monday of this ISO week, subtract L weeks -> target iso year/week
            # Compute once per group.
        # We'll do robust lag via date instead of year/week arithmetic.
        # Compute lag keys per group:
        any_date = df.loc[idxs, col_date].iloc[0]
        # get Monday of this ISO week:
        monday = pd.Timestamp.fromisocalendar(int(y), int(w), 1)
        lag_keys = {}
        for L in LAGS_WEEKS:
            lag_monday = monday - pd.Timedelta(weeks=L)
            lag_y, lag_w = to_iso_year_week(lag_monday)
            lag_keys[L] = (lag_y, lag_w)

        for L in LAGS_WEEKS:
            key = lag_keys[L]
            if key not in trees:
                continue
            pack = trees[key]
            nbrs = pack["tree"].query_radius(rows_rad, r=radius_rad)

            # Aggregate neighbor stats for each row
            near_pos_count = np.zeros(len(nbrs), dtype=float)
            near_n_samples = np.zeros(len(nbrs), dtype=float)
            near_mosq_sum  = np.zeros(len(nbrs), dtype=float)

            for i, neigh_idx in enumerate(nbrs):
                if len(neigh_idx) == 0:
                    continue
                near_pos_count[i] = pack["pos_count"][neigh_idx].sum()
                near_n_samples[i] = pack["n_samples"][neigh_idx].sum()
                near_mosq_sum[i]  = pack["mosq_sum"][neigh_idx].sum()

            near_pos_rate = np.divide(
                near_pos_count,
                np.where(near_n_samples == 0, np.nan, near_n_samples)
            )
            near_pos_rate = np.nan_to_num(near_pos_rate, nan=0.0)

            df.loc[idxs, f"near_pos_count_lag{L}"] = near_pos_count
            df.loc[idxs, f"near_n_samples_lag{L}"] = near_n_samples
            df.loc[idxs, f"near_mosq_sum_lag{L}"] = near_mosq_sum
            df.loc[idxs, f"near_pos_rate_lag{L}"] = near_pos_rate

    # Extra stable transforms
    for L in LAGS_WEEKS:
        df[f"log_near_mosq_sum_lag{L}"] = np.log1p(df[f"near_mosq_sum_lag{L}"])

    # Keep only needed columns + labels for training
    out_cols = [
        "Latitude","Longitude","Year","Week",
        "sin_doy","cos_doy",
        "log_mosq",
        "near_pos_count_lag1","near_pos_rate_lag1","log_near_mosq_sum_lag1","near_n_samples_lag1",
        "near_pos_count_lag2","near_pos_rate_lag2","log_near_mosq_sum_lag2","near_n_samples_lag2",
        "SPECIES","target",
    ]

    feat = df[out_cols].copy()

    # Make sure numeric columns are numeric
    num_cols = [c for c in feat.columns if c not in ["SPECIES"]]
    feat[num_cols] = feat[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    feat.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}  rows={len(feat)}  cols={len(feat.columns)}")
    print("NOTE: Weather not included here (original CSV had none). You can add later.")

if __name__ == "__main__":
    main()
