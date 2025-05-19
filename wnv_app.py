import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from shapely.geometry import Point, MultiPoint
import folium
from datetime import datetime

# === 1. Load & clean historical data ===
df = pd.read_csv("wnv_data.csv")
df = df.dropna(subset=['Latitude', 'Longitude', 'RESULT', 'SEASON YEAR', 'WEEK'])
df['RESULT'] = LabelEncoder().fit_transform(df['RESULT'])  # 1 = positive, 0 = negative

# === 2. Select positives around current week (±2 weeks), with year‐based weighting ===
current_week = datetime.now().isocalendar()[1]
window = 2
mask = df['WEEK'].between(current_week - window, current_week + window)
positives = df[(df['RESULT'] == 1) & mask]

if positives.empty:
    print("⚠️ No positives in ±2-week window; using all positives.")
    positives = df[df['RESULT'] == 1]

# Coordinates and weights for KDE
coords_pos = positives[['Latitude', 'Longitude']].values
min_year = positives['SEASON YEAR'].min()
weights_pos = (positives['SEASON YEAR'] - min_year + 1).values

# === 3. Build the Chicago grid ===
lat_min, lat_max = 41.6445, 42.0230
lon_min, lon_max = -87.9409, -87.5237
step = 0.005  # ~500m grid

grid_pts = [
    [lat, lon]
    for lat in np.arange(lat_min, lat_max + step, step)
    for lon in np.arange(lon_min, lon_max + step, step)
]
grid_df = pd.DataFrame(grid_pts, columns=['Latitude', 'Longitude'])

# === 4. Clip out Lake Michigan by convex hull of all historic points ===
hull = MultiPoint([
    Point(lon, lat)
    for lat, lon in zip(df['Latitude'], df['Longitude'])
]).convex_hull

inside = [
    hull.contains(Point(lon, lat))
    for lat, lon in zip(grid_df['Latitude'], grid_df['Longitude'])
]
grid_df = grid_df[inside].reset_index(drop=True)

# === 5. Fit a weighted KDE on the positive samples ===
kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
kde.fit(coords_pos, sample_weight=weights_pos)

log_dens = kde.score_samples(grid_df[['Latitude', 'Longitude']])
grid_df['risk'] = np.exp(log_dens)

# === 6. Prepare color interpolation ===
min_risk, max_risk = grid_df['risk'].min(), grid_df['risk'].max()

def interpolate_risk_color(r, r_min, r_max):
    """Map r∈[r_min,r_max] to a hex color from pale yellow → red."""
    if r_max == r_min:
        return '#FF0000'
    ratio = (r - r_min) / (r_max - r_min)
    red = 255
    green = int(255 * (1 - ratio))
    blue = int(224 * (1 - ratio))
    return f'#{red:02X}{green:02X}{blue:02X}'

# === 7. Cluster‐based trap placement ===
def get_trap_cluster_centers(n):
    """
    Use weighted KMeans to find n cluster centers of high‐risk areas.
    Sample weights = grid_df['risk'].
    """
    coords = grid_df[['Latitude', 'Longitude']].values
    sample_w = grid_df['risk'].values
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(coords, sample_weight=sample_w)
    centers = kmeans.cluster_centers_
    # Each center is [lat, lon]
    return [(lat, lon) for lat, lon in centers]

# === 8. Map generation ===
def generate_map(num_traps=5, threshold=None):
    """
    num_traps: number of cluster‐center traps.
    threshold: if set, hide any grid cell with risk < threshold.
    """
    m = folium.Map(
        location=[41.8781, -87.6298],
        zoom_start=11,
        tiles='CartoDB positron',
        max_bounds=True
    )

    # Optionally filter out low‐risk cells
    plot_df = grid_df if threshold is None else grid_df[grid_df['risk'] >= threshold]

    # Add clickable risk circles
    for _, row in plot_df.iterrows():
        ratio = (row['risk'] - min_risk) / (max_risk - min_risk) if max_risk != min_risk else 1
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=interpolate_risk_color(row['risk'], min_risk, max_risk),
            fill=True,
            fill_opacity=0.2 + 0.6 * ratio,
            weight=0,
            popup=f"Risk: {row['risk']:.4f}"
        ).add_to(m)

    # Place traps at cluster centers
    for i, (lat, lon) in enumerate(get_trap_cluster_centers(num_traps), start=1):
        folium.Marker(
            location=[lat, lon],
            popup=f"Trap #{i}",
            icon=folium.DivIcon(html=f'''
                <div style="
                  font-size:12px;
                  color:#fff;
                  background:#000;
                  padding:2px 4px;
                  border-radius:3px;
                ">{i}</div>
            ''')
        ).add_to(m)

    # Constrain view to Chicago
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

    m.save("wnv_chicago_map.html")
    print("Saved → wnv_chicago_map.html")

# === 9. Run it! ===
generate_map(num_traps=5, threshold=None)
