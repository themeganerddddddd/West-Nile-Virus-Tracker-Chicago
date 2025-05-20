from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # ✅ NEW
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from shapely.geometry import shape, Point
import alphashape
import dimod
from dimod.reference.samplers import SimulatedAnnealingSampler
from itertools import combinations
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)
CORS(app)  # ✅ Allow CORS from any frontend domain (like onrender.com)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ = radians(lat2 - lat1)
    dλ = radians(lon2 - lon1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def train_models(csv_path="west_nile_virus_data.csv"):
    df = pd.read_csv(csv_path).dropna(
        subset=['Latitude','Longitude','RESULT','SEASON YEAR','WEEK']
    )
    df['RESULT_BIN'] = (df['RESULT']=='positive').astype(int)

    df['time']     = df['SEASON YEAR'] + (df['WEEK'] - 1)/52.0
    df['week_sin'] = np.sin(2*np.pi*(df['WEEK'] - 1)/52.0)
    df['week_cos'] = np.cos(2*np.pi*(df['WEEK'] - 1)/52.0)

    features = ['Latitude','Longitude','time','week_sin','week_cos']
    X, y = df[features], df['RESULT_BIN']

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X, y)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64,32),
        activation='relu',
        max_iter=200,
        random_state=42
    )
    mlp_model.fit(X, y)

    points = [(lon, lat) for lat, lon in zip(df['Latitude'], df['Longitude'])]
    concave = alphashape.alphashape(points, alpha=0.02)

    return xgb_model, mlp_model, concave

def build_grid(xgb_model, mlp_model, concave,
               year, week,
               lat_bounds, lon_bounds,
               step=0.005):
    time_now = year + (week - 1)/52.0
    sin_w    = sin(2*np.pi*(week - 1)/52.0)
    cos_w    = cos(2*np.pi*(week - 1)/52.0)

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    rows = []
    for lat in np.arange(lat_min, lat_max+step, step):
        for lon in np.arange(lon_min, lon_max+step, step):
            rows.append((lat, lon, time_now, sin_w, cos_w))
    grid_df = pd.DataFrame(
        rows,
        columns=['Latitude','Longitude','time','week_sin','week_cos']
    )

    mask = [
        concave.contains(Point(lon, lat))
        for lat, lon in zip(grid_df['Latitude'], grid_df['Longitude'])
    ]
    grid_df = grid_df[mask].reset_index(drop=True)

    Xg = grid_df[['Latitude','Longitude','time','week_sin','week_cos']]
    grid_df['risk_xgb'] = xgb_model.predict_proba(Xg)[:,1]
    grid_df['risk_mlp'] = mlp_model.predict_proba(Xg)[:,1]
    grid_df['risk']     = 0.5 * (grid_df['risk_xgb'] + grid_df['risk_mlp'])

    return grid_df, grid_df['risk'].min(), grid_df['risk'].max()

def select_traps(sub_df, K, K1=3, C=0.5, lam=1.0, M=200, penalty=10.0):
    candidates = sub_df.nlargest(M, 'risk').reset_index(drop=True)
    risks      = candidates['risk'].values
    N          = len(candidates)

    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.Vartype.BINARY)
    for i, r in enumerate(risks):
        bqm.add_variable(i, C - r)
    for i, j in combinations(range(N), 2):
        lat1, lon1 = candidates.loc[i, ['Latitude','Longitude']]
        lat2, lon2 = candidates.loc[j, ['Latitude','Longitude']]
        d = haversine(lat1, lon1, lat2, lon2)
        if d > 0:
            bqm.add_interaction(i, j, lam / d)

    A = penalty
    lin_pen = A * (1 - 2*K1)
    for i in range(N):
        bqm.add_variable(i, lin_pen)
    quad_pen = 2 * A
    for i, j in combinations(range(N), 2):
        bqm.add_interaction(i, j, quad_pen)

    sampler = SimulatedAnnealingSampler()
    ss = sampler.sample(bqm, num_reads=50)
    best = ss.first.sample
    chosen = [i for i, bit in best.items() if bit == 1]

    selected = set(chosen)
    for _ in range(K - len(selected)):
        i_best = max(
            (i for i in range(N) if i not in selected),
            key=lambda i: risks[i]
        )
        selected.add(i_best)

    return candidates.loc[list(selected), ['Latitude','Longitude']].values.tolist()

# ——————————————————————————————————————————————

@app.route('/')
def index():
    year = request.args.get('year', type=int, default=datetime.now().year)
    week = request.args.get('week', type=int, default=datetime.now().isocalendar()[1])
    return render_template('index.html', sel_year=year, sel_week=week)

@app.route('/grid')
def grid_data():
    year = request.args.get('year', type=int, default=datetime.now().year)
    week = request.args.get('week', type=int, default=datetime.now().isocalendar()[1])
    xgb_model, mlp_model, concave = train_models()
    grid_df, min_risk, max_risk = build_grid(
        xgb_model, mlp_model, concave,
        year, week,
        lat_bounds=(41.6445, 42.0230),
        lon_bounds=(-87.9409, -87.5237),
        step=0.005
    )
    return jsonify(
        grid=grid_df[['Latitude', 'Longitude', 'risk']].to_dict(orient='records'),
        min_risk=min_risk,
        max_risk=max_risk
    )

@app.route('/traps', methods=['POST'])
def traps():
    data = request.get_json()
    K, year, week = data['num_traps'], data['year'], data['week']
    app.logger.info(f"traps requested K={K}, year={year}, week={week}")
    app.logger.info(f"polygon geojson: {data['polygon']}")
    poly = shape(data['polygon'])

    xgb_model, mlp_model, concave = train_models()
    grid_df, _, _ = build_grid(
        xgb_model, mlp_model, concave,
        year, week,
        lat_bounds=(41.6445, 42.0230),
        lon_bounds=(-87.9409, -87.5237),
        step=0.005
    )

    mask = [
        poly.covers(Point(lon, lat))
        for lat, lon in zip(grid_df['Latitude'], grid_df['Longitude'])
    ]
    inside = sum(mask)
    app.logger.info(f"{inside} grid points inside polygon")

    sub = grid_df[mask].copy()
    centers = select_traps(sub, K) if inside > 0 else []
    return jsonify(centers=centers)

if __name__ == '__main__':
    app.run(debug=True)
