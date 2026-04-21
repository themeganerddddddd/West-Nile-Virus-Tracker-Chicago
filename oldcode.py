# Required installations: 
# pip install pandas geopandas matplotlib openweathermap-client scikit-learn neal

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel
import requests
import time

# -------------------- CONFIG --------------------
API_KEY = '79e121d53c70ecf6ebd8a573f98d702e'  # Replace with your key
CITY = 'Chicago'
GRID_SIZE = 0.01  # degrees ~1km

# -------------------- STEP 1: Load Data --------------------
# Assume CSV file downloaded from https://data.cityofchicago.org/Public-Safety/West-Nile-Virus-Cases-in-Humans-2005-2023/7ibz-npxh
wnv_data = pd.read_csv('west_nile_virus_data.csv')  # Historical mosquito/WNV data

# Filter and process
wnv_data = wnv_data.dropna(subset=['Latitude', 'Longitude'])
# Assuming the column with the result is named 'RESULT'
if 'RESULT' in wnv_data.columns:
    wnv_data['WnvPresent'] = wnv_data['RESULT'].map({'negative': 0, 'positive': 1})
    wnv_data['WnvPresent'] = wnv_data['WnvPresent'].astype(int)
else:
    print("Column 'RESULT' not found in the data.")

# -------------------- STEP 2: Add Real-Time Weather --------------------
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp'], data['main']['humidity'], data['wind']['speed']
    else:
        return np.nan, np.nan, np.nan

sample_lat, sample_lon = wnv_data.iloc[0][['Latitude', 'Longitude']]
temp, humidity, wind = get_weather(sample_lat, sample_lon)
wnv_data['Temp'] = temp
wnv_data['Humidity'] = humidity
wnv_data['Wind'] = wind

# -------------------- STEP 3: Train Risk Model --------------------
features = ['Latitude', 'Longitude', 'Temp', 'Humidity', 'Wind']
target = 'WnvPresent'

X_train, X_test, y_train, y_test = train_test_split(wnv_data[features], wnv_data[target], test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))

# -------------------- STEP 4: Generate Grid for Risk Prediction --------------------
lat_min, lat_max = wnv_data['Latitude'].min(), wnv_data['Latitude'].max()
lon_min, lon_max = wnv_data['Longitude'].min(), wnv_data['Longitude'].max()

lat_range = np.arange(lat_min, lat_max, GRID_SIZE)
lon_range = np.arange(lon_min, lon_max, GRID_SIZE)

grid_points = [(lat, lon) for lat in lat_range for lon in lon_range]
weather_data = [get_weather(lat, lon) for lat, lon in grid_points]

grid_df = pd.DataFrame(grid_points, columns=['Latitude', 'Longitude'])
grid_df[['Temp', 'Humidity', 'Wind']] = weather_data
risk_scores = model.predict_proba(grid_df[features])[:, 1]
grid_df['Risk'] = risk_scores

# -------------------- STEP 5: Optimize Trap Placement (QUBO) --------------------
# Define QUBO: place 1 trap in highest risk areas, penalize overlapping coverage
trap_budget = 5
num_points = len(grid_df)
coverage_radius = 0.02  # degrees

# Linear terms: minimize the risk score
linear = {i: -risk_scores[i] for i in range(num_points)}

# Quadratic terms: penalize overlapping traps within coverage radius
quadratic = {}
for i in range(num_points):
    for j in range(i + 1, num_points):
        lat1, lon1 = grid_points[i]
        lat2, lon2 = grid_points[j]
        dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
        if dist < coverage_radius:
            quadratic[(i, j)] = 0.5  # Penalty for overlap

# Create the Binary Quadratic Model (BQM)
bqm = BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')

# Constraint: total traps placed = trap_budget
# Add slack penalty to enforce the trap budget constraint
slack_weight = 3.0
for i in range(num_points):
    linear[i] += slack_weight * 2  # Update the linear term directly

# Rebuild the BQM with updated linear terms
bqm = BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')

# Sample using Simulated Annealing
sampler = SimulatedAnnealingSampler()
response = sampler.sample(bqm, num_reads=100)
best = response.first.sample
selected_indices = [i for i, val in best.items() if val == 1]

print("Selected trap indices:", selected_indices)

# -------------------- STEP 6: Visualize --------------------
plt.figure(figsize=(10, 8))
plt.scatter(grid_df['Longitude'], grid_df['Latitude'], c=grid_df['Risk'], cmap='Reds', s=40, alpha=0.5)
selected = grid_df.iloc[selected_indices]
plt.scatter(selected['Longitude'], selected['Latitude'], c='blue', s=100, label='Trap Location')
plt.title('Optimal Mosquito Trap Placement (Risk Heatmap)')
plt.colorbar(label='WNV Risk')
plt.legend()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.tight_layout()
plt.show()


