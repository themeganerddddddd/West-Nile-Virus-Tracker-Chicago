import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

# Load data
df = pd.read_csv('west_nile_virus_data_with_weather.csv', parse_dates=['date'])

# Verify columns
print(df.columns)

# Correctly specify the mosquito species column
df = df.dropna(subset=['SPECIES'])  # replace with your actual column name

# Features for species classification
features = [
    'Latitude', 'Longitude', 'Year', 'Week',
    'temp', 'humidity', 'rain', 'wind_speed'
    # Add other relevant features
]

X = df[features].fillna(0)
y = df['SPECIES']  # Replace this with your actual column name

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and calibrate model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
model_sp_calibrated = CalibratedClassifierCV(base_model, cv=5)
model_sp_calibrated.fit(X_train, y_train)

# Evaluate
accuracy = model_sp_calibrated.score(X_test, y_test)
print(f"Species model accuracy: {accuracy:.4f}")

# Save model
dump(model_sp_calibrated, 'model_sp_calibrated.joblib')
print("Model saved successfully.")
