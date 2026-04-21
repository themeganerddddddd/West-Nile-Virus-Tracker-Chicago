import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import dump

# Load your data
df = pd.read_csv('west_nile_virus_data_with_weather.csv', parse_dates=['date'])

# Preprocess data to create the target variable (positive cases)
df = df.dropna(subset=['RESULT'])
df['target'] = (df['RESULT'].str.contains('Positive')).astype(int)

# Select features used in your model (ensure they match app.py features)
features = [
    'Latitude', 'Longitude', 'Year', 'Week',
    'temp', 'humidity', 'rain', 'wind_speed'
    # Add other features as needed from your app.py file
]

X = df[features].fillna(0)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the HistGradientBoostingClassifier
model_inf_ensemble = HistGradientBoostingClassifier(max_iter=200, random_state=42)
model_inf_ensemble.fit(X_train, y_train)

# Evaluate the model accuracy (optional, but recommended)
accuracy = model_inf_ensemble.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Save the trained model
dump(model_inf_ensemble, 'model_inf_ensemble.joblib')
print("Model saved successfully as 'model_inf_ensemble.joblib'.")
