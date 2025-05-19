import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# -------------------- LOAD DATA --------------------
df = pd.read_csv("west_nile_virus_data.csv")
df = df.dropna(subset=["Latitude", "Longitude"])

# -------------------- CLEANING & FEATURE ENGINEERING --------------------
# Encode RESULT column
df["WnvPresent"] = df["RESULT"].map({"positive": 1, "negative": 0}).astype(int)

# Combine Season Year and Week
df["YearWeek"] = df["SEASON YEAR"].astype(str) + "-" + df["WEEK"].astype(str).str.zfill(2)

# Convert to numerical for training
df["WeekNum"] = df["WEEK"]
df["Year"] = df["SEASON YEAR"]

# Feature columns and target
features = ["Latitude", "Longitude", "Year", "WeekNum"]
target = "WnvPresent"

# -------------------- TRAIN MODEL --------------------
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))

# -------------------- GENERATE PREDICTIONS FOR CURRENT WEEK --------------------
# Define grid over Chicago
lat_range = np.arange(df["Latitude"].min(), df["Latitude"].max(), 0.01)
lon_range = np.arange(df["Longitude"].min(), df["Longitude"].max(), 0.01)

grid_points = [(lat, lon) for lat in lat_range for lon in lon_range]

# Set prediction for current year/week
now = datetime.now()
current_year = 2025
current_week = now.isocalendar().week

predict_df = pd.DataFrame(grid_points, columns=["Latitude", "Longitude"])
predict_df["Year"] = current_year
predict_df["WeekNum"] = current_week

# Predict risk probability
probs = model.predict_proba(predict_df[features])[:, 1]
predict_df["Risk"] = probs

# -------------------- MAP RISK HOTSPOTS --------------------
# Filter high-risk predictions
threshold = 0.5  # adjust as needed
high_risk_df = predict_df[predict_df["Risk"] >= threshold]

fig = px.scatter_map(
    high_risk_df,
    lat="Latitude",
    lon="Longitude",
    color="Risk",
    size="Risk",
    color_continuous_scale="YlOrRd",
    zoom=10,
    height=600,
    map_style="carto-positron",
    title=f"Predicted WNV Risk Hotspots – Week {current_week} of 2025"
)
fig.show()
