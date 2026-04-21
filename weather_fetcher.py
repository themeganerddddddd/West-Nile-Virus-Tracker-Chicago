"""
weather_fetcher.py

This script reads the raw West Nile virus data CSV,
fetches historical weather for Chicago using Meteostat,
merges them on the collection date, and writes out
an enriched CSV for downstream use in the Flask app.
"""
import pandas as pd
from datetime import datetime
from meteostat import Point, Daily, Hourly

# --- CONFIGURATION ---
RAW_CSV = 'west_nile_virus_data.csv'
OUTPUT_CSV = 'west_nile_virus_data_with_weather.csv'
# Chicago central point
CHICAGO_CENTER = Point(41.881832, -87.623177)

# --- FUNCTIONS ---
def load_wnv_data(path):
    df = pd.read_csv(path)
    # normalize column names if needed
    if 'SEASON YEAR' in df.columns and 'WEEK' in df.columns:
        df.rename(columns={'SEASON YEAR':'Year','WEEK':'Week'}, inplace=True)
    # compute Monday date of ISO week
    df['date'] = df.apply(
        lambda r: datetime.fromisocalendar(int(r['Year']), int(r['Week']), 1),
        axis=1
    )
    return df


def fetch_weather(df):
    start, end = df['date'].min(), df['date'].max()
    # fetch daily aggregated data
    daily = Daily(CHICAGO_CENTER, start, end).fetch().reset_index()
    daily = daily[['time','tavg','prcp','wspd']].rename(
        columns={'time':'date','tavg':'temp','prcp':'rain','wspd':'wind_speed'}
    )
    # fetch hourly humidity and average per day
    humidity = []
    for d in daily['date']:
        hourly = Hourly(CHICAGO_CENTER, d, d).fetch()
        humidity.append(hourly.get('rhum', pd.Series()).mean())
    daily['humidity'] = humidity
    return daily


def main():
    print(f"Loading raw data from {RAW_CSV}...")
    wnv = load_wnv_data(RAW_CSV)
    print(f"Fetching weather from {wnv['date'].min().date()} to {wnv['date'].max().date()}...")
    weather = fetch_weather(wnv)
    print("Merging weather into WNV data...")
    merged = wnv.merge(weather, on='date', how='left')
    # fill missing with median
    for col in ['temp','rain','wind_speed','humidity']:
        merged[col].fillna(merged[col].median(), inplace=True)
    print(f"Writing enriched data to {OUTPUT_CSV}...")
    merged.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == '__main__':
    main()
