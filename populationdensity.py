import pandas as pd
import geopandas as gpd

# Load block group shapefile
bg = gpd.read_file("tl_2022_17_bg.shp")

# Load population data and match on GEOID prefix
pop = pd.read_csv("Population_by_2010_Census_Block_20250617.csv", dtype={"CENSUS BLOCK FULL": str})
pop["BG_GEOID"] = pop["CENSUS BLOCK FULL"].str[:12]
pop_grouped = pop.groupby("BG_GEOID", as_index=False)["TOTAL POPULATION"].sum()
pop_grouped.rename(columns={"TOTAL POPULATION": "block_pop"}, inplace=True)

# Ensure consistent GEOID format
bg["GEOID"] = bg["GEOID"].astype(str).str.zfill(12)
bg = bg.merge(pop_grouped, left_on="GEOID", right_on="BG_GEOID", how="left")

# Project to EPSG:3857 for accurate area and centroid
bg_proj = bg.to_crs(epsg=3857)
bg_proj["area_km2"] = bg_proj.geometry.area / 1e6

# Compute centroids in projected CRS, then convert to lat/lon
centroids = bg_proj.geometry.centroid.to_crs(epsg=4326)
bg["Latitude"] = centroids.y
bg["Longitude"] = centroids.x

# Add area and compute density
bg["area_km2"] = bg_proj["area_km2"]
bg["pop_density"] = bg["block_pop"] / bg["area_km2"]

# Output CSV
bg[["GEOID", "Latitude", "Longitude", "block_pop", "area_km2", "pop_density"]].to_csv("pop_density.csv", index=False)

print("✅ Population density saved to 'pop_density.csv'")
