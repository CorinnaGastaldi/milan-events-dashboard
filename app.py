import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point, box
import numpy as np

st.set_page_config(page_title="Analisi Eventi Milano", layout="wide")

st.title("📊 Analisi spaziale eventi a Milano")

# --- 1️⃣ Carica dataset ---
data_file = st.file_uploader("Carica file CSV con eventi", type=["csv"])
if not data_file:
    st.stop()

df = pd.read_csv(data_file)
df = df.dropna(subset=['lat', 'long'])
df['year'] = pd.to_datetime(df['data_inizio'], errors='coerce').dt.year
df = df[~df['year'].isin([2016, 2017, 2018, 2025])]

# --- 2️⃣ Carica shapefile zone ---
shp_file = st.file_uploader("Carica shapefile (.shp o .zip)", type=["shp", "zip"])
if not shp_file:
    st.stop()

zones_gdf = gpd.read_file(shp_file).to_crs(epsg=3857)

# --- 3️⃣ Conversione eventi in GeoDataFrame ---
geometry = [Point(xy) for xy in zip(df['long'], df['lat'])]
events_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

# --- 4️⃣ Bounding box Milano ---
bbox_wgs84 = box(9.10, 45.40, 9.30, 45.55)
bbox = gpd.GeoSeries([bbox_wgs84], crs="EPSG:4326").to_crs(epsg=3857)
xmin, ymin, xmax, ymax = bbox.total_bounds

# --- 5️⃣ Selezione anno ---
years = sorted(events_gdf['year'].dropna().unique())
year = st.selectbox("Seleziona un anno", years)
events_year = events_gdf[events_gdf['year'] == year]

# --- 6️⃣ Calcolo densità eventi per zona ---
joined = gpd.sjoin(events_year, zones_gdf, how="inner", predicate='within')
counts = joined.groupby('area_id').size().rename('event_count')
zones_year = zones_gdf.set_index('area_id').join(counts).fillna(0)
zones_year['area_km2'] = zones_year.geometry.area / 1e6
zones_year['density'] = zones_year['event_count'] / zones_year['area_km2']

# --- 7️⃣ Visualizzazione mappa ---
fig, ax = plt.subplots(figsize=(10, 8))
zones_year.plot(
    column='density',
    ax=ax,
    cmap='cividis',
    scheme='fisher_jenks',
    k=min(20, zones_year['density'].nunique()),
    legend=True
)
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title(f"Densità eventi - {year}")
ax.axis('off')

st.pyplot(fig)
