import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import folium
from scipy.spatial import ConvexHull
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point, box
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.cluster import DBSCAN

st.set_page_config(
    page_title="Milan Events - Interactive Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# STYLE
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# PATH CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
EVENTS_FOLDER = os.path.join(BASE_DIR, "dataset_annuali_mi")
SHAPEFILE_PATH = os.path.join(BASE_DIR, "zonizzazione progetto", "Shapefile_uniBicocca.shp")

@st.cache_data(show_spinner="Loading events...")
def load_event_data(folder=EVENTS_FOLDER):
    """Load all event CSVs"""
    if not os.path.exists(folder):
        st.error(f"Folder not found: {folder}")
        st.info(f"Current directory: {os.getcwd()}")
        return None
    
    all_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    
    if not all_files:
        st.error(f"No CSV files found in: {folder}")
        return None
    
    dfs = []
    
    for f in all_files:
        try:
            year = int(f.split(".")[0])
            df = pd.read_csv(os.path.join(folder, f))
            
            if 'lat' not in df.columns or 'long' not in df.columns:
                st.sidebar.warning(f"{f} missing lat/long columns")
                continue
                
            df = df.dropna(subset=['lat', 'long'])
            df['year'] = year
            
            # Convert data_inizio to datetime if present
            if 'data_inizio' in df.columns:
                df['data_inizio'] = pd.to_datetime(df['data_inizio'], errors='coerce')
                df['month'] = df['data_inizio'].dt.month
                df['month_name'] = df['data_inizio'].dt.strftime('%B')
                df['season'] = df['month'].apply(get_season)
                df['quarter'] = df['data_inizio'].dt.quarter
            
            dfs.append(df)
        except Exception as e:
            st.sidebar.warning(f"Error in {f}: {str(e)[:50]}")
    
    if not dfs:
        st.error("No data loaded correctly")
        return None
    
    data = pd.concat(dfs, ignore_index=True)
    return data

def get_season(month):
    """Determine season from month"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def create_report_stats(data_subset, period_label):
    """Create statistics for report"""
    stats = {
        'period': period_label,
        'total_events': len(data_subset),
        'events_per_day': len(data_subset) / 30 if len(data_subset) > 0 else 0
    }
    
    if 'macro_categoria' in data_subset.columns:
        stats['top_category'] = data_subset['macro_categoria'].mode()[0] if len(data_subset) > 0 else 'N/A'
        stats['num_categories'] = data_subset['macro_categoria'].nunique()
    
    return stats

def generate_pdf_report(data, period_info):
    """Generate PDF report"""
    buffer = BytesIO()
    
    report_data = []
    
    for year in sorted(data['year'].unique()):
        year_data = data[data['year'] == year]
        
        if 'season' in data.columns:
            for season in ["Winter", "Spring", "Summer", "Autumn"]:
                season_data = year_data[year_data['season'] == season]
                
                report_row = {
                    'Year': year,
                    'Period': season,
                    'Total_Events': len(season_data),
                    'Daily_Average': len(season_data) / 90
                }
                
                if 'macro_categoria' in season_data.columns and len(season_data) > 0:
                    report_row['Main_Category'] = season_data['macro_categoria'].mode()[0]
                    report_row['Number_Categories'] = season_data['macro_categoria'].nunique()
                
                report_data.append(report_row)
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return buffer

@st.cache_data(show_spinner="Loading shapefile...")
def load_shapefile(path=SHAPEFILE_PATH):
    """Load zone shapefile"""
    if not os.path.exists(path):
        st.error(f"Shapefile not found: {path}")
        return None
    
    try:
        zones_gdf = gpd.read_file(path).to_crs(epsg=3857)
        return zones_gdf
    except Exception as e:
        st.error(f"Shapefile error: {e}")
        return None

def create_folium_map(subset, map_type="markers", colormap=None):
    """Create Folium map with different visualization types"""
    center_lat = subset['lat'].mean()
    center_lon = subset['long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    if map_type == "heatmap":
        heat_data = [[row['lat'], row['long']] for _, row in subset.iterrows()]
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
            
    else:
        if colormap and 'macro_categoria' in subset.columns:
            categories = subset['macro_categoria'].unique()
            colors = px.colors.qualitative.Set3[:len(categories)]
            color_map = dict(zip(categories, colors))
            
            for _, row in subset.iterrows():
                cat = row.get('macro_categoria', 'Other')
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=5,
                    popup=f"<b>{cat}</b><br>{row.get('titolo', '')}",
                    color=color_map.get(cat, '#3186cc'),
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
        else:
            for _, row in subset.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=4,
                    popup=row.get('macro_categoria', 'N/A'),
                    color="#3186cc",
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
    
    return m

def calculate_zone_density(events_gdf, zones_gdf, year):
    """Calculate event density per zone"""
    events_year = events_gdf[events_gdf['year'] == year]
    joined = gpd.sjoin(events_year, zones_gdf, how="inner", predicate='within')
    counts = joined.groupby('area_id').size().rename('event_count')
    
    zones_year = zones_gdf.set_index('area_id').join(counts).fillna(0)
    zones_year['area_km2'] = zones_year.geometry.area / 1e6
    zones_year['density'] = zones_year['event_count'] / zones_year['area_km2']
    
    return zones_year

# --- HEADER ---
st.markdown('<p class="main-header">Milan Events - Interactive Dashboard</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Configuration")

page = st.sidebar.radio("Select view:", 
                        ["Event Map", "Zone Analysis", "Statistics", "Clustering"])

# Load data
data = load_event_data()

if data is not None:
    years = sorted(data['year'].unique())
    
    # PAGE 1: EVENT MAP
    if page == "Event Map":
        st.sidebar.markdown("---")
        
        # Comparison mode
        compare_mode = st.sidebar.checkbox("Compare two periods", value=False)
        
        if not compare_mode:
            # STANDARD MODE
            time_filter = st.sidebar.radio("Time filter:", 
                                           ["Year", "Season", "Month", "Quarter"])
            
            if time_filter == "Year":
                selected_year = st.sidebar.slider("Select year:", 
                                                 min_value=int(years[0]), 
                                                 max_value=int(years[-1]), 
                                                 value=int(years[-1]))
                subset = data[data['year'] == selected_year]
                time_label = f"Year {selected_year}"
                
            elif time_filter == "Season":
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    selected_year = st.selectbox("Year:", sorted(data['year'].unique()))
                with col2:
                    season_options = ["Winter", "Spring", "Summer", "Autumn"]
                    selected_season = st.selectbox("Season:", season_options)
                
                subset = data[(data['year'] == selected_year) & (data['season'] == selected_season)]
                time_label = f"{selected_season} {selected_year}"
                
            elif time_filter == "Month":
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    selected_year = st.selectbox("Year:", sorted(data['year'].unique()))
                with col2:
                    if 'month' in data.columns:
                        month_names = {
                            1: "January", 2: "February", 3: "March", 4: "April",
                            5: "May", 6: "June", 7: "July", 8: "August",
                            9: "September", 10: "October", 11: "November", 12: "December"
                        }
                        available_months = sorted(data[data['year'] == selected_year]['month'].dropna().unique())
                        month_options = [month_names[int(m)] for m in available_months]
                        selected_month_name = st.selectbox("Month:", month_options)
                        selected_month = [k for k, v in month_names.items() if v == selected_month_name][0]
                
                subset = data[(data['year'] == selected_year) & (data['month'] == selected_month)]
                time_label = f"{selected_month_name} {selected_year}"
                
            else:  # Quarter
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    selected_year = st.selectbox("Year:", sorted(data['year'].unique()))
                with col2:
                    quarter_options = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]
                    selected_quarter_name = st.selectbox("Quarter:", quarter_options)
                    selected_quarter = int(selected_quarter_name[1])
                
                subset = data[(data['year'] == selected_year) & (data['quarter'] == selected_quarter)]
                time_label = f"{selected_quarter_name} {selected_year}"
            
            map_type = st.sidebar.selectbox("Map type:", 
                                            ["markers", "heatmap"],
                                            format_func=lambda x: {
                                                "markers": "Colored Markers",
                                                "heatmap": "Heatmap",
                                            }[x])
            
            show_categories = st.sidebar.checkbox("Color by category", value=True)
            
            # Category filter (if present)
            if 'macro_categoria' in subset.columns:
                categories = ['All'] + sorted(subset['macro_categoria'].unique().tolist())
                selected_cat = st.sidebar.selectbox("Filter category:", categories)
                
                if selected_cat != 'All':
                    subset = subset[subset['macro_categoria'] == selected_cat]
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Period", time_label)
            with col2:
                st.metric("Events displayed", len(subset))
            with col3:
                if 'macro_categoria' in subset.columns:
                    st.metric("Categories", subset['macro_categoria'].nunique())
            with col4:
                if time_filter == "Year":
                    avg_daily = len(subset) / 365
                elif time_filter == "Season":
                    avg_daily = len(subset) / 90
                elif time_filter == "Month":
                    avg_daily = len(subset) / 30
                else:
                    avg_daily = len(subset) / 90
                st.metric("Daily average", f"{avg_daily:.1f}")
            
            st.markdown("---")
            
            # Create and display map
            if len(subset) > 0:
                m = create_folium_map(subset, map_type, colormap=show_categories)
                st_folium(m, width=1400, height=700, returned_objects=[])
                
                # Event table
                with st.expander("View event table"):
                    display_cols = [col for col in ['titolo', 'macro_categoria', 'data_inizio', 'lat', 'long'] 
                                   if col in subset.columns]
                    st.dataframe(subset[display_cols].head(50), use_container_width=True)
            else:
                st.warning("No events found for selected filters")
        
        else:
            # COMPARISON MODE
            st.markdown("### Compare two periods")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Period 1")
                year1 = st.selectbox("Year 1:", sorted(data['year'].unique()), key="year1")
                if 'season' in data.columns:
                    season1 = st.selectbox("Season 1:", ["Winter", "Spring", "Summer", "Autumn"], key="season1")
                    subset1 = data[(data['year'] == year1) & (data['season'] == season1)]
                    label1 = f"{season1} {year1}"
                else:
                    subset1 = data[data['year'] == year1]
                    label1 = f"Year {year1}"
            
            with col2:
                st.markdown("#### Period 2")
                year2 = st.selectbox("Year 2:", sorted(data['year'].unique()), key="year2")
                if 'season' in data.columns:
                    season2 = st.selectbox("Season 2:", ["Winter", "Spring", "Summer", "Autumn"], key="season2")
                    subset2 = data[(data['year'] == year2) & (data['season'] == season2)]
                    label2 = f"{season2} {year2}"
                else:
                    subset2 = data[data['year'] == year2]
                    label2 = f"Year {year2}"
            
            # Comparative metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                diff = len(subset2) - len(subset1)
                st.metric(f"Events {label1}", len(subset1))
                st.metric(f"Events {label2}", len(subset2), delta=diff)
            
            with col2:
                if 'macro_categoria' in subset1.columns:
                    diff_cat = subset2['macro_categoria'].nunique() - subset1['macro_categoria'].nunique()
                    st.metric(f"Categories {label1}", subset1['macro_categoria'].nunique())
                    st.metric(f"Categories {label2}", subset2['macro_categoria'].nunique(), delta=diff_cat)
            
            with col3:
                pct_change = ((len(subset2) - len(subset1)) / len(subset1) * 100) if len(subset1) > 0 else 0
                st.metric("Change %", f"{pct_change:+.1f}%")
            
            st.markdown("---")
            
            # Side-by-side maps
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {label1}")
                if len(subset1) > 0:
                    m1 = create_folium_map(subset1, "heatmap", colormap=False)
                    st_folium(m1, width=650, height=500, returned_objects=[])
                else:
                    st.warning("No data for this period")
            
            with col2:
                st.markdown(f"#### {label2}")
                if len(subset2) > 0:
                    m2 = create_folium_map(subset2, "heatmap", colormap=False)
                    st_folium(m2, width=650, height=500, returned_objects=[])
                else:
                    st.warning("No data for this period")
            
            # Category comparison
            if 'macro_categoria' in data.columns:
                st.markdown("### Category comparison")
                
                cat1 = subset1['macro_categoria'].value_counts().head(10)
                cat2 = subset2['macro_categoria'].value_counts().head(10)
                
                comparison_df = pd.DataFrame({
                    label1: cat1,
                    label2: cat2
                }).fillna(0).astype(int)
                
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(name=label1, x=comparison_df.index, y=comparison_df[label1]))
                fig_compare.add_trace(go.Bar(name=label2, x=comparison_df.index, y=comparison_df[label2]))
                fig_compare.update_layout(barmode='group', title="Top 10 Categories - Comparison")
                st.plotly_chart(fig_compare, use_container_width=True)
    
    # PAGE 2: VODAFONE & EVENTS ANALYSIS
    elif page == "Zone Analysis":
        st.sidebar.markdown("---")
        st.markdown("### Spatial Analysis 2023: Vodafone Users vs Cultural Events")
        
        # --- Caricamento dati ---
        try:
            df_vodafone = pd.read_csv("ma-bicocca.presenze.csv")
            vodafone_available = True
        except:
            st.error("Vodafone data file not found: ma-bicocca.presenze.csv")
            vodafone_available = False
        
        zones_gdf = load_shapefile()
        
        if zones_gdf is not None and vodafone_available:
            
            analysis_type = st.sidebar.radio("Analysis type:", 
                                            ["Comparison", "Seasonal Comparison"])
            
            geometry = [Point(xy) for xy in zip(data['long'], data['lat'])]
            events_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
            
            #VODAFONE
            zones_gdf_3857 = zones_gdf.to_crs(epsg=3857)
        
            df_voda_filtered = df_vodafone[df_vodafone["LOCATION"].isin(zones_gdf["area_id"])]
            
            df_voda_sum = df_voda_filtered.groupby("LOCATION", as_index=False)["UNIQUE_USERS"].sum()
            df_voda_sum.rename(columns={"LOCATION": "area_id", "UNIQUE_USERS": "total_users"}, inplace=True)
            
            zones_voda = zones_gdf_3857.merge(df_voda_sum, on="area_id", how="left")
            zones_voda["total_users"] = zones_voda["total_users"].fillna(0)
            
            #EVENTI
            events_year = events_gdf[events_gdf['year'] == 2023]
            events_joined = gpd.sjoin(events_year, zones_gdf_3857, how="left", predicate='within')
            events_count = events_joined.groupby("area_id").size().reset_index(name="n_eventi")
            
            zones_events = zones_gdf_3857.merge(events_count, on="area_id", how="left")
            zones_events["n_eventi"] = zones_events["n_eventi"].fillna(0)
            
            #BOUNDING BOX 
            events_bounds = events_year.total_bounds  # [minx, miny, maxx, maxy]
            
            # Aggiungi un buffer per non tagliare ai margini (5% di margine)
            x_range = events_bounds[2] - events_bounds[0]
            y_range = events_bounds[3] - events_bounds[1]
        
            
            xmin = events_bounds[0] 
            xmax = events_bounds[2] 
            ymin = events_bounds[1] 
            ymax = events_bounds[3] 
            
            #COMPARISON: Side-by-side Vodafone vs Eventi
            if analysis_type == "Comparison":
                st.markdown("#### Spatial Distribution Comparison")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Vodafone Users", f"{zones_voda['total_users'].sum():,.0f}")
                with col2:
                    st.metric("Total Events", f"{zones_events['n_eventi'].sum():,.0f}")
                with col3:
                    st.metric("Analyzed Zones", len(zones_events))
                
                st.markdown("---")
                
                #Mappe side-by-side
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                fig.suptitle('Spatial Distribution: Vodafone Users vs Cultural Events - Milan 2023', 
                            fontsize=18, fontweight='bold', y=0.98)
                
                # Map 1: Vodafone Users
                zones_voda.plot(column='total_users',
                                ax=axes[0],
                                cmap='YlOrRd',
                                scheme='FisherJenks',
                                k=50,
                                legend=False,
                                linewidth=0.2,
                                edgecolor='white',
                                alpha=0.85)
                cx.add_basemap(axes[0], source=cx.providers.CartoDB.Positron, alpha=0.5)
                axes[0].set_xlim(xmin, xmax)
                axes[0].set_ylim(ymin, ymax)
                axes[0].axis('off')
                axes[0].set_title("Vodafone User Density", fontsize=14, pad=15)
                
                total_users = zones_voda['total_users'].sum()
                max_zone_users = zones_voda['total_users'].max()
                axes[0].text(0.02, 0.98, f'Total: {total_users:,.0f} users\nPeak zone: {max_zone_users:,.0f}',
                            transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Map 2: Eventi
                zones_events.plot(column='n_eventi',
                                ax=axes[1],
                                cmap='PuBuGn',
                                scheme='FisherJenks',
                                k=35,
                                legend=False,
                                linewidth=0.2,
                                edgecolor='white',
                                alpha=0.85)
                cx.add_basemap(axes[1], source=cx.providers.CartoDB.Positron, alpha=0.5)
                axes[1].set_xlim(xmin, xmax)
                axes[1].set_ylim(ymin, ymax)
                axes[1].axis('off')
                axes[1].set_title("Cultural Event Density", fontsize=14, pad=15)
                
                total_events = zones_events['n_eventi'].sum()
                max_zone_events = zones_events['n_eventi'].max()
                axes[1].text(0.02, 0.98, f'Total: {total_events:,.0f} events\nPeak zone: {max_zone_events:,.0f}',
                            transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            
            #SEASONAL COMPARISON
            elif analysis_type == "Seasonal Comparison":
                st.markdown("#### Seasonal Comparison: Vodafone Users vs Events")
                
                events_year_df = data[data['year'] == 2023].copy()
                
                if 'data_inizio' not in events_year_df.columns:
                    st.error("Column 'data_inizio' not found in events data")
                else:
                    events_year_df['data_inizio'] = pd.to_datetime(events_year_df['data_inizio'], errors='coerce')
                    events_year_df['month'] = events_year_df['data_inizio'].dt.month
                    
                    def get_season(month):
                        if pd.isna(month):
                            return None
                        if month in [12,1,2]: return 'Winter'
                        if month in [3,4,5]: return 'Spring'
                        if month in [6,7,8]: return 'Summer'
                        if month in [9,10,11]: return 'Autumn'
                        return None
                    
                    events_year_df['season'] = events_year_df['month'].apply(get_season)
                    events_year_df = events_year_df.dropna(subset=['season'])
                    
                    geometry_season = [Point(xy) for xy in zip(events_year_df['long'], events_year_df['lat'])]
                    events_gdf_season = gpd.GeoDataFrame(events_year_df, geometry=geometry_season, crs="EPSG:4326").to_crs(epsg=3857)
                    events_joined_season = gpd.sjoin(events_gdf_season, zones_gdf_3857, how="left", predicate='within')
                    
                    #VODAFONE 
                    df_voda_filtered['DATE_ID'] = pd.to_datetime(df_voda_filtered['DATE_ID'], format='%Y%m%d', errors='coerce')
                    df_voda_filtered['month'] = df_voda_filtered['DATE_ID'].dt.month
                    df_voda_filtered['season'] = df_voda_filtered['month'].apply(get_season)
                    df_voda_filtered = df_voda_filtered.dropna(subset=['season'])
                    
                    seasons_order = ['Winter','Spring','Summer','Autumn']
                    
                    fig, axes = plt.subplots(4, 2, figsize=(18, 28))
                    fig.suptitle('Seasonal Comparison: Vodafone Users vs Cultural Events - Milan 2023', 
                                fontsize=20, fontweight='bold', y=0.995)
                    
                    for idx, season in enumerate(seasons_order):
                        ax_vodafone = axes[idx, 0]
                        ax_events = axes[idx, 1]
                        
                        #VODAFONE 
                        voda_season = df_voda_filtered[df_voda_filtered['season'] == season]
                        voda_season_agg = voda_season.groupby("LOCATION", as_index=False)["UNIQUE_USERS"].sum()
                        voda_season_agg.rename(columns={"LOCATION": "area_id", "UNIQUE_USERS": "total_users"}, inplace=True)
                        
                        zones_voda_season = zones_gdf_3857.merge(voda_season_agg, on="area_id", how="left")
                        zones_voda_season["total_users"] = zones_voda_season["total_users"].fillna(0)
                        
                        zones_voda_season.plot(column='total_users',
                                            ax=ax_vodafone,
                                            cmap='YlOrRd',
                                            k=40,
                                            legend=False,
                                            linewidth=0.2,
                                            edgecolor='white',
                                            alpha=0.85)
                        cx.add_basemap(ax_vodafone, source=cx.providers.CartoDB.Positron, alpha=0.5)
                        ax_vodafone.set_xlim(xmin, xmax)
                        ax_vodafone.set_ylim(ymin, ymax)
                        ax_vodafone.axis('off')
                        
                        total_voda = zones_voda_season['total_users'].sum()
                        ax_vodafone.set_title(f"{season} - Vodafone Users\n(Total: {total_voda:,.0f})", 
                                            fontsize=12, fontweight='bold', pad=10)
                        
                        #EVENTI per stagione
                        season_events = events_joined_season[events_joined_season['season'] == season]
                        season_count = season_events.groupby("area_id").size().reset_index(name="n_eventi")
                        zones_season_events = zones_gdf_3857.merge(season_count, on="area_id", how="left")
                        zones_season_events["n_eventi"] = zones_season_events["n_eventi"].fillna(0)
                        
                        zones_season_events.plot(column='n_eventi',
                                            ax=ax_events,
                                            cmap='PuBuGn',
                                            k=35,
                                            legend=False,
                                            linewidth=0.2,
                                            edgecolor='white',
                                            alpha=0.85)
                        cx.add_basemap(ax_events, source=cx.providers.CartoDB.Positron, alpha=0.5)
                        ax_events.set_xlim(xmin, xmax)
                        ax_events.set_ylim(ymin, ymax)
                        ax_events.axis('off')
                        
                        total_evt = len(season_events)
                        ax_events.set_title(f"{season} - Cultural Events\n(Total: {total_evt})", 
                                        fontsize=12, fontweight='bold', pad=10)
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.99])
                    st.pyplot(fig)
                    plt.close()
    
    # PAGE 3: STATISTICS
    elif page == "Statistics":
        st.markdown("### Temporal Statistics")
        
        # Events per year
        yearly_counts = data.groupby('year').size().reset_index(name='count')
        
        fig1 = px.line(yearly_counts, x='year', y='count', 
                      title="Number of Events per Year",
                      labels={'year': 'Year', 'count': 'Number of Events'},
                      markers=True)
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Seasonal analysis
        if 'season' in data.columns:
            st.markdown("### Seasonal Trends")
            
            col1, col2 = st.columns(2)
            
            with col1:
                seasonal_counts = data.groupby(['year', 'season']).size().reset_index(name='count')
                fig_season = px.line(seasonal_counts, x='year', y='count', color='season',
                                    title="Events by Season Over Time",
                                    labels={'year': 'Year', 'count': 'Events'},
                                    color_discrete_map={
                                        "Winter": "#89CFF0",
                                        "Spring": "#FFB6C1",
                                        "Summer": "#FFD700",
                                        "Autumn": "#FF8C00"
                                    })
                st.plotly_chart(fig_season, use_container_width=True)
            
            with col2:
                season_totals = data['season'].value_counts()
                fig_season_pie = px.pie(values=season_totals.values, 
                                       names=season_totals.index,
                                       title="Total Distribution by Season",
                                       color=season_totals.index,
                                       color_discrete_map={
                                           "Winter": "#89CFF0",
                                           "Spring": "#FFB6C1",
                                           "Summer": "#FFD700",
                                           "Autumn": "#FF8C00"
                                       })
                st.plotly_chart(fig_season_pie, use_container_width=True)
        
        # Monthly analysis
        if 'month' in data.columns:
            st.markdown("### Monthly Trends")
            
            month_names_map = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }
            
            monthly_data = data.groupby(['year', 'month']).size().reset_index(name='count')
            monthly_data['month_label'] = monthly_data['month'].map(month_names_map)
            
            # Monthly heatmap
            pivot_month = monthly_data.pivot(index='month_label', columns='year', values='count').fillna(0)
            pivot_month = pivot_month.reindex([month_names_map[i] for i in range(1, 13)])
            
            fig_month_heat = px.imshow(pivot_month,
                                      labels=dict(x="Year", y="Month", color="Events"),
                                      title="Monthly Events Heatmap",
                                      aspect="auto",
                                      color_continuous_scale="YlOrRd")
            fig_month_heat.update_layout(height=500)
            st.plotly_chart(fig_month_heat, use_container_width=True)
            
            # Average events per month (all years)
            avg_by_month = data.groupby('month').size() / data['year'].nunique()
            avg_by_month.index = avg_by_month.index.map(month_names_map)
            
            fig_avg_month = px.bar(x=avg_by_month.index, y=avg_by_month.values,
                                  title="Average Events per Month (all years)",
                                  labels={'x': 'Month', 'y': 'Average Events'},
                                  color=avg_by_month.values,
                                  color_continuous_scale="Blues")
            st.plotly_chart(fig_avg_month, use_container_width=True)
        
        # Category distribution (if present)
        if 'macro_categoria' in data.columns:
            st.markdown("### Category Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cat_counts = data['macro_categoria'].value_counts().head(10)
                fig2 = px.bar(x=cat_counts.index, y=cat_counts.values,
                             title="Top 10 Categories",
                             labels={'x': 'Category', 'y': 'Number of Events'})
                fig2.update_xaxes(tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = px.pie(data, names='macro_categoria', 
                             title="Category Distribution (all years)",
                             hole=0.3)
                st.plotly_chart(fig3, use_container_width=True)
            
            # Categories by season
            if 'season' in data.columns:
                st.markdown("### Categories by Season")
                cat_season = data.groupby(['season', 'macro_categoria']).size().reset_index(name='count')
                top_cats = data['macro_categoria'].value_counts().head(8).index
                cat_season = cat_season[cat_season['macro_categoria'].isin(top_cats)]
                
                fig_cat_season = px.bar(cat_season, x='season', y='count', color='macro_categoria',
                                       title="Top 8 Categories by Season",
                                       labels={'season': 'Season', 'count': 'Events'},
                                       barmode='stack')
                st.plotly_chart(fig_cat_season, use_container_width=True)

    # PAGE 4: CLUSTERING
    elif page == "Clustering":
        st.sidebar.markdown("---")
        st.markdown("### Event Clustering Analysis")
        
        # Selezione algoritmo
        algorithm = st.sidebar.selectbox(
            "Algorithm:",
            options=["K-Means", "DBSCAN"],
            index=0
        )
        
        if algorithm == "K-Means":
            # ========== K-MEANS SECTION ==========
            st.sidebar.markdown("#### K-Means Parameters")
            
            # Select number of clusters
            cluster_options = []
            base_folder = BASE_DIR
            
            # Search for all eventi_kmeans_* folders
            for item in os.listdir(base_folder):
                if item.startswith("eventi_kmeans_") and os.path.isdir(os.path.join(base_folder, item)):
                    try:
                        n_clusters = int(item.split("_")[-1])
                        cluster_options.append(n_clusters)
                    except:
                        pass
            
            cluster_options = sorted(cluster_options)
            
            if not cluster_options:
                st.error("No clustering folders found")
            else:
                selected_n_clusters = st.sidebar.selectbox(
                    "Number of clusters:",
                    options=cluster_options,
                    format_func=lambda x: f"{x} clusters"
                )
                
                # List available years for selected cluster number
                kmeans_folder = os.path.join(BASE_DIR, f"eventi_kmeans_{selected_n_clusters}")
                kmeans_files = [f for f in os.listdir(kmeans_folder) if f.endswith(".csv")]
                available_years = sorted([int(f.split("_")[-1].split(".")[0]) for f in kmeans_files])
                
                selected_year = st.sidebar.selectbox("Select year:", available_years)
                
                # Load corresponding CSV
                kmeans_file = os.path.join(kmeans_folder, f"eventi_kmeans_{selected_n_clusters}_{selected_year}.csv")
                kmeans_data = pd.read_csv(kmeans_file)
                
                if 'lat' not in kmeans_data.columns or 'long' not in kmeans_data.columns or 'Cluster' not in kmeans_data.columns:
                    st.error("Selected file missing required columns (lat, long, Cluster).")
                else:
                    st.sidebar.markdown(f"Events loaded: {len(kmeans_data)}")
                    n_clusters = kmeans_data['Cluster'].nunique()
                    
                    # Option to show cluster report
                    show_report = st.sidebar.checkbox("Show detailed cluster report", value=False)
                    
                    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
                    colors = [f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})" for r, g, b, _ in colors]
                    
                    # General metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Year", selected_year)
                    with col2:
                        st.metric("Total events", len(kmeans_data))
                    with col3:
                        st.metric("Number of clusters", n_clusters)
                    with col4:
                        avg_events = len(kmeans_data) / n_clusters
                        st.metric("Avg events/cluster", f"{avg_events:.1f}")
                    
                    st.markdown("---")
                    
                    if not show_report:
                        # STANDARD MAP VIEW
                        map_clusters = folium.Map(
                            location=[kmeans_data['lat'].mean(), kmeans_data['long'].mean()],
                            zoom_start=11,
                            tiles="CartoDB positron"
                        )
                        
                        # Draw clusters
                        for cluster_num in range(n_clusters):
                            cluster_points = kmeans_data[kmeans_data['Cluster'] == cluster_num][['lat', 'long']].to_numpy()
                            
                            if len(cluster_points) >= 3:
                                try:
                                    hull = ConvexHull(cluster_points)
                                    hull_points = cluster_points[hull.vertices]
                                    folium.Polygon(
                                        locations=[(lat, lon) for lat, lon in hull_points],
                                        color=colors[cluster_num],
                                        fill=True,
                                        fill_color=colors[cluster_num],
                                        fill_opacity=0.3,
                                        popup=f"Cluster {cluster_num}"
                                    ).add_to(map_clusters)
                                except Exception as e:
                                    st.warning(f"ConvexHull failed for cluster {cluster_num}: {e}")
                            
                            # Individual points
                            for lat, lon in cluster_points:
                                folium.CircleMarker(
                                    location=(lat, lon),
                                    radius=2,
                                    color=colors[cluster_num],
                                    fill=True,
                                    fill_opacity=0.7
                                ).add_to(map_clusters)
                        
                        st_folium(map_clusters, width=1400, height=700)
                        
                        # Cluster statistics
                        st.markdown("### Cluster Statistics")
                        cluster_stats = []
                        
                        for cluster_num in range(n_clusters):
                            cluster_subset = kmeans_data[kmeans_data['Cluster'] == cluster_num]
                            
                            stat_row = {
                                'Cluster': cluster_num,
                                'Event Count': len(cluster_subset),
                                'Percentage': f"{(len(cluster_subset)/len(kmeans_data)*100):.1f}%",
                                'Avg Lat': f"{cluster_subset['lat'].mean():.4f}",
                                'Avg Long': f"{cluster_subset['long'].mean():.4f}"
                            }
                            
                            if 'macro_categoria' in cluster_subset.columns:
                                top_cat = cluster_subset['macro_categoria'].mode()
                                stat_row['Main Category'] = top_cat[0] if len(top_cat) > 0 else 'N/A'
                                stat_row['Num. Categories'] = cluster_subset['macro_categoria'].nunique()
                            
                            cluster_stats.append(stat_row)
                        
                        stats_df = pd.DataFrame(cluster_stats)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Comparative charts
                        st.markdown("### Cluster Comparative Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            cluster_counts = kmeans_data['Cluster'].value_counts().sort_index()
                            fig_dist = px.bar(
                                x=cluster_counts.index,
                                y=cluster_counts.values,
                                title="Event Distribution by Cluster",
                                labels={'x': 'Cluster', 'y': 'Number of Events'},
                                color=cluster_counts.values,
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with col2:
                            fig_pie = px.pie(
                                values=cluster_counts.values,
                                names=[f"Cluster {i}" for i in cluster_counts.index],
                                title="Event Proportion by Cluster",
                                hole=0.4
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Cluster table
                        with st.expander("View events table with Cluster"):
                            display_cols = [col for col in ['titolo', 'macro_categoria', 'data_inizio', 'lat', 'long', 'Cluster'] 
                                        if col in kmeans_data.columns]
                            st.dataframe(kmeans_data[display_cols].head(50), use_container_width=True)
                    
                    else:
                        pass
        
        else:
            # ========== CLUSTERING DBSCAN ==========
            st.sidebar.markdown("#### DBSCAN Clustering Parameters")

            # Parametri DBSCAN con slide/select box
            eps_value = st.sidebar.select_slider(
                "Eps value:",
                options=[0.001, 0.0015, 0.002],
                value=0.001
            )

            min_samples_value = st.sidebar.select_slider(
                "Min samples value:",
                options=[1, 10, 50],
                value=10
            )

            # Carica dati con cluster originale
            clustered_file = f"subcluster_results/subclusters_eps0.003_min50.csv"
            if not os.path.exists(clustered_file):
                st.error(f"File not found: {clustered_file}")
            else:
                data = pd.read_csv(clustered_file)

                if 'lat' not in data.columns or 'long' not in data.columns or 'subcluster' not in data.columns:
                    st.error("File must contain 'lat', 'long' and 'subcluster' columns")
                else:
                    # Filtra solo il cluster target
                    cluster_data = data[data['subcluster'] == 0].copy()
                    st.sidebar.markdown(f"Events in selected cluster: {len(cluster_data)}")

                    if len(cluster_data) == 0:
                        st.warning(f"Cluster {0} is empty!")
                    else:
                        coords = cluster_data[['lat', 'long']].to_numpy()
                        lat_min, lat_max = coords[:,0].min() - 0.005, coords[:,0].max() + 0.005
                        lon_min, lon_max = coords[:,1].min() - 0.005, coords[:,1].max() + 0.005

                        # Bottone per eseguire clustering
                        if st.sidebar.button("Run Clustering DBSCAN"):
                            st.info("Computing DBSCAN clusters...")

                            # Esegui DBSCAN
                            from sklearn.cluster import DBSCAN
                            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric='euclidean')
                            sub_labels = dbscan.fit_predict(coords)

                            # Statistiche
                            n_subclusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
                            n_noise = list(sub_labels).count(-1)
                            noise_percent = (n_noise / len(sub_labels)) * 100

                            # Aggiorna cluster_data
                            cluster_data['subsubcluster'] = sub_labels
                            cluster_data['hierarchical_id'] = cluster_data.apply(
                                lambda row: f"{0}.{row['subsubcluster']}" if row['subsubcluster'] != -1 else f"{0}.-1", axis=1
                            )

                            # Griglia matplotlib
                            # Griglia con due plot affiancati
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

                            # --- Sinistra: punti originali (senza cluster) ---
                            ax1.scatter(coords[:, 1], coords[:, 0], color='blue', s=50, alpha=0.7, edgecolor='k', linewidth=0.3)
                            ax1.set_title(f"Original Cluster (without DBSCAN)\nPoints: {len(coords)}", fontsize=14)
                            ax1.set_xlim(lon_min, lon_max)
                            ax1.set_ylim(lat_min, lat_max)
                            ax1.set_xlabel("Longitude")
                            ax1.set_ylabel("Latitude")
                            ax1.grid(True, alpha=0.3, linestyle='--')

                            # --- Destra: punti con sub-cluster DBSCAN ---
                            scatter = ax2.scatter(coords[:, 1], coords[:, 0], c=sub_labels, cmap='tab20', s=50, alpha=0.7, edgecolor='k', linewidth=0.3)
                            n_subclusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
                            n_noise = list(sub_labels).count(-1)
                            noise_percent = (n_noise / len(sub_labels)) * 100
                            ax2.set_title(f"DBSCAN Sub-clusters\nSub-clusters: {n_subclusters}, Outliers: {n_noise} ({noise_percent:.1f}%)", fontsize=14)
                            ax2.set_xlim(lon_min, lon_max)
                            ax2.set_ylim(lat_min, lat_max)
                            ax2.set_xlabel("Longitude")
                            ax2.set_ylabel("Latitude")
                            ax2.grid(True, alpha=0.3, linestyle='--')

                            # Aggiungi legenda dei cluster
                            handles, labels = scatter.legend_elements(prop="colors")
                            ax2.legend(handles, [str(int(l)) for l in range(n_subclusters)], title="Sub-clusters", bbox_to_anchor=(1.05, 1))

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()


                            # Tabella statistiche
                            stats_dict = {
                                'parent_cluster': [0],
                                'eps': [eps_value],
                                'min_samples': [min_samples_value],
                                'n_subclusters': [n_subclusters],
                                'n_noise': [n_noise],
                                'noise_percent': [noise_percent]
                            }
                            stats_df = pd.DataFrame(stats_dict)
                            st.markdown("### Clustering Statistics")
                            st.dataframe(stats_df, use_container_width=True)

                            st.write("---")
                            if st.button("Mostra su mappa"):
    
                            #MAPPA FOLIUM
                                import folium
                                import matplotlib.cm as cm
                                import matplotlib.colors as colors
                                from branca.colormap import LinearColormap
                                from streamlit_folium import st_folium

                                
                                m = folium.Map(location=[coords[:,0].mean(), coords[:,1].mean()], zoom_start=12, tiles="CartoDB positron")
                                unique_clusters = cluster_data['subsubcluster'].unique()
                                norm = colors.Normalize(vmin=min(unique_clusters), vmax=max(unique_clusters))
                                colormap = cm.get_cmap('tab20', len(unique_clusters))

                                for _, row in cluster_data.iterrows():
                                    cluster_id = row['subsubcluster']
                                    color = colors.to_hex(colormap(norm(cluster_id)))
                                    folium.CircleMarker(
                                        location=[row['lat'], row['long']],
                                        radius=3,
                                        color=color,
                                        fill=True,
                                        fill_opacity=0.7,
                                        weight=0.5
                                    ).add_to(m)

                                st.markdown("### Clustering Map")
                                st_folium(m, width=1000, height=500)
                            



else:
    st.error("Unable to load data. Please verify that the 'dataset_annuali_mi' folder exists and contains CSV files.")

# FOOTER
st.markdown("---")
st.markdown("*Milan Events Dashboard - Developed with Streamlit*")