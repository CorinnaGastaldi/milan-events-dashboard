import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point, box
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="Eventi Milano - Dashboard Interattiva", 
    layout="wide",
    initial_sidebar_state="expanded"
)

#STILE 
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

#CONFIGURAZIONE PERCORSI 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
EVENTS_FOLDER = os.path.join(BASE_DIR, "dataset_annuali_mi")
SHAPEFILE_PATH = os.path.join(BASE_DIR, "zonizzazione progetto", "Shapefile_uniBicocca.shp")

@st.cache_data(show_spinner="Caricamento eventi in corso...")
def load_event_data(folder=EVENTS_FOLDER):
    """Carica tutti i CSV degli eventi"""
    if not os.path.exists(folder):
        st.error(f"Cartella non trovata: {folder}")
        st.info(f"Directory corrente: {os.getcwd()}")
        return None
    
    all_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    
    if not all_files:
        st.error(f"Nessun CSV trovato in: {folder}")
        return None
    
    #st.sidebar.success(f"Trovati {len(all_files)} file CSV")
    dfs = []
    
    for f in all_files:
        try:
            year = int(f.split(".")[0])
            df = pd.read_csv(os.path.join(folder, f))
            
            if 'lat' not in df.columns or 'long' not in df.columns:
                st.sidebar.warning(f"{f} manca colonne lat/long")
                continue
                
            df = df.dropna(subset=['lat', 'long'])
            df['year'] = year
            
            #converti data_inizio in datetime se presente
            if 'data_inizio' in df.columns:
                df['data_inizio'] = pd.to_datetime(df['data_inizio'], errors='coerce')
                df['month'] = df['data_inizio'].dt.month
                df['month_name'] = df['data_inizio'].dt.strftime('%B')
                df['season'] = df['month'].apply(get_season)
                df['quarter'] = df['data_inizio'].dt.quarter
            
            dfs.append(df)
        except Exception as e:
            st.sidebar.warning(f"Errore in {f}: {str(e)[:50]}")
    
    if not dfs:
        st.error("Nessun dato caricato correttamente")
        return None
    
    data = pd.concat(dfs, ignore_index=True)
    #st.sidebar.info(f"Totale eventi: {len(data)}")
    return data

def get_season(month):
    """Determina la stagione dal mese"""
    if month in [12, 1, 2]:
        return "Inverno"
    elif month in [3, 4, 5]:
        return "Primavera"
    elif month in [6, 7, 8]:
        return "Estate"
    else:
        return "Autunno"

def create_report_stats(data_subset, period_label):
    """Crea statistiche per il report"""
    stats = {
        'periodo': period_label,
        'totale_eventi': len(data_subset),
        'eventi_giorno': len(data_subset) / 30 if len(data_subset) > 0 else 0
    }
    
    if 'macro_categoria' in data_subset.columns:
        stats['top_categoria'] = data_subset['macro_categoria'].mode()[0] if len(data_subset) > 0 else 'N/A'
        stats['num_categorie'] = data_subset['macro_categoria'].nunique()
    
    return stats

def generate_pdf_report(data, period_info):
    """Genera report PDF (simulato con CSV per semplicità)"""
    buffer = BytesIO()
    
    #report dettagliato
    report_data = []
    
    for year in sorted(data['year'].unique()):
        year_data = data[data['year'] == year]
        
        if 'season' in data.columns:
            for season in ["Inverno", "Primavera", "Estate", "Autunno"]:
                season_data = year_data[year_data['season'] == season]
                
                report_row = {
                    'Anno': year,
                    'Periodo': season,
                    'Totale_Eventi': len(season_data),
                    'Media_Giornaliera': len(season_data) / 90
                }
                
                if 'macro_categoria' in season_data.columns and len(season_data) > 0:
                    report_row['Categoria_Principale'] = season_data['macro_categoria'].mode()[0]
                    report_row['Numero_Categorie'] = season_data['macro_categoria'].nunique()
                
                report_data.append(report_row)
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return buffer

@st.cache_data(show_spinner="Caricamento shapefile...")
def load_shapefile(path=SHAPEFILE_PATH):
    """Carica shapefile delle zone"""
    if not os.path.exists(path):
        st.error(f"Shapefile non trovato: {path}")
        return None
    
    try:
        zones_gdf = gpd.read_file(path).to_crs(epsg=3857)
        #st.sidebar.success(f"Shapefile caricato: {len(zones_gdf)} zone")
        return zones_gdf
    except Exception as e:
        st.error(f"Errore shapefile: {e}")
        return None

def create_folium_map(subset, map_type="markers", colormap=None):
    """Crea mappa Folium con diversi tipi di visualizzazione"""
    center_lat = subset['lat'].mean()
    center_lon = subset['long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    if map_type == "heatmap":
        # Heatmap
        heat_data = [[row['lat'], row['long']] for _, row in subset.iterrows()]
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
        
    elif map_type == "cluster":
        # Marker Cluster
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in subset.iterrows():
            folium.Marker(
                location=[row['lat'], row['long']],
                popup=f"{row.get('macro_categoria', 'N/A')}<br>{row.get('titolo', '')}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)
            
    else:
        # Markers circolari con colori per categoria
        if colormap and 'macro_categoria' in subset.columns:
            categories = subset['macro_categoria'].unique()
            colors = px.colors.qualitative.Set3[:len(categories)]
            color_map = dict(zip(categories, colors))
            
            for _, row in subset.iterrows():
                cat = row.get('macro_categoria', 'Altro')
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
    """Calcola densità eventi per zona"""
    events_year = events_gdf[events_gdf['year'] == year]
    joined = gpd.sjoin(events_year, zones_gdf, how="inner", predicate='within')
    counts = joined.groupby('area_id').size().rename('event_count')
    
    zones_year = zones_gdf.set_index('area_id').join(counts).fillna(0)
    zones_year['area_km2'] = zones_year.geometry.area / 1e6
    zones_year['density'] = zones_year['event_count'] / zones_year['area_km2']
    
    return zones_year

# --- HEADER ---
st.markdown('<p class="main-header">Eventi Milano - Dashboard Interattiva</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Configurazione")

page = st.sidebar.radio("Seleziona visualizzazione:", 
                        ["Mappa Eventi", "Analisi Zone", "Statistiche"])

# Carica dati
data = load_event_data()

if data is not None:
    years = sorted(data['year'].unique())
    
    #PAGINA 1: MAPPA EVENTI
    if page == "Mappa Eventi":
        st.sidebar.markdown("---")
        
        # Modalità comparazione
        compare_mode = st.sidebar.checkbox("Confronta due periodi", value=False)
        
        if not compare_mode:
            # MODALITÀ STANDARD 
            # Filtro temporale
            time_filter = st.sidebar.radio("Filtro temporale:", 
                                           ["Anno", "Stagione", "Mese", "Trimestre"])
            
            if time_filter == "Anno":
                selected_year = st.sidebar.slider("Seleziona anno:", 
                                                 min_value=int(years[0]), 
                                                 max_value=int(years[-1]), 
                                                 value=int(years[-1]))
                subset = data[data['year'] == selected_year]
                time_label = f"Anno {selected_year}"
                
            elif time_filter == "Stagione":
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    selected_year = st.selectbox("Anno:", sorted(data['year'].unique()))
                with col2:
                    season_options = ["Inverno", "Primavera", "Estate", "Autunno"]
                    selected_season = st.selectbox("Stagione:", season_options)
                
                subset = data[(data['year'] == selected_year) & (data['season'] == selected_season)]
                time_label = f"{selected_season} {selected_year}"
                
            elif time_filter == "Mese":
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    selected_year = st.selectbox("Anno:", sorted(data['year'].unique()))
                with col2:
                    if 'month' in data.columns:
                        month_names = {
                            1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile",
                            5: "Maggio", 6: "Giugno", 7: "Luglio", 8: "Agosto",
                            9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre"
                        }
                        available_months = sorted(data[data['year'] == selected_year]['month'].dropna().unique())
                        month_options = [month_names[int(m)] for m in available_months]
                        selected_month_name = st.selectbox("Mese:", month_options)
                        selected_month = [k for k, v in month_names.items() if v == selected_month_name][0]
                
                subset = data[(data['year'] == selected_year) & (data['month'] == selected_month)]
                time_label = f"{selected_month_name} {selected_year}"
                
            else:  # Trimestre
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    selected_year = st.selectbox("Anno:", sorted(data['year'].unique()))
                with col2:
                    quarter_options = ["Q1 (Gen-Mar)", "Q2 (Apr-Giu)", "Q3 (Lug-Set)", "Q4 (Ott-Dic)"]
                    selected_quarter_name = st.selectbox("Trimestre:", quarter_options)
                    selected_quarter = int(selected_quarter_name[1])
                
                subset = data[(data['year'] == selected_year) & (data['quarter'] == selected_quarter)]
                time_label = f"{selected_quarter_name} {selected_year}"
            
            map_type = st.sidebar.selectbox("Tipo mappa:", 
                                            ["markers", "heatmap"],
                                            format_func=lambda x: {
                                                "markers": "Markers colorati",
                                                "heatmap": "Mappa di calore",
                                            }[x])
            
            show_categories = st.sidebar.checkbox("Colora per categoria", value=True)
            
            # Filtro categorie (se presente)
            if 'macro_categoria' in subset.columns:
                categories = ['Tutte'] + sorted(subset['macro_categoria'].unique().tolist())
                selected_cat = st.sidebar.selectbox("Filtra categoria:", categories)
                
                if selected_cat != 'Tutte':
                    subset = subset[subset['macro_categoria'] == selected_cat]
            
            # Metriche
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Periodo", time_label)
            with col2:
                st.metric("Eventi visualizzati", len(subset))
            with col3:
                if 'macro_categoria' in subset.columns:
                    st.metric("Categorie", subset['macro_categoria'].nunique())
            with col4:
                # Media eventi giornalieri nel periodo
                if time_filter == "Anno":
                    avg_daily = len(subset) / 365
                elif time_filter == "Stagione":
                    avg_daily = len(subset) / 90
                elif time_filter == "Mese":
                    avg_daily = len(subset) / 30
                else:
                    avg_daily = len(subset) / 90
                st.metric("Media giornaliera", f"{avg_daily:.1f}")
            
            st.markdown("---")
            
            # Crea e visualizza mappa
            if len(subset) > 0:
                m = create_folium_map(subset, map_type, colormap=show_categories)
                st_folium(m, width=1400, height=700, returned_objects=[])
                
                # Tabella eventi
                with st.expander("Visualizza tabella eventi"):
                    display_cols = [col for col in ['titolo', 'macro_categoria', 'data_inizio', 'lat', 'long'] 
                                   if col in subset.columns]
                    st.dataframe(subset[display_cols].head(50), use_container_width=True)
            else:
                st.warning("Nessun evento trovato per i filtri selezionati")
        
        else:
            #MODALITÀ COMPARAZIONE 
            st.markdown("### Confronto tra due periodi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Periodo 1")
                year1 = st.selectbox("Anno 1:", sorted(data['year'].unique()), key="year1")
                if 'season' in data.columns:
                    season1 = st.selectbox("Stagione 1:", ["Inverno", "Primavera", "Estate", "Autunno"], key="season1")
                    subset1 = data[(data['year'] == year1) & (data['season'] == season1)]
                    label1 = f"{season1} {year1}"
                else:
                    subset1 = data[data['year'] == year1]
                    label1 = f"Anno {year1}"
            
            with col2:
                st.markdown("#### Periodo 2")
                year2 = st.selectbox("Anno 2:", sorted(data['year'].unique()), key="year2")
                if 'season' in data.columns:
                    season2 = st.selectbox("Stagione 2:", ["Inverno", "Primavera", "Estate", "Autunno"], key="season2")
                    subset2 = data[(data['year'] == year2) & (data['season'] == season2)]
                    label2 = f"{season2} {year2}"
                else:
                    subset2 = data[data['year'] == year2]
                    label2 = f"Anno {year2}"
            
            # Metriche comparative
            col1, col2, col3 = st.columns(3)
            with col1:
                diff = len(subset2) - len(subset1)
                st.metric(f"Eventi {label1}", len(subset1))
                st.metric(f"Eventi {label2}", len(subset2), delta=diff)
            
            with col2:
                if 'macro_categoria' in subset1.columns:
                    diff_cat = subset2['macro_categoria'].nunique() - subset1['macro_categoria'].nunique()
                    st.metric(f"Categorie {label1}", subset1['macro_categoria'].nunique())
                    st.metric(f"Categorie {label2}", subset2['macro_categoria'].nunique(), delta=diff_cat)
            
            with col3:
                pct_change = ((len(subset2) - len(subset1)) / len(subset1) * 100) if len(subset1) > 0 else 0
                st.metric("Variazione %", f"{pct_change:+.1f}%")
            
            st.markdown("---")
            
            # Mappe affiancate
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {label1}")
                if len(subset1) > 0:
                    m1 = create_folium_map(subset1, "heatmap", colormap=False)
                    st_folium(m1, width=650, height=500, returned_objects=[])
                else:
                    st.warning("Nessun dato per questo periodo")
            
            with col2:
                st.markdown(f"#### {label2}")
                if len(subset2) > 0:
                    m2 = create_folium_map(subset2, "heatmap", colormap=False)
                    st_folium(m2, width=650, height=500, returned_objects=[])
                else:
                    st.warning("Nessun dato per questo periodo")
            
            # Confronto categorie
            if 'macro_categoria' in data.columns:
                st.markdown("### Confronto categorie")
                
                cat1 = subset1['macro_categoria'].value_counts().head(10)
                cat2 = subset2['macro_categoria'].value_counts().head(10)
                
                comparison_df = pd.DataFrame({
                    label1: cat1,
                    label2: cat2
                }).fillna(0).astype(int)
                
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(name=label1, x=comparison_df.index, y=comparison_df[label1]))
                fig_compare.add_trace(go.Bar(name=label2, x=comparison_df.index, y=comparison_df[label2]))
                fig_compare.update_layout(barmode='group', title="Top 10 categorie - Confronto")
                st.plotly_chart(fig_compare, use_container_width=True)
    
    #PAGINA 2: ANALISI ZONE 
    elif page == "Analisi Zone":
        st.sidebar.markdown("---")
        
        zones_gdf = load_shapefile()
        
        if zones_gdf is not None:
            # Converti eventi in GeoDataFrame
            geometry = [Point(xy) for xy in zip(data['long'], data['lat'])]
            events_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
            
            selected_year = st.sidebar.slider("Seleziona anno:", 
                                             min_value=int(years[0]), 
                                             max_value=int(years[-1]), 
                                             value=int(years[-1]))
            
            color_scheme = st.sidebar.selectbox("Schema colori:", 
                                               ["viridis", "plasma", "cividis"])
            
            # Calcola densità
            zones_year = calculate_zone_density(events_gdf, zones_gdf, selected_year)
            
            # Metriche
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Zone totali", len(zones_year))
            with col2:
                st.metric("Eventi totali", int(zones_year['event_count'].sum()))
            with col3:
                st.metric("Densità media", f"{zones_year['density'].mean():.1f} ev/km²")
            
            st.markdown("---")
            
            # Crea mappa con matplotlib
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Determina il numero di classi per la visualizzazione
            n_classes = min(10, max(3, zones_year['density'].nunique()))
            
            try:
                zones_year.plot(
                    column='density',
                    ax=ax,
                    cmap=color_scheme,
                    scheme='fisher_jenks',
                    k=n_classes,
                    legend=True,
                    legend_kwds={
                        'label': "Eventi per km²",
                        'orientation': "vertical",
                        'shrink': 0.8
                    }
                )
            except Exception as e:
                # Fallback senza classificazione se fisher_jenks fallisce
                st.warning(f"Usando visualizzazione continua: {e}")
                zones_year.plot(
                    column='density',
                    ax=ax,
                    cmap=color_scheme,
                    legend=True
                )
            
            # Basemap
            bbox_wgs84 = box(9.10, 45.40, 9.30, 45.55)
            gdf_bbox = gpd.GeoSeries([bbox_wgs84], crs="EPSG:4326").to_crs(epsg=3857)
            xmin, ymin, xmax, ymax = gdf_bbox.total_bounds
            
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"Densità eventi per zona - {selected_year}", fontsize=18, weight='bold')
            ax.axis('off')
            
            st.pyplot(fig)
            
            # Top 10 zone
            st.markdown("### Top 10 zone per densità eventi")
            top_zones = zones_year.nlargest(10, 'density')[['event_count', 'area_km2', 'density']]
            top_zones.columns = ['Eventi', 'Area (km²)', 'Densità (ev/km²)']
            st.dataframe(top_zones.style.format({'Area (km²)': '{:.2f}', 'Densità (ev/km²)': '{:.1f}'}), 
                        use_container_width=True)
        else:
            st.error("Impossibile caricare il file shapefile")
    
    #PAGINA 3: STATISTICHE 
    elif page == "Statistiche":
        st.markdown("### Statistiche temporali")
        
        # Eventi per anno
        yearly_counts = data.groupby('year').size().reset_index(name='count')
        
        fig1 = px.line(yearly_counts, x='year', y='count', 
                      title="Numero eventi per anno",
                      labels={'year': 'Anno', 'count': 'Numero eventi'},
                      markers=True)
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Analisi stagionale
        if 'season' in data.columns:
            st.markdown("### Andamento stagionale")
            
            col1, col2 = st.columns(2)
            
            with col1:
                seasonal_counts = data.groupby(['year', 'season']).size().reset_index(name='count')
                fig_season = px.line(seasonal_counts, x='year', y='count', color='season',
                                    title="Eventi per stagione nel tempo",
                                    labels={'year': 'Anno', 'count': 'Eventi'},
                                    color_discrete_map={
                                        "Inverno": "#89CFF0",
                                        "Primavera": "#FFB6C1",
                                        "Estate": "#FFD700",
                                        "Autunno": "#FF8C00"
                                    })
                st.plotly_chart(fig_season, use_container_width=True)
            
            with col2:
                season_totals = data['season'].value_counts()
                fig_season_pie = px.pie(values=season_totals.values, 
                                       names=season_totals.index,
                                       title="Distribuzione totale per stagione",
                                       color=season_totals.index,
                                       color_discrete_map={
                                           "Inverno": "#89CFF0",
                                           "Primavera": "#FFB6C1",
                                           "Estate": "#FFD700",
                                           "Autunno": "#FF8C00"
                                       })
                st.plotly_chart(fig_season_pie, use_container_width=True)
        
        # Analisi mensile
        if 'month' in data.columns:
            st.markdown("### Andamento mensile")
            
            month_names_map = {
                1: "Gen", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mag", 6: "Giu",
                7: "Lug", 8: "Ago", 9: "Set", 10: "Ott", 11: "Nov", 12: "Dic"
            }
            
            monthly_data = data.groupby(['year', 'month']).size().reset_index(name='count')
            monthly_data['month_label'] = monthly_data['month'].map(month_names_map)
            
            # Heatmap mensile
            pivot_month = monthly_data.pivot(index='month_label', columns='year', values='count').fillna(0)
            pivot_month = pivot_month.reindex([month_names_map[i] for i in range(1, 13)])
            
            fig_month_heat = px.imshow(pivot_month,
                                      labels=dict(x="Anno", y="Mese", color="Eventi"),
                                      title="Heatmap mensile degli eventi",
                                      aspect="auto",
                                      color_continuous_scale="YlOrRd")
            fig_month_heat.update_layout(height=500)
            st.plotly_chart(fig_month_heat, use_container_width=True)
            
            # Media eventi per mese (tutti gli anni)
            avg_by_month = data.groupby('month').size() / data['year'].nunique()
            avg_by_month.index = avg_by_month.index.map(month_names_map)
            
            fig_avg_month = px.bar(x=avg_by_month.index, y=avg_by_month.values,
                                  title="Media eventi per mese (tutti gli anni)",
                                  labels={'x': 'Mese', 'y': 'Media eventi'},
                                  color=avg_by_month.values,
                                  color_continuous_scale="Blues")
            st.plotly_chart(fig_avg_month, use_container_width=True)
        
        # Distribuzione per categoria (se presente)
        if 'macro_categoria' in data.columns:
            st.markdown("### Distribuzione per categoria")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cat_counts = data['macro_categoria'].value_counts().head(10)
                fig2 = px.bar(x=cat_counts.index, y=cat_counts.values,
                             title="Top 10 categorie",
                             labels={'x': 'Categoria', 'y': 'Numero eventi'})
                fig2.update_xaxes(tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = px.pie(data, names='macro_categoria', 
                             title="Distribuzione categorie (tutti gli anni)",
                             hole=0.3)
                st.plotly_chart(fig3, use_container_width=True)
            
            # Categorie per stagione
            if 'season' in data.columns:
                st.markdown("### Categorie per stagione")
                cat_season = data.groupby(['season', 'macro_categoria']).size().reset_index(name='count')
                top_cats = data['macro_categoria'].value_counts().head(8).index
                cat_season = cat_season[cat_season['macro_categoria'].isin(top_cats)]
                
                fig_cat_season = px.bar(cat_season, x='season', y='count', color='macro_categoria',
                                       title="Top 8 categorie per stagione",
                                       labels={'season': 'Stagione', 'count': 'Eventi'},
                                       barmode='stack')
                st.plotly_chart(fig_cat_season, use_container_width=True)

else:
    st.error("Impossibile caricare i dati. Verifica che la cartella 'dataset_annuali_mi' esista e contenga file CSV.")

#FOOTER
st.markdown("---")
st.markdown("*Dashboard Eventi Milano - Sviluppata con Streamlit*")