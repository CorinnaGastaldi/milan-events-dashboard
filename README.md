# Milan Events - Interactive Dashboard
### Description
Milan Events Interactive Dashboard is an interactive web application developed for in-depth analysis and visualization of cultural and entertainment events 
in the city of Milan. The project allows exploration of complex datasets through interactive maps, advanced statistical analysis, 
correlations with territorial presence data, and clustering algorithms, offering a comprehensive tool for understanding the spatial and temporal distribution 
of events across the Milanese territory.

---

### Objectives
- Visualize the geographic distribution of events across the territory through customizable interactive maps.
- Analyze correlations between cultural events and presence flows detected by Vodafone data.
- Identify temporal, seasonal, and monthly patterns in event programming.
- Apply clustering algorithms (K-Means and DBSCAN) to discover natural event aggregations and high-density zones.
- Provide filtering tools and temporal comparison for personalized analysis.
- Generate detailed statistical reports to support cultural and territorial planning decisions.

---
### Technologies used
- Python 3.8+
- Streamlit

---
### Project Structure
The main folder contains the following subfolders and files:

### Folders:

- `/dataset_annuali_mi`: event datasets organized in annual CSV files (2019-2024)
- `/zonizzazione progetto`: territorial zone shapefiles for spatial analysis
- `/eventi_kmeans_[N]`: folders containing pre-calculated K-Means clustering results with different numbers of clusters
- `/subcluster_results`: DBSCAN hierarchical clustering results

### Main Files:

- `app.py`: main Streamlit application file with all visualization and analysis logic
- `ma-bicocca.presenze.csv`: Vodafone presence data for comparative analysis
- `requirements.txt`: list of required Python dependencies

### Dataset Structure
Each annual CSV file contains the following information for each event:
- `titolo`: event name
- `macro_categoria`: event category
- `data_inizio`: event start date and time
- `lat`: geographic position latitude
- `long`: geographic position longitude
Other event-specific metadata

Vodafone presence data is structured with:
- `LOCATION`: territorial zone identifier
- `DATE_ID`: detection date
- `UNIQUE_USERS`: number of unique detected users

---
### Usage instructions
Clone or download the project:

```bash
git clone [link-del-repository]
cd milan-events-dashboard
```
Install dependencies:
```bash
pip install -r requirements.txt
```

Start the application:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at http://localhost:8501. 
On first launch, the system will automatically load all event dataset files from 2019 to 2024, 
preparing the data for subsequent analyses.

---
### Additional Notes

- On first launch, the application requires a few seconds to load the complete decade-long dataset. Data is cached to speed up subsequent interactions.
- Clustering algorithm parameters can be adjusted in the dedicated section to optimize results based on specific analysis needs.
- For K-Means clustering, pre-calculated results are available for different cluster configurations, optimizing application performance.
- Map visualization of large quantities of events (over 10,000 points) may require a few seconds for complete rendering.
- Vodafone data is available exclusively for the year 2023, limiting comparative analysis to this time period.

---
### Authors
Corinna Gastaldi

---
### Contact 
gastaldicorinna@gmail.com





