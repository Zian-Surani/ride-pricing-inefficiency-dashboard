# Ride Pricing Inefficiency Analysis Dashboard

A Streamlit dashboard that analyzes whether ride pricing intensity stays aligned with demand using the NYC Yellow Taxi dataset.

This project does not use machine learning. It focuses on:

- cleaning real taxi trip data
- building a surge proxy from fare and distance
- comparing observed surge against demand-based expected surge
- finding inefficient pricing patterns through graphs, rankings, and anomaly detection

## Dashboard Preview

The screenshots below show the dashboard from top to bottom, including the filters, KPI cards, charts, map, rankings, exports, and final interpretation section.

### Screenshot Gallery

#### 1. Overview, Filters, and Project Explanation

![Overview and filters](<screenshots/Screenshot 2026-03-28 163417.png>)

#### 2. Methodology, Limitations, and Key Insights

![Methodology and key insights](<screenshots/Screenshot 2026-03-28 163429.png>)

#### 3. Cleaning Summary and Data Preview

![Cleaning summary](<screenshots/Screenshot 2026-03-28 163443.png>)

#### 4. Time Analysis

![Time analysis charts](<screenshots/Screenshot 2026-03-28 163504.png>)

#### 5. Daily Pattern and Duration Analysis

![Daily and duration analysis](<screenshots/Screenshot 2026-03-28 163516.png>)

#### 6. Inefficiency Detection

![Inefficiency detection](<screenshots/Screenshot 2026-03-28 163524.png>)

#### 7. Distribution and Outlier Analysis

![Distribution and outliers](<screenshots/Screenshot 2026-03-28 163529.png>)

#### 8. Map, Heatmap, and Correlation

![Map and correlation](<screenshots/Screenshot 2026-03-28 163536.png>)

#### 9. Summary Statistics and Rankings

![Summary statistics and rankings](<screenshots/Screenshot 2026-03-28 163543.png>)

#### 10. Processed Data, Downloads, and Final Interpretation

![Downloads and conclusion](<screenshots/Screenshot 2026-03-28 163549.png>)

## Project Objective

The main idea of this project is to test the assumption:

`Surge is proportional to demand`

In this dashboard:

- `Demand = number of trips`
- `Observed Surge = fare_amount / trip_distance`
- `Expected Surge = normalized demand`
- `Deviation = observed surge - expected surge`

If observed surge is much higher or lower than expected surge, the dashboard treats that mismatch as pricing inefficiency.

## Dataset

- Dataset: NYC Yellow Taxi Trip Records
- Source type: real-world production taxi trip data
- Included sample: `data.csv`
- Additional lookup files:
  - `taxi_zone_lookup.csv`
  - `taxi_zone_centroids.csv`

Core columns used:

- `tpep_pickup_datetime`
- `tpep_dropoff_datetime`
- `fare_amount`
- `trip_distance`
- `PULocationID`

## Features

- Data cleaning summary with rows removed, invalid trips removed, and retained percentage
- Before vs after cleaning preview
- Hour filter
- Date range filter
- Weekday filter
- Weekday vs weekend filter
- Pickup zone filter with real place names
- KPI overview
- Demand vs surge time-series chart
- Demand vs surge scatter plot
- Surge distribution histogram
- Surge by hour box plot
- Deviation chart for inefficiency detection
- Abnormal pickup zone analysis
- Pickup zone map view
- Hourly heatmap
- Correlation matrix
- Trip duration analysis
- Summary statistics panel
- Top 5 busiest hours
- Top 5 surge hours
- Top 5 most efficient hours
- Top 5 least efficient hours
- Outlier detection view
- CSV and PNG export buttons
- Final interpretation section

## Project Workflow

1. Load the taxi dataset from CSV
2. Remove invalid records
3. Create engineered fields:
   - pickup date
   - weekday
   - day type
   - hour
   - surge proxy
   - trip duration
4. Group data by hour and pickup zone
5. Compute expected surge from demand
6. Measure deviation and detect anomalies
7. Display results in a Streamlit dashboard

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- pydeck

## Project Structure

```text
.
|-- app.py
|-- analysis.py
|-- data.csv
|-- taxi_zone_lookup.csv
|-- taxi_zone_centroids.csv
|-- requirements.txt
|-- run_instructions.txt
|-- screenshots/
|   |-- Screenshot 2026-03-28 163417.png
|   |-- Screenshot 2026-03-28 163429.png
|   |-- Screenshot 2026-03-28 163443.png
|   |-- Screenshot 2026-03-28 163504.png
|   |-- Screenshot 2026-03-28 163516.png
|   |-- Screenshot 2026-03-28 163524.png
|   |-- Screenshot 2026-03-28 163529.png
|   |-- Screenshot 2026-03-28 163536.png
|   |-- Screenshot 2026-03-28 163543.png
|   `-- Screenshot 2026-03-28 163549.png
`-- README.md
```

## How to Run

### First-time setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Next runs

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

The app usually opens at:

`http://localhost:8501`

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If Streamlit shows stale cached data:

```powershell
python -m streamlit cache clear
```

## Key Interpretation

This dashboard is built to support the conclusion that:

- pricing is not always perfectly aligned with demand
- inefficiencies exist in real ride-pricing systems
- data analysis and visualization can expose those inefficiencies clearly

## Notes

- The bundled `data.csv` is a sample for easy execution.
- You can upload a larger NYC Yellow Taxi CSV directly from the dashboard sidebar.
- The map view uses official taxi zone lookup and centroid metadata for named pickup locations.
