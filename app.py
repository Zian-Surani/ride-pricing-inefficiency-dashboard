from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pydeck as pdk
import streamlit as st

from analysis import (
    BaseAnalysis,
    WEEKDAY_ORDER,
    add_zone_names,
    clean_and_engineer_data,
    load_zone_centroids,
    load_zone_lookup,
    plot_correlation_heatmap,
    plot_daily_patterns,
    plot_deviation,
    plot_demand_vs_surge_scatter,
    plot_duration_by_hour,
    plot_hourly_heatmap,
    plot_location_deviation,
    plot_outlier_scatter,
    plot_surge_box_by_hour,
    plot_surge_distribution,
    plot_time_series,
    run_full_analysis,
    summarize_filtered_data,
)

DATA_PATH = Path("data.csv")
ZONE_LOOKUP_PATH = Path("taxi_zone_lookup.csv")
ZONE_CENTROIDS_PATH = Path("taxi_zone_centroids.csv")
REQUIRED_ENGINEERED_COLUMNS = {
    "pickup_date",
    "pickup_day_name",
    "day_type",
    "hour",
    "surge",
    "trip_duration_minutes",
    "surge_zscore",
    "is_surge_outlier",
}


@st.cache_data(show_spinner=False)
def load_base_analysis(file_bytes: bytes | None) -> BaseAnalysis:
    if file_bytes is None:
        return run_full_analysis(DATA_PATH)
    return run_full_analysis(BytesIO(file_bytes))


def ensure_current_analysis_schema(base_analysis: BaseAnalysis) -> BaseAnalysis:
    cleaned_df = getattr(base_analysis, "cleaned_df", None)
    raw_df = getattr(base_analysis, "raw_df", None)
    cleaning_breakdown = getattr(base_analysis, "cleaning_breakdown", None)
    cleaning_report = getattr(base_analysis, "cleaning_report", {})

    missing_columns = (
        REQUIRED_ENGINEERED_COLUMNS.difference(cleaned_df.columns)
        if cleaned_df is not None
        else REQUIRED_ENGINEERED_COLUMNS
    )
    missing_report_keys = {"retained_percentage"}.difference(cleaning_report.keys())

    if raw_df is None:
        return base_analysis

    if missing_columns or cleaning_breakdown is None or missing_report_keys:
        return clean_and_engineer_data(raw_df)

    return base_analysis


@st.cache_data(show_spinner=False)
def load_zone_lookup_table() -> pd.DataFrame:
    return load_zone_lookup(ZONE_LOOKUP_PATH)


@st.cache_data(show_spinner=False)
def load_zone_centroid_table() -> pd.DataFrame:
    return load_zone_centroids(ZONE_CENTROIDS_PATH)


def figure_to_png_bytes(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=180)
    buffer.seek(0)
    plt.close(figure)
    return buffer.getvalue()


def render_figure(figure: plt.Figure) -> None:
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_graph_note(what: str, how: str, why: str) -> None:
    st.markdown(
        f"**What it shows:** {what}\n\n"
        f"**How to read it:** {how}\n\n"
        f"**Why it matters:** {why}"
    )


def build_insights(filtered_analysis, outlier_rate: float) -> list[str]:
    insights: list[str] = []

    if not filtered_analysis.busiest_hours.empty:
        busiest = filtered_analysis.busiest_hours.iloc[0]
        insights.append(
            f"The busiest hour is **{int(busiest['hour'])}:00**, with **{int(busiest['demand']):,} trips**."
        )
    if not filtered_analysis.inefficient_hours.empty:
        worst = filtered_analysis.inefficient_hours.iloc[0]
        direction = "higher" if worst["deviation"] > 0 else "lower"
        insights.append(
            f"The strongest pricing mismatch appears at **{int(worst['hour'])}:00**, "
            f"where observed surge is **{abs(worst['deviation']):.2f}** points {direction} than expected."
        )
    if not filtered_analysis.location_summary.empty:
        zone = filtered_analysis.location_summary.iloc[0]
        insights.append(
            f"The most abnormal pickup zone is **{zone['pickup_zone']} ({zone['pickup_borough']})**, "
            f"with deviation **{zone['deviation']:.2f}**."
        )
    if not filtered_analysis.daily_summary.empty:
        top_day = filtered_analysis.daily_summary.sort_values("demand", ascending=False).iloc[0]
        insights.append(
            f"The highest-demand date in the filtered view is **{top_day['pickup_date']}**, "
            f"with **{int(top_day['demand']):,} trips**."
        )
    insights.append(
        f"Extreme surge outliers make up **{outlier_rate:.2f}%** of the filtered trips, "
        "showing that a small subset of trips can distort pricing intensity."
    )

    return insights[:5]


def build_map_data(location_summary: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    map_df = location_summary.merge(
        centroids.rename(columns={"locationid": "pulocationid"}),
        on="pulocationid",
        how="left",
        suffixes=("", "_centroid"),
    )
    map_df = map_df.dropna(subset=["latitude", "longitude"]).copy()
    if map_df.empty:
        return map_df

    max_abs = map_df["absolute_deviation"].max() or 1
    map_df["radius"] = 150 + (map_df["demand"] / max(map_df["demand"].max(), 1)) * 900

    colors = []
    for _, row in map_df.iterrows():
        intensity = row["absolute_deviation"] / max_abs
        shade = int(120 + intensity * 120)
        alpha = int(150 + intensity * 80)
        if row["deviation"] >= 0:
            colors.append([shade, 70, 70, alpha])
        else:
            colors.append([70, 110, shade, alpha])
    map_df["color"] = colors
    return map_df


def render_downloads(filtered_analysis) -> None:
    st.subheader("Exports")

    filtered_csv = filtered_analysis.filtered_df.to_csv(index=False).encode("utf-8")
    hourly_csv = filtered_analysis.hourly_summary.to_csv(index=False).encode("utf-8")
    location_csv = filtered_analysis.location_summary.to_csv(index=False).encode("utf-8")

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    dl_col1.download_button(
        "Download Filtered Clean Data",
        data=filtered_csv,
        file_name="filtered_clean_taxi_data.csv",
        mime="text/csv",
    )
    dl_col2.download_button(
        "Download Hourly Summary",
        data=hourly_csv,
        file_name="hourly_summary.csv",
        mime="text/csv",
    )
    dl_col3.download_button(
        "Download Location Summary",
        data=location_csv,
        file_name="location_summary.csv",
        mime="text/csv",
    )

    time_chart_bytes = figure_to_png_bytes(plot_time_series(filtered_analysis.hourly_summary))
    deviation_chart_bytes = figure_to_png_bytes(plot_deviation(filtered_analysis.hourly_summary))

    img_col1, img_col2 = st.columns(2)
    img_col1.download_button(
        "Download Time Series PNG",
        data=time_chart_bytes,
        file_name="demand_vs_surge.png",
        mime="image/png",
    )
    img_col2.download_button(
        "Download Deviation PNG",
        data=deviation_chart_bytes,
        file_name="surge_deviation.png",
        mime="image/png",
    )


st.set_page_config(
    page_title="Ride Pricing Inefficiency Analysis Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ride Pricing Inefficiency Analysis Dashboard")
st.caption(
    "A graph-heavy descriptive analytics project using NYC Yellow Taxi data to test whether "
    "pricing intensity stays aligned with demand."
)

with st.sidebar:
    st.header("Filters")
    uploaded_file = st.file_uploader(
        "Upload your own NYC taxi CSV (optional)", type=["csv"]
    )

base_analysis = load_base_analysis(
    uploaded_file.getvalue() if uploaded_file is not None else None
)
base_analysis = ensure_current_analysis_schema(base_analysis)
zone_lookup = load_zone_lookup_table()
zone_centroids = load_zone_centroid_table()
cleaned_with_zones = add_zone_names(base_analysis.cleaned_df.copy(), zone_lookup)

location_options = (
    cleaned_with_zones[
        ["pulocationid", "pickup_zone", "pickup_borough", "pickup_zone_label"]
    ]
    .drop_duplicates()
    .sort_values(["pickup_borough", "pickup_zone"])
)
location_labels = dict(
    zip(location_options["pulocationid"], location_options["pickup_zone_label"])
)

min_date = cleaned_with_zones["pickup_date"].min()
max_date = cleaned_with_zones["pickup_date"].max()

with st.sidebar:
    hour_range = st.slider("Select hour range", 0, 23, (0, 23))
    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
    day_type = st.selectbox("Select day type", ["All", "Weekday", "Weekend"])
    selected_weekdays = st.multiselect(
        "Select weekdays",
        options=WEEKDAY_ORDER,
        default=WEEKDAY_ORDER,
    )
    selected_location = st.selectbox(
        "Select pickup location",
        options=["All", *location_options["pulocationid"].tolist()],
        format_func=lambda value: "All pickup locations"
        if value == "All"
        else location_labels.get(value, f"Location ID {value}"),
    )
    st.markdown("---")
    st.write(f"Rows loaded: **{len(base_analysis.raw_df):,}**")
    st.write(f"Rows after cleaning: **{len(base_analysis.cleaned_df):,}**")

filtered_analysis = summarize_filtered_data(
    cleaned_with_zones,
    hour_range=hour_range,
    location=selected_location,
    date_range=date_range,
    weekday_names=selected_weekdays,
    day_type=day_type,
)

if filtered_analysis.filtered_df.empty:
    st.warning("No trips match the current filters. Adjust the date, hour, weekday, or location filters.")
    st.stop()

hourly_summary = filtered_analysis.hourly_summary
active_hours = hourly_summary.loc[hourly_summary["demand"] > 0]

peak_hour_value = (
    int(active_hours.loc[active_hours["demand"].idxmax(), "hour"])
    if not active_hours.empty
    else "N/A"
)
avg_surge_value = filtered_analysis.filtered_df["surge"].mean()
avg_duration_value = filtered_analysis.filtered_df["trip_duration_minutes"].mean()
outlier_count = int(filtered_analysis.outliers.shape[0])
outlier_rate = (outlier_count / len(filtered_analysis.filtered_df)) * 100
selected_location_text = (
    "All pickup locations"
    if selected_location == "All"
    else location_labels.get(selected_location, f"Location ID {selected_location}")
)

st.header("0. Project Explanation")
intro_col1, intro_col2 = st.columns([1.3, 1])
with intro_col1:
    st.markdown(
        """
        This project studies **ride pricing inefficiency** using real NYC Yellow Taxi trip data.
        The dashboard is intentionally focused on **analysis and visualization**, not prediction.

        **Core logic**

        - **Demand** = number of trips in a time window
        - **Observed surge** = `fare_amount / trip_distance`
        - **Expected surge** = normalized demand
        - **Deviation** = observed surge minus expected surge

        If deviation becomes large, pricing is not moving in line with the demand baseline.
        That gap is treated as evidence of inefficiency.
        """
    )
with intro_col2:
    st.info(
        "Current filter context:\n\n"
        f"- Dates: **{date_range[0]}** to **{date_range[1]}**\n"
        f"- Hours: **{hour_range[0]} to {hour_range[1]}**\n"
        f"- Day type: **{day_type}**\n"
        f"- Pickup location: **{selected_location_text}**"
    )

method_col1, method_col2 = st.columns(2)
with method_col1:
    st.subheader("Methodology")
    st.markdown(
        """
        - Load the CSV data with pandas.
        - Remove null rows and invalid trips.
        - Create `surge = fare_amount / trip_distance`.
        - Create time features: date, weekday, weekend/weekday, and hour.
        - Group the data by hour and pickup zone.
        - Estimate expected surge from normalized demand.
        - Detect anomalies using deviation and z-score logic.
        """
    )
with method_col2:
    st.subheader("Limitations")
    st.markdown(
        """
        - The project uses **surge as a proxy**, not the platform's private pricing formula.
        - `fare_amount / trip_distance` captures pricing intensity, but not every hidden business rule.
        - The bundled CSV is a **sample**, so dashboard patterns are representative rather than exhaustive.
        - Taxi zone centroids are used for the map, so the geo view is a zone-level summary, not trip-level GPS.
        """
    )

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
overview_col1.metric("Total Trips", f"{len(filtered_analysis.filtered_df):,}")
overview_col2.metric("Average Surge", f"{avg_surge_value:.2f}")
overview_col3.metric("Peak Hour", peak_hour_value)
overview_col4.metric("Average Duration", f"{avg_duration_value:.1f} min")

st.subheader("Key Insights")
for insight in build_insights(filtered_analysis, outlier_rate):
    st.markdown(f"- {insight}")

st.header("1. Cleaning Summary")
clean_col1, clean_col2, clean_col3, clean_col4 = st.columns(4)
clean_col1.metric("Rows Removed", f"{base_analysis.cleaning_report['rows_removed']:,}")
clean_col2.metric(
    "Null Values Handled", f"{base_analysis.cleaning_report['null_rows_removed']:,}"
)
clean_col3.metric(
    "Invalid Trips Removed", f"{base_analysis.cleaning_report['invalid_trips_removed']:,}"
)
clean_col4.metric(
    "Retained Rows", f"{base_analysis.cleaning_report['retained_percentage']:.2f}%"
)

cleaning_breakdown = base_analysis.cleaning_breakdown.copy()
st.dataframe(cleaning_breakdown, use_container_width=True)

with st.expander("Before vs After Cleaning Preview", expanded=False):
    preview_col1, preview_col2 = st.columns(2)
    preview_col1.subheader("Before Cleaning")
    preview_col1.dataframe(base_analysis.before_sample.round(2), use_container_width=True)
    preview_col2.subheader("After Cleaning")
    preview_col2.dataframe(base_analysis.after_sample.round(2), use_container_width=True)

st.header("2. Time and Date Analysis")
time_col1, time_col2 = st.columns(2)
with time_col1:
    render_figure(plot_time_series(hourly_summary))
    render_graph_note(
        what="Hourly demand is shown with observed surge and expected surge on the same graph.",
        how="If the observed surge line follows expected surge closely, pricing is behaving more in line with demand. Clear gaps indicate mismatch.",
        why="This is the main chart for testing the core project hypothesis.",
    )
with time_col2:
    render_figure(plot_demand_vs_surge_scatter(hourly_summary))
    render_graph_note(
        what="Each point represents one hour, comparing demand with average surge.",
        how="A strong upward pattern means higher demand and higher surge move together. A scattered pattern means the relationship is inconsistent.",
        why="It summarizes how tightly pricing intensity is linked to demand.",
    )

date_col1, date_col2 = st.columns(2)
with date_col1:
    render_figure(plot_daily_patterns(filtered_analysis.daily_summary))
    render_graph_note(
        what="This chart shows how daily demand and daily average surge move over the selected dates.",
        how="Compare whether surge rises on the same dates where demand rises, or whether they drift apart.",
        why="It makes the new date and day filters meaningful by exposing date-level mismatch.",
    )
with date_col2:
    render_figure(plot_duration_by_hour(filtered_analysis.duration_summary))
    render_graph_note(
        what="The line chart shows average trip duration by hour.",
        how="Higher points indicate longer average trips during that hour.",
        why="Trip duration can influence fare intensity and helps explain whether inefficiency may be related to longer trips.",
    )

st.header("3. Inefficiency Detection")
ineff_col1, ineff_col2 = st.columns(2)
with ineff_col1:
    render_figure(plot_deviation(hourly_summary))
    render_graph_note(
        what="Deviation measures the gap between observed surge and expected surge for each hour.",
        how="Bars above zero show pricing intensity above the demand baseline. Bars below zero show pricing below the baseline. Highlighted bars are anomalies.",
        why="This is the direct inefficiency measure used throughout the project.",
    )
with ineff_col2:
    render_figure(plot_location_deviation(filtered_analysis.location_summary))
    render_graph_note(
        what="This graph ranks pickup zones by how abnormal their pricing deviation is.",
        how="Zones farther from zero are more unusual. Positive values mean stronger-than-expected pricing and negative values mean weaker-than-expected pricing.",
        why="It identifies where inefficiency appears geographically across NYC pickup zones.",
    )

comparison_col1, comparison_col2 = st.columns(2)
with comparison_col1:
    st.subheader("Top 5 Most Efficient Hours")
    st.dataframe(filtered_analysis.efficient_hours, use_container_width=True)
with comparison_col2:
    st.subheader("Top 5 Least Efficient Hours")
    st.dataframe(filtered_analysis.inefficient_hours, use_container_width=True)

abnormal_locations = filtered_analysis.location_summary.head(10)[
    [
        "pickup_zone",
        "pickup_borough",
        "pulocationid",
        "demand",
        "observed_surge",
        "expected_surge",
        "deviation",
        "is_anomaly",
    ]
].rename(
    columns={
        "pickup_zone": "pickup_zone_name",
        "pickup_borough": "borough",
        "pulocationid": "pickup_location_id",
        "observed_surge": "avg_surge",
    }
)
st.dataframe(abnormal_locations.round(2), use_container_width=True)

st.header("4. Distribution and Outliers")
dist_col1, dist_col2 = st.columns(2)
with dist_col1:
    render_figure(plot_surge_distribution(filtered_analysis.filtered_df))
    render_graph_note(
        what="The histogram shows how surge values are distributed across trips.",
        how="A long right tail means a smaller set of trips has much higher pricing intensity than the rest.",
        why="It helps show whether surge behavior is broad and stable or driven by extremes.",
    )
with dist_col2:
    render_figure(plot_surge_box_by_hour(filtered_analysis.filtered_df))
    render_graph_note(
        what="The box plot compares the spread of surge values across hours.",
        how="Higher median lines mean stronger typical surge, while taller boxes mean more variation within that hour.",
        why="It compares both central tendency and variability of pricing intensity.",
    )

outlier_col1, outlier_col2 = st.columns([2, 1])
with outlier_col1:
    render_figure(plot_outlier_scatter(filtered_analysis.filtered_df))
    render_graph_note(
        what="Trips are plotted by distance and surge, with extreme surge trips highlighted.",
        how="Red points are outliers based on surge z-score. The log scale makes extreme values easier to see.",
        why="It visually validates the outlier detection step and highlights unusual trips.",
    )
with outlier_col2:
    st.subheader("Outlier Summary")
    st.write(f"Extreme surge trips: **{outlier_count:,}**")
    st.write(f"Outlier rate: **{outlier_rate:.2f}%**")
    st.write(
        "Trips are flagged as outliers when the surge proxy has an absolute z-score of 3 or more."
    )

st.header("5. Geo View, Heatmap, and Correlation")
map_df = build_map_data(filtered_analysis.location_summary, zone_centroids)

map_col1, map_col2 = st.columns(2)
with map_col1:
    st.subheader("Pickup Zone Deviation Map")
    if map_df.empty:
        st.info("No map data is available for the current filters.")
    else:
        tooltip = {
            "html": (
                "<b>{pickup_zone}</b><br/>"
                "Borough: {pickup_borough}<br/>"
                "Demand: {demand}<br/>"
                "Observed surge: {observed_surge}<br/>"
                "Expected surge: {expected_surge}<br/>"
                "Deviation: {deviation}"
            ),
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            stroked=True,
            get_line_color=[30, 30, 30, 180],
            line_width_min_pixels=1,
        )
        view_state = pdk.ViewState(latitude=40.73, longitude=-73.94, zoom=9.6, pitch=0)
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_provider="carto",
                map_style="light",
            ),
            use_container_width=True,
        )
    render_graph_note(
        what="The map shows pickup zones as circles positioned at zone centroids and colored by deviation.",
        how="Red circles indicate stronger-than-expected pricing, blue circles indicate weaker-than-expected pricing, and larger circles represent higher demand.",
        why="It turns the location analysis into a real geographic story instead of a table of IDs.",
    )

with map_col2:
    render_figure(plot_hourly_heatmap(hourly_summary))
    render_graph_note(
        what="The heatmap compresses average hourly surge into a color-based summary.",
        how="Hotter cells indicate higher average surge for that hour.",
        why="It gives a quick scan of pricing hot spots across the day.",
    )

corr_col1, corr_col2 = st.columns(2)
with corr_col1:
    if filtered_analysis.correlation_matrix.empty:
        st.info("Not enough data is available for the correlation matrix under the current filters.")
    else:
        render_figure(plot_correlation_heatmap(filtered_analysis.correlation_matrix))
        render_graph_note(
            what="This matrix shows the pairwise correlation between demand, surge, fare, distance, duration, and deviation.",
            how="Values near 1 show strong positive relation, values near -1 show strong negative relation, and values near 0 show weak relation.",
            why="It adds numerical support to the visual story told by the charts.",
        )
with corr_col2:
    st.subheader("Weekday Summary")
    st.dataframe(filtered_analysis.weekday_summary, use_container_width=True)
    st.markdown(
        "This table helps compare demand, surge, and duration across the days of the week, "
        "which supports the weekday/weekend filter."
    )

st.header("6. Summary Statistics")
summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.subheader("Descriptive Statistics")
    st.dataframe(filtered_analysis.summary_statistics, use_container_width=True)
with summary_col2:
    fare_mean = filtered_analysis.filtered_df["fare_amount"].mean()
    distance_mean = filtered_analysis.filtered_df["trip_distance"].mean()
    fare_std = filtered_analysis.filtered_df["fare_amount"].std()
    st.subheader("Key Stats")
    st.write(f"Mean fare: **{fare_mean:.2f}**")
    st.write(f"Mean distance: **{distance_mean:.2f}**")
    st.write(f"Fare standard deviation: **{fare_std:.2f}**")
    st.write(f"Mean trip duration: **{avg_duration_value:.2f} minutes**")
    st.write("The table on the left is the full `describe()` output for the filtered numeric data.")

st.header("7. Ranking Analysis")
rank_col1, rank_col2 = st.columns(2)
with rank_col1:
    st.subheader("Top 5 Busiest Hours")
    st.dataframe(filtered_analysis.busiest_hours, use_container_width=True)
with rank_col2:
    st.subheader("Top 5 Surge Hours")
    st.dataframe(filtered_analysis.surge_hours, use_container_width=True)

st.header("8. Processed Data")
st.dataframe(
    filtered_analysis.filtered_df[
        [
            "tpep_pickup_datetime",
            "pickup_date",
            "pickup_day_name",
            "day_type",
            "fare_amount",
            "trip_distance",
            "trip_duration_minutes",
            "surge",
            "hour",
            "pulocationid",
            "pickup_zone",
            "pickup_borough",
            "is_surge_outlier",
        ]
    ].head(100).round(2),
    use_container_width=True,
)

st.header("9. Downloads")
render_downloads(filtered_analysis)

st.header("10. Final Interpretation")
demand_surge_corr = None
if not filtered_analysis.correlation_matrix.empty and {
    "demand",
    "surge",
}.issubset(filtered_analysis.correlation_matrix.columns):
    demand_surge_corr = filtered_analysis.correlation_matrix.loc["demand", "surge"]

if demand_surge_corr is None:
    corr_text = "could not be estimated confidently under the current filters"
else:
    corr_text = f"is **{demand_surge_corr:.2f}**"

st.markdown(
    f"""
    The dashboard shows that **pricing intensity is not always perfectly aligned with demand**.
    In the current filtered view, the demand-surge correlation {corr_text}, while the deviation
    charts and abnormal pickup zones show specific hours and places where observed surge moves
    away from the demand-based baseline.

    This supports the project conclusion that **inefficiencies exist in real ride-pricing systems**,
    and that **data analysis plus visualization can expose those flaws clearly** without building
    a machine learning model.
    """
)
