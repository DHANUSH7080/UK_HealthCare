import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from streamlit_option_menu import option_menu
from datetime import datetime
import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

# NHS statistical region code -> display name (verified against provider locations)
REGION_NAMES = {
    'Y56': 'London',
    'Y58': 'South West',
    'Y59': 'South East',
    'Y60': 'Midlands',
    'Y61': 'East of England',
    'Y62': 'North West',
    'Y63': 'North East and Yorkshire',
}

# Page config
st.set_page_config(page_title="NHS Dynamic Dashboard", layout="wide")

# Custom CSS — "liquid glass" theme: dark base, blurred colour orbs for depth,
# frosted translucent panels throughout (sidebar, cards, charts, tables, inputs).
st.markdown("""
<style>
:root {
    --accent: #2dd4bf;
    --accent-2: #8b5cf6;
    --accent-3: #f472b6;
    --glass-bg: rgba(255,255,255,0.055);
    --glass-border: rgba(255,255,255,0.12);
    --text-primary: #f5f5f7;
    --text-secondary: #9a9aa5;
}

html, body, .stApp {
    background: #0a0b0f !important;
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Soft blurred colour orbs behind everything for the glass depth effect */
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    border-radius: 50%;
    filter: blur(110px);
    z-index: 0;
    opacity: 0.35;
    pointer-events: none;
}
.stApp::before {
    width: 560px; height: 560px;
    background: radial-gradient(circle, var(--accent), transparent 70%);
    top: -180px; left: -140px;
}
.stApp::after {
    width: 520px; height: 520px;
    background: radial-gradient(circle, var(--accent-2), transparent 70%);
    bottom: -160px; right: -120px;
}

[data-testid="stSidebar"] {
    background: transparent !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: var(--glass-bg);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border-right: 1px solid var(--glass-border);
}

h1, h2, h3, h4, h5 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px;
}
p, span, label, .stMarkdown, .stCaption {
    color: var(--text-secondary);
}

/* Glass styling for Streamlit's native inputs */
div[data-baseweb="select"] > div,
.stTextArea textarea,
.stTextInput input {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
}
.stButton > button {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    color: var(--text-primary);
    backdrop-filter: blur(10px);
}
.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent) !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--glass-border);
}
[data-testid="stNotification"] {
    border-radius: 14px !important;
    backdrop-filter: blur(14px);
}
[data-testid="stMetricValue"] {
    color: var(--text-primary);
}

/* Glass metric card, used via create_metric_card() */
.metric-card {
    background: var(--glass-bg);
    backdrop-filter: blur(22px) saturate(180%);
    -webkit-backdrop-filter: blur(22px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08);
    transition: transform 0.25s ease, border-color 0.25s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(45, 212, 191, 0.4);
}
.metric-label {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--text-primary);
}
.metric-delta {
    display: inline-block;
    margin-top: 0.6rem;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 999px;
    background: rgba(45, 212, 191, 0.14);
    color: #5eead4;
}

.plot-container {
    background: var(--glass-bg);
    backdrop-filter: blur(22px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.glass-alert {
    background: rgba(244, 114, 182, 0.1);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(244, 114, 182, 0.3);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.glass-alert h4 { color: #f9a8d4 !important; margin: 0 0 0.3rem; font-size: 1rem; }
.glass-alert p { color: var(--text-primary); margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data(file_mtime):
    # file_mtime is unused inside the function but is part of the cache key:
    # whenever the CSV on disk changes, its mtime changes, so Streamlit
    # treats this as a new call and re-reads the file instead of serving
    # a stale cached DataFrame after a redeploy.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "NHS_Trusts_Merged_2024_2025.csv")
    return pd.read_csv(file_path, parse_dates=["Month"])  # <- convert on load

@st.cache_data
def load_region_compliance(file_mtime):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "Target_Compliance_Region_2025_2026.csv")
    comp = pd.read_csv(file_path)
    comp["Month_dt"] = pd.to_datetime(comp["Month"], format="%B-%Y")
    return comp.sort_values("Month_dt")

@st.cache_data
def load_provider_compliance(file_mtime):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "Target_Compliance_Provider_2025_2026.csv")
    comp = pd.read_csv(file_path)
    comp["Month_dt"] = pd.to_datetime(comp["Month"], format="%B-%Y")
    return comp.sort_values("Month_dt")

@st.cache_data
def load_region_geojson(file_mtime):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "nhs_regions.geojson")
    with open(file_path) as f:
        return json.load(f)

_base_dir = os.path.dirname(os.path.abspath(__file__))
_csv_path = os.path.join(_base_dir, "..", "data", "NHS_Trusts_Merged_2024_2025.csv")
df = load_data(os.path.getmtime(_csv_path))

_region_comp_path = os.path.join(_base_dir, "..", "data", "Target_Compliance_Region_2025_2026.csv")
_provider_comp_path = os.path.join(_base_dir, "..", "data", "Target_Compliance_Provider_2025_2026.csv")
_geojson_path = os.path.join(_base_dir, "..", "data", "nhs_regions.geojson")


# Navigation
with st.sidebar:
    page = option_menu(
        "NHS Dashboard",
        ["Home - Trends", "Target Compliance", "Regional Map", "Anomaly Detection", "Forecasting", "Raw Data", "Chat with NHS AI"],
        icons=['bar-chart-line', 'flag', 'map', 'exclamation-triangle', 'graph-up', 'table', 'chat-dots'],
        menu_icon="hospital",
        default_index=0,
        styles={
            "container": {"padding": "8px", "background-color": "transparent"},
            "icon": {"color": "#9a9aa5", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px", "text-align": "left", "margin": "3px 0",
                "border-radius": "14px", "color": "#c9c9d3", "padding": "10px 14px",
            },
            "nav-link-selected": {
                "background-color": "rgba(45, 212, 191, 0.14)",
                "border": "1px solid rgba(45, 212, 191, 0.35)",
                "color": "#f5f5f7",
                "border-radius": "14px",
            },
        },
    )

# Metric Card
def create_metric_card(label, value, delta=None):
    card = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-delta">{delta}</div>' if delta else ''}
    </div>
    """
    return st.markdown(card, unsafe_allow_html=True)

# Home Page
if page == "Home - Trends":
    st.title("📈 NHS Performance Dashboard")
    trust_options = ["All Trusts"] + sorted(df["Provider Name"].unique())
    selected_trust = st.selectbox("Select Trust to Analyze:", trust_options)

    if selected_trust == "All Trusts":
        filtered_df = df
        trust_suffix = "(All Trusts)"
    else:
        filtered_df = df[df["Provider Name"] == selected_trust]
        trust_suffix = f"({selected_trust})"

    col1, col2, col3 = st.columns(3)
    with col1:
        current_wait = filtered_df['Average (median) waiting time (in weeks)'].iloc[-1]
        create_metric_card(f"Current Median Wait {trust_suffix}", f"{current_wait:.1f} weeks")
    with col2:
        incomplete_pathways = filtered_df['Total number of incomplete pathways'].sum()
        create_metric_card(f"Total Incomplete Pathways {trust_suffix}", f"{incomplete_pathways:,}")
    with col3:
        avg_change = filtered_df['Average (median) waiting time (in weeks)'].pct_change().mean() * 100
        create_metric_card(f"Monthly Change {trust_suffix}", f"{avg_change:.1f}%", f"{avg_change:.1f}% from last month")

    st.subheader(f"📆 Monthly Waiting Time Trends {trust_suffix}")
    trust_avg = filtered_df.groupby("Month")["Average (median) waiting time (in weeks)"].mean().reset_index()
    fig = px.line(trust_avg, x="Month", y="Average (median) waiting time (in weeks)",
                  template="plotly_dark", line_shape="spline",
                  markers=True, color_discrete_sequence=['#2dd4bf'])
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e5ea'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🏥 Compare Provider Performance", expanded=True):
        providers = st.multiselect("Select providers to compare:", df['Provider Name'].unique())
        if providers:
            provider_df = df[df['Provider Name'].isin(providers)]
            fig = px.area(provider_df, x="Month", y="Total number of incomplete pathways",
                          color="Provider Name", template="plotly_dark",
                          line_group="Provider Name", hover_name="Provider Name",
                          color_discrete_sequence=['#2dd4bf', '#8b5cf6', '#f472b6', '#fbbf24', '#60a5fa', '#34d399'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e5ea'))
            st.plotly_chart(fig, use_container_width=True)

# Target Compliance
elif page == "Target Compliance":
    st.title("🎯 18-Week / 52-Week Target Compliance")
    st.caption(
        "Tracks performance against NHS England's elective recovery targets: 65% of patients "
        "seen within 18 weeks (interim goal reached nationally in March 2026), rising to the "
        "92% constitutional standard by 2029, while holding 52+ week waits under 1%. "
        "Coverage: April 2025 – April 2026 (the dataset's earlier months predate this breakdown)."
    )

    region_comp = load_region_compliance(os.path.getmtime(_region_comp_path))
    provider_comp = load_provider_compliance(os.path.getmtime(_provider_comp_path))

    national = region_comp.groupby("Month_dt", as_index=False)[
        ["Total number of incomplete pathways", "Total within 18 weeks", "Total 52 plus weeks"]
    ].sum()
    national["% within 18 weeks"] = national["Total within 18 weeks"] / national["Total number of incomplete pathways"] * 100
    national["% 52 plus weeks"] = national["Total 52 plus weeks"] / national["Total number of incomplete pathways"] * 100
    national = national.sort_values("Month_dt")
    latest = national.iloc[-1]

    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_card(
            "% Within 18 Weeks (National, latest)",
            f"{latest['% within 18 weeks']:.1f}%",
            f"{latest['% within 18 weeks'] - 65:+.1f} pts vs 65% interim target"
        )
    with col2:
        create_metric_card(
            "% Waiting 52+ Weeks (National, latest)",
            f"{latest['% 52 plus weeks']:.2f}%",
            "target: under 1%"
        )
    with col3:
        create_metric_card(
            "Gap to 92% Constitutional Standard",
            f"{92 - latest['% within 18 weeks']:.1f} pts",
            "goal: by 2029"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=national["Month_dt"], y=national["% within 18 weeks"],
                              name="% within 18 weeks", mode="lines+markers", line=dict(color="#2dd4bf")))
    fig.add_hline(y=65, line_dash="dash", line_color="#fbbf24",
                  annotation_text="65% interim target (Mar 2026)", annotation_position="bottom right")
    fig.add_hline(y=92, line_dash="dot", line_color="#34d399",
                  annotation_text="92% constitutional standard (2029)", annotation_position="top right")
    fig.update_layout(template="plotly_dark", height=450, title="National % Within 18 Weeks",
                       xaxis_title="Month", yaxis_title="% within 18 weeks",
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e5ea'))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=national["Month_dt"], y=national["% 52 plus weeks"],
                               name="% 52+ weeks", mode="lines+markers", line=dict(color="#f472b6")))
    fig2.add_hline(y=1, line_dash="dash", line_color="#34d399", annotation_text="<1% target")
    fig2.update_layout(template="plotly_dark", height=350, title="National % Waiting 52+ Weeks",
                        xaxis_title="Month", yaxis_title="% waiting 52+ weeks",
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e5ea'))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📍 Latest Month — Regional Breakdown")
    latest_month_dt = region_comp["Month_dt"].max()
    latest_region = region_comp[region_comp["Month_dt"] == latest_month_dt].copy()
    latest_region["Region Name"] = latest_region["Region Code"].map(REGION_NAMES)
    latest_region = latest_region.sort_values("% within 18 weeks", ascending=False)

    fig3 = px.bar(latest_region, x="Region Name", y="% within 18 weeks", template="plotly_dark",
                  color="% within 18 weeks", color_continuous_scale="Tealgrn", range_color=[40, 80])
    fig3.add_hline(y=65, line_dash="dash", line_color="white", annotation_text="65% target")
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e5ea'))
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        latest_region[["Region Name", "Total number of incomplete pathways", "% within 18 weeks", "% 52 plus weeks"]]
        .style.format({
            "Total number of incomplete pathways": "{:,}",
            "% within 18 weeks": "{:.1f}%",
            "% 52 plus weeks": "{:.2f}%"
        }),
        use_container_width=True
    )

    with st.expander("🏥 Trust-level performance (latest month)"):
        latest_provider = provider_comp[provider_comp["Month_dt"] == provider_comp["Month_dt"].max()].copy()
        latest_provider["Region Name"] = latest_provider["Region Code"].map(REGION_NAMES)
        latest_provider = latest_provider.sort_values("% within 18 weeks")
        st.dataframe(
            latest_provider[["Provider Name", "Region Name", "Total number of incomplete pathways",
                              "% within 18 weeks", "% 52 plus weeks"]]
            .style.format({
                "Total number of incomplete pathways": "{:,}",
                "% within 18 weeks": "{:.1f}%",
                "% 52 plus weeks": "{:.2f}%"
            }),
            height=400,
            use_container_width=True
        )

# Regional Map
elif page == "Regional Map":
    st.title("🗺️ Regional Map — NHS England")

    metric_choice = st.radio(
        "Metric to map:",
        ["Average median wait (weeks)", "% within 18 weeks", "% 52+ weeks"],
        horizontal=True
    )

    geojson = load_region_geojson(os.path.getmtime(_geojson_path))

    if metric_choice == "Average median wait (weeks)":
        map_source = df.groupby(["Region Code", "Month"], as_index=False)["Average (median) waiting time (in weeks)"].mean()
        months_avail = sorted(map_source["Month"].unique())
        sel_month = st.select_slider("Month", options=months_avail, value=months_avail[-1],
                                      format_func=lambda d: d.strftime("%b %Y"))
        plot_df = map_source[map_source["Month"] == sel_month].copy()
        color_col = "Average (median) waiting time (in weeks)"
        color_scale = "Tealgrn"
    else:
        region_comp = load_region_compliance(os.path.getmtime(_region_comp_path))
        col_map = {"% within 18 weeks": "% within 18 weeks", "% 52+ weeks": "% 52 plus weeks"}
        color_col = col_map[metric_choice]
        months_avail = sorted(region_comp["Month_dt"].unique())
        sel_month = st.select_slider("Month", options=months_avail, value=months_avail[-1],
                                      format_func=lambda d: d.strftime("%b %Y"))
        plot_df = region_comp[region_comp["Month_dt"] == sel_month].copy()
        color_scale = "RdYlGn" if metric_choice == "% within 18 weeks" else "RdYlGn_r"

    plot_df["Region Name"] = plot_df["Region Code"].map(REGION_NAMES)

    fig = px.choropleth(
        plot_df, geojson=geojson, locations="Region Code", featureidkey="properties.Region_Code",
        color=color_col, color_continuous_scale=color_scale, hover_name="Region Name",
        projection="mercator"
    )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor='rgba(0,0,0,0)')
    fig.update_layout(template="plotly_dark", height=650, margin={"r": 0, "t": 30, "l": 0, "b": 0},
                       paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e5ea'))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        plot_df[["Region Name", color_col]].sort_values(color_col, ascending=False)
        .style.format({color_col: "{:.1f}"}),
        use_container_width=True
    )

# Anomaly Detection
elif page == "Anomaly Detection":
    st.title("🚨 Anomaly Detection Center")
    col1, col2 = st.columns(2)
    with col1:
        sensitivity = st.slider("Detection Sensitivity", 0.01, 0.5, 0.1, 0.01)
    with col2:
        min_duration = st.slider("Minimum Anomaly Duration (months)", 1, 6, 2)

    clf = IsolationForest(contamination=sensitivity, random_state=42)
    X = df[["Average (median) waiting time (in weeks)"]].fillna(0)
    df["Anomaly"] = clf.fit_predict(X)
    anomalies = df[df["Anomaly"] == -1]

    if not anomalies.empty:
        for _, row in anomalies.iterrows():
            with st.container():
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #ef4444, #dc2626);
                            padding: 1rem;
                            border-radius: 12px;
                            margin: 0.5rem 0;">
                    <h4 style="color: white;">⚠️ Anomaly Detected: {row['Provider Name']} - {row['Month'].strftime('%b %Y')}</h4>
                    <p style="color: white;">Waiting Time: {row['Average (median) waiting time (in weeks)']:.1f} weeks | Region: {row['Region Code']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 No anomalies detected with current settings")

# Forecasting
elif page == "Forecasting":
    st.title("🔮 Waiting Time Predictions")
    trust_options = ["All Trusts"] + sorted(df["Provider Name"].unique())
    selected_trust = st.selectbox("Select Trust to Forecast:", trust_options)

    if selected_trust == "All Trusts":
        forecast_df = df.groupby("Month")["Average (median) waiting time (in weeks)"].mean().reset_index()
    else:
        forecast_df = df[df["Provider Name"] == selected_trust]

    col1, col2 = st.columns(2)
    with col1:
        forecast_months = st.slider("Forecast Period (months)", 3, 12, 6)
    with col2:
        confidence_interval = st.slider("Confidence Interval", 0.8, 0.99, 0.95)

    if len(forecast_df) < 2:
        st.warning("Not enough historical data to generate forecast for this trust")
        st.stop()

    prophet_df = forecast_df.rename(columns={'Month': 'ds', 'Average (median) waiting time (in weeks)': 'y'})[['ds', 'y']].dropna()
    model = Prophet(interval_width=confidence_interval)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_months, freq='M')
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction', line=dict(color='#3b82f6')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(color='#9333ea', dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(color='#9333ea', dash='dot')))
    fig.update_layout(template="plotly_dark", title=f"{forecast_months}-Month Forecast for {selected_trust}", xaxis_title="Date", yaxis_title="Waiting Time (weeks)", hovermode="x unified", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Details")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_months).rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Weeks',
        'yhat_lower': 'Minimum Estimate',
        'yhat_upper': 'Maximum Estimate'
    }).style.format({'Predicted Weeks': '{:.1f}', 'Minimum Estimate': '{:.1f}', 'Maximum Estimate': '{:.1f}'}))

# Raw Data
elif page == "Raw Data":
    st.title("📊 Data Explorer")
    col1, col2 = st.columns(2)
    with col1:
        regions = st.multiselect("Filter Regions", df['Region Code'].unique())
    with col2:
        providers = st.multiselect("Filter Providers", df['Provider Name'].unique())

    filtered_df = df
    if regions:
        filtered_df = filtered_df[filtered_df['Region Code'].isin(regions)]
    if providers:
        filtered_df = filtered_df[filtered_df['Provider Name'].isin(providers)]

    st.dataframe(
        filtered_df.style.format({
            'Average (median) waiting time (in weeks)': '{:.1f}',
            'Total number of incomplete pathways': '{:,}'
        }).background_gradient(cmap='magma'),
        height=600,
        use_container_width=True
    )

# Chat with NHS AI
elif page == "Chat with NHS AI":
    st.title("🤖 Chat with NHS AI Assistant")
    st.markdown("Ask anything about NHS trends, forecasts, or healthcare analytics. The assistant knows your current dataset!")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    # ✅ Summarize dataset context (use key stats only)
    latest_month = df['Month'].max().strftime('%B %Y')
    summary_text = f"""
You are an NHS data assistant. You must answer based on the current dataset of monthly waiting times (April 2024 to March 2025).

**Dataset Summary**:
- Number of records: {len(df)}
- Time range: {df['Month'].min().strftime('%B %Y')} to {df['Month'].max().strftime('%B %Y')}
- Number of Trusts: {df['Provider Name'].nunique()}
- Average waiting time (latest month: {latest_month}): {df[df['Month'] == df['Month'].max()]['Average (median) waiting time (in weeks)'].mean():.2f} weeks
- Highest incomplete pathways: {df['Total number of incomplete pathways'].max():,}
- Sample Trusts: {", ".join(df['Provider Name'].unique()[:3])}...

You must answer clearly using ONLY the dataset knowledge. Do not guess beyond this information unless the question is general NHS knowledge.
    """

    # User input
    user_input = st.text_area("Ask NHS AI a question", placeholder="e.g., What are the recent waiting time trends?")
    if user_input:
        with st.spinner("Thinking..."):
            prompt = summary_text + f"\n\nUser Question: {user_input}"

            response = requests.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}"
                },
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [
                        {"role": "system", "content": "You are a helpful NHS data assistant."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                st.success("✅ NHS AI Response:")
                st.markdown(answer)
            else:
                st.error("❌ Failed to get a response from the AI. Please try again.")
