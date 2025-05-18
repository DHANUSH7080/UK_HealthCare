import streamlit as st
import pandas as pd
from pandas import to_datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from streamlit_option_menu import option_menu
from datetime import datetime
import os
import requests
from dotenv import load_dotenv
load_dotenv()

# Page config
st.set_page_config(page_title="NHS Dynamic Dashboard", layout="wide")

# Custom CSS
st.markdown("""
<style>
:root {
    --primary: #3b82f6;
    --secondary: #9333ea;
}

body {
    background: linear-gradient(to right bottom, #0f172a, #1e293b);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

.sidebar .sidebar-content {
    background: linear-gradient(to bottom right, #1e293b, #0f172a);
    border-right: 1px solid #334155;
}

h1, h2, h3, h4, h5 {
    background: linear-gradient(45deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
}

.stSelectbox>div>div>div {
    color: #000 !important;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
}

.plot-container {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border-radius: 15px;
    padding: 1rem;
    box-shadow: 0 4px 25px rgba(59, 130, 246, 0.2);
}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "NHS_Trusts_Merged_2024_2025.csv")
    return pd.read_csv(file_path)

df = load_data()
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')

# Navigation
with st.sidebar:
    page = option_menu(
        "NHS Dashboard",
        ["Home - Trends", "Anomaly Detection", "Forecasting", "Raw Data", "Chat with NHS AI"],
        icons=['bar-chart-line', 'exclamation-triangle', 'graph-up', 'table', 'chat-dots'],
        menu_icon="hospital",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#e2e8f0"},
            "nav-link-selected": {"background": "linear-gradient(45deg, #3b82f6, #9333ea)", "border-radius": "8px"},
        },
    )

# Metric Card
def create_metric_card(label, value, delta=None):
    card = f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #94a3b8;">{label}</div>
        <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">{value}</div>
        {f'<div style="color: #10b981; font-size: 0.8rem;">▲ {delta}</div>' if delta else ''}
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
                  markers=True, color_discrete_sequence=['#3b82f6'])
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#334155'),
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
                          color_discrete_sequence=px.colors.sequential.Magma)
            st.plotly_chart(fig, use_container_width=True)

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
       for _, row in df.iterrows():
           month = to_datetime(row['Month'])  # safely convert
           st.markdown(f"""
           <div style="background-color: red; padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
              <h4 style="color: white;">⚠️ Anomaly Detected: {month.strftime('%B %Y')}</h4>
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
    latest_month = pd.to_datetime(df['Month'].max()).strftime('%B %Y')
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
