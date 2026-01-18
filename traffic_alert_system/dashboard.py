import streamlit as st
import sqlite3
import pandas as pd
import time
import plotly.express as px

# Configuration
DB_PATH = 'traffic_data.db'

st.set_page_config(
    page_title="City Traffic Analytics",
    page_icon="🚦",
    layout="wide",
)

def load_data():
    conn = sqlite3.connect(DB_PATH)
    # Load last 1000 records for performance
    df = pd.read_sql_query("SELECT * FROM traffic_logs ORDER BY timestamp DESC LIMIT 2000", conn)
    conn.close()
    
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

# Header
st.title("🚦 Real-Time Traffic Analytics Dashboard")
st.markdown("### Monitoring Traffic Flow & Classification")

# Auto-refresh logic
placeholder = st.empty()

while True:
    df = load_data()
    
    with placeholder.container():
        if df.empty:
            st.warning("No data found. Please run the traffic detection system first.")
        else:
            # Metrics
            total_vehicles = len(df)
            last_minute_count = len(df[df['timestamp'] > (time.time() - 60)])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Vehicles Detected", total_vehicles)
            col2.metric("Flow Rate (Vehicles/Min)", last_minute_count)
            col3.metric("Most Common Vehicle", df['vehicle_type'].mode()[0] if not df['vehicle_type'].empty else "N/A")

            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Vehicle Type Distribution")
                fig_pie = px.pie(df, names='vehicle_type', hole=0.4, title="Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                st.subheader("Traffic Flow Over Time")
                # Resample count per minute
                df_resampled = df.set_index('datetime').resample('1min').count()['id'].reset_index()
                df_resampled.columns = ['Time', 'Count']
                
                fig_line = px.line(df_resampled, x='Time', y='Count', title="Vehicles Per Minute")
                st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("### Recent Detections Log")
            st.dataframe(df[['datetime', 'vehicle_type', 'confidence', 'vehicle_id']].head(10), use_container_width=True)

    # Refresh every 2 seconds
    time.sleep(2)
