# utils_eda.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

_this_file_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    """Loads the cleaned datasets required for the app."""
    try:
        # --- Construct absolute paths to the data files ---
        path_engineered = os.path.join(_this_file_dir, 'cleaned_engineered_data.csv')
        path_merged = os.path.join(_this_file_dir, 'cleaned_merged_data.csv')

        engineered_data = pd.read_csv(path_engineered)
        engineered_data['Date'] = pd.to_datetime(engineered_data['Date'])
        ts_data = pd.read_csv(path_merged)
        ts_data['Date'] = pd.to_datetime(ts_data['Date'])
        return engineered_data, ts_data
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure 'cleaned_engineered_data.csv' and 'cleaned_merged_data.csv' are present in the main directory.")
        return None, None
    

def create_target_variable_plots(df):
    """Creates all plots related to the target variable analysis."""
    st.markdown("#### Distribution of Insect Counts")
    fig_hist = px.histogram(df, x='Number of insects', labels={'Number of insects': 'Number of Insects', 'count': 'Frequency'}, nbins=30, color_discrete_sequence=['skyblue'])
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Insect Counts by Location")
    fig_box = px.box(df, x='Location', y='Number of insects', color='Location', labels={'Number of insects': 'Number of Insects'})
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("#### Daily Insect Activity Over Time")
    daily_stats = df.groupby('Date')['Number of insects'].agg(['sum', 'mean']).reset_index()
    fig_time = px.line(daily_stats, x='Date', y=['sum', 'mean'], labels={'value': 'Number of Insects', 'variable': 'Metric'})
    st.plotly_chart(fig_time, use_container_width=True)

def create_correlation_plots(df):
    """Creates all plots related to correlation analysis."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_cols].corr()
    st.markdown("#### Full Feature Correlation Matrix")
    fig_corr_full = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr_full, use_container_width=True)

def create_weather_analysis_plots(df):
    """Creates all plots for the weather patterns section."""
    st.markdown("#### Temperature vs. Insect Activity")
    fig_temp = px.scatter(df, x='Average Temperature', y='Number of insects', color='Number of insects', color_continuous_scale='Viridis')
    st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("#### Humidity vs. Insect Activity")
    fig_humidity = px.scatter(df, x='Average Humidity', y='Number of insects', color='Number of insects', color_continuous_scale='Plasma')
    st.plotly_chart(fig_humidity, use_container_width=True)
    