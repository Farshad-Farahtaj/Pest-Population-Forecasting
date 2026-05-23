# utils_regression.py
# This file contains all helper functions extracted from Notebook_2_Regression_Modeling.ipynb

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

_this_file_dir = os.path.dirname(os.path.abspath(__file__))
_models_dir = os.path.join(_this_file_dir, 'models') # Path to the models folder

# --- DATA & MODEL LOADING ---

@st.cache_data
def get_data_splits(engineered_data, ts_data):
    """Creates consistent train/test splits for all models."""
    ts_daily = ts_data.groupby('Date').agg({
        'Number of insects': 'sum',
        'Average Temperature': 'mean',
        'Average Humidity': 'mean'
    }).reset_index().sort_values('Date')
    
    train_size = int(0.8 * len(ts_daily))
    train_dates = ts_daily['Date'].iloc[:train_size]
    
    ml_train = engineered_data[engineered_data['Date'] <= train_dates.max()].copy()
    ml_test = engineered_data[engineered_data['Date'] > train_dates.max()].copy()
    
    ts_train = ts_daily.iloc[:train_size].copy()
    ts_test = ts_daily.iloc[train_size:].copy()
    
    return ml_train, ml_test, ts_train, ts_test

@st.cache_resource
def load_regression_models():
    """Loads all pre-trained regression models and the scaler."""
    try:
        models = {
            "ARIMAX": joblib.load(os.path.join(_models_dir, 'arimax_model.joblib')),
            "SARIMAX": joblib.load(os.path.join(_models_dir, 'sarimax_model.joblib')),
            "Prophet": joblib.load(os.path.join(_models_dir, 'prophet_model.joblib')),
            "Random Forest": joblib.load(os.path.join(_models_dir, 'rf_model.joblib')),
            "XGBoost": joblib.load(os.path.join(_models_dir, 'xgb_model.joblib')),
            "LightGBM": joblib.load(os.path.join(_models_dir, 'lgb_model.joblib'))
        }
        scaler = joblib.load(os.path.join(_models_dir, 'scaler.joblib'))
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure all .joblib files are in the 'models/' directory.")
        return None, None



# --- UTILITY FUNCTIONS ---

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def ensure_non_negative_int(predictions):
    """Ensure predictions are non-negative integers."""
    return np.maximum(np.round(predictions), 0).astype(int)

def generate_future_dates(last_date, days=7):
    """Generate future dates safely."""
    return pd.date_range(start=last_date, periods=days + 1, freq='D')[1:]

def aggregate_ml_data_for_plotting(ml_data, y_pred):
    """Aggregate ML prediction data by date for clean plotting."""
    df = ml_data[['Date', 'Number of insects']].copy()
    df['Predicted'] = y_pred
    agg_df = df.groupby('Date').agg({'Number of insects': 'sum', 'Predicted': 'sum'}).reset_index().sort_values('Date')
    return agg_df

# --- PLOTTING FUNCTIONS ---

def create_continuous_forecast_plot(historical_actual, test_actual, test_pred, future_pred, 
                                   dates_hist, dates_test, dates_future, title, 
                                   confidence_lower=None, confidence_upper=None,
                                   future_confidence_lower=None, future_confidence_upper=None):
    """
    Create a comprehensive and continuous forecast visualization with confidence intervals.
    """
    fig = go.Figure()
    
    # Plot historical actual data
    fig.add_trace(go.Scatter(x=dates_hist, y=historical_actual, mode='lines', name='Historical Data', line=dict(color='#1f77b4')))
    
    # Plot test period actual data
    fig.add_trace(go.Scatter(x=dates_test, y=test_actual, mode='lines+markers', name='Actual (Test Period)', line=dict(color='#2ca02c'), marker=dict(size=6)))
    
    # Plot test period predictions
    fig.add_trace(go.Scatter(x=dates_test, y=ensure_non_negative_int(test_pred), mode='lines+markers', name='Test Predictions', line=dict(color='#ff7f0e', dash='dash'), marker=dict(symbol='x', size=6)))
    
    # Plot future forecast predictions
    if future_pred is not None and dates_future is not None:
        fig.add_trace(go.Scatter(x=dates_future, y=ensure_non_negative_int(future_pred), mode='lines+markers', name='7-Day Forecast', line=dict(color='#d62728', dash='dot'), marker=dict(symbol='star', size=6)))

    # Add confidence intervals for test predictions
    if confidence_lower is not None and confidence_upper is not None:
        fig.add_trace(go.Scatter(x=dates_test, y=ensure_non_negative_int(confidence_upper), mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=dates_test, y=ensure_non_negative_int(confidence_lower), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', name='95% Confidence (Test)'))

    # Add confidence intervals for future forecast
    if future_confidence_lower is not None and future_confidence_upper is not None:
        fig.add_trace(go.Scatter(x=dates_future, y=ensure_non_negative_int(future_confidence_upper), mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=dates_future, y=ensure_non_negative_int(future_confidence_lower), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(214, 39, 40, 0.2)', name='95% Confidence (Forecast)'))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Number of Insects', template='plotly_white', height=500, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def generate_ml_confidence_intervals(model, X_data, n_bootstrap=25):
    """Generate confidence intervals for ML models using optimized bootstrap sampling."""
    predictions = []
    n_samples = X_data.shape[0]
    if n_samples == 0:
        return np.array([]), np.array([])
    for _ in range(n_bootstrap):
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X_data[bootstrap_indices]
        noise = np.random.normal(0, 0.01, X_bootstrap.shape)
        X_noisy = X_bootstrap + noise
        pred = model.predict(X_noisy)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
    return ensure_non_negative_int(lower_bound), ensure_non_negative_int(upper_bound)

def create_champion_comparison_plot(full_actual_dates, full_actual_y, test_dates, test_actual_y,
                                    pred1_y, pred1_name, pred2_y, pred2_name, title):
    """Creates a side-by-side plot for two champion models against actuals for the full timeline."""
    fig = go.Figure()
    # Full historical actuals
    fig.add_trace(go.Scatter(x=full_actual_dates, y=full_actual_y, mode='lines', name='Actual Data', line=dict(color='#1f77b4', width=3)))
    # Model 1 Predictions
    fig.add_trace(go.Scatter(x=test_dates, y=pred1_y, mode='lines', name=pred1_name, line=dict(color='#ff7f0e', dash='dash')))
    # Model 2 Predictions
    fig.add_trace(go.Scatter(x=test_dates, y=pred2_y, mode='lines', name=pred2_name, line=dict(color='#2ca02c', dash='dot')))
    
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Number of Insects', template='plotly_white', height=500, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# This is the main plotting function we will use everywhere
def create_full_forecast_plot(title, train_dates, train_y, test_dates, test_y, test_pred, 
                             future_dates, future_pred, test_ci_lower=None, test_ci_upper=None, 
                             future_ci_lower=None, future_ci_upper=None):
    """A master function to create a complete forecast plot with history, test predictions, and future forecast."""
    fig = go.Figure()
    
    # 1. Historical Data
    fig.add_trace(go.Scatter(x=train_dates, y=train_y, mode='lines', name='Historical Data', line=dict(color='#1f77b4')))
    
    # 2. Actual Test Data
    fig.add_trace(go.Scatter(x=test_dates, y=test_y, mode='lines', name='Actual (Test)', line=dict(color='#2ca02c', width=2.5)))
    
    # 3. Test Predictions
    fig.add_trace(go.Scatter(x=test_dates, y=test_pred, mode='lines', name='Predicted (Test)', line=dict(color='#ff7f0e', dash='dash')))
    if test_ci_lower is not None and test_ci_upper is not None:
        fig.add_trace(go.Scatter(x=test_dates, y=test_ci_upper, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=test_dates, y=test_ci_lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', name='95% Confidence'))

    # 4. Future Forecast
    if future_pred is not None:
        fig.add_trace(go.Scatter(x=future_dates, y=future_pred, mode='lines', name='7-Day Forecast', line=dict(color='#d62728', dash='dot')))
        if future_ci_lower is not None and future_ci_upper is not None:
            fig.add_trace(go.Scatter(x=future_dates, y=future_ci_upper, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=future_dates, y=future_ci_lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(214, 39, 40, 0.2)', name='Future Confidence'))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Number of Insects', template='plotly_white', height=500, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
