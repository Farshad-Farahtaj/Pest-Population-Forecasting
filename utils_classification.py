# utils_classification.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import joblib
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

_this_file_dir = os.path.dirname(os.path.abspath(__file__))
_models_dir = os.path.join(_this_file_dir, 'models') # Path to the models folder


@st.cache_resource
def load_classification_artifacts():
    """Loads all pre-trained classification models and artifacts."""
    try:
        from tensorflow.keras.models import load_model

        # --- CORRECTED FILENAMES ---
        # Load all the trained model objects with the correct names from your saving script
        models = {
            "RandomForest": joblib.load(os.path.join(_models_dir, 'clf_random_forest.joblib')),
            "XGBoost": joblib.load(os.path.join(_models_dir, 'clf_xgboost.joblib')),
            "LightGBM": joblib.load(os.path.join(_models_dir, 'clf_lightgbm.joblib')),
            "LSTM": load_model(os.path.join(_models_dir, 'clf_lstm.h5')),
            "GRU": load_model(os.path.join(_models_dir, 'clf_gru.h5')),
        }

        # Load all saved results and other artifacts
        artifacts = {}
        with open(os.path.join(_models_dir, 'part1_standard_tournament_results.json'), 'r') as f:
            artifacts['part1_results'] = json.load(f)
        with open(os.path.join(_models_dir, 'part2_deep_learning_tournament_results.json'), 'r') as f:
            artifacts['part2_results'] = json.load(f)

        # Load the scaler used for the standard ML models
        scaler = joblib.load(os.path.join(_models_dir, 'clf_scaler_ml.joblib'))
        
        return models, artifacts, scaler
    except Exception as e:
        st.error(f"Error loading classification artifacts: {e}. Please ensure you have run the saving script in your notebook to generate all necessary files in the 'models' directory.")
        return None, None, None

    

def plot_class_imbalance(df):
    """Plots the class imbalance from the dataframe."""
    st.subheader("⚖️ Class Imbalance Diagnosis")
    df_copy = df.copy()
    df_copy['New catches'] = (df_copy['New catches'] > 0).astype(int)
    class_counts = df_copy['New catches'].value_counts().sort_index()
    fig = px.bar(x=['No Catch (0)', 'Catch (1)'], y=class_counts.values, title="🎯 Target Variable Distribution", color=class_counts.values, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_classification_results(y_true, y_pred, y_proba, model_name, color):
    """Generates and displays a full suite of classification plots."""
    st.markdown("##### Classification Report")
    st.text(classification_report(y_true, y_pred, target_names=['No Catch', 'Catch']))

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = ff.create_annotated_heatmap(z=cm, x=['Predicted No', 'Predicted Yes'], y=['Actual No', 'Actual Yes'], colorscale='Blues', showscale=False)
        fig_cm.update_layout(title=f'Confusion Matrix: {model_name}', height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc:.4f}', line=dict(color=color, width=3)))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(title=f'ROC Curve: {model_name}', height=400, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', legend=dict(x=0.05, y=0.95))
        st.plotly_chart(fig_roc, use_container_width=True)

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(10)
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title=f'Top 10 Feature Importances: {model_name}')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def plot_dl_history(history, model_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Model Loss', 'Model AUC'])
    epochs = list(range(1, len(history['loss']) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=history['loss'], name='Training Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history.get('auc', []), name='Training AUC'), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history.get('val_auc', []), name='Validation AUC'), row=1, col=2)
    fig.update_layout(title_text=f"Training History: {model_name}", height=400)
    st.plotly_chart(fig, use_container_width=True)