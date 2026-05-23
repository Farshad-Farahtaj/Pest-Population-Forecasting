# Pest Population Forecasting

![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/streamlit-deployed-red)

Dual-task ML pipeline predicting insect population counts (regression) and catch-event occurrence (classification) from meteorological and entomological sensor data. Replaces manual field observation with automated risk estimation.

## Live Demo

[Pest Risk Forecasting Dashboard](https://huggingface.co/spaces/parhamkhoshsolat/pest-prediction-dashboard)

## Problem

Field managers estimated pest risk through manual observation: slow, weather-dependent, and hard to scale. This pipeline takes sensor inputs and outputs a population count estimate and a binary catch-event prediction, with feature importance reports for non-technical stakeholders.

## Approach

### Data

Merged meteorological sensor readings with historical entomological catch records. Lag features capture delayed weather-pest relationships. Full preprocessing covers imputation, scaling, and encoding.

### Models

**Regression** (evaluated on MAE): Random Forest, Gradient Boosting, Ridge, SVR, XGBoost, LightGBM

**Classification** (evaluated on F1): Random Forest, Gradient Boosting, Logistic Regression, SVM, XGBoost, LightGBM

Best models selected through stratified cross-validation. Feature importance reports generated for all final models.

## Project Structure

```
├── notebooks/
│   ├── Notebook_1_Data_Preprocessing_&_EDA.ipynb
│   ├── Notebook_2_Regression_Modeling.ipynb
│   └── Notebook_3_Classification_Modeling.ipynb
├── models/
├── utils_eda.py
├── utils_regression.py
├── utils_classification.py
├── app.py
├── cleaned_engineered_data.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/parhamkhoshsolat/pest-population-forecasting.git
cd pest-population-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## Course

Information Systems & Business Intelligence — University of Naples Federico II
Grade: 30/30

## License

MIT
