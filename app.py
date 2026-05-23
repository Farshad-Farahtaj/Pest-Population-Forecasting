# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils_eda
import utils_regression
import utils_classification

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pest Forecasting Project", page_icon="🐞", layout="wide")


# --- LOAD DATA AND MODELS (runs once) ---
@st.cache_data
def load_all_data():
    """Load all datasets and create splits."""
    engineered_data, ts_data = utils_eda.load_data()
    if engineered_data is not None and ts_data is not None:
        ml_train, ml_test, ts_train, ts_test = utils_regression.get_data_splits(engineered_data, ts_data)
        return engineered_data, ts_data, ml_train, ml_test, ts_train, ts_test
    return [None] * 6

@st.cache_resource
def load_all_models():
    """Load all trained regression models and the scaler."""
    models, scaler = utils_regression.load_regression_models()
    return models, scaler

@st.cache_resource
def load_classification_artifacts():
    """Loads all pre-trained classification models and artifacts."""
    models, artifacts, scaler = utils_classification.load_classification_artifacts()
    return models, artifacts, scaler


# --- Execute all loading functions ---
engineered_df, ts_data, ml_train, ml_test, ts_train, ts_test = load_all_data()
if engineered_df is not None:
    locations = engineered_df['Location'].unique()
models, scaler = load_all_models()
class_models, class_artifacts, class_scaler = load_classification_artifacts()


# --- Main Application Tabbing Structure ---
homepage_tab, presentation_tab, forecasting_tab = st.tabs([
    "🏠 Project Homepage",
    "📊 Presentation",
    "🔮 Live Forecasting"
])


# --- HOMEPAGE TAB ---
with homepage_tab:
    st.title("🐞 Pest Forecasting with Meteorological Data")
    st.header("Information Systems & Business Intelligence Final Project")
    st.markdown("Welcome to our final project! In this series, we demonstrate how data science and business intelligence can help predict pest outbreaks using weather and capture data.")
    st.markdown("---")
    st.header("📖 Project Overview")
    st.markdown("Our goal is to analyze meteorological and pest capture data to build predictive models for:\n- **Regression:** Predicting the exact number of insects caught each day.\n- **Classification:** Predicting whether there will be new pest captures (yes/no).\n\nWe use a combination of data science and business intelligence techniques to turn raw data into actionable insights for pest management.")
    st.markdown("---")
    st.header("👨‍💻 Meet the Team")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Seyyedalireza Khosh Solat")
        st.markdown("- Designed ML model tournaments architecture\n- Implemented advanced ML algorithms\n- Developed time series forecasting components\n- Led model evaluation methodology")
    with col2:
        st.subheader("Giovanni Montanile")
        st.markdown("- Developed Power BI dashboards\n- Created business intelligence solutions\n- Designed interactive visualizations\n- Implemented data reporting systems")
    with col3:
        st.subheader("Farshad Farahtaj")
        st.markdown("- Led data preprocessing workflow\n- Conducted exploratory data analysis\n- Performed feature engineering\n- Implemented data validation protocols")


# --- PRESENTATION TAB (WITH NESTED SUB-TABS) ---
with presentation_tab:
    st.title("Technical Presentation and Model Analysis")

    # Define sub-tabs within the Presentation tab
    eda_sub_tab, reg_sub_tab, class_sub_tab = st.tabs([
        "📊 Data Loading & EDA",
        "🏆 Regression Tournament",
        "🏆 Classification Tournament"
    ])

    # --- SUB-TAB 1: DATA LOADING AND EDA ---
    with eda_sub_tab:
        st.header("🚦 Data Preprocessing & Exploratory Data Analysis (EDA)")
        if engineered_df is not None and ts_data is not None:
            # 1. Data Quality and Loading
            st.subheader("1. Data Quality and Loading")
            st.markdown("""
            Our analysis is built upon two distinct, clean datasets derived from the same source. Each is tailored for a specific modeling approach:
            - **Engineered Dataset (for ML Models)**: This granular dataset contains records for each trap location per day. It includes a rich set of engineered features designed to provide deep context for our classification and regression machine learning models.
            - **Time-Series Dataset (for Forecasting Models)**: This dataset aggregates daily data across all locations, creating a single time series for total insect counts and average weather conditions. It's specifically structured for time-series forecasting models like ARIMAX and Prophet.
            """)
            st.markdown("##### Engineered Dataset Preview")
            st.dataframe(engineered_df.head())
            st.markdown("##### Time-Series Dataset Preview")
            st.dataframe(ts_data.head())


            # 2. Data Cleaning and Preprocessing
            st.subheader("2. Data Cleaning & Feature Engineering")
            st.markdown("""
            To move beyond raw data, we engineered several new features to provide more powerful predictive signals for our models. This process creates context that the algorithms can learn from more effectively. Our key engineered variables include:
            - **`Temp_Range`**: The difference between the day's maximum and minimum temperatures. A wider range can indicate specific weather patterns that might influence insect behavior.
            - **`Temp_Avg_3d` & `Humidity_Avg_3d`**: 3-day rolling averages for temperature and humidity. These capture recent weather trends, as insect activity is often influenced by conditions over the past few days, not just a single day's snapshot.
            - **`Insects_Lag1` & `Insects_Lag3`**: The number of insects caught one and three days prior. This is crucial for capturing the cyclical nature and momentum of pest populations.
            - **`Recent_Activity`**: A binary flag indicating if there have been any captures in the last 7 days. This helps the model differentiate between currently active and inactive periods at a location.
            - **`Days_Since_Cleaning`**: Tracks the number of days since a trap was last serviced. The effectiveness of a trap can change over time due to saturation or bait degradation.
            - **`Season`**: Categorizes dates into distinct agricultural seasons (e.g., Early Summer, Mid Summer), allowing the model to learn broader seasonal patterns in pest activity.
            """)

            # 3. Exploratory Data Analysis (EDA)
            st.subheader("3. Exploratory Data Analysis (EDA)")
            st.markdown("In this section, we visually inspect the data to uncover patterns, identify relationships, and validate our assumptions.")
            
            st.markdown("#### Analyzing the Target Variable: Insect Counts")
            st.markdown("First, we examine the distribution of our primary target variable, `Number of insects`. The histogram shows that most days have very few captures, with a long tail of high-capture events, indicating that major pest activity is infrequent but significant.")
            utils_eda.create_target_variable_plots(engineered_df)
            
            st.markdown("The box plot further breaks this down by location, revealing that some locations (like `Cicalino 2`) are significantly more active than others (`Imola 3`). This confirms that location is a critical predictive factor.")
            st.markdown("#### Total Insects Caught by Location")
            st.markdown("To make the disparity between locations even clearer, the bar chart below shows the total cumulative captures. `Cicalino 1` and `Cicalino 2` are clear hotspots for pest activity throughout the data collection period.")
            total_by_loc = engineered_df.groupby('Location')['Number of insects'].sum().sort_values(ascending=False).reset_index()
            fig_bar_loc = px.bar(
                total_by_loc, x='Location', y='Number of insects',
                title='Total Cumulative Insect Captures per Location',
                text='Number of insects', color='Location'
            )
            fig_bar_loc.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_bar_loc.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig_bar_loc, use_container_width=True)


            st.markdown("#### Is Pest Activity Seasonal?")
            st.markdown("We investigate if there are broad seasonal trends in pest activity. The bar chart shows the average daily captures for each defined season.")
            seasonal_activity = engineered_df.groupby('Season')['Number of insects'].mean().reset_index()
            fig_season = px.bar(
                seasonal_activity, x='Season', y='Number of insects',
                title='Average Daily Insect Captures by Season',
                labels={'Number of insects': 'Average Daily Insects Caught'},
                color='Season', text='Number of insects'
            )
            fig_season.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_season, use_container_width=True)
            st.markdown("Interestingly, there is no dramatic difference in average daily captures between seasons. This suggests that while seasonality might play a role, more immediate factors like short-term weather changes are likely stronger drivers of day-to-day pest activity.")

            st.markdown("#### Distribution of Catch vs. No-Catch Days")
            st.markdown("For our classification task, we care about whether *any* insects were caught. This pie chart shows that on roughly 60% of observed days for any given trap, no new insects were caught. This creates a slight class imbalance that our classification models must handle.")
            df_copy_pie = engineered_df.copy()
            df_copy_pie['Catch Event'] = (df_copy_pie['New catches'] > 0).map({True: 'Catch (>=1 insect)', False: 'No Catch (0 insects)'})
            catch_counts = df_copy_pie['Catch Event'].value_counts()
            fig_pie = px.pie(
                values=catch_counts.values,
                names=catch_counts.index,
                title="Proportion of Days with and without New Pest Catches",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.1])
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("4. Correlation and Feature Analysis")
            st.markdown("Correlation matrices help us understand the linear relationships between variables. A value close to 1 (dark blue) indicates a strong positive correlation, while a value close to -1 (dark red) indicates a strong negative correlation.")
            
            st.markdown("##### Correlation Matrix (Engineered Dataset)")
            utils_eda.create_correlation_plots(engineered_df)

            st.markdown("##### Correlation Matrix (Time-Series Dataset)")
            st.markdown("The time-series data shows a moderate positive correlation between temperature and insect captures, and a weaker one with humidity.")
            ts_numeric_cols = ts_data.select_dtypes(include=np.number).columns
            ts_corr_matrix = ts_data[ts_numeric_cols].corr(numeric_only=True)
            fig_corr_ts = px.imshow(ts_corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_corr_ts, use_container_width=True)

            st.markdown("##### Top Features Correlated with Insect Counts")
            st.markdown("This chart isolates the features with the strongest absolute correlation to `Number of insects`. As expected, lagged insect counts (`Insects_Lag1`, `Recent_Activity`) are the most powerful predictors, followed by our engineered temperature features.")
            numeric_cols_corr = engineered_df.select_dtypes(include=np.number).columns
            corr_matrix_top10 = engineered_df[numeric_cols_corr].corr(numeric_only=True)
            target_column_corr = 'Number of insects'
            if target_column_corr in corr_matrix_top10:
                corr_target = corr_matrix_top10[[target_column_corr]].drop(target_column_corr).sort_values(by=target_column_corr, ascending=False)
                corr_target_abs = corr_target.abs()
                top_10 = corr_target_abs.head(10)
                fig_top10 = px.bar(
                    top_10, x=top_10.index, y=target_column_corr,
                    title=f'Top 10 Features Correlated with "{target_column_corr}"',
                    labels={'x': 'Feature', 'y': 'Absolute Correlation'}, color=top_10.index
                )
                st.plotly_chart(fig_top10, use_container_width=True)

            st.subheader("5. Weather Analysis")
            st.markdown("Here, we visualize the relationship between key weather variables and insect captures more directly.")

            st.markdown("##### Temperature vs. Insect Activity")
            st.markdown("This scatter plot shows that higher insect counts tend to occur when the average temperature is between 20°C and 30°C. Very high or very low temperatures seem to suppress activity.")
            fig_temp = px.scatter(engineered_df, x='Average Temperature', y='Number of insects', color='Number of insects', color_continuous_scale='Viridis',
                                  title='Insect Activity by Average Daily Temperature', trendline="lowess", trendline_color_override="red")
            st.plotly_chart(fig_temp, use_container_width=True)
            
            st.markdown("##### Humidity vs. Insect Activity")
            st.markdown("The relationship with humidity is less defined, though there's a slight tendency for higher captures to occur in the 60-80% humidity range.")
            fig_humidity = px.scatter(engineered_df, x='Average Humidity', y='Number of insects', color='Number of insects', color_continuous_scale='Plasma',
                                    title='Insect Activity by Average Daily Humidity', trendline="lowess", trendline_color_override="blue")
            st.plotly_chart(fig_humidity, use_container_width=True)

            st.markdown("##### 3D View: Weather vs. Insect Activity")
            st.markdown("This 3D plot combines temperature, humidity, and insect counts. The highest peaks (largest insect captures) are concentrated in the middle ranges of both temperature and humidity, confirming our observations from the 2D plots.")
            fig_3d = px.scatter_3d(
                engineered_df, x='Average Temperature', y='Average Humidity', z='Number of insects',
                color='Number of insects', title='3D Relationship between Weather and Insect Captures',
                labels={'Average Temperature': 'Avg. Temp (°C)', 'Average Humidity': 'Avg. Humidity (%)', 'Number of insects': 'Insect Count'},
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
            st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown("##### Interactive Weather Conditions by Location")
            st.markdown("Use the dropdown below to explore the specific temperature and humidity trends over time for each trap location.")
            locations_weather = engineered_df['Location'].unique()
            selected_loc_weather = st.selectbox("Select a Location to View Weather Data", options=locations_weather, key="weather_loc_select")

            loc_df_weather = engineered_df[engineered_df['Location'] == selected_loc_weather].copy()
            loc_df_weather = loc_df_weather.sort_values('Date')

            fig_weather = go.Figure()
            fig_weather.add_trace(go.Scatter(x=loc_df_weather['Date'], y=loc_df_weather['Average Temperature'], mode='lines', name='Avg Temp (°C)', line=dict(color='red')))
            fig_weather.add_trace(go.Scatter(x=loc_df_weather['Date'], y=loc_df_weather['Average Humidity'], mode='lines', name='Avg Humidity (%)', line=dict(color='blue'), yaxis='y2'))
            fig_weather.update_layout(
                title=f'Weather Conditions Over Time for {selected_loc_weather}', xaxis_title='Date', yaxis_title='Average Temperature (°C)',
                yaxis=dict(titlefont=dict(color='red'), tickfont=dict(color='red')),
                yaxis2=dict(title='Average Humidity (%)', overlaying='y', side='right', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
                legend=dict(x=0, y=1.2, orientation="h")
            )
            st.plotly_chart(fig_weather, use_container_width=True)
        else:
            st.error("Data could not be loaded. Please check your CSV files.")

    # --- SUB-TAB 2: REGRESSION TOURNAMENT ---
    with reg_sub_tab:
        st.header("🏆 Insect Count Regression Tournament")
        st.markdown("""
        Welcome to our **tournament-style regression analysis**. The goal of regression is to predict a continuous value—in this case, the exact number of insects we expect to capture. We test two families of models:
        1.  **Time Series Models**: These models (ARIMAX, SARIMAX, Prophet) are specialists that primarily learn from the sequence and seasonality of past insect captures. They use weather data as additional "exogenous" variables to improve their forecasts.
        2.  **Machine Learning Models**: These models (Random Forest, XGBoost) are more general-purpose. They excel at finding complex, non-linear relationships between a wide array of features (like lagged values, temperature ranges, and days since cleaning) and the target variable.

        In the plots below, the **green line** represents the true, observed insect counts. The **orange dashed line** is our model's prediction. A better model will have its orange line follow the green line as closely as possible.
        """)

        if not all([models, scaler, ml_train is not None, ts_train is not None, ts_test is not None, ts_data is not None]):
            st.error("Regression models or data could not be loaded. Please ensure all necessary files are present and correctly named in the `/models` directory.")
        else:
            # Prepare data for all models
            y_train_ts, y_test_ts = ts_train['Number of insects'], ts_test['Number of insects']
            X_test_ts = ts_test[['Average Temperature', 'Average Humidity']].values
            feature_cols = ['Location_Code', 'Average Temperature', 'Average Humidity', 'Temp_Range', 'Temp_Avg_3d', 'Humidity_Avg_3d', 'Insects_Lag1', 'Insects_Lag3', 'Recent_Activity', 'Days_Since_Cleaning', 'Month', 'Day']
            X_test_ml = ml_test[feature_cols]
            X_test_ml_scaled = scaler.transform(X_test_ml)
            last_date = ts_test['Date'].iloc[-1]
            future_dates = utils_regression.generate_future_dates(last_date, days=7)

            st.markdown("---")
            st.subheader("Part 1: Time Series Models Tournament")
            with st.expander("View All Time Series Model Forecasts and Metrics", expanded=True):
                for model_name in ["ARIMAX", "SARIMAX", "Prophet"]:
                    st.subheader(f"Contestant: {model_name}")
                    model = models[model_name]

                    test_ci_lower, test_ci_upper = None, None
                    future_ci_lower, future_ci_upper = None, None

                    if model_name == "Prophet":
                        future_df = model.make_future_dataframe(periods=len(ts_test) + 7, freq='D')
                        ts_daily_aggregated = ts_data.groupby('Date').agg({'Average Temperature': 'mean', 'Average Humidity': 'mean'}).reset_index()
                        ts_daily_aggregated.rename(columns={'Date': 'ds'}, inplace=True)
                        full_df = pd.merge(future_df, ts_daily_aggregated, on='ds', how='left')
                        full_df.fillna(method='ffill', inplace=True)
                        if 'Average Temperature' in full_df.columns and 'Average Humidity' in full_df.columns:
                            full_df.rename(columns={'Average Temperature': 'temp', 'Average Humidity': 'humidity'}, inplace=True)
                        forecast = model.predict(full_df)
                        test_pred = forecast.iloc[len(ts_train):len(ts_train) + len(ts_test)]['yhat']
                        future_pred = forecast.iloc[-7:]['yhat']
                        test_ci_lower = forecast.iloc[len(ts_train):len(ts_train) + len(ts_test)]['yhat_lower']
                        test_ci_upper = forecast.iloc[len(ts_train):len(ts_train) + len(ts_test)]['yhat_upper']
                        future_ci_lower = forecast.iloc[-7:]['yhat_lower']
                        future_ci_upper = forecast.iloc[-7:]['yhat_upper']

                    else:
                        future_exog = np.tile(X_test_ts.mean(axis=0), (7, 1))
                        full_exog = np.concatenate([X_test_ts, future_exog])
                        forecast_results = model.get_forecast(steps=len(ts_test) + 7, exog=full_exog)
                        
                        full_forecast = forecast_results.predicted_mean
                        conf_int = forecast_results.conf_int()

                        test_pred = full_forecast[:len(ts_test)]
                        future_pred = full_forecast[len(ts_test):]
                        
                        test_ci_lower = conf_int[:len(ts_test), 0]
                        test_ci_upper = conf_int[:len(ts_test), 1]
                        future_ci_lower = conf_int[len(ts_test):, 0]
                        future_ci_upper = conf_int[len(ts_test):, 1]
                    
                    test_pred_clean = utils_regression.ensure_non_negative_int(test_pred)
                    metrics = utils_regression.calculate_metrics(y_test_ts, test_pred_clean)
                    st.write(pd.DataFrame([metrics]))

                    fig = utils_regression.create_full_forecast_plot(
                        title=f"{model_name} Forecast vs. Actuals",
                        train_dates=ts_train['Date'], train_y=y_train_ts,
                        test_dates=ts_test['Date'], test_y=y_test_ts, 
                        test_pred=test_pred_clean,
                        future_dates=future_dates, 
                        future_pred=utils_regression.ensure_non_negative_int(future_pred),
                        test_ci_lower=utils_regression.ensure_non_negative_int(test_ci_lower),
                        test_ci_upper=utils_regression.ensure_non_negative_int(test_ci_upper),
                        future_ci_lower=utils_regression.ensure_non_negative_int(future_ci_lower),
                        future_ci_upper=utils_regression.ensure_non_negative_int(future_ci_upper)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.info("🏅 **Time Series Champion:** ARIMAX (as determined in the notebook).")
            st.markdown("---")

            st.subheader("Part 2: Machine Learning Models Tournament")
            with st.expander("View All Machine Learning Model Forecasts and Metrics", expanded=True):
                y_test_agg_actual = ml_test.groupby('Date')['Number of insects'].sum()
                for model_name in ["Random Forest", "XGBoost", "LightGBM"]:
                    st.subheader(f"Contestant: {model_name}")
                    model = models[model_name]
                    test_pred_ml = model.predict(X_test_ml_scaled)
                    
                    lower_bound, upper_bound = utils_regression.generate_ml_confidence_intervals(model, X_test_ml_scaled)

                    plot_df = utils_regression.aggregate_ml_data_for_plotting(ml_test, test_pred_ml)
                    lower_plot_df = utils_regression.aggregate_ml_data_for_plotting(ml_test, lower_bound)
                    upper_plot_df = utils_regression.aggregate_ml_data_for_plotting(ml_test, upper_bound)
                    
                    test_pred_agg = plot_df['Predicted']
                    metrics = utils_regression.calculate_metrics(y_test_agg_actual, test_pred_agg)
                    st.write(pd.DataFrame([metrics]))

                    fig = utils_regression.create_full_forecast_plot(
                        title=f"{model_name} Forecast vs. Actuals",
                        train_dates=ts_train['Date'], train_y=ts_train['Number of insects'],
                        test_dates=plot_df['Date'], test_y=y_test_agg_actual, 
                        test_pred=utils_regression.ensure_non_negative_int(test_pred_agg),
                        test_ci_lower=utils_regression.ensure_non_negative_int(lower_plot_df['Predicted']),
                        test_ci_upper=utils_regression.ensure_non_negative_int(upper_plot_df['Predicted']),
                        future_dates=None, future_pred=None
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.info("🏅 **ML Champion:** Random Forest")
            st.markdown("---")

            st.subheader("🏁 Grand Finale & Location-Specific Showdown")
            st.markdown("After evaluating both model families, we compare the champions. The **Random Forest** model, with its ability to handle complex feature interactions, significantly outperforms the best time-series model (ARIMAX).")
            st.dataframe(pd.DataFrame({'Model': ['ARIMAX (Time Series)', 'Random Forest (ML)'],'Test MAE': [2.000, 0.340],'Test R²': [0.050, 0.353]}))
            st.success("🎉 **Overall Regression Champion: Random Forest**!")
            st.markdown("Below, we use the champion Random Forest model to generate forecasts for each individual trap location, demonstrating its performance on a granular level.")
            with st.expander("View Champion Forecasts for Each Location"):
                champion_model = models['Random Forest']
                for location in locations:
                    st.subheader(f"Forecast for: {location}")
                    
                    location_train_df = ml_train[ml_train['Location'] == location]
                    location_test_df = ml_test[ml_test['Location'] == location]

                    if location_test_df.empty:
                        st.warning(f"No test data available for {location}.")
                        continue

                    X_test_location = location_test_df[feature_cols]
                    X_test_location_scaled = scaler.transform(X_test_location)
                    location_pred = champion_model.predict(X_test_location_scaled)
                    
                    lower_bound_loc, upper_bound_loc = utils_regression.generate_ml_confidence_intervals(champion_model, X_test_location_scaled)
                    
                    train_plot_df = location_train_df.groupby('Date')['Number of insects'].sum().reset_index()
                    test_actual_plot_df = location_test_df.groupby('Date')['Number of insects'].sum().reset_index()
                    
                    pred_plot_df = location_test_df[['Date']].copy()
                    pred_plot_df['Predicted'] = location_pred
                    pred_plot_df = pred_plot_df.groupby('Date')['Predicted'].sum().reset_index()

                    lower_plot_df_loc = location_test_df[['Date']].copy()
                    lower_plot_df_loc['Predicted'] = lower_bound_loc
                    lower_plot_df_loc = lower_plot_df_loc.groupby('Date')['Predicted'].sum().reset_index()

                    upper_plot_df_loc = location_test_df[['Date']].copy()
                    upper_plot_df_loc['Predicted'] = upper_bound_loc
                    upper_plot_df_loc = upper_plot_df_loc.groupby('Date')['Predicted'].sum().reset_index()

                    fig = utils_regression.create_full_forecast_plot(
                        title=f"Random Forest Forecast for {location}",
                        train_dates=train_plot_df['Date'],
                        train_y=train_plot_df['Number of insects'],
                        test_dates=test_actual_plot_df['Date'],
                        test_y=test_actual_plot_df['Number of insects'],
                        test_pred=utils_regression.ensure_non_negative_int(pred_plot_df['Predicted']),
                        test_ci_lower=utils_regression.ensure_non_negative_int(lower_plot_df_loc['Predicted']),
                        test_ci_upper=utils_regression.ensure_non_negative_int(upper_plot_df_loc['Predicted']),
                        future_dates=None,
                        future_pred=None
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("---")


    # --- SUB-TAB 3: CLASSIFICATION TOURNAMENT ---
    with class_sub_tab:
        st.header("🎯 Pest Classification Tournament")
        st.markdown("""
        Welcome to our **tournament-style classification analysis**. The goal of classification is to predict a discrete category. Here, we simplify the problem: instead of predicting the *exact* number of insects, we predict whether there will be a **Catch (1 or more insects)** or **No Catch (0 insects)**. This binary prediction is often more actionable for making simple "yes/no" decisions (e.g., "Do we need to take action today?").

        We evaluate two main groups of models:
        1.  **Standard Classifiers**: Classic, powerful algorithms like RandomForest, XGBoost, and LightGBM that are excellent at learning from tabular data.
        2.  **Deep Learning Models**: Sequence-based models like LSTM and GRU that are designed to learn from patterns over time.

        For evaluation, we use metrics like the **F1-Score** and **AUC**, which are robust for imbalanced datasets like ours.
        """)

        if not all([class_models, class_artifacts, engineered_df is not None, ml_test is not None]):
            st.warning("Classification artifacts not found. Please run the saving script in your notebook.")
        else:
            st.markdown("### Tournament Structure")
            st.markdown("We pit the models against each other in two rounds, followed by a grand finale to determine the ultimate champion.")
            
            st.markdown("---")
            st.subheader("🥊 Standard Classifiers Tournament")
            st.markdown("Here, we assess the performance of our standard machine learning models. Pay attention to the **Confusion Matrix**, which shows how many predictions were correct vs. incorrect, and the **ROC Curve**, where a curve closer to the top-left corner indicates a better model.")
            part1_results = class_artifacts['part1_results']
            feature_names_std = part1_results['dataset_info']['features_used']
            ml_test_encoded = pd.get_dummies(ml_test, columns=['Location', 'Season'], drop_first=False)
            current_features_df = ml_test_encoded.reindex(columns=feature_names_std, fill_value=0)
            X_test_std = class_scaler.transform(current_features_df)
            y_test_std = (ml_test['New catches'] > 0).astype(int)

            with st.expander("View Standard Classifier Performance Analysis", expanded=True):
                for model_name, color in zip(["RandomForest", "XGBoost", "LightGBM"], ["green", "orange", "purple"]):
                    st.subheader(f"{model_name}")
                    model = class_models[model_name]
                    y_pred = model.predict(X_test_std)
                    y_proba = model.predict_proba(X_test_std)[:, 1]
                    utils_classification.plot_classification_results(y_test_std, y_pred, y_proba, model_name, color)
                    utils_classification.plot_feature_importance(model, feature_names_std, model_name)
                    st.markdown("---")

            st.markdown("---")
            st.subheader("🧠 Deep Learning Tournament")
            st.markdown("Next, we evaluate our deep learning models. In addition to the standard classification plots, we also show the **Training History** for Loss and AUC. This helps us diagnose if the models learned effectively over their training epochs.")
            part2_results = class_artifacts['part2_results']
            
            sequence_length = part2_results['data_info'].get('sequence_length', 14) 
            
            def create_dl_sequences(X_data, y_data, seq_length):
                xs, ys = [], []
                y_data_np = np.array(y_data)
                for i in range(len(X_data) - seq_length):
                    xs.append(X_data[i:(i + seq_length)])
                    ys.append(y_data_np[i + seq_length])
                return np.array(xs), np.array(ys)
            
            X_test_dl_features = X_test_std[:, :7]
            
            X_test_dl, y_test_dl = create_dl_sequences(X_test_dl_features, pd.Series(y_test_std), sequence_length)

            with st.expander("View Deep Learning Model Performance Analysis", expanded=True):
                for model_name, color in zip(["LSTM", "GRU"], ["blue", "cyan"]):
                    st.subheader(f"{model_name}")
                    model = class_models[model_name]
                    history = part2_results.get('histories', {}).get(model_name)
                    
                    y_proba_dl = model.predict(X_test_dl).ravel()
                    y_pred_dl = (y_proba_dl > 0.5).astype(int)
                    
                    if history:
                        utils_classification.plot_dl_history(history, model_name)
                    else:
                        st.info(f"Training history not found in artifacts for {model_name}. Skipping history plot.")

                    utils_classification.plot_classification_results(y_test_dl, y_pred_dl, y_proba_dl, model_name, color)
                    st.markdown("---")

            st.markdown("---")
            st.subheader("🏅 Grand Finale")
            st.markdown("The final showdown pits the champion of the Standard ML bracket against the champion of the Deep Learning bracket. Based on F1-Score and AUC, we declare the overall winner.")
            final_standings = pd.DataFrame({
                 'Champion': ['🥊 RandomForest (Standard ML)', '🧠 LSTM (Deep Learning)'],
                 'F1-Score': [part1_results['champion']['f1_score'], part2_results['champion']['f1_score']],
                 'AUC': [part1_results['champion']['auc_score'], part2_results['champion']['auc_score']]
            })
            def highlight_winner(row):
                if 'RandomForest' in row['Champion']:
                    return ['background-color: lightgreen'] * len(row)
                else:
                    return [''] * len(row)
            st.dataframe(final_standings.style.apply(highlight_winner, axis=1))
            st.balloons()
            st.success("🎉 **Overall Classification Champion:** RandomForest!")


# --- LIVE FORECASTING TAB ---
with forecasting_tab:
    st.title("🔮 Live Forecasting Panel")
    st.markdown("Use the champion models to get live predictions based on your input.")

    if not all([models, scaler, class_models, class_artifacts, class_scaler, engineered_df is not None, ml_test is not None]):
        st.error("Models or scalers could not be loaded. Please ensure all artifacts are present.")
    else:
        location_mapping = {loc: code for code, loc in enumerate(engineered_df['Location'].unique())}
        st.header("🐜 Predict Insect Count (Regression)")
        st.markdown("Fill in the details below to predict the exact number of insects.")
        with st.form("regression_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                location = st.selectbox("Select Location", options=list(location_mapping.keys()), key="reg_loc")
                insects_lag1 = st.number_input("Insects Caught Yesterday (Lag 1)", min_value=0, value=5)
                insects_lag3 = st.number_input("Insects Caught 3 Days Ago (Lag 3)", min_value=0, value=2)
                recent_activity = st.number_input("Recent Activity (e.g., sum of last 3 days)", min_value=0, value=10)
            with col2:
                avg_temp = st.slider("Average Temperature (°C)", 10.0, 40.0, 25.0)
                avg_humidity = st.slider("Average Humidity (%)", 20.0, 100.0, 60.0)
                temp_range = st.slider("Temperature Range (Max-Min)", 0.0, 25.0, 10.0)
                days_since_cleaning = st.number_input("Days Since Last Trap Cleaning", min_value=0, value=7)
            with col3:
                temp_avg_3d = st.slider("3-Day Average Temperature", 10.0, 40.0, 24.0)
                humidity_avg_3d = st.slider("3-Day Average Humidity", 20.0, 100.0, 65.0)
                forecast_date = st.date_input("Forecast Date", value=pd.to_datetime("today"), key="reg_date")
            submitted_reg = st.form_submit_button("Predict Insect Count")
            if submitted_reg:
                reg_feature_cols = ['Location_Code', 'Average Temperature', 'Average Humidity', 'Temp_Range', 'Temp_Avg_3d', 'Humidity_Avg_3d', 'Insects_Lag1', 'Insects_Lag3', 'Recent_Activity', 'Days_Since_Cleaning', 'Month', 'Day']
                reg_input_data = {'Location_Code': location_mapping[location],'Average Temperature': avg_temp, 'Average Humidity': avg_humidity,'Temp_Range': temp_range, 'Temp_Avg_3d': temp_avg_3d, 'Humidity_Avg_3d': humidity_avg_3d,'Insects_Lag1': insects_lag1, 'Insects_Lag3': insects_lag3,'Recent_Activity': recent_activity, 'Days_Since_Cleaning': days_since_cleaning,'Month': forecast_date.month, 'Day': forecast_date.day}
                input_df_reg = pd.DataFrame([reg_input_data])[reg_feature_cols]
                input_scaled_reg = scaler.transform(input_df_reg)
                champion_reg_model = models['Random Forest']
                prediction_reg = champion_reg_model.predict(input_scaled_reg)
                predicted_count = utils_regression.ensure_non_negative_int(prediction_reg)[0]
                st.metric(label="Predicted Insect Count", value=predicted_count)

        st.markdown("---")
        st.header("🎯 Predict Pest Activity (Classification)")
        st.markdown("Fill in the details below to predict whether there will be any new pest captures (Yes/No).")
        class_feature_cols = class_artifacts['part1_results']['dataset_info']['features_used']
        with st.form("classification_form"):
            col1_clf, col2_clf, col3_clf = st.columns(3)
            with col1_clf:
                clf_location = st.selectbox("Select Location", options=list(location_mapping.keys()), key="clf_loc")
                clf_season = st.selectbox("Select Season", options=["Early_Summer", "Mid_Summer", "Late_Summer"], key="clf_season")
                clf_avg_temp = st.slider("Average Temperature (°C)", 10.0, 40.0, 28.0, key="clf_temp")
            with col2_clf:
                clf_avg_humidity = st.slider("Average Humidity (%)", 20.0, 100.0, 75.0, key="clf_hum")
                clf_days_since_cleaning = st.number_input("Days Since Cleaning", min_value=0, value=15, key="clf_clean")
                clf_insects_lag1 = st.number_input("Insects Caught Yesterday", min_value=0, value=0, key="clf_lag1")
            with col3_clf:
                clf_forecast_date = st.date_input("Forecast Date", value=pd.to_datetime("today"), key="clf_date")
                clf_temp_range = st.slider("Temperature Range", 0.0, 25.0, 12.0, key="clf_tr")
            st.info(f"This model requires {len(class_feature_cols)} features. This form will construct them for you.")
            submitted_clf = st.form_submit_button("Predict Pest Activity")
            if submitted_clf:
                input_series = pd.Series(0, index=class_feature_cols)

                input_series['Average Temperature'] = clf_avg_temp
                input_series['Average Humidity'] = clf_avg_humidity
                input_series['Days_Since_Cleaning'] = clf_days_since_cleaning
                input_series['Insects_Lag1'] = clf_insects_lag1
                input_series['Temp_Range'] = clf_temp_range
                input_series['Month'] = clf_forecast_date.month
                input_series['Day'] = clf_forecast_date.day

                if 'Insects_Lag3' in input_series.index:
                    input_series['Insects_Lag3'] = 0 
                if 'Recent_Activity' in input_series.index:
                    input_series['Recent_Activity'] = 0
                if 'Temp_Avg_3d' in input_series.index:
                    input_series['Temp_Avg_3d'] = clf_avg_temp
                if 'Humidity_Avg_3d' in input_series.index:
                    input_series['Humidity_Avg_3d'] = clf_avg_humidity
                
                location_col_name = f"Location_{clf_location}"
                if location_col_name in input_series.index:
                    input_series[location_col_name] = 1
                    
                season_col_name = f"Season_{clf_season}"
                if season_col_name in input_series.index:
                    input_series[season_col_name] = 1

                weekday = clf_forecast_date.weekday()
                day_of_week_col_name = f"Day_of_week_{weekday}"
                if day_of_week_col_name in input_series.index:
                    input_series[day_of_week_col_name] = 1

                input_df_clf = pd.DataFrame([input_series]).reindex(columns=class_feature_cols)
                
                input_scaled_clf = class_scaler.transform(input_df_clf)
                champion_clf_model = class_models['RandomForest']
                prediction_clf = champion_clf_model.predict(input_scaled_clf)[0]
                prediction_proba = champion_clf_model.predict_proba(input_scaled_clf)[0]
                
                if prediction_clf == 1:
                    st.warning("🚨 Pest Activity Predicted!")
                else:
                    st.success("✅ No Pest Activity Predicted")
                
                st.metric(label="Confidence of Pest Activity", value=f"{prediction_proba[1]*100:.2f}%")
                st.progress(prediction_proba[1])
