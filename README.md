# bike-rental-timeSeriesForcasting

I built this project to predict hourly bike rental demand by blending traditional stats with heavy-hitting ML. I used SARIMAX to handle the seasonal trends and weather variables, then layered in XGBoost, CatBoost, and LightGBM to catch those tricky, non-linear demand spikes. I focused on optimizing RMSLE so the model stays accurate even during chaotic peak hours and holidays.
# bike-rental-timeSeriesForcasting

Machine learning project forecasting bike rental demand using SARIMAX, LightGBM, and CatBoost.

### Tech Stack
* Languages: Python
* ML: LightGBM, CatBoost, Scikit-Learn
* Stats: SARIMAX, SHAP (for model interpretability)
* Data: Pandas, NumPy, Matplotlib

### Project Flow
* EDA: Analyzed seasonal trends, hourly peaks, and weather correlations.
* Preprocessing: Addressed stationarity with ADF tests and engineered lag features.
* Modeling: Evaluated gradient boosting models against classical time-series approaches.
* Insights: Applied SHAP values to quantify the impact of temperature and calendar features on demand.

### Performance
The models were compared based on error metrics, with CatBoost showing the highest accuracy for this specific dataset:
* SARIMAX: 1.0532
* LightGBM: 0.4364
* CatBoost: 0.4297 (Best Performance)

### Usage
Run the Jupyter notebooks in the repository to view the full analysis, model comparisons, and SHAP interpretability plots.
