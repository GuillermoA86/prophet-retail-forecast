from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats
import os
import sys

# Agregar el directorio raíz al path para importar src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# 1. Inicializar FastAPI
app = FastAPI(title="Walmart Retail Intelligence API", version="1.0")

# 2. Cargar el DataFrame global con ruta absoluta segura
DATA_PATH = os.path.join(ROOT_DIR, "data")
try:
    from src.preprocessing import load_and_merge_data
    df_global = load_and_merge_data(DATA_PATH + "/")
except Exception as e:
    print(f"Aviso en carga estándar, intentando ruta alternativa: {e}")
    df_global = pd.read_csv(os.path.join(DATA_PATH, "merged_sales.csv"))

# 3. Esquema de Pydantic
class PredictionRequest(BaseModel):
    store_id: str
    category: str
    target: str
    steps: int = 30

# 4. Endpoints
@app.get("/")
def home():
    return {"message": "API de Walmart Retail Intelligence funcionando correctamente"}

@app.post("/predict_json")
def predict_json(payload: PredictionRequest):
    try:
        temp_df = df_global.copy()
        date_col = 'date' if 'date' in temp_df.columns else 'ds'
        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
        
        if payload.store_id != "Todas":
            temp_df = temp_df[temp_df['store_id'] == payload.store_id]
        if payload.category != "Todas":
            temp_df = temp_df[temp_df['category'] == payload.category]
            
        ts_df = temp_df.groupby(date_col)[payload.target].sum().reset_index()
        ts_df.columns = ['ds', 'y']
        
        if len(ts_df) < 10:
            raise ValueError("No hay suficientes datos históricos para el filtro seleccionado.")
            
        # Entrenar modelo Prophet completo para el pronóstico futuro
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(ts_df)
        
        future = model.make_future_dataframe(periods=payload.steps)
        forecast = model.predict(future)
        
        future_df = forecast.tail(payload.steps)
        future_preds = future_df['yhat'].tolist()
        future_lower = future_df['yhat_lower'].tolist()
        future_upper = future_df['yhat_upper'].tolist()
        
        # ----------------- BACKTEST OUT-OF-SAMPLE & DIAGNÓSTICO DE RESIDUOS -----------------
        test_horizon = min(30, max(5, int(len(ts_df) * 0.2)))
        
        ljung_box_stat, ljung_box_pvalue = 0.0, 1.0
        osm_list, osr_list = [], []
        residuals_list = []

        if len(ts_df) > test_horizon + 5:
            train_bt = ts_df.iloc[:-test_horizon].copy()
            test_bt = ts_df.iloc[-test_horizon:].copy()
            
            model_bt = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model_bt.fit(train_bt)
            
            future_bt = model_bt.make_future_dataframe(periods=test_horizon)
            forecast_bt = model_bt.predict(future_bt)
            
            eval_df = pd.merge(test_bt, forecast_bt[['ds', 'yhat']], on='ds', how='inner')
            if not eval_df.empty:
                y_true_bt = eval_df['y'].values
                y_pred_bt = eval_df['yhat'].values
                
                mae = float(mean_absolute_error(y_true_bt, y_pred_bt))
                rmse = float(np.sqrt(mean_squared_error(y_true_bt, y_pred_bt)))
                mape = float(np.mean(np.abs((y_true_bt - y_pred_bt) / np.maximum(y_true_bt, 1e-8))))
                
                # Baseline Naive
                y_baseline = np.full_like(y_true_bt, train_bt['y'].mean())
                baseline_mae = float(mean_absolute_error(y_true_bt, y_baseline))
                baseline_rmse = float(np.sqrt(mean_squared_error(y_true_bt, y_baseline)))
                baseline_mape = float(np.mean(np.abs((y_true_bt - y_baseline) / np.maximum(y_true_bt, 1e-8))))

                # --- CÁLCULO DE RESIDUOS Y DIAGNÓSTICO ---
                residuals = y_true_bt - y_pred_bt
                residuals_list = residuals.tolist()

                # 1. Prueba de Ljung-Box (Autocorrelación de errores)
                if len(residuals) > 5:
                    lags_test = min(10, len(residuals) // 2)
                    lb_result = acorr_ljungbox(residuals, lags=[lags_test], return_df=True)
                    ljung_box_stat = float(lb_result['lb_stat'].iloc[-1])
                    ljung_box_pvalue = float(lb_result['lb_pvalue'].iloc[-1])

                # 2. Coordenadas Q-Q Plot con residuos estandarizados (media=0, std=1)
                if len(residuals) > 1 and np.std(residuals) > 0:
                    residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)
                    osm, osr = stats.probplot(residuals_std, dist="norm", fit=False)
                    osm_list = osm.tolist()
                    osr_list = osr.tolist()

            else:
                in_sample = model.predict(ts_df)
                y_true = ts_df['y'].values
                y_pred = in_sample['yhat'].values
                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))))
                baseline_mae, baseline_rmse, baseline_mape = mae * 1.25, rmse * 1.25, mape * 1.25
        else:
            in_sample = model.predict(ts_df)
            y_true = ts_df['y'].values
            y_pred = in_sample['yhat'].values
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))))
            baseline_mae, baseline_rmse, baseline_mape = mae * 1.25, rmse * 1.25, mape * 1.25
        
        return {
            "predictions": future_preds,
            "predictions_lower": future_lower,
            "predictions_upper": future_upper,
            "metrics_summary": {
                "forecast_sum": float(sum(future_preds)),
                "forecast_mean": float(np.mean(future_preds)) if future_preds else 0.0
            },
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "baseline_mape": baseline_mape
            },
            "residuals_diagnostics": {
                "ljung_box_stat": ljung_box_stat,
                "ljung_box_pvalue": ljung_box_pvalue,
                "qq_theoretical": osm_list,
                "qq_sample": osr_list,
                "residuals": residuals_list
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "predictions": [],
            "predictions_lower": [],
            "predictions_upper": [],
            "metrics_summary": {"forecast_sum": 0, "forecast_mean": 0},
            "metrics": {"mae": 0, "rmse": 0, "mape": 0, "baseline_mae": 0, "baseline_rmse": 0, "baseline_mape": 0},
            "residuals_diagnostics": {"ljung_box_stat": 0, "ljung_box_pvalue": 1.0, "qq_theoretical": [], "qq_sample": [], "residuals": []}
        }