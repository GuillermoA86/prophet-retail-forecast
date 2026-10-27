from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    # Asegurar que la ruta termine con barra diagonal para evitar errores de concatenación
    df_global = load_and_merge_data(DATA_PATH + "/")
except Exception as e:
    print(f"Aviso en carga estándar, intentando ruta alternativa: {e}")
    # Fallback directo al archivo consolidado si existe
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
        
        if len(ts_df) < 2:
            raise ValueError("No hay suficientes datos históricos para el filtro seleccionado.")
            
        # Entrenar Prophet
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(ts_df)
        
        future = model.make_future_dataframe(periods=payload.steps)
        forecast = model.predict(future)
        
        future_df = forecast.tail(payload.steps)
        future_preds = future_df['yhat'].tolist()
        future_lower = future_df['yhat_lower'].tolist()
        future_upper = future_df['yhat_upper'].tolist()
        
        # Calcular métricas de ajuste histórico
        in_sample = model.predict(ts_df)
        y_true = ts_df['y'].values
        y_pred = in_sample['yhat'].values
        
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))))
        
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
                "mape": mape
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "predictions": [],
            "predictions_lower": [],
            "predictions_upper": [],
            "metrics_summary": {"forecast_sum": 0, "forecast_mean": 0},
            "metrics": {"mae": 0, "rmse": 0, "mape": 0}
        }