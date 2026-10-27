import os
import sys
import pandas as pd
import numpy as np

# Asegurar importación de módulos internos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_merge_data
from src.models import RetailForecaster, evaluate_backtesting


def run_training_pipeline():
    print("=" * 80)
    print("      WALMART RETAIL FORECASTING - PIPELINE DE ENTRENAMIENTO Y DIAGNÓSTICO     ")
    print("=" * 80)

    # 1. Cargar e integrar datos
    data_dir = "data/"
    if not os.path.exists(f"{data_dir}transactions.csv"):
        print(f"❌ Error: No se encontraron los archivos CSV en la ruta '{data_dir}'.")
        print("Asegúrate de colocar transactions.csv, stores.csv y calendar.csv en la carpeta data/.")
        return

    df = load_and_merge_data(data_dir=data_dir)
    print(f"✅ Integración exitosa. Total de registros cargados: {len(df):,}")

    # 2. Seleccionar combinaciones representativas para entrenamiento/prueba
    sample_stores = df['store_id'].unique()[:2]
    sample_categories = ['Abarrotes', 'Bebidas']
    target_variable = 'amount_total'

    steps_forecast = 30

    print("\n" + "-" * 80)
    print(f"EJECUTANDO PRUEBAS ESTADÍSTICAS Y EVALUACIÓN DE MODELOS (Pronóstico a {steps_forecast} días)")
    print("-" * 80)

    for store in sample_stores:
        for cat in sample_categories:
            print(f"\n📌 Tienda: {store} | Categoría: {cat} | Variable Objetivo: {target_variable}")
            
            # Instanciar el forecaster
            forecaster = RetailForecaster(df, store_id=store, category=cat, target_col=target_variable)

            # --- PRUEBAS ESTADÍSTICAS DE SERIES DE TIEMPO ---
            diagnostics = forecaster.get_statistical_diagnostics()
            print("\n  🔍 [Resultados de Pruebas Estadísticas]:")
            
            if "error" in diagnostics:
                print(f"     ⚠️ {diagnostics['error']}")
            else:
                # 1. ADF Test
                adf = diagnostics.get('ADF', {})
                print(f"     • Test ADF (Estacionariedad): Estadístico={adf.get('statistic')}, p-value={adf.get('p_value')} -> Estacionaria (95%): {adf.get('is_stationary_95%')}")
                
                # 2. KPSS Test
                kpss_res = diagnostics.get('KPSS', {})
                print(f"     • Test KPSS (Estacionariedad): Estadístico={kpss_res.get('statistic')}, p-value={kpss_res.get('p_value')} -> Estacionaria (95%): {kpss_res.get('is_stationary_95%')}")
                
                # 3. ARCH-LM Test
                arch = diagnostics.get('ARCH_LM', {})
                print(f"     • Test ARCH-LM (Heteroscedasticidad): Estadístico={arch.get('lagrange_multiplier_stat')}, p-value={arch.get('p_value')} -> Heteroscedasticidad Detectada: {arch.get('heteroscedasticity_detected')}")

            # --- PRONÓSTICOS DE MODELOS ---
            print("\n  🚀 [Entrenando y Generando Pronósticos]:")
            
            # ARIMA
            arima_pred = forecaster.fit_predict_arima(steps=steps_forecast)
            print(f"     ✔ ARIMA completado (Primeros 3 valores: {np.round(arima_pred[:3], 2)})")

            # Prophet
            prophet_pred = forecaster.fit_predict_prophet(steps=steps_forecast)
            print(f"     ✔ Prophet completado (Primeros 3 valores: {np.round(prophet_pred[:3], 2)})")

            # LSTM
            lstm_pred = forecaster.fit_predict_lstm(steps=steps_forecast)
            print(f"     ✔ LSTM completado (Primeros 3 valores: {np.round(lstm_pred[:3], 2)})")

            # --- METRICAS DE BACKTESTING ---
            # Se utiliza la última ventana real disponible para contrastar desempeño relativo
            real_series = forecaster.series.values
            if len(real_series) >= steps_forecast:
                y_true = real_series[-steps_forecast:]
                
                print("\n  📊 [Métricas de Backtesting en Último Periodo Conocido]:")
                metrics_arima = evaluate_backtesting(y_true, arima_pred)
                metrics_prophet = evaluate_backtesting(y_true, prophet_pred)
                metrics_lstm = evaluate_backtesting(y_true, lstm_pred)

                print(f"     • ARIMA   -> MAE: {metrics_arima['MAE']:,.2f} | RMSE: {metrics_arima['RMSE']:,.2f} | MAPE: {metrics_arima['MAPE (%)']}%")
                print(f"     • Prophet -> MAE: {metrics_prophet['MAE']:,.2f} | RMSE: {metrics_prophet['RMSE']:,.2f} | MAPE: {metrics_prophet['MAPE (%)']}%")
                print(f"     • LSTM    -> MAE: {metrics_lstm['MAE']:,.2f} | RMSE: {metrics_lstm['RMSE']:,.2f} | MAPE: {metrics_lstm['MAPE (%)']}%")

            print("-" * 80)

    print("\n✅ Pipeline de entrenamiento y pruebas finalizado correctamente.")


if __name__ == "__main__":
    run_training_pipeline()