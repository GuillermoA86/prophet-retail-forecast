import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def evaluate_backtesting(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}

class RetailForecaster:
    def __init__(self, train_df, store_id="Todas", category="Todas", target_col='amount_total'):
        self.store_id = store_id
        self.category = category
        self.target_col = target_col
        self.full_df = train_df
        
        sub = train_df.copy()
        if store_id != "Todas":
            sub = sub[sub['store_id'] == store_id]
        if category != "Todas":
            sub = sub[sub['category'] == category]
            
        sub = sub.sort_values('date')
        self.series = sub.set_index('date')[target_col].resample('D').sum().ffill()
        self.calendar_master = train_df[['date', 'is_holiday', 'is_payday', 'is_weekend', 'is_navidad_season', 'is_buen_fin', 'is_semana_santa']].drop_duplicates().set_index('date')

    def fit_predict_prophet(self, steps=30):
        if self.series.empty or len(self.series) == 0:
            zeros = np.zeros(steps)
            return zeros, zeros, zeros
            
        df_prophet = pd.DataFrame({
            'ds': self.series.index,
            'y': self.series.values
        }).merge(self.calendar_master, left_on='ds', right_index=True, how='left').fillna(0)

        model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        regressors = ['is_holiday', 'is_payday', 'is_weekend', 'is_navidad_season', 'is_buen_fin', 'is_semana_santa']
        for reg in regressors:
            if reg in df_prophet.columns:
                model.add_regressor(reg)
        
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=steps)
        future = future.merge(self.calendar_master, left_on='ds', right_index=True, how='left').fillna(0)
        
        forecast = model.predict(future)
        yhat = forecast['yhat'].tail(steps).values
        yhat_lower = forecast['yhat_lower'].tail(steps).values
        yhat_upper = forecast['yhat_upper'].tail(steps).values
        return yhat, yhat_lower, yhat_upper