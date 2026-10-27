import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# 🛠️ Agregar el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Walmart Retail Intelligence Dashboard", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    </style>
    """, unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

@st.cache_data
def load_base_data():
    from src.preprocessing import load_and_merge_data
    return load_and_merge_data("data/")

df = load_base_data()

st.title("🛒 Walmart Retail Intelligence & Demand Forecasting")
st.markdown("Tablero ejecutivo con análisis operativo, rentabilidad por metro cuadrado, segmentación y proyecciones con **Prophet**.")

# ----------------- SIDEBAR: FILTROS -----------------
st.sidebar.header("🔍 Filtros de Segmentación")

region_list = ["Todas"] + list(df['region'].unique())
selected_region = st.sidebar.selectbox("Región", region_list)

format_list = ["Todas"] + list(df['store_format'].unique())
selected_format = st.sidebar.selectbox("Formato de Tienda", format_list)

df_filtered = df.copy()
if selected_region != "Todas":
    df_filtered = df_filtered[df_filtered['region'] == selected_region]
if selected_format != "Todas":
    df_filtered = df_filtered[df_filtered['store_format'] == selected_format]

store_options = ["Todas las tiendas"] + list(df_filtered['store_id'].unique())
selected_store = st.sidebar.selectbox("Seleccionar Tienda", store_options)

categories = ["Todas"] + list(df['category'].unique())
selected_category = st.selectbox("Categoría de Producto", categories)

metric_labels = {
    "amount_total": "Ventas Totales",
    "units_sold": "Unidades Vendidas",
    "total_transactions": "Transacciones Totales"
}
selected_metric_key = st.selectbox(
    "Métrica Analizada / Pronosticada", 
    options=list(metric_labels.keys()), 
    format_func=lambda x: metric_labels[x]
)

unit_prefix = "MXN $" if selected_metric_key == "amount_total" else ""

# ----------------- VISTA 1: TODAS LAS TIENDAS (GLOBAL) -----------------
if selected_store == "Todas las tiendas":
    st.subheader("📊 Indicadores Globales de Operación (Cadena)")
    
    total_sales = df_filtered['amount_total'].sum()
    total_trans = df_filtered['total_transactions'].sum()
    avg_ticket_global = df_filtered['avg_ticket'].mean()
    total_cash = df_filtered['amount_cash'].sum()
    total_card = df_filtered['amount_card'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventas Totales", f"MXN ${total_sales:,.0f}")
    col2.metric("Transacciones", f"{total_trans:,.0f}")
    col3.metric("Ticket Promedio", f"MXN ${avg_ticket_global:,.2f}")
    col4.metric("Efectivo vs Tarjeta", f"Efectivo: {total_cash/(total_cash+total_card)*100:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🥧 Distribución de Monto: Efectivo vs Tarjeta")
        pay_amt = pd.DataFrame({"Método de Pago": ["Efectivo", "Tarjeta"], "Monto": [total_cash, total_card]})
        fig_pay_amt = px.pie(pay_amt, names='Método de Pago', values='Monto', hole=0.4)
        fig_pay_amt.update_layout(template="plotly_white")
        st.plotly_chart(fig_pay_amt, use_container_width=True)
        
    with c2:
        st.subheader("🥧 Distribución de Transacciones: Efectivo vs Tarjeta")
        pay_t = pd.DataFrame({"Método de Pago": ["Efectivo", "Tarjeta"], "Transacciones": [df_filtered['cash_transactions'].sum(), df_filtered['card_transactions'].sum()]})
        fig_pay_t = px.pie(pay_t, names='Método de Pago', values='Transacciones', hole=0.4)
        fig_pay_t.update_layout(template="plotly_white")
        st.plotly_chart(fig_pay_t, use_container_width=True)

    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("📊 Ventas por Región")
        reg_sales = df_filtered.groupby('region')['amount_total'].sum().reset_index()
        fig_reg = px.bar(reg_sales, x='region', y='amount_total', color='region', labels={'region': 'Región', 'amount_total': 'Ventas Totales (MXN)'})
        fig_reg.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_reg, use_container_width=True)
        
    with r2:
        st.subheader("📊 Ventas por Formato de Tienda")
        fmt_sales = df_filtered.groupby('store_format')['amount_total'].sum().reset_index()
        fig_fmt = px.bar(fmt_sales, x='store_format', y='amount_total', color='store_format', labels={'store_format': 'Formato de Tienda', 'amount_total': 'Ventas Totales (MXN)'})
        fig_fmt.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_fmt, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Ventas por Nivel Socioeconómico")
    nse_sales = df_filtered.groupby('socioeconomic_level')['amount_total'].sum().reset_index()
    fig_nse = px.bar(nse_sales, x='socioeconomic_level', y='amount_total', color='socioeconomic_level', labels={'socioeconomic_level': 'Nivel Socioeconómico', 'amount_total': 'Ventas Totales (MXN)'})
    fig_nse.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_nse, use_container_width=True)

# ----------------- VISTA 2: TIENDA ESPECÍFICA -----------------
else:
    store_info = df[df['store_id'] == selected_store].iloc[0]
    
    st.subheader(f"🏪 Detalle Operativo y Características de la Tienda: {selected_store}")
    
    sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8 = st.columns(8)
    sc1.metric("Formato", store_info['store_format'])
    sc2.metric("Región", store_info['region'])
    sc3.metric("Tamaño", f"{store_info['size_sqm']:,} m²")
    sc4.metric("Cajas", store_info['num_checkouts'])
    sc5.metric("Apertura", store_info['opening_year'])
    sc6.metric("NSE", store_info['socioeconomic_level'])
    sc7.metric("Farmacia", "Sí" if store_info['has_pharmacy'] else "No")
    sc8.metric("Gasolinera", "Sí" if store_info['has_fuel_station'] else "No")

    store_df = df_filtered[df_filtered['store_id'] == selected_store]
    if selected_category != "Todas":
        store_df = store_df[store_df['category'] == selected_category]

    s_val = store_df[selected_metric_key].sum()
    s_trans = store_df['total_transactions'].sum()
    s_ticket = store_df['avg_ticket'].mean()

    sqm = store_info['size_sqm']
    sales_per_sqm = s_val / sqm if sqm > 0 else 0

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Total {metric_labels[selected_metric_key]}", f"{unit_prefix}{s_val:,.0f}")
    col2.metric("Transacciones", f"{s_trans:,.0f}")
    col3.metric("Ticket Promedio", f"MXN ${s_ticket:,.2f}")
    col4.metric("Rendimiento por Metro Cuadrado", f"MXN ${sales_per_sqm:,.2f} / m²")

    st.markdown("---")
    st.subheader(f"📊 Rendimiento por Categoría en la Tienda ({metric_labels[selected_metric_key]})")
    store_cat = store_df.groupby('category')[selected_metric_key].sum().reset_index()
    fig_store_cat = px.bar(
        store_cat, 
        x='category', 
        y=selected_metric_key, 
        color='category', 
        text_auto='.2s',
        labels={'category': 'Categoría de Producto', selected_metric_key: metric_labels[selected_metric_key]}
    )
    fig_store_cat.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_store_cat, use_container_width=True)

# ----------------- PRONÓSTICO CON PROPHET (GLOBAL O INDIVIDUAL) -----------------
st.markdown("---")
st.subheader("🔮 Pronóstico de Demanda a 30 Días (Modelo Prophet con Ciclicidad)")

api_store_param = "Todas" if selected_store == "Todas las tiendas" else selected_store
api_cat_param = selected_category

if st.button("🚀 Generar Pronóstico con Prophet"):
    with st.spinner("Calculando estacionalidad, festivos y eventos con Prophet..."):
        try:
            response = requests.post(f"{API_URL}/predict_json", json={
                "store_id": api_store_param,
                "category": api_cat_param,
                "target": selected_metric_key,
                "steps": 30
            })
            
            data = response.json()
            
            if "error" in data and data["error"]:
                st.error(f"Error devuelto por la API: {data['error']}")
                st.stop()

            date_col = 'date' if 'date' in df.columns else 'ds'
            temp_df = df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            
            if api_store_param != "Todas":
                temp_df = temp_df[temp_df['store_id'] == api_store_param]
            if api_cat_param != "Todas":
                temp_df = temp_df[temp_df['category'] == api_cat_param]
                
            hist_series = temp_df.groupby(date_col)[selected_metric_key].sum().sort_index().tail(60)
            
            last_date = hist_series.index.max()
            future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 31)]
            prophet_preds = data.get('predictions', [])
            prophet_lower = data.get('predictions_lower', [])
            prophet_upper = data.get('predictions_upper', [])

            fig_ts = go.Figure()
            
            fig_ts.add_trace(go.Scatter(
                x=hist_series.index.strftime('%Y-%m-%d'), y=hist_series.values,
                mode='lines+markers', name='Histórico Real',
                line=dict(color='#0071CE', width=3)
            ))
            
            fig_ts.add_trace(go.Scatter(
                x=future_dates, y=prophet_upper,
                mode='lines', marker=dict(color="#444"),
                line=dict(width=0), showlegend=False
            ))
            
            fig_ts.add_trace(go.Scatter(
                x=future_dates, y=prophet_lower,
                mode='lines', marker=dict(color="#444"),
                line=dict(width=0), fill='tonexty',
                fillcolor='rgba(46, 139, 87, 0.2)', name='Intervalo de Confianza (95%)', showlegend=True
            ))
            
            fig_ts.add_trace(go.Scatter(
                x=future_dates, y=prophet_preds,
                mode='lines+markers', name='Pronóstico Prophet (30 días)',
                line=dict(color='#2E8B57', width=2, dash='dash')
            ))

            fig_ts.update_layout(
                title=f"Proyección con Intervalo de Confianza - Alcance: [{selected_store}] | Categoría: [{selected_category}] | Métrica: [{metric_labels[selected_metric_key]}]",
                xaxis_title="Fecha de Venta", 
                yaxis_title=metric_labels[selected_metric_key],
                template="plotly_white", 
                hovermode="x unified"
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Resumen de Predicción
            st.subheader("📊 Resumen de Predicción y Métricas de Evaluación")
            
            metrics_sum = data.get('metrics_summary', {})
            f_sum = metrics_sum.get('forecast_sum', 0)
            f_mean = metrics_sum.get('forecast_mean', 0)

            m1, m2, m3 = st.columns(3)
            m1.metric("Suma Proyectada (30 días)", f"{unit_prefix}{f_sum:,.0f}")
            m2.metric("Promedio Diario Proyectado", f"{unit_prefix}{f_mean:,.2f}")
            m3.metric("Modelo Activo", "Prophet (con Ciclicidad)")

            st.markdown("---")
            st.subheader("📐 Desempeño y Precisión del Modelo")

            # Renderizado único de las métricas de evaluación
            model_eval = data.get('metrics', {})
            if model_eval:
                mae = model_eval.get('mae', 0)
                rmse = model_eval.get('rmse', 0)
                mape = model_eval.get('mape', 0)

                mm1, mm2, mm3 = st.columns(3)
                mm1.metric("Error Absoluto Medio (MAE)", f"{mae:,.2f}")
                mm2.metric("Raíz del Error Cuadrático (RMSE)", f"{rmse:,.2f}")
                mm3.metric("Error Porcentual Absoluto (MAPE)", f"{mape:.2%}")
            else:
                st.info("ℹ️ No se encontraron métricas en la respuesta de la API.")

        except Exception as e:
            st.error(f"Error al conectar con la API o calcular Prophet: {e}. Asegúrate de que la API esté corriendo en el puerto 8000.")