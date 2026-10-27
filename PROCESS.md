# 🛒 Retail Intelligence & Demand Forecasting: Process & Methodology

## 1. Arquitectura y Visión General del Proyecto
Este proyecto implementa una solución de ingeniería de datos y machine learning de grado de producción para el análisis operativo y el pronóstico de demanda en tiendas de retail de Walmart. La arquitectura es de tipo orientada a servicios (SOA), desacoplando el motor analítico de series temporales y la API de inferencia (**FastAPI + Prophet**) de la interfaz de visualización ejecutiva (**Streamlit**), con soporte para despliegue contenerizado mediante **Docker Compose**.

---

## 2. Análisis Exploratorio y Limpieza Metodológica (EDA & Data Quality)
Durante la fase de preprocesamiento (`src/preprocessing.py`), se aplicaron los siguientes controles de calidad y análisis exploratorio:
* **Evaluación de Ceros y Nulos:** Se revisó exhaustivamente la presencia de ceros en las series temporales para evitar distorsiones o sesgos artificiales. No se encontraron nulos estructurales; los ceros identificados corresponden legítimamente a días sin transacciones registradas.
* **Corrección de Anomalías:** Se depuraron y corrigieron valores ilógicos o inconsistentes detectados específicamente en la tabla relacional `transactions`.
* **Detección de Valores Atípicos (Outliers):** No se detectaron anomalías extremas que ameritaran una imputación destructiva. Las caídas abruptas observadas en determinadas series temporales fueron contrastadas con el calendario operativo, confirmando que responden a patrones normales de reducción de demanda posteriores a días festivos (*post-holiday dip*).
* **Heterogeneidad de Series:** Dado que las tiendas y categorías presentan alta heterogeneidad, se optó por un modelado flexible capaz de capturar dinámicas locales sin sobreajuste.

---

## 3. Selección y Justificación del Modelo de Pronóstico
Se evaluaron y contrastaron tres enfoques metodológicos antes de seleccionar el modelo definitivo:
1. **ARIMA / SARIMA:** Descartado debido a su alta sensibilidad a múltiples estacionalidades superpuestas de largo plazo y su menor flexibilidad para incorporar regresores externos (como días festivos dinámicos) de forma automatizada y robusta.
2. **LSTM (Redes Neuronales Recurrentes):** Evaluado teóricamente; se determinó que para el volumen de datos históricos disponible por tienda/categoría, un modelo de deep learning como LSTM incurre en alto riesgo de *overfitting*, requiere un coste computacional elevado y no aporta mejoras marginales sustanciales frente a un modelo aditivo bien calibrado.
3. **Prophet (Seleccionado):** Elegido como el núcleo predictivo definitivo. Su estructura aditiva descompone la serie en tendencia, estacionalidad semanal, estacionalidad anual y efectos de días festivos. Maneja de manera nativa valores faltantes, es altamente interpretable desde el punto de vista económico y ofrece estabilidad computacional en producción.

---

## 4. Validación Rigurosa: Backtest & Cross-Validation sin Data Leakage
Para garantizar que el desempeño reportado en la API y el dashboard sea real y no producto de memorización (*in-sample overfitting*), la API implementa un esquema de validación **Out-of-Sample (Backtest)**:
* Se reserva un conjunto de prueba (*holdout*) correspondiente a los últimos periodos de la serie (horizonte de evaluación).
* El modelo de Prophet se entrena exclusivamente con los datos históricos anteriores a dicho horizonte ($t < T - h$).
* Las métricas de evaluación (**MAE, RMSE y MAPE**) se calculan comparando estrictamente las predicciones contra los valores reales ocultos del conjunto de prueba, garantizando **cero *data leakage***.
* Adicionalmente, se calcula un modelo base (*Naive Baseline*) para contrastar la mejora porcentual del modelo Prophet.

---

## 5. Pipeline del Sistema (End-to-End)
1. **Capa de Datos y Preprocesamiento (`src/preprocessing.py`):** Consolidación y limpieza metodológica de bases de datos transaccionales y de catálogos.
2. **Backend de Inferencia (`api/main.py`):** API construida con FastAPI que gestiona los filtros, ejecuta el entrenamiento dinámico de Prophet bajo demanda y calcula el backtest out-of-sample.
3. **Dashboard Ejecutivo (`dashboard/app.py`):** Interfaz en Streamlit con gráficos interactivos en Plotly, indicadores comparativos frente a promedios regionales/globales y visualización de intervalos de confianza al 95%.
4. **Contenerización (`Docker` & `docker-compose.yml`):** Orquestación mediante contenedores Docker para garantizar portabilidad y despliegue estandarizado.