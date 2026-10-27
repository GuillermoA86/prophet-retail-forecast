# 🛒 Retail Intelligence & Demand Forecasting

Solución analítica y predictiva de nivel corporativo diseñada para el análisis operativo y el pronóstico de demanda en tiendas de retail de Walmart. La arquitectura desacopla el motor analítico e inferencia (**FastAPI + Prophet**) de la interfaz ejecutiva (**Streamlit**), respaldado por un pipeline de validación sin *data leakage* (*out-of-sample backtest*) y orquestación con contenedores.

---

## 🚀 Descripción del Proyecto
El sistema permite segmentar operaciones por región, formato de tienda, nivel socioeconómico y categoría de producto, calculando indicadores clave como ventas totales, transacciones, ticket promedio, rentabilidad por metro cuadrado y proyecciones de demanda a 30 días con intervalos de confianza del 95% y validación mediante *backtest out-of-sample*.

---

## 📁 Estructura del Repositorio
```text
retail-forecast/
├── api/
│   ├── Dockerfile
│   └── main.py              # FastAPI backend, Prophet model & out-of-sample backtest
├── dashboard/
│   ├── Dockerfile
│   └── app.py               # Streamlit executive dashboard & comparative metrics
├── data/
│   ├── merged_sales.csv     # Master dataset de Walmart
│   └── ...                  
├── notebooks/               # Exploratory Data Analysis & experiments
├── src/
│   └── preprocessing.py     # Data cleaning, merging & quality control
├── PROCESS.md               # Metodología técnica detallada y justificación de modelos
├── docker-compose.yml       # Orquestación multi-contenedor (API + Dashboard)
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación principal del proyecto



🛠️ Requisitos Previos
Python 3.10 o superior instalado.

Git.

(Opcional) Docker Desktop con WSL 2 habilitado (si deseas probar la versión contenerizada).

💻 1. Pasos para Correr el Proyecto en Local (Entorno Virtual)
Para ejecutar la aplicación de forma nativa en tu máquina, sigue estos pasos:

Paso 1: Clonar el repositorio
Bash
git clone [https://github.com/GuillermoA86/prophet-retail-forecast.git](https://github.com/GuillermoA86/prophet-retail-forecast.git)
cd walmart-retail-forecast
Paso 2: Crear y activar el entorno virtual
Bash
python -m venv venv
# En Windows (PowerShell):
.\venv\Scripts\Activate
# En macOS / Linux:
source venv/bin/activate
Paso 3: Instalar las dependencias
Bash
pip install -r requirements.txt
Paso 4: Correr la API del Modelo (FastAPI)
Abre una primera terminal, asegúrate de tener el entorno virtual activado (venv) y ejecuta el servidor backend:

Bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Verificación: Puedes comprobar que la API está activa abriendo en tu navegador http://localhost:8000/docs para ver la documentación interactiva (Swagger UI).

Paso 5: Correr el Dashboard (Streamlit)
Abre una segunda terminal independiente, activa también el entorno virtual (venv) y ejecuta la interfaz gráfica:

Bash
streamlit run dashboard/app.py
Verificación: Se abrirá automáticamente una pestaña en tu navegador en http://localhost:8501 conectada a tu API local.

🐳 2. Pasos para Correr el Proyecto con Docker (Docker Compose)
Si prefieres ejecutar todo el sistema empaquetado en contenedores aislados:

Paso 1: Asegúrate de tener Docker abierto
Verifica que Docker Desktop esté encendido y funcionando en tu sistema.

Paso 2: Construir y levantar los servicios
Desde la carpeta raíz del proyecto, ejecuta el comando de orquestación:

Bash
docker compose up --build
(Este comando construirá las imágenes tanto para la API como para Streamlit y levantará ambos contenedores enlazados internamente).

Paso 3: Acceder a los servicios
Una vez que los contenedores estén corriendo, podrás acceder a ellos desde tu navegador:

Dashboard Ejecutivo (Streamlit): http://localhost:8501

Backend de Inferencia (FastAPI): http://localhost:8000/docs

Paso 4: Detener los contenedores
Para apagar y detener los contenedores de manera limpia, ejecuta en tu terminal:

Bash
docker compose down




