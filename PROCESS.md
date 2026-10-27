## 9. Documentación del Proceso (`PROCESS.md`)

```markdown
# Documentación del Proceso de Trabajo (`PROCESS.md`)

## 1. Enfoque de Negocio y Problema
El objetivo principal es dotar a la cadena de retail de una herramienta predictiva a 30 días para anticipar la demanda de inventarios, optimizar la reposición (*replenishment_signal*) y gestionar la capacidad operativa por tienda y categoría.

## 2. Metodología y Modelado
- **Integración de Datos:** Se unificaron las transacciones diarias con las características estáticas de las tiendas y el calendario fiscal/festivo de México.
- **Modelos Evaluados:**
  - **ARIMA:** Captura componentes autorregresivos y de media móvil lineales.
  - **Prophet:** Ideal para modelar estacionalidades complejas y efectos de días festivos específicos en México (*El Buen Fin*, quincenas, temporada navideña).
  - **LSTM:** Red neuronal recurrente para capturar dependencias no lineales a largo plazo en las series de transacciones.
- **Validación:** Se implementó backtesting comparando MAE, RMSE y MAPE en particiones temporales.