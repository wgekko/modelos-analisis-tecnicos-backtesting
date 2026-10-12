Stock Lab & Advanced Multivariate Backtesting Engine

Aplicación cuantitativa desarrollada en Python con Streamlit, diseñada para el análisis técnico avanzado, modelado de regresión lineal y penalizada (OLS, Ridge, Lasso, ElasticNet), validación cruzada temporal y simulación de backtesting bursátil con gestión de riesgo basada en ATR (Average True Range).

Características Principales

Modelos de Regresión Avanzados: Soporte para modelos OLS (Statsmodels) y regresiones penalizadas con validación cruzada temporal (TimeSeriesSplit y ventanas móviles).

Indicadores Técnicos Dinámicos: Cálculo automatizado de medias móviles (SMA/EMA), RSI de Wilder, MACD, Bandas de Bollinger, Oscilador Estocástico y ATR.

Optimización de Parámetros: Búsqueda automática de la mejor combinación de periodos para indicadores técnicos mediante optimización orientada a minimizar el MAE.

Motor de Backtesting con Gestión de Riesgo: Simulación realista de operativa bursátil incorporando comisiones/deslizamiento, Stop Loss y Take Profit dinámicos basados en múltiplos de ATR.

Visualizaciones Interactivas: Gráficos profesionales en modo oscuro (plotly_dark) para curvas de capital, errores diarios, precios e indicadores técnicos.

Estructura del Proyecto

El repositorio incluye versiones evolutivas del sistema analítico:

main.py: Versión principal optimizada con análisis OLS y evaluación rigurosa fuera de muestra.

main_version_1.py: Versión con optimización por Grid Search y análisis comparativo por indicador.

main_version_2.py: Versión avanzada con modelos de regresión penalizada (Ridge, Lasso, ElasticNet) y backtesting completo con Stop Loss / Take Profit.

Requisitos e Instalación

Asegúrate de tener Python 3.9 o superior instalado en tu entorno. Sigue estos pasos para clonar y ejecutar la aplicación localmente:

Crea y activa un entorno virtual (recomendado):

python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS / Linux:
source venv/bin/activate


Instala las dependencias necesarias:

pip install streamlit yfinance pandas numpy scikit-learn statsmodels plotly


🖥️ Uso de la Aplicación

Para poner en marcha la interfaz de usuario en tu navegador, ejecuta el siguiente comando en la terminal:

streamlit run main.py


(Opcionalmente puedes ejecutar streamlit run main_version_2.py para la variante multivariada con penalización).

Introduce el Ticker bursátil (ej. NVDA, AAPL, MSFT).

Selecciona el rango de fechas de análisis.

Configura los parámetros de la ventana de entrenamiento y gestión de riesgo en la barra lateral.

Pulsa «Ejecutar análisis» y espera a que el motor procese los datos históricos y las predicciones walk-forward.

🎨 Configuración Visual (Opcional)

Para mantener una estética limpia tipo terminal financiera en Streamlit, crea una carpeta .streamlit en la raíz del proyecto y un archivo config.toml:

[theme]
primaryColor = "#00FF7F"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#161b22"
textColor = "#e6edf3"
font = "sans serif"


⚠️ Disclaimer

Herramienta educativa de análisis cuantitativo y simulación algorítmica. Las predicciones históricas basadas en series temporales no garantizan resultados financieros futuros ni constituyen asesoramiento de inversión.

---
## Requisitos del Sistema y Dependencias

Asegúrate de tener instalado Python 3.8 o superior. Las librerías necesarias para ejecutar la aplicación son:

* `streamlit`
* `yfinance`
* `pandas`
* `numpy`
* `plotly`
* `scikit-learn`

---

## Instalación y Configuración Local

Sigue estos pasos para clonar y ejecutar el proyecto en tu entorno local:

1. **Clona el repositorio:**
   bash
   git clone ([h(https://github.com/wgekko/modelos-analisis-tecnicos-backtesting.git))](https://github.com/wgekko/modelos-analisis-tecnicos-backtesting.git)
   cd modelos-analisis-tecnicos-backtesting



<img width="1905" height="938" alt="main" src="https://github.com/user-attachments/assets/fbd77712-2c49-4185-9977-848e6b5bd97e" />


<img width="1898" height="939" alt="main1" src="https://github.com/user-attachments/assets/8c4e77da-f905-48ae-972f-939352f20037" />


<img width="1885" height="935" alt="main2" src="https://github.com/user-attachments/assets/1a685d1f-267a-4217-80ed-971bd974e43e" />


