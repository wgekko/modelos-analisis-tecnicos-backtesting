import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from datetime import date, timedelta


# ==========================================================
# CONFIGURACION
# ==========================================================

st.set_page_config(
    page_title="Stock Lab | Análisis cuantitativo",
    page_icon=":material/finance_mode:",
    layout="wide"
)

st.subheader(":material/finance_mode: Stock Lab")
st.caption(
    "Análisis técnico, regresión OLS y evaluación "
    "de predicciones bursátiles."
)
st.info("al ejecutar el análisis y backtesting - los resultados pueden demorar varios minutos")

# ==========================================================
# PARAMETROS
# ==========================================================

st.sidebar.subheader(":material/settings: Parámetros")

ticker = st.sidebar.text_input(
    "Ticker",
    value="NVDA"
).strip().upper()

fecha_inicio = st.sidebar.date_input(
    "Fecha inicial",
    value=date(2022, 10, 25)
)

fecha_fin = st.sidebar.date_input(
    "Fecha final",
    value=date.today()
)

window_size = st.sidebar.slider(
    "Ventana de entrenamiento",
    min_value=20,
    max_value=252,
    value=60,
    step=10,
    help="Número de sesiones anteriores utilizadas en cada ajuste."
)

indicadores = [
    "SMA_50",
    "EMA_50",
    "RSI",
    "MACD",
    "Signal_Line",
    "BB_Upper",
    "BB_Middle",
    "BB_Lower",
    "%K",
    "%D",
    "ATR",
    "Close_shifted"
]

indicador_seleccionado = st.sidebar.selectbox(
    "Indicador para el modelo OLS",
    indicadores,
    index=1
)

ejecutar = st.sidebar.button(
    "Ejecutar análisis",
    type="primary",
    width='stretch'
)


# ==========================================================
# DESCARGA DE DATOS
# ==========================================================

@st.cache_data(ttl=900, show_spinner=False)
def descargar_datos(ticker, fecha_inicio, fecha_fin):
    datos = yf.download(
        ticker,
        start=fecha_inicio,
        end=fecha_fin + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if datos.empty:
        return pd.DataFrame()

    # Compatibilidad con columnas MultiIndex.
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)

    columnas = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in datos.columns for c in columnas):
        raise ValueError(
            "Yahoo Finance no devolvió todas las columnas necesarias."
        )

    datos = datos[columnas].copy()

    for columna in columnas:
        datos[columna] = pd.to_numeric(
            datos[columna], errors="coerce"
        )

    datos = datos.replace([np.inf, -np.inf], np.nan)
    datos = datos.dropna(subset=columnas)
    datos = datos[~datos.index.duplicated(keep="last")]
    datos = datos.sort_index()

    return datos


# ==========================================================
# INDICADORES TECNICOS
# ==========================================================

def calcular_indicadores(datos):
    df = datos.copy()

    # Características disponibles antes de la sesión objetivo.
    close = df["Close"].shift(1)
    high = df["High"].shift(1)
    low = df["Low"].shift(1)

    df["Previous_Close"] = close
    df["Close_shifted"] = close

    df["SMA_50"] = close.rolling(50).mean()

    df["EMA_50"] = close.ewm(
        span=50,
        min_periods=50,
        adjust=False
    ).mean()

    # RSI de 14 periodos mediante suavizado de Wilder.
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - 100 / (1 + rs)

    df.loc[
        (avg_loss == 0) & (avg_gain > 0), "RSI"
    ] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss > 0), "RSI"
    ] = 0

    df.loc[
        (avg_gain == 0) & (avg_loss == 0), "RSI"
    ] = 50

    # MACD.
    ema12 = close.ewm(
        span=12, min_periods=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, min_periods=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["Signal_Line"] = df["MACD"].ewm(
        span=9, min_periods=9, adjust=False
    ).mean()

    # Bandas de Bollinger.
    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)

    df["BB_Middle"] = bb_middle
    df["BB_Upper"] = bb_middle + 2 * bb_std
    df["BB_Lower"] = bb_middle - 2 * bb_std

    # Estocástico: %K suavizado y %D.
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    rango = highest - lowest

    raw_k = 100 * (close - lowest) / rango
    raw_k = raw_k.mask(rango == 0, 0)

    df["%K"] = raw_k.rolling(3).mean()
    df["%D"] = df["%K"].rolling(3).mean()

    # ATR de 14 periodos.
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


# ==========================================================
# VALIDACION WALK-FORWARD
# ==========================================================

def evaluar_modelo(df, indicador, ventana):
    """
    Para cada fecha de prueba:
    - Entrena solo con las observaciones anteriores.
    - Utiliza el indicador y el cierre anterior.
    - Compara con el precio de cierre real.
    """

    resultados = []

    columnas = [indicador, "Previous_Close"]

    for i in range(ventana, len(df)):
        train = df.iloc[i - ventana:i]
        test = df.iloc[[i]]

        X_train = train[columnas].astype(float)
        y_train = train["Close"].astype(float)

        X_test = test[columnas].astype(float)

        if (
            not np.isfinite(X_train.to_numpy()).all()
            or not np.isfinite(y_train.to_numpy()).all()
            or not np.isfinite(X_test.to_numpy()).all()
        ):
            continue

        # Evitar entrenar con una variable constante.
        if X_train[indicador].nunique() < 2:
            continue

        X_train = sm.add_constant(
            X_train, has_constant="add"
        )

        X_test = sm.add_constant(
            X_test, has_constant="add"
        )

        try:
            modelo = sm.OLS(y_train, X_train).fit()

            prediccion = float(
                modelo.predict(X_test).iloc[0]
            )

        except (ValueError, np.linalg.LinAlgError):
            continue

        real = float(test["Close"].iloc[0])
        anterior = float(test["Previous_Close"].iloc[0])

        if not np.isfinite(prediccion):
            continue

        resultados.append({
            "Fecha": test.index[0],
            "Real": real,
            "Predicción OLS": prediccion,
            "Predicción base": anterior,
            "Error absoluto OLS": abs(real - prediccion),
            "Error absoluto base": abs(real - anterior)
        })

    return pd.DataFrame(resultados)


# ==========================================================
# METRICAS
# ==========================================================

def calcular_metricas(real, predicho):
    real = np.asarray(real, dtype=float)
    predicho = np.asarray(predicho, dtype=float)

    mae = mean_absolute_error(real, predicho)
    mse = mean_squared_error(real, predicho)
    rmse = np.sqrt(mse)

    mascara = real != 0

    mape = (
        np.mean(
            np.abs(
                (real[mascara] - predicho[mascara])
                / real[mascara]
            )
        ) * 100
        if mascara.any()
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    }


# ==========================================================
# EJECUCION
# ==========================================================

if ejecutar:
    if not ticker:
        st.error("Introduce un ticker válido.")

    elif fecha_inicio >= fecha_fin:
        st.error("La fecha inicial debe preceder a la final.")

    else:
        try:
            with st.spinner("Analizando datos y entrenando modelos..."):
                datos = descargar_datos(
                    ticker, fecha_inicio, fecha_fin
                )

                if datos.empty:
                    st.error(
                        "No se encontraron datos para ese ticker "
                        "y período."
                    )
                    st.stop()

                df = calcular_indicadores(datos)

                if len(df) <= window_size:
                    st.error(
                        "No hay suficientes observaciones. "
                        "Amplía el período o reduce la ventana."
                    )
                    st.stop()

                predicciones = evaluar_modelo(
                    df,
                    indicador_seleccionado,
                    window_size
                )

                if predicciones.empty:
                    st.error(
                        "No fue posible generar predicciones."
                    )
                    st.stop()

                metricas_ols = calcular_metricas(
                    predicciones["Real"],
                    predicciones["Predicción OLS"]
                )

                metricas_base = calcular_metricas(
                    predicciones["Real"],
                    predicciones["Predicción base"]
                )

                comparacion = pd.DataFrame(
                    [
                        {"Modelo": "OLS", **metricas_ols},
                        {
                            "Modelo": "Cierre anterior",
                            **metricas_base
                        }
                    ]
                )

                st.session_state["df"] = df
                st.session_state["predicciones"] = predicciones
                st.session_state["comparacion"] = comparacion
                st.session_state["ticker"] = ticker
                st.session_state["indicador"] = indicador_seleccionado

        except Exception as e:
            st.error(f"Error durante el análisis: {e}")


# ==========================================================
# RESULTADOS
# ==========================================================

if "predicciones" in st.session_state:
    df = st.session_state["df"]
    pred = st.session_state["predicciones"]
    comparacion = st.session_state["comparacion"]

    ticker_actual = st.session_state["ticker"]
    indicador_actual = st.session_state["indicador"]

    st.subheader(
        f"Resultados de {ticker_actual}"
    )

    st.caption(
        f"Variable técnica: {indicador_actual}. "
        "Evaluación temporal fuera de muestra mediante ventana móvil."
    )

    # Métricas principales.
    ols = comparacion.iloc[0]
    base = comparacion.iloc[1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAE OLS",
        f"{ols['MAE']:.4f}",
        delta=f"{ols['MAE'] - base['MAE']:.4f} vs. base",
        delta_color="inverse"
    )

    c2.metric("RMSE OLS", f"{ols['RMSE']:.4f}")
    c3.metric("MAPE OLS", f"{ols['MAPE (%)']:.2f}%")
    c4.metric("Predicciones", f"{len(pred)}")

    st.subheader("1. Comparación de modelos")

    st.dataframe(
        comparacion.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "MAPE (%)": "{:.2f}%"
        }),
        width='stretch',
        hide_index=True
    )

    st.caption(
        "Un error menor es preferible. El modelo OLS solo aporta "
        "valor si mejora al modelo base en datos no usados "
        "para ajustar cada predicción."
    )

    # Predicciones contra valores reales.
    st.subheader("2. Precio real frente a predicción")

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=pred["Fecha"],
        y=pred["Real"],
        name="Precio real",
        mode="lines"
    ))

    fig_pred.add_trace(go.Scatter(
        x=pred["Fecha"],
        y=pred["Predicción OLS"],
        name="Predicción OLS",
        mode="lines"
    ))

    fig_pred.add_trace(go.Scatter(
        x=pred["Fecha"],
        y=pred["Predicción base"],
        name="Cierre anterior (base)",
        mode="lines",
        line=dict(dash="dot")
    ))

    fig_pred.update_layout(
        template="plotly_white",
        height=500,
        hovermode="x unified",
        xaxis_title="Fecha",
        yaxis_title="Precio ajustado"
    )

    st.plotly_chart(fig_pred, width='stretch')

    # Errores diarios.
    st.subheader("3. Evolución del error")

    fig_error = go.Figure()

    fig_error.add_trace(go.Scatter(
        x=pred["Fecha"],
        y=pred["Error absoluto OLS"],
        name="Error OLS",
        mode="lines"
    ))

    fig_error.add_trace(go.Scatter(
        x=pred["Fecha"],
        y=pred["Error absoluto base"],
        name="Error modelo base",
        mode="lines"
    ))

    fig_error.update_layout(
        template="plotly_white",
        height=400,
        hovermode="x unified",
        xaxis_title="Fecha",
        yaxis_title="Error absoluto"
    )

    st.plotly_chart(fig_error, width='stretch')

    # Precio e indicadores.
    st.subheader("4. Análisis técnico")

    fig_ind = go.Figure()

    fig_ind.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Cierre ajustado",
        mode="lines"
    ))

    fig_ind.add_trace(go.Scatter(
        x=df.index,
        y=df["SMA_50"],
        name="SMA 50",
        mode="lines"
    ))

    fig_ind.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA_50"],
        name="EMA 50",
        mode="lines"
    ))

    fig_ind.add_trace(go.Scatter(
        x=df.index,
        y=df["BB_Upper"],
        name="Banda superior",
        line=dict(dash="dot")
    ))

    fig_ind.add_trace(go.Scatter(
        x=df.index,
        y=df["BB_Lower"],
        name="Banda inferior",
        line=dict(dash="dot")
    ))

    fig_ind.update_layout(
        template="plotly_white",
        height=500,
        hovermode="x unified",
        xaxis_title="Fecha",
        yaxis_title="Precio ajustado"
    )

    st.plotly_chart(fig_ind, width='stretch')

    # Indicadores de momentum.
    with st.expander("Ver RSI, MACD y estocástico"):
        fig_momentum = go.Figure()

        for columna in ["RSI", "MACD", "Signal_Line", "%K", "%D"]:
            fig_momentum.add_trace(go.Scatter(
                x=df.index,
                y=df[columna],
                name=columna,
                mode="lines"
            ))

        fig_momentum.update_layout(
            template="plotly_white",
            height=450,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_momentum,
            width='stretch'
        )

    # Datos descargables.
    with st.expander("Ver predicciones y datos"):
        st.dataframe(pred, width='stretch')

        st.download_button(
            "Descargar predicciones CSV",
            data=pred.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker_actual}_predicciones.csv",
            mime="text/csv"
        )

        st.download_button(
            "Descargar indicadores CSV",
            data=df.to_csv().encode("utf-8"),
            file_name=f"{ticker_actual}_indicadores.csv",
            mime="text/csv"
        )

else:
    st.info(
        "Selecciona los parámetros en la barra lateral "
        "y pulsa «Ejecutar análisis»."
    )

st.divider()

st.caption(
    "Herramienta educativa de análisis cuantitativo. "
    "Las predicciones históricas no garantizan resultados futuros "
    "ni constituyen asesoramiento financiero."
)
