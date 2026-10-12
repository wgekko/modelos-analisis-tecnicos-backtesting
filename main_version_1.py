import streamlit as st
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import numpy as np

from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date, timedelta


# =========================================================
# CONFIGURACION DE LA APLICACION
# =========================================================

st.set_page_config(
    page_title="Predicción bursátil y Backtesting",
    page_icon=":material/finance_mode:",
    layout="wide"
)

st.subheader(":material/finance_mode: Análisis, Optimización y Backtesting Bursátil")
st.markdown(
    "Modelo de regresión OLS con optimización dinámica de indicadores técnicos, "
    "evaluación predictiva y simulación de backtesting."
)

st.info("al ejecutar el análisis y backtesting - los resultados pueden demorar varios minutos")

# =========================================================
# BARRA LATERAL: PARAMETROS
# =========================================================

st.sidebar.subheader(":material/settings: Configuración")

ticker = st.sidebar.text_input(
    "Ticker bursátil",
    value="NVDA"
).strip().upper()

fecha_inicio = st.sidebar.date_input(
    "Fecha de inicio",
    value=date(2022, 10, 25)
)

fecha_fin = st.sidebar.date_input(
    "Fecha de fin",
    value=date.today()
)

window_size = st.sidebar.slider(
    "Ventana de entrenamiento (días bursátiles)",
    min_value=20,
    max_value=120,
    value=30,
    step=5
)

optimizar = st.sidebar.checkbox(
    "Optimizar parámetros de indicadores",
    value=True,
    help="Busca automáticamente la mejor combinación de periodos para SMA, EMA y RSI."
)

ejecutar = st.sidebar.button(
    "Ejecutar análisis y backtest",
    type="primary",
    width='stretch'
)


# =========================================================
# DESCARGA DE DATOS
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def descargar_datos(ticker, fecha_inicio, fecha_fin):
    df = yf.download(
        ticker,
        start=fecha_inicio,
        end=fecha_fin + timedelta(days=1),
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    columnas = ["Open", "High", "Low", "Close", "Volume"]
    faltantes = [c for c in columnas if c not in df.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas necesarias: {faltantes}")

    df = df[columnas].copy()

    for columna in columnas:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df = df.dropna(subset=columnas)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df


# =========================================================
# CALCULO DE INDICADORES TECNICOS (CON PARAMETROS DINAMICOS)
# =========================================================

def calcular_indicadores(df, p_sma=50, p_ema=50, p_rsi=14):
    df = df.copy()

    # Variables rezagadas
    df["Previous_Close"] = df["Close"].shift(1)
    df["Close_shifted"] = df["Close"].shift(1)
    df["Open_shifted"] = df["Open"].shift(1)
    df["High_shifted"] = df["High"].shift(1)
    df["Low_shifted"] = df["Low"].shift(1)

    close = df["Close_shifted"]
    high = df["High_shifted"]
    low = df["Low_shifted"]

    # SMA Dinámica
    df["SMA"] = close.rolling(window=p_sma, min_periods=p_sma).mean()

    # EMA Dinámica
    df["EMA"] = close.ewm(span=p_ema, min_periods=p_ema, adjust=False).mean()

    # RSI Dinámico
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / p_rsi, min_periods=p_rsi, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / p_rsi, min_periods=p_rsi, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    df["RSI"] = rsi

    # MACD Estándar (12, 26, 9)
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, min_periods=9, adjust=False).mean()

    # Bandas de Bollinger (20, 2)
    bb_middle = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    df["BB_Middle"] = bb_middle
    df["BB_Upper"] = bb_middle + 2 * bb_std
    df["BB_Lower"] = bb_middle - 2 * bb_std

    # Oscilador Estocástico (14, 3)
    lowest_low = low.rolling(window=14, min_periods=14).min()
    highest_high = high.rolling(window=14, min_periods=14).max()
    rango = highest_high - lowest_low
    raw_k = 100 * (close - lowest_low) / rango
    raw_k = raw_k.mask(rango == 0, 0)
    df["%K"] = raw_k.rolling(window=3, min_periods=3).mean()
    df["%D"] = df["%K"].rolling(window=3, min_periods=3).mean()

    # ATR (14)
    previous_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


# =========================================================
# OPTIMIZACION DE PARAMETROS (GRID SEARCH)
# =========================================================

def optimizar_parametros(datos_raw, window_size):
    """Prueba combinaciones de periodos para SMA, EMA y RSI minimizando el MAE global."""
    mejores_params = {"SMA": 50, "EMA": 50, "RSI": 14}
    mejor_mae = float("inf")

    # Grillas de prueba razonables
    sma_vals = [20, 30, 50, 100]
    ema_vals = [20, 30, 50, 100]
    rsi_vals = [9, 14, 21]

    for p_sma in sma_vals:
        for p_ema in ema_vals:
            for p_rsi in rsi_vals:
                try:
                    df_test = calcular_indicadores(datos_raw, p_sma, p_ema, p_rsi)
                    if len(df_test) <= window_size:
                        continue
                    
                    # Prueba rápida con SMA como proxy de optimización
                    _, res = ejecutar_modelos_simple(df_test, window_size, ["SMA"])
                    if "SMA" in res and res["SMA"]["predictions"]:
                        mae_actual = mean_absolute_error(res["SMA"]["actual"], res["SMA"]["predictions"])
                        if mae_actual < mejor_mae:
                            mejor_mae = mae_actual
                            mejores_params = {"SMA": p_sma, "EMA": p_ema, "RSI": p_rsi}
                except Exception:
                    continue

    return mejores_params


def ejecutar_modelos_simple(df, window_size, lista_indicadores):
    resultados = {ind: {"predictions": [], "actual": [], "dates": []} for ind in lista_indicadores}
    for i in range(window_size, len(df)):
        train_df = df.iloc[i - window_size:i]
        test_row = df.iloc[[i]]
        actual = float(test_row["Close"].iloc[0])
        fecha = test_row.index[0]

        for indicador in lista_indicadores:
            if indicador not in train_df.columns:
                continue
            X_train = train_df[[indicador, "Previous_Close"]].astype(float)
            y_train = train_df["Close"].astype(float)

            if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(y_train.to_numpy()).all():
                continue

            X_train = sm.add_constant(X_train, has_constant="add")
            X_test = sm.add_constant(test_row[[indicador, "Previous_Close"]].astype(float), has_constant="add")

            try:
                modelo = sm.OLS(y_train, X_train).fit()
                prediccion = float(modelo.predict(X_test).iloc[0])
                if pd.isna(prediccion):
                    continue
                resultados[indicador]["predictions"].append(prediccion)
                resultados[indicador]["actual"].append(actual)
                resultados[indicador]["dates"].append(fecha)
            except Exception:
                continue
    return lista_indicadores, resultados


# =========================================================
# MODELO OLS CON VENTANA MOVIL
# =========================================================

def ejecutar_modelos(df, window_size):
    indicadores = [
        "SMA", "EMA", "RSI", "MACD", "Signal_Line",
        "BB_Upper", "BB_Middle", "BB_Lower", "%K", "%D", "ATR", "Close_shifted"
    ]

    resultados = {
        ind: {"predictions": [], "actual": [], "daily_mae": [], "dates": []}
        for ind in indicadores
    }

    for i in range(window_size, len(df)):
        train_df = df.iloc[i - window_size:i]
        test_row = df.iloc[[i]]

        actual = float(test_row["Close"].iloc[0])
        fecha = test_row.index[0]

        for indicador in indicadores:
            if indicador not in train_df.columns:
                continue
            columnas_x = [indicador, "Previous_Close"]
            X_train = train_df[columnas_x].astype(float)
            y_train = train_df["Close"].astype(float)

            if (
                X_train.isna().any().any()
                or y_train.isna().any()
                or not np.isfinite(X_train.to_numpy()).all()
                or not np.isfinite(y_train.to_numpy()).all()
            ):
                continue

            X_train = sm.add_constant(X_train, has_constant="add")
            X_test = sm.add_constant(test_row[columnas_x].astype(float), has_constant="add")

            try:
                modelo = sm.OLS(y_train, X_train).fit()
                prediccion = float(modelo.predict(X_test).iloc[0])

                if pd.isna(prediccion):
                    continue

                resultados[indicador]["predictions"].append(prediccion)
                resultados[indicador]["actual"].append(actual)
                resultados[indicador]["daily_mae"].append(abs(actual - prediccion))
                resultados[indicador]["dates"].append(fecha)

            except (ValueError, KeyError, np.linalg.LinAlgError):
                continue

    return indicadores, resultados


# =========================================================
# MOTOR DE BACKTESTING
# =========================================================

def ejecutar_backtest(df_base, resultados_mejor, capital_inicial=10000.0):
    """
    Simula una estrategia basada en la predicción del OLS:
    - Si la predicción de cierre de mañana > cierre actual -> Comprar / Mantener Long (1)
    - Si la predicción <= cierre actual -> Vender / Estar en liquidez (0)
    """
    fechas = resultados_mejor["dates"]
    preds = resultados_mejor["predictions"]
    reales = resultados_mejor["actual"]

    if not fechas:
        return pd.DataFrame(), {}

    bt_df = pd.DataFrame({
        "Date": fechas,
        "Prediction": preds,
        "Actual": reales
    }).set_index("Date")

    # Unir con los datos originales para obtener los precios de cierre exactos
    bt_df = bt_df.join(df_base[["Close"]], how="inner")
    bt_df["Market_Return"] = bt_df["Close"].pct_change()

    # Generación de señal: Predicción > Precio actual de cierre
    bt_df["Signal"] = np.where(bt_df["Prediction"] > bt_df["Close"].shift(1), 1, 0)
    bt_df["Strategy_Return"] = bt_df["Signal"].shift(1) * bt_df["Market_Return"]

    bt_df = bt_df.dropna()

    # Curvas de capital
    bt_df["Buy_Hold_Equity"] = capital_inicial * (1 + bt_df["Market_Return"]).cumprod()
    bt_df["Strategy_Equity"] = capital_inicial * (1 + bt_df["Strategy_Return"]).cumprod()

    # Métricas clave de rendimiento
    dias_totales = len(bt_df)
    retorno_total_strat = (bt_df["Strategy_Equity"].iloc[-1] / capital_inicial) - 1
    retorno_total_bh = (bt_df["Buy_Hold_Equity"].iloc[-1] / capital_inicial) - 1

    # Sharpe Ratio anualizado (asumiendo 252 ruedas bursátiles, tasa libre de riesgo = 0)
    strat_vol = bt_df["Strategy_Return"].std() * np.sqrt(252)
    sharpe_ratio = (bt_df["Strategy_Return"].mean() * 252) / strat_vol if strat_vol > 0 else 0.0

    # Maximum Drawdown
    rolling_max = bt_df["Strategy_Equity"].cummax()
    drawdown = (bt_df["Strategy_Equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Win Rate de operaciones
    trades = bt_df["Strategy_Return"][bt_df["Strategy_Return"] != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0.0

    metricas_bt = {
        "Retorno Estrategia": retorno_total_strat,
        "Retorno Buy & Hold": retorno_total_bh,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Operaciones Realizadas": len(trades)
    }

    return bt_df, metricas_bt


# =========================================================
# EJECUCION PRINCIPAL
# =========================================================

if ejecutar:
    if not ticker:
        st.error("Introduce un ticker bursátil válido.")
    elif fecha_inicio >= fecha_fin:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
    else:
        try:
            with st.spinner("Descargando datos, optimizando indicadores y ejecutando backtest..."):
                datos_raw = descargar_datos(ticker, fecha_inicio, fecha_fin)

                if datos_raw.empty:
                    st.error("No se encontraron datos para el ticker y fechas seleccionadas.")
                    st.stop()

                # Optimización o valores por defecto
                if optimizar:
                    params = optimizar_parametros(datos_raw, window_size)
                else:
                    params = {"SMA": 50, "EMA": 50, "RSI": 14}

                df = calcular_indicadores(datos_raw, params["SMA"], params["EMA"], params["RSI"])

                if len(df) <= window_size:
                    st.error("Muy pocos datos para la ventana seleccionada. Amplía el rango o reduce la ventana.")
                    st.stop()

                indicadores, resultados = ejecutar_modelos(df, window_size)

                # Tabla de precisión
                filas = []
                for indicador in indicadores:
                    preds = resultados[indicador]["predictions"]
                    reales = resultados[indicador]["actual"]
                    if len(reales) == 0:
                        continue
                    filas.append({
                        "Indicador": indicador,
                        "MAE": mean_absolute_error(reales, preds),
                        "MSE": mean_squared_error(reales, preds),
                        "Predicciones": len(reales)
                    })

                accuracy_df = pd.DataFrame(filas).sort_values("MAE").reset_index(drop=True)

                # Mejor indicador para Backtesting
                mejor_ind_nombre = accuracy_df.iloc[0]["Indicador"]
                bt_df, metricas_bt = ejecutar_backtest(df, resultados[mejor_ind_nombre])

                # Guardar en Session State
                st.session_state["df"] = df
                st.session_state["resultados"] = resultados
                st.session_state["indicadores"] = indicadores
                st.session_state["accuracy_df"] = accuracy_df
                st.session_state["ticker"] = ticker
                st.session_state["window_size"] = window_size
                st.session_state["params"] = params
                st.session_state["bt_df"] = bt_df
                st.session_state["metricas_bt"] = metricas_bt
                st.session_state["mejor_ind"] = mejor_ind_nombre

        except Exception as e:
            st.error(f"Se produjo un error durante la ejecución: {e}")


# =========================================================
# PRESENTACION DE RESULTADOS
# =========================================================

if "accuracy_df" in st.session_state:
    df = st.session_state["df"]
    resultados = st.session_state["resultados"]
    indicadores = st.session_state["indicadores"]
    accuracy_df = st.session_state["accuracy_df"]
    ticker_actual = st.session_state["ticker"]
    window_actual = st.session_state["window_size"]
    params = st.session_state["params"]
    bt_df = st.session_state["bt_df"]
    metricas_bt = st.session_state["metricas_bt"]
    mejor_ind = st.session_state["mejor_ind"]

    st.subheader(f"Resultados de Análisis y Optimización — {ticker_actual}")

    # Parámetros óptimos usados
    if st.session_state.get("params"):
        st.info(f"⚙️ **Parámetros óptimos encontrados:** SMA: {params['SMA']} | EMA: {params['EMA']} | RSI: {params['RSI']}")

    # Resumen métricas principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sesiones analizadas", len(df))
    col2.metric("Mejor indicador (MAE)", mejor_ind)
    col3.metric("Retorno Estrategia", f"{metricas_bt.get('Retorno Estrategia', 0)*100:.2f}%")
    col4.metric("Sharpe Ratio", f"{metricas_bt.get('Sharpe Ratio', 0):.2f}")

    # SECCION 1: BACKTESTING Y RENDIMIENTO FINANCIERO
    st.subheader("1. Backtesting de la Estrategia vs. Buy & Hold")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Retorno Buy & Hold", f"{metricas_bt.get('Retorno Buy & Hold', 0)*100:.2f}%")
    col_b.metric("Máximo Drawdown", f"{metricas_bt.get('Maximum Drawdown', 0)*100:.2f}%")
    col_c.metric("Win Rate", f"{metricas_bt.get('Win Rate', 0)*100:.1f}%")

    if not bt_df.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Strategy_Equity"], mode="lines", name="Estrategia OLS", line=dict(color="green", width=2)))
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Buy_Hold_Equity"], mode="lines", name="Buy & Hold (Activo)", line=dict(color="gray", dash="dot")))
        fig_bt.update_layout(
            title="Curva de Capital Acumulado (Capital Inicial: $10,000)",
            xaxis_title="Fecha",
            yaxis_title="Capital ($)",
            template="plotly_dark",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_bt, width='stretch')

    # SECCION 2: TABLA DE PRECISION
    st.subheader("2. Comparación de Precisión por Indicador")
    st.dataframe(
        accuracy_df.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "Predicciones": "{:.0f}"}),
        width='stretch',
        hide_index=True
    )

    # SECCION 3: GRAFICO DE PRECIO E INDICADORES OPTIMIZADOS
    st.subheader("3. Precio de cierre e indicadores optimizados")
    fig_precio = go.Figure()
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Precio de cierre"))
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["SMA"], mode="lines", name=f"SMA ({params['SMA']})"))
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["EMA"], mode="lines", name=f"EMA ({params['EMA']})"))
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines", name="Banda superior", line=dict(dash="dot")))
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines", name="Banda inferior", line=dict(dash="dot")))
    
    fig_precio.update_layout(
        title=f"{ticker_actual}: Precio e Indicadores con Parámetros Optimizados",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark",
        height=550,
        hovermode="x unified"
    )
    st.plotly_chart(fig_precio, width='stretch')

    # SECCION 4: MACD
    st.subheader("4. MACD y Línea de Señal")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], mode="lines", name="Línea de señal"))
    fig_macd.update_layout(title="MACD", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white", height=350, hovermode="x unified")
    st.plotly_chart(fig_macd, width='stretch')

    # Datos y descarga
    with st.expander("Ver datos e indicadores calculados"):
        st.dataframe(df.sort_index(ascending=False), width='stretch')

    st.download_button(
        "Descargar métricas CSV",
        data=accuracy_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker_actual}_metricas.csv",
        mime="text/csv"
    )

else:
    st.info("Configura el ticker, fechas y parámetros en la barra lateral y pulsa «Ejecutar análisis y backtest».")