import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# CONFIGURACION DE LA APLICACION
# =========================================================

st.set_page_config(
    page_title="Predicción Multivariada y Backtesting Avanzado",
    page_icon=":material/finance_mode:",
    layout="wide"
)

st.subheader(":material/finance_mode: Análisis Multivariado, Backtesting y Gestión de Riesgo")
st.markdown(
    "Modelo de regresión penalizada (Ridge/Lasso/ElasticNet) con indicadores técnicos dinámicos, "
    "validación cruzada temporal y simulación de backtesting con Stop Loss/Take Profit basados en ATR."
)

st.info("al ejecutar el análisis y backtesting - los resultados pueden demorar varios minutos")

# =========================================================
# BARRA LATERAL: PARAMETROS
# =========================================================

st.sidebar.subheader(":material/settings: Configuración General")

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
    "Ventana de entrenamiento móvil (días)",
    min_value=20,
    max_value=120,
    value=40,
    step=5
)
st.info("al ejecutar el análisis y backtesting - los resultados pueden demorar varios minutos")
st.sidebar.subheader("Modelo Multivariado")
modelo_tipo = st.sidebar.selectbox(
    "Algoritmo de Regresión",
    ["Ridge", "Lasso", "ElasticNet"],
    index=0
)

alpha_val = st.sidebar.slider(
    "Parámetro de Regularización (Alpha)",
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.1
)

st.sidebar.subheader("Backtesting y Gestión de Riesgo")
comision = st.sidebar.slider(
    "Comisión / Deslizamiento por operación (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.05
) / 100.0

sl_mult = st.sidebar.slider(
    "Stop Loss (múltiplos de ATR)",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)

tp_mult = st.sidebar.slider(
    "Take Profit (múltiplos de ATR)",
    min_value=1.0,
    max_value=8.0,
    value=3.0,
    step=0.5
)

optimizar = st.sidebar.checkbox(
    "Optimizar periodos de indicadores (SMA/EMA/RSI)",
    value=True
)

ejecutar = st.sidebar.button(
    "Ejecutar análisis completo",
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
# CALCULO DE INDICADORES TECNICOS
# =========================================================

def calcular_indicadores(df, p_sma=50, p_ema=50, p_rsi=14):
    df = df.copy()

    df["Previous_Close"] = df["Close"].shift(1)
    df["Close_shifted"] = df["Close"].shift(1)
    df["Open_shifted"] = df["Open"].shift(1)
    df["High_shifted"] = df["High"].shift(1)
    df["Low_shifted"] = df["Low"].shift(1)

    close = df["Close_shifted"]
    high = df["High_shifted"]
    low = df["Low_shifted"]

    # SMA y EMA dinámicas
    df["SMA"] = close.rolling(window=p_sma, min_periods=p_sma).mean()
    df["EMA"] = close.ewm(span=p_ema, min_periods=p_ema, adjust=False).mean()

    # RSI dinámico
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

    # MACD
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, min_periods=9, adjust=False).mean()

    # Bandas de Bollinger
    bb_middle = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    df["BB_Middle"] = bb_middle
    df["BB_Upper"] = bb_middle + 2 * bb_std
    df["BB_Lower"] = bb_middle - 2 * bb_std

    # Oscilador Estocástico
    lowest_low = low.rolling(window=14, min_periods=14).min()
    highest_high = high.rolling(window=14, min_periods=14).max()
    rango = highest_high - lowest_low
    raw_k = 100 * (close - lowest_low) / rango
    raw_k = raw_k.mask(rango == 0, 0)
    df["%K"] = raw_k.rolling(window=3, min_periods=3).mean()
    df["%D"] = df["%K"].rolling(window=3, min_periods=3).mean()

    # ATR
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
# OPTIMIZACION DE PARAMETROS DE INDICADORES
# =========================================================

def optimizar_parametros_indicadores(datos_raw, window_size, modelo_tipo, alpha_val):
    mejores_params = {"SMA": 50, "EMA": 50, "RSI": 14}
    mejor_mae = float("inf")

    sma_vals = [20, 50, 100]
    ema_vals = [20, 50, 100]
    rsi_vals = [9, 14, 21]

    for p_sma in sma_vals:
        for p_ema in ema_vals:
            for p_rsi in rsi_vals:
                try:
                    df_t = calcular_indicadores(datos_raw, p_sma, p_ema, p_rsi)
                    if len(df_t) <= window_size:
                        continue
                    mae_cv, _ = evaluar_timeseries_split(df_t, modelo_tipo, alpha_val, window_size)
                    if mae_cv < mejor_mae and mae_cv > 0:
                        mejor_mae = mae_cv
                        mejores_params = {"SMA": p_sma, "EMA": p_ema, "RSI": p_rsi}
                except Exception:
                    continue
    return mejores_params


# =========================================================
# VALIDACION CRUZADA TEMPORAL (TimeSeriesSplit)
# =========================================================

def evaluar_timeseries_split(df, modelo_tipo, alpha_val, window_size):
    features = [
        "SMA", "EMA", "RSI", "MACD", "Signal_Line",
        "BB_Upper", "BB_Middle", "BB_Lower", "%K", "%D", "ATR", "Previous_Close"
    ]
    features = [f for f in features if f in df.columns]

    X = df[features].astype(float)
    y = df["Close"].astype(float)

    tscv = TimeSeriesSplit(n_splits=3)
    maes = []
    mses = []
    scaler = StandardScaler()

    for train_index, test_index in tscv.split(X):
        if len(train_index) < window_size:
            continue
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_tr, y_te = y.iloc[train_index], y.iloc[test_index]

        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        if modelo_tipo == "Ridge":
            m = Ridge(alpha=alpha_val)
        elif modelo_tipo == "Lasso":
            m = Lasso(alpha=alpha_val)
        else:
            m = ElasticNet(alpha=alpha_val, l1_ratio=0.5)

        m.fit(X_tr_sc, y_tr)
        preds = m.predict(X_te_sc)

        maes.append(mean_absolute_error(y_te, preds))
        mses.append(mean_squared_error(y_te, preds))

    return (np.mean(maes) if maes else 0.0), (np.mean(mses) if mses else 0.0)


# =========================================================
# MODELO MULTIVARIADO CON VENTANA MOVIL
# =========================================================

def ejecutar_modelo_multivariado(df, window_size, modelo_tipo, alpha_val):
    features = [
        "SMA", "EMA", "RSI", "MACD", "Signal_Line",
        "BB_Upper", "BB_Middle", "BB_Lower", "%K", "%D", "ATR", "Previous_Close"
    ]
    features = [f for f in features if f in df.columns]

    predictions = []
    actuals = []
    dates = []
    daily_mae = []

    scaler = StandardScaler()

    for i in range(window_size, len(df)):
        train_df = df.iloc[i - window_size:i]
        test_row = df.iloc[[i]]

        actual = float(test_row["Close"].iloc[0])
        fecha = test_row.index[0]

        X_train = train_df[features].astype(float)
        y_train = train_df["Close"].astype(float)

        if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(y_train.to_numpy()).all():
            continue

        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(test_row[features].astype(float))

        if modelo_tipo == "Ridge":
            model = Ridge(alpha=alpha_val)
        elif modelo_tipo == "Lasso":
            model = Lasso(alpha=alpha_val)
        else:
            model = ElasticNet(alpha=alpha_val, l1_ratio=0.5)

        try:
            model.fit(X_train_sc, y_train)
            prediccion = float(model.predict(X_test_sc)[0])

            if pd.isna(prediccion):
                continue

            predictions.append(prediccion)
            actuals.append(actual)
            daily_mae.append(abs(actual - prediccion))
            dates.append(fecha)
        except Exception:
            continue

    resultados_dict = {
        "dates": dates,
        "predictions": predictions,
        "actual": actuals,
        "daily_mae": daily_mae
    }
    return resultados_dict


# =========================================================
# MOTOR DE BACKTESTING CON STOP LOSS Y TAKE PROFIT (ATR)
# =========================================================

def ejecutar_backtest(df, resultados, sl_mult, tp_mult, comision, capital_inicial=10000.0):
    fechas = resultados["dates"]
    preds = resultados["predictions"]
    reales = resultados["actual"]

    if not fechas:
        return pd.DataFrame(), {}

    bt_df = pd.DataFrame({
        "Date": fechas,
        "Prediction": preds,
        "Actual": reales
    }).set_index("Date")

    bt_df = bt_df.join(df[["Open", "High", "Low", "Close", "ATR"]], how="inner")
    bt_df["Signal"] = np.where(bt_df["Prediction"] > bt_df["Close"].shift(1), 1, 0)

    cash = capital_inicial
    shares = 0
    in_position = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0

    portfolio_values = []

    for i in range(1, len(bt_df)):
        fecha = bt_df.index[i]
        row = bt_df.iloc[i]
        prev_row = bt_df.iloc[i-1]

        open_p = row['Open']
        high_p = row['High']
        low_p = row['Low']
        close_p = row['Close']
        atr = prev_row['ATR']
        signal = prev_row['Signal']

        if not in_position and signal == 1:
            entry_price = open_p * (1 + comision)
            shares = cash / entry_price
            cash = 0
            in_position = True
            sl_price = entry_price - (sl_mult * atr)
            tp_price = entry_price + (tp_mult * atr)
        elif in_position:
            exit_price = None
            if low_p <= sl_price:
                exit_price = sl_price * (1 - comision)
                in_position = False
            elif high_p >= tp_price:
                exit_price = tp_price * (1 - comision)
                in_position = False
            elif signal == 0:
                exit_price = open_p * (1 - comision)
                in_position = False

            if not in_position:
                cash = shares * exit_price
                shares = 0

        current_val = cash if not in_position else shares * close_p
        portfolio_values.append({
            'Date': fecha,
            'Portfolio_Value': current_val,
            'Close': close_p
        })

    bt_res = pd.DataFrame(portfolio_values).set_index('Date')
    if bt_res.empty:
        return pd.DataFrame(), {}

    bt_res['Buy_Hold_Equity'] = capital_inicial * (bt_res['Close'] / bt_res['Close'].iloc[0])
    retorno_strat = (bt_res['Portfolio_Value'].iloc[-1] / capital_inicial) - 1
    retorno_bh = (bt_res['Buy_Hold_Equity'].iloc[-1] / capital_inicial) - 1

    strat_returns = bt_res['Portfolio_Value'].pct_change().dropna()
    strat_vol = strat_returns.std() * np.sqrt(252)
    sharpe = (strat_returns.mean() * 252) / strat_vol if strat_vol > 0 else 0.0

    rolling_max = bt_res['Portfolio_Value'].cummax()
    dd = (bt_res['Portfolio_Value'] - rolling_max) / rolling_max
    max_dd = dd.min()

    metricas_bt = {
        "Retorno Estrategia": retorno_strat,
        "Retorno Buy & Hold": retorno_bh,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_dd
    }

    return bt_res, metricas_bt


# =========================================================
# EJECUCION PRINCIPAL
# =========================================================

if ejecutar:
    if not ticker:
        st.error("Introduce un ticker válido.")
    elif fecha_inicio >= fecha_fin:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
    else:
        try:
            with st.spinner("Procesando datos, optimizando y ejecutando modelo multivariado..."):
                datos_raw = descargar_datos(ticker, fecha_inicio, fecha_fin)

                if datos_raw.empty:
                    st.error("No se encontraron datos para el ticker y fechas seleccionadas.")
                    st.stop()

                if optimizar:
                    params = optimizar_parametros_indicadores(datos_raw, window_size, modelo_tipo, alpha_val)
                else:
                    params = {"SMA": 50, "EMA": 50, "RSI": 14}

                df = calcular_indicadores(datos_raw, params["SMA"], params["EMA"], params["RSI"])

                if len(df) <= window_size:
                    st.error("Datos insuficientes para la ventana de entrenamiento seleccionada.")
                    st.stop()

                # Validación cruzada temporal
                mae_cv, mse_cv = evaluar_timeseries_split(df, modelo_tipo, alpha_val, window_size)

                # Ejecutar modelo multivariado en ventana móvil
                resultados = ejecutar_modelo_multivariado(df, window_size, modelo_tipo, alpha_val)

                if not resultados["predictions"]:
                    st.error("El modelo no generó predicciones válidas.")
                    st.stop()

                # Backtesting con SL / TP y Comisiones
                bt_res, metricas_bt = ejecutar_backtest(df, resultados, sl_mult, tp_mult, comision)

                # Guardar en Session State
                st.session_state["df"] = df
                st.session_state["resultados"] = resultados
                st.session_state["bt_res"] = bt_res
                st.session_state["metricas_bt"] = metricas_bt
                st.session_state["ticker"] = ticker
                st.session_state["params"] = params
                st.session_state["mae_cv"] = mae_cv
                st.session_state["mse_cv"] = mse_cv
                st.session_state["modelo_tipo"] = modelo_tipo

        except Exception as e:
            st.error(f"Error durante la ejecución: {e}")


# =========================================================
# PRESENTACION DE RESULTADOS EN STREAMLIT
# =========================================================

if "bt_res" in st.session_state:
    df = st.session_state["df"]
    resultados = st.session_state["resultados"]
    bt_res = st.session_state["bt_res"]
    metricas_bt = st.session_state["metricas_bt"]
    ticker_actual = st.session_state["ticker"]
    params = st.session_state["params"]
    mae_cv = st.session_state["mae_cv"]
    mse_cv = st.session_state["mse_cv"]
    mod_usado = st.session_state["modelo_tipo"]

    st.subheader(f"Resultados de Análisis Multivariado — {ticker_actual}")

    st.info(
        f":material/modeling: **Modelo:** {mod_usado} | "
        f":material/modeling: **Parámetros óptimos:** SMA: {params['SMA']} | EMA: {params['EMA']} | RSI: {params['RSI']} | "
        f":material/modeling: **MAE (CV Temporal):** {mae_cv:.4f}"
    )

    # Tarjetas de métricas principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retorno Estrategia", f"{metricas_bt.get('Retorno Estrategia', 0)*100:.2f}%")
    col2.metric("Retorno Buy & Hold", f"{metricas_bt.get('Retorno Buy & Hold', 0)*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{metricas_bt.get('Sharpe Ratio', 0):.2f}")
    col4.metric("Máximo Drawdown", f"{metricas_bt.get('Maximum Drawdown', 0)*100:.2f}%")

    # SECCION 1: GRAFICO DE BACKTESTING (CURVA DE CAPITAL)
    st.subheader("1. Curva de Capital: Estrategia Multivariada vs. Buy & Hold")
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_res.index, y=bt_res["Portfolio_Value"], mode="lines", name="Estrategia con Riesgo (ATR)", line=dict(color="#00FF7F", width=2)))
    fig_bt.add_trace(go.Scatter(x=bt_res.index, y=bt_res["Buy_Hold_Equity"], mode="lines", name="Buy & Hold", line=dict(color="#A9A9A9", dash="dot")))
    fig_bt.update_layout(
        title="Evolución del Capital ($10,000 iniciales con comisiones y SL/TP)",
        xaxis_title="Fecha",
        yaxis_title="Capital ($)",
        template="plotly_dark",
        height=450,
        hovermode="x unified"
    )
    st.plotly_chart(fig_bt, width='stretch')

    # SECCION 2: GRAFICO DE ERRORES DIARIOS
    st.subheader("2. Evolución del Error Absoluto Diario")
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=resultados["dates"], y=resultados["daily_mae"], mode="lines", name="Error Absoluto", line=dict(color="#FF4500")))
    fig_err.update_layout(
        title="Error Absoluto Diario (Predicción vs. Real)",
        xaxis_title="Fecha",
        yaxis_title="Error Absoluto ($)",
        template="plotly_dark",
        height=350,
        hovermode="x unified"
    )
    st.plotly_chart(fig_err, width='stretch')

    # SECCION 3: PRECIOS E INDICADORES TECNICOS
    st.subheader("3. Precio de Cierre e Indicadores Técnicos")
    fig_precio = go.Figure()
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Precio de cierre"))
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["SMA"], mode="lines", name=f"SMA ({params['SMA']})"))
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["EMA"], mode="lines", name=f"EMA ({params['EMA']})"))
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines", name="Banda superior", line=dict(dash="dot")))
    fig_precio.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines", name="Banda inferior", line=dict(dash="dot")))
    fig_precio.update_layout(
        title=f"{ticker_actual}: Precio e Indicadores Principales",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig_precio, width='stretch')

    # Datos y descarga
    with st.expander("Ver datos completos de la simulación"):
        st.dataframe(bt_res.sort_index(ascending=False), width='stretch')

    st.download_button(
        "Descargar resultados de Backtest (CSV)",
        data=bt_res.to_csv().encode("utf-8"),
        file_name=f"{ticker_actual}_backtest_multivariado.csv",
        mime="text/csv"
    )

else:
    st.info("Configura los parámetros en la barra lateral y pulsa «Ejecutar análisis completo».")