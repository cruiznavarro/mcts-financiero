"""
mcts_portfolio_relativo.py
==========================
Monte Carlo Tree Search para optimización de cartera multi-activo
con recompensa relativa al índice de referencia (SPY).

Correcciones respecto a mcts_portfolio.py (que fracasó con ROI -6%):
  1. Espacio de acciones reducido a 7 acciones semánticas (en vez de 70)
     → árbol bien explorado con 500 iteraciones (~70 visitas/acción)
  2. Recompensa relativa vs SPY (en vez de retorno absoluto)
     → el agente aprende a superar al índice, no solo a ganar dinero
  3. Rebalanceo semanal (cada 5 días, en vez de diario)
     → costes de transacción reducidos en ~80%

Flujo del programa:
    1. Descargar precios históricos de N activos + SPY (yfinance)
    2. Para cada semana, ejecutar MCTS y elegir la mejor rotación
    3. Registrar evolución de la cartera y pesos
    4. Comparar con SPY B&H y Equal-Weight B&H
    5. Graficar y guardar reporte markdown

Dependencias: numpy, pandas, matplotlib, yfinance
"""

from __future__ import annotations

import math
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# SECCIÓN 0 — PARÁMETROS GLOBALES
# Modificar aquí para cambiar el experimento sin tocar el código.
# =============================================================================

TICKERS       = ["AAPL", "XOM", "GLD", "TLT", "IWM"]  # Activos de la cartera
SPY_TICK      = "SPY"                                    # Índice de referencia
PERIOD        = "2y"                                     # Período de descarga
CAPITAL_INIT  = 10_000.0                                 # Capital inicial ($)

ITERACIONES   = 5000    # Iteraciones MCTS por decisión
DIAS_ROLLOUT  = 20     # Horizonte de simulación GBM (días)
VENTANA_CALIB = 60     # Ventana rolling para calibrar mu y Sigma (días)
N_PATHS       = 20     # Trayectorias GBM por rollout (balance velocidad/varianza)
C_UCB         = math.sqrt(2)  # Constante de exploración UCT (estándar Kocsis 2006)

TILT          = 0.20   # Magnitud de rotación por acción (20% del valor total)
TX_COST       = 0.001  # Coste de transacción (0.1% del turnover)
REBAL_FREQ    = 5      # Rebalancear cada N días (5 = semanal)
VENTANA_MOM   = 20     # Ventana para calcular momentum relativo (días)

SEMILLA       = 42
OUTPUT_DIR    = "results"


# =============================================================================
# SECCIÓN 1 — DESCARGA DE DATOS
# =============================================================================

def descargar_precios(
    tickers: list[str],
    spy_tick: str,
    periodo: str,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Descarga precios de cierre ajustados de N activos + SPY.

    Retorna
    -------
    prices   : Matriz (T, N) de precios de cierre de los activos.
    spy      : Vector (T,) de precios de cierre de SPY.
    dates    : DatetimeIndex con las fechas de trading.
    tickers  : Lista de símbolos en el orden de las columnas de prices.
    """
    todos = tickers + [spy_tick]
    print(f"[INFO] Descargando {todos} ({periodo})...")

    df = yf.download(todos, period=periodo, progress=False, auto_adjust=True)["Close"]
    df = df.dropna()

    if df.empty:
        raise ValueError("No hay datos. Revisa los tickers y el período.")

    print(f"[INFO] {len(df)} sesiones  |  {df.index[0].date()} → {df.index[-1].date()}")

    prices = df[tickers].to_numpy(dtype=float)   # (T, N)
    spy    = df[spy_tick].to_numpy(dtype=float)  # (T,)
    dates  = df.index

    for i, t in enumerate(tickers):
        print(f"       {t:6s}  {prices[0, i]:.2f}$ → {prices[-1, i]:.2f}$  "
              f"({(prices[-1,i]/prices[0,i]-1)*100:+.1f}%)")
    print(f"       {spy_tick:6s}  {spy[0]:.2f}$ → {spy[-1]:.2f}$  "
          f"({(spy[-1]/spy[0]-1)*100:+.1f}%)")

    return prices, spy, dates, tickers


# =============================================================================
# SECCIÓN 2 — ESPACIO DE ACCIONES
#
# En vez del símplex completo (70 acciones para N=5, g=4), usamos 7 acciones
# semánticas que el árbol puede explorar completamente con 500 iteraciones.
#
#   ROTAR_HACIA_i : Desplazar TILT del valor total hacia el activo i,
#                   redistribuyendo el exceso desde los demás activos.
#   IGUAL_PESO    : Rebalancear a 1/N para todos los activos.
#   MANTENER      : No operar (pesos sin cambio, sin coste de transacción).
# =============================================================================

def construir_acciones(tickers: list[str]) -> list[str]:
    """Construye la lista de acciones a partir de los tickers."""
    return [f"ROTAR_HACIA_{t}" for t in tickers] + ["IGUAL_PESO", "MANTENER"]


def aplicar_accion(pesos: np.ndarray, accion_idx: int,
                   tickers: list[str], tilt: float = TILT) -> np.ndarray:
    """
    Aplica una acción al vector de pesos actual.

    ROTAR_HACIA_i: aumenta el peso del activo i en `tilt`, reduciendo
                   los demás proporcionalmente a sus pesos actuales.
    IGUAL_PESO:    fuerza pesos iguales (1/N).
    MANTENER:      devuelve los mismos pesos (sin coste de transacción).

    Parámetros
    ----------
    pesos     : Pesos actuales. Shape (N,), suma = 1.
    accion_idx: Índice en la lista devuelta por construir_acciones().
    tickers   : Lista de símbolos (para interpretar el índice).
    tilt      : Magnitud de la rotación.

    Retorna
    -------
    np.ndarray : Nuevos pesos. Shape (N,), suma = 1.
    """
    N = len(tickers)

    if accion_idx == N + 1:          # MANTENER
        return pesos.copy()

    if accion_idx == N:              # IGUAL_PESO
        return np.ones(N) / N

    # ROTAR_HACIA_i
    i = accion_idx
    new_w = pesos.copy()
    delta = min(tilt, 1.0 - pesos[i])  # no sobrepasar peso=1
    new_w[i] += delta

    # Reducir proporcionalmente desde el resto
    idx_resto = [j for j in range(N) if j != i]
    suma_resto = sum(pesos[j] for j in idx_resto)
    if suma_resto > 1e-10:
        for j in idx_resto:
            new_w[j] = max(0.0, pesos[j] - delta * pesos[j] / suma_resto)

    # Renormalizar por seguridad numérica
    total = new_w.sum()
    if total > 1e-10:
        new_w /= total
    else:
        new_w = np.ones(N) / N

    return new_w


# =============================================================================
# SECCIÓN 3 — ESTADO Y SEÑALES RELATIVAS
#
# El estado captura "comparar precios": en vez de precios absolutos,
# usamos rankings de momentum y retorno relativo vs SPY.
# Esto hace el estado invariante a la escala del precio.
# =============================================================================

def _ranking_normalizado(valores: np.ndarray) -> np.ndarray:
    """Convierte un vector a ranking normalizado en [0, 1]."""
    N = len(valores)
    if N <= 1:
        return np.array([0.5])
    orden = np.argsort(valores)
    rango = np.empty(N)
    rango[orden] = np.arange(N) / (N - 1)
    return rango


def calcular_momentum(prices: np.ndarray, dia: int,
                      ventana: int = VENTANA_MOM) -> np.ndarray:
    """
    Retorno log de cada activo en la ventana, normalizado a ranking [0,1].

    momentum[i] = 1 → activo i tiene el mejor retorno de los últimos `ventana` días.
    momentum[i] = 0 → activo i tiene el peor retorno.
    """
    inicio = max(0, dia - ventana)
    if inicio >= dia:
        return np.ones(prices.shape[1]) * 0.5
    log_ret = np.log(prices[dia] / prices[inicio])
    return _ranking_normalizado(log_ret)


def calcular_vol_relativa(prices: np.ndarray, dia: int,
                          ventana: int = VENTANA_MOM) -> np.ndarray:
    """
    Volatilidad de cada activo en la ventana, normalizada a ranking [0,1].

    vol_rel[i] = 1 → activo más volátil (mayor riesgo).
    vol_rel[i] = 0 → activo menos volátil.
    """
    inicio = max(0, dia - ventana)
    if dia - inicio < 2:
        return np.ones(prices.shape[1]) * 0.5
    log_rets = np.diff(np.log(prices[inicio:dia + 1]), axis=0)
    vols = np.std(log_rets, axis=0, ddof=1)
    return _ranking_normalizado(vols)


def calcular_ret_vs_spy(prices: np.ndarray, spy: np.ndarray,
                        dia: int, ventana: int = VENTANA_MOM) -> np.ndarray:
    """
    Retorno de cada activo relativo a SPY en la ventana.

    ret_vs_spy[i] > 0 → activo i superó a SPY.
    ret_vs_spy[i] < 0 → activo i quedó por debajo de SPY.
    """
    inicio = max(0, dia - ventana)
    if inicio >= dia:
        return np.zeros(prices.shape[1])
    log_ret_activos = np.log(prices[dia] / prices[inicio])  # (N,)
    log_ret_spy     = math.log(spy[dia] / spy[inicio])
    return log_ret_activos - log_ret_spy


def crear_estado(dia: int, pesos: np.ndarray, valor: float,
                 prices: np.ndarray, spy: np.ndarray) -> dict:
    """
    Crea el estado completo del agente en el día `dia`.

    El estado incluye:
      - pesos     : pesos actuales de la cartera
      - valor     : valor total de la cartera
      - momentum  : ranking de momentum de 20d (quién sube más)
      - vol_rel   : ranking de volatilidad de 20d (quién oscila más)
      - ret_vs_spy: retorno relativo a SPY de 20d (quién supera al índice)
    """
    return {
        "dia":        dia,
        "pesos":      pesos.copy(),
        "valor":      valor,
        "momentum":   calcular_momentum(prices, dia),
        "vol_rel":    calcular_vol_relativa(prices, dia),
        "ret_vs_spy": calcular_ret_vs_spy(prices, spy, dia),
    }


# =============================================================================
# SECCIÓN 4 — CALIBRACIÓN GBM MULTIVARIADO
#
# Reutiliza la lógica de mcts_portfolio.py: calibra mu y Sigma con una
# ventana rolling para evitar lookback bias, y aplica descomposición de
# Cholesky para simular activos correlacionados correctamente.
# =============================================================================

def calibrar_gbm_multi(prices_ext: np.ndarray, dia: int) -> tuple:
    """
    Estima mu (drift diario) y Sigma (covarianza diaria) para los N+1
    activos (N activos + SPY) usando los últimos VENTANA_CALIB días.

    Parámetros
    ----------
    prices_ext : Matriz (T, N+1) — últimas N columnas son los activos,
                 la columna N es SPY.
    dia        : Día actual.

    Retorna
    -------
    (mu, Sigma) : mu shape (N+1,), Sigma shape (N+1, N+1).
    """
    M = prices_ext.shape[1]
    inicio = max(0, dia - VENTANA_CALIB)
    ventana = prices_ext[inicio:dia + 1]

    if len(ventana) >= 3:
        log_ret = np.diff(np.log(ventana), axis=0)   # (w-1, M)
        mu      = np.mean(log_ret, axis=0)            # (M,)
        Sigma   = np.cov(log_ret, rowvar=False)       # (M, M)
        Sigma  += np.eye(M) * 1e-8                    # regularización
    else:
        mu    = np.zeros(M)
        Sigma = np.eye(M) * (0.01 ** 2)

    return mu, Sigma


def cholesky_safe(Sigma: np.ndarray) -> np.ndarray:
    """Descomposición de Cholesky con fallback diagonal si no es DP."""
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return np.diag(np.sqrt(np.maximum(np.diag(Sigma), 1e-10)))


# =============================================================================
# SECCIÓN 5 — LOS 4 PASOS DEL MCTS
#
# Arquitectura funcional (igual que mcts_simple.py).
# Los nodos son diccionarios simples para máxima legibilidad.
# =============================================================================

def crear_nodo(accion_idx: int | None, padre: dict | None) -> dict:
    """Crea un nodo del árbol MCTS."""
    return {
        "accion_idx": accion_idx,  # índice en ACCIONES (None = raíz)
        "padre":      padre,
        "hijos":      [],
        "n":          0,           # visitas
        "w":          0.0,         # recompensa acumulada
    }


def ucb1(nodo: dict, c: float = C_UCB) -> float:
    """
    Calcula el valor UCT del nodo.

    UCT(i) = w/n  +  c * sqrt(ln(N_padre) / n)
             ────    ─────────────────────────
             expl.   exploración

    Un nodo no visitado (n=0) devuelve infinito para garantizar que
    todo nodo se visite al menos una vez.
    """
    if nodo["n"] == 0:
        return float("inf")
    padre_n = nodo["padre"]["n"] if nodo["padre"] else 1
    return (nodo["w"] / nodo["n"]
            + c * math.sqrt(math.log(max(padre_n, 1)) / nodo["n"]))


def seleccionar(raiz: dict) -> dict:
    """
    Paso 1 — Selección (UCT):
    Recorre el árbol eligiendo el hijo con mayor UCT hasta llegar
    a un nodo que todavía puede expandirse.
    """
    nodo = raiz
    while nodo["hijos"] and len(nodo["hijos"]) == len(nodo.get("_acciones", [])):
        nodo = max(nodo["hijos"], key=ucb1)
    return nodo


def _acciones_no_exploradas(nodo: dict, n_acciones: int) -> list[int]:
    indices_explorados = {h["accion_idx"] for h in nodo["hijos"]}
    return [i for i in range(n_acciones) if i not in indices_explorados]


def expandir(nodo: dict, n_acciones: int, rng: np.random.Generator) -> dict:
    """
    Paso 2 — Expansión:
    Añade un hijo nuevo para una acción no explorada (elegida al azar).
    """
    no_exploradas = _acciones_no_exploradas(nodo, n_acciones)
    accion_idx = int(rng.choice(no_exploradas))
    hijo = crear_nodo(accion_idx=accion_idx, padre=nodo)
    nodo["hijos"].append(hijo)
    return hijo


def rollout(
    estado: dict,
    accion_idx: int,
    prices_ext: np.ndarray,
    tickers: list[str],
    rng: np.random.Generator,
) -> float:
    """
    Paso 3 — Simulación (Rollout) con GBM multivariado.

    Aplica la acción candidata al estado actual, luego simula N_PATHS
    trayectorias GBM de DIAS_ROLLOUT días para los N activos + SPY.

    Recompensa = E[log(1 + ret_cartera)] − E[log(1 + ret_SPY)]
                 ─────────────────────────────────────────────
                 Positivo → cartera supera al índice (genera alfa).
                 Negativo → cartera queda por debajo del índice.

    Usar el mismo Z para cartera y SPY garantiza que se comparan
    bajo exactamente la misma trayectoria de mercado simulada.
    Esta es la misma filosofía de recompensa que mcts_simple.py.
    """
    N = len(tickers)
    pesos_nuevo = aplicar_accion(estado["pesos"], accion_idx, tickers)

    mu, Sigma = calibrar_gbm_multi(prices_ext, estado["dia"])
    mu_ann    = mu * 252
    sigma_ann = Sigma * 252
    L         = cholesky_safe(sigma_ann)               # (N+1, N+1)

    T   = DIAS_ROLLOUT / 252.0
    Z   = rng.standard_normal((N + 1, N_PATHS))        # (N+1, N_PATHS)
    eps = L @ Z * math.sqrt(T)                         # (N+1, N_PATHS)

    drift       = (mu_ann - 0.5 * np.diag(sigma_ann)) * T   # (N+1,)
    log_ret_mat = drift[:, None] + eps                       # (N+1, N_PATHS)
    asset_ret   = np.exp(log_ret_mat) - 1.0                  # (N+1, N_PATHS)

    port_ret = pesos_nuevo @ asset_ret[:N]   # (N_PATHS,) — retorno cartera
    spy_ret  = asset_ret[N]                  # (N_PATHS,) — retorno SPY

    # Log-retorno relativo: alfa esperado de esta acción
    recompensa = float(
        np.mean(np.log(1.0 + np.maximum(port_ret, -0.999)))
        - np.mean(np.log(1.0 + np.maximum(spy_ret, -0.999)))
    )
    return recompensa


def retropropagar(nodo: dict, recompensa: float) -> None:
    """
    Paso 4 — Retropropagación:
    Actualiza w y n desde el nodo hoja hasta la raíz.
    """
    actual = nodo
    while actual is not None:
        actual["n"] += 1
        actual["w"] += recompensa
        actual = actual["padre"]


# =============================================================================
# SECCIÓN 6 — BUCLE MCTS
# =============================================================================

def ejecutar_mcts(
    estado: dict,
    prices_ext: np.ndarray,
    tickers: list[str],
    rng: np.random.Generator,
    registrar_convergencia: bool = False,
) -> tuple[int, dict]:
    """
    Ejecuta ITERACIONES ciclos de MCTS y devuelve la mejor acción.

    Parámetros
    ----------
    estado                : Estado actual de la cartera.
    prices_ext            : Precios (T, N+1) — activos + SPY.
    tickers               : Lista de símbolos de los activos.
    rng                   : Generador de números aleatorios.
    registrar_convergencia: Si True, guarda el historial de Q(a)/N(a) por
                            iteración para el gráfico de convergencia UCB.

    Retorna
    -------
    (mejor_accion_idx, historial_convergencia)
    """
    acciones = construir_acciones(tickers)
    n_acciones = len(acciones)
    raiz = crear_nodo(accion_idx=None, padre=None)

    # historial[accion_idx] = lista de Q(a)/N(a) por iteración
    historial: dict[int, list[float]] = {i: [] for i in range(n_acciones)}

    for it in range(ITERACIONES):
        # — Paso 1: Selección
        hoja = raiz
        while hoja["hijos"] and not _acciones_no_exploradas(hoja, n_acciones):
            hoja = max(hoja["hijos"], key=ucb1)

        # — Paso 2: Expansión
        if _acciones_no_exploradas(hoja, n_acciones):
            hoja = expandir(hoja, n_acciones, rng)

        # — Paso 3: Rollout
        accion_idx_rollout = (hoja["accion_idx"]
                              if hoja["accion_idx"] is not None
                              else int(rng.integers(n_acciones)))
        recompensa = rollout(estado, accion_idx_rollout, prices_ext, tickers, rng)

        # — Paso 4: Retropropagación
        retropropagar(hoja, recompensa)

        # Registrar convergencia UCB
        if registrar_convergencia:
            for hijo in raiz["hijos"]:
                idx = hijo["accion_idx"]
                q   = hijo["w"] / hijo["n"] if hijo["n"] > 0 else 0.0
                historial[idx].append(q)

    # Elegir la acción con mayor Q (c=0: explotación pura, sin exploración)
    mejor_hijo = max(raiz["hijos"], key=lambda h: h["w"] / h["n"] if h["n"] > 0 else -math.inf)
    return mejor_hijo["accion_idx"], historial


# =============================================================================
# SECCIÓN 7 — BACKTEST
# =============================================================================

def backtest(
    prices: np.ndarray,
    spy: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> dict:
    """
    Ejecuta el backtest completo del agente MCTS.

    Mecánica del rebalanceo semanal
    --------------------------------
    · Cada REBAL_FREQ días, MCTS elige la mejor acción y se aplica el nuevo
      peso objetivo (cobrando costes de transacción por turnover).
    · Entre rebalanceos, los pesos derivan con el mercado (drift natural):
          w_raw = w * (1 + ret)
          w     = w_raw / sum(w_raw)
      Esto es correcto: si AAPL sube mucho, su peso real aumenta aunque
      no hayamos rebalanceado.

    Benchmarks
    ----------
    · SPY B&H:   comprar SPY el primer día y mantenerlo.
    · EW B&H:    comprar igual cantidad de cada activo el primer día y
                 no rebalancear (Buy & Hold Equal-Weight).
    """
    T, N = prices.shape
    rng  = np.random.default_rng(SEMILLA)

    # prices_ext: columnas [activos | SPY]
    prices_ext = np.hstack([prices, spy[:, None]])  # (T, N+1)

    # Estado inicial: pesos iguales
    pesos = np.ones(N) / N
    valor = CAPITAL_INIT

    valores:         list[float]       = [CAPITAL_INIT]
    pesos_historia:  list[np.ndarray]  = [pesos.copy()]
    acciones_hist:   list[tuple]       = []
    historial_conv:  dict | None       = None
    dia_conv        = T // 2           # día central para graficar convergencia

    # — Benchmarks
    spy_shares   = CAPITAL_INIT / spy[0]
    ew_shares    = (CAPITAL_INIT / N) / prices[0]       # (N,)

    spy_valores: list[float] = [CAPITAL_INIT]
    ew_valores:  list[float] = [CAPITAL_INIT]

    print(f"\n[MCTS] Iniciando backtest | {T-1} sesiones | "
          f"rebalanceo cada {REBAL_FREQ} días | {ITERACIONES} iter/decisión")
    print(f"       Activos: {tickers} | Referencia: {SPY_TICK}\n")

    for dia in range(T - 1):

        # — Decisión MCTS (solo en días de rebalanceo)
        if dia % REBAL_FREQ == 0:
            estado = crear_estado(dia, pesos, valor, prices, spy)
            registrar = (dia == dia_conv)
            accion_idx, conv = ejecutar_mcts(
                estado, prices_ext, tickers, rng,
                registrar_convergencia=registrar,
            )
            if registrar:
                historial_conv = conv

            acciones = construir_acciones(tickers)
            pesos_nuevo = aplicar_accion(pesos, accion_idx, tickers)

            # Coste de transacción proporcional al turnover
            turnover = float(np.sum(np.abs(pesos_nuevo - pesos)))
            valor   *= (1.0 - turnover * TX_COST)
            pesos    = pesos_nuevo

            acciones_hist.append((dia, accion_idx, acciones[accion_idx]))

        # — Retornos reales del mercado (día siguiente)
        ret = (prices[dia + 1] - prices[dia]) / prices[dia]   # (N,)

        # Actualizar valor de la cartera
        valor *= float(np.dot(pesos, 1.0 + ret))

        # Drift natural de los pesos (sin rebalancear)
        pesos_raw = pesos * (1.0 + ret)
        pesos     = pesos_raw / pesos_raw.sum()

        valores.append(valor)
        pesos_historia.append(pesos.copy())

        # Benchmarks
        spy_valores.append(spy_shares * spy[dia + 1])
        ew_valores.append(float(np.sum(ew_shares * prices[dia + 1])))

        # Progreso
        pct = (dia + 1) / (T - 1) * 100
        print(
            f"\r[MCTS] {dia+1}/{T-1} ({pct:.0f}%)  "
            f"Cartera: {valor:>10,.0f}$  "
            f"SPY: {spy_valores[-1]:>10,.0f}$  "
            f"Pesos: [{' '.join(f'{w:.2f}' for w in pesos)}]",
            end="", flush=True,
        )

    print()
    return {
        "valores":        valores,
        "spy_valores":    spy_valores,
        "ew_valores":     ew_valores,
        "pesos_historia": [p for p in pesos_historia],
        "acciones_hist":  acciones_hist,
        "historial_conv": historial_conv,
        "tickers":        tickers,
        "dates":          dates,
    }


# =============================================================================
# SECCIÓN 8 — MÉTRICAS FINANCIERAS
# =============================================================================

RISK_FREE_ANNUAL = 0.045   # Tasa libre de riesgo (~4.5% anual)


def calcular_metricas(valores: list[float], etiqueta: str = "") -> dict:
    """
    Calcula métricas de rendimiento y riesgo sobre la serie de valores.

    ROI          : Retorno total acumulado.
    Retorno anual: Tasa anual equivalente.
    Sharpe       : Retorno medio ajustado por volatilidad (anualizado).
    Sortino      : Como Sharpe, pero solo penaliza la volatilidad negativa.
    Max Drawdown : Caída máxima desde el pico histórico.
    Calmar       : Retorno anual / |Max Drawdown|.
    """
    arr = np.array(valores)
    ret = np.diff(arr) / arr[:-1]
    rf_d = RISK_FREE_ANNUAL / 252

    mean_r = float(np.mean(ret))
    std_r  = float(np.std(ret, ddof=1)) + 1e-10

    sharpe = (mean_r - rf_d) / std_r * math.sqrt(252)

    neg_ret  = ret[ret < rf_d]
    down_std = float(np.std(neg_ret, ddof=1)) + 1e-10 if len(neg_ret) > 1 else std_r
    sortino  = (mean_r - rf_d) / down_std * math.sqrt(252)

    peak   = np.maximum.accumulate(arr)
    dd     = (arr - peak) / peak
    max_dd = float(np.min(dd))

    n_days     = len(arr)
    annual_ret = (arr[-1] / arr[0]) ** (252 / n_days) - 1
    calmar     = annual_ret / abs(max_dd) if max_dd != 0 else float("inf")

    return {
        "etiqueta":   etiqueta,
        "roi":        float((arr[-1] - arr[0]) / arr[0]),
        "annual_ret": float(annual_ret),
        "sharpe":     sharpe,
        "sortino":    sortino,
        "max_dd":     max_dd,
        "calmar":     calmar,
        "final":      float(arr[-1]),
    }


def imprimir_metricas(m_mcts: dict, m_spy: dict, m_ew: dict) -> None:
    """Imprime tabla comparativa de métricas."""
    print("\n" + "=" * 65)
    print(f"{'Métrica':<22} {'MCTS':>12} {'SPY B&H':>12} {'EW B&H':>12}")
    print("=" * 65)
    filas = [
        ("Valor final ($)",   "final",      "{:>12,.2f}"),
        ("ROI acumulado",     "roi",        "{:>+12.2%}"),
        ("Retorno anual",     "annual_ret", "{:>+12.2%}"),
        ("Sharpe",            "sharpe",     "{:>12.4f}"),
        ("Sortino",           "sortino",    "{:>12.4f}"),
        ("Max Drawdown",      "max_dd",     "{:>+12.2%}"),
        ("Calmar",            "calmar",     "{:>12.4f}"),
    ]
    for label, key, fmt in filas:
        vals = [m_mcts[key], m_spy[key], m_ew[key]]
        print(f"  {label:<20} " + "  ".join(fmt.format(v) for v in vals))
    print("=" * 65 + "\n")


# =============================================================================
# SECCIÓN 9 — VISUALIZACIONES
# =============================================================================

COLORES = plt.cm.tab10(np.linspace(0, 0.9, 10))


def graficar(resultados: dict, output_dir: str = OUTPUT_DIR) -> None:
    """
    Genera dos figuras:

    Figura 1 (4 paneles): Portafolio normalizado | Pesos en el tiempo |
                          Drawdown comparativo    | Alfa acumulado vs SPY

    Figura 2: Convergencia UCB — cómo aprende el árbol entre las 7 acciones.
    """
    os.makedirs(output_dir, exist_ok=True)

    valores  = np.array(resultados["valores"])
    spy_v    = np.array(resultados["spy_valores"])
    ew_v     = np.array(resultados["ew_valores"])
    pesos_h  = np.array(resultados["pesos_historia"])   # (T, N)
    dates    = resultados["dates"]
    tickers  = resultados["tickers"]
    N        = len(tickers)
    acciones = construir_acciones(tickers)

    base    = valores[0]
    v_n     = valores / base * 100
    spy_n   = spy_v   / base * 100
    ew_n    = ew_v    / base * 100

    # ── Figura 1: 4 paneles ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("MCTS Portfolio Relativo — Backtest",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 2.5, 1.5, 1.5], hspace=0.45)

    # Panel 1: Portafolio normalizado base 100
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, v_n,   lw=2.0, color="#1f77b4", label="MCTS",      zorder=3)
    ax1.plot(dates, spy_n, lw=1.5, color="#ff7f0e", ls="--", label="SPY B&H")
    ax1.plot(dates, ew_n,  lw=1.5, color="#2ca02c", ls=":",  label="EW B&H")
    ax1.axhline(100, color="gray", lw=0.8)
    ax1.set_title("Evolución de la Cartera (base 100)", fontsize=11,
                  fontweight="bold")
    ax1.set_ylabel("Valor (base 100)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Pesos MCTS en el tiempo (área apilada)
    ax2 = fig.add_subplot(gs[1])
    bottom = np.zeros(len(dates))
    for i, ticker in enumerate(tickers):
        ax2.fill_between(dates, bottom, bottom + pesos_h[:, i],
                         color=COLORES[i], alpha=0.85, label=ticker)
        bottom += pesos_h[:, i]

    # Marcadores verticales en días de rebalanceo
    for dia, _, nombre in resultados["acciones_hist"]:
        if dia < len(dates):
            ax2.axvline(dates[dia], color="white", lw=0.4, alpha=0.5)

    ax2.set_title("Distribución de Pesos por Activo", fontsize=11,
                  fontweight="bold")
    ax2.set_ylabel("Peso")
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=8, ncol=N, loc="upper right",
               bbox_to_anchor=(1.0, 1.18))
    ax2.grid(True, alpha=0.2)

    # Panel 3: Drawdown
    ax3 = fig.add_subplot(gs[2])
    peak_mcts = np.maximum.accumulate(valores)
    dd_mcts   = (valores - peak_mcts) / peak_mcts * 100
    ax3.fill_between(dates, dd_mcts, 0, color="#d62728", alpha=0.55,
                     label="MCTS DD")
    peak_spy = np.maximum.accumulate(spy_v)
    dd_spy   = (spy_v - peak_spy) / peak_spy * 100
    ax3.plot(dates, dd_spy, lw=1.0, color="#ff7f0e", ls="--",
             label="SPY DD", alpha=0.8)
    ax3.set_title("Drawdown (%)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("DD (%)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Alfa acumulado (MCTS / SPY - 1)
    ax4 = fig.add_subplot(gs[3])
    alfa = valores / spy_v - 1.0
    ax4.plot(dates, alfa * 100, lw=1.5, color="#9467bd")
    ax4.fill_between(dates, alfa * 100, 0,
                     where=(alfa >= 0), color="#2ca02c", alpha=0.35,
                     label="Alfa positivo")
    ax4.fill_between(dates, alfa * 100, 0,
                     where=(alfa < 0), color="#d62728", alpha=0.35,
                     label="Alfa negativo")
    ax4.axhline(0, color="gray", lw=0.8)
    ax4.set_title("Alfa Acumulado vs SPY (%)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("(MCTS/SPY − 1) %")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ruta1 = os.path.join(output_dir, "backtest_portfolio.png")
    plt.savefig(ruta1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico guardado → {ruta1}")

    # ── Figura 2: Convergencia UCB ───────────────────────────────────────────
    historial = resultados.get("historial_conv")
    if historial:
        fig2, ax = plt.subplots(figsize=(10, 5))
        for idx, vals in historial.items():
            if vals:
                ax.plot(vals, label=acciones[idx], lw=1.4,
                        color=COLORES[idx % len(COLORES)])
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title("Convergencia UCB — Q(a)/N(a) por iteración",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteración MCTS")
        ax.set_ylabel("Recompensa media (alfa vs SPY)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        ruta2 = os.path.join(output_dir, "convergencia_mcts.png")
        plt.savefig(ruta2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Convergencia UCB → {ruta2}")


def guardar_reporte(
    m_mcts: dict, m_spy: dict, m_ew: dict,
    resultados: dict, output_dir: str = OUTPUT_DIR,
) -> None:
    """Guarda un reporte markdown con configuración y resultados."""
    os.makedirs(output_dir, exist_ok=True)
    ruta = os.path.join(output_dir, "backtest_report.md")

    acciones_resumen: dict[str, int] = {}
    for _, _, nombre in resultados["acciones_hist"]:
        acciones_resumen[nombre] = acciones_resumen.get(nombre, 0) + 1

    lineas = [
        "# Resultados del Backtest — MCTS Portfolio Relativo",
        "",
        f"> Generado el {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Configuración",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| Activos | `{TICKERS}` |",
        f"| Benchmark | `{SPY_TICK}` |",
        f"| Período | `{PERIOD}` |",
        f"| Capital inicial | `${CAPITAL_INIT:,.0f}` |",
        f"| Iteraciones MCTS | `{ITERACIONES}` |",
        f"| Días rollout | `{DIAS_ROLLOUT}` |",
        f"| Ventana calibración | `{VENTANA_CALIB}` |",
        f"| Trayectorias GBM | `{N_PATHS}` |",
        f"| Tilt por rotación | `{TILT:.0%}` |",
        f"| Coste transacción | `{TX_COST:.1%}` |",
        f"| Frecuencia rebalanceo | `cada {REBAL_FREQ} días` |",
        "",
        "## Resultados financieros",
        "",
        "| Métrica | MCTS | SPY B&H | EW B&H |",
        "|---|---|---|---|",
        f"| Valor final ($) | {m_mcts['final']:,.2f} | {m_spy['final']:,.2f} | {m_ew['final']:,.2f} |",
        f"| ROI acumulado | {m_mcts['roi']:+.2%} | {m_spy['roi']:+.2%} | {m_ew['roi']:+.2%} |",
        f"| Retorno anual | {m_mcts['annual_ret']:+.2%} | {m_spy['annual_ret']:+.2%} | {m_ew['annual_ret']:+.2%} |",
        f"| Sharpe | {m_mcts['sharpe']:.4f} | {m_spy['sharpe']:.4f} | {m_ew['sharpe']:.4f} |",
        f"| Sortino | {m_mcts['sortino']:.4f} | {m_spy['sortino']:.4f} | {m_ew['sortino']:.4f} |",
        f"| Max Drawdown | {m_mcts['max_dd']:+.2%} | {m_spy['max_dd']:+.2%} | {m_ew['max_dd']:+.2%} |",
        f"| Calmar | {m_mcts['calmar']:.4f} | {m_spy['calmar']:.4f} | {m_ew['calmar']:.4f} |",
        "",
        "## Distribución de acciones tomadas",
        "",
        "| Acción | Veces |",
        "|---|---|",
    ]
    for nombre, cnt in sorted(acciones_resumen.items(), key=lambda x: -x[1]):
        lineas.append(f"| {nombre} | {cnt} |")

    lineas += [
        "",
        "## Notas metodológicas",
        "",
        "- **Espacio de acciones**: 7 acciones semánticas (ROTAR_HACIA_i × N + IGUAL_PESO + MANTENER).",
        "- **Recompensa relativa**: `E[log(1+ret_cartera)] − E[log(1+ret_SPY)]` → aprende a superar al índice.",
        "- **GBM multivariado**: calibración rolling con Cholesky, SPY incluido como activo N+1.",
        "- **Rebalanceo semanal**: cada 5 días para reducir costes de transacción.",
        "- **Drift de pesos**: entre rebalanceos, los pesos evolucionan con el mercado (sin coste).",
        "",
        "![Backtest](backtest_portfolio.png)",
        "![Convergencia UCB](convergencia_mcts.png)",
    ]

    with open(ruta, "w") as f:
        f.write("\n".join(lineas))
    print(f"[INFO] Reporte guardado → {ruta}")


# =============================================================================
# SECCIÓN 10 — PUNTO DE ENTRADA
# =============================================================================

def main() -> None:
    prices, spy, dates, tickers = descargar_precios(TICKERS, SPY_TICK, PERIOD)

    resultados = backtest(prices, spy, dates, tickers)

    valores_mcts = resultados["valores"]
    spy_valores  = resultados["spy_valores"]
    ew_valores   = resultados["ew_valores"]

    m_mcts = calcular_metricas(valores_mcts, "MCTS")
    m_spy  = calcular_metricas(spy_valores,  "SPY B&H")
    m_ew   = calcular_metricas(ew_valores,   "EW B&H")

    imprimir_metricas(m_mcts, m_spy, m_ew)

    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    graficar(resultados, output_dir)
    guardar_reporte(m_mcts, m_spy, m_ew, resultados, output_dir)

    print("[OK] Backtest completado.")


if __name__ == "__main__":
    main()
