"""
mcts_momentum_filtrado.py
=========================
Estrategia de cartera en dos etapas:

  ETAPA 1 — Filtro mensual por momentum ajustado por volatilidad
  ---------------------------------------------------------------
  De un universo amplio de ~20 ETFs (tecnología, metales raros,
  emergentes y sectores complementarios), selecciona los TOP_N
  activos con mayor ratio  retorno_acumulado / volatilidad  en los
  últimos MOM_LOOKBACK días.  Esto garantiza que el MCTS solo
  trabaje con activos que ya están en tendencia alcista sólida.

  ETAPA 2 — MCTS con recompensa relativa (reutiliza la lógica existente)
  -----------------------------------------------------------------------
  El MCTS de mcts_portfolio_relativo.py rota semanalmente entre los
  TOP_N activos seleccionados, buscando superar al SPY.  Su sesgo
  defensivo ahora se aplica sobre un universo pre-filtrado, así que
  incluso las acciones conservadoras (MANTENER, IGUAL_PESO) mantienen
  exposición a activos con momentum positivo.

  Cada mes se renueva el universo: los activos que pierden momentum
  salen y entran los nuevos líderes.  El MCTS re-arranca con pesos
  iguales sobre el nuevo universo.

Universo de ETFs
────────────────
  Tecnología:      QQQ  XLK  SOXX  SMH  IGV
  Metales raros:   REMX  LIT  COPX  SLV  PICK
  Emergentes:      EEM  INDA  EWZ  VWO  KWEB
  Complementarios: XLE  XLV  XBI  GDX

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
# =============================================================================

# ── Universo amplio con categorías (para diversificación forzada) ──
UNIVERSO_CATEGORIAS = {
    "tech": [
        "QQQ",   # Nasdaq 100
        "XLK",   # Technology Select SPDR
        "SOXX",  # Semiconductors
        "SMH",   # VanEck Semiconductor
        "IGV",   # iShares Expanded Tech-Software
    ],
    "metales": [
        "REMX",  # VanEck Rare Earth/Strategic Metals
        "LIT",   # Global X Lithium & Battery Tech
        "COPX",  # Global X Copper Miners
        "SLV",   # iShares Silver Trust
        "PICK",  # iShares MSCI Global Metals & Mining
    ],
    "emergentes": [
        "EEM",   # iShares MSCI Emerging Markets
        "INDA",  # iShares MSCI India
        "EWZ",   # iShares MSCI Brazil
        "VWO",   # Vanguard FTSE Emerging Markets
        "KWEB",  # KraneShares CSI China Internet
    ],
    "complementarios": [
        "XLE",   # Energy Select SPDR
        "XLV",   # Health Care Select SPDR
        "XBI",   # SPDR S&P Biotech
        "GDX",   # VanEck Gold Miners
    ],
}

# Lista plana para descarga
UNIVERSO = [t for cat in UNIVERSO_CATEGORIAS.values() for t in cat]

# Mapa inverso: ticker → categoría
TICKER_CAT = {t: cat for cat, tickers in UNIVERSO_CATEGORIAS.items()
              for t in tickers}

SPY_TICK      = "SPY"
PERIOD        = "3y"       # 3 años: 1 año warm-up + 2 años backtest
CAPITAL_INIT  = 10_000.0

# ── Filtro momentum (Etapa 1) ──
TOP_N         = 5          # Activos seleccionados cada mes
MAX_POR_CAT   = 5          # Máximo ETFs de la misma categoría (sin límite efectivo)
CORR_UMBRAL   = 0.95       # Correlación máxima permitida entre seleccionados
MOM_LOOKBACK  = 63         # Ventana momentum (~3 meses de trading)
MOM_SKIP      = 5          # Skip reciente: excluir últimos 5 días (1 semana)
RESELECT_FREQ = 21         # Renovar universo cada ~21 días (mensual)
INCLUIR_CASH  = False      # El stop-loss gestiona la defensa, MCTS solo rota

# ── Stop-loss dinámico (desactivado) ──
STOPLOSS_DD   = -0.99      # Umbral inalcanzable → stop-loss desactivado
STOPLOSS_CASH = 0.50       # Fracción forzada a cash en stop-loss
STOPLOSS_REENTRY_DD = -0.15  # Reentrar cuando DD mejore a -15%

# ── MCTS (Etapa 2) ──
ITERACIONES   = 5000        # 5000 iter para 7 acciones (~710 visitas/acción)
DIAS_ROLLOUT  = 20
VENTANA_CALIB = 60
N_PATHS       = 50          # Más trayectorias → rollouts más estables, menos ruido por decisión
C_UCB         = math.sqrt(2)

TILT          = 0.20
TX_COST       = 0.001
REBAL_FREQ    = 5          # Rebalanceo MCTS cada 5 días (semanal)

SEMILLA       = 42
OUTPUT_DIR    = "results_momentum"


# =============================================================================
# SECCIÓN 1 — DESCARGA DE DATOS
# =============================================================================

def descargar_universo(
    universo: list[str],
    spy_tick: str,
    periodo: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Descarga precios de cierre ajustados de todo el universo + SPY.

    Retorna
    -------
    df_precios : DataFrame (T, U) con precios de cada ETF del universo.
    spy_series : Series (T,) con precios de SPY.
    """
    todos = list(set(universo + [spy_tick]))
    print(f"[INFO] Descargando {len(todos)} tickers ({periodo})...")

    df = yf.download(todos, period=periodo, progress=False, auto_adjust=True)["Close"]
    df = df.dropna(how="all").ffill().dropna()

    # Eliminar tickers con demasiados NaN (ETFs muy nuevos)
    umbral_nan = 0.10
    tickers_validos = [t for t in universo if t in df.columns
                       and df[t].isna().mean() < umbral_nan]
    tickers_descartados = [t for t in universo if t not in tickers_validos]

    if tickers_descartados:
        print(f"[WARN] Descartados por datos insuficientes: {tickers_descartados}")

    df = df[[spy_tick] + tickers_validos].dropna()

    print(f"[INFO] {len(df)} sesiones  |  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"[INFO] Universo válido: {len(tickers_validos)} ETFs")

    spy_series = df[spy_tick]
    df_precios = df[tickers_validos]

    return df_precios, spy_series


# =============================================================================
# SECCIÓN 2 — ETAPA 1: FILTRO MOMENTUM AJUSTADO POR VOLATILIDAD
# =============================================================================

def calcular_mom_vol_adj(precios: np.ndarray, ventana: int,
                         skip: int = MOM_SKIP,
                         spy_precios: np.ndarray | None = None) -> np.ndarray:
    """
    Calcula el momentum ajustado por volatilidad con skip-month.

        mom_vol_adj = (retorno_activo - retorno_SPY)(t-ventana, t-skip) / vol_anualizada

    Cuando se proporciona spy_precios, el retorno es RELATIVO a SPY: selecciona
    activos que han superado al benchmark en la ventana de lookback, no los que
    simplemente han subido más en términos absolutos.
    Sin spy_precios, usa retorno absoluto (comportamiento anterior).

    El skip-month excluye los últimos `skip` días del cálculo de retorno.
    La volatilidad se calcula sobre toda la ventana para mayor estabilidad.

    Parámetros
    ----------
    precios    : (T, N) — precios históricos de los ETFs.
    ventana    : Número de días totales de lookback.
    skip       : Días recientes a excluir del retorno (skip-month).
    spy_precios: (T,) — precios de SPY alineados con precios (opcional).

    Retorna
    -------
    scores : (N,) — ratio (momentum relativo a SPY) / volatilidad.
    """
    N = precios.shape[1]
    scores = np.full(N, np.nan)

    if len(precios) < ventana + 1:
        return scores

    tramo = precios[-(ventana + 1):]
    log_ret = np.diff(np.log(tramo), axis=0)  # (ventana, N)

    # Retorno acumulado EXCLUYENDO los últimos `skip` días
    if skip > 0 and len(tramo) > skip + 1:
        ret_acum = np.log(tramo[-(skip + 1)] / tramo[0])  # (N,)
    else:
        ret_acum = np.log(tramo[-1] / tramo[0])

    # Restar retorno SPY para obtener alpha relativo al benchmark
    if spy_precios is not None and len(spy_precios) >= ventana + 1:
        spy_tramo = spy_precios[-(ventana + 1):]
        if skip > 0 and len(spy_tramo) > skip + 1:
            ret_spy = float(np.log(spy_tramo[-(skip + 1)] / spy_tramo[0]))
        else:
            ret_spy = float(np.log(spy_tramo[-1] / spy_tramo[0]))
        ret_acum = ret_acum - ret_spy  # exceso de retorno vs SPY

    # Volatilidad anualizada (sobre toda la ventana, más estable)
    vol = np.std(log_ret, axis=0, ddof=1) * math.sqrt(252)  # (N,)

    mask = vol > 1e-8
    scores[mask] = ret_acum[mask] / vol[mask]

    return scores


def seleccionar_top_n(
    df_precios: pd.DataFrame,
    dia_idx: int,
    top_n: int = TOP_N,
    lookback: int = MOM_LOOKBACK,
    spy_series: pd.Series | None = None,
) -> list[str]:
    """
    Selecciona los TOP_N ETFs con mayor momentum ajustado por volatilidad
    RELATIVO A SPY, aplicando tres restricciones de diversificación:

      1. Máximo MAX_POR_CAT ETFs de la misma categoría (tech, metales, etc.)
      2. Correlación < CORR_UMBRAL con los ETFs ya seleccionados.
      3. Solo selecciona ETFs con momentum relativo positivo (score > 0),
         es decir, ETFs que han superado a SPY en la ventana de lookback.

    Algoritmo greedy: recorre candidatos por score descendente y añade
    al conjunto solo si pasa las tres restricciones.
    """
    inicio = max(0, dia_idx - lookback)
    tramo = df_precios.iloc[inicio:dia_idx + 1].to_numpy()
    tickers = list(df_precios.columns)

    spy_arr = spy_series.iloc[inicio:dia_idx + 1].to_numpy() if spy_series is not None else None
    scores = calcular_mom_vol_adj(tramo, lookback, spy_precios=spy_arr)

    # Calcular matriz de correlación sobre log-retornos recientes
    log_rets = np.diff(np.log(tramo), axis=0) if len(tramo) > 1 else np.zeros((1, len(tickers)))
    corr_mat = np.corrcoef(log_rets.T) if log_rets.shape[0] > 2 else np.eye(len(tickers))

    # Ordenar por score descendente (solo válidos y positivos)
    indices_validos = np.where((~np.isnan(scores)) & (scores > 0))[0]
    ranking = indices_validos[np.argsort(scores[indices_validos])[::-1]]

    seleccion: list[int] = []
    cat_conteo: dict[str, int] = {}

    for idx in ranking:
        if len(seleccion) >= top_n:
            break

        ticker = tickers[idx]
        cat = TICKER_CAT.get(ticker, "otro")

        # Restricción 1: máximo por categoría
        if cat_conteo.get(cat, 0) >= MAX_POR_CAT:
            continue

        # Restricción 2: correlación con los ya seleccionados
        demasiado_correlado = False
        for sel_idx in seleccion:
            if abs(corr_mat[idx, sel_idx]) > CORR_UMBRAL:
                demasiado_correlado = True
                break
        if demasiado_correlado:
            continue

        seleccion.append(idx)
        cat_conteo[cat] = cat_conteo.get(cat, 0) + 1

    # Fallback 1: relajar restricciones pero mantener momentum positivo
    if len(seleccion) < top_n:
        for idx in ranking:
            if idx not in seleccion and len(seleccion) < top_n:
                seleccion.append(idx)

    # Fallback 2: si ninguno tiene momentum positivo, tomar los menos malos
    if len(seleccion) < 2:
        todos_validos = np.where(~np.isnan(scores))[0]
        ranking_all = todos_validos[np.argsort(scores[todos_validos])[::-1]]
        for idx in ranking_all:
            if idx not in seleccion and len(seleccion) < top_n:
                seleccion.append(idx)

    return [tickers[i] for i in seleccion]


# =============================================================================
# SECCIÓN 3 — ESPACIO DE ACCIONES MCTS (idéntico a mcts_portfolio_relativo)
# =============================================================================

def construir_acciones(tickers: list[str]) -> list[str]:
    """
    Acciones disponibles para el MCTS:
      ROTAR_HACIA_i : Tiltar hacia el activo i
      IGUAL_PESO    : Rebalancear a 1/N
      MANTENER      : No operar
      IR_A_CASH     : Mover TILT a cash (tasa libre de riesgo)
    """
    acciones = [f"ROTAR_HACIA_{t}" for t in tickers] + ["IGUAL_PESO", "MANTENER"]
    if INCLUIR_CASH:
        acciones.append("IR_A_CASH")
    return acciones


def aplicar_accion(pesos: np.ndarray, accion_idx: int,
                   tickers: list[str], tilt: float = TILT) -> np.ndarray:
    """
    Aplica una acción. Los pesos suman <= 1.0.
    Si suman < 1.0, la diferencia (1 - sum) es la fracción en cash.
    """
    N = len(tickers)

    if accion_idx == N + 1:          # MANTENER
        return pesos.copy()

    if accion_idx == N:              # IGUAL_PESO
        return np.ones(N) / N

    # IR_A_CASH (solo si INCLUIR_CASH)
    if INCLUIR_CASH and accion_idx == N + 2:
        new_w = pesos.copy()
        fraccion_invertida = new_w.sum()
        reduccion = min(tilt, fraccion_invertida - 0.05)  # mínimo 5% invertido
        if fraccion_invertida > 1e-10 and reduccion > 0:
            factor = (fraccion_invertida - reduccion) / fraccion_invertida
            new_w *= factor
        return new_w

    # ROTAR_HACIA_i — mueve `tilt` HACIA el activo i, tomando de
    # los otros activos Y del cash proporcionalmente.
    i = accion_idx
    new_w = pesos.copy()
    fraccion_invertida = new_w.sum()
    fraccion_cash = 1.0 - fraccion_invertida

    delta = min(tilt, 1.0 - pesos[i])

    # Repartir el delta entre los otros activos y el cash
    total_fuente = sum(pesos[j] for j in range(N) if j != i) + fraccion_cash
    if total_fuente > 1e-10:
        # Reducir otros activos proporcionalmente
        for j in range(N):
            if j != i:
                new_w[j] = max(0.0, pesos[j] - delta * pesos[j] / total_fuente)
        # El cash se reduce implícitamente (1 - sum(new_w) será menor)

    new_w[i] += delta

    # Clamp: no superar 1.0 en total
    total = new_w.sum()
    if total > 1.0 + 1e-10:
        new_w /= total
    elif total < 1e-10:
        new_w = np.ones(N) / N

    return new_w


# =============================================================================
# SECCIÓN 4 — ESTADO Y SEÑALES RELATIVAS
# =============================================================================

def _ranking_normalizado(valores: np.ndarray) -> np.ndarray:
    N = len(valores)
    if N <= 1:
        return np.array([0.5])
    orden = np.argsort(valores)
    rango = np.empty(N)
    rango[orden] = np.arange(N) / (N - 1)
    return rango


def calcular_momentum(prices: np.ndarray, dia: int,
                      ventana: int = 20) -> np.ndarray:
    inicio = max(0, dia - ventana)
    if inicio >= dia:
        return np.ones(prices.shape[1]) * 0.5
    log_ret = np.log(prices[dia] / prices[inicio])
    return _ranking_normalizado(log_ret)


def calcular_vol_relativa(prices: np.ndarray, dia: int,
                          ventana: int = 20) -> np.ndarray:
    inicio = max(0, dia - ventana)
    if dia - inicio < 2:
        return np.ones(prices.shape[1]) * 0.5
    log_rets = np.diff(np.log(prices[inicio:dia + 1]), axis=0)
    vols = np.std(log_rets, axis=0, ddof=1)
    return _ranking_normalizado(vols)


def calcular_ret_vs_spy(prices: np.ndarray, spy: np.ndarray,
                        dia: int, ventana: int = 20) -> np.ndarray:
    inicio = max(0, dia - ventana)
    if inicio >= dia:
        return np.zeros(prices.shape[1])
    log_ret_activos = np.log(prices[dia] / prices[inicio])
    log_ret_spy     = math.log(spy[dia] / spy[inicio])
    return log_ret_activos - log_ret_spy


def crear_estado(dia: int, pesos: np.ndarray, valor: float,
                 prices: np.ndarray, spy: np.ndarray) -> dict:
    return {
        "dia":        dia,
        "pesos":      pesos.copy(),
        "valor":      valor,
        "momentum":   calcular_momentum(prices, dia),
        "vol_rel":    calcular_vol_relativa(prices, dia),
        "ret_vs_spy": calcular_ret_vs_spy(prices, spy, dia),
    }


# =============================================================================
# SECCIÓN 5 — CALIBRACIÓN GBM MULTIVARIADO
# =============================================================================

def calibrar_gbm_multi(prices_ext: np.ndarray, dia: int) -> tuple:
    M = prices_ext.shape[1]
    inicio = max(0, dia - VENTANA_CALIB)
    ventana = prices_ext[inicio:dia + 1]

    if len(ventana) >= 3 and M >= 2:
        log_ret = np.diff(np.log(ventana), axis=0)
        mu      = np.mean(log_ret, axis=0)
        Sigma   = np.cov(log_ret, rowvar=False)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]])
        Sigma  += np.eye(M) * 1e-8
    else:
        mu    = np.zeros(M)
        Sigma = np.eye(M) * (0.01 ** 2)

    return mu, Sigma


def cholesky_safe(Sigma: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return np.diag(np.sqrt(np.maximum(np.diag(Sigma), 1e-10)))


# =============================================================================
# SECCIÓN 6 — LOS 4 PASOS DEL MCTS
# =============================================================================

def crear_nodo(accion_idx: int | None, padre: dict | None) -> dict:
    return {
        "accion_idx": accion_idx,
        "padre":      padre,
        "hijos":      [],
        "n":          0,
        "w":          0.0,
    }


def ucb1(nodo: dict, c: float = C_UCB) -> float:
    if nodo["n"] == 0:
        return float("inf")
    padre_n = nodo["padre"]["n"] if nodo["padre"] else 1
    return (nodo["w"] / nodo["n"]
            + c * math.sqrt(math.log(max(padre_n, 1)) / nodo["n"]))


def _acciones_no_exploradas(nodo: dict, n_acciones: int) -> list[int]:
    indices_explorados = {h["accion_idx"] for h in nodo["hijos"]}
    return [i for i in range(n_acciones) if i not in indices_explorados]


def expandir(nodo: dict, n_acciones: int, rng: np.random.Generator) -> dict:
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
    N = len(tickers)
    pesos_nuevo = aplicar_accion(estado["pesos"], accion_idx, tickers)

    mu, Sigma = calibrar_gbm_multi(prices_ext, estado["dia"])
    mu_ann    = mu * 252
    sigma_ann = Sigma * 252
    L         = cholesky_safe(sigma_ann)

    T   = DIAS_ROLLOUT / 252.0
    Z   = rng.standard_normal((N + 1, N_PATHS))
    eps = L @ Z * math.sqrt(T)

    drift       = (mu_ann - 0.5 * np.diag(sigma_ann)) * T
    log_ret_mat = drift[:, None] + eps
    asset_ret   = np.exp(log_ret_mat) - 1.0

    # Retorno de la parte invertida + fracción en cash a tasa libre
    fraccion_invertida = float(pesos_nuevo.sum())
    fraccion_cash = 1.0 - fraccion_invertida
    rf_periodo = RISK_FREE_ANNUAL * T

    port_ret = pesos_nuevo @ asset_ret[:N] + fraccion_cash * rf_periodo
    spy_ret  = asset_ret[N]

    recompensa = float(
        np.mean(np.log(1.0 + np.maximum(port_ret, -0.999)))
        - np.mean(np.log(1.0 + np.maximum(spy_ret, -0.999)))
    )
    return recompensa


def retropropagar(nodo: dict, recompensa: float) -> None:
    actual = nodo
    while actual is not None:
        actual["n"] += 1
        actual["w"] += recompensa
        actual = actual["padre"]


# =============================================================================
# SECCIÓN 7 — BUCLE MCTS
# =============================================================================

def ejecutar_mcts(
    estado: dict,
    prices_ext: np.ndarray,
    tickers: list[str],
    rng: np.random.Generator,
    registrar_convergencia: bool = False,
) -> tuple[int, dict]:
    acciones = construir_acciones(tickers)
    n_acciones = len(acciones)
    raiz = crear_nodo(accion_idx=None, padre=None)

    historial: dict[int, list[float]] = {i: [] for i in range(n_acciones)}

    for it in range(ITERACIONES):
        hoja = raiz
        while hoja["hijos"] and not _acciones_no_exploradas(hoja, n_acciones):
            hoja = max(hoja["hijos"], key=ucb1)

        if _acciones_no_exploradas(hoja, n_acciones):
            hoja = expandir(hoja, n_acciones, rng)

        accion_idx_rollout = (hoja["accion_idx"]
                              if hoja["accion_idx"] is not None
                              else int(rng.integers(n_acciones)))
        recompensa = rollout(estado, accion_idx_rollout, prices_ext, tickers, rng)

        retropropagar(hoja, recompensa)

        if registrar_convergencia:
            for hijo in raiz["hijos"]:
                idx = hijo["accion_idx"]
                q   = hijo["w"] / hijo["n"] if hijo["n"] > 0 else 0.0
                historial[idx].append(q)

    mejor_hijo = max(raiz["hijos"],
                     key=lambda h: h["w"] / h["n"] if h["n"] > 0 else -math.inf)
    return mejor_hijo["accion_idx"], historial


# =============================================================================
# SECCIÓN 8 — BACKTEST CON DOS ETAPAS
# =============================================================================

def backtest(
    df_precios: pd.DataFrame,
    spy_series: pd.Series,
) -> dict:
    """
    Backtest completo con filtro momentum mensual + MCTS semanal.

    Mecánica
    --------
    · Día 0 del backtest = MOM_LOOKBACK (necesitamos historial para el 1er filtro).
    · Cada RESELECT_FREQ días: Etapa 1 selecciona TOP_N activos.
      Al cambiar universo, se re-mapean pesos al nuevo conjunto de activos.
    · Cada REBAL_FREQ días: Etapa 2 ejecuta MCTS sobre los TOP_N activos.
    · Entre rebalanceos: drift natural de pesos con el mercado.
    """
    rng = np.random.default_rng(SEMILLA)

    all_prices = df_precios.to_numpy()   # (T_total, U)
    spy_arr    = spy_series.to_numpy()   # (T_total,)
    dates_all  = df_precios.index
    all_tickers = list(df_precios.columns)

    # El backtest empieza tras el warm-up de momentum
    inicio_bt = MOM_LOOKBACK + VENTANA_CALIB
    T_total   = len(all_prices)

    if inicio_bt >= T_total - 1:
        raise ValueError(f"No hay suficientes datos para backtest. "
                         f"Necesita >{inicio_bt} sesiones, hay {T_total}.")

    # ── Estado inicial ──
    tickers_activos = seleccionar_top_n(df_precios, inicio_bt, spy_series=spy_series)
    N_act = len(tickers_activos)
    idx_activos = [all_tickers.index(t) for t in tickers_activos]

    pesos = np.ones(N_act) / N_act
    valor = CAPITAL_INIT

    # ── Registros ──
    valores:         list[float]       = [CAPITAL_INIT]
    _p0 = dict(zip(tickers_activos, pesos))
    _p0["CASH"] = 0.0
    pesos_historia:  list[dict]        = [_p0]
    acciones_hist:   list[tuple]       = []
    seleccion_hist:  list[tuple]       = [(inicio_bt, tickers_activos[:])]
    historial_conv:  dict | None       = None

    # ── Benchmarks ──
    spy_shares   = CAPITAL_INIT / spy_arr[inicio_bt]
    # EW B&H sobre la primera selección
    ew_idx_fijos = idx_activos[:]  # índices fijos para EW B&H
    ew_shares    = (CAPITAL_INIT / N_act) / all_prices[inicio_bt, ew_idx_fijos]

    spy_valores: list[float] = [CAPITAL_INIT]
    ew_valores:  list[float] = [CAPITAL_INIT]

    dates_bt = dates_all[inicio_bt:]
    dias_desde_reselect = 0
    dias_desde_rebal    = 0
    dia_conv = (T_total - inicio_bt) // 2 + inicio_bt

    # ── Stop-loss dinámico ──
    peak_valor       = CAPITAL_INIT
    stoploss_activo  = False
    stoploss_eventos = 0

    print(f"\n[MCTS+MOM] Backtest  |  {T_total - inicio_bt - 1} sesiones")
    print(f"           Universo: {len(all_tickers)} ETFs → top {TOP_N} mensuales")
    print(f"           Reselección: cada {RESELECT_FREQ}d  |  Rebalanceo MCTS: cada {REBAL_FREQ}d")
    print(f"           Stop-loss: DD < {STOPLOSS_DD:.0%} → {STOPLOSS_CASH:.0%} cash")
    print(f"           Skip-month: {MOM_SKIP}d  |  Iteraciones MCTS: {ITERACIONES}")
    print(f"           Selección inicial: {tickers_activos}\n")

    for dia in range(inicio_bt, T_total - 1):
        dia_rel = dia - inicio_bt

        # ── ETAPA 1: ¿Renovar universo? ──
        if dia_rel > 0 and dias_desde_reselect >= RESELECT_FREQ:
            nuevos_tickers = seleccionar_top_n(df_precios, dia, spy_series=spy_series)

            if set(nuevos_tickers) != set(tickers_activos):
                # Re-mapear pesos: activos que permanecen conservan su peso,
                # activos nuevos reciben peso proporcional de los que salen.
                pesos_dict = dict(zip(tickers_activos, pesos))
                peso_salientes = sum(pesos_dict.get(t, 0.0)
                                     for t in tickers_activos
                                     if t not in nuevos_tickers)
                entrantes = [t for t in nuevos_tickers
                             if t not in tickers_activos]
                n_entrantes = len(entrantes)

                nuevos_pesos = np.zeros(len(nuevos_tickers))
                for j, t in enumerate(nuevos_tickers):
                    if t in pesos_dict:
                        nuevos_pesos[j] = pesos_dict[t]
                    elif n_entrantes > 0:
                        nuevos_pesos[j] = peso_salientes / n_entrantes

                # No normalizar a 1 si hay fracción en cash
                total_p = nuevos_pesos.sum()
                if total_p < 1e-10:
                    nuevos_pesos = np.ones(len(nuevos_tickers)) / len(nuevos_tickers)

                # Coste de transacción por la rotación de universo
                turnover_univ = float(np.sum(np.abs(nuevos_pesos - np.ones(len(nuevos_tickers)) / len(nuevos_tickers))))
                turnover_univ = min(turnover_univ, peso_salientes * 2)
                valor *= (1.0 - turnover_univ * TX_COST)

                tickers_activos = nuevos_tickers
                N_act = len(tickers_activos)
                idx_activos = [all_tickers.index(t) for t in tickers_activos]
                pesos = nuevos_pesos

                seleccion_hist.append((dia, tickers_activos[:]))

            dias_desde_reselect = 0

        # ── STOP-LOSS DINÁMICO ──
        peak_valor = max(peak_valor, valor)
        dd_actual  = (valor - peak_valor) / peak_valor

        if not stoploss_activo and dd_actual < STOPLOSS_DD:
            # Activar stop-loss: forzar STOPLOSS_CASH a cash
            pesos_pre_sl = pesos.copy()
            factor = 1.0 - STOPLOSS_CASH
            pesos = pesos * factor
            turnover_sl = float(np.sum(np.abs(pesos - pesos_pre_sl)))
            valor *= (1.0 - turnover_sl * TX_COST)
            stoploss_activo = True
            stoploss_eventos += 1
            acciones_hist.append((dia, -1, f"STOP_LOSS (DD={dd_actual:.1%})"))

        elif stoploss_activo and dd_actual > STOPLOSS_REENTRY_DD:
            # Desactivar stop-loss: redistribuir cash a pesos iguales
            pesos_pre = pesos.copy()
            pesos = np.ones(N_act) / N_act
            turnover_re = float(np.sum(np.abs(pesos - pesos_pre)))
            valor *= (1.0 - turnover_re * TX_COST)
            stoploss_activo = False
            acciones_hist.append((dia, -2, f"REENTRY (DD={dd_actual:.1%})"))

        # ── ETAPA 2: ¿Rebalancear con MCTS? (solo si stop-loss no activo) ──
        if not stoploss_activo and (dias_desde_rebal >= REBAL_FREQ or dia_rel == 0):
            # Construir prices_ext para los activos activos + SPY
            prices_activos = all_prices[:, idx_activos]  # (T_total, N_act)
            prices_ext = np.hstack([prices_activos,
                                    spy_arr[:, None]])   # (T_total, N_act+1)

            estado = crear_estado(dia, pesos, valor, prices_activos, spy_arr)
            registrar = (dia == dia_conv)
            accion_idx, conv = ejecutar_mcts(
                estado, prices_ext, tickers_activos, rng,
                registrar_convergencia=registrar,
            )
            if registrar:
                historial_conv = conv

            acciones = construir_acciones(tickers_activos)
            pesos_nuevo = aplicar_accion(pesos, accion_idx, tickers_activos)

            turnover = float(np.sum(np.abs(pesos_nuevo - pesos)))
            valor   *= (1.0 - turnover * TX_COST)
            pesos    = pesos_nuevo

            acciones_hist.append((dia, accion_idx, acciones[accion_idx]))
            dias_desde_rebal = 0

        # ── Retornos reales del mercado ──
        ret = (all_prices[dia + 1, idx_activos] -
               all_prices[dia, idx_activos]) / all_prices[dia, idx_activos]

        # Fracción en cash gana tasa libre de riesgo diaria
        fraccion_invertida = float(pesos.sum())
        fraccion_cash = 1.0 - fraccion_invertida
        rf_diaria = RISK_FREE_ANNUAL / 252

        ret_total = float(np.dot(pesos, 1.0 + ret)) + fraccion_cash * (1.0 + rf_diaria)
        valor *= ret_total

        pesos_raw = pesos * (1.0 + ret)
        pesos_sum = pesos_raw.sum()
        if pesos_sum > 1e-10:
            # Mantener la proporción cash/invertido correcta
            pesos = pesos_raw / (pesos_sum + fraccion_cash * (1.0 + rf_diaria)) if fraccion_cash > 0.01 else pesos_raw / pesos_sum
        else:
            pesos = np.ones(N_act) / N_act

        valores.append(valor)
        pesos_dict_dia = dict(zip(tickers_activos, pesos))
        pesos_dict_dia["CASH"] = fraccion_cash
        pesos_historia.append(pesos_dict_dia)

        spy_valores.append(spy_shares * spy_arr[dia + 1])
        ew_valores.append(float(np.sum(ew_shares * all_prices[dia + 1, ew_idx_fijos])))

        dias_desde_reselect += 1
        dias_desde_rebal    += 1

        # Progreso
        sl_tag = " [SL]" if stoploss_activo else ""
        pct = (dia_rel + 1) / (T_total - inicio_bt - 1) * 100
        print(
            f"\r[MCTS+MOM] {dia_rel+1}/{T_total-inicio_bt-1} ({pct:.0f}%)  "
            f"Cartera: {valor:>10,.0f}$  SPY: {spy_valores[-1]:>10,.0f}$  "
            f"DD: {dd_actual:+.1%}{sl_tag}  "
            f"Cash: {fraccion_cash:.0%}",
            end="", flush=True,
        )

    print()
    print(f"[INFO] Stop-loss activado {stoploss_eventos} veces durante el backtest")
    return {
        "valores":         valores,
        "spy_valores":     spy_valores,
        "ew_valores":      ew_valores,
        "pesos_historia":  pesos_historia,
        "acciones_hist":   acciones_hist,
        "seleccion_hist":  seleccion_hist,
        "historial_conv":  historial_conv,
        "all_tickers":     all_tickers,
        "stoploss_eventos": stoploss_eventos,
        "dates":           dates_all[inicio_bt:],
    }


# =============================================================================
# SECCIÓN 9 — MÉTRICAS FINANCIERAS
# =============================================================================

RISK_FREE_ANNUAL = 0.045


def calcular_metricas(valores: list[float], etiqueta: str = "") -> dict:
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
    print("\n" + "=" * 65)
    print(f"{'Métrica':<22} {'MCTS+MOM':>12} {'SPY B&H':>12} {'EW B&H':>12}")
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
# SECCIÓN 10 — VISUALIZACIONES
# =============================================================================

COLORES = plt.cm.tab10(np.linspace(0, 0.9, 10))
COLORES20 = plt.cm.tab20(np.linspace(0, 0.95, 20))


def graficar(resultados: dict, output_dir: str = OUTPUT_DIR) -> None:
    """
    Genera tres figuras:

    Figura 1 (4 paneles): Portafolio normalizado | Rotación de universo |
                          Drawdown comparativo   | Alfa acumulado vs SPY

    Figura 2: Convergencia UCB

    Figura 3: Scores de momentum por ETF a lo largo del backtest
    """
    os.makedirs(output_dir, exist_ok=True)

    valores  = np.array(resultados["valores"])
    spy_v    = np.array(resultados["spy_valores"])
    ew_v     = np.array(resultados["ew_valores"])
    dates    = resultados["dates"]

    base  = valores[0]
    v_n   = valores / base * 100
    spy_n = spy_v   / base * 100
    ew_n  = ew_v    / base * 100

    # ── Figura 1: 4 paneles ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 17))
    fig.suptitle("MCTS + Filtro Momentum — Backtest",
                 fontsize=14, fontweight="bold", y=0.995)

    # ── Bloque de parámetros bajo el título ─────────────────────────────────
    sl_str = f"{STOPLOSS_DD:.0%}" if STOPLOSS_DD > -0.90 else "desactivado"
    param_line1 = (
        f"Universo: {len(resultados.get('all_tickers', UNIVERSO))} ETFs  │  "
        f"TOP_N: {TOP_N}  │  "
        f"Lookback: {MOM_LOOKBACK}d  │  "
        f"Skip: {MOM_SKIP}d  │  "
        f"Momentum: relativo vs SPY  │  "
        f"Corr umbral: {CORR_UMBRAL}"
    )
    param_line2 = (
        f"Reselección: {RESELECT_FREQ}d  │  "
        f"Rebalanceo: {REBAL_FREQ}d  │  "
        f"Iteraciones MCTS: {ITERACIONES}  │  "
        f"Tilt: {TILT:.0%}  │  "
        f"TX cost: {TX_COST:.1%}  │  "
        f"Stop-loss: {sl_str}"
    )
    fig.text(0.5, 0.975, param_line1, ha="center", va="top",
             fontsize=8.5, color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                       edgecolor="#cccccc", alpha=0.8))
    fig.text(0.5, 0.962, param_line2, ha="center", va="top",
             fontsize=8.5, color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                       edgecolor="#cccccc", alpha=0.8))

    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 2.5, 1.5, 1.5], hspace=0.45,
                           top=0.945, bottom=0.04)

    # Panel 1: Portafolio normalizado base 100
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, v_n,   lw=2.0, color="#1f77b4", label="MCTS+MOM",   zorder=3)
    ax1.plot(dates, spy_n, lw=1.5, color="#ff7f0e", ls="--", label="SPY B&H")
    ax1.plot(dates, ew_n,  lw=1.5, color="#2ca02c", ls=":",  label="EW B&H (sel. inicial)")
    ax1.axhline(100, color="gray", lw=0.8)
    ax1.set_title("Evolución de la Cartera (base 100)", fontsize=11,
                  fontweight="bold")
    ax1.set_ylabel("Valor (base 100)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Qué activos estaban seleccionados en cada momento
    ax2 = fig.add_subplot(gs[1])
    seleccion_hist = resultados["seleccion_hist"]
    all_tickers = resultados["all_tickers"]

    # Crear matriz de presencia (días x tickers del universo)
    n_dias = len(dates)
    presencia = np.zeros((n_dias, len(all_tickers)))

    sel_actual = seleccion_hist[0][1]
    sel_idx = 0
    for d in range(n_dias):
        dia_abs = d  # relativo al inicio del backtest
        # ¿Hay una nueva selección?
        if sel_idx + 1 < len(seleccion_hist):
            next_dia, next_sel = seleccion_hist[sel_idx + 1]
            next_dia_rel = next_dia - (len(resultados["dates"]) -
                                        n_dias)  # ajustar a índice bt
            # Buscar en el offset correcto
            if d >= (seleccion_hist[sel_idx + 1][0] -
                     seleccion_hist[0][0]):
                sel_actual = next_sel
                sel_idx += 1

        for t in sel_actual:
            if t in all_tickers:
                presencia[d, all_tickers.index(t)] = 1

    # Mostrar como heatmap
    tickers_activos_alguna_vez = [t for i, t in enumerate(all_tickers)
                                   if presencia[:, i].sum() > 0]
    idx_activos_ever = [all_tickers.index(t) for t in tickers_activos_alguna_vez]
    presencia_filtrada = presencia[:, idx_activos_ever].T

    ax2.imshow(presencia_filtrada, aspect="auto", cmap="Blues",
               interpolation="nearest",
               extent=[0, n_dias, len(tickers_activos_alguna_vez), 0])
    ax2.set_yticks(np.arange(len(tickers_activos_alguna_vez)) + 0.5)
    ax2.set_yticklabels(tickers_activos_alguna_vez, fontsize=8)
    ax2.set_title("Rotación del Universo (azul = seleccionado)", fontsize=11,
                  fontweight="bold")
    ax2.set_xlabel("Sesión de trading")

    # Líneas verticales en días de reselección
    for dia_sel, _ in seleccion_hist:
        d_rel = dia_sel - seleccion_hist[0][0]
        if 0 <= d_rel < n_dias:
            ax2.axvline(d_rel, color="red", lw=0.5, alpha=0.5)

    # Panel 3: Drawdown
    ax3 = fig.add_subplot(gs[2])
    peak_mcts = np.maximum.accumulate(valores)
    dd_mcts   = (valores - peak_mcts) / peak_mcts * 100
    ax3.fill_between(dates, dd_mcts, 0, color="#d62728", alpha=0.55,
                     label="MCTS+MOM DD")
    peak_spy = np.maximum.accumulate(spy_v)
    dd_spy   = (spy_v - peak_spy) / peak_spy * 100
    ax3.plot(dates, dd_spy, lw=1.0, color="#ff7f0e", ls="--",
             label="SPY DD", alpha=0.8)
    ax3.set_title("Drawdown (%)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("DD (%)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Alfa acumulado
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

    ruta1 = os.path.join(output_dir, "backtest_momentum.png")
    plt.savefig(ruta1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico guardado → {ruta1}")

    # ── Figura 2: Convergencia UCB ───────────────────────────────────────────
    historial = resultados.get("historial_conv")
    if historial:
        fig2, ax = plt.subplots(figsize=(10, 5))
        # Necesitamos los tickers del momento de la convergencia
        for idx, vals in historial.items():
            if vals:
                ax.plot(vals, lw=1.4,
                        color=COLORES[idx % len(COLORES)],
                        label=f"Acción {idx}")
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
    os.makedirs(output_dir, exist_ok=True)
    ruta = os.path.join(output_dir, "backtest_report.md")

    # Conteo de acciones
    acciones_resumen: dict[str, int] = {}
    for _, _, nombre in resultados["acciones_hist"]:
        acciones_resumen[nombre] = acciones_resumen.get(nombre, 0) + 1

    # Conteo de ETFs seleccionados
    etf_conteo: dict[str, int] = {}
    for _, tickers in resultados["seleccion_hist"]:
        for t in tickers:
            etf_conteo[t] = etf_conteo.get(t, 0) + 1

    lineas = [
        "# Resultados — MCTS + Filtro Momentum Ajustado por Volatilidad",
        "",
        f"> Generado el {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Configuración",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| Universo amplio | `{len(UNIVERSO)} ETFs` |",
        f"| Activos seleccionados/mes | `{TOP_N}` |",
        f"| Lookback momentum | `{MOM_LOOKBACK} días (~3 meses)` |",
        f"| Skip-month | `{MOM_SKIP} días` |",
        f"| Max por categoría | `{MAX_POR_CAT}` |",
        f"| Umbral correlación | `{CORR_UMBRAL}` |",
        f"| Renovación universo | `cada {RESELECT_FREQ} días (~mensual)` |",
        f"| Stop-loss DD | `{STOPLOSS_DD:.0%}` |",
        f"| Stop-loss cash | `{STOPLOSS_CASH:.0%}` |",
        f"| Reentrada DD | `{STOPLOSS_REENTRY_DD:.0%}` |",
        f"| Benchmark | `{SPY_TICK}` |",
        f"| Período datos | `{PERIOD}` |",
        f"| Capital inicial | `${CAPITAL_INIT:,.0f}` |",
        f"| Iteraciones MCTS | `{ITERACIONES}` |",
        f"| Días rollout | `{DIAS_ROLLOUT}` |",
        f"| Ventana calibración GBM | `{VENTANA_CALIB}` |",
        f"| Trayectorias GBM | `{N_PATHS}` |",
        f"| Tilt por rotación | `{TILT:.0%}` |",
        f"| Coste transacción | `{TX_COST:.1%}` |",
        f"| Frecuencia rebalanceo MCTS | `cada {REBAL_FREQ} días` |",
        "",
        "## Resultados financieros",
        "",
        "| Métrica | MCTS+MOM | SPY B&H | EW B&H |",
        "|---|---|---|---|",
        f"| Valor final ($) | {m_mcts['final']:,.2f} | {m_spy['final']:,.2f} | {m_ew['final']:,.2f} |",
        f"| ROI acumulado | {m_mcts['roi']:+.2%} | {m_spy['roi']:+.2%} | {m_ew['roi']:+.2%} |",
        f"| Retorno anual | {m_mcts['annual_ret']:+.2%} | {m_spy['annual_ret']:+.2%} | {m_ew['annual_ret']:+.2%} |",
        f"| Sharpe | {m_mcts['sharpe']:.4f} | {m_spy['sharpe']:.4f} | {m_ew['sharpe']:.4f} |",
        f"| Sortino | {m_mcts['sortino']:.4f} | {m_spy['sortino']:.4f} | {m_ew['sortino']:.4f} |",
        f"| Max Drawdown | {m_mcts['max_dd']:+.2%} | {m_spy['max_dd']:+.2%} | {m_ew['max_dd']:+.2%} |",
        f"| Calmar | {m_mcts['calmar']:.4f} | {m_spy['calmar']:.4f} | {m_ew['calmar']:.4f} |",
        "",
        "## Distribución de acciones MCTS",
        "",
        "| Acción | Veces |",
        "|---|---|",
    ]
    for nombre, cnt in sorted(acciones_resumen.items(), key=lambda x: -x[1]):
        lineas.append(f"| {nombre} | {cnt} |")

    lineas += [
        "",
        "## ETFs más seleccionados por el filtro momentum",
        "",
        "| ETF | Veces seleccionado |",
        "|---|---|",
    ]
    for nombre, cnt in sorted(etf_conteo.items(), key=lambda x: -x[1]):
        lineas.append(f"| {nombre} | {cnt} |")

    lineas += [
        "",
        "## Notas metodológicas",
        "",
        "- **Etapa 1 — Filtro momentum con skip-month**: cada mes se seleccionan los TOP_N ETFs con mayor",
        f"  `retorno_log_acumulado(t-{MOM_LOOKBACK}, t-{MOM_SKIP}) / volatilidad_anualizada`.",
        "  El skip-month excluye el último mes para evitar reversión a la media (Jegadeesh & Titman).",
        "- **Diversificación forzada**: máx 2 ETFs por categoría + correlación < 0.85 entre seleccionados.",
        "- **Etapa 2 — MCTS relativo con opción CASH**: rota semanalmente entre los activos seleccionados,",
        "  optimizando `E[log(1+ret_cartera)] − E[log(1+ret_SPY)]`. Incluye acción IR_A_CASH.",
        f"- **Stop-loss dinámico**: si DD < {STOPLOSS_DD:.0%}, fuerza {STOPLOSS_CASH:.0%} a cash.",
        f"  Reentrada cuando DD > {STOPLOSS_REENTRY_DD:.0%}. Eventos de stop-loss: {resultados.get('stoploss_eventos', 0)}.",
        "- **Re-mapeo de pesos**: al cambiar universo, los activos que permanecen conservan",
        "  su peso; los nuevos reciben el peso proporcional de los que salen.",
        "",
        "![Backtest](backtest_momentum.png)",
        "![Convergencia UCB](convergencia_mcts.png)",
    ]

    with open(ruta, "w") as f:
        f.write("\n".join(lineas))
    print(f"[INFO] Reporte guardado → {ruta}")


# =============================================================================
# SECCIÓN 11 — PUNTO DE ENTRADA
# =============================================================================

def main() -> None:
    df_precios, spy_series = descargar_universo(UNIVERSO, SPY_TICK, PERIOD)

    resultados = backtest(df_precios, spy_series)

    m_mcts = calcular_metricas(resultados["valores"],     "MCTS+MOM")
    m_spy  = calcular_metricas(resultados["spy_valores"], "SPY B&H")
    m_ew   = calcular_metricas(resultados["ew_valores"],  "EW B&H")

    imprimir_metricas(m_mcts, m_spy, m_ew)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              OUTPUT_DIR)
    graficar(resultados, output_dir)
    guardar_reporte(m_mcts, m_spy, m_ew, resultados, output_dir)

    print("[OK] Backtest MCTS + Filtro Momentum completado.")


if __name__ == "__main__":
    main()
