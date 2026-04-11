"""
run_backtest.py
===============
Punto de entrada para el backtest de MCTS Portfolio Multi-Activo.

Activos por defecto (cartera diversificada):
    SPY  — S&P 500 ETF          (renta variable EE.UU.)
    QQQ  — Nasdaq-100 ETF       (tecnología)
    GLD  — Gold ETF             (materias primas / refugio)
    TLT  — iShares 20Y+ T-Bond  (renta fija larga)
    EFA  — MSCI EAFE ETF        (renta variable internacional)

Uso
---
    python run_backtest.py

Ajusta la sección CONFIG para cambiar activos, período y parámetros MCTS.
"""

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

# Añadir el directorio al path para importar el módulo
sys.path.insert(0, os.path.dirname(__file__))

from mcts_portfolio import PortfolioBacktester

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — ajusta aquí los parámetros
# ═════════════════════════════════════════════════════════════════════════════

TICKERS      = ["SPY", "QQQ", "GLD", "TLT", "EFA"]
PERIOD       = "2y"            # Período yfinance: "1y", "2y", "3y", etc.
INITIAL_CASH = 10_000.0        # Capital inicial ($)

# MCTS
GRANULARITY  = 4               # Paso de pesos: 1/4 = 25%.  Acciones = C(N+g-1, g-1)
MCTS_ITERS   = 300             # Iteraciones MCTS por decisión de rebalanceo
ROLLOUT_DAYS = 20              # Horizonte GBM en el rollout (días)
WINDOW       = 60              # Ventana rolling para calibrar μ y Σ (días)
C_UCB        = 1.41421356      # Constante de exploración (√2)
K_W          = 2.0             # Progressive Widening: k_w
ALPHA_W      = 0.5             # Progressive Widening: alpha_w
TX_COST      = 0.001           # Coste de transacción (0.1% del turnover)
RNG_SEED     = 42

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "results")

# ═════════════════════════════════════════════════════════════════════════════


def download_prices(tickers: list[str], period: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Descarga precios de cierre ajustados para todos los tickers.

    Returns
    -------
    prices : np.ndarray shape (T, N)
    dates  : pd.DatetimeIndex length T
    """
    print(f"[DATA] Descargando {tickers} — período: {period}")
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    # Extraer columna Close (multi-ticker → multiindex)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][tickers]
    else:
        close = raw[["Close"]]
        close.columns = tickers

    close = close.dropna()
    print(f"[DATA] {len(close)} sesiones descargadas ({close.index[0].date()} → "
          f"{close.index[-1].date()})")

    return close.to_numpy(dtype=np.float64), close.index


def main() -> None:
    # ── 1. Datos ──────────────────────────────────────────────────────────────
    prices, dates = download_prices(TICKERS, PERIOD)
    T, N = prices.shape

    # ── 2. Backtester ─────────────────────────────────────────────────────────
    backtester = PortfolioBacktester(
        n_assets=N,
        granularity=GRANULARITY,
        mcts_iters=MCTS_ITERS,
        rollout_days=ROLLOUT_DAYS,
        window=WINDOW,
        c_ucb=C_UCB,
        k_w=K_W,
        alpha_w=ALPHA_W,
        tx_cost=TX_COST,
        rng_seed=RNG_SEED,
    )

    # ── 3. Ejecutar ───────────────────────────────────────────────────────────
    results = backtester.run(prices, initial_cash=INITIAL_CASH)

    # ── 4. Métricas ───────────────────────────────────────────────────────────
    m_mcts = PortfolioBacktester.compute_metrics(results["portfolio_values"], "MCTS")
    m_bah  = PortfolioBacktester.compute_metrics(results["bah_values"],       "B&H")
    m_eqr  = PortfolioBacktester.compute_metrics(results["eqr_values"],       "EWR")

    print("\n" + "═" * 60)
    print(f"{'Métrica':<22} {'MCTS':>10} {'Buy&Hold':>10} {'EW Reb.':>10}")
    print("─" * 60)
    for key, label in [("roi", "ROI acumulado"),
                        ("annual_ret", "Retorno anual"),
                        ("sharpe",  "Sharpe"),
                        ("sortino", "Sortino"),
                        ("max_dd",  "Max Drawdown"),
                        ("calmar",  "Calmar")]:
        fmt = lambda v: f"{v*100:+.2f}%" if key in ("roi","annual_ret","max_dd") else f"{v:.4f}"
        print(f"{label:<22} {fmt(m_mcts[key]):>10} {fmt(m_bah[key]):>10} {fmt(m_eqr[key]):>10}")
    print("═" * 60)
    print(f"{'Valor final ($)':<22} {m_mcts['final']:>10,.2f} "
          f"{m_bah['final']:>10,.2f} {m_eqr['final']:>10,.2f}")
    print("═" * 60)

    # ── 5. Visualización y reporte ────────────────────────────────────────────
    PortfolioBacktester.plot(results, TICKERS, dates, OUTPUT_DIR)

    config = {
        "Tickers":        " | ".join(TICKERS),
        "Período":        PERIOD,
        "Sesiones":       T,
        "Granularidad":   f"1/{GRANULARITY} ({results['n_actions']} acciones)",
        "Iteraciones MCTS": MCTS_ITERS,
        "Horizonte rollout": f"{ROLLOUT_DAYS} días",
        "Ventana rolling":  f"{WINDOW} días",
        "Capital inicial":  f"{INITIAL_CASH:,.0f} $",
        "Coste transacción": f"{TX_COST*100:.1f}% turnover",
        "Prog. Widening":   f"k_w={K_W}, alpha_w={ALPHA_W}",
        "Semilla RNG":      RNG_SEED,
    }
    PortfolioBacktester.save_report(m_mcts, m_bah, m_eqr, config, OUTPUT_DIR)


if __name__ == "__main__":
    main()
