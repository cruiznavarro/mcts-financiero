"""
mcts_portfolio.py
=================
Monte Carlo Tree Search para Optimización de Cartera Multi-Activo.

Extensión del modelo mcts_trading.py a N activos con:

  1. Estado vectorial     — pesos y valor de cartera (no precio individual)
  2. Símplex discretizado — espacio de acciones = todas las asignaciones de
                            pesos válidas con paso 1/granularity
  3. Progressive Widening — número de hijos crece con ceil(k_w * n^alpha_w)
                            para controlar la explosión combinatoria
  4. GBM multivariado     — rollout con matriz de covarianzas (Cholesky)
  5. Sin lookback bias    — la expansión usa retornos GBM simulados,
                            no precios históricos futuros
  6. Recompensa log-ret   — más estable numéricamente que el valor absoluto
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constantes ───────────────────────────────────────────────────────────────

C_UCB:            float = math.sqrt(2)   # Constante de exploración estándar
RISK_FREE_ANNUAL: float = 0.045          # Tasa libre de riesgo (~4.5 % anual)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Espacio de acciones: símplex discretizado
# ═════════════════════════════════════════════════════════════════════════════

def enumerate_simplex(n_assets: int, granularity: int) -> list[np.ndarray]:
    """
    Enumera todos los vectores de pesos en el símplex estándar Δ^{n-1}
    con paso 1/granularity.

    Tamaño del espacio: C(n_assets + granularity - 1, granularity)

    Ejemplos
    --------
    n=2, g=4 →  5 vectores
    n=4, g=4 → 35 vectores
    n=5, g=4 → 70 vectores
    n=5, g=5 → 126 vectores
    """
    results: list[np.ndarray] = []

    def _recurse(remaining: int, n_left: int, current: list[float]) -> None:
        if n_left == 1:
            results.append(np.array(current + [remaining / granularity],
                                    dtype=np.float64))
            return
        for i in range(remaining + 1):
            _recurse(remaining - i, n_left - 1, current + [i / granularity])

    _recurse(granularity, n_assets, [])
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. Estado de la cartera
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class State:
    """
    Estado inmutable de la cartera en el día `day`.

    Attributes
    ----------
    day             : Índice del día en la serie de precios.
    weights         : Pesos actuales como tuple (inmutable, suma = 1). Shape (N,)
    portfolio_value : Valor total de la cartera en moneda base.
    """
    day:             int
    weights:         tuple   # tuple[float, ...] — inmutable para hashabilidad
    portfolio_value: float

    def apply_action(
        self,
        new_weights:      np.ndarray,
        price_returns:    np.ndarray,
        transaction_cost: float = 0.001,
    ) -> "State":
        """
        Aplica una reasignación de pesos, cobra costes de transacción y
        avanza un paso de tiempo.

        Secuencia de eventos (orden correcto)
        --------------------------------------
        1. Rebalancear a new_weights: pagar coste proporcional al turnover.
        2. El mercado evoluciona: la cartera crece según price_returns.

        Parameters
        ----------
        new_weights      : Nuevos pesos objetivo. Shape (N,)
        price_returns    : Retornos del período siguiente. Shape (N,)
        transaction_cost : Fracción del turnover cobrada como comisión.
        """
        old_w = np.array(self.weights, dtype=np.float64)
        new_w = np.asarray(new_weights, dtype=np.float64)

        # Turnover = suma de cambios absolutos en pesos
        turnover = float(np.sum(np.abs(new_w - old_w)))
        cost     = self.portfolio_value * turnover * transaction_cost

        pv_after_cost = max(self.portfolio_value - cost, 0.0)
        new_value     = pv_after_cost * float(np.dot(new_w, 1.0 + price_returns))

        return State(
            day=self.day + 1,
            weights=tuple(new_w.tolist()),
            portfolio_value=max(new_value, 0.0),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Nodo con Progressive Widening
# ═════════════════════════════════════════════════════════════════════════════

class Node:
    """
    Nodo del árbol MCTS con Progressive Widening.

    El PW limita el número de hijos que puede tener un nodo a:
        max_children(n) = ceil(k_w * n^alpha_w)

    Esto evita que nodos poco visitados generen centenares de hijos (lo que
    fragmentaría las estadísticas) mientras permite explorar más acciones
    a medida que el nodo acumula visitas.

    Parámetros típicos: k_w=2.0, alpha_w=0.5  (raíz cuadrada).
    """

    def __init__(
        self,
        state:      State,
        action_idx: Optional[int] = None,
        parent:     Optional["Node"] = None,
        k_w:        float = 2.0,
        alpha_w:    float = 0.5,
    ) -> None:
        self.state      = state
        self.action_idx = action_idx    # índice en ACTIONS_ALL (None = raíz)
        self.parent     = parent
        self.children:  list[Node] = []
        self.n:         int   = 0       # visitas
        self.w:         float = 0.0     # recompensa acumulada
        self.k_w        = k_w
        self.alpha_w    = alpha_w

    # ── UCT ──────────────────────────────────────────────────────────────────

    def ucb1(self, c: float = C_UCB) -> float:
        if self.n == 0:
            return math.inf
        expl = self.w / self.n
        if self.parent is None or self.parent.n == 0:
            return expl
        return expl + c * math.sqrt(math.log(self.parent.n) / self.n)

    def best_child(self, c: float = C_UCB) -> "Node":
        return max(self.children, key=lambda ch: ch.ucb1(c))

    # ── Expansión con PW ─────────────────────────────────────────────────────

    def max_children_allowed(self) -> int:
        if self.n == 0:
            return 1
        return max(1, math.ceil(self.k_w * (self.n ** self.alpha_w)))

    def can_expand(self, n_total_actions: int) -> bool:
        cap = min(self.max_children_allowed(), n_total_actions)
        return len(self.children) < cap

    def tried_indices(self) -> set[int]:
        return {ch.action_idx for ch in self.children
                if ch.action_idx is not None}

    def add_child(self, action_idx: int, state: State) -> "Node":
        child = Node(state=state, action_idx=action_idx, parent=self,
                     k_w=self.k_w, alpha_w=self.alpha_w)
        self.children.append(child)
        return child


# ═════════════════════════════════════════════════════════════════════════════
# 4. Motor MCTS multi-activo
# ═════════════════════════════════════════════════════════════════════════════

class MCTS_Portfolio:
    """
    Motor de Monte Carlo Tree Search para optimización de cartera.

    Parámetros
    ----------
    prices_matrix : Matriz de precios históricos shape (T, N).
    actions       : Lista de pesos válidos (símplex discretizado).
    iterations    : Iteraciones MCTS por decisión.
    rollout_days  : Horizonte de simulación GBM en el rollout.
    window        : Días de ventana rolling para calibrar μ y Σ.
    c_ucb         : Constante de exploración UCT.
    k_w, alpha_w  : Parámetros de Progressive Widening.
    tx_cost       : Coste de transacción proporcional al turnover.
    rng_seed      : Semilla para reproducibilidad.
    """

    def __init__(
        self,
        prices_matrix: np.ndarray,
        actions:       list[np.ndarray],
        iterations:    int   = 300,
        rollout_days:  int   = 20,
        window:        int   = 60,
        c_ucb:         float = C_UCB,
        k_w:           float = 2.0,
        alpha_w:       float = 0.5,
        tx_cost:       float = 0.001,
        rng_seed:      int   = 42,
    ) -> None:
        self.prices      = prices_matrix          # (T, N)
        self.actions     = actions
        self.n_actions   = len(actions)
        self.iterations  = iterations
        self.rollout_days = rollout_days
        self.window      = window
        self.c_ucb       = c_ucb
        self.k_w         = k_w
        self.alpha_w     = alpha_w
        self.tx_cost     = tx_cost
        self.rng         = np.random.default_rng(rng_seed)

    # ── Calibración GBM multivariado ─────────────────────────────────────────

    def _calibrate(self, day: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Estima μ (drift diario) y Σ (covarianza diaria) con ventana rolling.

        Returns
        -------
        mu    : Vector drift diario.  Shape (N,)
        sigma : Matriz covarianza diaria. Shape (N, N)
        """
        N     = self.prices.shape[1]
        start = max(0, day - self.window)
        end   = min(day + 1, len(self.prices))
        win   = self.prices[start:end]

        if len(win) >= 3:
            log_ret = np.diff(np.log(win), axis=0)    # (w-1, N)
            mu      = np.mean(log_ret, axis=0)         # (N,)
            sigma   = np.cov(log_ret, rowvar=False)    # (N, N)
            # Regularización para garantizar definida positiva
            sigma  += np.eye(N) * 1e-8
        else:
            mu    = np.zeros(N)
            sigma = np.eye(N) * (0.01 ** 2)

        return mu, sigma

    def _cholesky_safe(self, matrix: np.ndarray) -> np.ndarray:
        """Cholesky con fallback diagonal si la matriz no es definida positiva."""
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            return np.diag(np.sqrt(np.maximum(np.diag(matrix), 1e-10)))

    # ── Paso 3: Rollout GBM multivariado ─────────────────────────────────────

    def _rollout(self, state: State, weights: np.ndarray) -> float:
        """
        Estima el log-retorno de la cartera usando GBM multivariado.

        Modelo (solución exacta, sin error de discretización):
            log S_T^i = log S_0^i + (μ_i - ½ Σ_ii) * T  +  (L Z)_i √T
        donde L = chol(Σ_anual) y Z ~ N(0, I).

        Recompensa = retorno medio de la cartera sobre N_PATHS trayectorias.

        Ventajas sobre el rollout aleatorio del modelo base:
        - Captura correlaciones entre activos.
        - Usa drift y volatilidad calibrados con datos reales recientes.
        - Promedio de trayectorias reduce la varianza del estimador.
        """
        N_PATHS = 50
        N       = len(weights)
        T       = self.rollout_days / 252.0

        mu, sigma = self._calibrate(state.day)
        mu_ann    = mu * 252
        sigma_ann = sigma * 252
        L         = self._cholesky_safe(sigma_ann)

        # Z ~ N(0, I), shape (N, N_PATHS)
        Z   = self.rng.standard_normal((N, N_PATHS))
        eps = L @ Z * math.sqrt(T)                           # (N, N_PATHS)

        drift       = (mu_ann - 0.5 * np.diag(sigma_ann)) * T  # (N,)
        log_ret_mat = drift[:, None] + eps                      # (N, N_PATHS)

        # Retorno de la cartera en cada trayectoria
        asset_ret   = np.exp(log_ret_mat) - 1.0               # (N, N_PATHS)
        port_ret    = weights @ asset_ret                      # (N_PATHS,)

        return float(np.mean(port_ret))

    # ── Paso 1: Selección ────────────────────────────────────────────────────

    def _select(self, node: Node) -> Node:
        current = node
        while not current.can_expand(self.n_actions) and current.children:
            current = current.best_child(self.c_ucb)
        return current

    # ── Paso 2: Expansión (sin lookback bias) ────────────────────────────────

    def _expand(self, node: Node) -> Node:
        """
        Crea un hijo para una acción no probada.

        Para evitar lookback bias, el precio del siguiente día se simula
        con un paso GBM (no se usa el precio histórico real futuro).
        """
        untried    = [i for i in range(self.n_actions)
                      if i not in node.tried_indices()]
        action_idx = int(self.rng.choice(untried))
        new_weights = self.actions[action_idx]

        day       = node.state.day
        mu, sigma = self._calibrate(day)
        mu_ann    = mu * 252
        sigma_ann = sigma * 252
        L         = self._cholesky_safe(sigma_ann)

        # Un paso GBM simulado (1 día)
        T   = 1.0 / 252.0
        z   = self.rng.standard_normal(mu.shape[0])
        log_ret = (mu_ann - 0.5 * np.diag(sigma_ann)) * T + L @ z * math.sqrt(T)
        sim_ret = np.exp(log_ret) - 1.0

        new_state = node.state.apply_action(new_weights, sim_ret, self.tx_cost)
        return node.add_child(action_idx, new_state)

    # ── Paso 4: Retropropagación ─────────────────────────────────────────────

    def _backpropagate(self, node: Node, reward: float) -> None:
        current: Optional[Node] = node
        while current is not None:
            current.n += 1
            current.w += reward
            current = current.parent

    # ── Bucle principal ───────────────────────────────────────────────────────

    def run(self, root_state: State) -> Node:
        """
        Ejecuta `iterations` ciclos MCTS y devuelve la raíz del árbol.
        """
        root = Node(state=root_state, k_w=self.k_w, alpha_w=self.alpha_w)

        for _ in range(self.iterations):
            leaf = self._select(root)

            if leaf.can_expand(self.n_actions):
                leaf = self._expand(leaf)

            w = (self.actions[leaf.action_idx]
                 if leaf.action_idx is not None
                 else np.array(leaf.state.weights))

            reward = self._rollout(leaf.state, w)
            self._backpropagate(leaf, reward)

        return root

    def best_weights(self, root: Node) -> np.ndarray:
        """
        Devuelve los pesos óptimos: hijo con mayor Q (c=0, explotación pura).
        """
        if not root.children:
            return np.array(root.state.weights)
        best = root.best_child(c=0.0)
        if best.action_idx is not None:
            return self.actions[best.action_idx]
        return np.array(root.state.weights)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Backtester con benchmarks
# ═════════════════════════════════════════════════════════════════════════════

class PortfolioBacktester:
    """
    Evalúa el agente MCTS sobre datos históricos y lo compara con:
      - Buy & Hold Equal-Weight (1/N, sin rebalanceo)
      - Equal-Weight Rebalanced diario (1/N, con costes de transacción)

    Métricas reportadas
    -------------------
    ROI acumulado, Ratio de Sharpe anualizado, Máximo Drawdown, Ratio de Calmar.
    """

    def __init__(
        self,
        n_assets:     int,
        granularity:  int   = 4,
        mcts_iters:   int   = 300,
        rollout_days: int   = 20,
        window:       int   = 60,
        c_ucb:        float = C_UCB,
        k_w:          float = 2.0,
        alpha_w:      float = 0.5,
        tx_cost:      float = 0.001,
        rng_seed:     int   = 42,
    ) -> None:
        self.n_assets    = n_assets
        self.granularity = granularity
        self.actions     = enumerate_simplex(n_assets, granularity)
        self.n_actions   = len(self.actions)
        self.mcts_iters  = mcts_iters
        self.rollout_days = rollout_days
        self.window      = window
        self.c_ucb       = c_ucb
        self.k_w         = k_w
        self.alpha_w     = alpha_w
        self.tx_cost     = tx_cost
        self.rng_seed    = rng_seed

        n_actions = len(self.actions)
        print(f"[CONFIG] Activos: {n_assets} | Granularidad: 1/{granularity} "
              f"| Acciones posibles: {n_actions} | Iteraciones MCTS: {mcts_iters}")

    # ── Ejecutar backtest ─────────────────────────────────────────────────────

    def run(
        self,
        prices:       np.ndarray,
        initial_cash: float = 10_000.0,
    ) -> dict:
        """
        Ejecuta el backtest completo sobre la matriz de precios.

        Parameters
        ----------
        prices       : Matriz (T, N) de precios de cierre ajustados.
        initial_cash : Capital inicial.

        Returns
        -------
        dict con series temporales y listas de acciones tomadas.
        """
        T, N = prices.shape
        agent = MCTS_Portfolio(
            prices_matrix=prices,
            actions=self.actions,
            iterations=self.mcts_iters,
            rollout_days=self.rollout_days,
            window=self.window,
            c_ucb=self.c_ucb,
            k_w=self.k_w,
            alpha_w=self.alpha_w,
            tx_cost=self.tx_cost,
            rng_seed=self.rng_seed,
        )

        eq_w  = np.ones(N) / N
        state = State(day=0, weights=tuple(eq_w.tolist()),
                      portfolio_value=initial_cash)

        pv_series:  list[float]       = [initial_cash]
        w_history:  list[np.ndarray]  = [eq_w.copy()]
        act_history: list[np.ndarray] = []

        for day in range(T - 1):
            state = State(day=day,
                          weights=w_history[-1],
                          portfolio_value=pv_series[-1])

            root    = agent.run(state)
            new_w   = agent.best_weights(root)
            act_history.append(new_w.copy())

            # Retornos históricos reales para la ejecución
            ret   = (prices[day + 1] - prices[day]) / prices[day]
            state = state.apply_action(new_w, ret, self.tx_cost)

            pv_series.append(state.portfolio_value)
            w_history.append(np.array(state.weights))

            pct = (day + 1) / (T - 1) * 100
            print(
                f"\r[MCTS] {day+1}/{T-1} ({pct:.0f}%)  "
                f"PV: {state.portfolio_value:>10,.0f}$  "
                f"w: [{' '.join(f'{w:.2f}' for w in new_w)}]",
                end="", flush=True,
            )
        print()

        # ── Benchmark 1: Buy & Hold Equal-Weight ─────────────────────────────
        bah_shares = initial_cash * eq_w / prices[0]   # (N,)
        bah_series = [float(np.sum(bah_shares * prices[d])) for d in range(T)]

        # ── Benchmark 2: Equal-Weight Rebalanced (con costes) ────────────────
        eqr_pv  = initial_cash
        eqr_series: list[float] = [initial_cash]
        for day in range(T - 1):
            ret     = (prices[day + 1] - prices[day]) / prices[day]
            # Pesos tras evolución del mercado (sin rebalancear)
            w_drift = eq_w * (1.0 + ret)
            w_drift = w_drift / w_drift.sum()
            turnover = float(np.sum(np.abs(eq_w - w_drift)))
            cost    = eqr_pv * turnover * self.tx_cost
            eqr_pv  = (eqr_pv - cost) * float(np.dot(eq_w, 1.0 + ret))
            eqr_series.append(eqr_pv)

        return {
            "portfolio_values": pv_series,
            "bah_values":       bah_series,
            "eqr_values":       eqr_series,
            "weight_history":   w_history,
            "action_history":   act_history,
            "initial_cash":     initial_cash,
            "n_actions":        self.n_actions,
        }

    # ── Métricas financieras ──────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        values:     list[float],
        label:      str   = "",
        risk_free:  float = RISK_FREE_ANNUAL,
    ) -> dict:
        """
        Calcula ROI, Sharpe, Sortino, Max Drawdown y Calmar.
        """
        arr     = np.array(values)
        ret     = np.diff(arr) / arr[:-1]
        rf_d    = risk_free / 252

        mean_r  = float(np.mean(ret))
        std_r   = float(np.std(ret, ddof=1)) + 1e-10
        sharpe  = (mean_r - rf_d) / std_r * math.sqrt(252)

        # Sortino: solo volatilidad negativa
        neg_ret   = ret[ret < rf_d]
        down_std  = float(np.std(neg_ret, ddof=1)) + 1e-10 if len(neg_ret) > 1 else std_r
        sortino   = (mean_r - rf_d) / down_std * math.sqrt(252)

        # Máximo Drawdown
        peak   = np.maximum.accumulate(arr)
        dd     = (arr - peak) / peak
        max_dd = float(np.min(dd))

        # Calmar = retorno anual / |max drawdown|
        n_days     = len(arr)
        annual_ret = (arr[-1] / arr[0]) ** (252 / n_days) - 1
        calmar     = annual_ret / abs(max_dd) if max_dd != 0 else float("inf")

        roi = (arr[-1] - arr[0]) / arr[0]

        return {
            "label":      label,
            "roi":        roi,
            "annual_ret": annual_ret,
            "sharpe":     sharpe,
            "sortino":    sortino,
            "max_dd":     max_dd,
            "calmar":     calmar,
            "final":      float(arr[-1]),
        }

    # ── Visualización ─────────────────────────────────────────────────────────

    @staticmethod
    def plot(
        results:    dict,
        tickers:    list[str],
        dates:      pd.DatetimeIndex,
        output_dir: str = "results",
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        pv   = np.array(results["portfolio_values"])
        bah  = np.array(results["bah_values"])
        eqr  = np.array(results["eqr_values"])
        wh   = np.array(results["weight_history"])  # (T, N)
        N    = wh.shape[1]

        base  = pv[0]
        pv_n  = pv  / base * 100
        bah_n = bah / base * 100
        eqr_n = eqr / base * 100

        fig = plt.figure(figsize=(15, 11))
        fig.suptitle("MCTS Portfolio Optimizer — Backtest",
                     fontsize=14, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1.5], hspace=0.45)

        # ── Panel 1: Evolución normalizada ────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(dates, pv_n,  lw=2.0, color="#1f77b4", label="MCTS Portfolio",   zorder=3)
        ax1.plot(dates, bah_n, lw=1.5, color="#ff7f0e", ls="--", label="Buy & Hold (1/N)")
        ax1.plot(dates, eqr_n, lw=1.5, color="#2ca02c", ls=":",  label="EW Rebalanced")
        ax1.axhline(100, color="gray", lw=0.8, ls="-")
        ax1.set_title("Evolución de la Cartera (base 100)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Valor (base 100)")
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Distribución de pesos (área apilada) ─────────────────────
        ax2 = fig.add_subplot(gs[1])
        colors = plt.cm.tab10(np.linspace(0, 0.9, N))
        bottom = np.zeros(len(dates))
        for i, ticker in enumerate(tickers):
            ax2.fill_between(dates, bottom, bottom + wh[:, i],
                             color=colors[i], alpha=0.85, label=ticker)
            bottom = bottom + wh[:, i]
        ax2.set_title("Distribución de Pesos por Activo", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Peso")
        ax2.set_ylim(0, 1.02)
        ax2.legend(fontsize=8, ncol=N, loc="upper right",
                   bbox_to_anchor=(1.0, 1.15))
        ax2.grid(True, alpha=0.2)

        # ── Panel 3: Drawdown ─────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        peak_mcts = np.maximum.accumulate(pv)
        dd_mcts   = (pv - peak_mcts) / peak_mcts * 100
        ax3.fill_between(dates, dd_mcts, 0, color="#d62728", alpha=0.55,
                         label="MCTS DD")
        peak_bah  = np.maximum.accumulate(bah)
        dd_bah    = (bah - peak_bah) / peak_bah * 100
        ax3.plot(dates, dd_bah, lw=1.0, color="#ff7f0e", ls="--",
                 label="B&H DD", alpha=0.7)
        ax3.set_title("Drawdown (%)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("DD (%)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        path = os.path.join(output_dir, "backtest_portfolio.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Gráfico guardado en {path}")

    # ── Reporte markdown ──────────────────────────────────────────────────────

    @staticmethod
    def save_report(
        metrics_mcts: dict,
        metrics_bah:  dict,
        metrics_eqr:  dict,
        config:       dict,
        output_dir:   str = "results",
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "backtest_report.md")

        def fmt_pct(x: float) -> str:
            return f"{x * 100:+.2f}%"

        def fmt_f(x: float) -> str:
            return f"{x:.4f}"

        lines = [
            "# Resultados del Backtest — MCTS Portfolio Multi-Activo",
            "",
            f"> Generado el {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Configuración",
            "",
            "| Parámetro | Valor |",
            "|---|---|",
        ]
        for k, v in config.items():
            lines.append(f"| {k} | `{v}` |")

        lines += [
            "",
            "## Resultados financieros",
            "",
            "| Métrica | MCTS | Buy & Hold (1/N) | EW Rebalanced |",
            "|---|---|---|---|",
            f"| Valor final [$] | {metrics_mcts['final']:,.2f} | "
            f"{metrics_bah['final']:,.2f} | {metrics_eqr['final']:,.2f} |",
            f"| ROI acumulado | {fmt_pct(metrics_mcts['roi'])} | "
            f"{fmt_pct(metrics_bah['roi'])} | {fmt_pct(metrics_eqr['roi'])} |",
            f"| Retorno anual | {fmt_pct(metrics_mcts['annual_ret'])} | "
            f"{fmt_pct(metrics_bah['annual_ret'])} | {fmt_pct(metrics_eqr['annual_ret'])} |",
            f"| Sharpe | {fmt_f(metrics_mcts['sharpe'])} | "
            f"{fmt_f(metrics_bah['sharpe'])} | {fmt_f(metrics_eqr['sharpe'])} |",
            f"| Sortino | {fmt_f(metrics_mcts['sortino'])} | "
            f"{fmt_f(metrics_bah['sortino'])} | {fmt_f(metrics_eqr['sortino'])} |",
            f"| Max Drawdown | {fmt_pct(metrics_mcts['max_dd'])} | "
            f"{fmt_pct(metrics_bah['max_dd'])} | {fmt_pct(metrics_eqr['max_dd'])} |",
            f"| Calmar | {fmt_f(metrics_mcts['calmar'])} | "
            f"{fmt_f(metrics_bah['calmar'])} | {fmt_f(metrics_eqr['calmar'])} |",
            "",
            "## Notas metodológicas",
            "",
            "- **Espacio de acciones:** símplex discretizado (pesos con paso `1/granularity`, suma = 1).",
            "- **Progressive Widening:** `max_children = ceil(k_w * n^alpha_w)` para controlar branching.",
            "- **Rollout:** GBM multivariado con Cholesky, calibrado con ventana rolling.",
            "- **Sin lookback bias:** la expansión usa retornos GBM simulados, no precios reales futuros.",
            "- **Costes de transacción:** proporcionales al turnover de pesos.",
            "",
            "![Backtest](backtest_portfolio.png)",
        ]

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"[INFO] Reporte guardado en {path}")
