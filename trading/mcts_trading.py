"""
mcts_trading.py
===============
Esqueleto de Trading Algorítmico basado en Monte Carlo Tree Search (MCTS).

Trabajo de Fin de Grado — Aplicación de Matemática Discreta
(Teoría de Grafos, Árboles y Optimización Combinatoria) al Algorithmic Trading.

Estructura:
    - State          : Estado del entorno (día, cash, acciones, precio).
    - Action         : Acciones discretas (Comprar 10%, Vender 10%, Mantener).
    - Node           : Nodo del árbol de búsqueda con UCB1.
    - MCTS_Agent     : Motor MCTS con los 4 pasos clásicos.
    - Backtester     : Evaluación: ROI acumulado, Ratio de Sharpe, visualización.
    - download_data  : Descarga de datos históricos con yfinance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

C_UCB: float = math.sqrt(2)   # Constante de exploración UCB1 (√2 es el estándar)
ACTIONS_ALL: list["Action"] = []  # Se rellena tras definir el Enum


# ---------------------------------------------------------------------------
# 1. Acciones discretas
# ---------------------------------------------------------------------------

class Action(Enum):
    """Acciones disponibles en cada paso de decisión del agente."""
    BUY_10  = auto()   # Comprar con el 10 % del efectivo disponible
    SELL_10 = auto()   # Vender el 10 % de las acciones en propiedad
    HOLD    = auto()   # Mantener posición sin operar


ACTIONS_ALL = list(Action)


# ---------------------------------------------------------------------------
# 2. Estado del entorno
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    """
    Representa el estado completo del entorno en un instante dado.

    Attributes
    ----------
    day    : Índice del día actual dentro de la serie de precios.
    cash   : Efectivo disponible en moneda base (USD, EUR, etc.).
    shares : Número de acciones en propiedad (puede ser fraccionario).
    price  : Precio de cierre del activo en este día.
    """
    day:    int
    cash:   float
    shares: float
    price:  float

    def portfolio_value(self) -> float:
        """Valor total de la cartera: efectivo + valor de mercado de las acciones."""
        return self.cash + self.shares * self.price

    def apply_action(self, action: Action, next_price: float) -> "State":
        """
        Aplica una acción y devuelve el *nuevo* estado resultante (inmutable).

        Parameters
        ----------
        action     : Acción a ejecutar (BUY_10, SELL_10 o HOLD).
        next_price : Precio del activo en el siguiente paso de tiempo.

        Returns
        -------
        State : Estado resultante tras la acción.

        Notes
        -----
        - No se modelan comisiones aquí; se pueden añadir como parámetro.
        - Las acciones son "best-effort": si no hay efectivo para comprar o
          acciones para vender, equivalen a HOLD.
        """
        cash, shares = self.cash, self.shares

        if action == Action.BUY_10:
            amount_to_invest = cash * 0.10
            shares_bought    = amount_to_invest / self.price if self.price > 0 else 0.0
            cash   -= amount_to_invest
            shares += shares_bought

        elif action == Action.SELL_10:
            shares_to_sell = shares * 0.10
            cash   += shares_to_sell * self.price
            shares -= shares_to_sell

        # HOLD: sin cambios en efectivo ni acciones

        return State(
            day=self.day + 1,
            cash=cash,
            shares=shares,
            price=next_price,
        )


# ---------------------------------------------------------------------------
# 3. Nodo del árbol
# ---------------------------------------------------------------------------

class Node:
    """
    Nodo del árbol de búsqueda MCTS.

    Attributes
    ----------
    state    : Estado del entorno asociado a este nodo.
action   : Acción que llevó al padre a este nodo (None para la raíz).
    parent   : Nodo padre (None para la raíz).
    children : Lista de nodos hijo ya expandidos.
    n        : Número de veces que este nodo ha sido visitado.
    w        : Recompensa total acumulada en todas las simulaciones que
               pasaron por este nodo.
    """

    def __init__(
        self,
        state:  State,
        action: Optional[Action] = None,
        parent: Optional["Node"] = None,
    ) -> None:
        self.state:    State          = state
        self.action:   Optional[Action] = action
        self.parent:   Optional["Node"] = parent
        self.children: list["Node"]   = []
        self.n:        int            = 0
        self.w:        float          = 0.0

    # ------------------------------------------------------------------
    # UCB1 / UCT
    # ------------------------------------------------------------------

    def ucb1(self, c: float = C_UCB) -> float:
        """
        Calcula el valor UCT (Upper Confidence Bound for Trees) del nodo.

        Fórmula
        -------
            UCT = w_i / n_i  +  c * sqrt(ln(N_i) / n_i)

        donde N_i es el número de visitas del nodo *padre*.

        Returns
        -------
        float : Valor UCT. Devuelve +∞ si el nodo no ha sido visitado (n=0)
                para garantizar que todo nodo sea explorado al menos una vez.
        """
        if self.n == 0:
            return math.inf

        if self.parent is None or self.parent.n == 0:
            exploitation = self.w / self.n
            return exploitation

        exploitation = self.w / self.n
        exploration  = c * math.sqrt(math.log(self.parent.n) / self.n)
        return exploitation + exploration

    # ------------------------------------------------------------------
    # Expansión
    # ------------------------------------------------------------------

    def tried_actions(self) -> set[Action]:
        """Devuelve el conjunto de acciones ya expandidas como hijos."""
        return {child.action for child in self.children if child.action is not None}

    def untried_actions(self) -> list[Action]:
        """Devuelve las acciones aún no exploradas desde este nodo."""
        tried = self.tried_actions()
        return [a for a in ACTIONS_ALL if a not in tried]

    def is_fully_expanded(self) -> bool:
        """True si todas las acciones posibles ya tienen un nodo hijo."""
        return len(self.untried_actions()) == 0

    def best_child(self, c: float = C_UCB) -> "Node":
        """
        Devuelve el hijo con el mayor valor UCT.

        Parameters
        ----------
        c : Constante de exploración. Usar c=0 para explotación pura
            (elegir la mejor acción conocida al final del árbol).
        """
        return max(self.children, key=lambda child: child.ucb1(c))

    def add_child(self, action: Action, state: State) -> "Node":
        """Crea y registra un nodo hijo para la acción dada."""
        child = Node(state=state, action=action, parent=self)
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        action_name = self.action.name if self.action else "ROOT"
        return (
            f"Node(action={action_name}, day={self.state.day}, "
            f"n={self.n}, w={self.w:.2f}, "
            f"portfolio={self.state.portfolio_value():.2f})"
        )


# ---------------------------------------------------------------------------
# 4. Motor MCTS
# ---------------------------------------------------------------------------

class MCTS_Agent:
    """
    Agente de Monte Carlo Tree Search para decisiones de trading.

    Implementa los 4 pasos del algoritmo:
        1. Selección   — navegar el árbol con UCB1 hasta un nodo hoja.
        2. Expansión   — añadir un nuevo hijo no explorado.
        3. Rollout     — simular el futuro con Movimiento Browniano Geométrico.
        4. Backprop    — propagar la recompensa hacia la raíz.

    Parameters
    ----------
    prices       : Array de precios históricos de cierre (numpy).
    iterations   : Número de iteraciones MCTS por paso de decisión.
    rollout_days : Horizonte de simulación en el rollout (en días).
    c_ucb        : Constante de exploración UCB1 (default √2).
    rng_seed     : Semilla para reproducibilidad.
    """

    def __init__(
        self,
        prices:       np.ndarray,
        iterations:   int   = 200,
        rollout_days: int   = 10,
        c_ucb:        float = C_UCB,
        rng_seed:     int   = 42,
    ) -> None:
        self.prices       = prices
        self.iterations   = iterations
        self.rollout_days = rollout_days
        self.c_ucb        = c_ucb
        self.rng           = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Paso 1: Selección
    # ------------------------------------------------------------------

    def _select(self, node: Node) -> Node:
        """
        Navega el árbol desde `node` hacia abajo usando UCB1 hasta
        encontrar un nodo hoja (no expandido completamente) o terminal.

        Returns
        -------
        Node : Nodo hoja seleccionado para expansión.
        """
        current = node
        while current.is_fully_expanded() and current.children:
            current = current.best_child(self.c_ucb)
        return current

    # ------------------------------------------------------------------
    # Paso 2: Expansión
    # ------------------------------------------------------------------

    def _expand(self, node: Node) -> Node:
        """
        Escoge una acción no probada y crea el nodo hijo correspondiente.

        El precio del siguiente día se toma directamente de la serie
        histórica cuando está disponible.

        Returns
        -------
        Node : El nuevo nodo hijo creado.
        """
        untried = node.untried_actions()
        action  = self.rng.choice(untried)  # type: ignore[arg-type]

        current_day = node.state.day
        # Si hay precio histórico real lo usamos; si no, usamos el actual
        if current_day + 1 < len(self.prices):
            next_price = float(self.prices[current_day + 1])
        else:
            next_price = node.state.price

        new_state = node.state.apply_action(action, next_price)
        return node.add_child(action, new_state)

    # ------------------------------------------------------------------
    # Paso 3: Simulación estocástica (Rollout con GBM)
    # ------------------------------------------------------------------

    def _rollout(self, node: Node) -> float:
        """
        Evalúa el nodo simulando el precio futuro con Movimiento Browniano
        Geométrico (GBM) — solución exacta sin error de discretización.

        Modelo GBM (Black-Scholes):
            S_T = S_0 * exp((μ - σ²/2) * T  +  σ * √T * Z)
            Z ~ N(0, 1)

        donde μ y σ se estiman con las log-rentabilidades de los últimos
        20 días de la serie histórica (calibración rolling).

        Returns
        -------
        float : Valor de la cartera al final del horizonte de simulación.

        Notes
        -----
        - Se simulan N_PATHS trayectorias en paralelo y se devuelve la media
          para reducir la varianza del estimador.
        - T se mide en años (días / 252).
        """
        N_PATHS = 50  # Número de trayectorias paralelas para el estimador Monte Carlo

        state = node.state
        S0    = state.price
        day   = state.day

        # --- Calibración de μ y σ con ventana rolling de 20 días ---
        window_start = max(0, day - 20)
        window_end   = min(day + 1, len(self.prices))
        window_prices = self.prices[window_start:window_end]

        if len(window_prices) >= 3:
            log_returns = np.diff(np.log(window_prices))
            mu_daily    = float(np.mean(log_returns))
            sigma_daily = float(np.std(log_returns, ddof=1)) + 1e-8  # evitar σ=0
        else:
            # Fallback conservador si no hay datos suficientes
            mu_daily, sigma_daily = 0.0, 0.01

        # --- Solución exacta GBM (sin discretización) ---
        T = self.rollout_days / 252.0          # Horizonte en años
        mu_annual    = mu_daily * 252
        sigma_annual = sigma_daily * math.sqrt(252)

        Z   = self.rng.standard_normal(N_PATHS)  # Z ~ N(0,1), vectorizado
        S_T = S0 * np.exp(
            (mu_annual - 0.5 * sigma_annual ** 2) * T
            + sigma_annual * math.sqrt(T) * Z
        )

        # --- Aplicar acción del nodo y calcular valor medio de la cartera ---
        portfolio_values = state.cash + state.shares * S_T
        return float(np.mean(portfolio_values))

    # ------------------------------------------------------------------
    # Paso 4: Retropropagación
    # ------------------------------------------------------------------

    def _backpropagate(self, node: Node, reward: float) -> None:
        """
        Propaga la recompensa obtenida en el rollout hacia la raíz,
        actualizando `w` (recompensa total) y `n` (visitas) en cada nodo.

        Parameters
        ----------
        node   : Nodo desde el que comenzar a subir.
        reward : Valor de la cartera obtenido en la simulación.
        """
        current: Optional[Node] = node
        while current is not None:
            current.n += 1
            current.w += reward
            current = current.parent

    # ------------------------------------------------------------------
    # Bucle principal MCTS
    # ------------------------------------------------------------------

    def run(self, root_state: State) -> Node:
        """
        Ejecuta el algoritmo MCTS durante `self.iterations` iteraciones
        y devuelve el nodo raíz con el árbol construido.

        Parameters
        ----------
        root_state : Estado inicial desde el que construir el árbol.

        Returns
        -------
        Node : Raíz del árbol de decisiones tras las iteraciones.
        """
        root = Node(state=root_state)

        for _ in range(self.iterations):
            # 1. Selección
            leaf = self._select(root)

            # 2. Expansión (si el nodo no está completamente expandido)
            if not leaf.is_fully_expanded():
                leaf = self._expand(leaf)

            # 3. Simulación (rollout estocástico)
            reward = self._rollout(leaf)

            # 4. Retropropagación
            self._backpropagate(leaf, reward)

        return root

    def best_action(self, root: Node) -> Action:
        """
        Devuelve la acción recomendada tras construir el árbol:
        el hijo con mayor tasa de recompensa media (explotación pura, c=0).

        Parameters
        ----------
        root : Nodo raíz devuelto por `run()`.

        Returns
        -------
        Action : Acción óptima según el árbol.
        """
        if not root.children:
            return Action.HOLD
        best = root.best_child(c=0.0)
        return best.action if best.action is not None else Action.HOLD


# ---------------------------------------------------------------------------
# 5. Backtesting y evaluación
# ---------------------------------------------------------------------------

class Backtester:
    """
    Evalúa el desempeño del MCTS_Agent sobre datos históricos.

    Métricas calculadas:
        - Rendimiento Acumulado (ROI) del agente y de Buy & Hold.
        - Ratio de Sharpe anualizado.
        - Visualización de la ruta de decisiones.
    """

    def __init__(
        self,
        mcts_iterations: int   = 100,
        rollout_days:    int   = 5,
        c_ucb:           float = C_UCB,
        rng_seed:        int   = 42,
    ) -> None:
        self.mcts_iterations = mcts_iterations
        self.rollout_days    = rollout_days
        self.c_ucb           = c_ucb
        self.rng_seed        = rng_seed

    # ------------------------------------------------------------------

    def run_agent(
        self,
        prices:        np.ndarray,
        initial_cash:  float = 10_000.0,
    ) -> tuple[list[State], list[Action]]:
        """
        Ejecuta el agente MCTS sobre toda la serie de precios, tomando
        una decisión en cada día.

        Parameters
        ----------
        prices       : Array de precios de cierre históricos.
        initial_cash : Capital inicial en la misma moneda que los precios.

        Returns
        -------
        tuple[list[State], list[Action]] :
            - Historia de estados visitados por el agente.
            - Lista de acciones tomadas en cada paso.
        """
        agent = MCTS_Agent(
            prices       = prices,
            iterations   = self.mcts_iterations,
            rollout_days = self.rollout_days,
            c_ucb        = self.c_ucb,
            rng_seed     = self.rng_seed,
        )

        state   = State(day=0, cash=initial_cash, shares=0.0, price=float(prices[0]))
        history = [state]
        actions: list[Action] = []

        for day in range(len(prices) - 1):
            state  = State(
                day=day,
                cash=history[-1].cash,
                shares=history[-1].shares,
                price=float(prices[day]),
            )
            root   = agent.run(state)
            action = agent.best_action(root)
            actions.append(action)

            next_price = float(prices[day + 1])
            state      = state.apply_action(action, next_price)
            history.append(state)
            print(f"\r[INFO] Día {day + 1}/{len(prices) - 1}  "
                  f"Acción: {action.name:<8}  "
                  f"Cartera: {state.portfolio_value():,.2f}", end="", flush=True)

        print()  # salto de línea tras el progreso
        return history, actions

    # ------------------------------------------------------------------

    @staticmethod
    def buy_and_hold(prices: np.ndarray, initial_cash: float = 10_000.0) -> float:
        """
        Calcula el valor final de una estrategia pasiva Buy & Hold:
        invertir todo el capital en el primer día y mantener.

        Returns
        -------
        float : Valor final de la cartera Buy & Hold.
        """
        shares_bought = initial_cash / float(prices[0])
        return shares_bought * float(prices[-1])

    # ------------------------------------------------------------------

    @staticmethod
    def cumulative_roi(history: list[State], bah_value: float) -> dict:
        """
        Calcula el Rendimiento Acumulado (ROI) del agente y lo compara
        con Buy & Hold.

        Parameters
        ----------
        history   : Historia de estados devuelta por `run_agent`.
        bah_value : Valor final de Buy & Hold.

        Returns
        -------
        dict con claves:
            - 'initial_value'     : Capital inicial.
            - 'final_value_agent' : Valor final del agente MCTS.
            - 'final_value_bah'   : Valor final de Buy & Hold.
            - 'roi_agent'         : ROI del agente (fracción).
            - 'roi_bah'           : ROI de Buy & Hold (fracción).
            - 'portfolio_series'  : Serie temporal de valores de cartera.
        """
        initial = history[0].portfolio_value()
        final   = history[-1].portfolio_value()

        return {
            "initial_value":      initial,
            "final_value_agent":  final,
            "final_value_bah":    bah_value,
            "roi_agent":          (final - initial) / initial,
            "roi_bah":            (bah_value - initial) / initial,
            "portfolio_series":   [s.portfolio_value() for s in history],
        }

    # ------------------------------------------------------------------

    @staticmethod
    def sharpe_ratio(
        portfolio_series: list[float],
        risk_free:        float = 0.0,
        periods_per_year: int   = 252,
    ) -> float:
        """
        Calcula el Ratio de Sharpe anualizado.

        Fórmula
        -------
            S_a = E[R_a - R_b] / σ_a  * √(periods_per_year)

        donde R_a son las rentabilidades diarias del agente y R_b es la
        tasa libre de riesgo diaria.

        Parameters
        ----------
        portfolio_series  : Serie de valores de la cartera (un valor por día).
        risk_free         : Tasa libre de riesgo *anual* (default 0).
        periods_per_year  : Factor de anualización (252 para días bursátiles).

        Returns
        -------
        float : Ratio de Sharpe anualizado. Devuelve NaN si σ = 0.
        """
        values  = np.array(portfolio_series, dtype=float)
        returns = np.diff(values) / values[:-1]

        daily_rf      = risk_free / periods_per_year
        excess_returns = returns - daily_rf

        sigma = float(np.std(excess_returns, ddof=1))
        if sigma == 0:
            return float("nan")

        return float(np.mean(excess_returns) / sigma * math.sqrt(periods_per_year))

    # ------------------------------------------------------------------

    @staticmethod
    def plot_decision_tree(root: Node, max_depth: int = 3) -> None:
        """
        Visualiza el árbol de decisiones construido por MCTS con matplotlib.

        Se muestra hasta `max_depth` niveles desde la raíz.
        Cada nodo indica la acción, las visitas (n) y la recompensa media.

        Parameters
        ----------
        root      : Nodo raíz del árbol MCTS.
        max_depth : Profundidad máxima a visualizar.
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        ax.set_title("Árbol de Decisiones MCTS", fontsize=14, fontweight="bold")

        ACTION_COLORS = {
            Action.BUY_10:  "#2ecc71",   # verde
            Action.SELL_10: "#e74c3c",   # rojo
            Action.HOLD:    "#3498db",   # azul
            None:           "#95a5a6",   # gris (raíz)
        }

        def _draw(node: Node, x: float, y: float, dx: float, depth: int) -> None:
            if depth > max_depth:
                return

            mean_reward = node.w / node.n if node.n > 0 else 0.0
            label = (
                f"{node.action.name if node.action else 'ROOT'}\n"
                f"n={node.n}  w̄={mean_reward:.0f}"
            )
            color = ACTION_COLORS.get(node.action, "#95a5a6")

            ax.text(
                x, y, label,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8),
                fontsize=8,
            )

            n_children = len(node.children)
            if n_children == 0:
                return

            child_dx = dx / max(n_children, 1)
            x_start  = x - dx / 2 + child_dx / 2

            for i, child in enumerate(node.children):
                cx = x_start + i * child_dx
                cy = y - 1.5
                ax.annotate(
                    "", xy=(cx, cy + 0.35), xytext=(x, y - 0.35),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
                )
                _draw(child, cx, cy, child_dx, depth + 1)

        _draw(root, x=0.5, y=max_depth * 1.5, dx=1.0, depth=0)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(roi_dict: dict, sharpe: float) -> None:
        """Imprime un resumen legible de los resultados del backtest."""
        sep = "=" * 50
        print(sep)
        print("  RESUMEN DEL BACKTEST — MCTS TRADING")
        print(sep)
        print(f"  Capital inicial    : {roi_dict['initial_value']:>12,.2f}")
        print(f"  Valor final MCTS   : {roi_dict['final_value_agent']:>12,.2f}")
        print(f"  Valor final B&H    : {roi_dict['final_value_bah']:>12,.2f}")
        print(f"  ROI agente MCTS    : {roi_dict['roi_agent']:>+11.2%}")
        print(f"  ROI Buy & Hold     : {roi_dict['roi_bah']:>+11.2%}")
        print(f"  Ratio de Sharpe    : {sharpe:>12.4f}")
        print(sep)

    @staticmethod
    def save_chart(
        prices:           np.ndarray,
        portfolio_series: list[float],
        actions:          list[Action],
        bah_value:        float,
        initial_cash:     float,
        ticker:           str,
        output_path:      str,
    ) -> None:
        """
        Genera y guarda un gráfico de tres paneles:
          1. Evolución del precio del activo con marcadores de acción.
          2. Valor de la cartera MCTS vs Buy & Hold normalizado.
          3. Distribución de acciones tomadas (pie chart).
        """
        import matplotlib.gridspec as gridspec
        from matplotlib.lines import Line2D

        days = range(len(prices))

        # Construir serie Buy & Hold normalizada
        shares_bah  = initial_cash / prices[0]
        bah_series  = [shares_bah * p for p in prices]

        # Separar días por acción para los marcadores
        buy_days  = [i for i, a in enumerate(actions) if a == Action.BUY_10]
        sell_days = [i for i, a in enumerate(actions) if a == Action.SELL_10]

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"MCTS Trading — Backtest {ticker} (1 año)\n"
            f"Iteraciones MCTS: 200 · Horizonte rollout: 10 días · Capital inicial: {initial_cash:,.0f} $",
            fontsize=13, fontweight="bold", y=0.98,
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

        # ── Panel 1: Precio + señales ───────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(days, prices, color="#2c3e50", lw=1.5, label=f"Precio {ticker}")
        ax1.scatter(buy_days,  prices[buy_days],  color="#2ecc71", s=25, zorder=5, label="BUY 10%")
        ax1.scatter(sell_days, prices[sell_days], color="#e74c3c", s=25, zorder=5, label="SELL 10%")
        ax1.set_title("Precio del activo y señales de decisión MCTS", fontsize=11)
        ax1.set_xlabel("Día de trading")
        ax1.set_ylabel(f"Precio ({ticker}) [$]")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # ── Panel 2: Cartera MCTS vs B&H ───────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(range(len(portfolio_series)), portfolio_series,
                 color="#8e44ad", lw=2, label="MCTS Agent")
        ax2.plot(days, bah_series,
                 color="#e67e22", lw=2, linestyle="--", label="Buy & Hold")
        ax2.axhline(initial_cash, color="gray", lw=1, linestyle=":", label="Capital inicial")
        ax2.set_title("Valor de la cartera", fontsize=11)
        ax2.set_xlabel("Día de trading")
        ax2.set_ylabel("Valor [$]")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # ── Panel 3: Distribución de acciones ──────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        action_counts = {
            "BUY 10%":   sum(1 for a in actions if a == Action.BUY_10),
            "SELL 10%":  sum(1 for a in actions if a == Action.SELL_10),
            "HOLD":      sum(1 for a in actions if a == Action.HOLD),
        }
        colors_pie = ["#2ecc71", "#e74c3c", "#3498db"]
        wedges, texts, autotexts = ax3.pie(
            action_counts.values(),
            labels=action_counts.keys(),
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax3.set_title("Distribución de acciones MCTS", fontsize=11)

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Gráfico guardado en: {output_path}")

    @staticmethod
    def save_report(
        roi_dict:     dict,
        sharpe:       float,
        actions:      list[Action],
        ticker:       str,
        period:       str,
        n_days:       int,
        iterations:   int,
        rollout_days: int,
        output_path:  str,
    ) -> None:
        """Genera y guarda un documento Markdown con los resultados del backtest."""
        import datetime

        action_counts = {
            "BUY_10":  sum(1 for a in actions if a == Action.BUY_10),
            "SELL_10": sum(1 for a in actions if a == Action.SELL_10),
            "HOLD":    sum(1 for a in actions if a == Action.HOLD),
        }
        total_ops = len(actions)

        lines = [
            "# Resultados del Backtest — MCTS Trading",
            "",
            f"> Generado el {datetime.date.today().isoformat()}",
            "",
            "## Configuración del experimento",
            "",
            f"| Parámetro            | Valor          |",
            f"|----------------------|----------------|",
            f"| Activo (ticker)      | `{ticker}`     |",
            f"| Período              | {period}       |",
            f"| Sesiones analizadas  | {n_days} días  |",
            f"| Iteraciones MCTS     | {iterations}   |",
            f"| Horizonte rollout    | {rollout_days} días |",
            f"| Capital inicial      | {roi_dict['initial_value']:,.2f} $ |",
            "",
            "## Resultados financieros",
            "",
            f"| Métrica              | MCTS Agent     | Buy & Hold     |",
            f"|----------------------|----------------|----------------|",
            f"| Valor final [$]      | {roi_dict['final_value_agent']:>14,.2f} | {roi_dict['final_value_bah']:>14,.2f} |",
            f"| ROI acumulado        | {roi_dict['roi_agent']:>+13.2%} | {roi_dict['roi_bah']:>+13.2%} |",
            f"| Ratio de Sharpe      | {sharpe:>14.4f} | —              |",
            "",
            "## Distribución de decisiones MCTS",
            "",
            f"| Acción   | Veces | % del total |",
            f"|----------|-------|-------------|",
            f"| BUY 10%  | {action_counts['BUY_10']:>5} | {action_counts['BUY_10']/total_ops:>10.1%} |",
            f"| SELL 10% | {action_counts['SELL_10']:>5} | {action_counts['SELL_10']/total_ops:>10.1%} |",
            f"| HOLD     | {action_counts['HOLD']:>5} | {action_counts['HOLD']/total_ops:>10.1%} |",
            "",
            "## Notas metodológicas",
            "",
            "- **Modelo estocástico:** Movimiento Browniano Geométrico (GBM) con solución exacta.",
            "- **Calibración:** μ y σ estimados con ventana rolling de 20 días de log-rentabilidades.",
            "- **Exploración UCB1:** constante c = √2 (estándar teórico).",
            "- **Rollout:** media de 50 trayectorias GBM paralelas por simulación.",
            "- **Reproducibilidad:** semilla fija `rng_seed=42` en todo el experimento.",
            "",
            "## Gráfico",
            "",
            "![Backtest](backtest_chart.png)",
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[INFO] Informe guardado en: {output_path}")


# ---------------------------------------------------------------------------
# 6. Descarga de datos con yfinance
# ---------------------------------------------------------------------------

def download_data(ticker: str = "TSLA", period: str = "1y") -> np.ndarray:
    """
    Descarga los precios de cierre ajustados de un activo usando yfinance.

    Parameters
    ----------
    ticker : Símbolo del activo (ej. "TSLA", "AAPL", "BTC-USD").
    period : Período de descarga en formato yfinance (ej. "1y", "6mo", "2y").

    Returns
    -------
    np.ndarray : Array 1-D con los precios de cierre en orden cronológico.

    Raises
    ------
    ValueError : Si no se pudieron descargar datos para el ticker indicado.
    """
    print(f"[INFO] Descargando datos de {ticker} ({period})...")
    df: pd.DataFrame = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No se encontraron datos para el ticker '{ticker}'.")

    close_col = "Close"
    prices = df[close_col].dropna().to_numpy(dtype=float).flatten()
    print(f"[INFO] {len(prices)} sesiones descargadas. "
          f"Primer precio: {prices[0]:.2f}  Último: {prices[-1]:.2f}")
    return prices


# ---------------------------------------------------------------------------
# 7. Demo principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    TICKER       = "TSLA"
    PERIOD       = "1y"
    INITIAL_CASH = 10_000.0
    ITERATIONS   = 200
    ROLLOUT_DAYS = 10
    RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Descarga de datos (año completo) ---
    prices = download_data(ticker=TICKER, period=PERIOD)

    # --- Configuración del Backtester ---
    backtester = Backtester(
        mcts_iterations=ITERATIONS,
        rollout_days=ROLLOUT_DAYS,
        c_ucb=C_UCB,
        rng_seed=42,
    )

    # --- Backtest sobre el año completo ---
    print("[INFO] Ejecutando backtest anual con agente MCTS...")
    history, actions = backtester.run_agent(prices, initial_cash=INITIAL_CASH)

    # --- Métricas ---
    bah_value = Backtester.buy_and_hold(prices, INITIAL_CASH)
    roi_dict  = Backtester.cumulative_roi(history, bah_value)
    sharpe    = Backtester.sharpe_ratio(roi_dict["portfolio_series"])

    Backtester.print_summary(roi_dict, sharpe)

    # --- Guardar gráfico ---
    chart_path = os.path.join(RESULTS_DIR, "backtest_chart.png")
    Backtester.save_chart(
        prices           = prices,
        portfolio_series = roi_dict["portfolio_series"],
        actions          = actions,
        bah_value        = bah_value,
        initial_cash     = INITIAL_CASH,
        ticker           = TICKER,
        output_path      = chart_path,
    )

    # --- Guardar informe Markdown ---
    report_path = os.path.join(RESULTS_DIR, "backtest_report.md")
    Backtester.save_report(
        roi_dict     = roi_dict,
        sharpe       = sharpe,
        actions      = actions,
        ticker       = TICKER,
        period       = PERIOD,
        n_days       = len(prices),
        iterations   = ITERATIONS,
        rollout_days = ROLLOUT_DAYS,
        output_path  = report_path,
    )

    print(f"\n[OK] Archivos guardados en: {RESULTS_DIR}/")
