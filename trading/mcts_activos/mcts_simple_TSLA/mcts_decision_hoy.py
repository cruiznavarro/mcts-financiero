"""
mcts_decision_hoy.py
====================
Herramienta de apoyo a la decisión de trading para ANTES de la apertura.

Descarga los datos históricos más recientes, inicializa el estado con
la cartera REAL del usuario y ejecuta MCTS una sola vez para recomendar
la mejor acción del día (comprar, vender o mantener).

Uso
---
    1. Ajusta TU_EFECTIVO y TUS_ACCIONES con tu situación real.
    2. Ejecuta el script antes de la apertura del mercado.
    3. Lee el ranking de acciones y la recomendación final.

Dependencias: numpy, matplotlib, yfinance
(todas presentes si ya funciona mcts_simple.py)

──────────────────────────────────────────────────────────────────────────────
FUNDAMENTOS TEÓRICOS: MCTS + UCT
──────────────────────────────────────────────────────────────────────────────
Monte Carlo Tree Search (MCTS) es un algoritmo de búsqueda en árbol que
combina búsqueda heurística con simulación aleatoria (Coulom 2006, Kocsis &
Szepesvári 2006). En cada iteración ejecuta cuatro fases:

  1. SELECCIÓN   — Desde la raíz, desciende por el árbol eligiendo hijos
                   con la política UCT (Upper Confidence Bounds for Trees).

  2. EXPANSIÓN   — Al llegar a un nodo hoja no completamente explorado,
                   añade uno o varios nodos nuevos al árbol.

  3. SIMULACIÓN  — Desde el nuevo nodo, ejecuta un rollout aleatorio
                   (simulación de Monte Carlo de trayectorias de precio)
                   hasta el horizonte temporal definido por DIAS_ROLLOUT.

  4. RETROPROPAGACIÓN — El resultado del rollout sube por el árbol
                   actualizando la recompensa acumulada Q y el contador N
                   de visitas de cada nodo ancestro.

La política UCT para la fase de SELECCIÓN es:

        UCT(i) = Q(i)/N(i)  +  C · √( ln N(padre) / N(i) )
                 ───────────    ────────────────────────────
                 explotación          exploración

  · Q(i)      : recompensa acumulada del nodo i  (suma de log-retornos)
  · N(i)      : número de visitas al nodo i
  · N(padre)  : visitas al nodo padre
  · C         : constante de exploración (C = √2 en UCB1, Auer et al. 2002)

El término exploratorio √(ln N_padre / N(i)) tiende a ∞ cuando N(i)→0,
garantizando que todo nodo no visitado se expanda antes de reincidir en
nodos ya conocidos. Con suficientes iteraciones, Q(a)/N(a) converge al
valor esperado real de cada acción (propiedad de consistencia de UCT).

En este script el ÁRBOL representa:
  · NODO RAÍZ   : estado real de la cartera hoy (efectivo, acciones, RSI, MA20)
  · RAMAS       : acciones posibles del espacio A (COMPRAR_X%, VENDER_Y%, HOLD)
  · NODOS HIJOS : estados resultantes de aplicar cada acción sobre la raíz
  · RECOMPENSA  : log-retorno del portafolio relativo a Buy & Hold en el rollout
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

# Añadimos la carpeta del programa base al path para reutilizar sus funciones
sys.path.insert(0, os.path.dirname(__file__))

from mcts_simple import (
    descargar_precios,          # obtiene la serie temporal P[0..T] del activo
    crear_estado,               # construye el dict que representa un NODO del árbol MCTS
                                #   nodo = { dia, efectivo, acciones, precio, rsi, ma20 }
    valor_cartera,              # V(s) = efectivo + acciones × precio  (función de valor)
    ejecutar_mcts,              # núcleo MCTS: selección UCT → expansión → rollout → backprop
    graficar_convergencia_ucb,  # traza Q(a)/N(a) vs. iteración para cada acción
    TICKER,                     # símbolo del activo (TSLA)
    PERIOD,                     # ventana histórica descargada (p. ej. "6mo")
    SEMILLA,                    # semilla RNG → reproducibilidad de las simulaciones MC
    ACCIONES,                   # espacio de acciones completo A (antes de filtrar)
    ITERACIONES,                # K = número de iteraciones MCTS (profundidad estadística)
    DIAS_ROLLOUT,               # H = horizonte de cada simulación Monte Carlo (días)
    VENTANA_CALIB,              # ventana para estimar μ y σ del GBM en los rollouts
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# PARÁMETROS DE TU CARTERA REAL
# Modifica estos valores antes de ejecutar.
# =============================================================================

TU_EFECTIVO   = 100.0    # dinero disponible en cuenta (USD)
TUS_ACCIONES  = 0.0      # número de acciones de TSLA en cartera


# =============================================================================
# FUNCIÓN PRINCIPAL: DECISIÓN DE HOY
# =============================================================================

def decision_hoy(efectivo: float, num_acciones: float) -> None:
    """
    Ejecuta MCTS una sola vez sobre el estado real actual y recomienda
    la mejor acción para la sesión de hoy.

    Parámetros
    ----------
    efectivo     : Dinero disponible en cuenta (USD).
    num_acciones : Número de acciones del activo en cartera.

    ──────────────────────────────────────────────────────────────────────
    FLUJO MCTS EN ESTA FUNCIÓN
    ──────────────────────────────────────────────────────────────────────
    Paso 1 — Construir el NODO RAÍZ s₀ con el estado real de hoy.
             s₀ encapsula toda la información observable: precio, efectivo,
             acciones, RSI(14) y MA(20). Es el punto de partida del árbol.

    Paso 2 — Llamar a ejecutar_mcts(s₀), que repite K=ITERACIONES veces:
               a) Selección  : baja por el árbol usando UCT(i) hasta un nodo
                               hoja o un nodo no completamente expandido.
               b) Expansión  : crea un nodo hijo para una acción no probada.
               c) Rollout    : simula H=DIAS_ROLLOUT días de precio con GBM
                               calibrado sobre los últimos VENTANA_CALIB días.
                               reward = log(V_final/V_inicial) − log(P_final/P_inicial)
                               (retorno relativo al benchmark Buy & Hold)
               d) Backprop   : para cada nodo en la ruta raíz → hoja:
                               Q(nodo) += reward ;  N(nodo) += 1

    Paso 3 — Decisión final:  a* = argmax_a  Q(a) / N(a)
             (acción con mayor recompensa media estimada entre los hijos
             directos de la raíz, tras K simulaciones Monte Carlo)
    ──────────────────────────────────────────────────────────────────────
    """

    # ── 1. Descargar datos históricos hasta ayer ───────────────────────────
    # precios[t] = precio de cierre del día t, con t = 0..T (T = ayer).
    # Esta serie cumple dos roles en MCTS:
    #   a) CALIBRACIÓN del proceso de precios para los rollouts:
    #      μ = media de log-retornos diarios  (drift del GBM)
    #      σ = desv. estándar de log-retornos (volatilidad del GBM)
    #      Ambos se estiman sobre los últimos VENTANA_CALIB días.
    #   b) INICIALIZACIÓN del estado raíz: RSI(14) y MA(20) se calculan
    #      sobre los últimos 14/20 días de esta serie.
    precios = descargar_precios(TICKER, PERIOD)
    dia_hoy = len(precios) - 1   # índice del último día disponible (ayer)

    # ── 2. Construir el estado raíz s₀ del árbol MCTS ─────────────────────
    # El nodo raíz contiene la observación completa del entorno:
    #   s₀ = { dia, efectivo, acciones, precio, rsi, ma20 }
    # En la formulación MDP subyacente al MCTS:
    #   · s₀ es el estado inicial del proceso de decisión
    #   · Las acciones A son las transiciones posibles desde s₀
    #   · La función de recompensa r(s,a,s') se estima con los rollouts GBM
    estado = crear_estado(
        dia          = dia_hoy,
        efectivo     = efectivo,
        acciones     = num_acciones,
        precio       = float(precios[dia_hoy]),
        precios_hist = precios,
    )

    valor_actual = valor_cartera(estado)      # V(s₀) = efectivo + acciones × precio
    precio_hoy   = float(precios[dia_hoy])

    # ── 3. Resumen del estado actual ───────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  ESTADO DE CARTERA — {TICKER}  (cierre de ayer)")
    print(sep)
    print(f"  Precio cierre ayer  : {precio_hoy:>10.2f} $")
    print(f"  Efectivo disponible : {efectivo:>10,.2f} $")
    print(f"  Acciones en cartera : {num_acciones:>10.2f}")
    print(f"  Valor posición      : {num_acciones * precio_hoy:>10,.2f} $")
    print(f"  Valor total         : {valor_actual:>10,.2f} $")
    print(f"  RSI (14d)           : {estado['rsi']:>10.1f}")
    print(f"  MA20                : {estado['ma20']:>10.2f} $")
    # RSI y MA20 enriquecen el vector de estado del nodo raíz.
    # En un MCTS con política de rollout informada podrían usarse para
    # sesgar el muestreo aleatorio (p. ej., favorecer compras si RSI < 30).
    _interpretar_indicadores(estado, precio_hoy)
    print(sep)

    # ── 4. Ejecutar MCTS ───────────────────────────────────────────────────
    # Núcleo del algoritmo: K iteraciones del ciclo selección–expansión–
    # rollout–backpropagación sobre el árbol con raíz en s₀.
    #
    # Complejidad por iteración: O(|A| + H)
    #   · |A| = tamaño del espacio de acciones (número de ramas)
    #   · H   = DIAS_ROLLOUT (longitud de cada simulación Monte Carlo)
    #
    # La constante de exploración C en UCT controla el trade-off:
    #   · C grande → explora más ramas nuevas antes de profundizar
    #   · C pequeño → explota las ramas ya conocidas como buenas
    # El valor óptimo teórico es C = √2 (Kocsis & Szepesvári 2006),
    # aunque en la práctica se ajusta empíricamente por dominio.
    print(f"\n  Ejecutando MCTS ({ITERACIONES} iteraciones, "
          f"horizonte {DIAS_ROLLOUT} días)...\n")

    # Semilla fija → mismos números aleatorios en cada ejecución del día,
    # garantizando reproducibilidad de la recomendación.
    rng    = np.random.default_rng(SEMILLA)
    accion, historial = ejecutar_mcts(
        estado, precios, rng, registrar_convergencia=True
    )
    # historial = { acción_a: [Q(a)/N(a) tras iter 1, …, tras iter K] }
    # La última entrada de cada lista es la estimación final de la
    # recompensa media de esa acción: E[r | s₀, a] ≈ Q(a)/N(a).
    # Por la ley de grandes números, Q(a)/N(a) → E[r|s₀,a] cuando K→∞.

    # ── 5. Ranking de acciones por recompensa media final ──────────────────
    # Tomamos el valor convergido Q(a)/N(a) al final de las K iteraciones.
    # En AlphaGo/AlphaZero se usa argmax N(a) (más robusto ante outliers);
    # aquí usamos argmax Q(a)/N(a) que es equivalente cuando los rollouts
    # son estacionarios y K es suficientemente grande.
    medias = {a: vals[-1] for a, vals in historial.items() if vals}
    ranking = sorted(medias.items(), key=lambda x: x[1], reverse=True)

    print(f"  {'Acción':<14} {'Reward vs B&H':>14}  {'Señal':>8}")
    print("  " + "-" * 42)
    for a, v in ranking:
        # v = Q(a)/N(a): recompensa media relativa al benchmark Buy & Hold
        # v > 0  →  la acción genera alfa positivo frente a no operar
        # v < 0  →  la acción destruye valor respecto al benchmark pasivo
        barra  = _barra(v, medias)
        marca  = "  ◄ ELEGIDA" if a == accion else ""
        print(f"  {a:<14} {v:>+14.5f}  {barra}{marca}")

    print(f"\n{sep}")
    print(f"  RECOMENDACIÓN PARA HOY: {accion}")
    _interpretar_accion(accion, efectivo, num_acciones, precio_hoy)
    print(f"{sep}\n")

    # ── 6. Gráfico de convergencia UCB ────────────────────────────────────
    # Traza Q(a)/N(a) vs. número de iteración para cada acción.
    # Permite verificar convergencia: si las curvas se estabilizan antes
    # de K iteraciones, el presupuesto computacional es suficiente.
    # Si aún oscilan al final, conviene aumentar ITERACIONES.
    # Tasa teórica de convergencia: error estándar ~ O(1/√N) por CLT →
    # doblar iteraciones reduce la incertidumbre aproximadamente un 30 %.
    graficar_convergencia_ucb(historial)

    # ── 7. Gráfico adicional: ranking visual de acciones ──────────────────
    # Muestra el estado final del árbol MCTS: Q(a)/N(a) para cada acción,
    # ordenado de mejor a peor. La acción a* se resalta con borde naranja.
    _graficar_ranking(ranking, accion)

    import subprocess
    subprocess.Popen(["open",
                      "mcts_convergencia_hoy.png",
                      "mcts_ranking_acciones.png"])


# =============================================================================
# FUNCIONES AUXILIARES DE PRESENTACIÓN
# =============================================================================

def _interpretar_indicadores(estado: dict, precio: float) -> None:
    """
    Imprime una lectura rápida de RSI y MA20.

    Nota MCTS: RSI y MA20 son variables del vector de estado del nodo raíz.
    En un MCTS con política de rollout informada (tree policy no uniforme),
    estos indicadores técnicos podrían sesgar el muestreo aleatorio de
    acciones durante la simulación: por ejemplo, incrementar la probabilidad
    de elegir COMPRAR cuando RSI < 30 (sobreventa) y precio < MA20.
    En la implementación actual el rollout es uniforme (política por defecto).
    """
    rsi  = estado["rsi"]
    ma20 = estado["ma20"]

    señal_rsi = ("sobrecompra — posible corrección"  if rsi > 70
                 else "sobreventa — posible rebote"  if rsi < 30
                 else "zona neutral")
    señal_ma  = ("precio SOBRE MA20 — tendencia alcista" if precio > ma20
                 else "precio BAJO MA20 — tendencia bajista")

    print(f"  RSI → {señal_rsi}")
    print(f"  MA  → {señal_ma}")


def _interpretar_accion(accion: str, efectivo: float,
                         num_acciones: float, precio: float) -> None:
    """
    Traduce la acción elegida a términos concretos de cartera.

    En la nomenclatura MCTS: convierte a* (índice en el espacio discreto A)
    a una orden ejecutable en el mercado, calculando el importe o número de
    acciones según la fracción fija asignada a cada etiqueta de acción.
    Esta conversión no forma parte del árbol MCTS; es solo la interpretación
    de la decisión óptima hallada por el algoritmo.
    """
    from mcts_simple import _FRACCION_ACCION
    # _FRACCION_ACCION: A → [0,1]  mapea cada acción a su fracción de cartera
    #   COMPRAR_25 → 0.25 del efectivo disponible
    #   VENDER_50  → 0.50 de las acciones en cartera
    #   HOLD       → 0.00 (sin transacción)
    fraccion = _FRACCION_ACCION.get(accion, 0.0)

    if accion.startswith("COMPRAR"):
        importe        = efectivo * fraccion
        acciones_comp  = importe / precio if precio > 0 else 0
        print(f"  Comprar {fraccion:.0%} del efectivo disponible")
        print(f"  → Invertir aprox. {importe:,.2f} $ "
              f"({acciones_comp:.4f} acciones a {precio:.2f}$)")
    elif accion.startswith("VENDER"):
        acciones_vend  = num_acciones * fraccion
        importe        = acciones_vend * precio
        print(f"  Vender {fraccion:.0%} de las acciones en cartera")
        print(f"  → Vender aprox. {acciones_vend:.4f} acciones "
              f"(~{importe:,.2f} $)")
    else:
        # HOLD: acción nula → el árbol MCTS no encontró ventaja en operar
        print(f"  No operar hoy. Mantener la cartera sin cambios.")


def _barra(valor: float, todos: dict, ancho: int = 8) -> str:
    """
    Genera una mini barra de texto proporcional al valor relativo.

    Normaliza Q(a)/N(a) al intervalo [0, ancho] para representar
    visualmente la diferencia relativa entre las recompensas medias
    de cada acción. Es solo una ayuda de presentación en terminal;
    no influye en la decisión MCTS.
    """
    vmin = min(todos.values())
    vmax = max(todos.values())
    rango = vmax - vmin if vmax != vmin else 1.0
    proporcion = (valor - vmin) / rango
    llenos = round(proporcion * ancho)
    return "█" * llenos + "░" * (ancho - llenos)


def _graficar_ranking(ranking: list, accion_elegida: str) -> None:
    """
    Genera un gráfico de barras horizontales con el ranking de acciones
    coloreado por tipo (compra = verde, venta = rojo, mantener = azul).

    INTERPRETACIÓN EN TÉRMINOS MCTS:
      · Cada barra representa un nodo hijo directo de la raíz s₀.
      · La longitud de la barra = Q(a)/N(a), la recompensa media estimada
        de aplicar la acción a sobre el estado real de hoy.
      · Eje X = 0 es el benchmark Buy & Hold: valores > 0 indican que la
        acción supera al inversor pasivo en las simulaciones Monte Carlo.
      · La barra con borde naranja es a* = argmax Q(a)/N(a): la acción
        que el árbol MCTS recomienda ejecutar hoy.
      · El orden descendente refleja el ranking inducido por UCT tras K
        iteraciones de exploración–explotación.
    """
    import matplotlib.pyplot as plt
    from mcts_simple import COLORES_ACCION

    acciones = [r[0] for r in ranking]
    valores  = [r[1] for r in ranking]
    colores  = [COLORES_ACCION.get(a, "#95a5a6") for a in acciones]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Ranking MCTS de acciones — {TICKER} (decisión de hoy)\n"
        f"Recompensa media relativa a Buy & Hold tras {ITERACIONES} iteraciones",
        fontsize=11, fontweight="bold",
    )

    bars = ax.barh(acciones, valores, color=colores, edgecolor="white",
                   linewidth=0.6, height=0.65)

    # Resaltar a* (acción elegida por MCTS) con borde naranja grueso
    for bar, a in zip(bars, acciones):
        if a == accion_elegida:
            bar.set_edgecolor("#f39c12")
            bar.set_linewidth(2.5)

    # x = 0: umbral neutro → a la derecha se supera al benchmark B&H
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Recompensa media Q(a)/N(a) relativa a B&H (log-retorno)")
    ax.set_title(f"Acción recomendada: {accion_elegida}",
                 fontsize=10, style="italic", color="#e67e22")

    # Etiquetas numéricas: muestran Q(a)/N(a) con 5 decimales por barra
    for bar, val in zip(bars, valores):
        x = bar.get_width()
        ax.text(x + (0.00005 if x >= 0 else -0.00005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.5f}",
                va="center", ha="left" if x >= 0 else "right",
                fontsize=8)

    ax.invert_yaxis()  # la mejor acción (mayor Q/N) queda en la parte superior
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig("mcts_ranking_acciones.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Gráfico ranking guardado: mcts_ranking_acciones.png")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    decision_hoy(
        efectivo     = TU_EFECTIVO,
        num_acciones = TUS_ACCIONES,
    )
