"""
mcts_decision_TSLA.py
=====================
Herramienta de apoyo a la decisión de trading para ANTES de la apertura.

Descarga los datos históricos más recientes, inicializa el estado con
la cartera REAL del usuario y ejecuta MCTS una sola vez para recomendar
la mejor acción del día (comprar, vender o mantener).

Además registra cada ejecución en un CSV de log para poder evaluar
a posteriori la calidad de las recomendaciones.

Uso manual
----------
    1. Ajusta TU_EFECTIVO y TUS_ACCIONES con tu situación real.
    2. Ejecuta el script antes de la apertura del mercado.
    3. Lee el ranking de acciones y la recomendación final.

Automatización
--------------
    Gestionado por launchd (macOS). Ver com.mcts.tsla.decision.plist.
    Se ejecuta automáticamente cada día laborable a las 9:00 AM.
    Los resultados se guardan en decisiones_TSLA.csv y en los PNGs.

Log CSV
-------
    decisiones_TSLA.csv — una fila por ejecución con:
      fecha, ticker, precio_cierre, rsi_14, ma20,
      accion_recomendada, valor_cartera, reward_mejor, ranking_json

Dependencias: numpy, matplotlib, yfinance
(todas presentes si ya funciona mcts_simple.py)

Gestión del agente launchd
--------------------------
    # Ver si está registrado
    launchctl list | grep mcts

    # Ejecutar manualmente ahora (para probar)
    launchctl start com.mcts.tsla.decision

    # Desactivar
    launchctl unload ~/Library/LaunchAgents/com.mcts.tsla.decision.plist

    # Ver logs
    tail -f /Users/carlosruiznavarro/MonteCarlo/mcts/trading/mcts_tsla/launchd_stdout.log

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

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date


# =============================================================================
# PARÁMETROS DE TU CARTERA REAL
# Modifica estos valores antes de ejecutar.
# =============================================================================

TU_EFECTIVO   = 100.0    # dinero disponible en cuenta (USD)
TUS_ACCIONES  = 0.0      # número de acciones de TSLA en cartera

# Ruta del log CSV — mismo directorio que este script
_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_CSV = os.path.join(_DIR, "decisiones_TSLA.csv")

# Cabecera del CSV (se escribe solo si el archivo no existe aún)
_CSV_CABECERA = [
    "fecha", "ticker", "precio_cierre", "rsi_14", "ma20",
    "accion_recomendada", "valor_cartera", "reward_mejor", "ranking_json",
]


# =============================================================================
# FUNCIÓN DE REGISTRO EN CSV
# =============================================================================

def _registrar_decision(
    fecha: str,
    precio: float,
    rsi: float,
    ma20: float,
    accion: str,
    valor: float,
    ranking: list,
) -> None:
    """
    Añade una fila al CSV de log con la decisión del día.

    Si el archivo no existe lo crea con cabecera.
    Si ya existe una fila para la misma fecha la sobreescribe
    (evita duplicados en caso de reejecutar el mismo día).

    Parámetros
    ----------
    fecha   : Fecha de la decisión (YYYY-MM-DD).
    precio  : Precio de cierre del activo.
    rsi     : RSI(14) calculado sobre la serie histórica.
    ma20    : Media móvil de 20 días.
    accion  : Acción recomendada por MCTS (p. ej. "COMPRAR_25%").
    valor   : Valor total de la cartera en ese momento.
    ranking : Lista de tuplas (accion, Q/N) ordenada de mejor a peor.
    """
    archivo_nuevo = not os.path.exists(LOG_CSV)

    # Leemos filas existentes para detectar duplicado de fecha
    filas = []
    if not archivo_nuevo:
        with open(LOG_CSV, newline="", encoding="utf-8") as f:
            filas = list(csv.DictReader(f))

    # Fila nueva a insertar/actualizar
    reward_mejor = ranking[0][1] if ranking else float("nan")
    fila_nueva = {
        "fecha":             fecha,
        "ticker":            TICKER,
        "precio_cierre":     f"{precio:.4f}",
        "rsi_14":            f"{rsi:.2f}",
        "ma20":              f"{ma20:.4f}",
        "accion_recomendada": accion,
        "valor_cartera":     f"{valor:.4f}",
        "reward_mejor":      f"{reward_mejor:.6f}",
        "ranking_json":      json.dumps({a: round(v, 6) for a, v in ranking}),
    }

    # Sustituir si ya existe fila para esta fecha, si no añadir
    actualizado = False
    for i, fila in enumerate(filas):
        if fila.get("fecha") == fecha:
            filas[i] = fila_nueva
            actualizado = True
            break
    if not actualizado:
        filas.append(fila_nueva)

    # Reescribir el CSV completo
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_CABECERA)
        writer.writeheader()
        writer.writerows(filas)

    print(f"[LOG] Decisión registrada en: {LOG_CSV}")


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
    precios = descargar_precios(TICKER, PERIOD)
    dia_hoy = len(precios) - 1

    # ── 2. Construir el estado raíz s₀ del árbol MCTS ─────────────────────
    estado = crear_estado(
        dia          = dia_hoy,
        efectivo     = efectivo,
        acciones     = num_acciones,
        precio       = float(precios[dia_hoy]),
        precios_hist = precios,
    )

    valor_actual = valor_cartera(estado)
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
    _interpretar_indicadores(estado, precio_hoy)
    print(sep)

    # ── 4. Ejecutar MCTS ───────────────────────────────────────────────────
    print(f"\n  Ejecutando MCTS ({ITERACIONES} iteraciones, "
          f"horizonte {DIAS_ROLLOUT} días)...\n")

    rng    = np.random.default_rng(SEMILLA)
    accion, historial = ejecutar_mcts(
        estado, precios, rng, registrar_convergencia=True
    )

    # ── 5. Ranking de acciones por recompensa media final ──────────────────
    medias  = {a: vals[-1] for a, vals in historial.items() if vals}
    ranking = sorted(medias.items(), key=lambda x: x[1], reverse=True)

    print(f"  {'Acción':<14} {'Reward vs B&H':>14}  {'Señal':>8}")
    print("  " + "-" * 42)
    for a, v in ranking:
        barra  = _barra(v, medias)
        marca  = "  ◄ ELEGIDA" if a == accion else ""
        print(f"  {a:<14} {v:>+14.5f}  {barra}{marca}")

    print(f"\n{sep}")
    print(f"  RECOMENDACIÓN PARA HOY: {accion}")
    _interpretar_accion(accion, efectivo, num_acciones, precio_hoy)
    print(f"{sep}\n")

    # ── 6. Guardar decisión en el log CSV ─────────────────────────────────
    _registrar_decision(
        fecha   = date.today().isoformat(),
        precio  = precio_hoy,
        rsi     = estado["rsi"],
        ma20    = estado["ma20"],
        accion  = accion,
        valor   = valor_actual,
        ranking = ranking,
    )

    # ── 7. Gráficos ────────────────────────────────────────────────────────
    graficar_convergencia_ucb(historial)
    _graficar_ranking(ranking, accion)

    # ── 8. Guardar datos de las gráficas ──────────────────────────────────────
    import csv
    DATOS_DIR = os.path.join(_DIR, "datos_graficos")
    os.makedirs(DATOS_DIR, exist_ok=True)

    # datos_ranking_acciones.csv — Q(a)/N(a) final de cada acción
    with open(os.path.join(DATOS_DIR, "datos_ranking_acciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["accion", "q_sobre_n", "elegida"])
        for a_r, v_r in ranking:
            w.writerow([a_r, round(v_r, 6), "si" if a_r == accion else "no"])
    print(f"[OK] Datos ranking: {os.path.join(DATOS_DIR, 'datos_ranking_acciones.csv')}")

    # datos_convergencia_ucb.csv — evolución Q(a)/N(a) por iteración
    if historial:
        acciones_ucb = list(historial.keys())
        max_iter = max(len(v) for v in historial.values())
        with open(os.path.join(DATOS_DIR, "datos_convergencia_ucb.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iteracion"] + acciones_ucb)
            for i in range(max_iter):
                row = [i + 1] + [
                    round(historial[a_u][i], 6) if i < len(historial[a_u]) else ""
                    for a_u in acciones_ucb
                ]
                w.writerow(row)
        print(f"[OK] Datos convergencia: {os.path.join(DATOS_DIR, 'datos_convergencia_ucb.csv')}")

    # Abrir imágenes solo en ejecución interactiva (no en launchd/cron)
    if sys.stdout.isatty():
        import subprocess
        subprocess.Popen(["open",
                          os.path.join(_DIR, "mcts_convergencia_hoy.png"),
                          os.path.join(_DIR, "mcts_ranking_acciones.png")])


# =============================================================================
# FUNCIONES AUXILIARES DE PRESENTACIÓN
# =============================================================================

def _interpretar_indicadores(estado: dict, precio: float) -> None:
    """
    Imprime una lectura rápida de RSI y MA20.
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
    """
    from mcts_simple import _FRACCION_ACCION
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
        print(f"  No operar hoy. Mantener la cartera sin cambios.")


def _barra(valor: float, todos: dict, ancho: int = 8) -> str:
    """
    Genera una mini barra de texto proporcional al valor relativo.
    """
    vmin = min(todos.values())
    vmax = max(todos.values())
    rango = vmax - vmin if vmax != vmin else 1.0
    proporcion = (valor - vmin) / rango
    llenos = round(proporcion * ancho)
    return "█" * llenos + "░" * (ancho - llenos)


def _graficar_ranking(ranking: list, accion_elegida: str) -> None:
    """
    Genera un gráfico de barras horizontales con el ranking de acciones.
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

    for bar, a in zip(bars, acciones):
        if a == accion_elegida:
            bar.set_edgecolor("#f39c12")
            bar.set_linewidth(2.5)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Recompensa media Q(a)/N(a) relativa a B&H (log-retorno)")
    ax.set_title(f"Acción recomendada: {accion_elegida}",
                 fontsize=10, style="italic", color="#e67e22")

    for bar, val in zip(bars, valores):
        x = bar.get_width()
        ax.text(x + (0.00005 if x >= 0 else -0.00005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.5f}",
                va="center", ha="left" if x >= 0 else "right",
                fontsize=8)

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, "mcts_ranking_acciones.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráfico ranking guardado: {os.path.join(_DIR, 'mcts_ranking_acciones.png')}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    decision_hoy(
        efectivo     = TU_EFECTIVO,
        num_acciones = TUS_ACCIONES,
    )
