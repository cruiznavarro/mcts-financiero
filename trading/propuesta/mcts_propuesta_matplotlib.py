import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ArrowStyle
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────
BG   = "#0d1117"
CARD = "#161b22"
ACC1 = "#58a6ff"   # blue
ACC2 = "#3fb950"   # green
ACC3 = "#f78166"   # red/orange
ACC4 = "#d2a8ff"   # purple
TXT  = "#e6edf3"
MUT  = "#8b949e"

def set_dark(fig, axes=None):
    fig.patch.set_facecolor(BG)
    if axes is None:
        return
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUT)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(CARD)

def fancy_box(ax, x, y, w, h, color, label, sublabel="", fontsize=11):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.04", linewidth=1.5,
                         edgecolor=color, facecolor=color + "22")
    ax.add_patch(box)
    ax.text(x, y + (0.06 if sublabel else 0), label,
            ha="center", va="center", color=TXT, fontsize=fontsize, fontweight="bold")
    if sublabel:
        ax.text(x, y - 0.13, sublabel, ha="center", va="center",
                color=MUT, fontsize=8)

def arrow(ax, x0, y0, x1, y1, color=MUT, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTADA
# ══════════════════════════════════════════════════════════════════════════════
def page_cover(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    set_dark(fig, ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # Accent bar
    bar = FancyBboxPatch((0.3, 3.8), 9.4, 0.06,
                         boxstyle="round,pad=0.01", facecolor=ACC1, linewidth=0)
    ax.add_patch(bar)

    ax.text(5, 7.8, "P R O P U E S T A   D E   P R O Y E C T O",
            ha="center", va="center", color=MUT, fontsize=13, fontweight="bold")
    ax.text(5, 6.6,
            "Agente de Trading con MCTS\nEstocástico aplicado a TSLA",
            ha="center", va="center", color=TXT, fontsize=26, fontweight="bold",
            linespacing=1.4)
    ax.text(5, 3.3, "Monte Carlo Tree Search · Procesos de Decisión de Markov · GBM",
            ha="center", va="center", color=MUT, fontsize=11)

    # Three pill-badges
    pills = [("MDP", ACC1), ("MCTS", ACC2), ("Backtesting", ACC3)]
    xs = [3, 5, 7]
    for (label, col), x in zip(pills, xs):
        badge = FancyBboxPatch((x - 0.65, 2.4), 1.3, 0.55,
                               boxstyle="round,pad=0.08", facecolor=col + "33",
                               edgecolor=col, linewidth=1.2)
        ax.add_patch(badge)
        ax.text(x, 2.67, label, ha="center", va="center",
                color=col, fontsize=11, fontweight="bold")

    ax.text(5, 1.6, "Datos: Yahoo Finance (TSLA) · Benchmark: Buy & Hold",
            ha="center", va="center", color=MUT, fontsize=10)
    ax.text(5, 0.5, "Marzo 2026",
            ha="center", va="center", color=MUT + "88", fontsize=9)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MDP FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
def page_mdp(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    set_dark(fig, ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    ax.text(5, 9.4, "1 · Formulación: Proceso de Decisión de Markov",
            ha="center", va="center", color=TXT, fontsize=16, fontweight="bold")
    ax.axhline(9.0, color=ACC1, lw=1, xmin=0.05, xmax=0.95)

    # Central cycle diagram
    cx, cy, r = 5, 5.3, 2.0
    nodes = [
        ("Estado\n$s_t$",     0,    ACC1),
        ("Acción\n$a_t$",    90,    ACC2),
        ("Entorno\n(Mercado)", 180,  ACC3),
        ("Recompensa\n$R_t$", 270,   ACC4),
    ]
    node_positions = []
    for label, angle_deg, color in nodes:
        angle = np.radians(angle_deg)
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        node_positions.append((x, y, color))
        fancy_box(ax, x, y, 1.55, 0.75, color, label, fontsize=10)

    # Arrows in cycle
    for i, (x0, y0, c0) in enumerate(node_positions):
        x1, y1, _ = node_positions[(i + 1) % len(node_positions)]
        mx = (x0 + x1) / 2 + cx * 0.04
        my = (y0 + y1) / 2 + cy * 0.04
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->, head_width=0.25",
                                   color=c0, lw=1.8,
                                   connectionstyle="arc3,rad=0.25"))

    # Info boxes below
    info = [
        (2.0, 2.2, ACC1, "Estado $s_t$",
         "$s_t = (P_t,\\; C_t,\\; H_t)$\nPrecio · Capital · Holdings\n$V_t = C_t + H_t \\cdot P_t$"),
        (5.0, 2.2, ACC2, "Acciones $a_t$",
         "$a_t \\in \\{-k\\%,\\; 0,\\; +k\\%\\}$\nVender · Mantener · Comprar"),
        (8.0, 2.2, ACC4, "Recompensa $R_t$",
         "$R_t = \\ln\\!\\left(\\dfrac{V_t}{V_{t-1}}\\right)$\nRetorno logarítmico"),
    ]
    for x, y, col, title, body in info:
        box = FancyBboxPatch((x - 1.5, y - 1.05), 3.0, 2.1,
                             boxstyle="round,pad=0.05", linewidth=1.3,
                             edgecolor=col, facecolor=col + "18")
        ax.add_patch(box)
        ax.text(x, y + 0.72, title, ha="center", va="center",
                color=col, fontsize=10, fontweight="bold")
        ax.axhline(y + 0.45, color=col + "55", lw=0.8,
                   xmin=(x - 1.3) / 10, xmax=(x + 1.3) / 10)
        ax.text(x, y - 0.15, body, ha="center", va="center",
                color=TXT, fontsize=8.5, linespacing=1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MCTS ÁRBOL
# ══════════════════════════════════════════════════════════════════════════════
def page_mcts(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    set_dark(fig, ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    ax.text(5, 9.4, "2 · Motor MCTS — 4 Fases",
            ha="center", va="center", color=TXT, fontsize=16, fontweight="bold")
    ax.axhline(9.0, color=ACC2, lw=1, xmin=0.05, xmax=0.95)

    # ── tree nodes ──────────────────────────────────────────────────────────
    # root
    ROOT = (5, 7.8)
    # depth-1
    D1 = [(2.5, 6.2), (5, 6.2), (7.5, 6.2)]
    # depth-2 (children of D1[2] = selected branch)
    D2 = [(6.5, 4.6), (7.5, 4.6), (8.5, 4.6)]
    # rollout horizon
    ROLL = [(7.5, 3.2), (7.5, 2.0)]

    def node(pos, color=MUT, r=0.32, label="", sublabel=""):
        c = plt.Circle(pos, r, color=color + "33", linewidth=1.8, ec=color, zorder=3)
        ax.add_patch(c)
        if label:
            ax.text(pos[0], pos[1] + (0.07 if sublabel else 0), label,
                    ha="center", va="center", color=TXT, fontsize=8, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(pos[0], pos[1] - 0.16, sublabel,
                    ha="center", va="center", color=MUT, fontsize=6.5, zorder=4)

    def edge(p0, p1, color=MUT, lw=1.2, style="-"):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
                linestyle=style, zorder=2)

    # Edges root → D1
    for d in D1:
        edge(ROOT, d)
    # Edges D1[2] (selected) → D2
    for d in D2:
        edge(D1[2], d, color=ACC2, lw=1.8)
    # Rollout from D2[1]
    edge(D2[1], ROLL[0], color=ACC3, lw=1.5, style="--")
    edge(ROLL[0], ROLL[1], color=ACC3, lw=1.5, style="--")
    # Backprop arrows
    for pos in [D2[1], D1[2], ROOT]:
        ax.annotate("", xy=(pos[0] - 0.05, pos[1] + 0.35),
                    xytext=(pos[0] - 0.05, pos[1] - 0.35),
                    arrowprops=dict(arrowstyle="->", color=ACC4, lw=1.5))

    # Draw nodes
    node(ROOT, ACC1, label="$s_0$", sublabel="N=42")
    for i, d in enumerate(D1):
        col = ACC2 if i == 2 else MUT
        node(d, col, label=f"$s_{{1,{i+1}}}$",
             sublabel=f"n={np.random.randint(5,20)}")
    node(D1[2], ACC2, r=0.34, label="$s_{1,3}$", sublabel="n=17 ★")
    for j, d in enumerate(D2):
        col = ACC3 if j == 1 else MUT
        node(d, col, label=f"$s_{{2,{j+1}}}$",
             sublabel="nuevo" if j == 1 else "")
    for k, r in enumerate(ROLL):
        node(r, ACC3, r=0.28, label=f"$\\tau_{{{k+1}}}$")

    # Phase labels
    phases = [
        (1.0, 7.0, ACC1, "① SELECCIÓN", "UCT elige la rama\ncon mayor Q + exploración"),
        (1.0, 5.5, ACC2, "② EXPANSIÓN",  "Instancia hijo\nnuevo no explorado"),
        (1.0, 3.9, ACC3, "③ ROLLOUT",    "Simula el mercado\ncon GBM hasta T"),
        (1.0, 2.5, ACC4, "④ BACKPROP",  "Propaga R hacia\nla raíz, actualiza Q"),
    ]
    for x, y, col, title, desc in phases:
        badge = FancyBboxPatch((x - 0.9, y - 0.52), 2.6, 1.04,
                               boxstyle="round,pad=0.05", facecolor=col + "1a",
                               edgecolor=col, linewidth=1.2)
        ax.add_patch(badge)
        ax.text(x + 0.4, y + 0.22, title, ha="center", va="center",
                color=col, fontsize=9, fontweight="bold")
        ax.text(x + 0.4, y - 0.18, desc, ha="center", va="center",
                color=MUT, fontsize=7.5, linespacing=1.4)

    # UCT formula
    formula_box = FancyBboxPatch((3.5, 0.4), 5.8, 1.0,
                                 boxstyle="round,pad=0.05",
                                 facecolor=ACC1 + "15", edgecolor=ACC1, linewidth=1.2)
    ax.add_patch(formula_box)
    ax.text(6.4, 1.05, "UCT — Fórmula de Selección",
            ha="center", va="center", color=ACC1, fontsize=9, fontweight="bold")
    ax.text(6.4, 0.72,
            r"$a^* = \arg\max_a \left[ Q(s,a) + c\sqrt{\frac{\ln N(s)}{n(s,a)}} \right]$",
            ha="center", va="center", color=TXT, fontsize=10)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GBM SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def page_gbm(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    set_dark(fig, axes)
    fig.suptitle("3 · Simulación del Mercado — Movimiento Browniano Geométrico",
                 color=TXT, fontsize=14, fontweight="bold", y=0.97)

    ax1, ax2 = axes

    # ── Left: GBM trajectories ───────────────────────────────────────────────
    np.random.seed(42)
    S0, mu, sigma, T, dt = 250, 0.0003, 0.018, 252, 1
    steps = int(T / dt)
    t = np.arange(steps)

    colors_traj = plt.cm.cool(np.linspace(0.2, 0.9, 40))
    for i, c in enumerate(colors_traj):
        Z = np.random.randn(steps)
        increments = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        S = S0 * np.cumprod(increments)
        alpha = 0.25 if i < 38 else 1.0
        lw = 0.5 if i < 38 else 1.8
        ax1.plot(t, S, color=c, alpha=alpha, lw=lw)

    # Highlight two special paths
    for color, seed in [(ACC2, 7), (ACC3, 13)]:
        np.random.seed(seed)
        Z = np.random.randn(steps)
        S = S0 * np.cumprod(np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z))
        ax1.plot(t, S, color=color, lw=2.2, label=f"Trayectoria muestra")

    ax1.axhline(S0, color=MUT, lw=1, linestyle="--", alpha=0.6, label="$P_0$")
    ax1.set_title("Rollout: 40 trayectorias GBM", color=TXT, fontsize=11, pad=8)
    ax1.set_xlabel("Días de trading", color=TXT)
    ax1.set_ylabel("Precio TSLA ($)", color=TXT)
    ax1.legend(facecolor=CARD, edgecolor=MUT, labelcolor=TXT, fontsize=8)

    # ── Right: formula + parameters ─────────────────────────────────────────
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")

    ax2.text(5, 9.3, "Modelo estocástico del precio",
             ha="center", va="center", color=TXT, fontsize=12, fontweight="bold")

    # Formula box
    fbox = FancyBboxPatch((0.5, 7.5), 9.0, 1.45,
                          boxstyle="round,pad=0.05",
                          facecolor=ACC3 + "18", edgecolor=ACC3, linewidth=1.3)
    ax2.add_patch(fbox)
    ax2.text(5, 8.62,
             r"$P_{t+1} = P_t \cdot \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t "
             r"+ \sigma\sqrt{\Delta t}\cdot Z\right]$",
             ha="center", va="center", color=TXT, fontsize=11)
    ax2.text(5, 7.75, r"$Z \sim \mathcal{N}(0, 1)$ — ruido gaussiano",
             ha="center", va="center", color=MUT, fontsize=9)

    # Parameter table
    params = [
        ("$\\mu$",     "Deriva diaria",       "0.03% histórico TSLA", ACC1),
        ("$\\sigma$",  "Volatilidad diaria",  "1.8% (empírico)",       ACC2),
        ("$\\Delta t$","Paso temporal",       "1 día",                 ACC4),
        ("$T$",        "Horizonte rollout",   "252 días (1 año)",      ACC3),
    ]
    y = 6.8
    for sym, name, val, col in params:
        row = FancyBboxPatch((0.5, y - 0.34), 9.0, 0.68,
                             boxstyle="round,pad=0.03",
                             facecolor=col + "15", edgecolor=col + "55", linewidth=0.8)
        ax2.add_patch(row)
        ax2.text(1.2,  y, sym,  ha="center", va="center", color=col,  fontsize=11, fontweight="bold")
        ax2.text(3.5,  y, name, ha="center", va="center", color=TXT,  fontsize=9)
        ax2.text(7.5,  y, val,  ha="center", va="center", color=MUT,  fontsize=9)
        y -= 0.78

    # Intuition note
    note = FancyBboxPatch((0.5, 2.1), 9.0, 1.6,
                          boxstyle="round,pad=0.05",
                          facecolor=ACC4 + "15", edgecolor=ACC4, linewidth=1.0)
    ax2.add_patch(note)
    ax2.text(5, 3.35, "¿Por qué GBM?", ha="center", va="center",
             color=ACC4, fontsize=10, fontweight="bold")
    ax2.text(5, 2.75,
             "Captura que los retornos (no los precios) son gaussianos,\n"
             "impone precio siempre positivo y permite calibrar\n"
             "$\\mu$ y $\\sigma$ directamente de datos históricos.",
             ha="center", va="center", color=MUT, fontsize=8.5, linespacing=1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BACKTESTING & MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
def page_backtest(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    set_dark(fig, axes)
    fig.suptitle("4 · Evaluación Empírica — Backtesting",
                 color=TXT, fontsize=14, fontweight="bold", y=0.97)

    ax1, ax2 = axes

    # ── Left: simulated equity curves ────────────────────────────────────────
    np.random.seed(99)
    days = 252
    t = np.arange(days)

    # Buy & Hold
    bh_returns = np.random.normal(0.0003, 0.018, days)
    bh_curve = np.exp(np.cumsum(bh_returns))

    # MCTS (slightly better Sharpe)
    mc_returns = np.random.normal(0.0006, 0.014, days)
    mc_curve = np.exp(np.cumsum(mc_returns))

    ax1.fill_between(t, 1, bh_curve, alpha=0.08, color=MUT)
    ax1.fill_between(t, 1, mc_curve, alpha=0.12, color=ACC2)
    ax1.plot(t, bh_curve, color=ACC3, lw=2.0, label="Buy & Hold")
    ax1.plot(t, mc_curve, color=ACC2, lw=2.2, label="Agente MCTS")
    ax1.axhline(1.0, color=MUT, lw=0.8, linestyle="--")

    ax1.set_title("Comparación de curvas de capital (simulado)",
                  color=TXT, fontsize=10, pad=8)
    ax1.set_xlabel("Días de trading", color=TXT)
    ax1.set_ylabel("Valor portafolio normalizado", color=TXT)
    ax1.legend(facecolor=CARD, edgecolor=MUT, labelcolor=TXT, fontsize=9)

    # ── Right: metrics cards ─────────────────────────────────────────────────
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.text(5, 9.4, "Métricas de evaluación",
             ha="center", va="center", color=TXT, fontsize=12, fontweight="bold")

    metrics = [
        ("ROI",  "Rendimiento acumulado",
         r"$\dfrac{V_T - V_0}{V_0}$",
         "Mide el retorno total del portafolio\nsobre todo el periodo de backtest.", ACC1),
        ("Sharpe", "Ratio de Sharpe",
         r"$S = \dfrac{\mathbb{E}[R_p - R_f]}{\sigma_p}$",
         "Retorno ajustado por riesgo.\nMayor Sharpe = mejor perfil riesgo/retorno.", ACC2),
    ]

    y0 = 8.2
    for name, full, formula, desc, col in metrics:
        h = 3.4
        card = FancyBboxPatch((0.4, y0 - h), 9.2, h - 0.15,
                              boxstyle="round,pad=0.06",
                              facecolor=col + "18", edgecolor=col, linewidth=1.3)
        ax2.add_patch(card)
        ax2.text(5, y0 - 0.38, full, ha="center", va="center",
                 color=col, fontsize=11, fontweight="bold")
        ax2.text(5, y0 - 1.3, formula, ha="center", va="center",
                 color=TXT, fontsize=13)
        ax2.text(5, y0 - 2.55, desc, ha="center", va="center",
                 color=MUT, fontsize=8.5, linespacing=1.5)
        y0 -= h + 0.2

    # Bottom note
    ax2.text(5, 0.55,
             "Validación out-of-sample con datos de Yahoo Finance (TSLA)\n"
             "usando walk-forward cross-validation.",
             ha="center", va="center", color=MUT, fontsize=8.5, linespacing=1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
OUTPUT = "/Users/carlosruiznavarro/MCTS_Trading_Propuesta.pdf"

with PdfPages(OUTPUT) as pdf:
    page_cover(pdf)
    page_mdp(pdf)
    page_mcts(pdf)
    page_gbm(pdf)
    page_backtest(pdf)

    d = pdf.infodict()
    d["Title"]   = "Agente de Trading MCTS Estocástico — TSLA"
    d["Author"]  = "Propuesta de Proyecto"
    d["Subject"] = "Monte Carlo Tree Search aplicado a Finanzas"

print(f"PDF generado: {OUTPUT}")
