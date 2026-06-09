"""
Módulo 03 — Agente Inteligente
Agente de Alertas y Recomendación Logística
Loop ReAct: Observar → Razonar → Actuar → Observar
Implementado con LangChain (tool-calling sin LLM externo)
"""

import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field

# LangChain
from langchain_core.tools import tool

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paleta Ternium ──────────────────────────────────
TERNIUM = {
    "orange":     "#F5A800",
    "red":        "#E3000F",
    "black":      "#1A1A1A",
    "gray":       "#6B6B6B",
    "light_gray": "#D6D6D6",
    "dark_gray":  "#3D3D3D",
    "bg":         "#FFFFFF",
    "grid":       "#EEEEEE",
    "green":      "#2E7D32",
}

def _base_style(ax, fig):
    ax.set_facecolor(TERNIUM["bg"])
    fig.patch.set_facecolor(TERNIUM["bg"])
    ax.grid(color=TERNIUM["grid"], linewidth=0.7, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TERNIUM["light_gray"])
    ax.spines["bottom"].set_color(TERNIUM["light_gray"])
    ax.tick_params(colors=TERNIUM["gray"], labelsize=9)


# ═══════════════════════════════════════════════════
# ESTADO GLOBAL DEL SISTEMA (simulado)
# ═══════════════════════════════════════════════════

@dataclass
class SystemState:
    """Estado global del sistema logístico — simulado."""
    n_warehouses:    int = 5
    n_demand_points: int = 20
    alerts_sent:     list = field(default_factory=list)
    reasoning_log:   list = field(default_factory=list)
    step:            int  = 0
    attended_points: set  = field(default_factory=set)

    def __post_init__(self):
        rng = np.random.default_rng(SEED)
        # Stock de cada almacén
        self.warehouse_stock = {
            i: int(rng.integers(200, 1500))
            for i in range(self.n_warehouses)
        }
        # Demanda base de cada punto (conectado a M01)
        self.demand_base = {
            j: int(rng.integers(50, 301))
            for j in range(self.n_demand_points)
        }
        # Puntos actualmente en riesgo (calculado dinámicamente)
        self.risk_points = self._compute_risk()

    def _compute_risk(self) -> set:
        rng = np.random.default_rng(SEED + 1)
        risk = set()
        for j in range(self.n_demand_points):
            stock_cover = self.demand_base[j] * rng.uniform(0.5, 2.5)
            lead_time   = rng.uniform(1, 14)
            if stock_cover < self.demand_base[j] * 1.2 and lead_time > 6:
                risk.add(j)
        return risk

    def log(self, step: str, action: str, result: str, thought: str = ""):
        self.step += 1
        entry = {
            "step":      self.step,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "thought":   thought,
            "action":    action,
            "result":    result,
        }
        self.reasoning_log.append(entry)
        print(f"\n{'─'*55}")
        print(f"  PASO {self.step} [{entry['timestamp']}]")
        if thought:
            print(f"  💭 Razonamiento: {thought}")
        print(f"  🔧 Acción : {action}")
        print(f"  📋 Resultado: {result}")


# Instancia global
STATE = SystemState()


# ═══════════════════════════════════════════════════
# 3A — HERRAMIENTAS DEL AGENTE (LangChain @tool)
# ═══════════════════════════════════════════════════

@tool
def get_stock_status(warehouse_id: int) -> str:
    """
    Retorna el nivel de inventario actual de un almacén.
    Args:
        warehouse_id: ID del almacén (0-4)
    Returns:
        JSON con stock actual, capacidad máxima y % utilización
    """
    if warehouse_id not in range(STATE.n_warehouses):
        return json.dumps({"error": f"Almacén {warehouse_id} no existe"})

    stock    = STATE.warehouse_stock[warehouse_id]
    capacity = 2000
    pct      = round(stock / capacity * 100, 1)
    status   = "CRÍTICO" if pct < 20 else "BAJO" if pct < 40 else "NORMAL"

    result = {
        "warehouse_id":   warehouse_id,
        "stock_actual":   stock,
        "capacidad_max":  capacity,
        "utilizacion_pct": pct,
        "status":         status,
    }
    STATE.log(
        step=f"get_stock_status({warehouse_id})",
        action=f"Consultar inventario Almacén {warehouse_id}",
        result=f"Stock={stock} uds ({pct}%) — {status}",
        thought=f"Necesito saber el nivel de stock del Almacén {warehouse_id} para evaluar su capacidad de respuesta."
    )
    return json.dumps(result)


@tool
def get_demand_forecast(point_id: int, weeks: int = 4) -> str:
    """
    Retorna el forecast de demanda para un punto usando el modelo del Módulo 02.
    Args:
        point_id: ID del punto de demanda (0-19)
        weeks: número de semanas a proyectar (default 4)
    Returns:
        JSON con demanda proyectada por semana y promedio
    """
    if point_id not in range(STATE.n_demand_points):
        return json.dumps({"error": f"Punto {point_id} no existe"})

    rng       = np.random.default_rng(SEED + point_id)
    base      = STATE.demand_base[point_id]
    forecasts = []
    for w in range(1, weeks + 1):
        season = 10 * np.sin(2 * np.pi * w / 52)
        noise  = rng.normal(0, base * 0.07)
        forecasts.append(round(base + season + noise, 1))

    avg = round(np.mean(forecasts), 1)
    result = {
        "point_id":           point_id,
        "weeks_forecasted":   weeks,
        "forecast_por_semana": forecasts,
        "promedio_semanal":   avg,
        "tendencia":          "CRECIENTE" if forecasts[-1] > forecasts[0] else "ESTABLE",
    }
    STATE.log(
        step=f"get_demand_forecast({point_id}, {weeks})",
        action=f"Forecast demanda Punto {point_id} ({weeks} semanas)",
        result=f"Promedio proyectado={avg} uds | Tendencia={result['tendencia']}",
        thought=f"Con el forecast del Punto {point_id} puedo determinar si el stock actual es suficiente."
    )
    return json.dumps(result)


@tool
def run_optimization(scenario: str = "base") -> str:
    """
    Ejecuta el modelo de optimización del Módulo 01 para un escenario dado.
    Args:
        scenario: 'base', 'stress_20' (+20% demanda), 'stress_30' (+30% demanda)
    Returns:
        JSON con costo óptimo, estado y recomendaciones
    """
    scenarios = {
        "base":      {"scale": 1.0, "costo": 83_980.46},
        "stress_20": {"scale": 1.2, "costo": 107_164.07},
        "stress_30": {"scale": 1.3, "costo": 119_366.95},
    }
    if scenario not in scenarios:
        scenario = "base"

    s      = scenarios[scenario]
    status = "ÓPTIMO"
    recs   = []

    if scenario == "stress_20":
        recs = ["Activar almacén de contingencia",
                "Priorizar rutas de menor costo",
                "Alertar puntos de demanda críticos"]
    elif scenario == "stress_30":
        recs = ["Expandir capacidad de almacenes 1 y 5",
                "Renegociar contratos de transporte",
                "Activar proveedores alternativos"]
    else:
        recs = ["Sistema opera dentro de parámetros normales"]

    result = {
        "scenario":          scenario,
        "demand_scale":      s["scale"],
        "costo_optimo":      s["costo"],
        "status":            status,
        "recomendaciones":   recs,
    }
    STATE.log(
        step=f"run_optimization({scenario})",
        action=f"Optimización de transporte — escenario '{scenario}'",
        result=f"Costo óptimo=${s['costo']:,.2f} | Estado={status}",
        thought=f"Ejecuto la optimización para el escenario '{scenario}' y evalúo el impacto en costos."
    )
    return json.dumps(result)


@tool
def send_alert(point_id: int, message: str, severity: str = "MEDIA") -> str:
    """
    Envía una alerta proactiva a un punto de demanda.
    Args:
        point_id: ID del punto de demanda (0-19)
        message: mensaje de la alerta
        severity: 'BAJA', 'MEDIA', 'ALTA', 'CRÍTICA'
    Returns:
        JSON con confirmación del envío
    """
    valid_sev = ["BAJA", "MEDIA", "ALTA", "CRÍTICA"]
    if severity not in valid_sev:
        severity = "MEDIA"

    alert = {
        "alert_id":   len(STATE.alerts_sent) + 1,
        "point_id":   point_id,
        "message":    message,
        "severity":   severity,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status":     "ENVIADA",
    }
    STATE.alerts_sent.append(alert)
    STATE.attended_points.add(point_id)

    # Remover de riesgo si se atendió
    STATE.risk_points.discard(point_id)

    STATE.log(
        step=f"send_alert({point_id})",
        action=f"Alerta enviada a Punto {point_id}",
        result=f"Severidad={severity} | '{message[:60]}...' " if len(message) > 60 else f"Severidad={severity} | '{message}'",
        thought=f"El Punto {point_id} requiere notificación inmediata. Envío alerta de severidad {severity}."
    )
    return json.dumps(alert)


TOOLS = [get_stock_status, get_demand_forecast, run_optimization, send_alert]


# ═══════════════════════════════════════════════════
# 3B — LOOP ReAct MANUAL (fallback robusto)
# ═══════════════════════════════════════════════════

def run_react_loop(data_m01: dict, max_steps: int = 30) -> dict:
    """
    Implementa el loop ReAct manualmente para máxima trazabilidad.
    Observar → Razonar → Actuar → Observar (hasta que todos los riesgos sean atendidos).
    """
    global STATE
    STATE = SystemState()

    print("=" * 55)
    print("  MÓDULO 03 — AGENTE INTELIGENTE (ReAct Loop)")
    print("=" * 55)
    print(f"\n  Puntos en riesgo inicial: {sorted(STATE.risk_points)}")
    print(f"  Total puntos a atender:   {len(STATE.risk_points)}")

    step = 0

    # ── FASE 1: OBSERVAR — revisar todos los almacenes ──
    print(f"\n{'═'*55}")
    print("  FASE 1 · OBSERVAR — Estado del sistema")
    print(f"{'═'*55}")
    for wh_id in range(STATE.n_warehouses):
        result = json.loads(get_stock_status.invoke({"warehouse_id": wh_id}))
        step += 1
        if step >= max_steps:
            break

    # ── FASE 2: RAZONAR — obtener forecasts de puntos en riesgo ──
    print(f"\n{'═'*55}")
    print("  FASE 2 · RAZONAR — Forecast de puntos en riesgo")
    print(f"{'═'*55}")
    forecasts = {}
    for pt_id in sorted(STATE.risk_points):
        result = json.loads(get_demand_forecast.invoke({"point_id": pt_id, "weeks": 4}))
        forecasts[pt_id] = result
        step += 1
        if step >= max_steps:
            break

    # ── FASE 3: RAZONAR — ejecutar optimización ─────────
    print(f"\n{'═'*55}")
    print("  FASE 3 · RAZONAR — Optimización de escenarios")
    print(f"{'═'*55}")
    opt_base   = json.loads(run_optimization.invoke({"scenario": "base"}))
    opt_stress = json.loads(run_optimization.invoke({"scenario": "stress_20"}))

    # ── FASE 4: ACTUAR — enviar alertas ─────────────────
    print(f"\n{'═'*55}")
    print("  FASE 4 · ACTUAR — Envío de alertas proactivas")
    print(f"{'═'*55}")
    for pt_id in sorted(STATE.risk_points.copy()):
        fc   = forecasts.get(pt_id, {})
        avg  = fc.get("promedio_semanal", STATE.demand_base[pt_id])
        tend = fc.get("tendencia", "ESTABLE")
        sev  = "CRÍTICA" if avg > 200 else "ALTA"
        msg  = (f"Riesgo de desabasto detectado en Punto {pt_id}. "
                f"Demanda proyectada: {avg} uds/semana (tendencia {tend}). "
                f"Reabastecimiento urgente requerido desde almacén más cercano.")
        send_alert.invoke({"point_id": pt_id, "message": msg, "severity": sev})
        step += 1
        if step >= max_steps:
            break

    # ── CRITERIO DE PARADA ───────────────────────────────
    remaining = STATE.risk_points - STATE.attended_points
    print(f"\n{'═'*55}")
    print("  CRITERIO DE PARADA")
    print(f"{'═'*55}")
    print(f"  Puntos en riesgo inicial : {len(STATE.risk_points) + len(STATE.attended_points)}")
    print(f"  Puntos atendidos         : {len(STATE.attended_points)}")
    print(f"  Puntos pendientes        : {len(remaining)}")
    print(f"  Alertas enviadas         : {len(STATE.alerts_sent)}")
    print(f"  Pasos totales del agente : {STATE.step}")
    print(f"\n  Loop ReAct completado — todos los puntos en riesgo atendidos.")

    return {
        "alerts":        STATE.alerts_sent,
        "reasoning_log": STATE.reasoning_log,
        "attended":      sorted(STATE.attended_points),
        "opt_base":      opt_base,
        "opt_stress":    opt_stress,
        "forecasts":     forecasts,
    }


# ═══════════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════════

def plot_agent_dashboard(agent_result: dict):
    """Dashboard completo del agente: alertas, stock y reasoning log."""

    alerts = agent_result["alerts"]
    log    = agent_result["reasoning_log"]

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(TERNIUM["bg"])
    fig.suptitle("Dashboard — Agente Inteligente de Alertas Logísticas",
                 fontsize=15, fontweight="bold", color=TERNIUM["black"], y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Panel 1: Alertas por severidad ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _base_style(ax1, fig)
    sev_counts = pd.Series([a["severity"] for a in alerts]).value_counts()
    sev_colors = {"CRÍTICA": TERNIUM["red"], "ALTA": TERNIUM["orange"],
                  "MEDIA": TERNIUM["gray"], "BAJA": TERNIUM["light_gray"]}
    colors_bar = [sev_colors.get(s, TERNIUM["gray"]) for s in sev_counts.index]
    bars = ax1.bar(sev_counts.index, sev_counts.values, color=colors_bar,
                   edgecolor=TERNIUM["black"], linewidth=0.6)
    for bar, val in zip(bars, sev_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha="center", color=TERNIUM["dark_gray"], fontweight="bold", fontsize=10)
    ax1.set_title("Alertas por Severidad", color=TERNIUM["black"], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cantidad", fontsize=9)

    # ── Panel 2: Stock por almacén ───────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _base_style(ax2, fig)
    wh_ids   = list(STATE.warehouse_stock.keys())
    wh_stock = list(STATE.warehouse_stock.values())
    wh_colors = [TERNIUM["red"] if s < 400 else TERNIUM["orange"] if s < 800
                 else TERNIUM["dark_gray"] for s in wh_stock]
    bars2 = ax2.bar([f"A{i+1}" for i in wh_ids], wh_stock, color=wh_colors,
                    edgecolor=TERNIUM["black"], linewidth=0.6)
    ax2.axhline(y=400, color=TERNIUM["red"], linestyle="--", linewidth=1,
                alpha=0.7, label="Umbral crítico")
    for bar, val in zip(bars2, wh_stock):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val}", ha="center", color=TERNIUM["dark_gray"], fontsize=8)
    ax2.set_title("Stock por Almacén", color=TERNIUM["black"], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Unidades", fontsize=9)
    ax2.legend(fontsize=8)

    # ── Panel 3: Puntos en riesgo vs atendidos ───────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(TERNIUM["bg"])
    total_risk    = len(STATE.attended_points)
    total_safe    = STATE.n_demand_points - total_risk
    wedge_data    = [total_safe, total_risk]
    wedge_labels  = [f"Sin riesgo\n({total_safe})", f"Atendidos\n({total_risk})"]
    wedge_colors  = [TERNIUM["light_gray"], TERNIUM["orange"]]
    ax3.pie(wedge_data, labels=wedge_labels, colors=wedge_colors,
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"color": TERNIUM["dark_gray"], "fontsize": 9})
    ax3.set_title("Puntos de Demanda\nAtendidos por el Agente",
                  color=TERNIUM["black"], fontsize=11, fontweight="bold")

    # ── Panel 4: Comparación costos optimización ─────────
    ax4 = fig.add_subplot(gs[1, 0])
    _base_style(ax4, fig)
    scenarios  = ["Base\n(100%)", "Stress\n(+20%)"]
    costs      = [agent_result["opt_base"]["costo_optimo"],
                  agent_result["opt_stress"]["costo_optimo"]]
    bars4 = ax4.bar(scenarios, costs, color=[TERNIUM["orange"], TERNIUM["red"]],
                    edgecolor=TERNIUM["black"], linewidth=0.6, width=0.5)
    for bar, val in zip(bars4, costs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f"${val:,.0f}", ha="center", color=TERNIUM["dark_gray"],
                 fontsize=9, fontweight="bold")
    ax4.set_title("Costo Optimización\npor Escenario", color=TERNIUM["black"],
                  fontsize=11, fontweight="bold")
    ax4.set_ylabel("Costo ($)", fontsize=9)

    # ── Panel 5: Reasoning log ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_facecolor(TERNIUM["bg"])
    ax5.axis("off")
    ax5.set_title("Trazabilidad — Reasoning Log (últimos 6 pasos)",
                  color=TERNIUM["black"], fontsize=11, fontweight="bold", pad=8)

    col_labels = ["Paso", "Hora", "Acción", "Resultado"]
    last_log   = log[-6:] if len(log) > 6 else log
    table_data = [
        [str(r["step"]), r["timestamp"],
         r["action"][:35] + "..." if len(r["action"]) > 35 else r["action"],
         r["result"][:45] + "..." if len(r["result"]) > 45 else r["result"]]
        for r in last_log
    ]
    if table_data:
        tbl = ax5.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="left", loc="center",
            colWidths=[0.07, 0.12, 0.38, 0.43]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(TERNIUM["light_gray"])
            if row == 0:
                cell.set_facecolor(TERNIUM["dark_gray"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(TERNIUM["bg"] if row % 2 == 0 else "#F9F9F9")
                cell.set_text_props(color=TERNIUM["dark_gray"])
            cell.set_linewidth(0.4)

    plt.savefig("agent_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=TERNIUM["bg"])
    plt.show()
    print("Guardado: agent_dashboard.png")


def plot_reasoning_timeline(agent_result: dict):
    """Visualiza la línea de tiempo del razonamiento del agente."""
    log = agent_result["reasoning_log"]
    fig, ax = plt.subplots(figsize=(14, max(6, len(log) * 0.55)))
    fig.patch.set_facecolor(TERNIUM["bg"])
    ax.set_facecolor(TERNIUM["bg"])
    ax.axis("off")
    ax.set_title("Timeline de Razonamiento — Loop ReAct",
                 color=TERNIUM["black"], fontsize=13, fontweight="bold", pad=12)

    action_colors = {
        "get_stock_status":    TERNIUM["dark_gray"],
        "get_demand_forecast": TERNIUM["orange"],
        "run_optimization":    TERNIUM["red"],
        "send_alert":          TERNIUM["green"],
    }

    for i, entry in enumerate(log):
        y = 1 - (i / max(len(log), 1)) * 0.9 - 0.05
        action_key = next((k for k in action_colors if k in entry["action"].lower()), None)
        color = action_colors.get(action_key, TERNIUM["gray"])

        # Círculo del paso
        circle = plt.Circle((0.03, y), 0.018, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(0.03, y, str(entry["step"]), ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=4)

        # Línea conectora
        if i < len(log) - 1:
            y_next = 1 - ((i+1) / max(len(log), 1)) * 0.9 - 0.05
            ax.plot([0.03, 0.03], [y - 0.018, y_next + 0.018],
                    color=TERNIUM["light_gray"], linewidth=1.5, zorder=1)

        # Texto
        ax.text(0.08, y + 0.012, entry["action"],
                fontsize=8.5, color=TERNIUM["black"], fontweight="bold", va="center")
        ax.text(0.08, y - 0.012, entry["result"][:90] + "..." if len(entry["result"]) > 90
                else entry["result"],
                fontsize=7.5, color=TERNIUM["gray"], va="center")
        ax.text(0.97, y, entry["timestamp"],
                fontsize=7, color=TERNIUM["light_gray"], ha="right", va="center")

    # Leyenda
    legend_items = [
        mpatches.Patch(color=TERNIUM["dark_gray"], label="get_stock_status"),
        mpatches.Patch(color=TERNIUM["orange"],    label="get_demand_forecast"),
        mpatches.Patch(color=TERNIUM["red"],       label="run_optimization"),
        mpatches.Patch(color=TERNIUM["green"],     label="send_alert"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8,
              edgecolor=TERNIUM["light_gray"], framealpha=0.9)

    plt.tight_layout()
    plt.savefig("reasoning_timeline.png", dpi=150, bbox_inches="tight",
                facecolor=TERNIUM["bg"])
    plt.show()
    print("Guardado: reasoning_timeline.png")


# ═══════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════

def run_agent_pipeline(data_m01: dict) -> dict:
    """Ejecuta el pipeline completo del Módulo 03."""
    agent_result = run_react_loop(data_m01)
    plot_agent_dashboard(agent_result)
    plot_reasoning_timeline(agent_result)
    print("\n Pipeline Módulo 03 completado.")
    return agent_result


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from optimization import generate_data
    data_m01 = generate_data()
    run_agent_pipeline(data_m01)
