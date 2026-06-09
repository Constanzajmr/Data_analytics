"""
Módulo 01 - Optimización
Problema de Transporte con Capacidad Limitada
Solver: OR-Tools (CP-SAT / Linear Solver)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from ortools.linear_solver import pywraplp

# ─────────────────────────────────────────
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# ─────────────────────────────────────────

SEED = 42

def generate_data(seed: int = SEED) -> dict:
    """
    Genera datos sintéticos reproducibles para el problema de transporte.

    Returns:
        dict con costos, capacidades, demandas y coordenadas.
    """
    rng = np.random.default_rng(seed)

    n_warehouses = 5
    n_demand_points = 20

    # Coordenadas geográficas simuladas (para visualización)
    warehouse_coords = rng.uniform(0, 100, size=(n_warehouses, 2))
    demand_coords    = rng.uniform(0, 100, size=(n_demand_points, 2))

    # Costos de transporte proporcionales a distancia euclidiana + ruido
    costs = np.zeros((n_warehouses, n_demand_points))
    for i in range(n_warehouses):
        for j in range(n_demand_points):
            dist = np.linalg.norm(warehouse_coords[i] - demand_coords[j])
            noise = rng.uniform(0.8, 1.2)
            costs[i, j] = round(dist * noise, 2)

    # Capacidades de almacén [500, 2000]
    capacities = rng.integers(500, 2001, size=n_warehouses)

    # Demandas de cada punto [50, 300]
    demands = rng.integers(50, 301, size=n_demand_points)

    # Verificar factibilidad básica
    total_supply = capacities.sum()
    total_demand = demands.sum()
    print(f"[DATA] Oferta total: {total_supply} | Demanda total: {total_demand}")
    if total_supply < total_demand:
        print("[WARN] Oferta insuficiente — se resolverá como problema parcial.")

    return {
        "n_warehouses":      n_warehouses,
        "n_demand_points":   n_demand_points,
        "costs":             costs,
        "capacities":        capacities,
        "demands":           demands,
        "warehouse_coords":  warehouse_coords,
        "demand_coords":     demand_coords,
        "warehouse_names":   [f"Almacén {i+1}" for i in range(n_warehouses)],
        "demand_names":      [f"Punto {j+1}"   for j in range(n_demand_points)],
    }


# ─────────────────────────────────────────
# 2. MODELO DE OPTIMIZACIÓN — OR-Tools
# ─────────────────────────────────────────

def solve_transport(data: dict, demand_scale: float = 1.0) -> dict:
    """
    Formula y resuelve el Problema de Transporte con Capacidad Limitada.

    Args:
        data:          Diccionario generado por generate_data().
        demand_scale:  Factor de escala para demanda (1.0 = base, 1.2 = +20%).

    Returns:
        dict con estado, costo óptimo, flujos y variables de decisión.
    """
    costs      = data["costs"]
    capacities = data["capacities"]
    demands    = (data["demands"] * demand_scale).astype(int)
    I          = data["n_warehouses"]
    J          = data["n_demand_points"]

    solver = pywraplp.Solver.CreateSolver("GLOP")  # Solver lineal de OR-Tools
    if not solver:
        raise RuntimeError("No se pudo inicializar OR-Tools GLOP solver.")

    solver.SetTimeLimit(60_000)  # 60 segundos máximo

    # ── Variables de decisión: x[i,j] = unidades enviadas de almacén i → punto j
    x = {}
    for i in range(I):
        for j in range(J):
            x[i, j] = solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}")

    # ── Función objetivo: minimizar costo total
    objective = solver.Objective()
    for i in range(I):
        for j in range(J):
            objective.SetCoefficient(x[i, j], costs[i, j])
    objective.SetMinimization()

    # ── Restricción 1: no superar capacidad de cada almacén
    for i in range(I):
        ct = solver.Constraint(0, float(capacities[i]), f"cap_{i}")
        for j in range(J):
            ct.SetCoefficient(x[i, j], 1)

    # ── Restricción 2: satisfacer demanda de cada punto
    for j in range(J):
        ct = solver.Constraint(float(demands[j]), solver.infinity(), f"dem_{j}")
        for i in range(I):
            ct.SetCoefficient(x[i, j], 1)

    # ── Resolver
    status = solver.Solve()

    status_map = {
        pywraplp.Solver.OPTIMAL:    "ÓPTIMO",
        pywraplp.Solver.FEASIBLE:   "FACTIBLE (no óptimo)",
        pywraplp.Solver.INFEASIBLE: "INFACTIBLE",
        pywraplp.Solver.UNBOUNDED:  "NO ACOTADO",
    }
    status_str = status_map.get(status, "DESCONOCIDO")
    print(f"[SOLVER] Estado: {status_str} | Escala demanda: {demand_scale:.0%}")

    flows = np.zeros((I, J))
    total_cost = None

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        total_cost = solver.Objective().Value()
        for i in range(I):
            for j in range(J):
                flows[i, j] = x[i, j].solution_value()
        print(f"[SOLVER] Costo total óptimo: {total_cost:,.2f}")
    else:
        print("[SOLVER] No se encontró solución factible.")

    return {
        "status":     status_str,
        "total_cost": total_cost,
        "flows":      flows,
        "demands":    demands,
        "data":       data,
    }


# ─────────────────────────────────────────
# 3. ANÁLISIS DE SENSIBILIDAD
# ─────────────────────────────────────────

def sensitivity_analysis(data: dict, scales: list = None) -> pd.DataFrame:
    """
    Analiza cómo varía el costo óptimo al escalar la demanda.

    Args:
        data:   Datos del problema.
        scales: Lista de factores de escala. Default: [1.0, 1.1, 1.2, 1.3].

    Returns:
        DataFrame con resultados por escenario.
    """
    if scales is None:
        scales = [1.0, 1.1, 1.2, 1.3]

    records = []
    for s in scales:
        result = solve_transport(data, demand_scale=s)
        records.append({
            "Escala demanda":     f"{s:.0%}",
            "Demanda total":      int(result["demands"].sum()),
            "Costo óptimo":       result["total_cost"],
            "Estado":             result["status"],
            "Δ costo vs base (%)": None,
        })

    df = pd.DataFrame(records)
    base_cost = df.loc[df["Escala demanda"] == "100%", "Costo óptimo"].values[0]
    df["Δ costo vs base (%)"] = df["Costo óptimo"].apply(
        lambda c: round((c - base_cost) / base_cost * 100, 2) if c else None
    )
    print("\n[SENSIBILIDAD]\n", df.to_string(index=False))
    return df


# ─────────────────────────────────────────
# 4. VISUALIZACIONES
# ─────────────────────────────────────────

def plot_flow_map(result: dict, title: str = "Mapa de Flujos — Solución Óptima"):
    """Visualiza los flujos de distribución entre almacenes y puntos de demanda."""
    data   = result["data"]
    flows  = result["flows"]
    w_coords = data["warehouse_coords"]
    d_coords = data["demand_coords"]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    max_flow = flows.max() if flows.max() > 0 else 1

    # Dibujar flujos
    for i in range(data["n_warehouses"]):
        for j in range(data["n_demand_points"]):
            if flows[i, j] > 0.1:
                alpha = 0.2 + 0.6 * (flows[i, j] / max_flow)
                lw    = 0.5 + 2.5 * (flows[i, j] / max_flow)
                ax.plot(
                    [w_coords[i, 0], d_coords[j, 0]],
                    [w_coords[i, 1], d_coords[j, 1]],
                    color="#00bfff", alpha=alpha, linewidth=lw, zorder=1
                )

    # Puntos de demanda
    ax.scatter(d_coords[:, 0], d_coords[:, 1],
               c="#ff6b6b", s=80, zorder=3, label="Puntos de demanda", edgecolors="white", linewidths=0.5)
    for j, name in enumerate(data["demand_names"]):
        ax.annotate(name, d_coords[j], fontsize=6, color="#aaaaaa",
                    xytext=(3, 3), textcoords="offset points")

    # Almacenes
    ax.scatter(w_coords[:, 0], w_coords[:, 1],
               c="#00e676", s=200, marker="s", zorder=4, label="Almacenes", edgecolors="white", linewidths=1)
    for i, name in enumerate(data["warehouse_names"]):
        ax.annotate(name, w_coords[i], fontsize=8, color="white", fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_title(f"{title}\nCosto total: {result['total_cost']:,.2f}",
                 color="white", fontsize=13, pad=15)
    ax.tick_params(colors="#555555")
    ax.legend(loc="upper left", facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.set_xlabel("Coordenada X", color="#888888")
    ax.set_ylabel("Coordenada Y", color="#888888")
    plt.tight_layout()
    plt.savefig("data/flow_map.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("[VIZ] Mapa de flujos guardado en data/flow_map.png")


def plot_cost_distribution(result: dict):
    """Visualiza distribución de costos por almacén."""
    data  = result["data"]
    flows = result["flows"]
    costs = data["costs"]

    cost_by_warehouse = [(flows[i] * costs[i]).sum() for i in range(data["n_warehouses"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    colors = ["#00bfff", "#00e676", "#ff6b6b", "#ffd93d", "#c77dff"]

    # Barras: costo por almacén
    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")
    bars = ax1.bar(data["warehouse_names"], cost_by_warehouse, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("Costo Total por Almacén", color="white", fontsize=12)
    ax1.set_ylabel("Costo", color="#888888")
    ax1.tick_params(colors="#888888")
    for bar, val in zip(bars, cost_by_warehouse):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val:,.0f}", ha="center", color="white", fontsize=8)

    # Pie: proporción de costo
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    wedges, texts, autotexts = ax2.pie(
        cost_by_warehouse, labels=data["warehouse_names"],
        colors=colors, autopct="%1.1f%%", startangle=140,
        textprops={"color": "white", "fontsize": 8}
    )
    ax2.set_title("Proporción de Costo por Almacén", color="white", fontsize=12)

    plt.tight_layout()
    plt.savefig("data/cost_distribution.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("[VIZ] Distribución de costos guardada en data/cost_distribution.png")


def plot_sensitivity(df_sensitivity: pd.DataFrame):
    """Visualiza el análisis de sensibilidad."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#1a1a2e")

    costs  = df_sensitivity["Costo óptimo"].values
    labels = df_sensitivity["Escala demanda"].values
    deltas = df_sensitivity["Δ costo vs base (%)"].values

    color_bars = ["#00e676" if d <= 0 else "#ff6b6b" for d in deltas]
    bars = ax.bar(labels, costs, color=color_bars, edgecolor="white", linewidth=0.5)

    for bar, cost, delta in zip(bars, costs, deltas):
        label = f"{cost:,.0f}\n(+{delta:.1f}%)" if delta > 0 else f"{cost:,.0f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cost * 0.01,
                label, ha="center", color="white", fontsize=9)

    ax.set_title("Análisis de Sensibilidad — Incremento de Demanda", color="white", fontsize=13)
    ax.set_xlabel("Escala de Demanda", color="#888888")
    ax.set_ylabel("Costo Óptimo", color="#888888")
    ax.tick_params(colors="#888888")
    plt.tight_layout()
    plt.savefig("data/sensitivity_analysis.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("[VIZ] Análisis de sensibilidad guardado en data/sensitivity_analysis.png")


# ─────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────

def run_optimization_pipeline(seed: int = SEED) -> dict:
    """
    Ejecuta el pipeline completo del Módulo 01:
    1. Genera datos
    2. Resuelve el modelo base
    3. Análisis de sensibilidad (+10%, +20%, +30%)
    4. Genera visualizaciones

    Returns:
        dict con result_base y df_sensitivity.
    """
    print("=" * 55)
    print("  MÓDULO 01 — OPTIMIZACIÓN DE TRANSPORTE")
    print("=" * 55)

    data        = generate_data(seed=seed)
    result_base = solve_transport(data, demand_scale=1.0)
    df_sens     = sensitivity_analysis(data, scales=[1.0, 1.1, 1.2, 1.3])

    plot_flow_map(result_base)
    plot_cost_distribution(result_base)
    plot_sensitivity(df_sens)

    print("\n Pipeline Módulo 01 completado.")
    return {"result_base": result_base, "df_sensitivity": df_sens}


if __name__ == "__main__":
    run_optimization_pipeline()
