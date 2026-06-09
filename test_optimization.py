"""
Tests — Módulo 01: Optimización
Basado en modulo_01.ipynb (versión modificada por el usuario)
"""
import sys
sys.path.append('src')
import numpy as np
import pandas as pd
import pytest
from optimization import generate_data, solve_transport, sensitivity_analysis

SEED = 42

# ── generate_data ────────────────────────────────────

def test_generate_data_n_warehouses():
    """Genera exactamente 5 almacenes."""
    data = generate_data(SEED)
    assert data['n_warehouses'] == 5

def test_generate_data_n_demand_points():
    """Genera exactamente 20 puntos de demanda."""
    data = generate_data(SEED)
    assert data['n_demand_points'] == 20

def test_generate_data_costs_shape():
    """La matriz de costos tiene forma (5, 20)."""
    data = generate_data(SEED)
    assert data['costs'].shape == (5, 20)

def test_generate_data_capacities_range():
    """Capacidades dentro del rango [500, 2000] especificado."""
    data = generate_data(SEED)
    assert data['capacities'].min() >= 500
    assert data['capacities'].max() <= 2000

def test_generate_data_demands_range():
    """Demandas dentro del rango [50, 300] especificado."""
    data = generate_data(SEED)
    assert data['demands'].min() >= 50
    assert data['demands'].max() <= 300

def test_generate_data_costs_positive():
    """Todos los costos de transporte son positivos."""
    data = generate_data(SEED)
    assert (data['costs'] > 0).all()

def test_generate_data_reproducible():
    """Mismo seed produce datos idénticos (reproducibilidad)."""
    d1 = generate_data(SEED)
    d2 = generate_data(SEED)
    np.testing.assert_array_equal(d1['costs'],      d2['costs'])
    np.testing.assert_array_equal(d1['capacities'], d2['capacities'])
    np.testing.assert_array_equal(d1['demands'],    d2['demands'])

def test_generate_data_different_seeds():
    """Seeds distintos producen datos distintos."""
    d1 = generate_data(42)
    d2 = generate_data(99)
    assert not np.array_equal(d1['costs'], d2['costs'])

# ── solve_transport ──────────────────────────────────

def test_solve_transport_status_optimal():
    """El solver alcanza estado ÓPTIMO para el escenario base."""
    data   = generate_data(SEED)
    result = solve_transport(data, demand_scale=1.0)
    assert result['status'] == 'ÓPTIMO'

def test_solve_transport_cost_positive():
    """El costo óptimo es positivo."""
    data   = generate_data(SEED)
    result = solve_transport(data)
    assert result['total_cost'] > 0

def test_solve_transport_flows_shape():
    """La matriz de flujos tiene forma (5, 20)."""
    data   = generate_data(SEED)
    result = solve_transport(data)
    assert result['flows'].shape == (5, 20)

def test_solve_transport_flows_non_negative():
    """Todos los flujos son no negativos."""
    data   = generate_data(SEED)
    result = solve_transport(data)
    assert (result['flows'] >= 0).all()

def test_solve_transport_capacity_respected():
    """Ningún almacén supera su capacidad."""
    data   = generate_data(SEED)
    result = solve_transport(data)
    for i in range(data['n_warehouses']):
        assert result['flows'][i].sum() <= data['capacities'][i] + 0.01

def test_solve_transport_demand_satisfied():
    """Todos los puntos de demanda son abastecidos."""
    data   = generate_data(SEED)
    result = solve_transport(data)
    for j in range(data['n_demand_points']):
        assert result['flows'][:, j].sum() >= data['demands'][j] - 0.01

# ── sensitivity_analysis ─────────────────────────────

def test_sensitivity_returns_dataframe():
    """sensitivity_analysis retorna un DataFrame."""
    data = generate_data(SEED)
    df   = sensitivity_analysis(data, scales=[1.0, 1.2])
    assert isinstance(df, pd.DataFrame)

def test_sensitivity_correct_rows():
    """El DataFrame tiene una fila por escenario."""
    data = generate_data(SEED)
    df   = sensitivity_analysis(data, scales=[1.0, 1.1, 1.2])
    assert len(df) == 3

def test_sensitivity_cost_increases_with_demand():
    """El costo crece al aumentar la demanda."""
    data        = generate_data(SEED)
    df          = sensitivity_analysis(data, scales=[1.0, 1.2])
    cost_base   = df[df['Escala demanda'] == '100%']['Costo óptimo'].values[0]
    cost_stress = df[df['Escala demanda'] == '120%']['Costo óptimo'].values[0]
    assert cost_stress > cost_base

def test_sensitivity_delta_column_exists():
    """El DataFrame incluye la columna de delta porcentual."""
    data = generate_data(SEED)
    df   = sensitivity_analysis(data, scales=[1.0, 1.2])
    assert 'Δ costo vs base (%)' in df.columns
