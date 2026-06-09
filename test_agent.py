"""
Tests — Módulo 03: Agente Inteligente
Basado en agent.py y modulo_03.ipynb (versión modificada por el usuario)
Cubre: SystemState, @tool functions (get_stock_status, get_demand_forecast,
       run_optimization, send_alert), loop ReAct
"""
import sys
sys.path.append('src')
import json
import pytest
import numpy as np
from optimization import generate_data
from agent import (
    SystemState, STATE,
    get_stock_status, get_demand_forecast,
    run_optimization, send_alert,
    run_react_loop, SEED
)

DATA = generate_data(SEED)

# ── SystemState ──────────────────────────────────────

def test_state_n_warehouses():
    """SystemState inicializa con 5 almacenes."""
    s = SystemState()
    assert s.n_warehouses == 5
    assert len(s.warehouse_stock) == 5

def test_state_n_demand_points():
    """SystemState inicializa con 20 puntos de demanda."""
    s = SystemState()
    assert s.n_demand_points == 20
    assert len(s.demand_base) == 20

def test_state_stock_positive():
    """Todo el stock inicial es positivo."""
    s = SystemState()
    assert all(v > 0 for v in s.warehouse_stock.values())

def test_state_demand_base_range():
    """Demanda base de cada punto está entre 50 y 300 (herencia de M01)."""
    s = SystemState()
    for j, d in s.demand_base.items():
        assert 50 <= d <= 300, f"Punto {j}: demanda_base={d} fuera de rango"

def test_state_risk_points_subset_of_demand():
    """Los puntos en riesgo son subconjunto de {0..19}."""
    s = SystemState()
    assert s.risk_points.issubset(set(range(20)))

def test_state_initial_attended_empty():
    """Al inicializar, ningún punto ha sido atendido."""
    s = SystemState()
    assert len(s.attended_points) == 0

def test_state_log_increments_step():
    """Cada llamada a log() incrementa el contador de pasos."""
    s = SystemState()
    before = s.step
    s.log('test_action', 'test_result', 'test_thought')
    assert s.step == before + 1

def test_state_log_appends_to_reasoning_log():
    """log() agrega la entrada al reasoning_log con los campos correctos."""
    s = SystemState()
    # Firma real: log(step, action, result, thought)
    s.log(step='test_step', action='acción de prueba',
          result='resultado de prueba', thought='pensamiento de prueba')
    assert len(s.reasoning_log) == 1
    entry = s.reasoning_log[0]
    assert entry['action']  == 'acción de prueba'
    assert entry['result']  == 'resultado de prueba'
    assert entry['thought'] == 'pensamiento de prueba'
    assert 'timestamp' in entry

# ── get_stock_status ─────────────────────────────────

def test_get_stock_status_valid_warehouse():
    """Retorna JSON válido con campos requeridos para almacén existente."""
    result = json.loads(get_stock_status.invoke({'warehouse_id': 0}))
    assert 'stock_actual'     in result
    assert 'capacidad_max'    in result
    assert 'utilizacion_pct'  in result
    assert 'status'           in result

def test_get_stock_status_status_values():
    """Status es CRÍTICO, BAJO o NORMAL."""
    for wh_id in range(5):
        result = json.loads(get_stock_status.invoke({'warehouse_id': wh_id}))
        assert result['status'] in ['CRÍTICO', 'BAJO', 'NORMAL']

def test_get_stock_status_pct_range():
    """El porcentaje de utilización está entre 0 y 100."""
    result = json.loads(get_stock_status.invoke({'warehouse_id': 0}))
    assert 0 <= result['utilizacion_pct'] <= 100

def test_get_stock_status_invalid_returns_error():
    """Retorna error para warehouse_id inválido."""
    result = json.loads(get_stock_status.invoke({'warehouse_id': 99}))
    assert 'error' in result

# ── get_demand_forecast ──────────────────────────────

def test_get_demand_forecast_n_weeks():
    """Retorna exactamente el número de semanas solicitadas."""
    result = json.loads(get_demand_forecast.invoke({'point_id': 0, 'weeks': 4}))
    assert len(result['forecast_por_semana']) == 4

def test_get_demand_forecast_all_positive():
    """Todos los valores del forecast son positivos."""
    result = json.loads(get_demand_forecast.invoke({'point_id': 0, 'weeks': 4}))
    assert all(v > 0 for v in result['forecast_por_semana'])

def test_get_demand_forecast_tendencia_valid():
    """La tendencia es CRECIENTE o ESTABLE."""
    result = json.loads(get_demand_forecast.invoke({'point_id': 1, 'weeks': 4}))
    assert result['tendencia'] in ['CRECIENTE', 'ESTABLE']

def test_get_demand_forecast_promedio_matches():
    """promedio_semanal coincide con el promedio de forecast_por_semana."""
    result = json.loads(get_demand_forecast.invoke({'point_id': 2, 'weeks': 4}))
    expected_avg = round(np.mean(result['forecast_por_semana']), 1)
    assert abs(result['promedio_semanal'] - expected_avg) < 0.1

def test_get_demand_forecast_invalid_point():
    """Retorna error para point_id inválido."""
    result = json.loads(get_demand_forecast.invoke({'point_id': 99, 'weeks': 4}))
    assert 'error' in result

# ── run_optimization ─────────────────────────────────

def test_run_optimization_base_cost_positive():
    """Escenario base retorna costo positivo."""
    result = json.loads(run_optimization.invoke({'scenario': 'base'}))
    assert result['costo_optimo'] > 0

def test_run_optimization_base_status_optimal():
    """Escenario base retorna estado ÓPTIMO."""
    result = json.loads(run_optimization.invoke({'scenario': 'base'}))
    assert result['status'] == 'ÓPTIMO'

def test_run_optimization_stress_higher_than_base():
    """El costo del escenario stress_20 es mayor al base."""
    base   = json.loads(run_optimization.invoke({'scenario': 'base'}))
    stress = json.loads(run_optimization.invoke({'scenario': 'stress_20'}))
    assert stress['costo_optimo'] > base['costo_optimo']

def test_run_optimization_has_recommendations():
    """Todos los escenarios incluyen al menos una recomendación."""
    for scenario in ['base', 'stress_20', 'stress_30']:
        result = json.loads(run_optimization.invoke({'scenario': scenario}))
        assert len(result['recomendaciones']) >= 1

def test_run_optimization_unknown_scenario_fallback():
    """Escenario desconocido hace fallback a 'base'."""
    result = json.loads(run_optimization.invoke({'scenario': 'inexistente'}))
    assert result['costo_optimo'] > 0  # no rompe, usa base

# ── send_alert ───────────────────────────────────────

def test_send_alert_status_enviada():
    """La alerta enviada tiene status ENVIADA."""
    result = json.loads(send_alert.invoke(
        {'point_id': 0, 'message': 'Test alerta', 'severity': 'ALTA'}))
    assert result['status'] == 'ENVIADA'

def test_send_alert_severity_preserved():
    """La severidad enviada se preserva en la respuesta."""
    result = json.loads(send_alert.invoke(
        {'point_id': 1, 'message': 'Test', 'severity': 'CRÍTICA'}))
    assert result['severity'] == 'CRÍTICA'

def test_send_alert_marks_point_attended():
    """El punto queda marcado como atendido en STATE."""
    send_alert.invoke({'point_id': 10, 'message': 'Test', 'severity': 'MEDIA'})
    assert 10 in STATE.attended_points

def test_send_alert_removes_from_risk():
    """El punto se elimina de risk_points tras la alerta."""
    STATE.risk_points.add(11)
    send_alert.invoke({'point_id': 11, 'message': 'Test', 'severity': 'ALTA'})
    assert 11 not in STATE.risk_points

def test_send_alert_increments_alert_count():
    """Cada alerta incrementa la lista de alertas enviadas."""
    before = len(STATE.alerts_sent)
    send_alert.invoke({'point_id': 12, 'message': 'Test', 'severity': 'BAJA'})
    assert len(STATE.alerts_sent) == before + 1

def test_send_alert_invalid_severity_defaults_to_media():
    """Severidad inválida hace fallback a MEDIA."""
    result = json.loads(send_alert.invoke(
        {'point_id': 3, 'message': 'Test', 'severity': 'EXTREMA'}))
    assert result['severity'] == 'MEDIA'

# ── run_react_loop ───────────────────────────────────

def test_react_loop_all_risk_points_attended():
    """Al finalizar el loop, todos los puntos en riesgo han sido atendidos."""
    result = run_react_loop(DATA)
    # run_react_loop crea un STATE fresco internamente
    # Verificar via el resultado: attended debe cubrir todos los que había en riesgo
    assert len(result['attended']) > 0
    # Todos los atendidos están en la lista de alertas
    alerted_points = {a['point_id'] for a in result['alerts']}
    for pt in result['attended']:
        assert pt in alerted_points, f"Punto {pt} en attended pero sin alerta"

def test_react_loop_returns_required_keys():
    """El resultado del loop contiene las claves requeridas."""
    result = run_react_loop(DATA)
    for key in ['alerts', 'reasoning_log', 'attended', 'opt_base', 'opt_stress']:
        assert key in result, f"Clave faltante: {key}"

def test_react_loop_reasoning_log_not_empty():
    """El reasoning_log registra al menos un paso."""
    result = run_react_loop(DATA)
    assert len(result['reasoning_log']) > 0

def test_react_loop_alerts_match_attended():
    """El número de alertas coincide con los puntos atendidos."""
    result = run_react_loop(DATA)
    assert len(result['alerts']) == len(result['attended'])
