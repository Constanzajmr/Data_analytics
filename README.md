# Logistics Challenge — Sistema Integrado de Optimización Logística

**Evaluación Técnica · Proceso de Selección**

Sistema computacional que resuelve tres decisiones simultáneas en una red logística industrial: cuánto producir/surtir, cómo asignar rutas y cuándo alertar proactivamente a clientes ante posibles retrasos.

---

## Estructura del Repositorio

```
logistics-challenge/
├── notebooks/
│   ├── modulo_01.ipynb             # M01: Optimización de Transporte
│   ├── modulo_02.ipynb             # M02: Machine Learning
│   ├── modulo_03_conectado.ipynb   # M03: Agente Inteligente
│   
├── src/
│   ├── optimization.py          # M01: Problema de Transporte con OR-Tools
│   ├── ml_models.py             # M02: Regresión + Clasificación + SHAP
│   └── agent.py                 # M03: Agente ReAct con LangChain
├── tests/
│   ├── test_optimization.py     # 18 tests — M01
│   ├── test_ml_models.py        # 22 tests — M02
│   └── test_agent.py            # 34 tests — M03
├── requirements.txt
├── report.pdf
└── README.md
```

---

## Cómo Reproducir

### Opción A — Google Colab (recomendado y usado para la creación de este código)
1. Abrir cada notebook en [colab.research.google.com](https://colab.research.google.com)
2. `Archivo → Subir notebook` → seleccionar el `.ipynb`
3. `Runtime → Run all` — la primera celda instala dependencias automáticamente

> El orden recomendado es: `modulo_01` → `modulo_02` → `modulo_03`

### Opción B — Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Constanzajmr/Data_analytics.git
cd Data_analytics

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar módulos
python src/optimization.py   # Módulo 01
python src/ml_models.py      # Módulo 02
python src/agent.py          # Módulo 03 (conectado)

# 4. Correr todos los tests
python -m pytest tests/ -v
```

---

## Módulos del Sistema

### Módulo 01 — Optimización de Transporte (`optimization.py` · `modulo_01.ipynb`)

Minimiza el costo total de distribución desde **5 almacenes** a **20 puntos de demanda** usando programación lineal.

- **Solver:** OR-Tools GLOP
- **Restricciones:** capacidad por almacén [500–2000 uds] + demanda por punto [50–300 uds]
- **Análisis de sensibilidad:** escenarios +10%, +20%, +30% sobre la demanda base

| Escenario | Demanda total | Costo óptimo | Δ vs base |
|-----------|--------------|--------------|-----------|
| Base (100%) | 3,541 uds | $83,980 | — |
| +10% | 3,885 uds | $95,360 | +13.6% |
| +20% | 4,241 uds | $107,164 | +27.6% |
| +30% | 4,595 uds | $119,367 | +42.1% |

**Visualizaciones:** mapa de flujos, distribución de costos por almacén, gráfica de sensibilidad.

---

### Módulo 02 — Machine Learning (`ml_models.py` · `modulo_02.ipynb`)

**2A — Regresión (Forecast de demanda)**
- Serie temporal sintética de 52 semanas conectada a los 20 puntos del M01
- Features: lags (1–4), rolling mean/std (3–4 semanas), variables cíclicas sin/cos
- Validación: walk-forward sin data leakage (train semanas 1–42 / test 43–52)
- Modelos: Random Forest + XGBoost

| Modelo | MAE | RMSE | MAPE |
|--------|-----|------|------|
| Random Forest | 13.87 | 18.36 | 7.95% |
| XGBoost | 14.40 | 18.82 | 8.43% |

**2B — Clasificación (Riesgo de desabasto)**
- Features: `stock_actual`, `lead_time`, `demanda_proyectada`, `distancia_almacen`, `cobertura`
- Balanceo de clases: SMOTE (67/33 → 50/50 en train)
- Modelo: Random Forest

| Clasificador | ROC-AUC | F1 |
|-------------|---------|-----|
| RF + SMOTE | 0.9186 | 0.8219 |

**2C — Interpretabilidad**
- SHAP values para Random Forest y XGBoost (regresión)
- Feature importance para el clasificador de riesgo
- Variable más importante en regresión: `roll_mean_4` (promedio móvil 4 semanas)
- Variable más importante en clasificación: `cobertura` (stock/demanda proyectada)

---

### Módulo 03 — Agente Inteligente (`agent.py` · `modulo_03.ipynb`)

Agente con loop ReAct que monitorea el sistema, detecta anomalías y genera alertas proactivas.

**Loop ReAct — 4 fases:**
```
OBSERVAR → RAZONAR (Forecast) → RAZONAR (Optimización) → ACTUAR → STOP
```

**Herramientas LangChain (`@tool`):**

| Tool | Descripción |
|------|-------------|
| `get_stock_status(warehouse_id)` | Retorna inventario actual del almacén |
| `get_demand_forecast(point_id, weeks)` | Forecast de demanda usando modelo del M02 |
| `run_optimization(scenario)` | Ejecuta OR-Tools del M01 para el escenario dado |
| `send_alert(point_id, message, severity)` | Simula envío de notificación proactiva |

**Criterio de parada:** todos los puntos en riesgo han sido atendidos (0 pendientes).

**Trazabilidad:** log completo de razonamiento por cada paso (timestamp, thought, action, result).

---

## Tests

```bash
python -m pytest tests/ -v
# 74 passed in 7.62s
```

| Archivo | Tests | Qué verifica |
|---------|-------|-------------|
| `test_optimization.py` | 18 | Dimensiones, rangos, reproducibilidad, factibilidad, capacidades, sensibilidad |
| `test_ml_models.py` | 22 | Serie temporal, no-leakage en lags, ROC-AUC ≥ 0.70, F1 ≥ 0.60, SMOTE |
| `test_agent.py` | 34 | SystemState, 4 tools, loop ReAct, criterio de parada, reasoning log |

---

## Stack Tecnológico

| Categoría | Librería |
|-----------|----------|
| Optimización | `ortools` (GLOP solver) |
| ML Regresión | `scikit-learn` (Random Forest), `xgboost` |
| Balanceo | `imbalanced-learn` (SMOTE) |
| Interpretabilidad | `shap` |
| Agente | `langchain-core` |
| Visualización | `matplotlib` (paleta Ternium) |
| Datos | `numpy`, `pandas` |

---

## Decisiones de Diseño en términos de aplicación

**¿Por qué OR-Tools GLOP?**
Solver lineal nativo de OR-Tools, más rápido que CBC para problemas de transporte continuos sin variables enteras.

**¿Por qué Random Forest sobre XGBoost en el agente?**
MAPE ligeramente menor (7.95% vs 8.43%) y más robusto ante datos fuera del rango de entrenamiento.

**¿Por qué SMOTE?**
El dataset genera ~67% clase 0 / 33% clase 1. SMOTE balancea el set de entrenamiento sintéticamente sin contaminar el test set.

**¿Por qué LangChain `@tool` sin LLM externo?**
El decorador `@tool` expone cada función con su docstring como descripción para cualquier LLM compatible. Para producción, se reemplaza el loop manual por `create_react_agent(llm, tools, prompt)` sin cambiar las herramientas.

---

## Reproducibilidad

Todos los módulos usan `SEED = 42`:

```python
np.random.default_rng(42)             # generación de datos
RandomForestRegressor(random_state=42)
XGBRegressor(random_state=42)
SMOTE(random_state=42)
train_test_split(..., random_state=42)
```

---



**Constanza** · [GitHub](https://github.com/Constanzajmr/Data_analytics)
