"""
Tests — Módulo 02: Machine Learning
Basado en ml_models.py y modulo_02.ipynb (versión modificada por el usuario)
Funciones: generate_time_series, build_regression_features, walk_forward_validation,
           generate_classification_data, train_classifier
"""
import sys
sys.path.append('src')
import numpy as np
import pandas as pd
import pytest
from optimization import generate_data
from ml_models import (
    generate_time_series, build_regression_features,
    walk_forward_validation, generate_classification_data,
    train_classifier, REGRESSION_FEATURES, CLASS_FEATURES, SEED
)

DATA = generate_data(SEED)

# ── generate_time_series ─────────────────────────────

def test_time_series_total_rows():
    """52 semanas × 20 puntos = 1040 filas."""
    df = generate_time_series(DATA)
    assert len(df) == 52 * 20

def test_time_series_n_points():
    """La serie tiene exactamente 20 puntos únicos."""
    df = generate_time_series(DATA)
    assert df['punto_id'].nunique() == 20

def test_time_series_n_weeks():
    """La serie tiene exactamente 52 semanas."""
    df = generate_time_series(DATA)
    assert df['semana'].nunique() == 52

def test_time_series_demand_positive():
    """Toda la demanda generada es positiva."""
    df = generate_time_series(DATA)
    assert (df['demanda'] > 0).all()

def test_time_series_base_demand_connected_to_m01():
    """base_demand refleja las demandas originales del Módulo 01."""
    df = generate_time_series(DATA)
    for j in range(DATA['n_demand_points']):
        base_in_ts = df[df['punto_id'] == j]['base_demand'].iloc[0]
        assert base_in_ts == DATA['demands'][j], \
            f"Punto {j}: base_demand={base_in_ts} != M01 demand={DATA['demands'][j]}"

def test_time_series_reproducible():
    """Mismo seed → misma serie temporal."""
    df1 = generate_time_series(DATA, seed=SEED)
    df2 = generate_time_series(DATA, seed=SEED)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                   df2.reset_index(drop=True))

# ── build_regression_features ────────────────────────

def test_regression_features_all_columns_present():
    """El dataset de features contiene todas las columnas de REGRESSION_FEATURES."""
    df      = generate_time_series(DATA)
    df_feat = build_regression_features(df)
    for col in REGRESSION_FEATURES:
        assert col in df_feat.columns, f"Columna faltante: {col}"

def test_regression_features_no_nan():
    """No hay NaN en el dataset de features tras dropna."""
    df      = generate_time_series(DATA)
    df_feat = build_regression_features(df)
    assert df_feat[REGRESSION_FEATURES].isna().sum().sum() == 0

def test_regression_features_lag1_no_leakage():
    """lag_1 corresponde a la demanda de la semana anterior (sin leakage)."""
    df      = generate_time_series(DATA)
    df_feat = build_regression_features(df)
    pt0     = df_feat[df_feat['punto_id'] == 0].sort_values('semana')
    orig    = df[df['punto_id'] == 0].sort_values('semana')
    row     = pt0.iloc[0]
    sem     = row['semana']
    prev    = orig[orig['semana'] == sem - 1]['demanda'].values
    if len(prev) > 0:
        assert abs(row['lag_1'] - prev[0]) < 0.01

def test_regression_features_semana_sin_cos_range():
    """semana_sin y semana_cos están en el rango [-1, 1]."""
    df      = generate_time_series(DATA)
    df_feat = build_regression_features(df)
    assert df_feat['semana_sin'].between(-1, 1).all()
    assert df_feat['semana_cos'].between(-1, 1).all()

def test_regression_features_trimestre_range():
    """Trimestre tiene valores entre 1 y 4."""
    df      = generate_time_series(DATA)
    df_feat = build_regression_features(df)
    assert df_feat['trimestre'].between(1, 4).all()

# ── walk_forward_validation ──────────────────────────

def test_walk_forward_returns_two_models():
    """walk_forward_validation retorna resultados para Random Forest y XGBoost."""
    df      = generate_time_series(DATA)
    results = walk_forward_validation(df)
    assert 'Random Forest' in results
    assert 'XGBoost' in results

def test_walk_forward_metrics_present():
    """Cada modelo incluye MAE, RMSE y MAPE."""
    df      = generate_time_series(DATA)
    results = walk_forward_validation(df)
    for model_name, res in results.items():
        assert 'mae'  in res, f"{model_name}: falta mae"
        assert 'rmse' in res, f"{model_name}: falta rmse"
        assert 'mape' in res, f"{model_name}: falta mape"

def test_walk_forward_mape_acceptable():
    """MAPE de ambos modelos está por debajo del 20% (umbral mínimo aceptable)."""
    df      = generate_time_series(DATA)
    results = walk_forward_validation(df)
    for model_name, res in results.items():
        assert res['mape'] < 20, f"{model_name}: MAPE={res['mape']}% >= 20%"

def test_walk_forward_mae_positive():
    """MAE es positivo para ambos modelos."""
    df      = generate_time_series(DATA)
    results = walk_forward_validation(df)
    for model_name, res in results.items():
        assert res['mae'] > 0

# ── generate_classification_data ────────────────────

def test_classification_data_rows():
    """El dataset de clasificación tiene 600 registros."""
    df = generate_classification_data(DATA, {})
    assert len(df) == 600

def test_classification_features_present():
    """Contiene todas las features de CLASS_FEATURES."""
    df = generate_classification_data(DATA, {})
    for feat in CLASS_FEATURES:
        assert feat in df.columns, f"Feature faltante: {feat}"

def test_classification_target_binary():
    """El target riesgo_alto es binario (0 o 1)."""
    df = generate_classification_data(DATA, {})
    assert set(df['riesgo_alto'].unique()).issubset({0, 1})

def test_classification_both_classes():
    """Ambas clases están presentes en el dataset."""
    df = generate_classification_data(DATA, {})
    assert df['riesgo_alto'].sum() > 0
    assert (df['riesgo_alto'] == 0).sum() > 0

def test_classification_cobertura_formula():
    """La cobertura = stock_actual / demanda_proyectada."""
    df = generate_classification_data(DATA, {})
    computed = (df['stock_actual'] / (df['demanda_proyectada'] + 1e-6)).round(3)
    pd.testing.assert_series_equal(computed, df['cobertura'], check_names=False,
                                    rtol=0.01)

# ── train_classifier ─────────────────────────────────

def test_classifier_roc_auc_above_070():
    """ROC-AUC ≥ 0.70 (umbral mínimo aceptable)."""
    df     = generate_classification_data(DATA, {})
    result = train_classifier(df)
    assert result['roc_auc'] >= 0.70, f"ROC-AUC={result['roc_auc']} < 0.70"

def test_classifier_f1_above_060():
    """F1 ≥ 0.60."""
    df     = generate_classification_data(DATA, {})
    result = train_classifier(df)
    assert result['f1'] >= 0.60, f"F1={result['f1']} < 0.60"

def test_classifier_confusion_matrix_shape():
    """La matriz de confusión tiene forma (2, 2)."""
    df     = generate_classification_data(DATA, {})
    result = train_classifier(df)
    assert result['cm'].shape == (2, 2)

def test_classifier_predictions_binary():
    """Las predicciones son 0 o 1."""
    df     = generate_classification_data(DATA, {})
    result = train_classifier(df)
    assert set(result['preds']).issubset({0, 1})
