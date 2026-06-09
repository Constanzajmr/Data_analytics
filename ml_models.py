"""
Módulo 02 — Machine Learning
2A: Regresión — Forecast de demanda (Random Forest + XGBoost)
2B: Clasificación — Riesgo de desabasto (con SMOTE)
2C: Interpretabilidad — SHAP values + Feature Importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             confusion_matrix, roc_auc_score, f1_score,
                             ConfusionMatrixDisplay, roc_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

SEED = 42

# ── Paleta Ternium ──────────────────────────────
TERNIUM = {
    "orange":     "#F5A800",
    "red":        "#E3000F",
    "black":      "#1A1A1A",
    "gray":       "#6B6B6B",
    "light_gray": "#D6D6D6",
    "dark_gray":  "#3D3D3D",
    "bg":         "#FFFFFF",
    "grid":       "#EEEEEE",
}
PALETTE = ["#F5A800", "#E3000F", "#3D3D3D", "#6B6B6B", "#D6D6D6"]

def _base_style(ax, fig):
    ax.set_facecolor(TERNIUM["bg"])
    fig.patch.set_facecolor(TERNIUM["bg"])
    ax.grid(color=TERNIUM["grid"], linewidth=0.7, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TERNIUM["light_gray"])
    ax.spines["bottom"].set_color(TERNIUM["light_gray"])
    ax.tick_params(colors=TERNIUM["gray"], labelsize=9)
    ax.xaxis.label.set_color(TERNIUM["dark_gray"])
    ax.yaxis.label.set_color(TERNIUM["dark_gray"])


# ═══════════════════════════════════════════════
# 2A — GENERACIÓN DE SERIE TEMPORAL
# ═══════════════════════════════════════════════

def generate_time_series(data_m01: dict, n_weeks: int = 52, seed: int = SEED) -> pd.DataFrame:
    """
    Genera series temporales de demanda para los 20 puntos del Módulo 01.
    Incluye tendencia, estacionalidad semanal y ruido aleatorio.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(1, n_weeks + 1)
    records = []

    base_demands = data_m01["demands"]          # demanda base de cada punto
    demand_names = data_m01["demand_names"]

    for idx, (name, base) in enumerate(zip(demand_names, base_demands)):
        # Tendencia lineal leve
        trend = base + weeks * rng.uniform(0.1, 0.5)
        # Estacionalidad: pico en semanas 10-20 y 40-50 (temporadas)
        seasonality = 15 * np.sin(2 * np.pi * weeks / 52) + \
                      8  * np.sin(4 * np.pi * weeks / 52)
        # Ruido gaussiano
        noise = rng.normal(0, base * 0.08, size=n_weeks)
        demand = np.clip(trend + seasonality + noise, 10, None).round(1)

        for w, d in zip(weeks, demand):
            records.append({
                "punto":        name,
                "punto_id":     idx,
                "semana":       w,
                "demanda":      d,
                "base_demand":  base,
            })

    df = pd.DataFrame(records)
    print(f"[DATA] Serie temporal generada: {len(df)} registros | "
          f"{df['punto'].nunique()} puntos × {n_weeks} semanas")
    return df


# ═══════════════════════════════════════════════
# 2A — FEATURE ENGINEERING PARA REGRESIÓN
# ═══════════════════════════════════════════════

def build_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features temporales para el forecast de demanda.
    """
    df = df.sort_values(["punto_id", "semana"]).copy()

    # Lags
    for lag in [1, 2, 3, 4]:
        df[f"lag_{lag}"] = df.groupby("punto_id")["demanda"].shift(lag)

    # Rolling stats
    for w in [3, 4]:
        df[f"roll_mean_{w}"] = df.groupby("punto_id")["demanda"] \
                                  .transform(lambda x: x.shift(1).rolling(w).mean())
        df[f"roll_std_{w}"]  = df.groupby("punto_id")["demanda"] \
                                  .transform(lambda x: x.shift(1).rolling(w).std())

    # Features temporales
    df["semana_sin"] = np.sin(2 * np.pi * df["semana"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana"] / 52)
    df["trimestre"]  = ((df["semana"] - 1) // 13 + 1).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


REGRESSION_FEATURES = [
    "punto_id", "semana", "base_demand",
    "lag_1", "lag_2", "lag_3", "lag_4",
    "roll_mean_3", "roll_mean_4", "roll_std_3", "roll_std_4",
    "semana_sin", "semana_cos", "trimestre"
]


# ═══════════════════════════════════════════════
# 2A — WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════

def walk_forward_validation(df: pd.DataFrame, n_test_weeks: int = 10) -> dict:
    """
    Validación walk-forward (sin data leakage).
    Entrena con semanas 1..T y predice semana T+1 iterativamente.
    Evalúa Random Forest y XGBoost.
    """
    df_feat = build_regression_features(df)
    weeks   = sorted(df_feat["semana"].unique())
    cutoff  = weeks[-n_test_weeks]

    train = df_feat[df_feat["semana"] <  cutoff]
    test  = df_feat[df_feat["semana"] >= cutoff]

    X_train = train[REGRESSION_FEATURES]
    y_train = train["demanda"]
    X_test  = test[REGRESSION_FEATURES]
    y_test  = test["demanda"]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=SEED, verbosity=0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results[name] = {
            "model":  model,
            "preds":  preds,
            "y_test": y_test.values,
            "mae":    round(mae,  2),
            "rmse":   round(rmse, 2),
            "mape":   round(mape, 2),
            "test_df": test,
        }
        print(f"[{name}] MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")

    return results


# ═══════════════════════════════════════════════
# 2B — GENERACIÓN DATOS CLASIFICACIÓN
# ═══════════════════════════════════════════════

def generate_classification_data(data_m01: dict, reg_results: dict,
                                  seed: int = SEED) -> pd.DataFrame:
    """
    Genera dataset de clasificación de riesgo de desabasto.
    Features: stock_actual, lead_time, demanda_proyectada, distancia_almacen
    Target: riesgo_alto (1) / riesgo_bajo (0)
    """
    rng  = np.random.default_rng(seed)
    n    = 600   # muestras sintéticas
    dems = data_m01["demands"]
    coords_w = data_m01["warehouse_coords"]
    coords_d = data_m01["demand_coords"]

    # Distancia promedio de cada punto al almacén más cercano
    dist_to_nearest = []
    for j in range(data_m01["n_demand_points"]):
        dists = [np.linalg.norm(coords_w[i] - coords_d[j])
                 for i in range(data_m01["n_warehouses"])]
        dist_to_nearest.append(min(dists))
    avg_dist = np.mean(dist_to_nearest)

    stock_actual       = rng.uniform(50,  500, n)
    lead_time          = rng.uniform(1,   14,  n)
    demanda_proyectada = rng.uniform(50,  350, n)
    distancia_almacen  = rng.uniform(min(dist_to_nearest), max(dist_to_nearest) * 1.2, n)
    cobertura          = stock_actual / (demanda_proyectada + 1e-6)

    # Regla de negocio: riesgo alto si cobertura < 1.2 Y lead_time > 7
    #                   o distancia alta y stock bajo
    riesgo = (
        ((cobertura < 1.2) & (lead_time > 7)) |
        ((distancia_almacen > avg_dist * 1.1) & (stock_actual < 150))
    ).astype(int)

    # Añadir algo de ruido al target para hacerlo más realista
    flip_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    riesgo[flip_idx] = 1 - riesgo[flip_idx]

    df = pd.DataFrame({
        "stock_actual":       stock_actual.round(1),
        "lead_time":          lead_time.round(1),
        "demanda_proyectada": demanda_proyectada.round(1),
        "distancia_almacen":  distancia_almacen.round(2),
        "cobertura":          cobertura.round(3),
        "riesgo_alto":        riesgo,
    })

    balance = df["riesgo_alto"].value_counts()
    print(f"[CLASIFICACIÓN] Registros: {len(df)} | "
          f"Riesgo bajo: {balance[0]} ({balance[0]/n:.0%}) | "
          f"Riesgo alto: {balance[1]} ({balance[1]/n:.0%})")
    return df


CLASS_FEATURES = ["stock_actual", "lead_time", "demanda_proyectada",
                  "distancia_almacen", "cobertura"]


# ═══════════════════════════════════════════════
# 2B — ENTRENAR CLASIFICADOR CON SMOTE
# ═══════════════════════════════════════════════

def train_classifier(df_cls: pd.DataFrame) -> dict:
    """
    Entrena clasificador de riesgo con SMOTE para balanceo de clases.
    Train 80% / Test 20% estratificado.
    """
    from sklearn.model_selection import train_test_split

    X = df_cls[CLASS_FEATURES]
    y = df_cls["riesgo_alto"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    print(f"[SMOTE] Antes — Train: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] Después — Train: {pd.Series(y_train_res).value_counts().to_dict()}")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=SEED, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    preds      = model.predict(X_test)
    proba      = model.predict_proba(X_test)[:, 1]
    roc_auc    = roc_auc_score(y_test, proba)
    f1         = f1_score(y_test, preds)
    cm         = confusion_matrix(y_test, preds)

    print(f"[CLASIFICADOR] ROC-AUC={roc_auc:.4f} | F1={f1:.4f}")
    print(f"[CLASIFICADOR] Matriz de confusión:\n{cm}")

    return {
        "model":    model,
        "X_test":   X_test,
        "y_test":   y_test,
        "preds":    preds,
        "proba":    proba,
        "roc_auc":  round(roc_auc, 4),
        "f1":       round(f1, 4),
        "cm":       cm,
        "features": CLASS_FEATURES,
    }


# ═══════════════════════════════════════════════
# 2C — INTERPRETABILIDAD (SHAP)
# ═══════════════════════════════════════════════

def compute_shap(model, X_sample: pd.DataFrame, model_name: str = "modelo") -> dict:
    """Calcula SHAP values para un modelo dado."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Para clasificadores RF, shap_values es lista [clase0, clase1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    print(f"[SHAP] {model_name} — valores calculados para {len(X_sample)} muestras")
    return {"explainer": explainer, "shap_values": shap_values, "X": X_sample}


# ═══════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════

def plot_forecast(reg_results: dict, df_ts: pd.DataFrame, punto_id: int = 0):
    """Visualiza forecast vs real para un punto de demanda."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.patch.set_facecolor(TERNIUM["bg"])
    fig.suptitle(f"Forecast de Demanda — Punto {punto_id + 1}",
                 fontsize=14, fontweight="bold", color=TERNIUM["black"], y=1.01)

    colors_model = {"Random Forest": TERNIUM["orange"], "XGBoost": TERNIUM["red"]}

    for ax, (model_name, res) in zip(axes, reg_results.items()):
        _base_style(ax, fig)
        test_df = res["test_df"]
        pt_test = test_df[test_df["punto_id"] == punto_id].sort_values("semana")

        if len(pt_test) == 0:
            ax.set_title(f"{model_name} — sin datos para este punto", color=TERNIUM["gray"])
            continue

        pt_idx  = test_df[test_df["punto_id"] == punto_id].index
        rel_idx = [list(test_df.index).index(i) for i in pt_idx if i in test_df.index]
        preds_pt = res["preds"][rel_idx] if rel_idx else []

        ax.plot(pt_test["semana"].values, pt_test["demanda"].values,
                color=TERNIUM["dark_gray"], linewidth=2, label="Real", marker="o", markersize=3)
        if len(preds_pt):
            ax.plot(pt_test["semana"].values, preds_pt,
                    color=colors_model[model_name], linewidth=2,
                    linestyle="--", label=f"Predicción", marker="s", markersize=3)

        ax.set_title(f"{model_name}   MAE={res['mae']}  RMSE={res['rmse']}  MAPE={res['mape']}%",
                     color=TERNIUM["black"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Semana", fontsize=9)
        ax.set_ylabel("Demanda (uds)", fontsize=9)
        ax.legend(fontsize=9, edgecolor=TERNIUM["light_gray"])

    plt.tight_layout()
    plt.savefig("forecast.png", dpi=150, bbox_inches="tight", facecolor=TERNIUM["bg"])
    plt.show()
    print("Guardado: forecast.png")


def plot_metrics_comparison(reg_results: dict):
    """Compara métricas MAE, RMSE, MAPE entre modelos."""
    model_names = list(reg_results.keys())
    metrics = ["mae", "rmse", "mape"]
    labels  = ["MAE", "RMSE", "MAPE (%)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor(TERNIUM["bg"])
    fig.suptitle("Comparación de Métricas — Regresión",
                 fontsize=13, fontweight="bold", color=TERNIUM["black"])

    bar_colors = [TERNIUM["orange"], TERNIUM["red"]]

    for ax, metric, label in zip(axes, metrics, labels):
        _base_style(ax, fig)
        vals = [reg_results[m][metric] for m in model_names]
        bars = ax.bar(model_names, vals, color=bar_colors,
                      edgecolor=TERNIUM["black"], linewidth=0.6, width=0.5)
        ax.set_title(label, color=TERNIUM["black"], fontsize=11, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val}", ha="center", color=TERNIUM["dark_gray"],
                    fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight", facecolor=TERNIUM["bg"])
    plt.show()
    print("Guardado: metrics_comparison.png")


def plot_classification_report(cls_result: dict):
    """Visualiza matriz de confusión y curva ROC."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(TERNIUM["bg"])
    fig.suptitle("Clasificación — Riesgo de Desabasto",
                 fontsize=13, fontweight="bold", color=TERNIUM["black"])

    # Matriz de confusión
    ax1 = axes[0]
    ax1.set_facecolor(TERNIUM["bg"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cls_result["cm"],
        display_labels=["Riesgo Bajo", "Riesgo Alto"])
    disp.plot(ax=ax1, colorbar=False,
              cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                  "ternium", [TERNIUM["bg"], TERNIUM["orange"]]))
    ax1.set_title(f"Matriz de Confusión\nF1={cls_result['f1']}",
                  color=TERNIUM["black"], fontsize=11, fontweight="bold")
    ax1.tick_params(colors=TERNIUM["gray"])

    # Curva ROC
    ax2 = axes[1]
    _base_style(ax2, fig)
    fpr, tpr, _ = roc_curve(cls_result["y_test"], cls_result["proba"])
    ax2.plot(fpr, tpr, color=TERNIUM["red"], linewidth=2.5,
             label=f"ROC-AUC = {cls_result['roc_auc']}")
    ax2.plot([0, 1], [0, 1], color=TERNIUM["light_gray"],
             linestyle="--", linewidth=1.2, label="Random")
    ax2.fill_between(fpr, tpr, alpha=0.08, color=TERNIUM["orange"])
    ax2.set_title("Curva ROC", color=TERNIUM["black"], fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tasa Falsos Positivos", fontsize=9)
    ax2.set_ylabel("Tasa Verdaderos Positivos", fontsize=9)
    ax2.legend(fontsize=9, edgecolor=TERNIUM["light_gray"])

    plt.tight_layout()
    plt.savefig("classification_report.png", dpi=150, bbox_inches="tight", facecolor=TERNIUM["bg"])
    plt.show()
    print("Guardado: classification_report.png")


def plot_feature_importance(reg_results: dict, cls_result: dict):
    """Visualiza feature importance para regresión y clasificación."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(TERNIUM["bg"])
    fig.suptitle("Interpretabilidad — Feature Importance",
                 fontsize=13, fontweight="bold", color=TERNIUM["black"])

    # Regresión — mejor modelo (menor MAPE)
    best_name = min(reg_results, key=lambda k: reg_results[k]["mape"])
    rf_model  = reg_results[best_name]["model"]
    imp_reg   = pd.Series(rf_model.feature_importances_,
                           index=REGRESSION_FEATURES).sort_values(ascending=True).tail(10)

    ax1 = axes[0]
    _base_style(ax1, fig)
    colors_reg = [TERNIUM["orange"] if v == imp_reg.max() else TERNIUM["gray"]
                  for v in imp_reg.values]
    ax1.barh(imp_reg.index, imp_reg.values, color=colors_reg,
             edgecolor=TERNIUM["black"], linewidth=0.5)
    ax1.set_title(f"Regresión — {best_name}\n(Top 10 features)",
                  color=TERNIUM["black"], fontsize=11, fontweight="bold")
    ax1.set_xlabel("Importancia", fontsize=9)

    # Clasificación
    imp_cls = pd.Series(cls_result["model"].feature_importances_,
                        index=CLASS_FEATURES).sort_values(ascending=True)
    ax2 = axes[1]
    _base_style(ax2, fig)
    colors_cls = [TERNIUM["red"] if v == imp_cls.max() else TERNIUM["gray"]
                  for v in imp_cls.values]
    ax2.barh(imp_cls.index, imp_cls.values, color=colors_cls,
             edgecolor=TERNIUM["black"], linewidth=0.5)
    ax2.set_title("Clasificación — Riesgo de Desabasto",
                  color=TERNIUM["black"], fontsize=11, fontweight="bold")
    ax2.set_xlabel("Importancia", fontsize=9)

    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight", facecolor=TERNIUM["bg"])
    plt.show()
    print("Guardado: feature_importance.png")


def plot_shap_summary(shap_data: dict, title: str = "SHAP Summary"):
    """Visualiza SHAP summary plot con paleta Ternium."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(TERNIUM["bg"])
    ax.set_facecolor(TERNIUM["bg"])

    shap.summary_plot(
        shap_data["shap_values"],
        shap_data["X"],
        plot_type="bar",
        color=TERNIUM["orange"],
        show=False
    )
    plt.title(title, color=TERNIUM["black"], fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=TERNIUM["bg"])
    plt.show()
    print(f"Guardado: {fname}")


# ═══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════

def run_ml_pipeline(data_m01: dict) -> dict:
    """
    Ejecuta el pipeline completo del Módulo 02.
    Requiere el dict generado por generate_data() del Módulo 01.
    """
    print("=" * 55)
    print("  MÓDULO 02 — MACHINE LEARNING")
    print("=" * 55)

    # 2A — Regresión
    print("\n── 2A: REGRESIÓN ──────────────────────────────")
    df_ts       = generate_time_series(data_m01)
    reg_results = walk_forward_validation(df_ts)

    # 2B — Clasificación
    print("\n── 2B: CLASIFICACIÓN ──────────────────────────")
    df_cls     = generate_classification_data(data_m01, reg_results)
    cls_result = train_classifier(df_cls)

    # 2C — Interpretabilidad
    print("\n── 2C: INTERPRETABILIDAD ──────────────────────")
    X_sample_cls = cls_result["X_test"].sample(100, random_state=SEED)
    shap_cls     = compute_shap(cls_result["model"], X_sample_cls, "Clasificador")

    best_reg_name = min(reg_results, key=lambda k: reg_results[k]["mape"])
    best_reg      = reg_results[best_reg_name]
    df_feat_reg   = build_regression_features(df_ts)
    X_sample_reg  = df_feat_reg[REGRESSION_FEATURES].sample(100, random_state=SEED)
    shap_reg      = compute_shap(best_reg["model"], X_sample_reg, "Regresión")

    # Visualizaciones
    print("\n── VISUALIZACIONES ────────────────────────────")
    plot_forecast(reg_results, df_ts, punto_id=0)
    plot_metrics_comparison(reg_results)
    plot_classification_report(cls_result)
    plot_feature_importance(reg_results, cls_result)
    plot_shap_summary(shap_cls, "SHAP — Clasificación Riesgo Desabasto")
    plot_shap_summary(shap_reg, "SHAP — Regresión Forecast Demanda")

    print("\n Pipeline Módulo 02 completado.")
    return {
        "df_ts":        df_ts,
        "reg_results":  reg_results,
        "df_cls":       df_cls,
        "cls_result":   cls_result,
        "shap_cls":     shap_cls,
        "shap_reg":     shap_reg,
    }


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from optimization import generate_data
    data_m01 = generate_data()
    run_ml_pipeline(data_m01)
