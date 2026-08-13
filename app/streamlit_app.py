"""
Dashboard interactivo — Predicción de mortalidad intrahospitalaria (MIMIC-III / AKI)
Autor: Alejandro Vázquez Alonso

Requiere los artefactos exportados desde el notebook de Kaggle:
xgb_model.pkl, imputer.pkl, scaler.pkl, feature_names.json, medians.json, threshold.json
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

st.set_page_config(
    page_title="MIMIC-III — Predicción de mortalidad en UCI",
    page_icon="🏥",
    layout="wide"
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
FIGURES_DIR = Path(__file__).parent / "figures"


@st.cache_resource
def load_artifacts():
    """Carga el modelo y los objetos de preprocesamiento. Cacheado para no recargar en cada interacción."""
    model = joblib.load(ARTIFACTS_DIR / "xgb_model.pkl")
    imputer = joblib.load(ARTIFACTS_DIR / "imputer.pkl")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(ARTIFACTS_DIR / "medians.json") as f:
        medians = json.load(f)
    with open(ARTIFACTS_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, imputer, scaler, feature_names, medians, threshold


def predict(model, imputer, scaler, feature_names, medians, threshold, inputs):
    # 1. Construir la fila completa (38 variables), con medianas de train por defecto.
    #    Como todas las variables no pedidas en el formulario ya se rellenan aquí con su
    #    mediana, la fila resultante nunca tiene valores faltantes — por lo que el paso
    #    de imputación del pipeline original (pensado para rellenar NaN) no es necesario
    #    aplicarlo de nuevo: no hay nada que imputar.
    row = {feat: inputs.get(feat, medians.get(feat, 0)) for feat in feature_names}
    X = pd.DataFrame([row], columns=feature_names)

    # 2. Escalar con el mismo StandardScaler ajustado en el pipeline original.
    scale_cols = list(scaler.feature_names_in_)
    X_scaled = scaler.transform(X[scale_cols])

    prob = model.predict_proba(X_scaled)[0, 1]
    pred = int(prob >= threshold)
    return prob, pred


# ============================================================
# SIDEBAR — navegación
# ============================================================
st.sidebar.title("🏥 MIMIC-III")
st.sidebar.caption("Predicción de mortalidad intrahospitalaria en UCI (AKI)")
page = st.sidebar.radio("Navegación", ["🔮 Predicción", "📊 Análisis exploratorio"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sobre este proyecto**\n\n"
    "Pipeline de ML (Logistic Regression, Random Forest, XGBoost) para predecir "
    "mortalidad en pacientes de UCI con lesión renal aguda (MIMIC-III).\n\n"
    "[Ver código completo en GitHub](https://github.com/alejandrovazquez-alonso/mimic-aki-mortality-prediction)"
)

# ============================================================
# Intentar cargar artefactos
# ============================================================
artifacts_missing = not (ARTIFACTS_DIR / "xgb_model.pkl").exists()

if artifacts_missing:
    st.error(
        "⚠️ No se encuentran los artefactos del modelo en `artifacts/`. "
        "Exporta `xgb_model.pkl`, `imputer.pkl`, `scaler.pkl`, `feature_names.json`, "
        "`medians.json` y `threshold.json` desde el notebook de Kaggle y colócalos en esa carpeta."
    )
    st.stop()

model, imputer, scaler, feature_names, medians, threshold = load_artifacts()

# ============================================================
# PÁGINA 1 — PREDICCIÓN
# ============================================================
if page == "🔮 Predicción":
    st.title("Predicción de mortalidad intrahospitalaria")
    st.markdown(
        "Introduce los valores clínicos de un paciente de UCI con lesión renal aguda (AKI). "
        "Las variables no incluidas en el formulario se completan automáticamente con la "
        "**mediana del conjunto de entrenamiento** (mismo criterio de imputación que en el pipeline)."
    )

    st.info(
        "⚕️ **Herramienta educativa / de portfolio.** No constituye una herramienta de decisión "
        "clínica validada. Ver [Limitations](https://github.com/alejandrovazquez-alonso/"
        "mimic-aki-mortality-prediction#limitations) en el repositorio.",
        icon="⚠️"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demografía")
        age = st.slider("Edad (años)", 18, 100, 65)
        gender_F = st.selectbox("Sexo", ["Hombre", "Mujer"])
        gender_F = 1 if gender_F == "Mujer" else 0

        st.subheader("Neurológico")
        gcs_mean = st.slider("GCS medio (Glasgow Coma Scale)", 3.0, 15.0, 13.0, 0.5)
        gcs_min = st.slider("GCS mínimo", 3.0, 15.0, 11.0, 0.5)

    with col2:
        st.subheader("Función renal")
        bun_mean = st.slider("BUN medio (mg/dL)", 5.0, 150.0, 25.0, 1.0)
        bun_max = st.slider("BUN máximo (mg/dL)", 5.0, 200.0, 35.0, 1.0)
        bic_mean = st.slider("Bicarbonato medio (mEq/L)", 5.0, 40.0, 22.0, 0.5)

    with col3:
        st.subheader("Hemodinámica")
        bp_mean = st.slider("Presión arterial media (mmHg)", 30.0, 150.0, 75.0, 1.0)
        bp_min = st.slider("Presión arterial mínima (mmHg)", 20.0, 120.0, 55.0, 1.0)

    st.markdown("---")

    if st.button("🔍 Calcular predicción", type="primary", use_container_width=True):
        inputs = {
            "age": age,
            "gender_F": gender_F,
            "gcs_mean": gcs_mean,
            "gcs_min": gcs_min,
            "bun_mean": bun_mean,
            "bun_max": bun_max,
            "bic_mean": bic_mean,
            "bp_mean": bp_mean,
            "bp_min": bp_min,
        }
        prob, pred = predict(model, imputer, scaler, feature_names, medians, threshold, inputs)

        st.markdown("### Resultado")
        rcol1, rcol2 = st.columns([1, 2])

        with rcol1:
            st.metric("Probabilidad estimada de mortalidad", f"{prob*100:.1f}%")
            st.metric("Umbral de decisión (Youden + F1)", f"{threshold:.2f}")

        with rcol2:
            if pred == 1:
                st.error(
                    f"**Alto riesgo estimado** (probabilidad {prob*100:.1f}% ≥ umbral {threshold*100:.0f}%).\n\n"
                    "Este resultado, en un contexto real, motivaría una valoración clínica más intensiva — "
                    "no es una indicación diagnóstica ni terapéutica por sí sola.",
                    icon="🔴"
                )
            else:
                st.success(
                    f"**Riesgo estimado por debajo del umbral** (probabilidad {prob*100:.1f}% < umbral {threshold*100:.0f}%).\n\n"
                    "No implica ausencia de riesgo — solo que, según el modelo, no se supera el punto de corte "
                    "operativo definido (τ=0.26, ver README del proyecto).",
                    icon="🟢"
                )

        st.caption(
            "El modelo (XGBoost, AUROC test = 0.8946) usa 36 variables; las 9 mostradas aquí son las de "
            "mayor peso clínico e importancia compartida entre Random Forest y XGBoost. El resto se "
            "completan con la mediana de entrenamiento — por eso el resultado es una aproximación "
            "simplificada, no la predicción exacta que daría el modelo con el registro completo del paciente."
        )

# ============================================================
# PÁGINA 2 — EDA / HALLAZGOS
# ============================================================
else:
    st.title("Análisis exploratorio y resultados del modelo")

    st.markdown("### Distribución de la variable objetivo")
    st.image(str(FIGURES_DIR / "distribucion_target.png"),
             caption="3,550 pacientes, 974 fallecidos (27.4% mortalidad, ratio 2.64:1)")

    st.markdown("---")
    st.markdown("### Comparativa de modelos")
    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression", "AUROC 0.872")
    col2.metric("Random Forest", "AUROC 0.879")
    col3.metric("XGBoost (ganador)", "AUROC 0.895")

    st.markdown("---")
    st.markdown("### Matriz de confusión — XGBoost")
    st.image(str(FIGURES_DIR / "matriz_confusion.png"),
              caption="Umbral de decisión seleccionado por criterio combinado AUROC + F1")

    st.markdown("---")
    st.markdown("### Importancia de variables — Random Forest vs. XGBoost")
    st.image(str(FIGURES_DIR / "importancia_variables.png"),
              caption="9 variables coincidentes en el top 10 de ambos modelos: age, bic_mean, bp_mean, "
                      "bp_min, bun_max, bun_mean, gcs_max, gcs_mean, gcs_min")

    st.markdown("---")
    st.markdown(
        "**Interpretación clínica:** la escala de Glasgow (GCS) refleja nivel de consciencia, el BUN "
        "es marcador directo de función renal, la presión arterial media refleja estabilidad "
        "hemodinámica, y la edad es un factor pronóstico transversal en UCI. Que dos algoritmos con "
        "lógicas de decisión distintas (bagging vs. boosting) converjan en las mismas variables refuerza "
        "la robustez del hallazgo."
    )

    st.markdown(
        "[Ver el análisis completo, código y limitaciones en GitHub →]"
        "(https://github.com/alejandrovazquez-alonso/mimic-aki-mortality-prediction)"
    )
