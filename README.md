# MIMIC-III — Dashboard interactivo (Streamlit)

Dashboard de predicción de mortalidad intrahospitalaria en pacientes de UCI con lesión renal
aguda (AKI), a partir del pipeline documentado en
[mimic-aki-mortality-prediction](https://github.com/alejandrovazquez-alonso/mimic-aki-mortality-prediction).

**⚠️ Herramienta educativa / de portfolio.** No es una herramienta de decisión clínica
validada. Ver las [limitaciones del proyecto original](https://github.com/alejandrovazquez-alonso/mimic-aki-mortality-prediction#limitations).

## Qué hace

- **Pestaña Predicción:** formulario con las 9 variables clínicas de mayor peso (edad, sexo,
  GCS, BUN, bicarbonato, presión arterial), que devuelve la probabilidad de mortalidad
  estimada por el modelo XGBoost y la clasificación según el umbral de decisión (τ=0.26).
  Las 29 variables restantes se completan automáticamente con la mediana del conjunto de
  entrenamiento.
- **Pestaña Análisis exploratorio:** resumen visual del proyecto — distribución de la
  variable objetivo, comparativa de modelos, matriz de confusión, e importancia de variables.

## Estructura

```
app/
├── streamlit_app.py       # código de la aplicación
├── requirements.txt
├── artifacts/              # modelo entrenado y objetos de preprocesamiento (exportados de Kaggle)
│   ├── xgb_model.pkl
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── medians.json
│   └── threshold.json
└── figures/                 # figuras reales extraídas del notebook ejecutado
    ├── distribucion_target.png
    ├── matriz_confusion.png
    └── importancia_variables.png
```

## Desplegar en Streamlit Community Cloud (gratis)

1. Sube esta carpeta (`app/` con todo su contenido) a un repositorio de GitHub — puede ser
   un repo nuevo, por ejemplo `mimic-dashboard`, o una subcarpeta del repo original.
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de
   GitHub.
3. **New app** → selecciona el repositorio → indica la ruta del archivo principal:
   `streamlit_app.py` (o `app/streamlit_app.py` si está en subcarpeta).
4. **Deploy** — el primer despliegue tarda 2-3 minutos mientras instala las dependencias.
5. Obtendrás una URL pública tipo `tu-usuario-mimic-dashboard.streamlit.app`, lista para
   compartir en LinkedIn, en el README del proyecto, o en una entrevista.

## Ejecutar en local (opcional, para probar antes de desplegar)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Nota técnica sobre los artefactos

El modelo y los objetos de preprocesamiento (`imputer.pkl`, `scaler.pkl`, `xgb_model.pkl`)
se generaron con **scikit-learn 1.6.1** en el entorno de Kaggle. `requirements.txt` fija
esa misma versión para evitar incompatibilidades de deserialización entre versiones.
