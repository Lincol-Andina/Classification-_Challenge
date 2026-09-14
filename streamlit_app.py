"""Dashboard precalculado — Gas Sensor Array Drift (UCI 270). Sin reentrenar (apto Streamlit Cloud)."""
from pathlib import Path
import json
import platform
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
RES = BASE / "results"
FIG = BASE / "figures"
DATA = BASE / "data" / "gas_drift.csv"

CLASS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
               4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}
MODELS = ["KNN", "GaussianNB", "LDA", "AdaBoost"]

st.set_page_config(page_title="Gas Drift — Classification Challenge", layout="wide")

st.title("Chemical Risk Classification — Gas Sensor Array Drift (UCI 270)")
st.markdown(
    "13,910 instancias · 128 sensores · 6 gases · "
    "Pipeline unificado `StandardScaler + GridSearchCV(5-Fold, f1_macro)` · split 80/20 `random_state=42`. "
    "**Sin reentrenar en Cloud:** esta app solo lee `results/*.csv` y `figures/*.png`."
)

try:
    import sklearn
    sklearn_v = sklearn.__version__
except Exception:
    sklearn_v = "no disponible"
st.caption(f"Entorno: Python {platform.python_version()} · scikit-learn {sklearn_v} · pandas {pd.__version__} "
           f"(pins en `requirements.txt` + `runtime.txt`). Si ves mismatch de versiones, revisa el deploy log.")


@st.cache_data
def load_summary():
    return pd.read_csv(RES / "metrics_summary.csv")


@st.cache_data
def load_cv():
    return pd.read_csv(RES / "cv_folds.csv")


try:
    summary = load_summary()
    cv = load_cv()
except FileNotFoundError as e:
    st.error(f"Falta archivo de resultados: {e}. Corre `python src/experiment.py` local y sube `results/`.")
    st.stop()

# Orden canónico para comparar
order = ["KNN", "LDA", "AdaBoost", "GaussianNB"]
summary["orden"] = summary["model"].apply(lambda m: order.index(m) if m in order else 99)
summary = summary.sort_values("orden").drop(columns="orden")

# Métricas top
cols = st.columns(4)
for c, (_, r) in zip(cols, summary.iterrows()):
    c.metric(r["model"], f"F1 {r['test_f1_macro']:.3f}",
             f"CV {r['cv_f1_macro_mean']:.3f} · fit {r['fit_time_s']}s")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Resumen", "Comparativa F1 / Costo", "Matrices confusión",
     "Fronteras PCA-2D", "Detalle modelo", "Muestra datos"])

with tab1:
    st.subheader("Tabla test 20% (n=2782) + CV 5-Fold")
    show = summary[["model", "best_params", "cv_f1_macro_mean", "cv_f1_macro_std",
                    "test_acc", "test_f1_macro", "test_bal_acc",
                    "fit_time_s", "predict_time_s"]].copy()
    st.dataframe(show, use_container_width=True)
    st.download_button("Descargar metrics_summary.csv",
                       show.to_csv(index=False).encode(),
                       "metrics_summary.csv", "text/csv")
    st.caption("Dataset: `data/gas_drift.csv` (parseado de `batch*.dat` UCI id=270). "
               "Clases: 1 Ethanol (2565), 2 Ethylene (2926), 3 Ammonia (1641), "
               "4 Acetaldehyde (1936), 5 Acetone (3009), 6 Toluene (1833).")
    p = FIG / "f1_bar.png"
    if p.exists():
        st.image(str(p), caption="Test F1-macro por modelo")
    p2 = FIG / "cost_vs_f1.png"
    if p2.exists():
        st.image(str(p2), caption="Costo (fit con GridSearch, log) vs F1")

with tab2:
    st.subheader("F1 por fold (CV externo 5-Fold)")
    st.bar_chart(cv.pivot(index="fold", columns="model", values="f1_macro"))
    st.dataframe(cv, use_container_width=True)

with tab3:
    st.subheader("Matrices de confusión (test)")
    sel = st.selectbox("Modelo", MODELS, index=0, key="cm")
    p = FIG / f"cm_{sel}.png"
    if p.exists():
        st.image(str(p), caption=f"Confusion Matrix — {sel}", use_container_width=True)
    else:
        st.warning(f"No se encontró {p.name}")

with tab4:
    st.subheader("Fronteras de decisión en PCA-2D (solo visualización)")
    st.caption("Clon del mejor estimador entrenado sobre 2 componentes PCA. KNN=islas locales, "
               "LDA=rectas, NB=elipses solapadas, AdaBoost=franjas escalonadas. "
               "No es la frontera real 128-D usada en las métricas (ver `src/experiment.py`).")
    sel2 = st.selectbox("Modelo", MODELS, index=0, key="bd")
    p = FIG / f"boundary_{sel2}.png"
    if p.exists():
        st.image(str(p), caption=f"Decision boundary (PCA-2D) — {sel2}", use_container_width=True)
    else:
        st.warning(f"No se encontró {p.name}")

with tab5:
    st.subheader("Mejores hiperparámetros")
    sel3 = st.selectbox("Modelo", MODELS, index=0, key="hp")
    row = summary.loc[summary["model"] == sel3].iloc[0]
    st.code(json.dumps(json.loads(row["best_params"]), indent=2), language="json")
    st.write(f"**CV F1-macro:** {row['cv_f1_macro_mean']:.4f} ± {row['cv_f1_macro_std']:.4f} · "
             f"**Test acc/F1:** {row['test_acc']:.4f} / {row['test_f1_macro']:.4f} · "
             f"**Fit:** {row['fit_time_s']}s")
    st.markdown(
        "**Verdict:** mejor equilibrio KNN (0.996, CV≈test generaliza). "
        "LDA segundo (0.948) y 15x más rápido en fit — mejor costo/rendimiento. "
        "AdaBoost (0.898) no justifica 125s. NB último (0.579): independencia no se cumple "
        "(128 sensores correlacionados), confunde Acetaldehyde/Toluene."
    )

with tab6:
    st.subheader("Muestra del dataset (solo lectura, head)")
    if DATA.exists():
        try:
            head = pd.read_csv(DATA, nrows=10)
            st.dataframe(head.iloc[:, :8].join(head[["class"]]), use_container_width=True)
            st.caption(f"`gas_drift.csv`: 13,910 filas × 129 cols (128 sensores + class). Vista: 10 filas × 8 primeras features.")
        except Exception as e:
            st.warning(f"No se pudo leer muestra: {e}")
    else:
        st.warning("`data/gas_drift.csv` no incluido en el deploy (esperable en Cloud para ahorrar peso). "
                   "Generable local con `python src/download_data.py`.")

st.divider()
st.caption("Vergara et al. 2012 (Gas Sensor Array Drift); UCI ML Repository id=270, CC-BY-4.0, solo investigación. "
           "Código pipeline: `src/experiment.py`. Notebook teórico: `notebooks/01_classification_challenge.ipynb`.")
