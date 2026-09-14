"""Dashboard precalculado — Gas Sensor Array Drift (UCI 270). Sin reentrenar (apto Streamlit Cloud)."""
from pathlib import Path
import json
import platform
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
RES = BASE / "results"
FIG = BASE / "figures"
DATA = BASE / "data" / "gas_drift.csv"
DICT = BASE / "data" / "feature_dictionary.csv"

CLASS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
               4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}

ORDER = ["KNN", "LDA", "AdaBoost", "GaussianNB"]
COLORS = {"KNN": "#4F8BF9", "LDA": "#14B8A6",
          "AdaBoost": "#EF4444", "GaussianNB": "#F59E0B"}
FAMILIES = {
    "KNN": ("Geométrica / distancias", "F1"),
    "LDA": ("Probabilística / discriminante", "F2"),
    "AdaBoost": ("Ensemble / estocástica", "F3"),
    "GaussianNB": ("Probabilística / generativa", "F2"),
}

st.set_page_config(page_title="Gas Drift — Classification Challenge",
                   page_icon=":bar_chart:", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

/* Hero */
.hero {
  background: linear-gradient(135deg, #0f2a52 0%, #1d4ed8 55%, #38bdf8 100%);
  color: #fff; border-radius: 18px; padding: 1.6rem 2.2rem; margin-bottom: 1.4rem;
  box-shadow: 0 10px 30px rgba(29,78,216,.25);
}
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 .25rem; }
.hero p { margin: .15rem 0; font-size: .98rem; opacity: .95; }
.hero .tag { display:inline-block; background: rgba(255,255,255,.16); padding:.2rem .7rem;
             border-radius: 999px; font-size: .78rem; margin-right: .5rem; font-weight:600; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.2rem 0; }
.kpi {
  border-radius: 14px; padding: 1rem 1.2rem; color:#fff; position: relative;
  box-shadow: 0 6px 16px rgba(0,0,0,.12); background:#fff;
}
.kpi .rank { position:absolute; top:.6rem; right:.8rem; background:rgba(0,0,0,.22);
             border-radius:999px; padding:.15rem .55rem; font-size:.75rem; font-weight:700; }
.kpi .name { font-size: 1rem; font-weight:700; color:#334155; margin-bottom:.1rem; }
.kpi .fam { font-size:.72rem; color:#64748b; font-weight:600; margin-bottom:.6rem; }
.kpi .f1 { font-size: 2.1rem; font-weight:800; line-height:1; }
.kpi .sub { font-size:.78rem; margin-top:.45rem; opacity:.95; }
.kpi .gap { margin-top:.3rem; font-size:.78rem; font-weight:600; }

/* Callout / verdict */
.callout {
  border-radius: 12px; padding: 1rem 1.3rem; margin: .6rem 0;
  border-left: 5px solid; background: #f8fafc;
}
.callout h4 { margin: 0 0 .3rem; font-weight:800; color:#0f172a; }
.callout p { margin: 0; color:#334155; font-size:.92rem; line-height:1.55; }
.callout.best { border-color:#4F8BF9; }
.callout.warn { border-color:#EF4444; }
.callout.good { border-color:#14B8A6; }
.callout.amber { border-color:#F59E0B; }

/* section title */
.sec-title { font-size: 1.25rem; font-weight: 800; color:#0f172a;
             margin: 1.6rem 0 .6rem; padding-bottom: .35rem; border-bottom: 2px solid #e2e8f0; }
.chip { display:inline-block; background:#eef2ff; color:#4338ca; font-weight:700;
        border-radius:999px; padding:.18rem .6rem; font-size:.75rem; margin:.1rem .2rem; }
.credit { font-size:.78rem; color:#64748b; text-align:center; margin-top:1.5rem; line-height:1.6; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: .5rem 1.1rem; font-weight:600; }
div[data-testid="stImage"] img { border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,.10); }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_summary():
    return pd.read_csv(RES / "metrics_summary.csv")


@st.cache_data(show_spinner=False)
def load_cv():
    return pd.read_csv(RES / "cv_folds.csv")


@st.cache_data(show_spinner=False)
def load_dict():
    return pd.read_csv(DICT)


try:
    summary = load_summary()
    cv = load_cv()
except FileNotFoundError as e:
    st.error(f"Falta archivo de resultados: {e}. Corre `python src/experiment.py` local y sube `results/`.")
    st.stop()

summary["orden"] = summary["model"].apply(lambda m: ORDER.index(m) if m in ORDER else 99)
summary = summary.sort_values("orden").drop(columns="orden").reset_index(drop=True)
best_f1 = float(summary["test_f1_macro"].max())
_rank_order = summary.sort_values("test_f1_macro", ascending=False)["model"].tolist()
rank_map = {m: i + 1 for i, m in enumerate(_rank_order)}


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>Gas Sensor Array Drift · Benchmark de Clasificación</h1>
  <p>
    <span class="tag">13,910 muestras</span>
    <span class="tag">128 features · 16 sensores</span>
    <span class="tag">6 clases de gas</span>
    <span class="tag">Pipeline unificado</span>
    <span class="tag">GridSearch + 5-Fold CV</span>
  </p>
  <p style="margin-top:.6rem;">Clasificación de gases por sensores químicos — KNN · LDA · AdaBoost · GaussianNB bajo condiciones idénticas
     (mismo split, mismo scaler, misma búsqueda de hiperparámetros).</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------
def kpi_card(name):
    row = summary.loc[summary["model"] == name].iloc[0]
    fam, code = FAMILIES[name]
    rank = rank_map[name]
    f1 = row["test_f1_macro"]
    gap = f1 - row["cv_f1_macro_mean"]
    gap_txt = "≈ CV · generaliza" if abs(gap) < 0.003 else (f"+{gap:+.3f} test vs CV" if gap >= 0 else f"{gap:+.3f} test vs CV")
    color = COLORS[name]
    return f"""<div class="kpi" style="background:linear-gradient(140deg,{color} 0%,{color}cc 100%);">
      <div class="rank">#{rank}</div>
      <div class="name">{name}</div>
      <div class="fam">{fam} · {code}</div>
      <div class="f1">{f1:.3f}</div>
      <div class="sub">CV {row['cv_f1_macro_mean']:.3f} ± {row['cv_f1_macro_std']:.3f} · acc {row['test_acc']:.3f}</div>
      <div class="gap">{gap_txt}</div>
    </div>"""

st.markdown('<div class="kpi-grid">' + "".join(kpi_card(m) for m in ORDER) + "</div>",
            unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_vista, tab_comp, tab_exp, tab_ver, tab_datos = st.tabs(
    ["Vista general", "Comparación detallada", "Explorador de modelos",
     "Veredicto (Phase 3)", "Datos y metodología"])

# ---------------------------------------------------------------- Vista general
with tab_vista:
    st.markdown('<div class="sec-title">F1-macro por modelo · test 20% vs CV externo 5-Fold</div>',
                unsafe_allow_html=True)
    colA, colB = st.columns([1.4, 1])

    with colA:
        fig = go.Figure()
        for name in ORDER:
            row = summary.loc[summary["model"] == name].iloc[0]
            fig.add_trace(go.Bar(
                x=[name], y=[row["test_f1_macro"]],
                name=name, marker_color=COLORS[name],
                error_y=dict(type="data", array=[row["cv_f1_macro_std"]],
                             visible=True, thickness=1.6, width=4),
                text=[f"{row['test_f1_macro']:.3f}"], textposition="outside",
            ))
            fig.add_trace(go.Scatter(
                x=[name], y=[row["cv_f1_macro_mean"]],
                mode="markers", name=f"CV {name}",
                marker=dict(symbol="diamond-open", size=13, color="#0f172a", line=dict(width=2)),
                showlegend=False,
            ))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10),
                          yaxis=dict(range=[0.5, 1.03], title="F1-macro"),
                          xaxis=dict(title=""), legend_title="",
                          bargap=0.35, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, config={"displayModeBar": False})

    with colB:
        st.markdown('<div class="sec-title">Tabla resumen</div>', unsafe_allow_html=True)
        tbl = summary[["model", "test_f1_macro", "cv_f1_macro_mean",
                       "cv_f1_macro_std", "test_acc", "fit_time_s"]].rename(columns={
            "model": "Modelo", "test_f1_macro": "F1 test", "cv_f1_macro_mean": "CV F1",
            "cv_f1_macro_std": "CV ±", "test_acc": "Acc test", "fit_time_s": "Fit (s)"})
        st.dataframe(tbl.style.format({c: "{:.3f}" for c in
                                       ["F1 test", "CV F1", "CV ±", "Acc test"]}),
                     use_container_width=True, hide_index=True)
        st.caption("Test: 20% (n=2782) · CV externo 5-Fold sobre train · fit = GridSearch completo.")

    st.markdown('<div class="sec-title">F1 por fold — estabilidad de cada modelo</div>', unsafe_allow_html=True)
    pivot = cv.pivot(index="fold", columns="model", values="f1_macro")[ORDER]
    fig2 = go.Figure()
    for name in ORDER:
        fig2.add_trace(go.Scatter(x=pivot.index + 1, y=pivot[name], mode="lines+markers",
                                  name=name,
                                  line=dict(color=COLORS[name], width=3),
                                  marker=dict(size=7),
                                  fill="tozeroy" if name == "KNN" else None,
                                  fillcolor="rgba(79,139,249,.08)"))
    fig2.update_layout(height=380, xaxis=dict(title="Fold (CV externo)", dtick=1),
                       yaxis=dict(title="F1-macro", range=[0.5, 1.0]),
                       margin=dict(l=10, r=10, t=20, b=10),
                       legend_title="", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, config={"displayModeBar": False})

# ---------------------------------------------------------------- Comparación
with tab_comp:
    st.markdown('<div class="sec-title">Costo de entrenamiento vs rendimiento</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        fig3 = go.Figure()
        for name in ORDER:
            row = summary.loc[summary["model"] == name].iloc[0]
            fig3.add_trace(go.Scatter(
                x=[row["fit_time_s"]], y=[row["test_f1_macro"]],
                mode="markers+text", name=name,
                marker=dict(size=18, color=COLORS[name], line=dict(color="#0f172a", width=1.5)),
                text=[name], textposition="top center", textfont=dict(size=12, color="#0f172a"),
            ))
        fig3.add_hline(y=best_f1, line_dash="dot", line_color="#94a3b8",
                       annotation_text=f"mejor F1 {best_f1:.3f}", annotation_font_size=11)
        fig3.update_layout(height=420, xaxis=dict(type="log", title="Fit time con GridSearch (s, log)"),
                           yaxis=dict(title="F1-macro test", range=[0.5, 1.03]),
                           margin=dict(l=10, r=10, t=20, b=10), legend_title="",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, config={"displayModeBar": False})
        st.caption("El mejor costo/rendimiento es el que queda arriba-izquierda: LDA (0.4 s, F1 0.948).")

    with colB:
        st.markdown('<div class="sec-title">Accuracy vs Balanced accuracy</div>', unsafe_allow_html=True)
        fig4 = go.Figure()
        metrics_wide = summary[["model", "test_acc", "test_bal_acc", "test_f1_weighted"]].melt(
            id_vars="model", var_name="metrica", value_name="valor")
        metric_labels = {"test_acc": "Accuracy", "test_bal_acc": "Balanced acc",
                         "test_f1_weighted": "F1 weighted"}
        for m in ["test_acc", "test_bal_acc", "test_f1_weighted"]:
            sub = metrics_wide[metrics_wide["metrica"] == m]
            fig4.add_trace(go.Bar(x=sub["model"], y=sub["valor"], name=metric_labels[m],
                                  text=[f"{v:.3f}" for v in sub["valor"]],
                                  textposition="outside"))
        fig4.update_layout(height=420, yaxis=dict(range=[0.5, 1.03], title="Score"),
                           xaxis=dict(title=""), barmode="group", legend_title="",
                           margin=dict(l=10, r=10, t=20, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, config={"displayModeBar": False})
        st.caption("Balanced accuracy pondera clases iguales (dataset levemente desbalanceado).")

    st.markdown('<div class="sec-title">Matrices de confusión y fronteras (PCA-2D)</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        for name in ORDER[:2]:
            p = FIG / f"cm_{name}.png"
            if p.exists():
                with st.expander(f"Confusión — {name}", expanded=True):
                    st.image(str(p), use_container_width=True)
    with g2:
        for name in ORDER[2:]:
            p = FIG / f"cm_{name}.png"
            if p.exists():
                with st.expander(f"Confusión — {name}", expanded=True):
                    st.image(str(p), use_container_width=True)
    b1, b2 = st.columns(2)
    with b1:
        for name in ORDER[:2]:
            p = FIG / f"boundary_{name}.png"
            if p.exists():
                with st.expander(f"Frontera (PCA-2D) — {name}", expanded=True):
                    st.image(str(p), use_container_width=True)
    with b2:
        for name in ORDER[2:]:
            p = FIG / f"boundary_{name}.png"
            if p.exists():
                with st.expander(f"Frontera (PCA-2D) — {name}", expanded=True):
                    st.image(str(p), use_container_width=True)
    st.caption("Fronteras: clon del mejor estimador reentrenado en 2 componentes PCA — ilustración, "
               "no la frontera real 128-D de las métricas.")

# ---------------------------------------------------------------- Explorador
with tab_exp:
    st.markdown('<div class="sec-title">Explorador de modelos</div>', unsafe_allow_html=True)
    sel = st.selectbox("Modelo", ORDER, index=0)
    row = summary.loc[summary["model"] == sel].iloc[0]
    fam, code = FAMILIES[sel]
    st.markdown(f'<span class="chip">Familia {code}</span> <span class="chip">{fam}</span>'
                f'<span class="chip">Rango #{rank_map[sel]} en F1</span>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Hiperparámetros óptimos (GridSearch)**")
        st.code(json.dumps(json.loads(row["best_params"]), indent=2), language="json")
        metrics = {
            "F1-macro (test)": f"{row['test_f1_macro']:.4f}",
            "Accuracy (test)": f"{row['test_acc']:.4f}",
            "Balanced acc": f"{row['test_bal_acc']:.4f}",
            "F1 weighted": f"{row['test_f1_weighted']:.4f}",
            "CV F1-macro": f"{row['cv_f1_macro_mean']:.4f} ± {row['cv_f1_macro_std']:.4f}",
            "Fit (GridSearch)": f"{row['fit_time_s']} s",
            "Predict (test)": f"{row['predict_time_s']} s",
        }
        st.markdown("**Métricas**")
        st.dataframe(pd.DataFrame({"Métrica": list(metrics), "Valor": list(metrics.values())}),
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Confusión y frontera**")
        pm = FIG / f"cm_{sel}.png"
        pb = FIG / f"boundary_{sel}.png"
        if pm.exists() and pb.exists():
            i1, i2 = st.columns(2)
            i1.image(str(pm), use_container_width=True)
            i2.image(str(pb), use_container_width=True)

    blurbs = {
        "KNN": ("Por qué gana", "life",
                "Vecinos locales con k=3, manhattan y peso por distancia capturan los clusters de cada gas. "
                "Con n=13k el espacio 128-D está densamente poblado y el voto local es muy estable (CV≈test)."),
        "LDA": ("Mejor costo/rendimiento", "rocket",
                "Clases aprox. gaussianas con covarianza similar → frontera lineal suficiente. 0.4 s de fit "
                "para F1 0.948: el modelo más eficiente del grupo."),
        "AdaBoost": ("El complejo no se justifica", "alert-triangle",
                     "125 s (200 stumps) para rendir por debajo de KNN y LDA. Insiste sobre el ruido/drift y "
                     "sobre-fragmenta la frontera."),
        "GaussianNB": ("Caso de fallo clásico", "x-octagon",
                       "Independencia condicional rota: 128 sensores correlacionados. F1 0.579 y fronteras "
                       "elípticas solapadas que confunden Acetaldehyde/Toluene."),
    }
    title, icon, msg = blurbs[sel]
    st.markdown(f'<div class="callout best"><h4>[{icon}] {title}</h4><p>{msg}</p></div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- Veredicto
with tab_ver:
    st.markdown('<div class="sec-title">Phase 3 — Las 3 preguntas del benchmark</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="callout best"><h4>1 · ¿Mejor equilibrio precisión-generalización (F1)?</h4>
<p><b>KNN</b> (test F1 0.995, CV 0.995 ± 0.001). CV ≈ test ⇒ no es una partición afortunada: generaliza.
LDA segundo (0.948) y 15× más rápido en fit. AdaBoost tercero (0.898) pese a su complejidad.
GaussianNB último (0.579): la independencia no se cumple con 128 sensores correlacionados.</p></div>
<div class="callout warn"><h4>2 · ¿El modelo complejo justifica su costo?</h4>
<p><b>No.</b> AdaBoost invierte 125 s (GridSearch) y rinde peor que LDA (0.4 s) y KNN (6.3 s).
El mejor costo/rendimiento es <b>LDA</b>. GaussianNB es el más barato pero inútil aquí;
KNN es el más lento en predicción (1.8 s) por ser lazy O(n·d).</p></div>
<div class="callout good"><h4>3 · Fronteras de decisión vs dimensionalidad (128-D)</h4>
<p><b>KNN</b> → islas locales flexibles (captura clusters; gana porque n=13k es denso).
<b>LDA</b> → fronteras lineales limpias (clases aprox. gaussianas separables).
<b>GaussianNB</b> → elipses ingenuas solapadas. <b>AdaBoost</b> → franjas escalonadas de stumps
(sobre-fragmenta y sobreajusta al drift). En alta dimensión el lineal simple (LDA) supera al
ingenuo (NB); KNN compensa la dimensionalidad con más datos.</p></div>
<div class="callout amber"><h4>Conclusión para el caso de uso</h4>
<p>Para inferencia sobre sensores: <b>LDA</b> si importa el coste (0.002 s por batch); <b>KNN</b> si la
precisión máxima es prioridad. Evitar AdaBoost y GaussianNB en este dominio.</p></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- Datos
with tab_datos:
    st.markdown('<div class="sec-title">Dataset & limpieza de features</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("""
<ul style="font-size:.93rem; line-height:1.8; color:#334155;">
<li><b>Fuente:</b> Gas Sensor Array Drift — UCI ML Repository (id=270), Vergara et al. 2012. CC-BY-4.0.</li>
<li><b>Forma:</b> 13,910 instancias × 128 features (16 sensores × 8 lecturas) + clase (6 gases).</li>
<li><b>Tarea aplicada:</b> clasificación de componentes químicos / sensores ambientales.</li>
<li><b>Limpieza:</b> features anónimas <code>f1..f128</code> → <code>Sensor01_R1..Sensor16_R8</code>
    (<code>src/clean_data.py</code>). Mapeo completo en <code>data/feature_dictionary.csv</code>.</li>
<li><b>Clases:</b> Ethanol (2565), Ethylene (2926), Ammonia (1641), Acetaldehyde (1936),
    Acetone (3009), Toluene (1833).</li>
</ul>
""", unsafe_allow_html=True)
    with c2:
        if DATA.exists():
            head = pd.read_csv(DATA, nrows=10)
            st.markdown("**Vista previa (10 filas × 8 sensores + clase)**")
            st.dataframe(head.iloc[:, :8].join(head[["class"]]),
                         use_container_width=True, hide_index=True)
        else:
            st.warning("`data/gas_drift.csv` no está en el deploy. Generable con "
                       "`python src/download_data.py` + `src/clean_data.py`.")

    try:
        dict_df = load_dict()
        sensors = sorted(set(dict_df["feature_nuevo"].str[:8]), key=lambda s: int(s[6:8]))
        colA, colM = st.columns([1, 2])
        with colA:
            tgt = st.selectbox("Filtrar por sensor", sensors)
        with colM:
            sub = dict_df[dict_df["feature_nuevo"].str.startswith(tgt) | (dict_df["feature_nuevo"] == "class")]
            st.dataframe(sub, use_container_width=True, hide_index=True)
        cL, cR, _ = st.columns([1, 1, 2])
        cL.download_button("⬇ Descargar feature_dictionary.csv",
                           dict_df.to_csv(index=False).encode(), "feature_dictionary.csv", "text/csv")
        cR.download_button("⬇ Descargar metrics_summary.csv",
                           load_summary().to_csv(index=False).encode(), "metrics_summary.csv", "text/csv")
    except Exception as e:
        st.warning(f"No se pudo cargar el diccionario: {e}")

    st.markdown('<div class="sec-title">Metodología — pipeline unificado</div>', unsafe_allow_html=True)
    with st.expander("Ver detalle del pipeline (igualdad de condiciones)"):
        st.markdown("""
1. **Mismo split:** `train_test_split(80/20, stratify, random_state=42)` → test n=2782.
2. **Mismo preproceso:** `Pipeline([StandardScaler, clf])` — scaler ajustado solo en train dentro de cada fold
   (sin leakage). Sin OneHot: todas las features son numéricas.
3. **Misma búsqueda:** `GridSearchCV(cv=Stratified 5-Fold, scoring=f1_macro)` + `cross_validate` externo 5-Fold.
   Grids: KNN `k[3,5,11,21] × weights × metric`; GaussianNB `var_smoothing`; LDA `solver × shrinkage`;
   AdaBoost `n_estimators[50,100,200] × lr × depth`.
4. **Reproducción:** `python src/clean_data.py` → `python src/experiment.py` (genera `results/` y `figures/`).
""")
        st.code("KNN      manhattan, k=3, weights=distance    -> F1 0.995  (6.3 s)\n"
                "LDA      lsqr, sin shrinkage                -> F1 0.948  (0.4 s)\n"
                "AdaBoost 200 arboles d=2, lr=1.0            -> F1 0.898  (125.2 s)\n"
                "GaussianNB var_smoothing=1e-9               -> F1 0.579  (0.3 s)", language=None)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
try:
    import sklearn
    env = f"Python {platform.python_version()} · scikit-learn {sklearn.__version__} · pandas {pd.__version__}"
except Exception:
    env = f"Python {platform.python_version()}"
st.markdown(
    '<div class="credit">Vergara et al. 2012 · UCI ML Repository id=270 · CC-BY-4.0 · uso solo investigación.<br>'
    f'{env} · Reporte precalculado: <code>results/report.html</code> · Pipeline: <code>src/experiment.py</code></div>',
    unsafe_allow_html=True)