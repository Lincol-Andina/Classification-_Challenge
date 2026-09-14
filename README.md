# Machine Learning Group Activity — Classification Challenge
## Models: KNN (F1) + GaussianNB (F2) + LDA (F2) + AdaBoost (F3) | Dataset: Gas Sensor Array Drift (UCI 270)

**Dataset descargado y usado:** `data/uci270/batch*.dat` (zip oficial UCI id=270) → parseado a `data/gas_drift.csv`
- 13,910 instancias (≥ 2,500 exigidos), 128 features numéricas, 6 clases:
  1 Ethanol (2565), 2 Ethylene (2926), 3 Ammonia (1641), 4 Acetaldehyde (1936), 5 Acetone (3009), 6 Toluene (1833)
- Tarea: clasificación de riesgo químico por sensores (encaja en "chemical risk + sensor data").

## Cómo correr
```
pip install ucimlrepo pandas scikit-learn matplotlib seaborn
python src/download_data.py   # parsea batch*.dat -> data/gas_drift.csv (ya hecho)
python src/clean_data.py      # renombra columnas f1..f128 -> Sensor01_R1..Sensor16_R8 (ya hecho)
python src/experiment.py      # pipeline unificado + GridSearch + CV + figuras
```

## Limpieza de datos (nombres legibles)
El dataset crudo (UCI id=270) trae columnas anónimas `f1..f128` — mediciones de 16 sensores químicos sin
encabezados oficiales. Para que cualquier persona entienda el dataset, `src/clean_data.py` los renombra:

- `f(i)` → `Sensor{(i-1)//8+1:02d}_R{(i-1)%8+1}`  →  **16 sensores × 8 lecturas**
- Ejemplos: `f1..f8 → Sensor01_R1..Sensor01_R8`, `f121..f128 → Sensor16_R1..Sensor16_R8`
- La columna `class` se conserva (Clase de gas 1..6).

Solo cambia la cabecera; valores y orden no se tocan. El pipeline (`experiment.py`) lee por posición, así que
los resultados no varían. El mapeo completo queda en `data/feature_dictionary.csv`. Usa solo estándar-library.

## Unified Pipeline (igualdad total)
1. Mismo CSV, mismo `train_test_split(80/20, stratify, random_state=42)`.
2. Mismo `Pipeline([StandardScaler, clf])` — scaler fiteado solo en train dentro de cada fold (sin leakage). Sin OneHot (todo numérico).
3. Mismo `GridSearchCV(cv=Stratified 5-Fold, scoring=f1_macro)` + mismo `cross_validate` externo 5-Fold.
4. Grids: KNN `k[3,5,11,21] x weights x metric`; NB `var_smoothing`; LDA `solver x shrinkage`; AdaBoost `n_estimators[50,100,200] x lr x depth[1,2]`.

## Resultados (test 20% = 2782 + CV 5-Fold)
| Modelo | Best params | CV F1-macro | Test acc / F1-macro | Fit (GridSearch) |
|---|---|---|---|---|
| KNN | manhattan, k=3, distance | 0.995 | 0.995 / 0.996 | 6.3 s |
| LDA | lsqr, sin shrinkage | 0.945 | 0.951 / 0.948 | 0.4 s |
| AdaBoost | 200 árboles d=2, lr=1.0 | 0.914 | 0.908 / 0.898 | 125.2 s |
| GaussianNB | var_smoothing=1e-9 | 0.568 | 0.578 / 0.579 | 0.3 s |

Ver `results/metrics_summary.csv`, `results/cv_folds.csv`, `figures/cm_*.png`, `f1_bar.png`, `cost_vs_f1.png`, `boundary_*.png`.

## Phase 1 — Matriz teórica (resumen)
|  | KNN | GaussianNB | LDA | AdaBoost |
|---|---|---|---|---|
| Principio | Voto de k vecinos por distancia | Bayes + gaussiana + independencia | Proyección que separa medias gaussianas (misma covarianza) | Suma ponderada de stumps que corrigen errores |
| Outliers | Muy sensible (un vecino raro vota) | Medio (media/varianza se sesgan) | Sensible (medias/covarianza) | Muy sensible (insiste en los difíciles/ruido) |
| Coste | Train O(1), predict O(n·d) | O(n·d) / O(d) | O(n·d²) aprox | O(T·n·d) el más caro |
| Suposición | Ninguna (no paramétrico) | Features independientes + gaussianas | Gaussiana por clase, misma covarianza → frontera lineal | Ninguna, pero necesita señal débil mejor que azar |

## Phase 3 — Verdict
1. **Mejor equilibrio precisión-generalización (F1): KNN (0.996).** CV≈test → generaliza. LDA segundo (0.948) y es 15x más rápido que KNN en fit. AdaBoost (0.898) tercero pese a ser el más complejo. NB último (0.579): la independencia no se cumple (128 sensores correlacionados) y confunde Acetaldehyde/Toluene.
2. **¿Justifica el complejo su coste? No.** AdaBoost tarda 125 s vs 0.4 s LDA / 6.3 s KNN y rinde peor. El mejor costo/rendimiento es LDA. NB es el más barato pero inútil aquí.
3. **Fronteras vs dimensionalidad (128-D, ver boundary_*.png en PCA-2D):** KNN → islas locales flexibles (captura clusters de gases, por eso gana); LDA → rectas limpias (funciona porque clases son aprox. gaussianas separables); NB → elipses ingenuas que se solapan (falla); AdaBoost → franjas escalonadas de stumps (sobre-fragmenta, sobreajusta al drift). En alta-D, los simples lineales (LDA) generalizan mejor que los ingenuos (NB); KNN gana porque n=13k es denso.

**Citas:** Vergara et al. 2012 (Gas Sensor Array Drift); Rodriguez-Lujan et al. (calibración). UCI ML Repository id=270, licencia CC-BY-4.0, solo investigación.
