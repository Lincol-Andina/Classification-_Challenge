"""Unified pipeline: same scaler, same CV, same scoring for KNN / GNB / LDA / AdaBoost.

Dataset: Gas Sensor Array Drift, UCI id=270, 13910 x 128, 6 classes.
"""
import os, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "gas_drift.csv")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
RANDOM = 42
CLASS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia", 4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}

MODELS = {
    "KNN": (KNeighborsClassifier(), {
        "clf__n_neighbors": [3, 5, 11, 21],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    }),
    "GaussianNB": (GaussianNB(), {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
    }),
    "LDA": (LinearDiscriminantAnalysis(), {
        "clf__solver": ["lsqr", "eigen"],
        "clf__shrinkage": [None, "auto"],
    }),
    "AdaBoost": (AdaBoostClassifier(random_state=RANDOM), {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.5, 1.0],
        "clf__estimator__max_depth": [1, 2],
    }),
}

def get_base_estimator(name):
    if name == "AdaBoost":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM),
                                  random_state=RANDOM)
    obj, _ = MODELS[name]
    # fresh copy
    from sklearn.base import clone
    return clone(obj)

def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(DATA)
    X = df.drop(columns=["class"]).values
    y = df["class"].values  # 1..6
    print(f"Data: X={X.shape}, classes={sorted(np.unique(y))}")

    # Unified split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)

    summary, cv_rows = [], []
    best_pipes = {}

    for name, (_, grid) in MODELS.items():
        base = get_base_estimator(name)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base)])
        gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring="f1_macro", n_jobs=-1, verbose=0)
        t0 = time.time()
        gs.fit(Xtr, ytr)
        fit_t = time.time() - t0
        best = gs.best_estimator_
        best_pipes[name] = best
        t1 = time.time()
        yp = best.predict(Xte)
        pred_t = time.time() - t1
        acc = accuracy_score(yte, yp)
        f1m = f1_score(yte, yp, average="macro")
        f1w = f1_score(yte, yp, average="weighted")
        bal = balanced_accuracy_score(yte, yp)
        # outer 5-fold CV on train with best params (fair generalization estimate)
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
        cvres = cross_validate(best, Xtr, ytr, cv=outer,
                               scoring=["accuracy", "f1_macro"], n_jobs=-1,
                               return_train_score=False)
        summary.append(dict(model=name, best_params=json.dumps(gs.best_params_),
                            cv_f1_macro_mean=cvres["test_f1_macro"].mean(),
                            cv_f1_macro_std=cvres["test_f1_macro"].std(),
                            cv_acc_mean=cvres["test_accuracy"].mean(),
                            test_acc=acc, test_f1_macro=f1m, test_f1_weighted=f1w,
                            test_bal_acc=bal, fit_time_s=round(fit_t, 2),
                            predict_time_s=round(pred_t, 3)))
        for i, s in enumerate(cvres["test_f1_macro"]):
            cv_rows.append(dict(model=name, fold=i, f1_macro=s,
                                acc=cvres["test_accuracy"][i]))
        print(f"[{name}] best={gs.best_params_} test_acc={acc:.4f} F1macro={f1m:.4f} fit={fit_t:.1f}s")
        print(classification_report(yte, yp, digits=3))
        # confusion matrix
        cm = confusion_matrix(yte, yp, labels=[1,2,3,4,5,6])
        disp = ConfusionMatrixDisplay(cm, display_labels=[CLASS_NAMES[i] for i in [1,2,3,4,5,6]])
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"cm_{name}.png"), dpi=130)
        plt.close(fig)

    pd.DataFrame(summary).to_csv(os.path.join(RES, "metrics_summary.csv"), index=False)
    pd.DataFrame(cv_rows).to_csv(os.path.join(RES, "cv_folds.csv"), index=False)

    # Bar: F1 test
    s = pd.DataFrame(summary).sort_values("test_f1_macro", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(s["model"], s["test_f1_macro"])
    ax.set_ylabel("F1-macro (test)")
    ax.set_title("Test F1-macro by model (Gas Drift)")
    for i, v in enumerate(s["test_f1_macro"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_bar.png"), dpi=130)
    plt.close(fig)

    # Cost vs performance scatter
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, r in s.iterrows():
        ax.scatter(r["fit_time_s"], r["test_f1_macro"], s=90)
        ax.annotate(r["model"], (r["fit_time_s"], r["test_f1_macro"]),
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Fit time incl. GridSearch (s, log)")
    ax.set_ylabel("Test F1-macro")
    ax.set_title("Cost vs Performance")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cost_vs_f1.png"), dpi=130)
    plt.close(fig)

    # Decision boundaries on PCA-2D (visualization only: same scaler+PCA for all)
    from sklearn.base import clone
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=2, random_state=RANDOM).fit(scaler.transform(Xtr))
    Xtr2 = pca.transform(scaler.transform(Xtr))
    for name, pipe in best_pipes.items():
        # Train a clone of the best estimator on 2D PCA space (visualization only)
        clf2d = clone(pipe.named_steps["clf"])
        clf2d.fit(Xtr2, ytr)
        fig, ax = plt.subplots(figsize=(6, 5))
        DecisionBoundaryDisplay.from_estimator(clf2d, Xtr2, response_method="predict",
                                              cmap="RdYlBu", alpha=0.5, ax=ax, eps=0.5)
        ax.scatter(Xtr2[:, 0], Xtr2[:, 1], c=ytr, cmap="Spectral", s=5, alpha=0.8,
                   edgecolors="k", linewidths=0.2)
        ax.set_title(f"Decision boundary (PCA-2D) - {name}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"boundary_{name}.png"), dpi=130)
        plt.close(fig)

    print("Saved results/ + figures/")

if __name__ == "__main__":
    main()
