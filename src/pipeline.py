"""Pipeline de modélisation du risque de diabète (CDC BRFSS 2015).

Fonctions réutilisées par `train.py` et par le notebook `02_modeling.ipynb`.
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
TARGET = "Diabetes_binary"
SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def load_splits(data_dir=ROOT / "data"):
    tr = pd.read_csv(Path(data_dir) / "train.csv")
    te = pd.read_csv(Path(data_dir) / "test.csv")
    return tr.drop(columns=TARGET), tr[TARGET].astype(int), te.drop(columns=TARGET), te[TARGET].astype(int)


def get_models():
    return {
        "Random Forest (baseline)": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Régression logistique": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED, verbose=-1),
    }


def oof_proba(model, X, y, n_splits=5):
    """Probabilités out-of-fold : servent à choisir le seuil sans toucher au test."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def pick_thresholds(y, proba, target_recall=0.80):
    """Deux points de fonctionnement choisis sur l'OOF : F1 max et dépistage (rappel >= cible)."""
    p, r, th = precision_recall_curve(y, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    thr_f1 = float(th[f1.argmax()])
    ok = np.where(r[:-1] >= target_recall)[0]
    thr_screen = float(th[ok[-1]]) if len(ok) else float(th[0])
    return {"f1_max": thr_f1, f"depistage_rappel_{int(target_recall * 100)}": thr_screen}


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"seuil": round(thr, 4), "precision": precision_score(y, pred), "rappel": recall_score(y, pred),
            "specificite": tn / (tn + fp), "f1": f1_score(y, pred),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def global_metrics(y, proba):
    return {"roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "brier": brier_score_loss(y, proba), "prevalence": float(np.mean(y))}


# ---------- figures ----------
COLORS = {"Random Forest (baseline)": "#9aa5b1", "Régression logistique": "#5b8def", "LightGBM": "#d6336c"}


def plot_roc_pr(y, probas, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=COLORS.get(name), lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=COLORS.get(name), lw=2, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].axhline(np.mean(y), color="k", ls="--", lw=1, label=f"Hasard ({np.mean(y):.2f})")
    ax[0].set(title="Courbe ROC (test)", xlabel="Taux de faux positifs", ylabel="Rappel")
    ax[1].set(title="Courbe précision-rappel (test)", xlabel="Rappel", ylabel="Précision")
    for a in ax:
        a.legend(fontsize=8, loc="lower right" if a is ax[0] else "upper right")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_tradeoff(y, proba, thresholds, path):
    ts = np.linspace(0.02, 0.8, 200)
    prec = [precision_score(y, proba >= t, zero_division=0) for t in ts]
    rec = [recall_score(y, proba >= t) for t in ts]
    f1 = [f1_score(y, proba >= t) for t in ts]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, rec, lw=2, label="Rappel (diabétiques détectés)", color="#d6336c")
    ax.plot(ts, prec, lw=2, label="Précision", color="#5b8def")
    ax.plot(ts, f1, lw=2, label="F1", color="#2b2d42")
    ax.axvline(0.5, color="#9aa5b1", ls=":", label="Seuil par défaut 0,5")
    for name, t in thresholds.items():
        ax.axvline(t, ls="--", lw=1.2, color="#2b2d42" if name == "f1_max" else "#d6336c",
                   label=f"{name} = {t:.2f}")
    ax.set(title="Choix du seuil de décision (LightGBM, test)", xlabel="Seuil", ylabel="Score", ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y, proba, thr, title, path):
    import seaborn as sns
    cm = confusion_matrix(y, proba >= thr)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="RdPu", cbar=False, ax=ax,
                xticklabels=["Non diabétique", "Diabétique"], yticklabels=["Non diabétique", "Diabétique"])
    ax.set(title=title, xlabel="Prédit", ylabel="Réel")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap(model, X, path, n=3000):
    import shap
    Xs = X.sample(n, random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure()
    shap.summary_plot(sv, Xs, show=False, max_display=12)
    plt.title("Contribution des variables au risque (SHAP)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
