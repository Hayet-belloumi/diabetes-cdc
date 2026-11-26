"""Entraîne et évalue les modèles, puis sauvegarde le modèle, les métriques et les figures.

Usage (depuis la racine du dépôt) :
    python src/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402


def main():
    reports, figs, models_dir = P.ROOT / "reports", P.ROOT / "reports" / "figures", P.ROOT / "models"
    for d in (reports, figs, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_te, y_te = P.load_splits()
    print(f"Train {X_tr.shape} | Test {X_te.shape} | prévalence {y_tr.mean():.3f}")

    rows, test_probas, fitted, thresholds = [], {}, {}, {}
    for name, model in P.get_models().items():
        oof = P.oof_proba(model, X_tr, y_tr)
        thr = P.pick_thresholds(y_tr, oof)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        g = P.global_metrics(y_te, proba)
        m05, mf1 = P.metrics_at(y_te, proba, 0.5), P.metrics_at(y_te, proba, thr["f1_max"])
        rows.append({"modele": name, "cv_roc_auc": P.roc_auc_score(y_tr, oof), **{f"test_{k}": v for k, v in g.items()},
                     "rappel@0.5": m05["rappel"], "f1@0.5": m05["f1"],
                     "seuil_f1": thr["f1_max"], "rappel@seuil_f1": mf1["rappel"], "f1@seuil_f1": mf1["f1"]})
        test_probas[name], fitted[name], thresholds[name] = proba, model, thr
        print(f"{name:26s} AUC {g['roc_auc']:.3f} | PR-AUC {g['pr_auc']:.3f} | F1@seuil {mf1['f1']:.3f}")

    table = pd.DataFrame(rows).round(4)
    table.to_csv(reports / "model_comparison.csv", index=False)

    best = "LightGBM"
    proba, thr = test_probas[best], thresholds[best]
    screen_key = [k for k in thr if k.startswith("depistage")][0]
    metrics = {
        "modele": best,
        "test": P.global_metrics(y_te, proba),
        "seuil_defaut_0.5": P.metrics_at(y_te, proba, 0.5),
        "seuil_f1_max": P.metrics_at(y_te, proba, thr["f1_max"]),
        screen_key: P.metrics_at(y_te, proba, thr[screen_key]),
        "note": "Seuils choisis par validation croisée 5 plis sur le train (out-of-fold), jamais sur le test.",
    }

    P.plot_roc_pr(y_te, test_probas, figs / "roc_pr_curves.png")
    P.plot_threshold_tradeoff(y_te, proba, thr, figs / "threshold_tradeoff.png")
    P.plot_confusion(y_te, proba, 0.5, "LightGBM, seuil 0,5", figs / "confusion_seuil_050.png")
    P.plot_confusion(y_te, proba, thr[screen_key], f"LightGBM, seuil dépistage {thr[screen_key]:.2f}",
                     figs / "confusion_depistage.png")
    importance = P.plot_shap(fitted[best], X_te, figs / "shap_summary.png")
    metrics["top_variables_shap"] = importance.head(10).round(4).to_dict()

    with open(reports / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False, default=float)
    joblib.dump({"model": fitted[best], "thresholds": thr, "features": list(X_tr.columns)},
                models_dir / "lightgbm.joblib")
    print("\n" + table.to_string(index=False))
    print(json.dumps({k: metrics[k] for k in ["seuil_f1_max", screen_key]}, indent=2, default=float))


if __name__ == "__main__":
    main()
