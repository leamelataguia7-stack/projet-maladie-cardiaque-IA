import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# -------------------------------------------------------------------
# 0. Définir les chemins de manière robuste
#    (indépendant du répertoire depuis lequel on lance le script)
# -------------------------------------------------------------------
# THIS_DIR = dossier où se trouve CE script (…/PROJET_FINAL/src)
THIS_DIR = os.path.dirname(__file__)
# BASE_DIR = dossier PROJET_FINAL
BASE_DIR = os.path.dirname(THIS_DIR)
# Dossier des datasets finaux
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "datasets_final")

# -------------------------------------------------------------------
# 1. Charger les données prétraitées (A2, sans SMOTE)
# -------------------------------------------------------------------
X_train_A2 = pd.read_csv(os.path.join(DATA_DIR, "X_train_A2.csv"))
X_test_scaled = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# -------------------------------------------------------------------
# 2. Entraîner LogReg_A2 et SVM_A2 (si ce n’est pas déjà fait)
#    (adapter les hyperparamètres si besoin pour rester cohérent
#     avec tes autres scripts)
# -------------------------------------------------------------------
logreg_A2 = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    class_weight=None,  # IMPORTANT : pas de pondération ici
)
logreg_A2.fit(X_train_A2, y_train)

svm_A2 = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    class_weight=None,  # idem : pas de pondération
)
svm_A2.fit(X_train_A2, y_train)

# -------------------------------------------------------------------
# 3. Fonction utilitaire pour chercher le seuil optimal
# -------------------------------------------------------------------
def evaluate_thresholds(y_true, y_proba, thresholds):
    """
    Calcule les métriques pour une grille de seuils.
    Retourne un DataFrame avec :
    threshold, accuracy, recall, precision, f1, auc_roc, auc_pr.
    """
    results = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        results.append(
            {
                "threshold": thr,
                "accuracy": acc,
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
            }
        )
    return pd.DataFrame(results)

# -------------------------------------------------------------------
# 4. Probabilités prédites sur le test
# -------------------------------------------------------------------
# Pour LogReg
y_proba_logreg_A2 = logreg_A2.predict_proba(X_test_scaled)[:, 1]
# Pour SVM
y_proba_svm_A2 = svm_A2.predict_proba(X_test_scaled)[:, 1]

# -------------------------------------------------------------------
# 5. Grille de seuils à tester (par exemple de 0.1 à 0.9)
# -------------------------------------------------------------------
thresholds = np.linspace(0.1, 0.9, 81)  # pas de 0.01

# -------------------------------------------------------------------
# 6. Recherche du seuil optimal pour LogReg_A2 (par F1 max)
# -------------------------------------------------------------------
df_thr_logreg = evaluate_thresholds(y_test, y_proba_logreg_A2, thresholds)
thr_opt_logreg = df_thr_logreg.loc[df_thr_logreg["f1"].idxmax(), "threshold"]
print("=== LogReg_A2 : seuil optimal (F1 max) ===")
print(df_thr_logreg.loc[df_thr_logreg["f1"].idxmax()])

# -------------------------------------------------------------------
# 7. Recherche du seuil optimal pour SVM_A2 (par F1 max)
# -------------------------------------------------------------------
df_thr_svm = evaluate_thresholds(y_test, y_proba_svm_A2, thresholds)
thr_opt_svm = df_thr_svm.loc[df_thr_svm["f1"].idxmax(), "threshold"]
print("\n=== SVM_A2 : seuil optimal (F1 max) ===")
print(df_thr_svm.loc[df_thr_svm["f1"].idxmax()])

# -------------------------------------------------------------------
# 8. Sauvegarder les résultats détaillés (scan de seuils)
# -------------------------------------------------------------------
df_thr_logreg.to_csv(
    os.path.join(DATA_DIR, "threshold_scan_logreg_A2.csv"),
    index=False,
)
df_thr_svm.to_csv(
    os.path.join(DATA_DIR, "threshold_scan_svm_A2.csv"),
    index=False,
)

# -------------------------------------------------------------------
# 9. Comparaison explicite 0.5 vs seuil optimal pour LogReg_A2 / SVM_A2
# -------------------------------------------------------------------
def metrics_at_threshold(y_true, y_proba, thr, name="model"):
    y_pred = (y_proba >= thr).astype(int)
    return {
        "model": name,
        "threshold": thr,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "auc_pr": average_precision_score(y_true, y_proba),
    }

row_logreg_05 = metrics_at_threshold(
    y_test, y_proba_logreg_A2, 0.5, "LogReg_A2 (thr=0.5)"
)
row_logreg_opt = metrics_at_threshold(
    y_test, y_proba_logreg_A2, thr_opt_logreg, "LogReg_A2 (thr_opt)"
)

row_svm_05 = metrics_at_threshold(
    y_test, y_proba_svm_A2, 0.5, "SVM_A2 (thr=0.5)"
)
row_svm_opt = metrics_at_threshold(
    y_test, y_proba_svm_A2, thr_opt_svm, "SVM_A2 (thr_opt)"
)

df_compare = pd.DataFrame(
    [row_logreg_05, row_logreg_opt, row_svm_05, row_svm_opt]
)
print("\n=== Comparaison 0.5 vs seuil optimal (A2, sans SMOTE) ===")
print(df_compare)

df_compare.to_csv(
    os.path.join(DATA_DIR, "threshold_compare_A2.csv"),
    index=False,
)
