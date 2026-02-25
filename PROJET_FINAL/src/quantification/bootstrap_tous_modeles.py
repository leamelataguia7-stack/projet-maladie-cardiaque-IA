# bootstrap_all_models.py
# Quantification de l'incertitude par bootstrap sur le JEU DE TEST
# pour les 10 modèles :
#   - LogReg_A2 / LogReg_B2
#   - RF_A1 / RF_B1
#   - GB_A1 / GB_B1
#   - XGB_A1 / XGB_B1
#   - SVM_A2 / SVM_B2
#
# Pour chaque modèle :
#   1) entraînement sur TOUT le train correspondant (A1, A2, B1, B2)
#   2) évaluation "point" sur le test
#   3) bootstrap du test (n_bootstrap échantillons)
#   4) IC 95 % pour Recall_cl1, F1_cl1, AUC-ROC, AUC-PR

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# -------------------------------------------------------------------
# 1) Paramètres généraux
# -------------------------------------------------------------------
N_BOOTSTRAP = 1000  # tu peux réduire à 300 si c'est trop long
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# -------------------------------------------------------------------
# 2) Localisation des données
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "processed" / "datasets_final"

print("=== Script de bootstrap sur le jeu de test pour TOUS les modèles ===")
print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATASET_DIR)

# -------------------------------------------------------------------
# 3) Chargement des jeux d'entraînement et de test
# -------------------------------------------------------------------
# Train
X_train_A1 = pd.read_csv(DATASET_DIR / "X_train_A1.csv")
X_train_A2 = pd.read_csv(DATASET_DIR / "X_train_A2.csv")
X_train_B1 = pd.read_csv(DATASET_DIR / "X_train_B1.csv")
X_train_B2 = pd.read_csv(DATASET_DIR / "X_train_B2.csv")

y_train = pd.read_csv(DATASET_DIR / "y_train.csv").iloc[:, 0]
y_train_B1 = pd.read_csv(DATASET_DIR / "y_train_B1.csv").iloc[:, 0]
y_train_B2 = pd.read_csv(DATASET_DIR / "y_train_B2.csv").iloc[:, 0]

y_train.name = "risque_chd_10ans"
y_train_B1.name = "risque_chd_10ans"
y_train_B2.name = "risque_chd_10ans"

# Test (le même pour tous)
X_test_raw = pd.read_csv(DATASET_DIR / "X_test_raw.csv")
X_test_scaled = pd.read_csv(DATASET_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(DATASET_DIR / "y_test.csv").iloc[:, 0]
y_test.name = "risque_chd_10ans"

print("\n--- Shapes des jeux ---")
print("X_train_A1 :", X_train_A1.shape)
print("X_train_A2 :", X_train_A2.shape)
print("X_train_B1 :", X_train_B1.shape)
print("X_train_B2 :", X_train_B2.shape)
print("X_test_raw :", X_test_raw.shape)
print("X_test_scaled :", X_test_scaled.shape)
print("y_train      :", y_train.shape)
print("y_train_B1   :", y_train_B1.shape)
print("y_train_B2   :", y_train_B2.shape)
print("y_test       :", y_test.shape)

# -------------------------------------------------------------------
# 4) Fonction utilitaire : calcul des métriques sur un (X, y) donné
# -------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Calcule les métriques principales :
    - Accuracy
    - Recall (classe 1)
    - Precision (classe 1)
    - F1 (classe 1)
    - AUC-ROC (si proba et deux classes)
    - AUC-PR (si proba et deux classes)
    """
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    if (y_prob is not None) and (len(np.unique(y_true)) == 2):
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_roc = np.nan
        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except ValueError:
            auc_pr = np.nan
    else:
        auc_roc = np.nan
        auc_pr = np.nan

    return acc, rec, prec, f1, auc_roc, auc_pr


# -------------------------------------------------------------------
# 5) Fonction utilitaire : bootstrap sur le test pour un modèle donné
# -------------------------------------------------------------------
def bootstrap_on_test(model, X_train, y_train, X_test, y_test, model_name):
    """
    1) Entraîne le modèle sur (X_train, y_train)
    2) Évalue sur le test (point estimate)
    3) Fait un bootstrap sur le test :
       - N_BOOTSTRAP échantillons
       - IC 95 % pour Recall_cl1, F1_cl1, AUC-ROC, AUC-PR
    """
    print(f"\n\n=== Bootstrap pour le modèle : {model_name} ===")
    print(f"Train : X={X_train.shape}, y={y_train.shape}")
    print(f"Test  : X={X_test.shape}, y={y_test.shape}")

    # 1) Entraînement
    model.fit(X_train, y_train)

    # 2) Évaluation "point" sur TOUT le test
    if hasattr(model, "predict_proba"):
        y_prob_test = model.predict_proba(X_test)[:, 1]
    else:
        y_prob_test = None

    if y_prob_test is not None:
        y_pred_test = (y_prob_test >= 0.5).astype(int)
    else:
        y_pred_test = model.predict(X_test)

    acc, rec, prec, f1, auc_roc, auc_pr = compute_metrics(
        y_test, y_pred_test, y_prob_test
    )

    print("\n--- Performance ponctuelle sur le test ---")
    print(f"Accuracy      : {acc:.3f}")
    print(f"Recall cl. 1  : {rec:.3f}")
    print(f"Precision cl1 : {prec:.3f}")
    print(f"F1 cl. 1      : {f1:.3f}")
    print(f"AUC-ROC       : {auc_roc:.3f}")
    print(f"AUC-PR        : {auc_pr:.3f}")

    # 3) Bootstrap sur le test
    n_test = len(y_test)
    recall_list = []
    f1_list = []
    aucroc_list = []
    aucpr_list = []

    print(f"\n--- Bootstrap sur le test ({N_BOOTSTRAP} itérations) ---")

    for b in range(N_BOOTSTRAP):
        # Tirage avec remise d'indices du test
        idx = rng.integers(0, n_test, size=n_test)
        y_true_b = y_test.iloc[idx].reset_index(drop=True)
        X_test_b = X_test.iloc[idx].reset_index(drop=True)

        if hasattr(model, "predict_proba"):
            y_prob_b = model.predict_proba(X_test_b)[:, 1]
            y_pred_b = (y_prob_b >= 0.5).astype(int)
        else:
            y_prob_b = None
            y_pred_b = model.predict(X_test_b)

        _, rec_b, _, f1_b, aucroc_b, aucpr_b = compute_metrics(
            y_true_b, y_pred_b, y_prob_b
        )

        recall_list.append(rec_b)
        f1_list.append(f1_b)
        aucroc_list.append(aucroc_b)
        aucpr_list.append(aucpr_b)

        # petit affichage ponctuel
        if (b + 1) in [1, 10, 50, 100, 500, N_BOOTSTRAP]:
            print(f"  itération {b+1}/{N_BOOTSTRAP}...")

    # Conversion en arrays
    recall_arr = np.array(recall_list)
    f1_arr = np.array(f1_list)
    aucroc_arr = np.array(aucroc_list)
    aucpr_arr = np.array(aucpr_list)

    def ic95(x):
        return np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5)

    rec_lo, rec_hi = ic95(recall_arr)
    f1_lo, f1_hi = ic95(f1_arr)
    aucroc_lo, aucroc_hi = ic95(aucroc_arr)
    aucpr_lo, aucpr_hi = ic95(aucpr_arr)

    print("\n--- Résumé bootstrap (IC 95 %) pour", model_name, "---")
    print(f"Recall cl. 1  : {rec:.3f}  (IC95% [{rec_lo:.3f} ; {rec_hi:.3f}])")
    print(f"F1 cl. 1      : {f1:.3f}  (IC95% [{f1_lo:.3f} ; {f1_hi:.3f}])")
    print(f"AUC-ROC       : {auc_roc:.3f}  (IC95% [{aucroc_lo:.3f} ; {aucroc_hi:.3f}])")
    print(f"AUC-PR        : {auc_pr:.3f}  (IC95% [{aucpr_lo:.3f} ; {aucpr_hi:.3f}])")
    print("------------------------------------------------------------")

    # On retourne éventuellement les résultats si tu veux les stocker plus tard
    return {
        "model": model_name,
        "recall": rec,
        "recall_ic": (rec_lo, rec_hi),
        "f1": f1,
        "f1_ic": (f1_lo, f1_hi),
        "auc_roc": auc_roc,
        "aucroc_ic": (aucroc_lo, aucroc_hi),
        "auc_pr": auc_pr,
        "aucpr_ic": (aucpr_lo, aucpr_hi),
    }


# -------------------------------------------------------------------
# 6) Définition des modèles (mêmes hyperparamètres que précédemment)
# -------------------------------------------------------------------
# LOGISTIC REGRESSION
logreg_A2 = LogisticRegression(max_iter=1000, solver="liblinear")
logreg_B2 = LogisticRegression(max_iter=1000, solver="liblinear")

# RANDOM FOREST
rf_A1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rf_B1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# GRADIENT BOOSTING
gb_A1 = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE,
)
gb_B1 = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE,
)

# XGBOOST
xgb_A1 = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
xgb_B1 = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# SVM
svm_A2 = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=RANDOM_STATE,
)
svm_B2 = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=RANDOM_STATE,
)

# -------------------------------------------------------------------
# 7) Lancement du bootstrap pour chaque modèle
# -------------------------------------------------------------------
results = []

# LOGREG
results.append(
    bootstrap_on_test(
        logreg_A2, X_train_A2, y_train, X_test_scaled, y_test, "LogReg_A2 (A2, sans SMOTE)"
    )
)
results.append(
    bootstrap_on_test(
        logreg_B2, X_train_B2, y_train_B2, X_test_scaled, y_test, "LogReg_B2 (B2, avec SMOTE)"
    )
)

# RANDOM FOREST
results.append(
    bootstrap_on_test(
        rf_A1, X_train_A1, y_train, X_test_raw, y_test, "RF_A1 (A1, sans SMOTE)"
    )
)
results.append(
    bootstrap_on_test(
        rf_B1, X_train_B1, y_train_B1, X_test_raw, y_test, "RF_B1 (B1, avec SMOTE)"
    )
)

# GRADIENT BOOSTING
results.append(
    bootstrap_on_test(
        gb_A1, X_train_A1, y_train, X_test_raw, y_test, "GB_A1 (A1, sans SMOTE)"
    )
)
results.append(
    bootstrap_on_test(
        gb_B1, X_train_B1, y_train_B1, X_test_raw, y_test, "GB_B1 (B1, avec SMOTE)"
    )
)

# XGBOOST
results.append(
    bootstrap_on_test(
        xgb_A1, X_train_A1, y_train, X_test_raw, y_test, "XGB_A1 (A1, sans SMOTE)"
    )
)
results.append(
    bootstrap_on_test(
        xgb_B1, X_train_B1, y_train_B1, X_test_raw, y_test, "XGB_B1 (B1, avec SMOTE)"
    )
)

# SVM
results.append(
    bootstrap_on_test(
        svm_A2, X_train_A2, y_train, X_test_scaled, y_test, "SVM_A2 (A2, sans SMOTE)"
    )
)
results.append(
    bootstrap_on_test(
        svm_B2, X_train_B2, y_train_B2, X_test_scaled, y_test, "SVM_B2 (B2, avec SMOTE)"
    )
)

print("\n=== Fin du bootstrap pour tous les modèles ===")

