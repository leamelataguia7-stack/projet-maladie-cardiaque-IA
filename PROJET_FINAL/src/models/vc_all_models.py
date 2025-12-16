# cv_all_models.py
# Validation croisée (5-fold) de tous les modèles :
# LogReg_A2/B2, RF_A1/B1, GB_A1/B1, XGB_A1/B1, SVM_A2/B2
#
# Objectif :
# - Estimer la performance moyenne et la variabilité (écart-type)
#   de chaque modèle sur le JEU D'ENTRAÎNEMENT (train uniquement).
# - Utiliser StratifiedKFold pour respecter le déséquilibre de classes.
#
# IMPORTANT :
# - On n'utilise PAS le jeu de test ici.
# - On ne fait qu'une validation INTERNE sur le train.

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# --------------------------------------------------------------------
# 1) Localisation des données
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "processed" / "datasets_final"

print("=== Script de validation croisée de TOUS les modèles ===")
print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATASET_DIR)

# --------------------------------------------------------------------
# 2) Chargement des jeux d'entraînement
# --------------------------------------------------------------------
# A1 : MICE brut (sans SMOTE, sans scaling)
X_train_A1 = pd.read_csv(DATASET_DIR / "X_train_A1.csv")

# A2 : MICE + scaling (sans SMOTE)
X_train_A2 = pd.read_csv(DATASET_DIR / "X_train_A2.csv")

# B1 : MICE + SMOTE (sans scaling)
X_train_B1 = pd.read_csv(DATASET_DIR / "X_train_B1.csv")
y_train_B1 = pd.read_csv(DATASET_DIR / "y_train_B1.csv").iloc[:, 0]

# B2 : MICE + SMOTE + scaling
X_train_B2 = pd.read_csv(DATASET_DIR / "X_train_B2.csv")
y_train_B2 = pd.read_csv(DATASET_DIR / "y_train_B2.csv").iloc[:, 0]

# y_train original (déséquilibré, sans SMOTE)
y_train = pd.read_csv(DATASET_DIR / "y_train.csv").iloc[:, 0]

y_train.name = "risque_chd_10ans"
y_train_B1.name = "risque_chd_10ans"
y_train_B2.name = "risque_chd_10ans"

print("\n--- Shapes des jeux d'entraînement ---")
print("X_train_A1 :", X_train_A1.shape)
print("X_train_A2 :", X_train_A2.shape)
print("X_train_B1 :", X_train_B1.shape)
print("X_train_B2 :", X_train_B2.shape)
print("y_train    :", y_train.shape)
print("y_train_B1 :", y_train_B1.shape)
print("y_train_B2 :", y_train_B2.shape)

# --------------------------------------------------------------------
# 3) Fonction utilitaire de validation croisée
# --------------------------------------------------------------------
def cross_validate_model(X, y, model, model_name, n_splits=5):
    """
    Effectue une validation croisée stratifiée (n_splits) sur (X, y)
    pour un modèle donné.

    Calcule et affiche :
      - Accuracy
      - Recall (classe 1)
      - Precision (classe 1)
      - F1 (classe 1)
      - AUC-ROC (si proba dispo)
      - AUC-PR (si proba dispo)
    """
    print(f"\n\n=== Validation croisée pour le modèle : {model_name} ===")
    print(f"Shape X : {X.shape}, Shape y : {y.shape}")
    print(f"Répartition de la cible (y) :\n{y.value_counts(normalize=True)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_list = []
    rec_list = []
    prec_list = []
    f1_list = []
    auc_roc_list = []
    auc_pr_list = []

    fold = 0
    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Entraînement du modèle sur le sous-train
        model.fit(X_tr, y_tr)

        # Probabilités si disponibles (nécessaires pour AUC)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = None

        # Prédictions binaires avec seuil 0.5 si proba dispo
        if y_prob is not None:
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_val)

        # Calcul des métriques
        acc = accuracy_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)

        # AUC-ROC et AUC-PR si proba dispo et les deux classes présentes
        if y_prob is not None and len(np.unique(y_val)) == 2:
            try:
                auc_roc = roc_auc_score(y_val, y_prob)
            except ValueError:
                auc_roc = np.nan
            try:
                auc_pr = average_precision_score(y_val, y_prob)
            except ValueError:
                auc_pr = np.nan
        else:
            auc_roc = np.nan
            auc_pr = np.nan

        acc_list.append(acc)
        rec_list.append(rec)
        prec_list.append(prec)
        f1_list.append(f1)
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)

        print(
            f"\nFold {fold}/{n_splits} - "
            f"Acc={acc:.3f}, Recall_cl1={rec:.3f}, "
            f"Precision_cl1={prec:.3f}, F1_cl1={f1:.3f}, "
            f"AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}"
        )

    # Résumé global (moyenne ± écart-type)
    def mean_std(a):
        return np.mean(a), np.std(a)

    acc_m, acc_s = mean_std(acc_list)
    rec_m, rec_s = mean_std(rec_list)
    prec_m, prec_s = mean_std(prec_list)
    f1_m, f1_s = mean_std(f1_list)
    aucroc_m, aucroc_s = mean_std(auc_roc_list)
    aucpr_m, aucpr_s = mean_std(auc_pr_list)

    print("\n--- Résumé CV pour", model_name, "---")
    print(f"Accuracy      : {acc_m:.3f} ± {acc_s:.3f}")
    print(f"Recall cl. 1  : {rec_m:.3f} ± {rec_s:.3f}")
    print(f"Precision cl1 : {prec_m:.3f} ± {prec_s:.3f}")
    print(f"F1 cl. 1      : {f1_m:.3f} ± {f1_s:.3f}")
    print(f"AUC-ROC       : {aucroc_m:.3f} ± {aucroc_s:.3f}")
    print(f"AUC-PR        : {aucpr_m:.3f} ± {aucpr_s:.3f}")
    print("------------------------------------------------------")


# --------------------------------------------------------------------
# 4) LOGISTIC REGRESSION (A2 et B2)
# --------------------------------------------------------------------
print("\n================ LOGISTIC REGRESSION (A2 / B2) ================")

logreg_A2 = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
)

cross_validate_model(X_train_A2, y_train, logreg_A2, "LogReg_A2 (A2, sans SMOTE)")

logreg_B2 = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
)

cross_validate_model(X_train_B2, y_train_B2, logreg_B2, "LogReg_B2 (B2, avec SMOTE)")


# --------------------------------------------------------------------
# 5) RANDOM FOREST (A1 et B1)
# --------------------------------------------------------------------
print("\n================ RANDOM FOREST (A1 / B1) ================")

rf_A1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

cross_validate_model(X_train_A1, y_train, rf_A1, "RF_A1 (A1, sans SMOTE)")

rf_B1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

cross_validate_model(X_train_B1, y_train_B1, rf_B1, "RF_B1 (B1, avec SMOTE)")


# --------------------------------------------------------------------
# 6) GRADIENT BOOSTING (A1 et B1)
# --------------------------------------------------------------------
print("\n================ GRADIENT BOOSTING (A1 / B1) ================")

gb_A1 = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

cross_validate_model(X_train_A1, y_train, gb_A1, "GB_A1 (A1, sans SMOTE)")

gb_B1 = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

cross_validate_model(X_train_B1, y_train_B1, gb_B1, "GB_B1 (B1, avec SMOTE)")


# --------------------------------------------------------------------
# 7) XGBOOST (A1 et B1)
# --------------------------------------------------------------------
print("\n================ XGBOOST (A1 / B1) ================")

xgb_A1 = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

cross_validate_model(X_train_A1, y_train, xgb_A1, "XGB_A1 (A1, sans SMOTE)")

xgb_B1 = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

cross_validate_model(X_train_B1, y_train_B1, xgb_B1, "XGB_B1 (B1, avec SMOTE)")


# --------------------------------------------------------------------
# 8) SVM (A2 et B2)
# --------------------------------------------------------------------
print("\n================ SVM (A2 / B2) ================")

svm_A2 = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42,
)

cross_validate_model(X_train_A2, y_train, svm_A2, "SVM_A2 (A2, sans SMOTE)")

svm_B2 = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42,
)

cross_validate_model(X_train_B2, y_train_B2, svm_B2, "SVM_B2 (B2, avec SMOTE)")

print("\n=== Fin de la validation croisée de tous les modèles ===")

