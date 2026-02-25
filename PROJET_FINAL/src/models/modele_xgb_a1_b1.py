# xgb_A1_B1.py
# Objectif : entraîner XGBoost sur A1 (sans SMOTE) puis sur B1 (avec SMOTE).

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

print("=== XGBoost sur A1 (sans SMOTE) ===")

# 1) Localisation des données -------------------------------------------------

# Ce script est dans PROJET_FINAL/src/models/
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "processed" / "datasets_final"

print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATASET_DIR)

# 2) Chargement des datasets A1 et test --------------------------------------

X_train_A1 = pd.read_csv(DATASET_DIR / "X_train_A1.csv")
X_test_raw = pd.read_csv(DATASET_DIR / "X_test_raw.csv")

y_train = pd.read_csv(DATASET_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATASET_DIR / "y_test.csv").iloc[:, 0]

y_train.name = "risque_chd_10ans"
y_test.name = "risque_chd_10ans"

print("\n--- Shapes du jeu A1 ---")
print("X_train_A1 :", X_train_A1.shape)
print("X_test_raw :", X_test_raw.shape)
print("y_train    :", y_train.shape)
print("y_test     :", y_test.shape)

print("\nRépartition de la cible dans y_train :")
print(y_train.value_counts(normalize=True))

# 3) Définition du modèle XGBoost pour A1 ------------------------------------

# Paramètres "raisonnables" pour un premier essai :
# - n_estimators = 300 arbres
# - learning_rate = 0.05
# - max_depth = 3
# - subsample et colsample_bytree pour limiter le surapprentissage
# - eval_metric = "logloss" (perte logistique)
# - scale_pos_weight laissé à 1 (on gère le déséquilibre via SMOTE dans B1)

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

print("\n=== Entraînement de XGBoost sur A1 (sans SMOTE) ===")
xgb_A1.fit(X_train_A1, y_train)
print(" Modèle XGB_A1 entraîné.")

# 4) Prédictions et évaluation sur le test (A1) -------------------------------

y_prob_A1 = xgb_A1.predict_proba(X_test_raw)[:, 1]
y_pred_A1 = (y_prob_A1 >= 0.5).astype(int)

print("\n=== Évaluation de XGB_A1 sur le jeu de test ===")

acc_A1 = accuracy_score(y_test, y_pred_A1)
print(f"Accuracy globale : {acc_A1:.3f}")

cm_A1 = confusion_matrix(y_test, y_pred_A1)
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm_A1)

tn_A1, fp_A1, fn_A1, tp_A1 = cm_A1.ravel()
print(f"\nDétails matrice : TN={tn_A1}, FP={fp_A1}, FN={fn_A1}, TP={tp_A1}")

print("\nRapport de classification (XGB_A1) :")
print(classification_report(y_test, y_pred_A1, digits=3))

auc_roc_A1 = roc_auc_score(y_test, y_prob_A1)
print(f"AUC-ROC (probabilités) : {auc_roc_A1:.3f}")

precision_A1, recall_A1, _ = precision_recall_curve(y_test, y_prob_A1)
auc_pr_A1 = average_precision_score(y_test, y_prob_A1)
print(f"AUC-PR (classe 1) : {auc_pr_A1:.3f}")

brier_A1 = brier_score_loss(y_test, y_prob_A1)
print(f"Brier score : {brier_A1:.3f} (plus petit = mieux)")

print("\n=== Fin du bloc XGB_A1 (sans SMOTE) ===")

# 5) XGBoost sur B1 (avec SMOTE) ---------------------------------------------

print("\n=== XGBoost sur B1 (avec SMOTE) ===")

X_train_B1 = pd.read_csv(DATASET_DIR / "X_train_B1.csv")
y_train_B1 = pd.read_csv(DATASET_DIR / "y_train_B1.csv").iloc[:, 0]
y_train_B1.name = "risque_chd_10ans"

print("\n--- Shapes du jeu B1 ---")
print("X_train_B1 :", X_train_B1.shape)
print("y_train_B1 :", y_train_B1.shape)

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

print("\n=== Entraînement de XGBoost sur B1 (avec SMOTE) ===")
xgb_B1.fit(X_train_B1, y_train_B1)
print("Modèle XGB_B1 entraîné.")

# Prédictions sur le même test réaliste
y_prob_B1 = xgb_B1.predict_proba(X_test_raw)[:, 1]
y_pred_B1 = (y_prob_B1 >= 0.5).astype(int)

print("\n=== Évaluation de XGB_B1 sur le jeu de test ===")

acc_B1 = accuracy_score(y_test, y_pred_B1)
print(f"Accuracy globale : {acc_B1:.3f}")

cm_B1 = confusion_matrix(y_test, y_pred_B1)
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm_B1)

tn_B1, fp_B1, fn_B1, tp_B1 = cm_B1.ravel()
print(f"\nDétails matrice : TN={tn_B1}, FP={fp_B1}, FN={fn_B1}, TP={tp_B1}")

print("\nRapport de classification (XGB_B1) :")
print(classification_report(y_test, y_pred_B1, digits=3))

auc_roc_B1 = roc_auc_score(y_test, y_prob_B1)
print(f"AUC-ROC (probabilités) : {auc_roc_B1:.3f}")

precision_B1, recall_B1, _ = precision_recall_curve(y_test, y_prob_B1)
auc_pr_B1 = average_precision_score(y_test, y_prob_B1)
print(f"AUC-PR (classe 1) : {auc_pr_B1:.3f}")

brier_B1 = brier_score_loss(y_test, y_prob_B1)
print(f"Brier score : {brier_B1:.3f} (plus petit = mieux)")

print("\n=== Fin du bloc XGB_B1 (avec SMOTE) ===")

