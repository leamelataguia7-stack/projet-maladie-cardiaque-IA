# rf_A1_B1.py
# Objectif : entraîner une Random Forest sur A1 (sans SMOTE) puis, plus tard, sur B1 (avec SMOTE).
# Étape actuelle : RF_A1 uniquement.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

print("=== Random Forest sur A1 (sans SMOTE) ===")

# 1) Localisation des données -------------------------------------------------

# Ce script est dans PROJET_FINAL/src/models/
# On remonte de 2 niveaux pour arriver à PROJET_FINAL
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "processed" / "datasets_final"

print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATASET_DIR)

# 2) Chargement du dataset A1 (MICE brut, sans scaling, sans SMOTE) ----------

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

# 3) Définition du modèle Random Forest --------------------------------------

# Choix "raisonnables" pour un premier modèle :
# - n_estimators = 300 arbres (stabilité)
# - max_depth = None (laisse l'arbre se développer, mais on limite min_samples_leaf)
# - min_samples_leaf = 10 pour éviter un surapprentissage trop fort
# - random_state = 42 pour la reproductibilité

rf_A1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

print("\n=== Entraînement de la Random Forest sur A1 (sans SMOTE) ===")
rf_A1.fit(X_train_A1, y_train)
print("✅ Modèle RF_A1 entraîné.")

# 4) Prédictions sur le jeu de test ------------------------------------------

# Probabilités pour la classe 1 (événement)
y_prob_A1 = rf_A1.predict_proba(X_test_raw)[:, 1]

# Prédictions binaires avec seuil 0.5
y_pred_A1 = (y_prob_A1 >= 0.5).astype(int)

# 5) Évaluation des performances ---------------------------------------------

print("\n=== Évaluation de RF_A1 sur le jeu de test ===")

acc_A1 = accuracy_score(y_test, y_pred_A1)
print(f"Accuracy globale : {acc_A1:.3f}")

cm_A1 = confusion_matrix(y_test, y_pred_A1)
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm_A1)

tn_A1, fp_A1, fn_A1, tp_A1 = cm_A1.ravel()
print(f"\nDétails matrice : TN={tn_A1}, FP={fp_A1}, FN={fn_A1}, TP={tp_A1}")

print("\nRapport de classification (RF_A1) :")
print(classification_report(y_test, y_pred_A1, digits=3))

auc_roc_A1 = roc_auc_score(y_test, y_prob_A1)
print(f"AUC-ROC (probabilités) : {auc_roc_A1:.3f}")

precision_A1, recall_A1, _ = precision_recall_curve(y_test, y_prob_A1)
auc_pr_A1 = average_precision_score(y_test, y_prob_A1)
print(f"AUC-PR (classe 1) : {auc_pr_A1:.3f}")

brier_A1 = brier_score_loss(y_test, y_prob_A1)
print(f"Brier score : {brier_A1:.3f} (plus petit = mieux)")

print("\n=== Fin du bloc RF_A1 (sans SMOTE) ===")

# 6) Random Forest sur B1 (avec SMOTE) --------------------------------------

print("\n=== Random Forest sur B1 (avec SMOTE) ===")

# Chargement du train B1 (MICE + SMOTE, sans scaling)
X_train_B1 = pd.read_csv(DATASET_DIR / "X_train_B1.csv")
y_train_B1 = pd.read_csv(DATASET_DIR / "y_train_B1.csv").iloc[:, 0]
y_train_B1.name = "risque_chd_10ans"

print("\n--- Shapes du jeu B1 ---")
print("X_train_B1 :", X_train_B1.shape)
print("y_train_B1 :", y_train_B1.shape)

# On garde le même test réaliste que pour A1 : X_test_raw, y_test

rf_B1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

print("\n=== Entraînement de la Random Forest sur B1 (avec SMOTE) ===")
rf_B1.fit(X_train_B1, y_train_B1)
print("✅ Modèle RF_B1 entraîné.")

# Probabilités pour la classe 1 sur le test
y_prob_B1 = rf_B1.predict_proba(X_test_raw)[:, 1]

# Prédictions binaires avec seuil 0.5
y_pred_B1 = (y_prob_B1 >= 0.5).astype(int)

print("\n=== Évaluation de RF_B1 sur le jeu de test ===")

acc_B1 = accuracy_score(y_test, y_pred_B1)
print(f"Accuracy globale : {acc_B1:.3f}")

cm_B1 = confusion_matrix(y_test, y_pred_B1)
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm_B1)

tn_B1, fp_B1, fn_B1, tp_B1 = cm_B1.ravel()
print(f"\nDétails matrice : TN={tn_B1}, FP={fp_B1}, FN={fn_B1}, TP={tp_B1}")

print("\nRapport de classification (RF_B1) :")
print(classification_report(y_test, y_pred_B1, digits=3))

auc_roc_B1 = roc_auc_score(y_test, y_prob_B1)
print(f"AUC-ROC (probabilités) : {auc_roc_B1:.3f}")

precision_B1, recall_B1, _ = precision_recall_curve(y_test, y_prob_B1)
auc_pr_B1 = average_precision_score(y_test, y_prob_B1)
print(f"AUC-PR (classe 1) : {auc_pr_B1:.3f}")

brier_B1 = brier_score_loss(y_test, y_prob_B1)
print(f"Brier score : {brier_B1:.3f} (plus petit = mieux)")

print("\n=== Fin du bloc RF_B1 (avec SMOTE) ===")
