# svm_A2_B2.py
# Objectif : entraîner un SVM sur A2 (sans SMOTE) puis sur B2 (avec SMOTE).

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

print("=== SVM sur A2 (sans SMOTE) ===")

# 1) Localisation des données -------------------------------------------------

# Ce script est dans PROJET_FINAL/src/models/
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "processed" / "datasets_final"

print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATASET_DIR)

# 2) Chargement des datasets A2 et test (scalés) ------------------------------

X_train_A2 = pd.read_csv(DATASET_DIR / "X_train_A2.csv")
X_test_scaled = pd.read_csv(DATASET_DIR / "X_test_scaled.csv")

y_train = pd.read_csv(DATASET_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATASET_DIR / "y_test.csv").iloc[:, 0]

y_train.name = "risque_chd_10ans"
y_test.name = "risque_chd_10ans"

print("\n--- Shapes du jeu A2 ---")
print("X_train_A2   :", X_train_A2.shape)
print("X_test_scaled:", X_test_scaled.shape)
print("y_train      :", y_train.shape)
print("y_test       :", y_test.shape)

print("\nRépartition de la cible dans y_train :")
print(y_train.value_counts(normalize=True))

# 3) Définition du modèle SVM pour A2 ----------------------------------------

# On utilise un SVM avec noyau RBF, adapté à des frontières non linéaires.
# C=1.0 et gamma="scale" sont des valeurs raisonnables pour un premier modèle.
# probability=True permet d'obtenir des probabilités (nécessaire pour AUC, Brier, etc.).

svm_A2 = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42,
)

print("\n=== Entraînement du SVM sur A2 (sans SMOTE) ===")
svm_A2.fit(X_train_A2, y_train)
print("✅ Modèle SVM_A2 entraîné.")

# 4) Prédictions et évaluation sur le test (A2) -------------------------------

y_prob_A2 = svm_A2.predict_proba(X_test_scaled)[:, 1]
y_pred_A2 = (y_prob_A2 >= 0.5).astype(int)

print("\n=== Évaluation de SVM_A2 sur le jeu de test ===")

acc_A2 = accuracy_score(y_test, y_pred_A2)
print(f"Accuracy globale : {acc_A2:.3f}")

cm_A2 = confusion_matrix(y_test, y_pred_A2)
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm_A2)

tn_A2, fp_A2, fn_A2, tp_A2 = cm_A2.ravel()
print(f"\nDétails matrice : TN={tn_A2}, FP={fp_A2}, FN={fn_A2}, TP={tp_A2}")

print("\nRapport de classification (SVM_A2) :")
print(classification_report(y_test, y_pred_A2, digits=3))

auc_roc_A2 = roc_auc_score(y_test, y_prob_A2)
print(f"AUC-ROC (probabilités) : {auc_roc_A2:.3f}")

precision_A2, recall_A2, _ = precision_recall_curve(y_test, y_prob_A2)
auc_pr_A2 = average_precision_score(y_test, y_prob_A2)
print(f"AUC-PR (classe 1) : {auc_pr_A2:.3f}")

brier_A2 = brier_score_loss(y_test, y_prob_A2)
print(f"Brier score : {brier_A2:.3f} (plus petit = mieux)")

print("\n=== Fin du bloc SVM_A2 (sans SMOTE) ===")

# 5) SVM sur B2 (avec SMOTE + scaling) ---------------------------------------

print("\n=== SVM sur B2 (avec SMOTE) ===")

X_train_B2 = pd.read_csv(DATASET_DIR / "X_train_B2.csv")
y_train_B2 = pd.read_csv(DATASET_DIR / "y_train_B2.csv").iloc[:, 0]
y_train_B2.name = "risque_chd_10ans"

print("\n--- Shapes du jeu B2 ---")
print("X_train_B2 :", X_train_B2.shape)
print("y_train_B2 :", y_train_B2.shape)

svm_B2 = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42,
)

print("\n=== Entraînement du SVM sur B2 (avec SMOTE) ===")
svm_B2.fit(X_train_B2, y_train_B2)
print("✅ Modèle SVM_B2 entraîné.")

# Prédictions sur le même test standardisé
y_prob_B2 = svm_B2.predict_proba(X_test_scaled)[:, 1]
y_pred_B2 = (y_prob_B2 >= 0.5).astype(int)

print("\n=== Évaluation de SVM_B2 sur le jeu de test ===")

acc_B2 = accuracy_score(y_test, y_pred_B2)
print(f"Accuracy globale : {acc_B2:.3f}")

cm_B2 = confusion_matrix(y_test, y_pred_B2)
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm_B2)

tn_B2, fp_B2, fn_B2, tp_B2 = cm_B2.ravel()
print(f"\nDétails matrice : TN={tn_B2}, FP={fp_B2}, FN={fn_B2}, TP={tp_B2}")

print("\nRapport de classification (SVM_B2) :")
print(classification_report(y_test, y_pred_B2, digits=3))

auc_roc_B2 = roc_auc_score(y_test, y_prob_B2)
print(f"AUC-ROC (probabilités) : {auc_roc_B2:.3f}")

precision_B2, recall_B2, _ = precision_recall_curve(y_test, y_prob_B2)
auc_pr_B2 = average_precision_score(y_test, y_prob_B2)
print(f"AUC-PR (classe 1) : {auc_pr_B2:.3f}")

brier_B2 = brier_score_loss(y_test, y_prob_B2)
print(f"Brier score : {brier_B2:.3f} (plus petit = mieux)")

print("\n=== Fin du bloc SVM_B2 (avec SMOTE) ===")

