# logreg_A2_B2.py
# Étape 0 : vérifier la qualité des données pour la régression logistique
#           sur le dataset A2 (MICE + standardisation, sans SMOTE).

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
print("=== Vérifications des données pour la régression logistique (A2) ===")

# 1) Localisation des fichiers ----------------------------------------------

# Ce script est dans PROJET_FINAL/src/models/
# On remonte donc de 2 niveaux pour arriver à PROJET_FINAL :
BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = BASE_DIR / "data" / "processed" / "datasets_final"

print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATASET_DIR)

# 2) Chargement du dataset A2 -----------------------------------------------

X_train = pd.read_csv(DATASET_DIR / "X_train_A2.csv")
X_test = pd.read_csv(DATASET_DIR / "X_test_scaled.csv")

y_train = pd.read_csv(DATASET_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATASET_DIR / "y_test.csv").iloc[:, 0]

y_train.name = "risque_chd_10ans"
y_test.name = "risque_chd_10ans"

print("\n--- Shapes des jeux A2 ---")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# 3) Vérifier l'absence de valeurs manquantes -------------------------------

print("\n=== Vérification des valeurs manquantes ===")
n_na_X_train = X_train.isna().sum().sum()
n_na_X_test = X_test.isna().sum().sum()
n_na_y_train = y_train.isna().sum()
n_na_y_test = y_test.isna().sum()

print(f"Nombre de NA dans X_train : {n_na_X_train}")
print(f"Nombre de NA dans X_test  : {n_na_X_test}")
print(f"Nombre de NA dans y_train : {n_na_y_train}")
print(f"Nombre de NA dans y_test  : {n_na_y_test}")

# 4) Types des variables ----------------------------------------------------

print("\n=== Types des variables explicatives (X_train) ===")
print(X_train.dtypes)

# 5) Corrélations fortes entre variables -----------------------------------

print("\n=== Corrélations fortes (|r| > 0.8) dans X_train ===")

# Matrice de corrélation absolue
corr = X_train.corr().abs()

# On ne garde que la partie supérieure de la matrice (pour ne pas dupliquer)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

high_corr_pairs = []

for col in upper.columns:
    for row in upper.index:
        value = upper.loc[row, col]
        if pd.notna(value) and value > 0.8:
            high_corr_pairs.append((row, col, value))

if not high_corr_pairs:
    print("Aucune paire de variables avec |r| > 0.8 (corrélation très forte).")
else:
    print("Paires avec corrélation forte :")
    for row, col, value in sorted(high_corr_pairs, key=lambda x: -x[2]):
        print(f"  - {row} / {col} : r = {value:.3f}")

# 6) Multicolinéarité : calcul des VIF --------------------------------------

print("\n=== Analyse de la multicolinéarité (VIF) ===")
# VIF = Variance Inflation Factor
# VIF > 5 ou 10 peut indiquer une multicolinéarité problématique pour la logistique.

# On ajoute une constante pour le modèle
X_vif = sm.add_constant(X_train)

vif_data = pd.DataFrame()
vif_data["variable"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print(vif_data)

print("\nInterprétation :")
print("- VIF ≈ 1 : pas de multicolinéarité.")
print("- VIF entre 1 et 5 : acceptable.")
print("- VIF > 5 (ou > 10) : multicolinéarité forte → coefficients de la logistique moins stables.")

print("\n=== Fin des vérifications A2 pour la régression logistique ===")

# 7) Modèle de régression logistique sur A2 (sans SMOTE) --------------------

print("\n=== Entraînement de la régression logistique sur A2 (sans SMOTE) ===")

# On utilise les objets déjà chargés :
# - X_train (X_train_A2)
# - X_test  (X_test_scaled)
# - y_train, y_test

log_reg_A2 = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=1000,
)

log_reg_A2.fit(X_train, y_train)
print(" Modèle LogReg_A2 entraîné.")

# Probabilités prédites pour la classe 1 (événement) sur le test
y_prob_A2 = log_reg_A2.predict_proba(X_test)[:, 1]

# Prédictions binaires avec seuil 0.5
y_pred_A2 = (y_prob_A2 >= 0.5).astype(int)

print("\n=== Évaluation de LogReg_A2 sur le jeu de test ===")

# Accuracy
acc_A2 = accuracy_score(y_test, y_pred_A2)
print(f"Accuracy globale : {acc_A2:.3f}")

# Matrice de confusion
cm_A2 = confusion_matrix(y_test, y_pred_A2)
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm_A2)

# Rapport de classification
print("\nRapport de classification (LogReg_A2) :")
print(classification_report(y_test, y_pred_A2, digits=3))

# AUC-ROC
auc_roc_A2 = roc_auc_score(y_test, y_prob_A2)
print(f"AUC-ROC (probabilités) : {auc_roc_A2:.3f}")

# AUC-PR
precision_A2, recall_A2, _ = precision_recall_curve(y_test, y_prob_A2)
auc_pr_A2 = average_precision_score(y_test, y_prob_A2)
print(f"AUC-PR (classe 1) : {auc_pr_A2:.3f}")

# Brier score
brier_A2 = brier_score_loss(y_test, y_prob_A2)
print(f"Brier score : {brier_A2:.3f} (plus petit = mieux)")

print("\n=== Fin du bloc LogReg_A2 ===")

# 8) Modèle de régression logistique sur B2 (avec SMOTE + scaling) ----------

print("\n=== Entraînement de la régression logistique sur B2 (avec SMOTE) ===")

# Chargement du train B2 (SMOTE + standardisation)
X_train_B2 = pd.read_csv(DATASET_DIR / "X_train_B2.csv")
y_train_B2 = pd.read_csv(DATASET_DIR / "y_train_B2.csv").iloc[:, 0]
y_train_B2.name = "risque_chd_10ans"

print("\n--- Shapes du jeu B2 ---")
print("X_train_B2 :", X_train_B2.shape)
print("y_train_B2 :", y_train_B2.shape)

# On garde le même test que pour A2 : X_test (X_test_scaled), y_test

log_reg_B2 = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=1000,
)

log_reg_B2.fit(X_train_B2, y_train_B2)
print(" Modèle LogReg_B2 entraîné.")

# Probabilités prédites pour la classe 1 sur le test
y_prob_B2 = log_reg_B2.predict_proba(X_test)[:, 1]

# Prédictions binaires avec seuil 0.5
y_pred_B2 = (y_prob_B2 >= 0.5).astype(int)

print("\n=== Évaluation de LogReg_B2 sur le jeu de test ===")

# Accuracy
acc_B2 = accuracy_score(y_test, y_pred_B2)
print(f"Accuracy globale : {acc_B2:.3f}")

# Matrice de confusion
cm_B2 = confusion_matrix(y_test, y_pred_B2)
print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm_B2)

# Pour ton futur tableau : extraire TN, FP, FN, TP
tn_B2, fp_B2, fn_B2, tp_B2 = cm_B2.ravel()
print(f"\nDétails matrice : TN={tn_B2}, FP={fp_B2}, FN={fn_B2}, TP={tp_B2}")

# Rapport de classification
print("\nRapport de classification (LogReg_B2) :")
print(classification_report(y_test, y_pred_B2, digits=3))

# AUC-ROC
auc_roc_B2 = roc_auc_score(y_test, y_prob_B2)
print(f"AUC-ROC (probabilités) : {auc_roc_B2:.3f}")

# AUC-PR
precision_B2, recall_B2, _ = precision_recall_curve(y_test, y_prob_B2)
auc_pr_B2 = average_precision_score(y_test, y_prob_B2)
print(f"AUC-PR (classe 1) : {auc_pr_B2:.3f}")

# Brier score
brier_B2 = brier_score_loss(y_test, y_prob_B2)
print(f"Brier score : {brier_B2:.3f} (plus petit = mieux)")

print("\n=== Fin du bloc LogReg_B2 ===")
