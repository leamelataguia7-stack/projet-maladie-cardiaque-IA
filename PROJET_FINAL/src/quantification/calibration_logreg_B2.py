# calibration_logreg_B2.py
# Objectif : évaluer la calibration et le seuil de décision du modèle LogReg_B2
# - Régression logistique sur B2 (SMOTE + scaling)
# - Évaluation sur le test : seuil 0.5 + recherche d'un seuil optimal (F1)
# - Brier score (calibration globale)
# - Préparation des données pour ROC / PR / calibration

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

print("=== Évaluation de la calibration du modèle LogReg_B2 ===")

# 1) Chemins et chargement des données ----------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "processed" / "datasets_final"

print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATASET_DIR)

# X_train_B2, X_test_scaled, y_train_B2, y_test
X_train_B2 = pd.read_csv(DATASET_DIR / "X_train_B2.csv")
y_train_B2 = pd.read_csv(DATASET_DIR / "y_train_B2.csv").iloc[:, 0]

X_test_scaled = pd.read_csv(DATASET_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(DATASET_DIR / "y_test.csv").iloc[:, 0]

y_train_B2.name = "risque_chd_10ans"
y_test.name = "risque_chd_10ans"

print("\n--- Shapes ---")
print("X_train_B2   :", X_train_B2.shape)
print("X_test_scaled:", X_test_scaled.shape)
print("y_train_B2   :", y_train_B2.shape)
print("y_test       :", y_test.shape)

print("\nRépartition de la cible dans y_test :")
print(y_test.value_counts(normalize=True))

# 2) Entraînement de la régression logistique sur B2 --------------------------
print("\n=== Entraînement de LogReg_B2 sur le jeu B2 ===")

log_reg_B2 = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

log_reg_B2.fit(X_train_B2, y_train_B2)
print("✅ Modèle LogReg_B2 ré-entraîné.")

# 3) Probabilités et métriques au seuil 0.5 ----------------------------------
print("\n=== Évaluation sur le jeu de test : seuil 0.5 ===")

# Probabilité d'appartenir à la classe 1 (événement)
y_prob = log_reg_B2.predict_proba(X_test_scaled)[:, 1]

# Prédiction binaire au seuil 0.5
threshold_default = 0.5
y_pred_05 = (y_prob >= threshold_default).astype(int)

acc_05 = accuracy_score(y_test, y_pred_05)
recall_05 = recall_score(y_test, y_pred_05, pos_label=1)
precision_05 = precision_score(y_test, y_pred_05, pos_label=1)
f1_05 = f1_score(y_test, y_pred_05, pos_label=1)

auc_roc = roc_auc_score(y_test, y_prob)
auc_pr = average_precision_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

cm_05 = confusion_matrix(y_test, y_pred_05)
tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()

print(f"Accuracy (seuil 0.5)      : {acc_05:.3f}")
print(f"Recall cl.1 (seuil 0.5)   : {recall_05:.3f}")
print(f"Precision cl.1 (0.5)      : {precision_05:.3f}")
print(f"F1 cl.1 (0.5)             : {f1_05:.3f}")
print(f"AUC-ROC (probas)          : {auc_roc:.3f}")
print(f"AUC-PR  (classe 1)        : {auc_pr:.3f}")
print(f"Brier score               : {brier:.3f}")
print("Matrice de confusion (0.5) :")
print(cm_05)
print(f"Détails : TN={tn_05}, FP={fp_05}, FN={fn_05}, TP={tp_05}")

# 4) Recherche d'un seuil optimal (max F1 classe 1) ---------------------------
print("\n=== Recherche d'un seuil optimal (max F1 classe 1) ===")

best_threshold = 0.5
best_f1 = f1_05

thresholds = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00

for th in thresholds:
    y_pred_th = (y_prob >= th).astype(int)
    f1_th = f1_score(y_test, y_pred_th, pos_label=1)
    if f1_th > best_f1:
        best_f1 = f1_th
        best_threshold = th

print(f"Seuil optimal (F1 max) ≈ {best_threshold:.2f}")
print(f"F1 classe 1 à ce seuil : {best_f1:.3f}")

# Recalcul des métriques au seuil optimal
y_pred_opt = (y_prob >= best_threshold).astype(int)
acc_opt = accuracy_score(y_test, y_pred_opt)
recall_opt = recall_score(y_test, y_pred_opt, pos_label=1)
precision_opt = precision_score(y_test, y_pred_opt, pos_label=1)

cm_opt = confusion_matrix(y_test, y_pred_opt)
tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()

print("\n=== Métriques au seuil optimal ===")
print(f"Accuracy (opt)      : {acc_opt:.3f}")
print(f"Recall cl.1 (opt)   : {recall_opt:.3f}")
print(f"Precision cl.1 (opt): {precision_opt:.3f}")
print(f"F1 cl.1 (opt)       : {best_f1:.3f}")
print("Matrice de confusion (seuil optimal) :")
print(cm_opt)
print(f"Détails : TN={tn_opt}, FP={fp_opt}, FN={fn_opt}, TP={tp_opt}")

print("\nRappel :")
print("- Le seuil 0.5 est un choix par défaut, adapté quand les coûts d'erreur sont symétriques.")
print("- Ici, on a cherché un seuil qui maximise le F1 de la classe 1 (événements),")
print("  ce qui met l'accent sur un compromis entre sensibilité (recall) et précision.")
print("=== Fin de l'évaluation de la calibration de LogReg_B2 ===")

