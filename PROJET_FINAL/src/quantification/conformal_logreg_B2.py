# PROJET_FINAL/src/quantification/conformal_logreg_B2.py
"""
Conformal prediction pour la régression logistique LogReg_B2 (dataset B2).

Principe :
- On utilise un schéma split-conformal.
- On divise le jeu d'entraînement B2 en :
    * un sous-ensemble "train_conformal" pour entraîner la régression logistique
    * un sous-ensemble "calibration" pour calculer les scores de non-conformité
- Score de non-conformité = 1 - p(chapeau)(y_vrai)
- Pour un niveau de confiance 1 - alpha (ex : 0.90), on calcule un quantile
  sur les scores de calibration. Ce seuil est ensuite utilisé pour construire,
  pour chaque individu du test, un ensemble prédictif {0}, {1} ou {0,1}.

Ce script ne modifie pas de fichiers, il affiche juste les résultats.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

print("=== Conformal prediction pour la régression logistique (LogReg_B2) ===")

# -------------------------------------------------------------------------
# 1) Chargement des données
# -------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "datasets_final"


print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATA_DIR)

# Jeux B2 (train équilibré + scaling) et jeu de test standardisé
X_train_B2 = pd.read_csv(DATA_DIR / "X_train_B2.csv")
y_train_B2 = pd.read_csv(DATA_DIR / "y_train_B2.csv").iloc[:, 0]
X_test_scaled = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

y_train_B2.name = "risque_chd_10ans"
y_test.name = "risque_chd_10ans"

print("\n--- Shapes ---")
print("X_train_B2   :", X_train_B2.shape)
print("y_train_B2   :", y_train_B2.shape)
print("X_test_scaled:", X_test_scaled.shape)
print("y_test       :", y_test.shape)

print("\nRépartition de la cible dans y_train_B2 :")
print(y_train_B2.value_counts(normalize=True))

print("\nRépartition de la cible dans y_test :")
print(y_test.value_counts(normalize=True))

# -------------------------------------------------------------------------
# 2) Split train B2 en 'train_conformal' et 'calibration'
# -------------------------------------------------------------------------

alpha = 0.10  # niveau de risque (=> 90% de confiance)
test_size_calib = 0.2  # 20% du train pour la calibration

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=test_size_calib,
    random_state=42
)

for idx_train_conf, idx_calib in sss.split(X_train_B2, y_train_B2):
    X_train_conf = X_train_B2.iloc[idx_train_conf].reset_index(drop=True)
    y_train_conf = y_train_B2.iloc[idx_train_conf].reset_index(drop=True)
    X_calib = X_train_B2.iloc[idx_calib].reset_index(drop=True)
    y_calib = y_train_B2.iloc[idx_calib].reset_index(drop=True)

print("\n--- Split pour conformal ---")
print("Train_conformal : X =", X_train_conf.shape, ", y =", y_train_conf.shape)
print("Calibration     : X =", X_calib.shape, ", y =", y_calib.shape)

# -------------------------------------------------------------------------
# 3) Entraîner la régression logistique sur train_conformal
# -------------------------------------------------------------------------

logreg = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=1000,
    class_weight=None
)

logreg.fit(X_train_conf, y_train_conf)
print("\n Modèle LogReg_B2 (conformal) entraîné sur le sous-échantillon d'apprentissage.")

# -------------------------------------------------------------------------
# 4) Scores de non-conformité sur l'ensemble de calibration
# -------------------------------------------------------------------------

proba_calib = logreg.predict_proba(X_calib)  # shape (n_calib, 2)

# Pour chaque individu de calibration, probabilité de la classe vraie
# y_calib est en float (0.0 / 1.0) -> on le convertit en entier
y_calib_int = y_calib.astype(int).values
p_true_calib = proba_calib[np.arange(len(y_calib_int)), y_calib_int]

# Score de non-conformité : 1 - p(chapeau)(y_vrai)
nonconformity_calib = 1.0 - p_true_calib

# Quantile empirique au niveau (1 - alpha)
# Formule classique en split-conformal :
# q_hat = quantile_{ceil((n_cal+1)*(1-alpha))/n_cal}
n_cal = len(nonconformity_calib)
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_hat = np.quantile(nonconformity_calib, q_level, interpolation="higher")

print("\n--- Non-conformité sur l'échantillon de calibration ---")
print(f"Nombre d'observations calibration : {n_cal}")
print(f"Niveau de confiance cible 1 - alpha = {1 - alpha:.2f}")
print(f"Niveau de quantile utilisé q_level = {q_level:.3f}")
print(f"Seuil de non-conformité q_hat ≈ {q_hat:.4f}")

# -------------------------------------------------------------------------
# 5) Construction des ensembles prédictifs sur le jeu de test
# -------------------------------------------------------------------------

proba_test = logreg.predict_proba(X_test_scaled)  # shape (n_test, 2)
p_test_class0 = proba_test[:, 0]
p_test_class1 = proba_test[:, 1]

# Non-conformité hypothétique si la vraie classe était 0 ou 1
nonconf_y0 = 1.0 - p_test_class0
nonconf_y1 = 1.0 - p_test_class1

prediction_sets = []
contains_true_label = []

for i in range(len(y_test)):
    S = []
    # On inclut la classe 0 si son score de non-conformité est ≤ q_hat
    if nonconf_y0[i] <= q_hat:
        S.append(0)
    # On inclut la classe 1 si son score de non-conformité est ≤ q_hat
    if nonconf_y1[i] <= q_hat:
        S.append(1)

    prediction_sets.append(S)
    contains_true_label.append(y_test.iloc[i] in S)

prediction_sets = np.array(prediction_sets, dtype=object)
contains_true_label = np.array(contains_true_label, dtype=bool)

# -------------------------------------------------------------------------
# 6) Résultats globaux : couverture et taille des ensembles
# -------------------------------------------------------------------------

coverage = contains_true_label.mean()

sizes = np.array([len(S) for S in prediction_sets])
prop_size1 = np.mean(sizes == 1)
prop_empty = np.mean(sizes == 0)
prop_both = np.mean(sizes == 2)

print("\n=== Résultats conformal sur le jeu de test ===")
print(f"Couverture empirique (vrai label ∈ ensemble prédictif) : {coverage:.3f}")
print(f"Niveau théorique attendu (1 - alpha)                    : {1 - alpha:.3f}")

print("\nDistribution de la taille des ensembles prédictifs :")
print(f" - Ensembles vides     (∅)   : {prop_empty*100:.1f} %")
print(f" - Ensembles à 1 classe      : {prop_size1*100:.1f} %")
print(f" - Ensembles à 2 classes {{0,1}} : {prop_both*100:.1f} %")

# Détail sur les ensembles à une seule classe
mask_singleton = (sizes == 1)
singletons = prediction_sets[mask_singleton]
true_labels_singletons = y_test.values[mask_singleton]

n_single_0 = np.sum([ (len(S) == 1 and S[0] == 0) for S in singletons ])
n_single_1 = np.sum([ (len(S) == 1 and S[0] == 1) for S in singletons ])

print("\nParmi les ensembles de taille 1 :")
print(f" - {{0}} : {n_single_0} cas")
print(f" - {{1}} : {n_single_1} cas")

# -------------------------------------------------------------------------
# 7) Quelques exemples concrets
# -------------------------------------------------------------------------

print("\n=== Exemples d'ensembles prédictifs (5 premières observations) ===")
for i in range(5):
    print(
        f"Individu {i} : p(y=1) = {p_test_class1[i]:.3f} "
        f"-> ensemble prédictif S(x) = {prediction_sets[i]} "
        f"(y_vrai = {y_test.iloc[i]})"
    )

print("\n=== Fin du script conformal_logreg_B2 ===")

