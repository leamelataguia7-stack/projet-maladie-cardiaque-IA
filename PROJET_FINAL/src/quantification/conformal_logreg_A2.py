"""
Conformal prediction pour la régression logistique A2 (sans SMOTE, avec scaling).

Approche : split-conformal classification binaire.

- Modèle de base : LogisticRegression entraînée sur X_train_A2 / y_train.
- Calibration : sous-échantillon stratifié du train (20 %).
- Score de non-conformité : s = 1 - p(y_vrai) sur l'échantillon de calibration.
- Quantile q_hat choisi pour viser une couverture (1 - alpha) ≈ 90 %.
- Pour chaque observation du test : ensemble prédictif Γ(x) ⊆ {0, 1}.
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report


# -------------------------------------------------------------------
# 1. Paramètres généraux de l'expérience
# -------------------------------------------------------------------

alpha = 0.20            # => on vise une couverture ≈ 90 %
test_size_calib = 0.20  # 20 % du train pour la calibration
RANDOM_STATE = 42       # pour la reproductibilité


# -------------------------------------------------------------------
# 2. Chemins vers les données
# -------------------------------------------------------------------

ROOT_DIR = os.getcwd()
print("\nRépertoire de travail courant (cwd) :", ROOT_DIR)

DATA_DIR = os.path.join(
    "PROJET_FINAL",
    "data",
    "processed",
    "datasets_final"
)

X_train_path = os.path.join(DATA_DIR, "X_train_A2.csv")
y_train_path = os.path.join(DATA_DIR, "y_train.csv")
X_test_path  = os.path.join(DATA_DIR, "X_test_scaled.csv")
y_test_path  = os.path.join(DATA_DIR, "y_test.csv")

print("\n--- Chemins des fichiers utilisés ---")
print("DATA_DIR      :", DATA_DIR)
print("X_train_path  :", X_train_path)
print("y_train_path  :", y_train_path)
print("X_test_path   :", X_test_path)
print("y_test_path   :", y_test_path)


# -------------------------------------------------------------------
# 3. Chargement des données A2 (sans SMOTE, avec scaling)
# -------------------------------------------------------------------

print("\n--- Chargement des données ---")
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).iloc[:, 0]
X_test  = pd.read_csv(X_test_path)
y_test  = pd.read_csv(y_test_path).iloc[:, 0]

print("X_train shape :", X_train.shape)
print("y_train shape :", y_train.shape)
print("X_test shape  :", X_test.shape)
print("y_test shape  :", y_test.shape)

print("\nRépartition de la cible dans y_train :")
print(y_train.value_counts(normalize=True))

print("\nRépartition de la cible dans y_test :")
print(y_test.value_counts(normalize=True))


# -------------------------------------------------------------------
# 4. Split du train en 'train_model' et 'calibration'
# -------------------------------------------------------------------

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=test_size_calib,
    random_state=RANDOM_STATE
)

for train_index, calib_index in sss.split(X_train, y_train):
    X_train_model = X_train.iloc[train_index].reset_index(drop=True)
    y_train_model = y_train.iloc[train_index].reset_index(drop=True)
    X_calib       = X_train.iloc[calib_index].reset_index(drop=True)
    y_calib       = y_train.iloc[calib_index].reset_index(drop=True)

print("\n--- Split train -> (train_model, calibration) ---")
print("X_train_model shape :", X_train_model.shape)
print("y_train_model shape :", y_train_model.shape)
print("X_calib shape       :", X_calib.shape)
print("y_calib shape       :", y_calib.shape)

print("\nRépartition de la cible dans y_train_model :")
print(y_train_model.value_counts(normalize=True))

print("\nRépartition de la cible dans y_calib :")
print(y_calib.value_counts(normalize=True))


# -------------------------------------------------------------------
# 5. Entraînement du modèle de base (LogisticRegression)
# -------------------------------------------------------------------

logreg = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE
)

logreg.fit(X_train_model, y_train_model)

print("\n--- Modèle de base entraîné : Logistic Regression A2 ---")


# -------------------------------------------------------------------
# 6. Scores de non-conformité sur l'échantillon de calibration
# -------------------------------------------------------------------

proba_calib = logreg.predict_proba(X_calib)  # shape (n_calib, 2)
y_calib_arr = y_calib.to_numpy().astype(int)

# probabilité de la vraie classe
p_true = proba_calib[np.arange(len(y_calib_arr)), y_calib_arr]

# score de non-conformité
nonconformity_scores = 1.0 - p_true

print("\n--- Scores de non-conformité sur la calibration ---")
print("Taille calibration :", len(nonconformity_scores))
print("Quelques scores :", nonconformity_scores[:10])


# -------------------------------------------------------------------
# 7. Calcul du quantile q_hat (split-conformal)
# -------------------------------------------------------------------

n_calib = len(nonconformity_scores)
sorted_scores = np.sort(nonconformity_scores)

k = int(np.ceil((n_calib + 1) * (1 - alpha)))  # indice 1-based
k = min(k, n_calib)
q_hat = sorted_scores[k - 1]

print("\n--- Quantile de non-conformité ---")
print(f"alpha = {alpha}")
print(f"n_calib = {n_calib}")
print(f"k (1-based) = {k}")
print(f"q_hat = {q_hat:.4f}")


# -------------------------------------------------------------------
# 8. Construction des ensembles prédictifs sur le test
# -------------------------------------------------------------------

proba_test = logreg.predict_proba(X_test)  # shape (n_test, 2)
y_test_arr = y_test.to_numpy().astype(int)

prediction_sets = []

for i in range(len(X_test)):
    p0, p1 = proba_test[i, 0], proba_test[i, 1]

    s0 = 1.0 - p0  # score si y=0
    s1 = 1.0 - p1  # score si y=1

    pred_set = []
    if s0 <= q_hat:
        pred_set.append(0)
    if s1 <= q_hat:
        pred_set.append(1)

    prediction_sets.append(pred_set)

prediction_sets = np.array(prediction_sets, dtype=object)


# -------------------------------------------------------------------
# 9. Évaluation : couverture et efficacité
# -------------------------------------------------------------------

# 9.1 Couverture empirique
covered = []
for i in range(len(y_test_arr)):
    covered.append(y_test_arr[i] in prediction_sets[i])

covered = np.array(covered)
coverage_empirical = covered.mean()

print("\n--- Couverture empirique sur le test ---")
print(f"Couverture (proportion où y_test ∈ Γ(x)) = {coverage_empirical:.3f}")

# 9.2 Taille moyenne des ensembles prédictifs
set_sizes = np.array([len(s) for s in prediction_sets])
avg_set_size = set_sizes.mean()

print("\n--- Efficacité des ensembles prédictifs ---")
print("Taille moyenne des ensembles Γ(x) :", avg_set_size)


# 9.3 Répartition des types d'ensembles : {0}, {1}, {0,1}
count_0 = np.sum([s == [0] for s in prediction_sets])
count_1 = np.sum([s == [1] for s in prediction_sets])
count_both = np.sum([set(s) == {0, 1} for s in prediction_sets])

n_test = len(y_test_arr)

print("\n--- Répartition des ensembles prédictifs ---")
print(f"Total test         : {n_test}")
print(f"Singleton {{0}}    : {count_0}  ({count_0 / n_test:.3f})")
print(f"Singleton {{1}}    : {count_1}  ({count_1 / n_test:.3f})")
print(f"Ensembles {{0,1}}  : {count_both}  ({count_both / n_test:.3f})")


# -------------------------------------------------------------------
# 10. Prédiction « classique » avec seuil 0,15
# -------------------------------------------------------------------

threshold = 0.15
y_pred_point = (proba_test[:, 1] >= threshold).astype(int)

print("\n--- Prédiction ponctuelle (seuil 0.15) ---")
print(classification_report(y_test_arr, y_pred_point, digits=3))
