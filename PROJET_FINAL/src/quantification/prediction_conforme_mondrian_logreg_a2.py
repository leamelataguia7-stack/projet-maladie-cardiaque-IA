"""
Mondrian (label-conditional) conformal prediction
pour la régression logistique A2 (sans SMOTE, avec scaling).

- Modèle de base : LogisticRegression entraînée sur TOUT X_train_A2 / y_train.
- Calibration : sous-échantillon stratifié du train (20 %) pour calculer
  les scores de non-conformité séparément par classe (0 et 1).
- Score de non-conformité : s(x, y) = 1 - P(Y = y | x).
- Quantiles :
    q0 = quantile (1 - alpha0) des scores chez les y=0 (classe majoritaire),
    q1 = quantile (1 - alpha1) des scores chez les y=1 (classe minoritaire).
- Sur le test : ensemble prédictif Γ(x) construit avec q0 et q1.

Objectif : obtenir une couverture plus équilibrée entre classes
(class-conditional / Mondrian CP), plus adaptée aux données déséquilibrées.
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report


# -------------------------------------------------------------------
# 1. Paramètres généraux
# -------------------------------------------------------------------

# Niveaux de risque par classe (peuvent être égaux ou différents)
alpha0 = 0.10   # risque cible pour la classe 0
alpha1 = 0.10   # risque cible pour la classe 1

test_size_calib = 0.20   # proportion du train utilisée comme calibration
RANDOM_STATE = 42        # reproductibilité


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
# 4. Définition du set de calibration (Mondrian)
# -------------------------------------------------------------------
# On choisit un sous-échantillon du train pour calibrer les scores,
# en gardant la stratification.
#
# NB : ici on entraîne le modèle FINAL sur TOUT X_train, y_train.
# Le set de calibration sert uniquement pour estimer les quantiles
# de non-conformité par classe (label-conditional / Mondrian).

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=test_size_calib,
    random_state=RANDOM_STATE
)

for _, calib_index in sss.split(X_train, y_train):
    X_calib = X_train.iloc[calib_index].reset_index(drop=True)
    y_calib = y_train.iloc[calib_index].reset_index(drop=True)

print("\n--- Set de calibration ---")
print("X_calib shape       :", X_calib.shape)
print("y_calib shape       :", y_calib.shape)

print("\nRépartition de la cible dans y_calib :")
print(y_calib.value_counts(normalize=True))


# -------------------------------------------------------------------
# 5. Entraînement du modèle final (Logistic Regression A2)
# -------------------------------------------------------------------

logreg = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE
)

logreg.fit(X_train, y_train)

print("\n--- Modèle FINAL entraîné : Logistic Regression A2 sur TOUT le train ---")


# -------------------------------------------------------------------
# 6. Scores de non-conformité sur la calibration, séparés par classe
# -------------------------------------------------------------------
# Mondrian / label-conditional :
#   - on sépare les scores pour y=0 et y=1,
#   - on calcule un quantile séparé pour chaque classe.

proba_calib = logreg.predict_proba(X_calib)  # shape (n_calib, 2)
y_calib_arr = y_calib.to_numpy().astype(int)

# Score de non-conformité s_i = 1 - P(Y = y_i | x_i)
p_true = proba_calib[np.arange(len(y_calib_arr)), y_calib_arr]
nonconformity_scores = 1.0 - p_true

# On sépare les scores par classe
mask_0 = (y_calib_arr == 0)
mask_1 = (y_calib_arr == 1)

scores_0 = nonconformity_scores[mask_0]
scores_1 = nonconformity_scores[mask_1]

print("\n--- Scores de non-conformité par classe (calibration) ---")
print(f"Nombre de calib. classe 0 : {len(scores_0)}")
print(f"Nombre de calib. classe 1 : {len(scores_1)}")
print("Exemples scores_0 :", scores_0[:10])
print("Exemples scores_1 :", scores_1[:10])


def mondrian_quantile(scores, alpha):
    """
    Calcule le quantile (1 - alpha) selon la formule split-conformal
    pour une classe donnée (scores = non-conformité de cette classe).
    """
    n = len(scores)
    sorted_scores = np.sort(scores)
    # formule standard : k = ceil((n + 1) * (1 - alpha))
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return sorted_scores[k - 1], n, k


q0, n0, k0 = mondrian_quantile(scores_0, alpha0)
q1, n1, k1 = mondrian_quantile(scores_1, alpha1)

print("\n--- Quantiles Mondrian (label-conditional) ---")
print(f"Classe 0 : alpha0 = {alpha0}, n0 = {n0}, k0 (1-based) = {k0}, q0 = {q0:.4f}")
print(f"Classe 1 : alpha1 = {alpha1}, n1 = {n1}, k1 (1-based) = {k1}, q1 = {q1:.4f}")


# -------------------------------------------------------------------
# 7. Construction des ensembles prédictifs Γ(x) sur le test
# -------------------------------------------------------------------
# Pour chaque x_test :
#   - s(x,0) = 1 - P(Y=0 | x)
#   - s(x,1) = 1 - P(Y=1 | x)
#   - inclure 0 si s(x,0) <= q0
#   - inclure 1 si s(x,1) <= q1

proba_test = logreg.predict_proba(X_test)  # shape (n_test, 2)
y_test_arr = y_test.to_numpy().astype(int)

prediction_sets = []

for i in range(len(X_test)):
    p0, p1 = proba_test[i, 0], proba_test[i, 1]

    s0 = 1.0 - p0
    s1 = 1.0 - p1

    pred_set = []
    if s0 <= q0:
        pred_set.append(0)
    if s1 <= q1:
        pred_set.append(1)

    prediction_sets.append(pred_set)

prediction_sets = np.array(prediction_sets, dtype=object)


# -------------------------------------------------------------------
# 8. Évaluation : couverture globale, par classe, et efficacité
# -------------------------------------------------------------------

n_test = len(y_test_arr)
is_covered = np.array([
    y_test_arr[i] in prediction_sets[i] for i in range(n_test)
])

coverage_global = is_covered.mean()

print("\n--- Couverture globale (Mondrian) sur le test ---")
print(f"Couverture globale = {coverage_global:.3f}")

# Couverture par classe
mask_test_0 = (y_test_arr == 0)
mask_test_1 = (y_test_arr == 1)

coverage_0 = is_covered[mask_test_0].mean()
coverage_1 = is_covered[mask_test_1].mean()

print("\n--- Couverture par classe ---")
print(f"Couverture classe 0 : {coverage_0:.3f}")
print(f"Couverture classe 1 : {coverage_1:.3f}")

# Taille moyenne des ensembles
set_sizes = np.array([len(s) for s in prediction_sets])
avg_set_size = set_sizes.mean()

print("\n--- Efficacité des ensembles prédictifs ---")
print("Taille moyenne des ensembles Γ(x) :", avg_set_size)

# Répartition des types d'ensembles : {0}, {1}, {0,1}
count_0 = np.sum([s == [0] for s in prediction_sets])
count_1 = np.sum([s == [1] for s in prediction_sets])
count_both = np.sum([set(s) == {0, 1} for s in prediction_sets])

print("\n--- Répartition des ensembles prédictifs ---")
print(f"Total test         : {n_test}")
print(f"Singleton {{0}}    : {count_0}  ({count_0 / n_test:.3f})")
print(f"Singleton {{1}}    : {count_1}  ({count_1 / n_test:.3f})")
print(f"Ensembles {{0,1}}  : {count_both}  ({count_both / n_test:.3f})")


# -------------------------------------------------------------------
# 9. Prédiction classique (seuil 0.15) pour comparaison
# -------------------------------------------------------------------
# On rappelle les performances du modèle final en point prediction
# avec le seuil ~F1-opt (0.15) que tu utilises déjà.

threshold = 0.15
y_pred_point = (proba_test[:, 1] >= threshold).astype(int)

print("\n--- Prédiction ponctuelle (seuil 0.15) ---")
print(classification_report(y_test_arr, y_pred_point, digits=3))

