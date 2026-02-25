"""
Prédiction conforme label-conditionnelle (Mondrian) pour LogReg A2
avec variation du seuil alpha.

- Modèle de base : LogisticRegression A2 (MICE + scaling, sans SMOTE),
  entraînée sur TOUT X_train_A2 / y_train.
- Calibration : sous-échantillon stratifié de 20 % du train.
- Score de non-conformité : s(x, y) = 1 - P(Y = y | x).
- Pour chaque alpha dans ALPHA_GRID :
    * on calcule q0 et q1 (quantiles par classe),
    * on construit les ensembles Γ(x) sur le test,
    * on calcule couverture globale, couverture par classe,
      taille moyenne des ensembles et répartition {0}, {1}, {0,1}.
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit


# -------------------------------------------------------------------
# 1. Paramètres généraux
# -------------------------------------------------------------------

# Grille des alpha à tester (tu peux changer les valeurs)
ALPHA_GRID = [0.05, 0.10, 0.20, 0.25]

TEST_SIZE_CALIB = 0.20   # proportion du train pour la calibration
RANDOM_STATE = 42        # reproductibilité

# -------------------------------------------------------------------
# 2. Chargement des données
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

y_train_arr = y_train.to_numpy().astype(int)
y_test_arr  = y_test.to_numpy().astype(int)


# -------------------------------------------------------------------
# 3. Définition du set de calibration (stratifié)
# -------------------------------------------------------------------

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE_CALIB,
    random_state=RANDOM_STATE
)

for _, calib_index in sss.split(X_train, y_train):
    X_calib = X_train.iloc[calib_index].reset_index(drop=True)
    y_calib = y_train.iloc[calib_index].reset_index(drop=True)

print("\n--- Set de calibration ---")
print("X_calib shape :", X_calib.shape)
print("y_calib shape :", y_calib.shape)
print("\nRépartition de la cible dans y_calib :")
print(y_calib.value_counts(normalize=True))

y_calib_arr = y_calib.to_numpy().astype(int)


# -------------------------------------------------------------------
# 4. Entraînement du modèle final (LogReg A2) sur TOUT le train
# -------------------------------------------------------------------

logreg = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE
)

logreg.fit(X_train, y_train)

print("\n--- Modèle FINAL entraîné : Logistic Regression A2 sur TOUT le train ---")

proba_calib = logreg.predict_proba(X_calib)   # shape (n_calib, 2)
proba_test  = logreg.predict_proba(X_test)    # shape (n_test, 2)


# -------------------------------------------------------------------
# 5. Fonctions utilitaires pour Mondrian
# -------------------------------------------------------------------

def mondrian_quantile(scores, alpha):
    """
    Quantile split-conformal pour une classe donnée.
    Renvoie (q, n, k) où q est le quantile (1 - alpha),
    n la taille de l'échantillon de calibration et k l'indice (1-based).
    """
    n = len(scores)
    sorted_scores = np.sort(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return sorted_scores[k - 1], n, k


def compute_mondrian_for_alpha(alpha, proba_calib, y_calib_arr, proba_test, y_test_arr):
    """
    Applique la prédiction conforme Mondrian pour un alpha donné.
    - Calcule q0 et q1 pour la classe 0 et 1.
    - Construit Γ(x) sur le test.
    - Calcule couverture globale, couverture par classe,
      taille moyenne et répartition {0}, {1}, {0,1}.
    Retourne un dict de résultats.
    """

    # Scores de non-conformité : s = 1 - p(y_true)
    p_true = proba_calib[np.arange(len(y_calib_arr)), y_calib_arr]
    scores = 1.0 - p_true

    # Séparation par classe
    mask_0 = (y_calib_arr == 0)
    mask_1 = (y_calib_arr == 1)

    scores_0 = scores[mask_0]
    scores_1 = scores[mask_1]

    q0, n0, k0 = mondrian_quantile(scores_0, alpha)
    q1, n1, k1 = mondrian_quantile(scores_1, alpha)

    # Construction des ensembles sur le test
    n_test = len(y_test_arr)
    prediction_sets = []

    for i in range(n_test):
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

    # Couverture globale
    is_covered = np.array([
        y_test_arr[i] in prediction_sets[i] for i in range(n_test)
    ])
    coverage_global = is_covered.mean()

    # Couverture par classe
    mask_test_0 = (y_test_arr == 0)
    mask_test_1 = (y_test_arr == 1)

    coverage_0 = is_covered[mask_test_0].mean()
    coverage_1 = is_covered[mask_test_1].mean()

    # Taille moyenne des ensembles
    set_sizes = np.array([len(s) for s in prediction_sets])
    avg_set_size = set_sizes.mean()

    # Répartition des ensembles
    count_0    = np.sum([s == [0] for s in prediction_sets])
    count_1    = np.sum([s == [1] for s in prediction_sets])
    count_both = np.sum([set(s) == {0, 1} for s in prediction_sets])

    return {
        "alpha": alpha,
        "q0": q0,
        "q1": q1,
        "n0_calib": n0,
        "n1_calib": n1,
        "k0": k0,
        "k1": k1,
        "coverage_global": coverage_global,
        "coverage_0": coverage_0,
        "coverage_1": coverage_1,
        "avg_set_size": avg_set_size,
        "count_0": count_0,
        "count_1": count_1,
        "count_both": count_both,
        "n_test": n_test,
    }


# -------------------------------------------------------------------
# 6. Boucle sur la grille d'alpha : Mondrian seulement
# -------------------------------------------------------------------

print("\n================ Prédiction conforme Mondrian : variation de alpha ================")

results_mondrian = []

for alpha in ALPHA_GRID:
    print(f"\n========== alpha = {alpha} ==========")

    res = compute_mondrian_for_alpha(
        alpha=alpha,
        proba_calib=proba_calib,
        y_calib_arr=y_calib_arr,
        proba_test=proba_test,
        y_test_arr=y_test_arr
    )
    results_mondrian.append(res)

    print("\n--- Résultats Mondrian pour alpha =", alpha, "---")
    print(f"q0 (classe 0)       : {res['q0']:.4f} (n0_calib={res['n0_calib']}, k0={res['k0']})")
    print(f"q1 (classe 1)       : {res['q1']:.4f} (n1_calib={res['n1_calib']}, k1={res['k1']})")
    print(f"Couverture globale  : {res['coverage_global']:.3f}")
    print(f"Couverture classe 0 : {res['coverage_0']:.3f}")
    print(f"Couverture classe 1 : {res['coverage_1']:.3f}")
    print(f"Taille moyenne Γ(x) : {res['avg_set_size']:.3f}")
    print(f"Singleton {{0}}     : {res['count_0']} ({res['count_0'] / res['n_test']:.3f})")
    print(f"Singleton {{1}}     : {res['count_1']} ({res['count_1'] / res['n_test']:.3f})")
    print(f"Ensembles {{0,1}}   : {res['count_both']} ({res['count_both'] / res['n_test']:.3f})")

print("\n================ Fin de l'analyse Mondrian (variation de alpha) ================")

