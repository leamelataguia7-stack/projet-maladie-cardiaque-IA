"""
Bootstrap des métriques du modèle final LogReg_A2 (seuil 0,15)
- Entraîne une régression logistique sur X_train_A2 / y_train
- Évalue sur X_test_scaled / y_test
- Calcule des IC95% bootstrap (B = 1000) pour plusieurs métriques :
  accuracy, recall (classe 1), précision (classe 1), F1 (classe 1),
  AUC-ROC, AUC-PR, Brier score.
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# ---------------------------------------------------------------------------
# 1. Paramètres généraux
# ---------------------------------------------------------------------------

# Chemins relatifs depuis la racine du repo
DATA_DIR = os.path.join("PROJET_FINAL", "data", "processed", "datasets_final")
OUTPUT_DIR = os.path.join("PROJET_FINAL", "outputs", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_A2.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test_scaled.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")

# Paramètres du bootstrap
N_BOOT = 1000
RANDOM_STATE = 42

# Seuil de décision du modèle final
DECISION_THRESHOLD = 0.15

# Nom de la colonne cible (adapter si différent)
TARGET_COL = "risque_chd_10ans"


# ---------------------------------------------------------------------------
# 2. Fonction utilitaire pour calculer les métriques
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_proba, threshold=0.5):
    """
    Calcule les principales métriques de performance à partir des probabilités
    prédites pour la classe 1.
    """
    # Probabilité de la classe 1
    p1 = y_proba

    # Prédiction binaire selon le seuil
    y_pred = (p1 >= threshold).astype(int)

    # Métriques globales
    acc = accuracy_score(y_true, y_pred)
    rec1 = recall_score(y_true, y_pred, pos_label=1)
    prec1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # AUC-ROC et AUC-PR (basées sur les probabilités)
    try:
        auc_roc = roc_auc_score(y_true, p1)
    except ValueError:
        auc_roc = np.nan  # si toutes les étiquettes sont identiques dans l'échantillon

    try:
        auc_pr = average_precision_score(y_true, p1)
    except ValueError:
        auc_pr = np.nan

    # Brier score (utilise les probabilités de la classe 1)
    brier = brier_score_loss(y_true, p1)

    return {
        "accuracy": acc,
        "recall_1": rec1,
        "precision_1": prec1,
        "f1_1": f1_1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier": brier,
    }


# ---------------------------------------------------------------------------
# 3. Chargement des données
# ---------------------------------------------------------------------------

print("Répertoire de travail courant (cwd) :", os.getcwd())
print("\n--- Chemins des fichiers utilisés ---")
print("DATA_DIR     :", DATA_DIR)
print("X_train_path :", X_TRAIN_PATH)
print("y_train_path :", Y_TRAIN_PATH)
print("X_test_path  :", X_TEST_PATH)
print("y_test_path  :", Y_TEST_PATH)

print("\n--- Chargement des données ---")
X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH)

# On s'assure que y_* sont des Series 1D
if TARGET_COL in y_train.columns:
    y_train = y_train[TARGET_COL]
else:
    y_train = y_train.squeeze()

if TARGET_COL in y_test.columns:
    y_test = y_test[TARGET_COL]
else:
    y_test = y_test.squeeze()

print("X_train shape :", X_train.shape)
print("y_train shape :", y_train.shape)
print("X_test shape  :", X_test.shape)
print("y_test shape  :", y_test.shape)

print("\nRépartition de la cible dans y_train :")
print(y_train.value_counts(normalize=True).rename("proportion"))

print("\nRépartition de la cible dans y_test :")
print(y_test.value_counts(normalize=True).rename("proportion"))


# ---------------------------------------------------------------------------
# 4. Entraînement du modèle final (Logistic Regression A2)
# ---------------------------------------------------------------------------

print("\n--- Entraînement du modèle final LogReg_A2 ---")

# ⚠️ IMPORTANT : adapter ici les hyperparamètres pour coller exactement
# à ceux que vous avez retenus pour LogReg_A2 dans vos analyses principales.
logreg = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE,
)

logreg.fit(X_train, y_train)

# Probabilités sur le test
proba_test = logreg.predict_proba(X_test)[:, 1]

# Point estimé sur le test complet
metrics_point = compute_metrics(y_test.values, proba_test, threshold=DECISION_THRESHOLD)

print("\n--- Métriques sur le test (point estimé, seuil = {:.2f}) ---".format(DECISION_THRESHOLD))
for k, v in metrics_point.items():
    print(f"{k:12s} : {v:.3f}")


# ---------------------------------------------------------------------------
# 5. Bootstrap sur le jeu de test
# ---------------------------------------------------------------------------

print("\n--- Bootstrap non paramétrique sur le test (B = {}) ---".format(N_BOOT))

rng = np.random.RandomState(RANDOM_STATE)
n_test = len(y_test)

# Dictionnaire pour stocker les métriques bootstrap
boot_results = {
    "accuracy": [],
    "recall_1": [],
    "precision_1": [],
    "f1_1": [],
    "auc_roc": [],
    "auc_pr": [],
    "brier": [],
}

for b in range(N_BOOT):
    # échantillon bootstrap d'indices
    idx = rng.randint(0, n_test, size=n_test)

    y_boot = y_test.values[idx]
    p_boot = proba_test[idx]

    m_b = compute_metrics(y_boot, p_boot, threshold=DECISION_THRESHOLD)

    for k in boot_results.keys():
        boot_results[k].append(m_b[k])

    if (b + 1) % 100 == 0:
        print(f"  - {b + 1} / {N_BOOT} rééchantillonnages traités")

# Conversion en DataFrame
boot_df = pd.DataFrame(boot_results)


# ---------------------------------------------------------------------------
# 6. Calcul des IC95% bootstrap
# ---------------------------------------------------------------------------

print("\n--- Intervalles de confiance bootstrap (percentiles 2,5 % et 97,5 %) ---")

ci_dict = {}
for metric in boot_results.keys():
    lower = np.percentile(boot_df[metric].dropna(), 2.5)
    upper = np.percentile(boot_df[metric].dropna(), 97.5)
    ci_dict[metric] = {
        "point_estimate": metrics_point[metric],
        "ci2.5": lower,
        "ci97.5": upper,
    }
    print(
        f"{metric:12s} : "
        f"point = {metrics_point[metric]:.3f}, "
        f"IC95% = [{lower:.3f} ; {upper:.3f}]"
    )

ci_df = (
    pd.DataFrame(ci_dict)
    .T.reset_index()
    .rename(columns={"index": "metric"})
)

# Sauvegarde des résultats
output_ci_path = os.path.join(OUTPUT_DIR, "logreg_A2_bootstrap_metrics.csv")
output_boot_samples_path = os.path.join(OUTPUT_DIR, "logreg_A2_bootstrap_samples_raw.csv")

ci_df.to_csv(output_ci_path, index=False)
boot_df.to_csv(output_boot_samples_path, index=False)

print("\nRésumés des IC95% sauvegardés dans :", output_ci_path)
print("Toutes les réalisations bootstrap sauvegardées dans :", output_boot_samples_path)
print("\nAnalyse bootstrap terminée.")
