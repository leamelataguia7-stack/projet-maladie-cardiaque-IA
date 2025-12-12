# ============================================================
# SCRIPT 03 - COMPARAISON DES MÉTHODES D'IMPUTATION
# Étape B1 : Comparaison des distributions
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use("seaborn-v0_8")

# Création du dossier de sortie
os.makedirs("PROJET_FINAL/COMPARAISONS_IMPUTATIONS", exist_ok=True)

# ============================================================
# 1. Chargement des datasets imputés (A1, KNN, MICE)
# ============================================================

X_train_simple = pd.read_csv("PROJET_FINAL/X_train_A1.csv")
X_train_knn = pd.read_csv("PROJET_FINAL/X_train_KNN.csv")
X_train_mice = pd.read_csv("PROJET_FINAL/X_train_MICE.csv")

print("Datasets chargés avec succès.")
print("Dimensions SIMPLE :", X_train_simple.shape)
print("Dimensions KNN    :", X_train_knn.shape)
print("Dimensions MICE   :", X_train_mice.shape)

# ============================================================
# 2. Variables à comparer
# ============================================================

variables_a_tester = [
    "cholesterol_total", 
    "tension_systolique", 
    "tension_diastolique", 
    "imc", 
    "glucose",
    "cigarettes_par_jour"
]

# ============================================================
# 3. Graphiques de comparaison (A1 vs KNN vs MICE)
# ============================================================

for col in variables_a_tester:
    plt.figure(figsize=(10, 6))

    sns.kdeplot(X_train_simple[col], label="Simple", linewidth=2)
    sns.kdeplot(X_train_knn[col], label="KNN", linewidth=2)
    sns.kdeplot(X_train_mice[col], label="MICE", linewidth=2)

    plt.title(f"Comparaison des distributions imputées : {col}")
    plt.xlabel(col)
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"PROJET_FINAL/COMPARAISONS_IMPUTATIONS/B1_comparaison_{col}.png")
    plt.show()

print("\n>>> Comparaison graphique (B1) terminée.")

# ============================================================
#      ÉVALUATION DES MÉTHODES
# B2 — Comparaison statistique
# B3 — Stabilité
# B4 — Mini-modélisation
# ============================================================

from scipy.stats import ks_2samp, levene
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import numpy as np

# ============================================================
# CHARGEMENT DES DONNÉES POUR B2, B3, B4
# ============================================================

# ============= Chargement des datasets depuis PROJET_FINAL =================

A1_train = pd.read_csv("PROJET_FINAL/X_train_A1.csv")
A1_test  = pd.read_csv("PROJET_FINAL/X_test_A1.csv")

KNN_train = pd.read_csv("PROJET_FINAL/X_train_KNN.csv")
KNN_test  = pd.read_csv("PROJET_FINAL/X_test_KNN.csv")

MICE_train = pd.read_csv("PROJET_FINAL/X_train_MICE.csv")
MICE_test  = pd.read_csv("PROJET_FINAL/X_test_MICE.csv")

# ================= Chargement des cibles =================

y_train_A1 = pd.read_csv("PROJET_FINAL/y_train_A1.csv").squeeze()
y_test_A1  = pd.read_csv("PROJET_FINAL/y_test_A1.csv").squeeze()

y_train_KNN = pd.read_csv("PROJET_FINAL/y_train_KNN.csv").squeeze()
y_test_KNN  = pd.read_csv("PROJET_FINAL/y_test_KNN.csv").squeeze()

y_train_MICE = pd.read_csv("PROJET_FINAL/y_train_MICE.csv").squeeze()
y_test_MICE  = pd.read_csv("PROJET_FINAL/y_test_MICE.csv").squeeze()

# Ajout de la colonne cible dans chaque dataset
A1_train["risque_chd_10ans"] = y_train_A1
A1_test["risque_chd_10ans"]  = y_test_A1

KNN_train["risque_chd_10ans"] = y_train_KNN
KNN_test["risque_chd_10ans"]  = y_test_KNN

MICE_train["risque_chd_10ans"] = y_train_MICE
MICE_test["risque_chd_10ans"]  = y_test_MICE

# Supprime les NaN restants côté KNN en utilisant la médiane/mode
for col in KNN_train.columns:
    if KNN_train[col].dtype == "object":
        KNN_train[col] = KNN_train[col].fillna(KNN_train[col].mode()[0])
        KNN_test[col] = KNN_test[col].fillna(KNN_test[col].mode()[0])
    else:
        KNN_train[col] = KNN_train[col].fillna(KNN_train[col].median())
        KNN_test[col] = KNN_test[col].fillna(KNN_test[col].median())


variables_continues = [
    "age", "cigarettes_par_jour", "cholesterol_total",
    "tension_systolique", "tension_diastolique", "imc",
    "frequence_cardiaque", "glucose"
]

TARGET = "risque_chd_10ans"

# On réinsère la cible dans chaque dataframe
A1_train[TARGET] = y_train_A1
KNN_train[TARGET] = y_train_KNN
MICE_train[TARGET] = y_train_MICE

A1_test[TARGET] = y_test_A1
KNN_test[TARGET] = y_test_KNN
MICE_test[TARGET] = y_test_MICE

# ============================================================
# B2 — COMPARAISON STATISTIQUE
# ============================================================

print("\n===== B2 — COMPARAISON STATISTIQUE =====")

results_stats = []

for col in variables_continues:
    ks_simple_mice = ks_2samp(A1_train[col], MICE_train[col])
    ks_knn_mice = ks_2samp(KNN_train[col], MICE_train[col])

    var_simple = levene(A1_train[col], MICE_train[col]).pvalue
    var_knn = levene(KNN_train[col], MICE_train[col]).pvalue

    results_stats.append([
        col,
        round(ks_simple_mice.pvalue, 4),
        round(var_simple, 4),
        round(ks_knn_mice.pvalue, 4),
        round(var_knn, 4)
    ])

df_stats = pd.DataFrame(results_stats,
                        columns=["Variable",
                                 "KS Simple vs MICE",
                                 "Var Simple vs MICE",
                                 "KS KNN vs MICE",
                                 "Var KNN vs MICE"])

print(df_stats)

df_stats.to_csv("PROJET_FINAL/COMPARAISONS_IMPUTATIONS/B2_comparaison_statistique.csv",
                index=False)

# ============================================================
# B3 — STABILITÉ PAR BOOTSTRAP
# ============================================================

print("\n===== B3 — STABILITÉ PAR BOOTSTRAP =====")

def bootstrap_variance(df, col, n_boot=200):
    variances = []
    for _ in range(n_boot):
        sample = resample(df[col])
        variances.append(np.var(sample))
    return np.mean(variances), np.std(variances)

stability_results = []

for col in variables_continues:
    var_A1 = bootstrap_variance(A1_train, col)
    var_KNN = bootstrap_variance(KNN_train, col)
    var_MICE = bootstrap_variance(MICE_train, col)

    stability_results.append([col,
                              var_A1[0], var_A1[1],
                              var_KNN[0], var_KNN[1],
                              var_MICE[0], var_MICE[1]])

df_stability = pd.DataFrame(stability_results,
                            columns=["Variable",
                                     "Var Moy A1", "Var SD A1",
                                     "Var Moy KNN", "Var SD KNN",
                                     "Var Moy MICE", "Var SD MICE"])

print(df_stability)

df_stability.to_csv("PROJET_FINAL/COMPARAISONS_IMPUTATIONS/B3_stabilite_imputation.csv",
                    index=False)

# ============================================================
# B4 — MINI-MODÉLISATION
# ============================================================

print("\n===== B4 — MINI-MODÉLISATION LOGISTIQUE =====")

def quick_model(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    return roc_auc_score(y_test, preds), f1_score(y_test, preds > 0.5)

metrics = []

AUC_A1, F1_A1 = quick_model(A1_train, A1_test)
AUC_KNN, F1_KNN = quick_model(KNN_train, KNN_test)
AUC_MICE, F1_MICE = quick_model(MICE_train, MICE_test)

metrics.append(["Imputation Simple", AUC_A1, F1_A1])
metrics.append(["KNN", AUC_KNN, F1_KNN])
metrics.append(["MICE", AUC_MICE, F1_MICE])

df_metrics = pd.DataFrame(metrics,
                          columns=["Méthode", "AUC", "F1-score"])

print(df_metrics)

df_metrics.to_csv("PROJET_FINAL/COMPARAISONS_IMPUTATIONS/B4_mini_modelisation.csv",
                  index=False)

print("\n>>> Analyse B2, B3, B4 terminée et fichiers sauvegardés.")

#######     Résumé Visuel B1     ####
import matplotlib.pyplot as plt
import seaborn as sns

variables = [
    "cholesterol_total", "tension_systolique",
    "tension_diastolique", "imc", "glucose",
    "cigarettes_par_jour"
]

plt.figure(figsize=(18, 12))
plt.suptitle("Comparaison des distributions imputées : Simple vs KNN vs MICE", fontsize=18)

for i, col in enumerate(variables, 1):
    plt.subplot(3, 2, i)
    sns.kdeplot(X_train_simple[col], label="Simple", linewidth=2)
    sns.kdeplot(X_train_knn[col], label="KNN", linewidth=2)
    sns.kdeplot(X_train_mice[col], label="MICE", linewidth=2)

    # Titre du sous-graphe
    plt.title(col, fontsize=12)

    # Supprimer les labels inutiles pour éviter duplication
    plt.xlabel("")     # enlève le nom sur l’axe X
    plt.ylabel("")

    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
