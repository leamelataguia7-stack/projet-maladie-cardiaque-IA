# ============================================================
#         CHAPITRE 2 - PREPROCESSING FRAMINGHAM
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# =======================
# 1. CHARGEMENT DES DONNÉES
# =======================

df = pd.read_csv("PROJET_FINAL/FRAMINGANG.csv", sep=";")
print("Données chargées :", df.shape)

# =======================
# 2. SUPPRESSION DES NAN DANS LA CIBLE
# =======================

print("Valeurs manquantes dans la cible :", df["TenYearCHD"].isna().sum())

df = df.dropna(subset=["TenYearCHD"])

print("Taille après suppression des NaN cible :", df.shape)

# ============================================================
#  RENOMMAGE DES VARIABLES EN FRANÇAIS
# ============================================================

df = df.rename(columns={
    "male": "sexe_masculin",
    "age": "age",
    "education": "niveau_education",
    "currentSmoker": "fumeur_actuel",
    "cigsPerDay": "cigarettes_par_jour",
    "BPMeds": "traitement_hypotenseur",
    "prevalentStroke": "antecedent_avc",
    "prevalentHyp": "hypertension_connue",
    "diabetes": "diabete",
    "totChol": "cholesterol_total",
    "sysBP": "tension_systolique",
    "diaBP": "tension_diastolique",
    "BMI": "imc",
    "heartRate": "frequence_cardiaque",
    "glucose": "glucose",
    "TenYearCHD": "risque_chd_10ans"
})

print(df.head())
print(df.columns)

# =======================
#  3. REDEFINIR X ET y APRÈS NETTOYAGE
# =======================

target = "risque_chd_10ans"

X = df.drop(columns=[target])
y = df[target]

print("Dimensions X :", X.shape)
print("Dimensions y :", y.shape)

# =======================
# 4. SPLIT STRATIFIÉ
# =======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTaille du Train :", X_train.shape)
print("Taille du Test  :", X_test.shape)
print("Proportion classe positive (Train) :", y_train.mean())
print("Proportion classe positive (Test)  :", y_test.mean())

# ============================================================
# 5. IMPUTATION SIMPLE (MEDIANE POUR NUMÉRIQUES, MODE POUR CATÉGORIELLES)
# ============================================================

# 1️) Définition des colonnes numériques
num_cols = [
    "age", "niveau_education", "cigarettes_par_jour",
    "cholesterol_total", "tension_systolique", "tension_diastolique",
    "imc", "frequence_cardiaque", "glucose"
]

# 2️) Définition des colonnes catégorielles
cat_cols = [
    "sexe_masculin", "fumeur_actuel", "traitement_hypotenseur",
    "antecedent_avc", "hypertension_connue", "diabete"
]

# --- Copier les datasets pour conserver les données brutes ---
X_train_simple = X_train.copy()
X_test_simple = X_test.copy()

# ============================================================
#  Imputation sur TRAIN UNIQUEMENT (anti-leakage)
# ============================================================

# Calcul des médianes et modes basés uniquement sur TRAIN
median_values = X_train_simple[num_cols].median()
mode_values = X_train_simple[cat_cols].mode().iloc[0]

# Appliquer l'imputation
X_train_simple[num_cols] = X_train_simple[num_cols].fillna(median_values)
X_train_simple[cat_cols] = X_train_simple[cat_cols].fillna(mode_values)

# ============================================================
#  Appliquer ces mêmes valeurs au TEST (jamais recalculer sur test)
# ============================================================

X_test_simple[num_cols] = X_test_simple[num_cols].fillna(median_values)
X_test_simple[cat_cols] = X_test_simple[cat_cols].fillna(mode_values)

# ============================================================
#  Vérification finale
# ============================================================

print("\n>>> Vérification des NaN après imputation simple :")
print("Train :\n", X_train_simple.isna().sum())
print("Test  :\n", X_test_simple.isna().sum())

# ============================================================
#  Sauvegarde (A1)
# ============================================================

X_train_simple.to_csv("PROJET_FINAL/X_train_A1.csv", index=False)
X_test_simple.to_csv("PROJET_FINAL/X_test_A1.csv", index=False)
y_train.to_csv("PROJET_FINAL/y_train_A1.csv", index=False)
y_test.to_csv("PROJET_FINAL/y_test_A1.csv", index=False)

print("\n>>> A1 (dataset imputé) sauvegardé avec succès.")

# ============================================================
# 6. IMPUTATION KNN
# ============================================================

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# --- Copier les datasets ---
X_train_knn = X_train.copy()
X_test_knn = X_test.copy()

# --- Colonnes numériques uniquement ---
num_cols = [
    "age", "niveau_education", "cigarettes_par_jour",
    "cholesterol_total", "tension_systolique", "tension_diastolique",
    "imc", "frequence_cardiaque", "glucose"
]

# ============================================================
# TEMPORARY SCALING (pour KNN uniquement)
# ============================================================
scaler_knn = StandardScaler()

X_train_scaled = X_train_knn[num_cols].copy()
X_test_scaled = X_test_knn[num_cols].copy()

# Fit sur TRAIN uniquement
scaler_knn.fit(X_train_scaled)

# Transformer TRAIN et TEST
X_train_scaled = scaler_knn.transform(X_train_scaled)
X_test_scaled = scaler_knn.transform(X_test_scaled)

# ============================================================
# KNN IMPUTATION
# ============================================================

knn_imputer = KNNImputer(n_neighbors=5)

# Fit sur TRAIN uniquement
X_train_knn_imputed = knn_imputer.fit_transform(X_train_scaled)

# Appliquer sur TEST
X_test_knn_imputed = knn_imputer.transform(X_test_scaled)

# ============================================================
# Remettre dans les dataframes
# ============================================================

X_train_knn[num_cols] = X_train_knn_imputed = pd.DataFrame(
    X_train_knn_imputed, columns=num_cols, index=X_train.index
)

X_test_knn[num_cols] = X_test_knn_imputed = pd.DataFrame(
    X_test_knn_imputed, columns=num_cols, index=X_test.index
)

print("\n>>> Imputation KNN terminée.")
print("Vérification des NaN (TRAIN) :\n", X_train_knn.isna().sum())
print("Vérification des NaN (TEST) :\n", X_test_knn.isna().sum())

# ============================================================
# SAUVEGARDE DES VERSIONS KNN
# ============================================================

X_train_knn.to_csv("PROJET_FINAL/X_train_KNN.csv", index=False)
X_test_knn.to_csv("PROJET_FINAL/X_test_KNN.csv", index=False)

print("\n>>> Imputation KNN sauvegardée.")

# ============================================================
# 7. COMPLEMENT : IMPUTATION PAR MODE POUR LES VARIABLES CATEGORIELLES
# ============================================================

# Mode calculé uniquement sur TRAIN
mode_values_cat = X_train[cat_cols].mode().iloc[0]

# Appliquer au TRAIN
X_train_knn[cat_cols] = X_train_knn[cat_cols].fillna(mode_values_cat)

# Appliquer au TEST
X_test_knn[cat_cols] = X_test_knn[cat_cols].fillna(mode_values_cat)

print("\n>>> Complément d'imputation (mode pour cat.) terminé.")
print("NaN restants TRAIN :\n", X_train_knn.isna().sum())
print("NaN restants TEST :\n", X_test_knn.isna().sum())

# ============================================================
# 8. IMPUTATION MICE (Iterative Imputer)
# ============================================================

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# --- Copier les datasets ---
X_train_mice = X_train.copy()
X_test_mice = X_test.copy()

# Colonnes numériques et catégorielles
num_cols = [
    "age", "niveau_education", "cigarettes_par_jour",
    "cholesterol_total", "tension_systolique", "tension_diastolique",
    "imc", "frequence_cardiaque", "glucose"
]

cat_cols = [
    "sexe_masculin", "fumeur_actuel", "traitement_hypotenseur",
    "antecedent_avc", "hypertension_connue", "diabete"
]

# Convertir les catégo en float pour MICE si besoin
X_train_mice[cat_cols] = X_train_mice[cat_cols].astype(float)
X_test_mice[cat_cols] = X_test_mice[cat_cols].astype(float)

# ============================================================
# MICE Imputation
# ============================================================

mice_imputer = IterativeImputer(random_state=42, max_iter=20)

# Fit sur TRAIN uniquement
X_train_mice_imputed = mice_imputer.fit_transform(X_train_mice)

# Appliquer sur TEST
X_test_mice_imputed = mice_imputer.transform(X_test_mice)

# Reconstruction en DataFrame
X_train_mice = pd.DataFrame(X_train_mice_imputed, columns=X_train.columns, index=X_train.index)
X_test_mice = pd.DataFrame(X_test_mice_imputed, columns=X_test.columns, index=X_test.index)

# Vérification
print("\n>>> Imputation MICE terminée.")
print("NaN TRAIN :", X_train_mice.isna().sum())
print("NaN TEST  :", X_test_mice.isna().sum())

# ============================================================
# Sauvegarde MICE
# ============================================================

X_train_mice.to_csv("PROJET_FINAL/X_train_MICE.csv", index=False)
X_test_mice.to_csv("PROJET_FINAL/X_test_MICE.csv", index=False)

print("\n>>> Imputation MICE sauvegardée.")

# ===============================================================
#       EXPORT DES CIBLES (y_train / y_test)
# ===============================================================

y_train.to_csv("PROJET_FINAL/y_train_A1.csv", index=False)
y_test.to_csv("PROJET_FINAL/y_test_A1.csv", index=False)

# Pour cohérence avec KNN et MICE (mêmes y)
y_train.to_csv("PROJET_FINAL/y_train_KNN.csv", index=False)
y_test.to_csv("PROJET_FINAL/y_test_KNN.csv", index=False)

y_train.to_csv("PROJET_FINAL/y_train_MICE.csv", index=False)
y_test.to_csv("PROJET_FINAL/y_test_MICE.csv", index=False)

print(">>> Export des fichiers y_train / y_test terminé.")
