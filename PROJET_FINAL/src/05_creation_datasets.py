# 05_creation_datasets.py
# Construction des jeux de données finaux pour la modélisation
# Étape actuelle : créer le dataset A1 (MICE brut)

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("=== Début du script 05_creation_datasets.py ===")

# 1) Définir les dossiers de travail -----------------------------------------

# Dossier PROJET_FINAL (on part du fichier actuel)
BASE_DIR = Path(__file__).resolve().parents[1]

# Dossier où se trouvent les fichiers imputés MICE
IMPUT_DIR = BASE_DIR / "data" / "processed" / "imputations"

# Dossier où on va enregistrer les jeux finaux
FINAL_DIR = BASE_DIR / "data" / "processed" / "datasets_final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

print("Dossier PROJET_FINAL :", BASE_DIR)
print("Dossier imputations  :", IMPUT_DIR)
print("Dossier final        :", FINAL_DIR)

# 2) Charger les fichiers MICE -----------------------------------------------

# les fichiers sont dans data/processed/imputations :
# - X_train_MICE.csv
# - X_test_MICE.csv
# - y_train_MICE.csv
# - y_test_MICE.csv

X_train = pd.read_csv(IMPUT_DIR / "X_train_MICE.csv")
X_test  = pd.read_csv(IMPUT_DIR / "X_test_MICE.csv")

# Pour y_train et y_test, on prend la première colonne
y_train = pd.read_csv(IMPUT_DIR / "y_train_MICE.csv").iloc[:, 0]
y_test  = pd.read_csv(IMPUT_DIR / "y_test_MICE.csv").iloc[:, 0]

# On donne un nom explicite à la colonne cible
y_train.name = "risque_chd_10ans"
y_test.name = "risque_chd_10ans"

print("\n--- Shapes des fichiers MICE ---")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

print("\nRépartition de la cible dans y_train :")
print(y_train.value_counts(normalize=True))

# 3) Créer le dataset A1 -----------------------------------------------------

# A1 = les données MICE telles quelles (brutes),
# que l'on fige comme base de référence pour les modèles.

X_train_A1 = X_train.copy()
X_test_raw = X_test.copy()

# Sauvegarde des fichiers A1 dans datasets_final
X_train_A1.to_csv(FINAL_DIR / "X_train_A1.csv", index=False)
X_test_raw.to_csv(FINAL_DIR / "X_test_raw.csv", index=False)
y_train.to_csv(FINAL_DIR / "y_train.csv", index=False)
y_test.to_csv(FINAL_DIR / "y_test.csv", index=False)

print("\n Dataset A1 créé et sauvegardé dans :", FINAL_DIR)
print("   - X_train_A1 :", X_train_A1.shape)
print("   - X_test_raw :", X_test_raw.shape)
print("   - y_train    :", y_train.shape)
print("   - y_test     :", y_test.shape)
print("\n=== Fin du script 05_creation_datasets.py (A1 uniquement) ===")

# 4) Créer le dataset A2 : standardisation des variables ---------------------

print("\n=== Création du dataset A2 (standardisation) ===")

# On part de X_train_A1 et X_test_raw créés juste avant
# StandardScaler va transformer chaque variable :
#   - moyenne ~ 0
#   - écart-type ~ 1

scaler = StandardScaler()

# On ajuste le scaler sur le TRAIN uniquement
X_train_A2_array = scaler.fit_transform(X_train_A1)

# Puis on applique la même transformation au TEST
X_test_scaled_array = scaler.transform(X_test_raw)

# On remet les résultats dans des DataFrame avec les mêmes colonnes
X_train_A2 = pd.DataFrame(X_train_A2_array, columns=X_train_A1.columns)
X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_test_raw.columns)

# Sauvegarde dans datasets_final
X_train_A2.to_csv(FINAL_DIR / "X_train_A2.csv", index=False)
X_test_scaled.to_csv(FINAL_DIR / "X_test_scaled.csv", index=False)

print(" Dataset A2 créé et sauvegardé dans :", FINAL_DIR)
print("   - X_train_A2   :", X_train_A2.shape)
print("   - X_test_scaled:", X_test_scaled.shape)

# Petite vérification : moyenne et écart-type sur quelques colonnes
print("\nVérification rapide sur les 3 premières colonnes de X_train_A2 :")
print("Moyennes :", X_train_A2.iloc[:, :3].mean())
print("Écarts-types :", X_train_A2.iloc[:, :3].std())

# 5) Création du dataset B1 : SMOTE (rééquilibrage des classes) --------------

print("\n=== Création du dataset B1 (SMOTE) ===")

# On part du dataset A1 (données brutes)
print("Répartition avant SMOTE :")
print(y_train.value_counts(normalize=True))

# On crée l'objet SMOTE
smote = SMOTE(random_state=42)

# On applique SMOTE sur le TRAIN uniquement
X_train_B1_array, y_train_B1 = smote.fit_resample(X_train_A1, y_train)

# On remet dans des DataFrame / Series
X_train_B1 = pd.DataFrame(X_train_B1_array, columns=X_train_A1.columns)
y_train_B1 = pd.Series(y_train_B1, name="risque_chd_10ans")

# Sauvegarde
X_train_B1.to_csv(FINAL_DIR / "X_train_B1.csv", index=False)
y_train_B1.to_csv(FINAL_DIR / "y_train_B1.csv", index=False)

print(" Dataset B1 créé et sauvegardé dans :", FINAL_DIR)
print("   - X_train_B1 :", X_train_B1.shape)
print("   - y_train_B1 :", y_train_B1.shape)

print("\nRépartition après SMOTE :")
print(y_train_B1.value_counts(normalize=True))

# 6) Création du dataset B2 : SMOTE + standardisation ------------------------

print("\n=== Création du dataset B2 (SMOTE + standardisation) ===")

# On réutilise le scaler déjà ajusté sur X_train_A1 (A2)
# On applique la même transformation à X_train_B1 et au test (X_test_raw)

X_train_B2_array = scaler.transform(X_train_B1)
X_train_B2 = pd.DataFrame(X_train_B2_array, columns=X_train_A1.columns)

# Pour la clarté, on réutilise X_test_scaled déjà créé pour A2
# donc pas besoin de recréer un nouveau fichier test

# Sauvegarde
X_train_B2.to_csv(FINAL_DIR / "X_train_B2.csv", index=False)
# y_train_B2 = y_train_B1, mais on le sauvegarde sous un autre nom pour être explicite
y_train_B1.to_csv(FINAL_DIR / "y_train_B2.csv", index=False)

print(" Dataset B2 créé et sauvegardé dans :", FINAL_DIR)
print("   - X_train_B2 :", X_train_B2.shape)
print("   - y_train_B2 :", y_train_B1.shape)
