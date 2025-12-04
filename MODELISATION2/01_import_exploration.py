import pandas as pd

# -------------------------------------------------------------
# 1. Chargement de la base Framingham
# -------------------------------------------------------------

fichier = "FRAMINGAMG.csv"

# Charger la base
df = pd.read_csv(fichier)

print("\n--- Aperçu des 5 premières lignes ---")
print(df.head())

print("\n--- Dimensions de la base (lignes, colonnes) ---")
print(df.shape)

print("\n--- Types de variables ---")
print(df.dtypes)

print("\n--- Statistiques descriptives des variables numériques ---")
print(df.describe())

print("\n--- Pourcentage de valeurs manquantes ---")
na_pct = df.isna().mean().sort_values(ascending=False) * 100
print(na_pct)
