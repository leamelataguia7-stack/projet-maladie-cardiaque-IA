########## ANALYSE DES VALEURS MANQUANTES ##########

# --- Import des librairies ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings

warnings.filterwarnings("ignore")

# --- Charger la base ---
df = pd.read_csv("PROJET_FINAL/FRAMINGANG.csv", sep=";")

print("\n===== Aperçu du dataset =====")
print(df.head())

print("\n===== Pourcentages de valeurs manquantes =====")
print(df.isna().mean().sort_values(ascending=False) * 100)

# --- Visualisation Missingno ---

# Barplot des valeurs manquantes
plt.figure(figsize=(6,4))
msno.bar(df)
plt.title("Taux de valeurs manquantes par variable")
plt.show()

# Matrice Missingno
plt.figure(figsize=(6,4))
msno.matrix(df)
plt.title("Matrice des valeurs manquantes")
plt.show()

# Heatmap Missingno (corrélations des patterns de manquants)
plt.figure(figsize=(6,6))
msno.heatmap(df)
plt.title("Heatmap des patterns de valeurs manquantes")
plt.show()

# Dendrogramme Missingno
plt.figure(figsize=(6,6))
msno.dendrogram(df)
plt.title("Dendrogramme des valeurs manquantes")
plt.show()
