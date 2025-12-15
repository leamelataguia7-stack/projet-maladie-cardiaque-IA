##### Importer les bibliothèques nécessaires #####
# --- MANIPULATION DES DONNÉES ---
import numpy as np
import pandas as pd

# --- VISUALISATION ---
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# --- IGNORER LES WARNING ENNUYEUX ---
import warnings
warnings.filterwarnings("ignore")

##### Charger le jeu de données Framingham #####
# Chargement du dataset
df = pd.read_csv("PROJET_FINAL/FRAMINGANG.csv", sep=";")
# Inspection initiale
print(df.head())
print(df.info())
print(df.describe())

##### ANALYSE DES DISTRIBUTIONS : SKEWNESS (La skewness mesure l’asymétrie d’une distribution),
#  KURTOSIS (La kurtosis mesure la "pointedness" (= épaisseur des queues),
#  MULTIMODALITÉ (Certaines variables peuvent avoir plusieurs pics → mélange de populations)
# ============================================================
#           ANALYSE DES DISTRIBUTIONS : SKEW & KURTOSIS
# ============================================================

print("\n===== SKEWNESS =====")
print(df.skew(numeric_only=True))

print("\n===== KURTOSIS =====")
print(df.kurtosis(numeric_only=True))

# ============================================================
#           VISUALISATION DES DISTRIBUTIONS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")

for col in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.show()

# ============================================================
#                HEATMAP DE CORRÉLATIONS
# ============================================================

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Heatmap des corrélations")
plt.show()

# ============================================================
#            ANALYSE DES VALEURS MANQUANTES (MISSINGNO)
# ============================================================

import missingno as msno

# --- Diagramme global des valeurs présentes vs manquantes ---
plt.figure(figsize=(10, 4))
msno.bar(df)
plt.title("Taux de valeurs manquantes par variable")
plt.show()

# --- Matrice Missingno ---
plt.figure(figsize=(10, 4))
msno.matrix(df)
plt.title("Matrice des valeurs manquantes")
plt.show()

# --- Heatmap des corrélations de valeurs manquantes ---
plt.figure(figsize=(10, 6))
msno.heatmap(df)
plt.title("Corrélations des patterns de valeurs manquantes")
plt.show()

# --- Dendrogramme (clustering des patterns de manquants) ---
plt.figure(figsize=(10, 6))
msno.dendrogram(df)
plt.title("Clustering des valeurs manquantes")
plt.show()
