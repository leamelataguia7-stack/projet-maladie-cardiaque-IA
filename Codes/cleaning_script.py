
#### Chargement de la base de donnée ###

import pandas as pd
df = pd.read_csv("../Données/BASE_BRUTE/FRAMINGANG.csv", sep=";")
df.head()

####Informations générales sur la base de donnée###
df.info()   

### Construction du Data Dictionary###

# Affichage de la liste des variables présentes dans le DataFrame
df.columns

## info sur les données
df.info()

#renommage des colonnes 

df = df.rename(columns={
    'male': 'sexe',
    'age': 'age',
    'education': 'niveau_etude',
    'currentSmoker': 'fumeur',
    'cigsPerDay': 'cigarettes_par_jour',
    'BPMeds': 'traitement_hypertenseur',
    'prevalentStroke': 'antecedent_avc',
    'prevalentHyp': 'hypertension',
    'diabetes': 'diabete',
    'totChol': 'cholesterol_total',
    'sysBP': 'tension_sys',
    'diaBP': 'tension_dia',
    'BMI': 'imc',
    'heartRate': 'frequence_cardiaque',
    'glucose': 'glucose',
    'TenYearCHD': 'maladie_cardiaque_10ans'
})

df.head()

categorical_vars = [
    'sexe', 'niveau_etude', 'fumeur', 'traitement_hypertenseur',
    'antecedent_avc', 'hypertension', 'diabete', 
    'maladie_cardiaque_10ans'
]

for col in categorical_vars:
    df[col] = df[col].astype('category')

df.info()

#description des variables catégorielles
df[categorical_vars].describe().T

for col in categorical_vars:
    print("\nVariable :", col)
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True) * 100)

    #valeurs manquantes
df.isnull().sum()

df. isnull(). mean() * 100

##Identification des variables quantitatives

quant_vars = [
    'age', 'cigarettes_par_jour', 'cholesterol_total', 
    'tension_sys', 'tension_dia', 'imc',
    'frequence_cardiaque', 'glucose'
]

df_quant_stats = df[quant_vars].describe().T

df_quant_stats

## Visualisation 
#Histogrammes de toutes les variables quantitatives

import matplotlib.pyplot as plt
import numpy as np

# Variables quantitatives
quant_vars = [
    'age', 'cigarettes_par_jour', 'cholesterol_total', 
    'tension_sys', 'tension_dia', 'imc',
    'frequence_cardiaque', 'glucose'
]

plt.figure(figsize=(15, 12))


for i, col in enumerate(quant_vars, 1):
    plt.subplot(3, 3, i)
    plt.hist(df[col].dropna())
    plt.title(col)
    plt.tight_layout()

plt.show()

#Boxplots

plt.figure(figsize=(15, 12))
plt.suptitle("Boxplots des variables quantitatives", fontsize=16)

for i, col in enumerate(quant_vars, 1):
    plt.subplot(3, 3, i)
    plt.boxplot(df[col].dropna())
    plt.title(col)
    plt.tight_layout()

plt.show()

#variables catégorielles

categorical_vars = [
    'sexe', 'niveau_etude', 'fumeur', 'traitement_hypertenseur',
    'antecedent_avc', 'hypertension', 'diabete', 'maladie_cardiaque_10ans'
]

plt.figure(figsize=(15, 12))
plt.suptitle("Diagrammes en barres des variables catégorielles", fontsize=16)

for i, col in enumerate(categorical_vars, 1):
    plt.subplot(4, 2, i)
    df[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.tight_layout()

plt.show()

# Matrice de corrélation

import seaborn as sns
import matplotlib.pyplot as plt

# Sélection des variables numériques
num_vars = [
    'age', 'cigarettes_par_jour', 'cholesterol_total',
    'tension_sys', 'tension_dia', 'imc',
    'frequence_cardiaque', 'glucose'
]

# Ajout de la variable cible pour la corrélation
corr_vars = num_vars + ['maladie_cardiaque_10ans']

# Matrice de corrélation
corr_matrix = df[corr_vars].corr()

# Affichage
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corrélation des variables quantitatives")
plt.show()

# corrélation avec la variable cible
corr_target = corr_matrix['maladie_cardiaque_10ans'].sort_values(ascending=False)
corr_target

# enregistrer la base 
df.to_csv("../Données/BASE_BRUTE/framingham_clean_phase1.csv", index=False)
