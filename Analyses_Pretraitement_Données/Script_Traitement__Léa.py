
# Phase 2 PREPROCESSING, preparation données pôur le maching learning 
#Chargement de la base 

import pandas as pd
df = pd.read_csv("../Données/BASE_BRUTE/framingham_clean_phase1.csv")

df.head()

### suppression de la colonne Sexe
df = df.drop(columns=["Sexe"])
df.head()   

#Séparer les variables (cible, catégorielles, quantitatives)

# Variable cible
target = "maladie_cardiaque_10ans"

# Variables catégorielles (binaires)
categorical_vars = [
    "sexe",
    "niveau_etude",
    "fumeur",
    "traitement_hypertenseur",
    "antecedent_avc",
    "hypertension",
    "diabete"
]

# Variables quantitatives (continues)
quantitative_vars = [
    "age",
    "cigarettes_par_jour",
    "cholesterol_total",
    "tension_sys",
    "tension_dia",
    "imc",
    "frequence_cardiaque",
    "glucose"
]

print("Variable cible :", target)
print("\nVariables catégorielles :", categorical_vars)
print("\nVariables quantitatives :", quantitative_vars)
print("\nNombre total :", len(categorical_vars) + len(quantitative_vars) + 1)

#Gestion des valeurs manquantes
#methodes d'amputation  (médiane + mode) pour les quantitatives et catégorielles respectivement

from sklearn.impute import SimpleImputer
import pandas as pd

# Création des imputers
imputer_quant = SimpleImputer(strategy="median")
imputer_cat   = SimpleImputer(strategy="most_frequent")

# Imputation des quantitatives
df[quantitative_vars] = imputer_quant.fit_transform(df[quantitative_vars])

# Imputation des catégorielles
df[categorical_vars] = imputer_cat.fit_transform(df[categorical_vars])

# Vérification après imputation
df.isnull().sum()

# Suppression de la ligne avec cible manquante (1 seule ligne)
df = df[df["maladie_cardiaque_10ans"].notna()]

# Suppression de la colonne temp
df = df.drop(columns=["temp"])

# Vérification finale
df.isnull().sum()

#Détection et traitement des Outliers/ valeurs extrèmes 
quantitative_vars = [
    "age",
    "cigarettes_par_jour",
    "cholesterol_total",
    "tension_sys",
    "tension_dia",
    "imc",
    "frequence_cardiaque",
    "glucose"
]
# detection outlers pour chaque variable quantitative

import numpy as np

# Fonction pour détecter les outliers avec IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers

# Détection des outliers pour chaque variable quantitative
for col in quantitative_vars:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col} : {len(outliers)} outliers détectés")

    #gestion des outliers par winsorisation

import numpy as np

def winsorize_iqr(data, column):
    """
    Applique une winsorisation basée sur l'IQR à une colonne numérique.
    Les valeurs en-dessous de Q1 - 1.5*IQR sont ramenées à la borne basse,
    celles au-dessus de Q3 + 1.5*IQR à la borne haute.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    data[column] = np.where(
        data[column] < lower, lower,
        np.where(data[column] > upper, upper, data[column])
    )
    return data

# Application de la winsorisation à toutes les variables quantitatives
for col in quantitative_vars:
    df = winsorize_iqr(df, col)

# Vérification : recomptage des outliers
for col in quantitative_vars:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col} : {len(outliers)} outliers après winsorisation")

    #Split Train/Test

# -------------------------------
#  Séparation Train / Test propre
# -------------------------------

from sklearn.model_selection import train_test_split

# 1. Définir X et y
X = df.drop(columns=["maladie_cardiaque_10ans"])   # variables explicatives
y = df["maladie_cardiaque_10ans"]                  # variable cible

# 2. Split train/test (80% - 20%) avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT : conserve le même ratio 0/1 dans train et test
)

# 3. Vérification des dimensions
print("Dimensions :")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# 4. Vérification du déséquilibre dans les deux échantillons
print("\nRépartition des classes dans TRAIN :")
print(y_train.value_counts(normalize=True)*100)

print("\nRépartition des classes dans TEST :")
print(y_test.value_counts(normalize=True)*100)

# Encodage des variables catégorielles
#Identification  des variables catégorielles

categorical_vars = ["sexe", "niveau_etude", "fumeur",
                    "traitement_hypertenseur", "antecedent_avc",
                    "hypertension", "diabete"]

quantitative_vars = ["age", "cigarettes_par_jour", "cholesterol_total",
                     "tension_sys", "tension_dia", "imc",
                     "frequence_cardiaque", "glucose"]

#One-Hot Encoding sur X_train (Transforme une variable en colonnes binaires )

import pandas as pd

# Encodage One-Hot sur TRAIN uniquement
X_train_encoded = pd.get_dummies(X_train, columns=categorical_vars, drop_first=True)

# Encodage One-Hot sur TEST (en utilisant les mêmes colonnes que TRAIN)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_vars, drop_first=True)

# Alignement des colonnes (très important !)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1)

# Remplacer les NaN créés dans X_test (colonnes manquantes)
X_test_encoded = X_test_encoded.fillna(0)

X_train_encoded.head(), X_test_encoded.head()

#Gestion du déséquilibre de la variable cible avec SMOTE (Synthetic Minority Over-sampling Technique)

!pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

# Initialisation de SMOTE
smote = SMOTE(random_state=42)

# Application de SMOTE sur le TRAIN uniquement
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

# Vérification du nouvel équilibre des classes
print("Avant SMOTE :", y_train.value_counts(normalize=True) * 100)
print("\nAprès SMOTE :", y_train_resampled.value_counts(normalize=True) * 100)

print("\nDimensions :")
print("X_train_resampled :", X_train_resampled.shape)
print("y_train_resampled :", y_train_resampled.shape)

#Standardisation (Scaling) pour les variables quantitatives et la stabilisation des modèles de machine learning et comparables 

from sklearn.preprocessing import StandardScaler

# Initialisation du scaler
scaler = StandardScaler()

# Fit uniquement sur le TRAIN RESAMPLED (apprentissage des moyennes et écarts-types)
scaler.fit(X_train_resampled[quantitative_vars])

# Transformation du TRAIN resampled
X_train_resampled[quantitative_vars] = scaler.transform(X_train_resampled[quantitative_vars])

# Transformation du TEST (sans fit)
X_test_encoded[quantitative_vars] = scaler.transform(X_test_encoded[quantitative_vars])

# Vérification
X_train_resampled[quantitative_vars].describe().T

#Standardisation (Scaling) pour les variables quantitatives et la stabilisation des modèles de machine learning et comparables 

from sklearn.preprocessing import StandardScaler

# Initialisation du scaler
scaler = StandardScaler()

# Fit uniquement sur le TRAIN RESAMPLED (apprentissage des moyennes et écarts-types)
scaler.fit(X_train_resampled[quantitative_vars])

# Transformation du TRAIN resampled
X_train_resampled[quantitative_vars] = scaler.transform(X_train_resampled[quantitative_vars])

# Transformation du TEST (sans fit)
X_test_encoded[quantitative_vars] = scaler.transform(X_test_encoded[quantitative_vars])

# Vérification
X_train_resampled[quantitative_vars].describe().T

# Enregistrement des DF 
df.to_csv("../Données/BASE_INTERMEDIAIRE/df_preprocessed.csv", index=False)
import pandas as pd

pd.DataFrame(X_train_final).to_csv("../Données/BASE_MODELISATION/X_train_final.csv", index=False)
pd.DataFrame(y_train_final).to_csv("../Données/BASE_MODELISATION/y_train_final.csv", index=False)

pd.DataFrame(X_test_final).to_csv("../Données/BASE_MODELISATION/X_test_final.csv", index=False)
pd.DataFrame(y_test_final).to_csv("../Données/BASE_MODELISATION/y_test_final.csv", index=False)