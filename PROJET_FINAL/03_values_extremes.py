###### Analyses des valeurs extrêmes ######

# ============================================================
#           BOXPLOTS MULTIPLES — TITRES SANS DOUBLONS
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger la base
df = pd.read_csv("PROJET_FINAL/FRAMINGANG.csv", sep=";")

# Dictionnaire de renommage
rename_dict = {
    "male": "sex",
    "age": "age",
    "education": "education",
    "currentSmoker": "smoker",
    "cigsPerDay": "cigarettes_per_day",
    "BPMeds": "bp_meds",
    "prevalentStroke": "prev_stroke",
    "prevalentHyp": "prev_hypertension",
    "diabetes": "diabetes",
    "totChol": "cholesterol",
    "sysBP": "systolic_bp",
    "diaBP": "diastolic_bp",
    "BMI": "bmi",
    "heartRate": "heart_rate",
    "glucose": "glucose",
    "TenYearCHD": "chd_risk"
}

df.rename(columns=rename_dict, inplace=True)

variables_continues = [
    "age", "cigarettes_per_day", "cholesterol",
    "systolic_bp", "diastolic_bp", "bmi",
    "heart_rate", "glucose"
]

plt.figure(figsize=(16, 12))
plt.suptitle("Boxplots des variables continues", fontsize=18)

for i, col in enumerate(variables_continues, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=df[col], color="skyblue")
    
    # Titre propre, une seule fois
    titre = col.replace("_", " ").capitalize()
    plt.title(titre, fontsize=14)
    
    # Supprimer le label en bas pour éviter les doublons
    plt.xlabel("")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ============================================================
#           CALCUL IQR + SEUILS + NOMBRE D'OUTLIERS
# ============================================================

import pandas as pd

outliers_info = []

for col in variables_continues:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    pct_outliers = len(outliers) / len(df) * 100

    outliers_info.append([
        col, Q1, Q3, IQR, lower, upper,
        len(outliers), round(pct_outliers, 2)
    ])

outliers_table = pd.DataFrame(outliers_info, columns=[
    "Variable", "Q1", "Q3", "IQR", "Lower Bound",
    "Upper Bound", "Nb Outliers", "Pct Outliers (%)"
])

print(outliers_table)

# ============================================================
#        TESTER SI LES OUTLIERS SONT EXPLIQUÉS CLINIQUEMENT
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

df = pd.read_csv("PROJET_FINAL/FRAMINGANG.csv", sep=";")

# Renommage pour cohérence
df.rename(columns={
    "cigsPerDay": "cigarettes_per_day",
    "totChol": "cholesterol",
    "sysBP": "systolic_bp",
    "diaBP": "diastolic_bp",
    "BMI": "bmi",
    "heartRate": "heart_rate"
}, inplace=True)

variables = ["cholesterol", "systolic_bp", "diastolic_bp", "bmi", "heart_rate", "glucose"]

# Fonction pour détecter les outliers IQR
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

# Analyse
for var in variables:
    df[var + "_outlier"] = detect_outliers(df[var])
    out_count = df[var + "_outlier"].sum()
    print(f"\n==== {var.upper()} ====")
    print(f"Outliers détectés : {out_count}")

    if var == "glucose":
        # Test du lien glucose ↔ diabetes
        table = pd.crosstab(df["glucose_outlier"], df["diabetes"])
        chi2, p, _, _ = chi2_contingency(table)
        print("Lien avec diabetes → p-value =", p)

    if var == "systolic_bp":
        table = pd.crosstab(df["systolic_bp_outlier"], df["prevalentHyp"])
        chi2, p, _, _ = chi2_contingency(table)
        print("Lien avec hypertension → p =", p)

    if var == "bmi":
        # test lien BMI ↔ sex
        groups = df.groupby("male")["bmi"]
        t, p = ttest_ind(groups.get_group(0), groups.get_group(1), nan_policy="omit")
        print("Différence par sexe → p =", p)
