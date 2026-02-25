# logreg_B2_interpretation.py
# Objectif : analyser en détail la régression logistique finale (LogReg_B2)
# - Chargement de X_train_B2 / y_train_B2
# - Ajustement d'un modèle logistique avec statsmodels
# - Extraction des coefficients, OR, IC95% et p-values
# - Interprétation possible : quels facteurs augmentent le risque ?

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

print("=== Analyse détaillée de la régression logistique LogReg_B2 ===")

# 1) Localisation des données -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "processed" / "datasets_final"

print("Dossier PROJET_FINAL   :", BASE_DIR)
print("Dossier datasets_final :", DATASET_DIR)

# 2) Chargement des données B2 (SMOTE + standardisation) ----------------------
X_train_B2 = pd.read_csv(DATASET_DIR / "X_train_B2.csv")
y_train_B2 = pd.read_csv(DATASET_DIR / "y_train_B2.csv").iloc[:, 0]
y_train_B2.name = "risque_chd_10ans"

print("\n--- Shapes ---")
print("X_train_B2 :", X_train_B2.shape)
print("y_train_B2 :", y_train_B2.shape)

print("\nAperçu des 5 premières lignes de X_train_B2 :")
print(X_train_B2.head())

print("\nRépartition de la cible (y_train_B2) :")
print(y_train_B2.value_counts(normalize=True))

# 3) Ajustement du modèle logistique avec statsmodels -------------------------
# On ajoute une constante (intercept)
X_sm = sm.add_constant(X_train_B2)

print("\nVariables utilisées dans le modèle :")
print(X_sm.columns)

# On ajuste un modèle Logit (régression logistique binaire)
logit_model = sm.Logit(y_train_B2, X_sm)
result = logit_model.fit(disp=0)  # disp=0 pour éviter le gros output

print("\n=== Résumé statsmodels (Logit) ===")
print(result.summary())

# 4) Extraction des coefficients, OR, IC95% et p-values -----------------------
params = result.params           # coefficients β
conf = result.conf_int()         # IC95% sur β
pvalues = result.pvalues         # p-values

# On renomme les colonnes de l'IC
conf.columns = ["beta_IC_inf", "beta_IC_sup"]

# On calcule les odds ratios et leurs IC
or_values = np.exp(params)
or_ic_inf = np.exp(conf["beta_IC_inf"])
or_ic_sup = np.exp(conf["beta_IC_sup"])

# Construction d'un tableau synthétique
summary_df = pd.DataFrame({
    "variable": params.index,
    "beta": params.values,
    "OR": or_values.values,
    "OR_IC_inf": or_ic_inf.values,
    "OR_IC_sup": or_ic_sup.values,
    "p_value": pvalues.values
})

# On trie les variables (par exemple par ordre alphabétique ou par |beta|)
summary_df = summary_df.sort_values(by="variable").reset_index(drop=True)

print("\n=== Tableau synthétique des coefficients (LogReg_B2) ===")
print(summary_df)

# 5) Petite mise en forme : on affiche sans la constante séparément ----------
print("\n=== Coefficients (sans la constante) triés par OR décroissant ===")
summary_no_const = summary_df[summary_df["variable"] != "const"].copy()
summary_no_const = summary_no_const.sort_values(by="OR", ascending=False)
print(summary_no_const)

# 6) Quelques rappels d'interprétation ---------------------------------------
print("\nRappel d'interprétation :")
print("- Chaque coefficient (beta) est estimé sur des variables standardisées (moyenne 0, écart-type 1).")
print("- L'odds ratio (OR) correspond donc à la variation du rapport de cotes")
print("  pour une augmentation d'un écart-type de la variable correspondante.")
print("- OR > 1 : la variable est associée à une augmentation du risque de maladie cardiaque à 10 ans.")
print("- OR < 1 : la variable est associée à une diminution du risque.")
print("- Les IC95% [OR_IC_inf ; OR_IC_sup] et les p-values permettent d'apprécier")
print("  la précision et la significativité statistique des associations.")
print("\n=== Fin de l'analyse de LogReg_B2 ===")

