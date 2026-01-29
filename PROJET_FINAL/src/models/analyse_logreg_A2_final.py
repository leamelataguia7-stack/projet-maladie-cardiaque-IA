"""
Analyse complète du modèle final LogReg A2 (sans SMOTE, avec scaling) :

1) Chargement des données finales (A2 / scaled)
2) Entraînement de la régression logistique sur TOUT le train
3) Évaluation sur le test :
   - Métriques au seuil 0.50 (classique)
   - Métriques au seuil 0.15 (seuil F1-opt)
   - AUC-ROC, AUC-PR, Brier score
4) Tracé des courbes :
   - ROC
   - Precision-Recall
   - Calibration (reliability diagram)
5) Tableau de régression + SHAP :
   - Refit via statsmodels.Logit pour obtenir beta, OR, IC95%, p-values
   - SHAP LinearExplainer pour importance moyenne absolue
   - Tri des variables par importance SHAP
   - Export du tableau en CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
    classification_report,
)
from sklearn.calibration import calibration_curve

import statsmodels.api as sm
import shap

# -------------------------------------------------------------------
# 0. Paramètres généraux
# -------------------------------------------------------------------

RANDOM_STATE = 42
THRESHOLD_FINAL = 0.15   # seuil F1-optimal conservé

ROOT_DIR = os.getcwd()
print("Répertoire de travail :", ROOT_DIR)

DATA_DIR = os.path.join("PROJET_FINAL", "data", "processed", "datasets_final")
OUTPUT_FIG_DIR = os.path.join("PROJET_FINAL", "outputs", "figures")
OUTPUT_TAB_DIR = os.path.join("PROJET_FINAL", "outputs", "tables")

os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_TAB_DIR, exist_ok=True)

# chemins
X_train_path = os.path.join(DATA_DIR, "X_train_A2.csv")
y_train_path = os.path.join(DATA_DIR, "y_train.csv")
X_test_path  = os.path.join(DATA_DIR, "X_test_scaled.csv")
y_test_path  = os.path.join(DATA_DIR, "y_test.csv")

print("\n--- Chemins des fichiers ---")
print("X_train_path :", X_train_path)
print("y_train_path :", y_train_path)
print("X_test_path  :", X_test_path)
print("y_test_path  :", y_test_path)

# -------------------------------------------------------------------
# 1. Chargement des données
# -------------------------------------------------------------------

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).iloc[:, 0]
X_test  = pd.read_csv(X_test_path)
y_test  = pd.read_csv(y_test_path).iloc[:, 0]

feature_names = X_train.columns.tolist()

print("\n--- Dimensions ---")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)

print("\nRépartition de la cible (train) :")
print(y_train.value_counts(normalize=True))

print("\nRépartition de la cible (test) :")
print(y_test.value_counts(normalize=True))

# -------------------------------------------------------------------
# 2. Entraînement du modèle final LogReg A2 sur TOUT le train
# -------------------------------------------------------------------

logreg = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE
)

logreg.fit(X_train, y_train)

print("\n--- Modèle final entraîné : LogReg A2 sur TOUT le train ---")

# Probabilités sur le test (classe 1 = événement)
y_proba_test = logreg.predict_proba(X_test)[:, 1]

# -------------------------------------------------------------------
# 3. Évaluation : seuil 0.50 vs seuil 0.15
# -------------------------------------------------------------------

def eval_at_threshold(y_true, y_proba, threshold):
    """
    Calcule les métriques principales pour un seuil donné.
    """
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    return acc, rec, prec, f1, y_pred

print("\n--- Évaluation au seuil 0.50 (classique) ---")
acc_05, rec_05, prec_05, f1_05, y_pred_05 = eval_at_threshold(y_test, y_proba_test, 0.50)
print(f"Accuracy : {acc_05:.3f}")
print(f"Recall   : {rec_05:.3f}")
print(f"Precision: {prec_05:.3f}")
print(f"F1       : {f1_05:.3f}")

print("\n--- Évaluation au seuil final 0.15 (F1-opt) ---")
acc_015, rec_015, prec_015, f1_015, y_pred_015 = eval_at_threshold(y_test, y_proba_test, THRESHOLD_FINAL)
print(f"Accuracy : {acc_015:.3f}")
print(f"Recall   : {rec_015:.3f}")
print(f"Precision: {prec_015:.3f}")
print(f"F1       : {f1_015:.3f}")

# AUC-ROC, AUC-PR
roc_auc = roc_auc_score(y_test, y_proba_test)
pr_auc  = average_precision_score(y_test, y_proba_test)  # AUC-PR

print("\n--- Discrimination globale ---")
print(f"AUC-ROC : {roc_auc:.3f}")
print(f"AUC-PR  : {pr_auc:.3f}")

# Brier score (calibration globale)
brier = brier_score_loss(y_test, y_proba_test)
print("\n--- Calibration globale ---")
print(f"Brier score : {brier:.4f}")

print("\n--- Classification report au seuil 0.15 ---")
print(classification_report(y_test, y_pred_015, digits=3))

# -------------------------------------------------------------------
# 4. Courbes ROC, PR, calibration
# -------------------------------------------------------------------

# 4.1 ROC
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_test)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Hasard")
plt.xlabel("1 - Spécificité (FPR)")
plt.ylabel("Sensibilité (TPR)")
plt.title("Courbe ROC - LogReg A2 (test)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
roc_path = os.path.join(OUTPUT_FIG_DIR, "logreg_A2_ROC_test.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure ROC sauvegardée : {roc_path}")

# 4.2 Precision-Recall
prec, rec, thresholds_pr = precision_recall_curve(y_test, y_proba_test)

plt.figure()
plt.plot(rec, prec, label=f"PR (AUC = {pr_auc:.3f})")
# ligne horizontale = prévalence (baseline)
prevalence = y_test.mean()
plt.hlines(prevalence, 0, 1, colors="gray", linestyles="--", label=f"Prévalence = {prevalence:.3f}")
plt.xlabel("Recall (sensibilité)")
plt.ylabel("Precision")
plt.title("Courbe Precision-Recall - LogReg A2 (test)")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
pr_path = os.path.join(OUTPUT_FIG_DIR, "logreg_A2_PR_test.png")
plt.savefig(pr_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure PR sauvegardée : {pr_path}")

# 4.3 Calibration (reliability diagram)
prob_true, prob_pred = calibration_curve(y_test, y_proba_test, n_bins=10, strategy="uniform")

plt.figure()
plt.plot(prob_pred, prob_true, "s-", label="LogReg A2 (test)")
plt.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")
plt.xlabel("Probabilité prédite")
plt.ylabel("Fréquence observée (classe 1)")
plt.title(f"Courbe de calibration - LogReg A2 (Brier = {brier:.3f})")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
calib_path = os.path.join(OUTPUT_FIG_DIR, "logreg_A2_calibration_test.png")
plt.savefig(calib_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure calibration sauvegardée : {calib_path}")

# -------------------------------------------------------------------
# 5. Tableau de régression + SHAP
# -------------------------------------------------------------------

print("\n--- Tableau de régression (statsmodels) + SHAP ---")

# 5.1 Refit en statsmodels pour obtenir beta, OR, IC95%, p-values
# On utilise X_train (déjà standardisé) + constante
X_sm = sm.add_constant(X_train, has_constant="add")
logit_model = sm.Logit(y_train, X_sm)
result = logit_model.fit(disp=False)

params = result.params          # coef (const + features)
conf_int = result.conf_int()    # IC 95% sur les coef
pvalues = result.pvalues

# On extrait seulement les variables explicatives (pas la constante)
coef = params[1:]               # enlève 'const'
conf = conf_int.iloc[1:, :]     # enlève 'const'
pvals = pvalues[1:]             # enlève 'const'

# OR et IC95% OR
or_values = np.exp(coef)
or_ci_low = np.exp(conf[0])
or_ci_high = np.exp(conf[1])

# 5.2 SHAP LinearExplainer sur le modèle sklearn
# On explique la probabilité de la classe 1
explainer = shap.LinearExplainer(logreg, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_train)  # shape (n_samples, n_features)

# Importance = moyenne absolue des SHAP values par variable
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# 5.3 Construction du DataFrame final
df_reg = pd.DataFrame({
    "variable": feature_names,
    "beta": coef.values,
    "OR": or_values.values,
    "OR_95CI_low": or_ci_low.values,
    "OR_95CI_high": or_ci_high.values,
    "p_value": pvals.values,
    "mean_abs_SHAP": mean_abs_shap,
})

# Tri des variables par importance SHAP décroissante
df_reg_sorted = df_reg.sort_values(by="mean_abs_SHAP", ascending=False).reset_index(drop=True)

# Arrondi pour affichage
df_reg_sorted_round = df_reg_sorted.copy()
cols_round = ["beta", "OR", "OR_95CI_low", "OR_95CI_high", "p_value", "mean_abs_SHAP"]
df_reg_sorted_round[cols_round] = df_reg_sorted_round[cols_round].round(4)

print("\n--- Aperçu du tableau de régression trié par importance SHAP ---")
print(df_reg_sorted_round)

# Sauvegarde en CSV
table_path = os.path.join(OUTPUT_TAB_DIR, "logreg_A2_regression_SHAP.csv")
df_reg_sorted_round.to_csv(table_path, index=False)
print(f"\nTableau exporté en CSV : {table_path}")

print("\nAnalyse du modèle final LogReg A2 terminée.")
