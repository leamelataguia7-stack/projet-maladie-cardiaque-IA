#Modèle 1 : Régression logistique – Version A1
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,  # AUC-PR
    confusion_matrix,
    classification_report,
    brier_score_loss

    #Charger les bases A1 (train/test bruts)
base_path = "../BASES_FINALES_B"  # adapte si besoin

X_train_A1 = pd.read_csv(f"{base_path}/X_train_A1_raw.csv")
y_train_A  = pd.read_csv(f"{base_path}/y_train_A_raw.csv")

X_test_raw = pd.read_csv(f"{base_path}/X_test_raw.csv")
y_test     = pd.read_csv(f"{base_path}/y_test.csv")

X_train_A1.shape, y_train_A.shape, X_test_raw.shape, y_test.shape

# applatement des variables cibles aux datasets d'entrainement
y_train_A = y_train_A.squeeze()
y_test    = y_test.squeeze()

#Vérification la distribution de la cible (déséquilibre)
print("Train A1 :")
print(y_train_A.value_counts(normalize=True) * 100)

print("\nTest :")
print(y_test.value_counts(normalize=True) * 100)

#Définition fonction d’évaluation standard
def eval_classification(y_true, y_pred, y_proba, model_name=""):
    """
    y_true  : vraies étiquettes (0/1)
    y_pred  : classes prédites (0/1)
    y_proba : probas prédites pour la classe 1
    """
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr  = average_precision_score(y_true, y_proba)
    brier   = brier_score_loss(y_true, y_proba)
    cm      = confusion_matrix(y_true, y_pred)

    print(f"====== Résultats : {model_name} ======")
    print("Matrice de confusion (rows = vrai, cols = prédit) :")
    print(cm)
    print("\nAccuracy      :", round(acc, 3))
    print("Recall        :", round(rec, 3))
    print("Precision     :", round(prec, 3))
    print("F1-score      :", round(f1, 3))
    print("AUC-ROC       :", round(auc_roc, 3))
    print("AUC-PR        :", round(auc_pr, 3))
    print("Brier score   :", round(brier, 3))
    print("\nClassification report :")
    print(classification_report(y_true, y_pred, digits=3))

    # On retourne aussi les scores dans un dict pour construire plus tard un tableau de synthèse
    return {
       "model": model_name,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier": brier
    }
    #Entraîner la Régression Logistique (A1 : non SMOTE, non scaling)
logit_A1 = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",      # solide pour ce type de données
    random_state=42
)

logit_A1.fit(X_train_A1, y_train_A)

#Prédictions sur le jeu de test
y_pred_A1 = logit_A1.predict(X_test_raw)
y_proba_A1 = logit_A1.predict_proba(X_test_raw)[:, 1]

#Évaluation 
results_A1 = eval_classification(
    y_true=y_test,
    y_pred=y_pred_A1,
    y_proba=y_proba_A1,
    model_name="LOGIT_A1 (non SMOTE, non scalé)"
)
#MODÈLE 2 — LOGIT_A2 : Régression logistique – Version A2 (non SMOTE, scaling)
#Charger les bases A2 (train/test scalés)   
import pandas as pd

base_path = "../BASES_FINALES_B"  # adapte si nécessaire

X_train_A2 = pd.read_csv(f"{base_path}/X_train_A2_scaled.csv")
y_train_A  = pd.read_csv(f"{base_path}/y_train_A_raw.csv")

X_test_scaled = pd.read_csv(f"{base_path}/X_test_scaled.csv")
y_test        = pd.read_csv(f"{base_path}/y_test.csv")

# convertir cibles en vecteur
y_train_A = y_train_A.squeeze()
y_test    = y_test.squeeze()

X_train_A2.shape, X_test_scaled.shape, y_train_A.shape, y_test.shape

#Vérifier que les données sont bien scalées
X_train_A2.describe().T

#Entraîner la régression logistique LOGIT_A2
from sklearn.linear_model import LogisticRegression

logit_A2 = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    random_state=42
)

logit_A2.fit(X_train_A2, y_train_A)

#predire sur le jeu de test
y_pred_A2  = logit_A2.predict(X_test_scaled)
y_proba_A2 = logit_A2.predict_proba(X_test_scaled)[:, 1]

#Évaluation avec la fonction standard
results_A2 = eval_classification(
    y_true=y_test,
    y_pred=y_pred_A2,
    y_proba=y_proba_A2,
    model_name="LOGIT_A2 (non SMOTE, SCALÉ)"
)
#MODÈLE 3 — LOGIT_B1 : Régression logistique – Version B1 (SMOTE, non scaling)
#Charger les bases B1 (train/test bruts)
import pandas as pd

base_path = "../BASES_FINALES_B"

# Train SMOTE non scalé
X_train_B1 = pd.read_csv(f"{base_path}/X_train_B1_smote.csv")
y_train_B1 = pd.read_csv(f"{base_path}/y_train_B_smote.csv").squeeze()

# Test non scalé
X_test_raw = pd.read_csv(f"{base_path}/X_test_raw.csv")
y_test     = pd.read_csv(f"{base_path}/y_test.csv").squeeze()

X_train_B1.shape, y_train_B1.shape, X_test_raw.shape, y_test.shape

#Vérifier que SMOTE a bien équilibré la cible
y_train_B1.value_counts(normalize=True)*100

#Entraîner le modèle LOGIT_B1
from sklearn.linear_model import LogisticRegression

logit_B1 = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    random_state=42
)

logit_B1.fit(X_train_B1, y_train_B1)

#Prédictions sur le test non scalé
y_pred_B1  = logit_B1.predict(X_test_raw)
y_proba_B1 = logit_B1.predict_proba(X_test_raw)[:, 1]

#Évaluation complète
results_B1 = eval_classification(
    y_true=y_test,
    y_pred=y_pred_B1,
    y_proba=y_proba_B1,
    model_name="LOGIT_B1 (SMOTE, non scalé)"
)
#MODÈLE 4 — LOGIT_B2 : Régression logistique – Version B2 (SMOTE, scaling)
#Charger les bases B2 (train/test scalés)   
import pandas as pd

base_path = "../BASES_FINALES_B"

# Train SMOTE + SCALING
X_train_B2 = pd.read_csv(f"{base_path}/X_train_B2_smote_scaled.csv")
y_train_B2 = pd.read_csv(f"{base_path}/y_train_B_smote.csv").squeeze()

# Test scalé
X_test_scaled = pd.read_csv(f"{base_path}/X_test_scaled.csv")
y_test        = pd.read_csv(f"{base_path}/y_test.csv").squeeze()

X_train_B2.shape, y_train_B2.shape, X_test_scaled.shape, y_test.shape

#Entraîner le modèle LOGIT_B2
from sklearn.linear_model import LogisticRegression

logit_B2 = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    random_state=42
)

logit_B2.fit(X_train_B2, y_train_B2)

#Prédictions sur le test scalé
y_pred_B2  = logit_B2.predict(X_test_scaled)
y_proba_B2 = logit_B2.predict_proba(X_test_scaled)[:, 1]
#Évaluation complète    
results_B2 = eval_classification(
    y_true=y_test,
    y_pred=y_pred_B2,
    y_proba=y_proba_B2,
    model_name="LOGIT_B2 (SMOTE + SCALING)"
)

#Random forest ((4 SOUS-MODÈLES))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss

#Fonction d’évaluation standard
def eval_rf(y_true, y_pred, y_proba, model_name):
    print(f"\n====== Résultats : {model_name} ======")

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de confusion (rows = vrai, cols = prédit) :")
    print(cm)

    # Scores
    acc = (y_pred == y_true).mean()
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1]+cm[0,1])>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    aucroc = roc_auc_score(y_true, y_proba)
    aucpr = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    print(f"\nAccuracy      : {acc:.3f}")
    print(f"Recall        : {recall:.3f}")
    print(f"Precision     : {precision:.3f}")
    print(f"F1-score      : {f1:.3f}")
    print(f"AUC-ROC       : {aucroc:.3f}")
    print(f"AUC-PR        : {aucpr:.3f}")
    print(f"Brier score   : {brier:.3f}")

    print("\nClassification report :")
    print(classification_report(y_true, y_pred))

    return {
        "model": model_name,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "aucroc": aucroc,
        "aucpr": aucpr,
        "brier": brier
    }

    #Chargement des bases 
base_path = "../BASES_FINALES_B"

# A1 / A2
X_train_A_raw   = pd.read_csv(f"{base_path}/X_train_A1_raw.csv")
y_train_A       = pd.read_csv(f"{base_path}/y_train_A_raw.csv").squeeze()
X_test_raw      = pd.read_csv(f"{base_path}/X_test_raw.csv")
X_test_scaled   = pd.read_csv(f"{base_path}/X_test_scaled.csv")
y_test          = pd.read_csv(f"{base_path}/y_test.csv").squeeze()

# B1 / B2
X_train_B1      = pd.read_csv(f"{base_path}/X_train_B1_smote.csv")
X_train_B2      = pd.read_csv(f"{base_path}/X_train_B2_smote_scaled.csv")
y_train_B       = pd.read_csv(f"{base_path}/y_train_B_smote.csv").squeeze()

#4 modèles Random Forest (complets avec évaluation)

# RF_A1 : non SMOTE, non scalé
rf_A1 = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight=None,
    random_state=42
)

rf_A1.fit(X_train_A_raw, y_train_A)

y_pred = rf_A1.predict(X_test_raw)
y_proba = rf_A1.predict_proba(X_test_raw)[:,1]

results_RF_A1 = eval_rf(y_test, y_pred, y_proba, "RF_A1 (non SMOTE, non scalé)")

#RF_A2 — Sans SMOTE, SCALING 
rf_A2 = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight=None,
    random_state=42
)

rf_A2.fit(X_train_A_raw, y_train_A)

y_pred = rf_A2.predict(X_test_scaled)
y_proba = rf_A2.predict_proba(X_test_scaled)[:,1]

results_RF_A2 = eval_rf(y_test, y_pred, y_proba, "RF_A2 (non SMOTE, SCALÉ)")

#RF_B1 — SMOTE, non scalé
rf_B1 = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42
)

rf_B1.fit(X_train_B1, y_train_B)

y_pred = rf_B1.predict(X_test_raw)
y_proba = rf_B1.predict_proba(X_test_raw)[:,1]

results_RF_B1 = eval_rf(y_test, y_pred, y_proba, "RF_B1 (SMOTE, non scalé)")


#RF_B2 — SMOTE + Scaling
rf_B2 = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42
)

rf_B2.fit(X_train_B2, y_train_B)

y_pred = rf_B2.predict(X_test_scaled)
y_proba = rf_B2.predict_proba(X_test_scaled)[:,1]

results_RF_B2 = eval_rf(y_test, y_pred, y_proba, "RF_B2 (SMOTE + SCALING)")


# modeles XGBoost Classifier (4 sous-modèles)
!pip install xgboost

# importer les bibliothèques nécessaires
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss
)

#fonction d’évaluation standard
def eval_xgb(y_true, y_pred, y_proba, model_name):
    print(f"\n====== Résultats : {model_name} ======")

    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de confusion (rows = vrai, cols = prédit) :")
    print(cm)

    acc = (y_pred == y_true).mean()
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1]+cm[0,1])>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    aucroc = roc_auc_score(y_true, y_proba)
    aucpr = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    print(f"\nAccuracy      : {acc:.3f}")
    print(f"Recall        : {recall:.3f}")
    print(f"Precision     : {precision:.3f}")
    print(f"F1-score      : {f1:.3f}")
    print(f"AUC-ROC       : {aucroc:.3f}")
    print(f"AUC-PR        : {aucpr:.3f}")
    print(f"Brier score   : {brier:.3f}")

    print("\nClassification report :")
    print(classification_report(y_true, y_pred))

    return {
        "model": model_na
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "aucroc": aucroc,
        "aucpr": aucpr,
        "brier": brier
    }

    #charger les données
base_path = "../BASES_FINALES_B"

# A1 / A2
X_train_A_raw   = pd.read_csv(f"{base_path}/X_train_A1_raw.csv")
y_train_A       = pd.read_csv(f"{base_path}/y_train_A_raw.csv").squeeze()
X_test_raw      = pd.read_csv(f"{base_path}/X_test_raw.csv")
X_test_scaled   = pd.read_csv(f"{base_path}/X_test_scaled.csv")
y_test          = pd.read_csv(f"{base_path}/y_test.csv").squeeze()

# B1 / B2
X_train_B1      = pd.read_csv(f"{base_path}/X_train_B1_smote.csv")
X_train_B2      = pd.read_csv(f"{base_path}/X_train_B2_smote_scaled.csv")
y_train_B       = pd.read_csv(f"{base_path}/y_train_B_smote.csv").squeeze()

#PARAMÈTRES XGBOOST (standards, adaptables)
xgb_params = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}

#MODÈLE 1 — XGB_A1 (Non SMOTE, Non Scalé)
xgb_A1 = xgb.XGBClassifier(**xgb_params)
xgb_A1.fit(X_train_A_raw, y_train_A)

y_pred = xgb_A1.predict(X_test_raw)
y_proba = xgb_A1.predict_proba(X_test_raw)[:,1]

results_XGB_A1 = eval_xgb(y_test, y_pred, y_proba, "XGB_A1 (non SMOTE, non scalé)")

#MODÈLE 2 — XGB_A2 (Non SMOTE, SCALÉ)
xgb_A2 = xgb.XGBClassifier(**xgb_params)
xgb_A2.fit(X_train_A_raw, y_train_A)

y_pred = xgb_A2.predict(X_test_scaled)
y_proba = xgb_A2.predict_proba(X_test_scaled)[:,1]

results_XGB_A2 = eval_xgb(y_test, y_pred, y_proba, "XGB_A2 (non SMOTE, SCALÉ)")

#MODÈLE 3 — XGB_B1 (SMOTE, non scalé)
xgb_B1 = xgb.XGBClassifier(**xgb_params)
xgb_B1.fit(X_train_B1, y_train_B)

y_pred = xgb_B1.predict(X_test_raw)
y_proba = xgb_B1.predict_proba(X_test_raw)[:,1]

results_XGB_B1 = eval_xgb(y_test, y_pred, y_proba, "XGB_B1 (SMOTE, non scalé)")

#MODÈLE 4 — XGB_B2 (SMOTE + SCALING)
xgb_B2 = xgb.XGBClassifier(**xgb_params)
xgb_B2.fit(X_train_B2, y_train_B)

y_pred = xgb_B2.predict(X_test_scaled)
y_proba = xgb_B2.predict_proba(X_test_scaled)[:,1]

results_XGB_B2 = eval_xgb(y_test, y_pred, y_proba, "XGB_B2 (SMOTE + SCALING)")

#IMPORTANCE DES VARIABLES (à mettre dans le mémoire)
importances = pd.DataFrame({
    "variable": X_train_B1.columns,
    "importance": xgb_B2.feature_importances_
}).sort_values("importance", ascending=False)

print(importances.head(20))

# models GRADIENT BOOSTING

# importation des bibliothèques nécessaires
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss
)

#Fonction d’évaluation standard
def eval_gb(y_true, y_pred, y_proba, model_name):
    print(f"\n====== Résultats : {model_name} ======")

    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de confusion (rows = vrai, cols = prédit) :")
    print(cm)

    acc = (y_pred == y_true).mean()
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1]+cm[0,1])>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    aucroc = roc_auc_score(y_true, y_proba)
    aucpr = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    print(f"\nAccuracy      : {acc:.3f}")
    print(f"Recall        : {recall:.3f}")
    print(f"Precision     : {precision:.3f}")
    print(f"F1-score      : {f1:.3f}")
    print(f"AUC-ROC       : {aucroc:.3f}")
    print(f"AUC-PR        : {aucpr:.3f}")
    print(f"Brier score   : {brier:.3f}")

    print("\nClassification report :")
    print(classification_report(y_true, y_pred))
 return {
        "model": model_name,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "aucroc": aucroc,
        "aucpr": aucpr,
        "brier": brier
    }

    #CHARGEMENT DES DONNÉES
base_path = "../BASES_FINALES_B"

# A1 / A2
X_train_A_raw   = pd.read_csv(f"{base_path}/X_train_A1_raw.csv")
y_train_A       = pd.read_csv(f"{base_path}/y_train_A_raw.csv").squeeze()
X_test_raw      = pd.read_csv(f"{base_path}/X_test_raw.csv")
X_test_scaled   = pd.read_csv(f"{base_path}/X_test_scaled.csv")
y_test          = pd.read_csv(f"{base_path}/y_test.csv").squeeze()

# B1 / B2
X_train_B1      = pd.read_csv(f"{base_path}/X_train_B1_smote.csv")
X_train_B2      = pd.read_csv(f"{base_path}/X_train_B2_smote_scaled.csv")
y_train_B       = pd.read_csv(f"{base_path}/y_train_B_smote.csv").squeeze()

vars#MODÈLE 1 — GB_A1 (Non SMOTE, Non Scalé)
gb_A1 = GradientBoostingClassifier(**gb_params)
gb_A1.fit(X_train_A_raw, y_train_A)

y_pred = gb_A1.predict(X_test_raw)
y_proba = gb_A1.predict_proba(X_test_raw)[:,1]

results_GB_A1 = eval_gb(y_test, y_pred, y_proba, "GB_A1 (non SMOTE, non scalé)")

#MODÈLE 2 — GB_A2 (Non SMOTE, SCALÉ)
gb_A2 = GradientBoostingClassifier(**gb_params)
gb_A2.fit(X_train_A_raw, y_train_A)

y_pred = gb_A2.predict(X_test_scaled)
y_proba = gb_A2.predict_proba(X_test_scaled)[:,1]

results_GB_A2 = eval_gb(y_test, y_pred, y_proba, "GB_A2 (non SMOTE, SCALÉ)")

#MODÈLE 3 — GB_B1 (SMOTE, Non Scalé)
gb_B1 = GradientBoostingClassifier(**gb_params)
gb_B1.fit(X_train_B1, y_train_B)

y_pred = gb_B1.predict(X_test_raw)
y_proba = gb_B1.predict_proba(X_test_raw)[:,1]

results_GB_B1 = eval_gb(y_test, y_pred, y_proba, "GB_B1 (SMOTE, non scalé)")

#MODÈLE 4 — GB_B2 (SMOTE + SCALING)
gb_B2 = GradientBoostingClassifier(**gb_params)
gb_B2.fit(X_train_B2, y_train_B)

y_pred = gb_B2.predict(X_test_scaled)
y_proba = gb_B2.predict_proba(X_test_scaled)[:,1]

results_GB_B2 = eval_gb(y_test, y_pred, y_proba, "GB_B2 (SMOTE + SCALING)")

#courbes ROC et PR pour les modèles GB
#importation des bibliothèques nécessaires
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

import pandas as pd

# === TRAIN (SMOTE) ===
X_train = pd.read_csv("../BASES_FINALES_B/X_train_B1_smote.csv")
y_train = pd.read_csv("../BASES_FINALES_B/y_train_B_smote.csv")

# === TEST (RAW) ===
X_test  = pd.read_csv("../BASES_FINALES_B/X_test_raw.csv")
y_test  = pd.read_csv("../BASES_FINALES_B/y_test.csv")

# Convertir en vecteur
y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

print("Shapes :")
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#entraînement du modèle GB_B1
from sklearn.ensemble import GradientBoostingClassifier

# Meilleur modèle identifié
gb_best = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_best.fit(X_train, y_train)

#Courbe ROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Probabilités
y_proba = gb_best.predict_proba(X_test)[:, 1]

# ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f'GB_B1 (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC – Gradient Boosting (B1)")
plt.legend()
plt.grid(True)
plt.show()

# Courbe Precision–Recall
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.figure(figsize=(7,6))
plt.plot(recall, precision, label=f'GB_B1 (AP = {ap:.3f})', linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Courbe Precision–Recall – Gradient Boosting (B1)")
plt.legend()
plt.grid(True)
plt.show()

#Trouver le meilleur seuil décisionnel
from sklearn.metrics import f1_score
import numpy as np

best_f1 = 0
best_threshold = 0

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Meilleur seuil = {best_threshold:.3f}")
print(f"F1-score associé = {best_f1:.3f}")


#Modèle SVM
#Imports et fonction d’évaluation
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    classification_report
)

def eval_model(name, y_true, y_pred, y_prob):
    print(f"\n====== Résultats : {name} ======")

    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de confusion (vrai x prédit) :")
    print(cm)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    print(f"\nAccuracy      : {acc:.3f}")
    print(f"Recall        : {rec:.3f}")
    print(f"Precision     : {prec:.3f}")
    print(f"F1-score      : {f1:.3f}")
    print(f"AUC-ROC       : {auc:.3f}")
    print(f"AUC-PR        : {auc_pr:.3f}")
    print(f"Brier score   : {brier:.3f}")

    print("\nClassification report :")
    print(classification_report(y_true, y_pred, zero_division=0))


#Chargement des données
path = "C:/Users/YOGA Slim/Documents/GitHub/projet-maladie-cardiaque-IA/BASES_FINALES_B/"

X_train_A1 = pd.read_csv(path + "X_train_A1_raw.csv")
X_train_A2 = pd.read_csv(path + "X_train_A2_scaled.csv")
X_train_B1 = pd.read_csv(path + "X_train_B1_smote.csv")
X_train_B2 = pd.read_csv(path + "X_train_B2_smote_scaled.csv")

y_train_A = pd.read_csv(path + "y_train_A_raw.csv").values.ravel()
y_train_B = pd.read_csv(path + "y_train_B_smote.csv").values.ravel()

X_test_raw = pd.read_csv(path + "X_test_raw.csv")
X_test_scaled = pd.read_csv(path + "X_test_scaled.csv")
y_test = pd.read_csv(path + "y_test.csv").values.ravel()


# modèles SVM (4 sous-modèles)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix
)
import numpy as np

# -----------------------------
# FONCTION D’ÉVALUATION GÉNÉRALE
# -----------------------------

def evaluate_svm(X_train, y_train, X_test, y_test, model_name):
    print("\n" + "="*15 + f" Résultats : {model_name} " + "="*15)

    # SVM – probabilité requise pour AUC
    model = SVC(kernel='rbf', probability=True, C=1, gamma='scale')

    # Entraînement
    model.fit(X_train, y_train)

    # Prédiction classes
    y_pred = model.predict(X_test)

    # Probabilités
    y_prob = model.predict_proba(X_test)[:, 1]

    # -----------------------
    #    MÉTRIQUES
    # -----------------------

    acc  = accuracy_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    aupr = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    print("Matrice de confusion (rows = vrai, cols = prédit) :")
    print(cm)
    print("\nAccuracy      :", round(acc,3))
    print("Recall        :", round(rec,3))
    print("Precision     :", round(prec,3))
    print("F1-score      :", round(f1,3))
    print("AUC-ROC       :", round(auc,3))
    print("AUC-PR        :", round(aupr,3))
    print("Brier score   :", round(brier,3))

    print("\nClassification report :")
    print(classification_report(y_test, y_pred))

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Recall": rec,
        "Precision": prec,
        "F1-score": f1,
        "AUC-ROC": auc,
        "AUC-PR": aupr,
        "Brier": brier
    }


# -----------------------------
#     Exécution des modèles
# -----------------------------

results_svm = []

# A1 : RAW — non SMOTE, non-scalé
results_svm.append(
    evaluate_svm(X_train_A1, y_train_A, X_test_raw, y_test, "SVM_A1")
)

# A2 : SCALÉ — non SMOTE + Scaling
results_svm.append(
    evaluate_svm(X_train_A2, y_train_A, X_test_scaled, y_test, "SVM_A2")
)

# B1 : SMOTE — non-scalé
results_svm.append(
    evaluate_svm(X_train_B1, y_train_B, X_test_raw, y_test, "SVM_B1")
)

# B2 : SMOTE + SCALING
results_svm.append(
    evaluate_svm(X_train_B2, y_train_B, X_test_scaled, y_test, "SVM_B2")
)


# -----------------------------
#     Tableau comparatif final
# -----------------------------
import pandas as pd

df_svm = pd.DataFrame(results_svm)
print("\n\n===== TABLEAU COMPARATIF SVM =====")
print(df_svm)
