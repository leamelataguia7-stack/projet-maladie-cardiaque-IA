import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# -------------------------------------------------------------------
# 1. Chargement des données prétraitées
#    (adapter les chemins si besoin)
# -------------------------------------------------------------------
BASE_PATH = "PROJET_FINAL/data/processed/datasets_final/"

X_train_A1 = pd.read_csv(BASE_PATH + "X_train_A1.csv")
X_train_A2 = pd.read_csv(BASE_PATH + "X_train_A2.csv")

X_test_raw = pd.read_csv(BASE_PATH + "X_test_raw.csv")
X_test_scaled = pd.read_csv(BASE_PATH + "X_test_scaled.csv")

y_train = pd.read_csv(BASE_PATH + "y_train.csv").values.ravel()
y_test = pd.read_csv(BASE_PATH + "y_test.csv").values.ravel()


# -------------------------------------------------------------------
# 2. Fonctions utilitaires
# -------------------------------------------------------------------
def evaluate_thresholds(y_true, y_proba, thresholds):
    """
    Calcule les métriques pour une grille de seuils.
    Retourne un DataFrame avec :
    threshold, accuracy, recall, precision, f1, auc_roc, auc_pr.
    """
    results = []
    # AUC-ROC / AUC-PR ne dépendent pas du seuil → on les calcule une fois
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred)
        results.append(
            {
                "threshold": thr,
                "accuracy": acc,
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "brier": brier,
            }
        )
    return pd.DataFrame(results)


def metrics_at_threshold(y_true, y_proba, thr, model_name, data_name):
    """
    Calcule les métriques pour un seuil donné.
    """
    y_pred = (y_proba >= thr).astype(int)
    return {
        "model": model_name,
        "data": data_name,
        "threshold": thr,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "auc_pr": average_precision_score(y_true, y_proba), 
        "brier": brier_score_loss(y_true, y_pred),
    }


# -------------------------------------------------------------------
# 3. Définition des modèles (sans SMOTE)
#    → A2 (scaled) pour LogReg / SVM
#    → A1 (raw)    pour RF / GB / XGB
# -------------------------------------------------------------------
models = {
    "LogReg_A2": {
        "estimator": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
        ),
        "X_train": X_train_A2,
        "X_test": X_test_scaled,
    },
    "SVM_A2": {
        "estimator": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight=None,
        ),
        "X_train": X_train_A2,
        "X_test": X_test_scaled,
    },
    "RF_A1": {
        "estimator": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "X_train": X_train_A1,
        "X_test": X_test_raw,
    },
    "GB_A1": {
        "estimator": GradientBoostingClassifier(
            random_state=42,
        ),
        "X_train": X_train_A1,
        "X_test": X_test_raw,
    },
    "XGB_A1": {
        "estimator": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "X_train": X_train_A1,
        "X_test": X_test_raw,
    },
}


# -------------------------------------------------------------------
# 4. Boucle principale : entraînement, scan de seuils, F1-optimal
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Grille de seuils
    thresholds = np.linspace(0.05, 0.9, 86)

    all_rows_compare = []
    all_scans = []

    for model_name, cfg in models.items():
        print(f"\n=== Traitement du modèle {model_name} ===")

        est = cfg["estimator"]
        Xtr = cfg["X_train"]
        Xte = cfg["X_test"]

        # 4.1. Entraînement
        est.fit(Xtr, y_train)

        # 4.2. Probabilités sur le test
        y_proba = est.predict_proba(Xte)[:, 1]

        # 4.3. Scan des seuils pour F1
        df_scan = evaluate_thresholds(y_test, y_proba, thresholds)
        df_scan["model"] = model_name
        all_scans.append(df_scan)

        # Seuil optimal (F1 max)
        idx_max = df_scan["f1"].idxmax()
        thr_opt = float(df_scan.loc[idx_max, "threshold"])
        print(f"Seuil F1-optimal pour {model_name} : {thr_opt:.3f}")
        print(df_scan.loc[idx_max])

        # 4.4. Lignes de comparaison (0.5 vs thr_opt)
        row_05 = metrics_at_threshold(
            y_test, y_proba, 0.50, model_name=model_name, data_name="sans_SMOTE_thr=0.5"
        )
        row_opt = metrics_at_threshold(
            y_test, y_proba, thr_opt, model_name=model_name, data_name="sans_SMOTE_thr_F1_opt"
        )

        all_rows_compare.extend([row_05, row_opt])

    # ----------------------------------------------------------------
    # 5. Sauvegarde des résultats
    # ----------------------------------------------------------------
    df_compare = pd.DataFrame(all_rows_compare)
    df_compare.to_csv(BASE_PATH + "threshold_compare_all_models_A_sans_SMOTE.csv", index=False)
    print("\n=== Tableau récapitulatif (0.5 vs F1-opt) sauvegardé dans :")
    print(BASE_PATH + "threshold_compare_all_models_A_sans_SMOTE.csv")

    df_scans = pd.concat(all_scans, ignore_index=True)
    df_scans.to_csv(BASE_PATH + "threshold_scans_all_models_A_sans_SMOTE.csv", index=False)
    print("=== Scans complets de seuils sauvegardés dans :")
    print(BASE_PATH + "threshold_scans_all_models_A_sans_SMOTE.csv")
