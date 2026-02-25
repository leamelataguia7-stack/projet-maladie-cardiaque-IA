**Auteur :** Melataguia Mekontchou Lea  
**Projet (M2) :** Prédiction du risque de maladie coronarienne à 10 ans à partir de la base Framingham, avec apprentissage automatique et quantification de l’incertitude.

Ce dépôt présente un pipeline complet de modélisation prédictive sur données tabulaires (cohorte de type **Framingham**) pour estimer le risque d’événement coronarien à 10 ans, avec une attention particulière portée à :  
- la gestion des **données manquantes** (imputation multivariée),  
- le **déséquilibre de classes** (≈ 15% d’événements),  
- la **calibration** et la **quantification explicite de l’incertitude** (bootstrap + prédiction conforme Mondrian).

---

## Objectifs

1. Décrire la cohorte et la qualité des données (valeurs manquantes, déséquilibre de classes).  
2. Comparer plusieurs approches de prédiction supervisée (régression logistique, SVM, Random Forest, Gradient Boosting, XGBoost).  
3. Évaluer les performances (métriques adaptées au déséquilibre) et quantifier l’incertitude au niveau global (bootstrap) et individuel (conformal prediction).

---

## Données

- **Source :** Framingham Heart Study (extraction dérivée à usage pédagogique / méthodologique).  
- **Taille :** N = 4 239 participants.  
- **Issue :** *TenYearCHD* (événement coronarien à 10 ans, binaire).  
- **Prévalence :** 15,2% d’événements (déséquilibre de classes).  
- **Manquants :** faibles pour la plupart des variables, plus élevés pour la glycémie (~9%).  

> ⚠️ Ce dépôt n’a pas vocation à fournir un outil clinique opérationnel. Une validation externe est nécessaire avant toute utilisation en pratique.

---

## Méthodes (résumé)

### Prétraitement
- Imputation des valeurs manquantes : **MICE** (Multiple Imputation by Chained Equations).  
- Standardisation des variables continues (pour modèles sensibles à l’échelle).  
- Gestion du déséquilibre : **SMOTE** (appliqué uniquement sur le jeu d’apprentissage).  
- Optimisation du **seuil de décision** (maximisation du F1-score de la classe positive).

### Modèles évalués
- Régression logistique (LogReg)  
- Support Vector Machine (SVM)  
- Random Forest (RF)  
- Gradient Boosting (GB)  
- XGBoost (XGB)

### Évaluation
- Jeu de test indépendant conservant la prévalence naturelle (N = 848).  
- Métriques : Recall, Precision, F1-score (classe positive), AUC-ROC, AUC-PR, Brier score.  
- Incertitude :
  - **Bootstrap** (B = 1000) sur le jeu de test (IC95% percentiles).  
  - **Prédiction conforme Mondrian** (label-conditional) pour l’incertitude individuelle.

---

## Résultats principaux (modèle final)

Le meilleur compromis a été obtenu avec une **régression logistique standardisée sans SMOTE** avec **seuil optimisé à 0,15** (LogReg_A2).

### Performance sur jeu de test (N = 848)
- **Accuracy :** 0,667  
- **Recall (classe 1) :** 0,605  
- **Precision (classe 1) :** 0,252  
- **F1-score (classe 1) :** 0,356  
- **AUC-ROC :** 0,697  
- **AUC-PR :** 0,290  
- **Brier score :** 0,122  

### Bootstrap (B = 1000) — IC95% (jeu de test)
- Accuracy : 0,667 [0,636 ; 0,697]  
- Recall classe 1 : 0,605 [0,517 ; 0,688]  
- Precision classe 1 : 0,252 [0,204 ; 0,303]  
- F1 classe 1 : 0,356 [0,297 ; 0,413]  
- AUC-ROC : 0,697 [0,652 ; 0,741]  
- AUC-PR : 0,290 [0,227 ; 0,374]  
- Brier : 0,122 [0,107 ; 0,138]

### Prédiction conforme Mondrian (α = 0,10 ; confiance 90%)
- **Couverture globale :** 0,902  
- Couverture classe 0 : 0,901  
- Couverture classe 1 : 0,907  
- Ensembles {0} : 33,0%  
- Ensembles {1} : 12,3%  
- Ensembles {0,1} : 54,7%  
- Taille moyenne de Γ(x) : 1,55  



