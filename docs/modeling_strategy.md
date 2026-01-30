# Stratégie de Modélisation

## Vue d'ensemble

```
1. Explorer les données (EDA)
      ↓
2. Créer des features
      ↓
3. Entraîner un modèle simple (baseline)
      ↓
4. Améliorer le modèle
      ↓
5. Faire les prédictions finales
```

---

## Étape 1 : Comprendre les données

**Notebook** : `01_EDA.ipynb`

Questions à se poser :
- Combien de lignes ? Combien de sites ?
- Y a-t-il des valeurs manquantes ?
- Comment sont distribuées les targets ?
- Quelles features sont corrélées aux targets ?

---

## Étape 2 : Créer des features

**Notebook** : `02_Feature_Engineering.ipynb`

Idées de features :
- **Temporelles** : mois, saison
- **Spectrales** : ratios entre bandes (ex: nir/green)
- **Climatiques** : autres variables TerraClimate

---

## Étape 3 : Modèle baseline

**Notebook** : `03_Modeling.ipynb`

Le benchmark utilise un **Random Forest** avec :
- 4 features : `swir22`, `NDMI`, `MNDWI`, `pet`
- Split 70/30 train/test

C'est notre point de départ à battre.

---

## Étape 4 : Améliorer le modèle

### Options pour améliorer :

1. **Plus de features**
   - Ajouter les features créées à l'étape 2
   - Tester si ça améliore le score

2. **Autres modèles**
   - LightGBM : souvent meilleur que Random Forest
   - XGBoost : très populaire en compétition

3. **Tuning des hyperparamètres**
   - Ajuster `n_estimators`, `max_depth`, etc.
   - Utiliser `GridSearchCV` ou `RandomizedSearchCV`

4. **Validation croisée**
   - Utiliser `cross_val_score` pour une évaluation plus fiable
   - Idéalement : validation spatiale (par site)

---

## Le piège à éviter : l'overfitting spatial

Le modèle sera testé sur des **nouveaux sites**.

Si on fait un split train/test classique, le modèle peut "tricher" en apprenant des choses spécifiques à chaque site.

**Solution** : Validation croisée par site (GroupKFold)

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
for train_idx, test_idx in cv.split(X, y, groups=site_ids):
    # train_idx et test_idx n'ont pas de sites en commun
```

---

## Métriques d'évaluation

| Métrique | Description | Objectif |
|----------|-------------|----------|
| **R²** | % de variance expliquée | Plus c'est proche de 1, mieux c'est |
| **RMSE** | Erreur moyenne | Plus c'est petit, mieux c'est |

---

## Checklist avant de soumettre

- [ ] Le modèle a été testé correctement
- [ ] Les prédictions n'ont pas de valeurs manquantes
- [ ] Le fichier est au bon format (CSV)
- [ ] Le code est reproductible (seed fixé)
