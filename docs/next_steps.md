# Prochaines Étapes

## Fait

| Étape | Détail |
|-------|--------|
| EDA | Exploration, valeurs manquantes, corrélations |
| Extraction données | Landsat (6 bandes + 4 indices), TerraClimate (10 variables) |
| Feature engineering | day_of_year, season, nir_green_ratio, swir_ratio |
| One-hot encoding | season → 3 colonnes binaires |
| Pipeline src/ | Fonctions réutilisables dans src/features et src/models |
| Modèle baseline | Random Forest, split 60/20/20, learning curves |

## À faire

### Priorité 1 : Améliorer le modèle actuel

| Tâche | Pourquoi |
|-------|----------|
| Analyser les learning curves | Vérifier si overfitting |
| Tuner max_depth | Réduire si overfitting |
| Tuner n_estimators | 100 → 200 ? |
| Tester sans certaines features | Simplifier si performances similaires |

### Priorité 2 : Tester d'autres modèles

| Modèle | Avantage |
|--------|----------|
| LightGBM | Plus rapide, souvent meilleur |
| XGBoost | Robuste |
| Ridge/Lasso | Simple, interprétable |

```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
```

### Priorité 3 : Validation spatiale

**Problème actuel** : Le split est aléatoire. Un même site peut être dans train ET test → fuite de données.

**Solution** : GroupKFold par site.

```python
from sklearn.model_selection import GroupKFold

site_ids = df['Latitude'].astype(str) + '_' + df['Longitude'].astype(str)
gkf = GroupKFold(n_splits=5)

for train_idx, test_idx in gkf.split(X, y, groups=site_ids):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # ...
```

### Priorité 4 : Améliorer le phosphore

Le phosphore (DRP) est difficile à prédire. Idées :

| Approche | Comment |
|----------|---------|
| Transformation log | `y['DRP'] = np.log1p(y['DRP'])` |
| Classification | Prédire bas/moyen/élevé au lieu d'une valeur |
| Features décalées | Utiliser satellite de J-7 (laisser le temps aux algues) |

### Priorité 5 : Soumission finale

1. Réentraîner sur tout le training set (sans split)
2. Appliquer le même pipeline au submission
3. Générer les prédictions
4. Sauvegarder au format demandé

```python
# Réentraîner sur tout
X_all = select_model_features(df_train)
y_all = df_train[TARGETS]
X_all_sc = scaler.fit_transform(X_all)
models_final = train_models(X_all_sc, y_all)

# Prédire submission
X_sub = select_model_features(df_submission)
X_sub_sc = scaler.transform(X_sub)
predictions = predict(models_final, X_sub_sc)

# Sauvegarder
predictions.to_csv('outputs/submissions/submission.csv', index=False)
```

## Checklist avant soumission

- [ ] Learning curves OK (pas d'overfitting majeur)
- [ ] Validation spatiale testée
- [ ] Performances sur test set notées
- [ ] Pipeline appliqué au submission
- [ ] Fichier submission.csv généré
- [ ] Format vérifié (colonnes, ordre)
