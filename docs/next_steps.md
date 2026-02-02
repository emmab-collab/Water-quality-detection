# Prochaines Étapes - EY Water Quality Challenge

## Ce qu'on a fait ✅

1. **EDA complète** : compréhension des données, valeurs manquantes, outliers, corrélations
2. **Feature Engineering** : ajout de `day_of_year`, `season`, ratios spectraux
3. **Amélioration des features** : passage de 4 à 7 features (ajout de `nir`, `green`, `swir16`)
4. **Documentation** : data dictionary, interprétation physique des corrélations

---

## Prochaines étapes 🚀

### 1. Récupérer plus de données TerraClimate

**Pourquoi ?** Les précipitations et la température pourraient aider à prédire le phosphore (ruissellement agricole, croissance des algues).

**Variables à récupérer :**

| Variable | Description | Priorité |
|----------|-------------|----------|
| `ppt` | Précipitations (mm) | ⭐⭐⭐ Haute |
| `tmax` | Température max (°C) | ⭐⭐⭐ Haute |
| `tmin` | Température min (°C) | ⭐⭐ Moyenne |
| `soil` | Humidité du sol | ⭐⭐ Moyenne |
| `def` | Déficit hydrique | ⭐ Basse |

**Comment faire :**

```python
# Option 1 : API TerraClimate via Google Earth Engine
import ee
ee.Initialize()

# Charger TerraClimate
terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')

# Filtrer par date et localisation
filtered = terraclimate.filterDate('2011-01-01', '2015-12-31')

# Extraire les variables
# ppt = précipitations, tmmx = temp max, tmmn = temp min
```

```python
# Option 2 : Télécharger directement depuis le site
# https://www.climatologylab.org/terraclimate.html
# Sélectionner : Afrique du Sud, 2011-2015, variables souhaitées
```

---

### 2. Récupérer plus de bandes Landsat

**Pourquoi ?** La bande rouge permettrait de calculer le NDVI (végétation autour des sites), et la bande thermique donnerait la température de l'eau.

**Bandes à récupérer :**

| Bande | Description | Priorité |
|-------|-------------|----------|
| `red` | Bande rouge | ⭐⭐⭐ Haute (pour NDVI) |
| `blue` | Bande bleue | ⭐⭐ Moyenne |
| `thermal` | Température surface | ⭐⭐⭐ Haute |

**Comment faire :**

```python
# Via Google Earth Engine
import ee
ee.Initialize()

# Landsat 7 ou 8 selon les dates
landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')

# Filtrer et extraire
# B2 = blue, B3 = green, B4 = red, B5 = nir, B6 = swir16, B7 = swir22
# B10 = thermal
```

---

### 3. Créer de nouvelles features

**À partir des données existantes :**

| Feature | Formule | Utilité |
|---------|---------|---------|
| `day_sin` | sin(2π × day_of_year / 365) | Encodage cyclique du jour |
| `day_cos` | cos(2π × day_of_year / 365) | Encodage cyclique du jour |
| `nir_green_ratio` | nir / green | Détection algues |
| `swir_ratio` | swir16 / swir22 | Humidité vs minéraux |

**À partir des nouvelles données (si récupérées) :**

| Feature | Formule | Utilité |
|---------|---------|---------|
| `NDVI` | (nir - red) / (nir + red) | Végétation autour du site |
| `ppt_7d` | Somme précipitations 7 derniers jours | Ruissellement récent |
| `temp_mean` | (tmax + tmin) / 2 | Température moyenne |

---

### 4. Améliorer le modèle

**Étapes :**

1. **Tester avec les nouvelles features** (7 features actuelles)
2. **Comparer les performances** avec le benchmark original (4 features)
3. **Optimiser les hyperparamètres** (GridSearch ou RandomSearch)
4. **Tester d'autres modèles** :
   - LightGBM (souvent meilleur que Random Forest)
   - XGBoost
   - Gradient Boosting

**Validation :**

- Utiliser une **validation spatiale** (sites différents en train/test)
- Pas juste un split aléatoire !

```python
from sklearn.model_selection import GroupKFold

# Grouper par site pour éviter la fuite de données
group_kfold = GroupKFold(n_splits=5)
for train_idx, test_idx in group_kfold.split(X, y, groups=site_ids):
    # ...
```

---

### 5. Traiter le problème du phosphore

Le phosphore est difficile à prédire (corrélations faibles). Idées :

| Approche | Description |
|----------|-------------|
| **Transformation log** | `log(phosphore)` pour réduire l'asymétrie |
| **Classification** | Prédire une classe (bas/moyen/élevé) au lieu d'une valeur |
| **Features décalées** | Utiliser les données satellite de J-7 ou J-14 (laisser le temps aux algues de pousser) |
| **Données externes** | Ajouter des données sur l'usage des sols (agricole, urbain...) |

---

## Ordre de priorité

1. ⭐⭐⭐ **Tester le modèle avec les 7 features actuelles** (rapide, déjà prêt)
2. ⭐⭐⭐ **Récupérer précipitations et température** (impact potentiel élevé)
3. ⭐⭐ **Ajouter la bande rouge + NDVI** (végétation = proxy du ruissellement)
4. ⭐⭐ **Optimiser les hyperparamètres**
5. ⭐ **Tester LightGBM/XGBoost**
6. ⭐ **Features décalées dans le temps** (si les autres n'améliorent pas assez)

---

## Ressources

- **TerraClimate** : https://www.climatologylab.org/terraclimate.html
- **Google Earth Engine** : https://earthengine.google.com/
- **Landsat bands** : https://www.usgs.gov/landsat-missions/landsat-8
- **scikit-learn GroupKFold** : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
