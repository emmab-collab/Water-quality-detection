# EY Challenge - Qualité de l'Eau

Prédire 3 indicateurs de qualité de l'eau en Afrique du Sud.

## Targets

| Variable | Description |
|----------|-------------|
| Total Alkalinity | Alcalinité (mg/L) |
| Electrical Conductance | Conductivité (µS/cm) |
| Dissolved Reactive Phosphorus | Phosphore (µg/L) |

## Structure

```
├── notebooks/
│   ├── 01_EDA.ipynb                 # Explorer les données
│   ├── 02_Feature_Engineering.ipynb # Nettoyer + créer features
│   └── 03_Modeling.ipynb            # Entraîner + évaluer
│
├── src/
│   ├── config.py          # Constantes (TARGETS, FEATURES)
│   ├── paths.py           # Chemins fichiers
│   ├── data/
│   │   └── load_data.py   # Charger les données
│   ├── features/
│   │   └── engineering.py # Nettoyage + features
│   ├── models/
│   │   └── train.py       # Split + train + evaluate
│   └── visualization/
│       └── plots.py       # Graphiques
│
├── data/
│   ├── raw/               # Données brutes
│   └── processed/         # Données Landsat/TerraClimate extraites
│
└── docs/
    └── next_steps.md      # Prochaines étapes
```

## Pipeline

### 1. Feature Engineering (notebook 02)

```python
from src.features import prepare_training, prepare_submission

df_train, medians = prepare_training(df_raw)      # Nettoie + crée features
df_sub = prepare_submission(df_sub_raw, medians)  # Impute + crée features
```

**Ce que fait `prepare_training` :**
1. Supprime les lignes avec NaN
2. Supprime les valeurs saturées (65535)
3. Calcule les médianes (pour imputer submission)
4. Crée `day_of_year`, `season`, `nir_green_ratio`, `swir_ratio`
5. One-hot encode `season`

### 2. Modeling (notebook 03)

```python
from src.features import select_model_features
from src.models import split_data, normalize, train_models, evaluate

X = select_model_features(df_train)
y = df_train[TARGETS]

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train_sc, X_val_sc, X_test_sc, scaler = normalize(X_train, X_val, X_test)
models = train_models(X_train_sc, y_train)
results = evaluate(models, X_test_sc, y_test)
```

## Features utilisées (23)

| Type | Features |
|------|----------|
| Landsat bandes (6) | blue, green, red, nir, swir16, swir22 |
| Landsat indices (4) | NDVI, NDWI, NDMI, MNDWI |
| TerraClimate (10) | pet, aet, ppt, tmax, tmin, soil, def, pdsi, vpd, ws |
| Créées (3) | day_of_year, nir_green_ratio, swir_ratio |
| Season encodé (3) | season_spring, season_summer, season_winter |

## Métriques

| Métrique | Interprétation |
|----------|----------------|
| R² | 1 = parfait, 0 = nul |
| RMSE | Plus petit = meilleur |

## Commandes

```bash
# Installer
pip install -r requirements.txt

# Exécuter les notebooks dans l'ordre
1. 01_EDA.ipynb
2. 02_Feature_Engineering.ipynb
3. 03_Modeling.ipynb
```
