# EY Challenge - Qualite de l'Eau

Predire 3 indicateurs de qualite de l'eau en Afrique du Sud a partir de donnees satellite et environnementales.

## Targets (ce qu'on predit)

| Variable | Unite | Description |
|----------|-------|-------------|
| Total Alkalinity | mg/L CaCO3 | Capacite tampon de l'eau |
| Electrical Conductance | uS/cm | Concentration en mineraux dissous |
| Dissolved Reactive Phosphorus | ug/L | Phosphore biodisponible |

## Structure du projet

```
PROJET EY/
|
|-- data/
|   |-- raw/                    # Donnees brutes (training, submission template)
|   |-- processed/              # Features extraites et fusionnees
|   |   |-- merged_training.csv     # FINAL: 9319 samples, 79 features
|   |   |-- merged_validation.csv   # FINAL: 200 samples pour soumission
|   |   +-- gemstat_features.csv    # Donnees externes (1415 samples)
|   +-- submissions/            # Fichiers de soumission
|
|-- notebooks/                  # Extraction des features (Google Earth Engine)
|   |-- 00_GETTING_STARTED.ipynb    # Reference du challenge
|   |-- 01_LANDSAT_EXTRACTION.ipynb # Bandes spectrales + indices
|   |-- 02_TERRACLIMATE_EXTRACTION.ipynb # Climat (34 features)
|   |-- 03_WORLDCOVER_EXTRACTION.ipynb   # Occupation du sol (8 classes)
|   |-- 04_SOILGRIDS_EXTRACTION.ipynb    # Proprietes du sol (6 features)
|   |-- 05_DEM_EXTRACTION.ipynb          # Elevation, pente, aspect
|   |-- 06_DATA_FUSION.ipynb             # Fusion de toutes les features
|   |-- 07_GEMSTAT_PREPARATION.ipynb     # Preparation donnees externes
|   +-- 08_GEMSTAT_FEATURES.ipynb        # Extraction features GEMStat
|
|-- scripts/                    # Scripts d'entrainement et soumission
|   |-- train_common_features.py    # Entrainement sur 51 features communes
|   |-- train_combined.py           # Analyse Training vs GEMStat
|   +-- generate_submission.py      # Generation du fichier de soumission
|
|-- src/                        # Code reutilisable
|   |-- config.py               # TARGETS, FEATURES, constantes
|   |-- models/train.py         # split_data, normalize, train_models, evaluate
|   +-- features/engineering.py # Fonctions de feature engineering
|
+-- docs/                       # Documentation
```

## Donnees disponibles

### Features (79 au total dans merged_training.csv)

| Source | Nb | Features |
|--------|---:|----------|
| Landsat | 22 | Bandes (blue, green, red, nir, swir16, swir22) + indices (NDVI, NDWI, NDMI, MNDWI) + std |
| TerraClimate | 34 | pet, aet, ppt, tmax, tmin, soil, def, pdsi, vpd, ws + lags + anomalies |
| WorldCover | 8 | lc_tree, lc_shrubland, lc_grassland, lc_cropland, lc_builtup, lc_bare, lc_water, lc_wetland |
| SoilGrids | 6 | soil_ph, soil_clay, soil_sand, soil_soc, soil_cec, soil_nitrogen |
| DEM | 3 | elevation, slope, aspect |
| Autres | 2 | water_type, distance_to_river |
| Meta | 3 | Latitude, Longitude, Sample Date |

### GEMStat (donnees externes)

- 1415 echantillons de l'Eastern Cape
- 51 features communes (pas de Landsat, water_type, distance_to_river)
- **Attention**: DRP etait en mg/L, converti en ug/L (x1000)

## Workflow actuel

### 1. Extraction des features (deja fait)

Les notebooks 01-06 extraient les features depuis Google Earth Engine.
**Resultat**: `merged_training.csv` et `merged_validation.csv`

### 2. Entrainement

```bash
# Option 1: Sur les 51 features communes (sans Landsat)
python scripts/train_common_features.py

# Option 2: Analyse comparative Training vs GEMStat
python scripts/train_combined.py
```

### 3. Soumission

```bash
python scripts/generate_submission.py
# -> data/submissions/submission_common_features.csv
```

## Resultats actuels

| Dataset | Total Alkalinity | Electrical Conductance | DRP |
|---------|-----------------|----------------------|-----|
| CV Training (R2) | 0.82 | 0.86 | 0.67 |
| CV Combine (R2) | 0.83 | 0.85 | 0.51 |
| Train->GEMStat (R2) | -0.54 | -0.30 | -0.12 |

**Probleme**: Domain shift entre Training et GEMStat - les modeles ne generalisent pas.

## Prochaines etapes

1. [ ] Extraire Landsat + distance_to_river pour GEMStat
2. [ ] Tester d'autres modeles (XGBoost, LightGBM)
3. [ ] Reduire l'overfitting (regularisation, moins de features)
4. [ ] Domain adaptation techniques

## Installation

```bash
pip install -r requirements.txt
```

## Fichiers archives

Les anciens notebooks et scripts sont dans `notebooks/_archive/` et `scripts/_archive/`.
