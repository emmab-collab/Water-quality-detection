# EY Challenge - Prédiction de la Qualité de l'Eau

Projet pour le challenge EY AI & Data : prédire la qualité de l'eau en Afrique du Sud à partir de données satellite et climatiques.

## Objectif

Prédire 3 indicateurs de qualité de l'eau :
- **Total Alkalinity** : alcalinité de l'eau
- **Electrical Conductance** : conductivité électrique
- **Dissolved Reactive Phosphorus** : phosphore réactif

## Structure du projet

```
├── Snowflake Notebooks Package/   # Données du challenge
│   ├── water_quality_training_dataset.csv
│   ├── landsat_features_training.csv
│   └── terraclimate_features_training.csv
│
├── notebooks/                     # Notebooks à exécuter
│   ├── 01_EDA.ipynb              # Explorer les données
│   ├── 02_Feature_Engineering.ipynb  # Créer des features
│   └── 03_Modeling.ipynb         # Entraîner le modèle
│
├── src/                          # Code Python
│   ├── config.py                 # Configuration (targets, features)
│   ├── paths.py                  # Chemins vers les fichiers
│   └── data/
│       └── load_data.py          # Fonctions de chargement
│
├── docs/                         # Documentation
│   ├── problem_definition.md     # Définition du problème
│   ├── data_dictionary.md        # Description des données
│   └── modeling_strategy.md      # Stratégie de modélisation
│
└── outputs/                      # Résultats
    └── submissions/              # Fichiers de soumission
```

## Comment démarrer ?

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Exécuter les notebooks dans l'ordre

1. **01_EDA.ipynb** : Comprendre les données
2. **02_Feature_Engineering.ipynb** : Créer de nouvelles features
3. **03_Modeling.ipynb** : Entraîner et évaluer le modèle

## Les données

| Fichier | Description |
|---------|-------------|
| `water_quality_training_dataset.csv` | Mesures de qualité d'eau (~9300 lignes) |
| `landsat_features_training.csv` | Données satellite (swir22, NDMI, MNDWI...) |
| `terraclimate_features_training.csv` | Données climat (pet...) |

### Description des colonnes

#### Colonnes d'identification

| Colonne | Description |
|---------|-------------|
| `Latitude` | Position GPS nord-sud du site de mesure |
| `Longitude` | Position GPS est-ouest du site de mesure |
| `Sample Date` | Date de prélèvement de l'échantillon |

#### Colonnes TARGETS (à prédire)

| Colonne | Description | Unité | Ce que ça mesure |
|---------|-------------|-------|------------------|
| `Total Alkalinity` | Alcalinité totale | mg/L | Capacité de l'eau à neutraliser les acides (lié au calcaire) |
| `Electrical Conductance` | Conductivité électrique | µS/cm | Quantité de minéraux dissous dans l'eau |
| `Dissolved Reactive Phosphorus` | Phosphore réactif dissous | µg/L | Nutriment qui peut causer l'eutrophisation (algues) |

#### Colonnes Landsat (satellite)

**Bandes spectrales brutes :**

| Colonne | Description | Utilité |
|---------|-------------|---------|
| `nir` | Near Infrared (proche infrarouge) | Détecte la végétation et l'humidité |
| `green` | Bande verte | Réflectance de l'eau |
| `swir16` | Shortwave Infrared 1 (1.6 µm) | Sensible à l'humidité du sol |
| `swir22` | Shortwave Infrared 2 (2.2 µm) | Sensible aux minéraux et à la turbidité |

**Indices spectraux calculés :**

| Colonne | Formule | Ce que ça détecte |
|---------|---------|-------------------|
| `NDMI` | (nir - **swir16**) / (nir + **swir16**) | Humidité de la végétation/surface |
| `MNDWI` | (green - **swir16**) / (green + **swir16**) | Présence d'eau (valeurs positives = eau) |

> **Note** : NDMI et MNDWI utilisent **swir16** (1.6 µm). Le **swir22** est ajouté séparément car il capte des informations différentes (minéraux, turbidité).

#### Colonne TerraClimate (climat)

| Colonne | Description | Unité | Ce que ça mesure |
|---------|-------------|-------|------------------|
| `pet` | Potential Evapotranspiration | mm | Demande en eau de l'atmosphère (lié à température, vent, ensoleillement) |

## Le benchmark

Le modèle de référence utilise :
- **4 features** : swir22, NDMI, MNDWI, pet
- **Random Forest** avec 100 arbres
- **Split** 70% train / 30% test

Notre objectif : faire mieux que ce benchmark.

## Métriques

- **R²** : entre 0 et 1, plus c'est proche de 1 mieux c'est
- **RMSE** : erreur moyenne, plus c'est petit mieux c'est

## Auteur

Emmanuelle Benhaim
