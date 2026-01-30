# Dictionnaire des Données

## Les fichiers de données

| Fichier | Description |
|---------|-------------|
| `water_quality_training_dataset.csv` | Mesures de qualité d'eau |
| `landsat_features_training.csv` | Données satellite |
| `terraclimate_features_training.csv` | Données climatiques |

---

## Variables à prédire (Targets)

| Colonne | Type | Description |
|---------|------|-------------|
| `Total Alkalinity` | nombre | Alcalinité totale (mg/L) |
| `Electrical Conductance` | nombre | Conductivité électrique (µS/cm) |
| `Dissolved Reactive Phosphorus` | nombre | Phosphore réactif (µg/L) |

### Valeurs de référence

#### Total Alkalinity (mg/L CaCO3)

| Niveau | Valeur | Interprétation |
|--------|--------|----------------|
| 🔴 Trop bas | < 20 mg/L | Eau acide, faible capacité tampon |
| 🟢 Normal | 30 - 150 mg/L | Bon pour la vie aquatique |
| 🟠 Élevé | 150 - 300 mg/L | Eau calcaire, acceptable |
| 🔴 Très élevé | > 300 mg/L | Problématique |

**Dans nos données** : 5 à 362 mg/L, moyenne 119 mg/L

#### Electrical Conductance (µS/cm)

| Niveau | Valeur | Interprétation |
|--------|--------|----------------|
| 🟢 Eau douce | < 500 µS/cm | Peu minéralisée |
| 🟢 Normal | 500 - 1000 µS/cm | Eau douce typique |
| 🟠 Élevé | 1000 - 2000 µS/cm | Minéralisation importante |
| 🔴 Saumâtre | > 2000 µS/cm | Problème de salinité |

**Dans nos données** : 15 à 1506 µS/cm, moyenne 485 µS/cm

#### Dissolved Reactive Phosphorus (µg/L)

| Niveau | Valeur | Interprétation |
|--------|--------|----------------|
| 🟢 Oligotrophe | < 10 µg/L | Eau pauvre en nutriments (claire) |
| 🟢 Mésotrophe | 10 - 20 µg/L | Niveau intermédiaire |
| 🟠 Eutrophe | 20 - 100 µg/L | Risque de prolifération d'algues |
| 🔴 Hypereutrophe | > 100 µg/L | Eutrophisation, problème écologique |

**Dans nos données** : 5 à 195 µg/L, moyenne 44 µg/L ⚠️ (beaucoup de sites eutrophes)

> **Sources** : [EPA](https://archive.epa.gov/water/archive/web/html/vms510.html), [Victoria Water](https://data.water.vic.gov.au/what-do-water-quality-parameters-mean), [Penn State](https://extension.psu.edu/interpreting-irrigation-water-tests)

---

## Variables de localisation

| Colonne | Type | Description |
|---------|------|-------------|
| `Site Name` | texte | Nom du site de mesure |
| `Latitude` | nombre | Latitude GPS |
| `Longitude` | nombre | Longitude GPS |
| `Sample Date` | date | Date de la mesure |

---

## Variables Landsat (satellite)

### Bandes spectrales

| Colonne | Description |
|---------|-------------|
| `nir` | Proche infrarouge (Near Infrared) |
| `green` | Bande verte |
| `swir16` | Infrarouge ondes courtes 1 |
| `swir22` | Infrarouge ondes courtes 2 |

### Indices spectraux

| Colonne | Formule | Utilité |
|---------|---------|---------|
| `NDMI` | (nir - **swir16**) / (nir + **swir16**) | Détection de l'humidité |
| `MNDWI` | (green - **swir16**) / (green + **swir16**) | Détection de l'eau |

> **Note** : NDMI et MNDWI utilisent **swir16** (1.6 µm), pas swir22.
> C'est pourquoi **swir22** est ajouté séparément dans le benchmark : il apporte une information complémentaire (sensible aux minéraux et à la turbidité).

---

## Variables TerraClimate (climat)

| Colonne | Description |
|---------|-------------|
| `pet` | Évapotranspiration potentielle (mm) |

---

## Le benchmark utilise seulement 4 features

```python
BENCHMARK_FEATURES = ['swir22', 'NDMI', 'MNDWI', 'pet']
```

C'est un bon point de départ, mais on peut ajouter plus de features pour améliorer le modèle.
