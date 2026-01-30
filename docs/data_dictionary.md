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

### Pourquoi utiliser des données satellite ?

Les satellites comme Landsat capturent la lumière à **différentes longueurs d'onde** (pas seulement la lumière visible). Chaque matériau (eau, végétation, sol, minéraux) **réfléchit différemment** selon la longueur d'onde.

C'est comme porter des "lunettes spéciales" qui révèlent des informations invisibles à l'œil nu.

### Comment les matériaux réfléchissent la lumière

| Longueur d'onde | Eau propre | Eau trouble | Algues |
|-----------------|------------|-------------|--------|
| Vert | Réfléchit | Réfléchit ++ | Réfléchit +++ |
| NIR (infrarouge) | Absorbe | Absorbe | Réfléchit |
| SWIR | Absorbe | Réfléchit + | Absorbe |

### Bandes spectrales

| Colonne | Description | Ce qu'elle détecte |
|---------|-------------|-------------------|
| `green` | Bande verte | Réflectance de l'eau, chlorophylle, algues |
| `nir` | Proche infrarouge (Near Infrared) | Végétation (forte réflexion), eau (absorption) |
| `swir16` | Infrarouge ondes courtes 1.6 µm | Humidité du sol et de l'eau |
| `swir22` | Infrarouge ondes courtes 2.2 µm | Minéraux dissous, turbidité (eau trouble) |

### Indices spectraux

| Colonne | Formule | Ce qu'il détecte |
|---------|---------|------------------|
| `NDMI` | (nir - **swir16**) / (nir + **swir16**) | Humidité : valeur haute = humide |
| `MNDWI` | (green - **swir16**) / (green + **swir16**) | Eau : valeur positive = présence d'eau |

### Exemples d'interprétation

| Type d'eau | MNDWI | SWIR22 | Explication |
|------------|-------|--------|-------------|
| Eau claire et propre | Élevé | Bas | L'eau absorbe l'infrarouge |
| Eau trouble (sédiments) | Moyen | Élevé | Les particules réfléchissent le SWIR |
| Eau avec algues | Variable | Bas | Les algues changent la réflectance verte |

> **Note** : NDMI et MNDWI utilisent **swir16** (1.6 µm), pas swir22.
> C'est pourquoi **swir22** est ajouté séparément dans le benchmark : il apporte une information complémentaire (sensible aux minéraux et à la turbidité).

---

## Variables TerraClimate (climat)

| Colonne | Description |
|---------|-------------|
| `pet` | Évapotranspiration potentielle (mm) |

---

## Nos features (améliorées par rapport au benchmark)

### Benchmark original d'EY (4 features)

```python
BENCHMARK_FEATURES_ORIGINAL = ['swir22', 'NDMI', 'MNDWI', 'pet']
```

### Nos features (7 features)

```python
BENCHMARK_FEATURES = ['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI', 'pet']
```

### Pourquoi on a ajouté `nir`, `green` et `swir16` ?

| Feature ajoutée | Pourquoi |
|-----------------|----------|
| `nir` | Détecte la végétation et les **algues** → lié au phosphore |
| `green` | Détecte la **chlorophylle** des algues → lié au phosphore |
| `swir16` | Déjà utilisé dans NDMI et MNDWI, mais apporte de l'info brute sur l'**humidité** |

**Objectif** : Améliorer la prédiction du **phosphore** (difficile à prédire avec le benchmark original car ses sources humaines ne sont pas visibles). En ajoutant `nir` et `green`, on peut détecter les **algues** qui sont une conséquence du phosphore.
