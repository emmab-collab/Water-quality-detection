# Définition du Problème - EY Challenge

## C'est quoi ce challenge ?

On doit **prédire la qualité de l'eau** en Afrique du Sud.

Au lieu de faire des mesures sur le terrain (cher et long), on utilise :
- Des images **satellite** (Landsat)
- Des données **climatiques** (TerraClimate)

## Les 3 choses à prédire (targets)

| Nom | C'est quoi ? |
|-----|--------------|
| **Total Alkalinity** | Capacité de l'eau à résister aux changements de pH |
| **Electrical Conductance** | Concentration en minéraux/ions |
| **Dissolved Reactive Phosphorus** | Quantité de phosphore (pollution agricole) |

## Les données disponibles

### 1. Données de qualité d'eau
- ~9,300 mesures
- ~200 sites en Afrique du Sud
- Période : 2011-2015

### 2. Données Landsat (satellite)
- Bandes spectrales : nir, green, swir16, swir22
- Indices calculés : NDMI, MNDWI

### 3. Données TerraClimate (climat)
- pet : évapotranspiration potentielle
- (et d'autres variables disponibles)

## Le défi principal

Le modèle sera testé sur des **sites qu'il n'a jamais vus**.

C'est plus difficile qu'un problème classique car :
- On ne peut pas "apprendre" les caractéristiques spécifiques de chaque site
- Le modèle doit généraliser à de nouveaux endroits

## Comment on est évalué ?

- **RMSE** (Root Mean Square Error) : plus c'est petit, mieux c'est
- On calcule un RMSE pour chaque target, puis la moyenne

## Ce qu'on doit livrer

1. Un fichier CSV avec les prédictions
2. Du code qui marche
3. Une bonne approche expliquée
