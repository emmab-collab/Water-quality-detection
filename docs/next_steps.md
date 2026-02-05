# Checklist - Améliorer la Prédiction de Qualité de l'Eau

## Situation actuelle

**R² max atteint : ~0.41** avec Random Forest sur 35 features (Landsat + TerraClimate + features créées)

**Problème identifié** : Les données actuelles ne capturent pas assez le contexte hydrologique et spatial.

---

## Phase 1 : Quick wins (données existantes)

### 1.1 ✅ Type de milieu (rivière vs plan d'eau)
- [x] Utiliser HydroLAKES + HydroRIVERS (méthode scientifique)
- [x] Notebook créé : `09_WATER_TYPE_CLASSIFICATION.ipynb`
- [x] Classification effectuée avec buffer 200m

**Résultat :**
| Type | Nb points | % |
|------|-----------|---|
| river | 7392 | 79% |
| unknown | 1900 | 20% |
| lake | 27 | <1% |

**Fichiers créés :** `water_type_training.csv`, `water_type_validation.csv`

### 1.2 ✅ Améliorer l'extraction Landsat (buffer + stats)
- [x] Notebook V2 créé : `06_LANDSAT_DATA_EXTRACTION_V2.ipynb`
- [x] Buffer de ~200m autour du point
- [x] Calcul moyenne + écart-type pour chaque bande/indice
- [x] **Exécuté**

**Nouvelles features (20 au lieu de 10) :**
- Bandes : `blue`, `blue_std`, `green`, `green_std`, etc.
- Indices : `NDVI`, `NDVI_std`, `NDWI`, `NDWI_std`, etc.

### 1.3 ✅ Agrégations temporelles climat (TerraClimate)
- [x] Notebook V2 créé : `05_TERRACLIMATE_DATA_EXTRACTION_V2.ipynb`
- [x] Lags mensuels (lag1, lag2, lag3)
- [x] Cumul 4 mois, moyenne 4 mois
- [x] Anomalie saisonnière
- [x] **Exécuté**

**Nouvelles features (34 au lieu de 10) :**
- Variables avec temporel : `ppt`, `soil`, `def`, `vpd`
- Suffixes : `_lag1`, `_lag2`, `_lag3`, `_sum4`, `_mean4`, `_anomaly`

---

## Phase 2 : Nouvelles sources de données

### 2.1 ✅ ESA WorldCover (occupation du sol)
- [x] Notebook créé : `08_ESA_WORLDCOVER_EXTRACTION.ipynb`
- [x] **Exécuté**
- [x] Extraire sur buffer 500m :
  - [x] % agriculture (`lc_cropland`)
  - [x] % zones urbaines (`lc_builtup`)
  - [x] % zones naturelles (`lc_tree`, `lc_grassland`)
  - [x] % zones humides (`lc_wetland`)

### 2.2 ✅ SoilGrids (propriétés du sol)
- [x] Notebook créé : `10_SOILGRIDS_EXTRACTION.ipynb`
- [x] **Exécuté** (via API REST ISRIC)
- [x] Variables extraites :
  - [x] pH du sol (`soil_ph`)
  - [x] % argiles (`soil_clay`)
  - [x] % sable (`soil_sand`)
  - [x] Carbone organique (`soil_soc`)
  - [x] CEC (`soil_cec`)
  - [x] Azote total (`soil_nitrogen`)

**Source** : API REST ISRIC (https://rest.isric.org)

### 2.3 ✅ DEM (topographie simple)
- [x] Notebook créé : `11_DEM_EXTRACTION.ipynb`
- [x] **Exécuté**
- [x] Variables extraites :
  - [x] Altitude du point (`elevation`)
  - [x] Pente locale (`slope`)
  - [x] Orientation (`aspect`)

**Source** : Copernicus DEM GLO-30 sur Planetary Computer

---

## Phase 3 : Hydrologie avancée (bassin versant)

### 3.1 Délinéation du bassin versant
- [ ] Télécharger DEM haute résolution (SRTM / HydroSHEDS)
- [ ] Installer PySheds ou WhiteboxTools
- [ ] Pour chaque point de mesure :
  - [ ] Délinéer le bassin amont
  - [ ] Calculer la surface drainée
  - [ ] Calculer la pente moyenne du bassin
  - [ ] Calculer l'ordre de Strahler
  - [ ] Calculer la distance au cours d'eau principal

### 3.2 Occupation du sol sur le bassin versant
- [ ] Recalculer ESA WorldCover sur le bassin (pas le buffer)
- [ ] % agriculture dans le bassin amont
- [ ] % zones urbaines dans le bassin amont
- [ ] % zones minières (si disponible)

### 3.3 Géologie / lithologie
- [ ] Trouver source de données géologiques Afrique du Sud
- [ ] Extraire :
  - [ ] Type dominant dans le bassin
  - [ ] % calcaire
  - [ ] % roches mafiques

---

## Phase 4 : Modélisation avancée

### 4.1 Séparer rivière / plan d'eau
- [x] Classification disponible (`water_type`)
- [ ] Option A : Deux modèles séparés
- [ ] Option B : Un modèle avec variable d'interaction
- [ ] Comparer les performances

### 4.2 Modèle par variable cible

| Variable | Drivers principaux | Approche |
|----------|-------------------|----------|
| **Alcalinité** | Sols + géologie | Modèle physique + ML sur résidus |
| **Conductivité** | Hydrologie + climat + sols | XGBoost avec toutes features |
| **Phosphore** | Pluie récente + occupation sol + satellite | Focus sur cumul pluie + agriculture |

### 4.3 Améliorer le modèle ML
- [ ] Tester XGBoost / LightGBM
- [ ] Log-transform si distribution asymétrique
- [ ] Validation croisée spatiale (GroupKFold par site/bassin)
- [ ] Stacking de modèles

---

## Phase 5 : Vérifications finales

### 5.1 Anti-pièges
- [ ] Pas de fuite temporelle (variable calculée après la date)
- [ ] Pas de fuite spatiale (même site dans train et test)
- [ ] Performance séparée par :
  - [ ] Type de milieu (rivière vs plan d'eau)
  - [ ] Variable chimique
  - [ ] Saison

### 5.2 Soumission finale
- [ ] Sélectionner le meilleur modèle
- [ ] Réentraîner sur tout le training set
- [ ] Appliquer le pipeline au submission
- [ ] Vérifier le format du fichier
- [ ] Générer `submission.csv`

---

## Résumé des priorités

| Priorité | Tâche | Impact estimé | Statut |
|----------|-------|---------------|--------|
| ✅ 1 | Type de milieu (rivière/plan d'eau) | Élevé | **FAIT** |
| ✅ 2 | Agrégations temporelles climat | Élevé | **FAIT** |
| ✅ 3 | Buffer + stats Landsat | Moyen | **FAIT** |
| ✅ 4 | ESA WorldCover | Moyen | **FAIT** |
| ✅ 5 | SoilGrids (pH, argiles) | Moyen | **FAIT** |
| ✅ 6 | DEM simple (altitude, pente) | Moyen | **FAIT** |
| 🟡 7 | Bassin versant | Élevé | Complexe |
| 🟡 8 | Géologie | Moyen | Complexe |
| 🟢 9 | XGBoost / LightGBM | Moyen | À faire |
| 🟢 10 | Modèle par variable | Moyen | À faire |

---

## Notebooks créés/modifiés

| Notebook | Statut | Description |
|----------|--------|-------------|
| `05_TERRACLIMATE_DATA_EXTRACTION_V2.ipynb` | ✅ Exécuté | Avec lags et cumuls mensuels |
| `06_LANDSAT_DATA_EXTRACTION_V2.ipynb` | ✅ Exécuté | Avec buffer 200m + stats |
| `08_ESA_WORLDCOVER_EXTRACTION.ipynb` | ✅ Exécuté | Occupation du sol |
| `09_WATER_TYPE_CLASSIFICATION.ipynb` | ✅ Exécuté | Lac vs rivière (HydroSHEDS) |
| `10_SOILGRIDS_EXTRACTION.ipynb` | ✅ Exécuté | Propriétés du sol (API ISRIC) |
| `11_DEM_EXTRACTION.ipynb` | ✅ Exécuté | Topographie (Copernicus DEM) |

---

## Prochaines actions immédiates

### Extractions terminées ✅
1. ~~**Exécuter** `08_ESA_WORLDCOVER_EXTRACTION.ipynb`~~ ✅ FAIT
2. ~~**Exécuter** `10_SOILGRIDS_EXTRACTION.ipynb`~~ ✅ FAIT
3. ~~**Créer et exécuter** `11_DEM_EXTRACTION.ipynb`~~ ✅ FAIT
4. ~~**Exécuter** `05_TERRACLIMATE_DATA_EXTRACTION_V2.ipynb`~~ ✅ FAIT
5. ~~**Exécuter** `06_LANDSAT_DATA_EXTRACTION_V2.ipynb`~~ ✅ FAIT

### Prochaines étapes
1. **Fusionner** tous les CSV en un seul dataset
2. **Réentraîner** le modèle avec toutes les nouvelles features
3. **Comparer** R² avant (~0.41) vs après
4. **Tester** XGBoost / LightGBM
