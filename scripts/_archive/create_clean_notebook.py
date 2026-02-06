import json
import os

# Créer un notebook propre avec la stratégie en 3 étapes
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(cell_id, content):
    notebook["cells"].append({
        "id": cell_id,
        "cell_type": "markdown",
        "metadata": {},
        "source": content
    })

def add_code(cell_id, content):
    notebook["cells"].append({
        "id": cell_id,
        "cell_type": "code",
        "metadata": {},
        "source": content,
        "outputs": [],
        "execution_count": None
    })

# =============================================================================
# CELLULE 1 : Introduction
# =============================================================================
add_markdown("intro", """# Landsat - Complétion du Dataset Existant

## Stratégie d'extraction

Nous avons déjà un fichier `landsat_features_training.csv` avec :
- **8234 lignes** avec données : nir, green, swir16, swir22, NDMI, MNDWI
- **1085 lignes** sans données (NaN)

```
┌─────────────────────────────────────────────────────────────┐
│  DONNÉES EXISTANTES (landsat_features_training.csv)         │
│  • 8234 lignes avec: nir, green, swir16, swir22, NDMI, MNDWI│
│  • 1085 lignes sans données (NaN)                           │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────────┐           ┌───────────────────────┐
│  ÉTAPE 1 (~2h)        │           │  ÉTAPE 2 (~20min)     │
│  8234 lignes          │           │  1085 lignes          │
│                       │           │                       │
│  Extraire: blue, red  │           │  Extraire: TOUTES     │
│  Calculer: NDVI, NDWI │           │  bandes (seuil 30%)   │
└───────────────────────┘           └───────────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 3: FUSION                                            │
│  Fichier final: 10 variables pour ~9300 lignes              │
│  blue, green, red, nir, swir16, swir22, NDVI, NDWI, NDMI,   │
│  MNDWI                                                      │
└─────────────────────────────────────────────────────────────┘
```

## Source des données

[Landsat Collection 2 Level 2 sur Planetary Computer](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2)""")

# =============================================================================
# CELLULE 2 : Installation
# =============================================================================
add_markdown("step-install", """---

## Installation des dépendances

⚠️ **Première exécution uniquement** : Redémarrer le kernel après cette cellule.""")

add_code("install-deps", """!pip install -q odc-stac planetary-computer pystac-client""")

# =============================================================================
# CELLULE 3 : Imports et Configuration
# =============================================================================
add_code("imports", """# =============================================================================
# IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

print("Imports OK!")""")

add_code("config", """# =============================================================================
# CONFIGURATION
# =============================================================================

# Toutes les bandes spectrales
ALL_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# Bandes à extraire pour l'étape 1 (nouvelles variables)
BANDS_STEP1 = ["blue", "red"]

# Période de recherche
DATE_RANGE = "2011-01-01/2015-12-31"

# Seuils de couverture nuageuse
MAX_CLOUD_COVER_STRICT = 10   # Pour l'étape 1
MAX_CLOUD_COVER_RELAXED = 30  # Pour l'étape 2 (données manquantes)

# Nombre de workers parallèles
N_WORKERS = 8

# Chemins des fichiers
EXISTING_FILE = "../data/processed/landsat_features_training.csv"
WATER_QUALITY_FILE = "../data/raw/water_quality_training_dataset.csv"
OUTPUT_DIR = "../data/processed"

print("Configuration:")
print(f"  - Étape 1: Extraire {BANDS_STEP1} (seuil nuages: {MAX_CLOUD_COVER_STRICT}%)")
print(f"  - Étape 2: Extraire {ALL_BANDS} (seuil nuages: {MAX_CLOUD_COVER_RELAXED}%)")
print(f"  - Workers: {N_WORKERS}")""")

# =============================================================================
# CELLULE 4 : Explication des bandes
# =============================================================================
add_markdown("explain-bands", """---

## Comprendre les données Landsat

### Les bandes spectrales

```
Spectre électromagnétique :

    Visible                    Infrarouge
    ├────────────────────┼─────────────────────────────────┤
    │  Blue  Green  Red  │  NIR     SWIR16     SWIR22      │
    └────────────────────┴─────────────────────────────────┘
     0.4    0.5   0.6    0.8      1.6        2.2    (μm)
```

| Bande | Utilité pour la qualité de l'eau |
|-------|----------------------------------|
| **blue** | Pénètre l'eau profonde, aérosols |
| **green** | Turbidité, algues |
| **red** | Chlorophylle, végétation |
| **nir08** | Distingue eau/végétation |
| **swir16** | Humidité du sol |
| **swir22** | Minéraux, types de sol |

### Indices spectraux

| Indice | Formule | Interprétation |
|--------|---------|----------------|
| **NDVI** | (NIR-Red)/(NIR+Red) | Végétation : >0.3 = dense |
| **NDWI** | (Green-NIR)/(Green+NIR) | Eau : >0 = eau |
| **NDMI** | (NIR-SWIR16)/(NIR+SWIR16) | Humidité : >0.4 = humide |
| **MNDWI** | (Green-SWIR16)/(Green+SWIR16) | Surfaces d'eau libre |""")

# =============================================================================
# CELLULE 5 : Fonction d'extraction
# =============================================================================
add_markdown("func-title", """---

## Fonctions d'extraction""")

add_code("func-extract", '''# =============================================================================
# FONCTION D'EXTRACTION GÉNÉRIQUE
# =============================================================================

def extract_landsat(df, bands, max_cloud_cover, n_workers=N_WORKERS):
    """
    Extrait les valeurs Landsat pour les points d'un DataFrame.

    Paramètres :
        df : DataFrame avec 'Latitude', 'Longitude', 'Sample Date'
        bands : Liste des bandes à extraire (ex: ["blue", "red"])
        max_cloud_cover : Seuil de couverture nuageuse max (%)
        n_workers : Nombre de threads parallèles

    Retourne :
        DataFrame avec les valeurs des bandes Landsat
    """

    df = df.copy().reset_index(drop=True)
    results = {band: np.full(len(df), np.nan) for band in bands}

    print(f"Extraction de {bands} pour {len(df)} points")
    print(f"  Seuil nuages: {max_cloud_cover}% | Workers: {n_workers}")
    print("="*60)

    def extract_point(args):
        """Extrait les données pour un point."""
        idx, lat, lon, sample_date = args

        try:
            # Vérifier la date
            if pd.isna(sample_date):
                return idx, {band: np.nan for band in bands}

            if isinstance(sample_date, str):
                sample_date = pd.to_datetime(sample_date, dayfirst=True, errors='coerce')
            if pd.isna(sample_date):
                return idx, {band: np.nan for band in bands}

            # Bounding box (~100m autour du point)
            buffer = 0.001  # ~100m en degrés
            bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]

            # Connexion au catalogue
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace,
            )

            # Recherche des scènes Landsat
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime=DATE_RANGE,
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            )
            items = list(search.item_collection())

            if not items:
                return idx, {band: np.nan for band in bands}

            # Sélectionner la scène la plus proche de la date
            sample_date_utc = sample_date.tz_localize("UTC") if sample_date.tzinfo is None else sample_date
            best_item = min(items, key=lambda x: abs(
                pd.to_datetime(x.properties["datetime"]).tz_convert("UTC") - sample_date_utc
            ))

            # Charger les données
            signed_item = pc.sign(best_item)
            data = stac_load([signed_item], bands=bands, bbox=bbox).isel(time=0)

            # Extraire la médiane pour chaque bande
            result = {}
            for band in bands:
                val = float(data[band].astype("float").median(skipna=True).values)
                result[band] = val if val != 0 else np.nan

            return idx, result

        except Exception:
            return idx, {band: np.nan for band in bands}

    # Préparer les arguments
    args_list = [
        (idx, df.loc[idx, 'Latitude'], df.loc[idx, 'Longitude'], df.loc[idx, 'Sample Date'])
        for idx in df.index
    ]

    # Extraction parallèle
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(extract_point, args): args[0] for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extraction"):
            try:
                idx, point_results = future.result()
                for band in bands:
                    if band in point_results:
                        results[band][idx] = point_results[band]
            except:
                pass

    return pd.DataFrame(results)

print("Fonction extract_landsat() définie!")''')

add_code("func-indices", '''# =============================================================================
# CALCUL DES INDICES SPECTRAUX
# =============================================================================

def compute_spectral_indices(df):
    """
    Calcule les indices spectraux à partir des bandes Landsat.
    Nécessite : nir (ou nir08), green, red, swir16
    """

    df = df.copy()
    eps = 1e-10  # Éviter division par zéro

    # Gérer le nom de la colonne NIR (nir ou nir08)
    nir_col = 'nir' if 'nir' in df.columns else 'nir08'

    # NDVI = (NIR - Red) / (NIR + Red)
    if 'red' in df.columns:
        df['NDVI'] = (df[nir_col] - df['red']) / (df[nir_col] + df['red'] + eps)

    # NDWI = (Green - NIR) / (Green + NIR)
    if 'green' in df.columns:
        df['NDWI'] = (df['green'] - df[nir_col]) / (df['green'] + df[nir_col] + eps)

    # NDMI = (NIR - SWIR16) / (NIR + SWIR16)
    if 'swir16' in df.columns:
        df['NDMI'] = (df[nir_col] - df['swir16']) / (df[nir_col] + df['swir16'] + eps)

    # MNDWI = (Green - SWIR16) / (Green + SWIR16)
    if 'green' in df.columns and 'swir16' in df.columns:
        df['MNDWI'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + eps)

    return df

print("Fonction compute_spectral_indices() définie!")''')

# =============================================================================
# CELLULE 6 : Chargement des données
# =============================================================================
add_markdown("load-title", """---

## Chargement des données existantes""")

add_code("load-data", """# =============================================================================
# CHARGER LES DONNÉES EXISTANTES
# =============================================================================

# Charger le fichier Landsat existant
existing_landsat = pd.read_csv(EXISTING_FILE)

# Charger les données de qualité d'eau (pour avoir les coordonnées et dates)
water_quality = pd.read_csv(WATER_QUALITY_FILE)

print(f"Fichier existant: {len(existing_landsat)} lignes")
print(f"Colonnes existantes: {list(existing_landsat.columns)}")

# Identifier les lignes avec données et sans données
# On vérifie si 'nir' (ou 'green') est NaN pour déterminer si on a des données
has_data_mask = existing_landsat['nir'].notna() if 'nir' in existing_landsat.columns else existing_landsat['green'].notna()

n_with_data = has_data_mask.sum()
n_without_data = (~has_data_mask).sum()

print(f"\\nRépartition:")
print(f"  - Lignes avec données: {n_with_data}")
print(f"  - Lignes sans données (NaN): {n_without_data}")""")

# =============================================================================
# CELLULE 7 : Étape 1
# =============================================================================
add_markdown("step1-title", """---

## ÉTAPE 1 : Extraire blue/red pour les lignes avec données

On extrait seulement les **nouvelles bandes** (blue, red) pour les lignes qui ont déjà des données.
Ensuite on calcule NDVI et NDWI.

⏱️ **Temps estimé** : ~2 heures pour ~8200 points""")

add_code("step1-extract", """# =============================================================================
# ÉTAPE 1 : EXTRACTION blue/red POUR LES LIGNES AVEC DONNÉES
# =============================================================================

# Sélectionner les lignes qui ont déjà des données
df_with_data = water_quality[has_data_mask].copy()
print(f"Étape 1: {len(df_with_data)} lignes à traiter")

# Extraire seulement blue et red
new_bands = extract_landsat(
    df_with_data,
    bands=BANDS_STEP1,
    max_cloud_cover=MAX_CLOUD_COVER_STRICT,
    n_workers=N_WORKERS
)

print(f"\\nExtraction terminée!")
print(f"Valeurs manquantes:")
print(new_bands.isna().sum())""")

add_code("step1-merge", """# =============================================================================
# ÉTAPE 1 : FUSIONNER AVEC LES DONNÉES EXISTANTES
# =============================================================================

# Créer une copie du dataframe existant pour les lignes avec données
step1_result = existing_landsat[has_data_mask].copy().reset_index(drop=True)

# Ajouter les nouvelles colonnes blue et red
step1_result['blue'] = new_bands['blue'].values
step1_result['red'] = new_bands['red'].values

# Calculer NDVI et NDWI (on a maintenant red et nir)
eps = 1e-10
step1_result['NDVI'] = (step1_result['nir'] - step1_result['red']) / (step1_result['nir'] + step1_result['red'] + eps)
step1_result['NDWI'] = (step1_result['green'] - step1_result['nir']) / (step1_result['green'] + step1_result['nir'] + eps)

print(f"Étape 1 terminée: {len(step1_result)} lignes")
print(f"Colonnes: {list(step1_result.columns)}")
display(step1_result.head())""")

# =============================================================================
# CELLULE 8 : Étape 2
# =============================================================================
add_markdown("step2-title", """---

## ÉTAPE 2 : Extraire TOUTES les bandes pour les lignes NaN

On extrait **toutes les bandes** pour les lignes qui n'ont pas de données.
On utilise un seuil de nuages plus tolérant (30%) pour avoir plus de chances de trouver des images.

⏱️ **Temps estimé** : ~20 minutes pour ~1000 points""")

add_code("step2-extract", """# =============================================================================
# ÉTAPE 2 : EXTRACTION COMPLÈTE POUR LES LIGNES NaN
# =============================================================================

# Sélectionner les lignes sans données
df_without_data = water_quality[~has_data_mask].copy()
print(f"Étape 2: {len(df_without_data)} lignes à traiter")

# Extraire toutes les bandes avec seuil relaxé
all_bands_result = extract_landsat(
    df_without_data,
    bands=ALL_BANDS,
    max_cloud_cover=MAX_CLOUD_COVER_RELAXED,
    n_workers=N_WORKERS
)

# Renommer nir08 -> nir
all_bands_result = all_bands_result.rename(columns={'nir08': 'nir'})

# Calculer les indices spectraux
all_bands_result = compute_spectral_indices(all_bands_result)

print(f"\\nExtraction terminée!")
print(f"Valeurs manquantes:")
print(all_bands_result.isna().sum())""")

add_code("step2-prepare", """# =============================================================================
# ÉTAPE 2 : PRÉPARER LE DATAFRAME
# =============================================================================

# Créer le dataframe pour l'étape 2 avec les mêmes colonnes que l'étape 1
step2_result = pd.DataFrame({
    'Latitude': df_without_data['Latitude'].values,
    'Longitude': df_without_data['Longitude'].values,
    'Sample Date': df_without_data['Sample Date'].values,
})

# Ajouter les colonnes Landsat dans le bon ordre
landsat_columns = ['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI', 'blue', 'red', 'NDVI', 'NDWI']
for col in landsat_columns:
    if col in all_bands_result.columns:
        step2_result[col] = all_bands_result[col].values
    else:
        step2_result[col] = np.nan

print(f"Étape 2 terminée: {len(step2_result)} lignes")
display(step2_result.head())""")

# =============================================================================
# CELLULE 9 : Étape 3
# =============================================================================
add_markdown("step3-title", """---

## ÉTAPE 3 : Fusion finale

On fusionne les résultats des étapes 1 et 2 pour créer le fichier final.""")

add_code("step3-merge", """# =============================================================================
# ÉTAPE 3 : FUSION FINALE
# =============================================================================

# S'assurer que step1_result a les coordonnées
if 'Latitude' not in step1_result.columns:
    step1_result.insert(0, 'Latitude', df_with_data['Latitude'].values)
    step1_result.insert(1, 'Longitude', df_with_data['Longitude'].values)
    step1_result.insert(2, 'Sample Date', df_with_data['Sample Date'].values)

# Définir l'ordre des colonnes
final_columns = ['Latitude', 'Longitude', 'Sample Date',
                 'blue', 'green', 'red', 'nir', 'swir16', 'swir22',
                 'NDVI', 'NDWI', 'NDMI', 'MNDWI']

# Réordonner les colonnes de step1_result
step1_final = step1_result.reindex(columns=final_columns)

# Réordonner les colonnes de step2_result
step2_final = step2_result.reindex(columns=final_columns)

# Concaténer les deux DataFrames
final_df = pd.concat([step1_final, step2_final], ignore_index=True)

print(f"Dataset final: {len(final_df)} lignes")
print(f"Colonnes: {list(final_df.columns)}")
print(f"\\nValeurs manquantes:")
print(final_df.isna().sum())""")

add_code("step3-save", """# =============================================================================
# SAUVEGARDE
# =============================================================================

# Sauvegarder le fichier final
output_path = os.path.join(OUTPUT_DIR, 'landsat_features_training_complete.csv')
final_df.to_csv(output_path, index=False)

print(f"Fichier sauvegardé : {output_path}")
print(f"  - {len(final_df)} lignes")
print(f"  - {len(final_df.columns)} colonnes")

# Statistiques finales
print(f"\\n" + "="*60)
print("RÉSUMÉ FINAL")
print("="*60)
print(f"Total lignes: {len(final_df)}")
print(f"\\nTaux de complétion par variable:")
for col in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'NDVI', 'NDWI', 'NDMI', 'MNDWI']:
    if col in final_df.columns:
        taux = (1 - final_df[col].isna().mean()) * 100
        print(f"  {col}: {taux:.1f}%")

display(final_df.head(10))""")

# =============================================================================
# CELLULE 10 : Validation (optionnel)
# =============================================================================
add_markdown("validation-title", """---

## (Optionnel) Extraction pour les données de validation""")

add_code("validation-extract", """# =============================================================================
# EXTRACTION VALIDATION (si nécessaire)
# =============================================================================

# Décommenter pour exécuter

# Validation_df = pd.read_csv('../data/raw/submission_template.csv')
# print(f"Validation : {len(Validation_df)} lignes")

# # Extraire toutes les bandes
# val_features = extract_landsat(
#     Validation_df,
#     bands=ALL_BANDS,
#     max_cloud_cover=MAX_CLOUD_COVER_RELAXED,
#     n_workers=N_WORKERS
# )
# val_features = val_features.rename(columns={'nir08': 'nir'})
# val_features = compute_spectral_indices(val_features)

# # Créer et sauvegarder
# landsat_val_df = pd.DataFrame({
#     'Latitude': Validation_df['Latitude'].values,
#     'Longitude': Validation_df['Longitude'].values,
#     'Sample Date': Validation_df['Sample Date'].values,
# })
# for col in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'NDVI', 'NDWI', 'NDMI', 'MNDWI']:
#     if col in val_features.columns:
#         landsat_val_df[col] = val_features[col].values

# output_path = os.path.join(OUTPUT_DIR, 'landsat_features_validation_complete.csv')
# landsat_val_df.to_csv(output_path, index=False)
# print(f"Fichier sauvegardé : {output_path}")""")

# =============================================================================
# CELLULE 11 : Résumé
# =============================================================================
add_markdown("summary", """---

## Résumé

### Stratégie appliquée

| Étape | Lignes | Bandes extraites | Seuil nuages |
|-------|--------|------------------|--------------|
| **1** | ~8234 | blue, red | 10% |
| **2** | ~1085 | TOUTES | 30% |
| **3** | ~9319 | Fusion | - |

### Variables finales (10 au total)

| Type | Variables |
|------|-----------|
| **Bandes (6)** | blue, green, red, nir, swir16, swir22 |
| **Indices (4)** | NDVI, NDWI, NDMI, MNDWI |

### Fichier créé

`landsat_features_training_complete.csv` : Dataset complet avec 10 variables Landsat""")

# Sauvegarder le notebook
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks', '06_LANDSAT_DATA_EXTRACTION.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook créé : {output_path}")
print(f"  - {len(notebook['cells'])} cellules")
