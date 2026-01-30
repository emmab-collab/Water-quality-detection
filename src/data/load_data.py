# -*- coding: utf-8 -*-
"""
Fonctions de chargement des données - EY Water Quality Challenge
================================================================

Fichier simple pour charger et préparer les données du challenge.
"""

import pandas as pd
import numpy as np


# =============================================================================
# FONCTIONS DE CHARGEMENT
# =============================================================================

def load_water_quality(filepath):
    """
    Charge le fichier principal avec les mesures de qualité de l'eau.

    Ce fichier contient:
    - Latitude, Longitude : où l'échantillon a été prélevé
    - Sample Date : quand
    - Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus : les 3 targets
    """
    df = pd.read_csv(filepath)
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    print(f"[OK] Water quality: {len(df)} lignes")
    return df


def load_landsat(filepath):
    """
    Charge les données satellite Landsat.

    Contient les bandes spectrales (nir, green, swir16, swir22)
    et les indices calculés (NDMI, MNDWI).
    """
    df = pd.read_csv(filepath)

    # Convertir la date en format datetime (pour pouvoir fusionner avec les autres fichiers)
    if 'Sample Date' in df.columns:
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')

    # Convertir en nombres (parfois ce sont des strings)
    for col in ['NDMI', 'MNDWI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"[OK] Landsat: {len(df)} lignes")
    return df


def load_terraclimate(filepath):
    """
    Charge les données climatiques TerraClimate.

    Contient pet (évapotranspiration potentielle) et potentiellement
    d'autres variables climatiques.
    """
    df = pd.read_csv(filepath)

    # Convertir la date en format datetime (pour pouvoir fusionner avec les autres fichiers)
    if 'Sample Date' in df.columns:
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')

    print(f"[OK] TerraClimate: {len(df)} lignes")
    return df


# =============================================================================
# FUSION DES DONNÉES
# =============================================================================

def merge_all_data(water_quality_df, landsat_df, terraclimate_df):
    """
    Fusionne les 3 fichiers en un seul DataFrame.

    On fusionne sur les colonnes communes : Latitude, Longitude, Sample Date.
    C'est plus sûr que de coller les fichiers côte à côte.
    """
    # Colonnes de jointure
    join_cols = ['Latitude', 'Longitude', 'Sample Date']

    # Première fusion : water_quality + landsat
    df = pd.merge(
        water_quality_df,
        landsat_df,
        on=join_cols,
        how='inner'
    )

    # Deuxième fusion : résultat + terraclimate
    df = pd.merge(
        df,
        terraclimate_df,
        on=join_cols,
        how='inner'
    )

    print(f"[OK] Données fusionnées: {len(df)} lignes, {len(df.columns)} colonnes")
    return df


# =============================================================================
# NETTOYAGE
# =============================================================================

def check_missing(df):
    """
    Affiche un résumé des valeurs manquantes.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]  # Garder seulement les colonnes avec des NaN

    if len(missing) == 0:
        print("Aucune valeur manquante!")
    else:
        print(f"\n{len(missing)} colonnes avec des valeurs manquantes:")
        for col in missing.index:
            pct = 100 * missing[col] / len(df)
            print(f"  - {col}: {missing[col]} manquants ({pct:.1f}%)")

    return missing


def fill_missing(df):
    """
    Remplit les valeurs manquantes avec la médiane de chaque colonne.

    Pourquoi la médiane? Elle est moins sensible aux valeurs extrêmes
    que la moyenne.
    """
    df = df.copy()  # Ne pas modifier l'original

    # Colonnes numériques seulement
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Remplir avec la médiane
    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    print("[OK] Valeurs manquantes remplies avec la médiane")
    return df


def detect_outliers_iqr(series):
    """
    Détecte les outliers avec la méthode IQR (Interquartile Range).

    Méthode :
    - Q1 = 25e percentile, Q3 = 75e percentile
    - IQR = Q3 - Q1
    - Outlier si : valeur < Q1 - 1.5×IQR  ou  valeur > Q3 + 1.5×IQR

    Retourne un dict avec : total, pct, low, high, lower_bound, upper_bound
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_low = (series < lower_bound).sum()
    outliers_high = (series > upper_bound).sum()
    total = outliers_low + outliers_high

    return {
        'total': total,
        'pct': 100 * total / len(series),
        'low': outliers_low,
        'high': outliers_high,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def get_outliers_summary(df, columns):
    """
    Crée un résumé des outliers pour plusieurs colonnes.

    Retourne un DataFrame avec le résumé des outliers par colonne.
    """
    results = []

    for col in columns:
        data = df[col].dropna()
        result = detect_outliers_iqr(data)
        results.append({
            'Variable': col,
            'Outliers': result['total'],
            'Pourcentage': round(result['pct'], 1),
            'Seuil_bas': round(result['lower_bound'], 2),
            'Seuil_haut': round(result['upper_bound'], 2)
        })

    return pd.DataFrame(results).sort_values('Pourcentage', ascending=False)


# =============================================================================
# PRÉPARATION POUR LE MODÈLE
# =============================================================================

def get_X_y(df, features=None):
    """
    Sépare les features (X) et les targets (y).

    Par défaut, utilise les mêmes features que le benchmark:
    - swir22, NDMI, MNDWI, pet

    Les targets sont toujours les 3 paramètres de qualité de l'eau.
    """
    # Features par défaut (comme le benchmark)
    if features is None:
        features = ['swir22', 'NDMI', 'MNDWI', 'pet']

    # Les 3 targets du challenge
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

    X = df[features]
    y = df[targets]

    print(f"[OK] X: {X.shape[0]} lignes, {X.shape[1]} features")
    print(f"[OK] y: {y.shape[0]} lignes, {y.shape[1]} targets")

    return X, y


def get_site_ids(df):
    """
    Crée un identifiant unique pour chaque site (combinaison lat/lon).

    C'est utile pour la validation spatiale: on veut s'assurer que
    les sites du test ne sont pas dans le train.
    """
    site_ids = df['Latitude'].round(4).astype(str) + "_" + df['Longitude'].round(4).astype(str)
    print(f"[OK] {site_ids.nunique()} sites uniques identifiés")
    return site_ids


# =============================================================================
# FONCTION PRINCIPALE (TOUT-EN-UN)
# =============================================================================

def load_all(wq_path, landsat_path, terra_path, features=None, fill_na=False):
    """
    Charge et prépare toutes les données en une seule fois.

    Paramètres:
    -----------
    wq_path : chemin vers water_quality_training_dataset.csv
    landsat_path : chemin vers landsat_features_training.csv
    terra_path : chemin vers terraclimate_features_training.csv
    features : liste des colonnes à utiliser (optionnel)
    fill_na : si True, remplit les valeurs manquantes avec la médiane (défaut: False)

    Retourne:
    ---------
    X : les features pour le modèle
    y : les targets à prédire
    site_ids : identifiant unique par site
    df : le DataFrame complet (si tu veux explorer)

    Exemple:
    --------
    X, y, sites, df = load_all(
        "chemin/water_quality_training_dataset.csv",
        "chemin/landsat_features_training.csv",
        "chemin/terraclimate_features_training.csv"
    )
    """
    print("="*50)
    print("CHARGEMENT DES DONNÉES")
    print("="*50)

    # 1. Charger les 3 fichiers
    wq = load_water_quality(wq_path)
    landsat = load_landsat(landsat_path)
    terra = load_terraclimate(terra_path)

    # 2. Fusionner
    df = merge_all_data(wq, landsat, terra)

    # 3. Afficher les valeurs manquantes
    print("\nValeurs manquantes:")
    check_missing(df)

    # 4. Remplir les valeurs manquantes seulement si demandé
    if fill_na:
        df = fill_missing(df)

    # 5. Créer les IDs de site
    site_ids = get_site_ids(df)

    # 6. Séparer X et y
    X, y = get_X_y(df, features)

    print("\n" + "="*50)
    print("PRÊT!")
    print("="*50)

    return X, y, site_ids, df


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Ce code s'exécute seulement si tu lances ce fichier directement
    print("Pour utiliser ce module:")
    print("")
    print("from src.data.load_data import load_all")
    print("")
    print('X, y, sites, df = load_all(')
    print('    "Snowflake Notebooks Package/water_quality_training_dataset.csv",')
    print('    "Snowflake Notebooks Package/landsat_features_training.csv",')
    print('    "Snowflake Notebooks Package/terraclimate_features_training.csv"')
    print(')')
