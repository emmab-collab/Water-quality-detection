# -*- coding: utf-8 -*-
"""
Feature Engineering - Nettoyage et création de features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

from src.config import ALL_FEATURES, LANDSAT_BANDS


# =============================================================================
# CONSTANTES
# =============================================================================

SATURATION_VALUE = 65535


# =============================================================================
# NETTOYAGE
# =============================================================================

def remove_missing_rows(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Supprime les lignes avec des valeurs manquantes dans les colonnes spécifiées."""
    return df.dropna(subset=columns)


def remove_saturated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes avec des valeurs Landsat saturées (65535)."""
    mask = (df[LANDSAT_BANDS] == SATURATION_VALUE).any(axis=1)
    return df[~mask]


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données training.
    1. Supprime les lignes avec NaN
    2. Supprime les lignes avec valeurs saturées
    """
    df = df.copy()
    n_initial = len(df)

    df = remove_missing_rows(df, ALL_FEATURES)
    df = remove_saturated_rows(df)

    print(f"Nettoyage: {n_initial} -> {len(df)} lignes")
    return df


# =============================================================================
# IMPUTATION
# =============================================================================

def compute_medians(df: pd.DataFrame) -> pd.Series:
    """Calcule les médianes des features."""
    return df[ALL_FEATURES].median()


def impute_with_medians(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Remplace les NaN par les médianes."""
    df = df.copy()
    for col in ALL_FEATURES:
        if col in df.columns and col in medians.index:
            df[col] = df[col].fillna(medians[col])
    return df


# =============================================================================
# CREATION DE FEATURES
# =============================================================================

def get_season(month: int) -> str:
    """Retourne la saison (hémisphère sud)."""
    if month in [12, 1, 2]:
        return 'summer'
    elif month in [3, 4, 5]:
        return 'autumn'
    elif month in [6, 7, 8]:
        return 'winter'
    else:
        return 'spring'


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute day_of_year et season."""
    df = df.copy()
    df['day_of_year'] = df['Sample Date'].dt.dayofyear
    df['season'] = df['Sample Date'].dt.month.apply(get_season)
    return df


def add_spectral_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute nir_green_ratio et swir_ratio."""
    df = df.copy()
    df['nir_green_ratio'] = df['nir'] / (df['green'] + 1e-6)
    df['swir_ratio'] = df['swir16'] / (df['swir22'] + 1e-6)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée toutes les nouvelles features."""
    df = add_temporal_features(df)
    df = add_spectral_ratios(df)
    return df


# =============================================================================
# ONE-HOT ENCODING
# =============================================================================

def encode_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode la colonne season.
    Supprime 'autumn' (référence) et la colonne 'season' originale.
    Résultat: season_spring, season_summer, season_winter
    """
    df = df.copy()
    dummies = pd.get_dummies(df['season'], prefix='season', drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=['season'])
    return df


# =============================================================================
# SELECTION DES FEATURES
# =============================================================================

# Liste des features numériques créées
CREATED_FEATURES = ['day_of_year', 'nir_green_ratio', 'swir_ratio']

# Liste des colonnes season encodées (autumn supprimée)
SEASON_ENCODED = ['season_spring', 'season_summer', 'season_winter']

# Toutes les features pour le modèle
MODEL_FEATURES = ALL_FEATURES + CREATED_FEATURES + SEASON_ENCODED


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne uniquement les colonnes pour le modèle."""
    available = [col for col in MODEL_FEATURES if col in df.columns]
    return df[available]


# =============================================================================
# PIPELINES
# =============================================================================

def prepare_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline training:
    1. Nettoie (supprime NaN et saturées)
    2. Calcule les médianes
    3. Crée les features
    4. Encode season

    Retourne: (df_prepared, medians)
    """
    df = clean_training_data(df)
    medians = compute_medians(df)
    df = create_features(df)
    df = encode_season(df)
    return df, medians


def prepare_submission(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """
    Pipeline submission:
    1. Impute les NaN avec les médianes
    2. Crée les features
    3. Encode season
    """
    df = impute_with_medians(df, medians)
    df = create_features(df)
    df = encode_season(df)
    return df
