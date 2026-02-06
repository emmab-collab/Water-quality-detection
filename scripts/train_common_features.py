# -*- coding: utf-8 -*-
"""
Entraînement sur les 51 features communes (Training + GEMStat).

Option 1: Sans Landsat, water_type, distance_to_river
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.config import TARGETS, RANDOM_SEED
from src.models.train import split_data, normalize, train_models, evaluate, print_results, get_feature_importance

# =============================================================================
# FEATURES COMMUNES (disponibles dans Training ET GEMStat)
# =============================================================================

# DEM features (3)
DEM_FEATURES = ['elevation', 'slope', 'aspect']

# WorldCover features (8)
WORLDCOVER_FEATURES = [
    'lc_tree', 'lc_shrubland', 'lc_grassland', 'lc_cropland',
    'lc_builtup', 'lc_bare', 'lc_water', 'lc_wetland'
]

# SoilGrids features (6)
SOILGRIDS_FEATURES = [
    'soil_ph', 'soil_clay', 'soil_sand', 'soil_soc', 'soil_cec', 'soil_nitrogen'
]

# TerraClimate features (34)
TERRACLIMATE_FEATURES = [
    'pet', 'aet', 'ppt', 'ppt_lag1', 'ppt_lag2', 'ppt_lag3', 'ppt_sum4', 'ppt_mean4', 'ppt_anomaly',
    'tmax', 'tmin',
    'soil', 'soil_lag1', 'soil_lag2', 'soil_lag3', 'soil_sum4', 'soil_mean4', 'soil_anomaly',
    'def', 'def_lag1', 'def_lag2', 'def_lag3', 'def_sum4', 'def_mean4', 'def_anomaly',
    'pdsi',
    'vpd', 'vpd_lag1', 'vpd_lag2', 'vpd_lag3', 'vpd_sum4', 'vpd_mean4', 'vpd_anomaly',
    'ws'
]

# Toutes les features communes (51)
COMMON_FEATURES = DEM_FEATURES + WORLDCOVER_FEATURES + SOILGRIDS_FEATURES + TERRACLIMATE_FEATURES

print(f"Features communes: {len(COMMON_FEATURES)}")

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

print("\n" + "="*60)
print("CHARGEMENT DES DONNÉES")
print("="*60)

# Training data
train_df = pd.read_csv('data/processed/merged_training.csv')
print(f"Training: {len(train_df)} échantillons")

# Validation data (EY)
val_df = pd.read_csv('data/processed/merged_validation.csv')
print(f"Validation EY: {len(val_df)} échantillons")

# GEMStat data
gemstat_df = pd.read_csv('data/processed/gemstat_features.csv')
print(f"GEMStat: {len(gemstat_df)} échantillons")

# Vérifier que les features communes sont disponibles
missing_train = [f for f in COMMON_FEATURES if f not in train_df.columns]
missing_gemstat = [f for f in COMMON_FEATURES if f not in gemstat_df.columns]

if missing_train:
    print(f"ERREUR: Features manquantes dans Training: {missing_train}")
if missing_gemstat:
    print(f"ERREUR: Features manquantes dans GEMStat: {missing_gemstat}")

# =============================================================================
# PRÉPARATION DES DONNÉES
# =============================================================================

print("\n" + "="*60)
print("PRÉPARATION DES DONNÉES")
print("="*60)

# Features et targets
X_train_full = train_df[COMMON_FEATURES]
y_train_full = train_df[TARGETS]

X_gemstat = gemstat_df[COMMON_FEATURES]
y_gemstat = gemstat_df[TARGETS]

# Gérer les NaN
print(f"\nNaN dans X_train: {X_train_full.isna().sum().sum()}")
print(f"NaN dans X_gemstat: {X_gemstat.isna().sum().sum()}")

# Remplacer NaN par la médiane
X_train_full = X_train_full.fillna(X_train_full.median())
X_gemstat = X_gemstat.fillna(X_gemstat.median())

# Split train/val/test
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X_train_full, y_train_full, test_size=0.2, val_size=0.2
)

# Normalisation
X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize(X_train, X_val, X_test)
X_gemstat_scaled = scaler.transform(X_gemstat)

# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

print("\n" + "="*60)
print("ENTRAÎNEMENT DU MODÈLE")
print("="*60)

models = train_models(X_train_scaled, y_train, max_depth=10, min_samples_leaf=1)

# =============================================================================
# ÉVALUATION
# =============================================================================

print("\n" + "="*60)
print("ÉVALUATION")
print("="*60)

# Sur les données de training
train_results = evaluate(models, X_train_scaled, y_train)
print_results(train_results, "TRAIN (Training Data)")

# Sur les données de validation
val_results = evaluate(models, X_val_scaled, y_val)
print_results(val_results, "VALIDATION (Training Data)")

# Sur les données de test
test_results = evaluate(models, X_test_scaled, y_test)
print_results(test_results, "TEST (Training Data)")

# Sur GEMStat
gemstat_results = evaluate(models, X_gemstat_scaled, y_gemstat)
print_results(gemstat_results, "GEMSTAT (External Validation)")

# =============================================================================
# IMPORTANCE DES FEATURES
# =============================================================================

print("\n" + "="*60)
print("TOP 10 FEATURES PAR TARGET")
print("="*60)

importance = get_feature_importance(models, COMMON_FEATURES, top_n=10)
for target, df in importance.items():
    print(f"\n{target}:")
    for i, row in df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

# =============================================================================
# RÉSUMÉ
# =============================================================================

print("\n" + "="*60)
print("RÉSUMÉ")
print("="*60)

print(f"\nFeatures utilisées: {len(COMMON_FEATURES)} (sans Landsat, water_type, distance_to_river)")
print(f"Données d'entraînement: {len(train_df)} échantillons")
print(f"Données GEMStat: {len(gemstat_df)} échantillons")

print("\nComparaison Test vs GEMStat:")
print("-" * 50)
for target in TARGETS:
    test_r2 = test_results[target]['R2']
    gemstat_r2 = gemstat_results[target]['R2']
    diff = gemstat_r2 - test_r2
    print(f"{target}:")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  GEMStat R²: {gemstat_r2:.3f}")
    print(f"  Diff: {diff:+.3f}")
