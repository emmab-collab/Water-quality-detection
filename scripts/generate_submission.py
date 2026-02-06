# -*- coding: utf-8 -*-
"""
Génération de la soumission pour le challenge EY.
Entraînement sur Training + GEMStat (51 features communes).
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from src.config import TARGETS, RANDOM_SEED

# =============================================================================
# FEATURES COMMUNES
# =============================================================================

COMMON_FEATURES = [
    # DEM (3)
    'elevation', 'slope', 'aspect',
    # WorldCover (8)
    'lc_tree', 'lc_shrubland', 'lc_grassland', 'lc_cropland',
    'lc_builtup', 'lc_bare', 'lc_water', 'lc_wetland',
    # SoilGrids (6)
    'soil_ph', 'soil_clay', 'soil_sand', 'soil_soc', 'soil_cec', 'soil_nitrogen',
    # TerraClimate (34)
    'pet', 'aet', 'ppt', 'ppt_lag1', 'ppt_lag2', 'ppt_lag3', 'ppt_sum4', 'ppt_mean4', 'ppt_anomaly',
    'tmax', 'tmin',
    'soil', 'soil_lag1', 'soil_lag2', 'soil_lag3', 'soil_sum4', 'soil_mean4', 'soil_anomaly',
    'def', 'def_lag1', 'def_lag2', 'def_lag3', 'def_sum4', 'def_mean4', 'def_anomaly',
    'pdsi',
    'vpd', 'vpd_lag1', 'vpd_lag2', 'vpd_lag3', 'vpd_sum4', 'vpd_mean4', 'vpd_anomaly',
    'ws'
]

print(f"Features: {len(COMMON_FEATURES)}")

# =============================================================================
# CHARGEMENT
# =============================================================================

print("\nChargement des données...")

train_df = pd.read_csv('data/processed/merged_training.csv')
gemstat_df = pd.read_csv('data/processed/gemstat_features.csv')
val_df = pd.read_csv('data/processed/merged_validation.csv')
template = pd.read_csv('data/raw/submission_template.csv')

print(f"Training: {len(train_df)}")
print(f"GEMStat: {len(gemstat_df)}")
print(f"Validation: {len(val_df)}")

# =============================================================================
# COMBINAISON TRAINING + GEMSTAT
# =============================================================================

combined_df = pd.concat([
    train_df[COMMON_FEATURES + TARGETS],
    gemstat_df[COMMON_FEATURES + TARGETS]
], ignore_index=True)

print(f"Combiné: {len(combined_df)}")

# =============================================================================
# PRÉPARATION
# =============================================================================

X_train = combined_df[COMMON_FEATURES].fillna(combined_df[COMMON_FEATURES].median())
y_train = combined_df[TARGETS]

X_val = val_df[COMMON_FEATURES].fillna(val_df[COMMON_FEATURES].median())

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# =============================================================================
# ENTRAÎNEMENT ET PRÉDICTION
# =============================================================================

print("\nEntraînement des modèles...")

predictions = {}
for target in TARGETS:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train[target])
    predictions[target] = model.predict(X_val_scaled)
    print(f"  {target}: OK")

# =============================================================================
# GÉNÉRATION SOUMISSION
# =============================================================================

submission = template.copy()
for target in TARGETS:
    submission[target] = predictions[target]

# Vérification
print("\nSoumission:")
print(submission.head())
print(f"\nShape: {submission.shape}")

# Stats des prédictions
print("\nStatistiques des prédictions:")
for target in TARGETS:
    print(f"  {target}: mean={predictions[target].mean():.2f}, min={predictions[target].min():.2f}, max={predictions[target].max():.2f}")

# Sauvegarde
output_path = 'data/submissions/submission_common_features.csv'
import os
os.makedirs('data/submissions', exist_ok=True)
submission.to_csv(output_path, index=False)
print(f"\nSoumission sauvegardée: {output_path}")
