# -*- coding: utf-8 -*-
"""
Entraînement sur données combinées (Training + GEMStat).
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from sklearn.model_selection import cross_val_score, KFold
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

print(f"Features communes: {len(COMMON_FEATURES)}")

# =============================================================================
# CHARGEMENT ET COMBINAISON
# =============================================================================

print("\n" + "="*60)
print("CHARGEMENT DES DONNÉES")
print("="*60)

train_df = pd.read_csv('data/processed/merged_training.csv')
gemstat_df = pd.read_csv('data/processed/gemstat_features.csv')

print(f"Training: {len(train_df)}")
print(f"GEMStat: {len(gemstat_df)}")

# Ajouter source
train_df['source'] = 'training'
gemstat_df['source'] = 'gemstat'

# Combiner
combined_df = pd.concat([
    train_df[COMMON_FEATURES + TARGETS + ['source']],
    gemstat_df[COMMON_FEATURES + TARGETS + ['source']]
], ignore_index=True)

print(f"Combiné: {len(combined_df)}")

# Stats par source
print("\nDistribution par source:")
print(combined_df['source'].value_counts())

# =============================================================================
# PRÉPARATION
# =============================================================================

X = combined_df[COMMON_FEATURES].fillna(combined_df[COMMON_FEATURES].median())
y = combined_df[TARGETS]

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# CROSS-VALIDATION SUR DONNÉES COMBINÉES
# =============================================================================

print("\n" + "="*60)
print("CROSS-VALIDATION (5-fold) SUR DONNÉES COMBINÉES")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for target in TARGETS:
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    scores = cross_val_score(model, X_scaled, y[target], cv=kf, scoring='r2')
    print(f"\n{target}:")
    print(f"  R² scores: {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean R²: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# =============================================================================
# TRAIN SUR TRAINING, TEST SUR GEMSTAT
# =============================================================================

print("\n" + "="*60)
print("TRAIN SUR TRAINING SEUL -> TEST SUR GEMSTAT")
print("="*60)

X_train = train_df[COMMON_FEATURES].fillna(train_df[COMMON_FEATURES].median())
y_train = train_df[TARGETS]
X_gemstat = gemstat_df[COMMON_FEATURES]
y_gemstat = gemstat_df[TARGETS]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_gemstat_scaled = scaler.transform(X_gemstat)

for target in TARGETS:
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train_scaled, y_train[target])

    from sklearn.metrics import r2_score, mean_squared_error
    y_pred_train = model.predict(X_train_scaled)
    y_pred_gemstat = model.predict(X_gemstat_scaled)

    print(f"\n{target}:")
    print(f"  Train R²: {r2_score(y_train[target], y_pred_train):.3f}")
    print(f"  GEMStat R²: {r2_score(y_gemstat[target], y_pred_gemstat):.3f}")

# =============================================================================
# TRAIN SUR COMBINÉ -> VALIDATION CROISÉE STRATIFIÉE PAR SOURCE
# =============================================================================

print("\n" + "="*60)
print("LEAVE-ONE-SOURCE-OUT: Train sur Training, Test sur GEMStat")
print("="*60)

# Train sur training only, test sur gemstat
X_train_only = combined_df[combined_df['source'] == 'training'][COMMON_FEATURES]
y_train_only = combined_df[combined_df['source'] == 'training'][TARGETS]
X_gemstat_only = combined_df[combined_df['source'] == 'gemstat'][COMMON_FEATURES]
y_gemstat_only = combined_df[combined_df['source'] == 'gemstat'][TARGETS]

X_train_only = X_train_only.fillna(X_train_only.median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_only)
X_gemstat_scaled = scaler.transform(X_gemstat_only)

for target in TARGETS:
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train_scaled, y_train_only[target])

    y_pred = model.predict(X_gemstat_scaled)
    r2 = r2_score(y_gemstat_only[target], y_pred)
    rmse = np.sqrt(mean_squared_error(y_gemstat_only[target], y_pred))

    print(f"{target}: R²={r2:.3f}, RMSE={rmse:.2f}")

# =============================================================================
# APPROCHE INVERSE: Train sur GEMStat, Test sur Training
# =============================================================================

print("\n" + "="*60)
print("INVERSE: Train sur GEMStat, Test sur Training")
print("="*60)

scaler = StandardScaler()
X_gemstat_scaled = scaler.fit_transform(X_gemstat_only)
X_train_scaled = scaler.transform(X_train_only)

for target in TARGETS:
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_gemstat_scaled, y_gemstat_only[target])

    y_pred = model.predict(X_train_scaled)
    r2 = r2_score(y_train_only[target], y_pred)
    rmse = np.sqrt(mean_squared_error(y_train_only[target], y_pred))

    print(f"{target}: R²={r2:.3f}, RMSE={rmse:.2f}")

# =============================================================================
# TRAIN SUR COMBINÉ, CV MIXTE
# =============================================================================

print("\n" + "="*60)
print("ENTRAÎNEMENT SUR DONNÉES COMBINÉES (pour soumission)")
print("="*60)

X_combined = combined_df[COMMON_FEATURES].fillna(combined_df[COMMON_FEATURES].median())
y_combined = combined_df[TARGETS]

scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

models = {}
for target in TARGETS:
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_combined_scaled, y_combined[target])
    models[target] = model

    y_pred = model.predict(X_combined_scaled)
    r2 = r2_score(y_combined[target], y_pred)
    print(f"{target}: Train R² = {r2:.3f}")

print("\nModèles entraînés sur données combinées (Training + GEMStat).")
print(f"Total échantillons: {len(combined_df)}")
