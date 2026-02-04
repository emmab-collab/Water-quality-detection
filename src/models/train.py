# -*- coding: utf-8 -*-
"""
Modélisation - Entraînement et évaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.config import TARGETS, RANDOM_SEED


# =============================================================================
# SPLIT
# =============================================================================

def split_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2
) -> Tuple:
    """
    Sépare les données en train/val/test.

    Retourne: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Train+Val vs Test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    # Train vs Val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=RANDOM_SEED
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# NORMALISATION
# =============================================================================

def normalize(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalise les features (fit sur train uniquement).

    Retourne: X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# =============================================================================
# ENTRAINEMENT
# =============================================================================

def train_models(
    X_train: np.ndarray,
    y_train: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_leaf: int = 1
) -> Dict[str, RandomForestRegressor]:
    """
    Entraîne un Random Forest pour chaque target.

    Retourne: {target_name: model}
    """
    models = {}
    for target in TARGETS:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        model.fit(X_train, y_train[target])
        models[target] = model
    return models


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    models: Dict[str, RandomForestRegressor],
    X: np.ndarray,
    y: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Évalue les modèles.

    Retourne: {target: {'R2': float, 'RMSE': float}}
    """
    results = {}
    for target in TARGETS:
        y_pred = models[target].predict(X)
        y_true = y[target]
        results[target] = {
            'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    return results


def print_results(results: Dict[str, Dict[str, float]], name: str = ""):
    """Affiche les résultats."""
    print(f"\n{name}")
    print("-" * 40)
    for target, metrics in results.items():
        print(f"{target}: R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.2f}")


# =============================================================================
# IMPORTANCE DES FEATURES
# =============================================================================

def get_feature_importance(
    models: Dict[str, RandomForestRegressor],
    feature_names: list,
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Retourne l'importance des features pour chaque modèle.

    Retourne: {target: DataFrame[feature, importance]}
    """
    result = {}
    for target in TARGETS:
        imp = models[target].feature_importances_
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': imp
        }).sort_values('importance', ascending=False).head(top_n)
        result[target] = df
    return result


# =============================================================================
# PREDICTION
# =============================================================================

def predict(
    models: Dict[str, RandomForestRegressor],
    X: np.ndarray
) -> pd.DataFrame:
    """Génère les prédictions pour toutes les targets."""
    predictions = {}
    for target in TARGETS:
        predictions[target] = models[target].predict(X)
    return pd.DataFrame(predictions)


# =============================================================================
# TUNING
# =============================================================================

def test_hyperparameters(
    X_train: np.ndarray,
    y_train: pd.DataFrame,
    X_val: np.ndarray,
    y_val: pd.DataFrame,
    max_depths: list = [3, 5, 7, 10, 15],
    min_samples_leafs: list = [1, 5, 10]
) -> pd.DataFrame:
    """
    Teste différentes combinaisons d'hyperparamètres.

    Retourne un DataFrame avec les résultats.
    """
    results = []

    for depth in max_depths:
        for min_leaf in min_samples_leafs:
            models = train_models(X_train, y_train, max_depth=depth, min_samples_leaf=min_leaf)

            train_res = evaluate(models, X_train, y_train)
            val_res = evaluate(models, X_val, y_val)

            for target in TARGETS:
                train_r2 = train_res[target]['R2']
                val_r2 = val_res[target]['R2']
                gap = train_r2 - val_r2

                results.append({
                    'target': target,
                    'max_depth': depth,
                    'min_samples_leaf': min_leaf,
                    'train_R2': round(train_r2, 3),
                    'val_R2': round(val_r2, 3),
                    'gap': round(gap, 3)
                })

    return pd.DataFrame(results)
