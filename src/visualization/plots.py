# -*- coding: utf-8 -*-
"""
Visualisation - Graphiques pour l'analyse et l'évaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from src.config import TARGETS


# =============================================================================
# CORRELATIONS
# =============================================================================

def plot_correlation(df: pd.DataFrame, features: List[str], targets: List[str] = TARGETS):
    """Affiche la heatmap de corrélation features vs targets."""
    corr = df[features + targets].corr().loc[features, targets]

    plt.figure(figsize=(8, len(features) * 0.4 + 1))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Corrélation Features vs Targets')
    plt.tight_layout()
    plt.show()


# =============================================================================
# PREDICTIONS VS REALITE
# =============================================================================

def plot_predictions(models: Dict, X: np.ndarray, y: pd.DataFrame, results: Dict):
    """Affiche les scatter plots prédictions vs réalité."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, target in enumerate(TARGETS):
        ax = axes[i]
        y_pred = models[target].predict(X)
        y_true = y[target]

        ax.scatter(y_true, y_pred, alpha=0.3, s=10)

        # Ligne parfaite
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=1)

        ax.set_xlabel('Réel')
        ax.set_ylabel('Prédit')
        ax.set_title(f"{target}\nR²={results[target]['R2']:.3f}")

    plt.tight_layout()
    plt.show()


# =============================================================================
# IMPORTANCE DES FEATURES
# =============================================================================

def plot_importance(models: Dict, feature_names: List[str], top_n: int = 10):
    """Affiche l'importance des features pour chaque modèle."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for i, target in enumerate(TARGETS):
        ax = axes[i]
        imp = models[target].feature_importances_
        idx = np.argsort(imp)[-top_n:]

        ax.barh(range(len(idx)), imp[idx])
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[j] for j in idx])
        ax.set_xlabel('Importance')
        ax.set_title(target)

    plt.tight_layout()
    plt.show()
