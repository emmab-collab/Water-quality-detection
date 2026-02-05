# -*- coding: utf-8 -*-
"""
Visualisation - Graphiques pour l'analyse et l'évaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor

from src.config import TARGETS, RANDOM_SEED


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


# =============================================================================
# LEARNING CURVES
# =============================================================================

def plot_learning_curves(
    X: np.ndarray,
    y: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    cv: int = 5,
    train_sizes: np.ndarray = None
):
    """
    Affiche les learning curves pour chaque target.

    Montre l'évolution du score R² sur train et validation
    en fonction de la taille du dataset d'entraînement.

    - Si les courbes train et val sont proches : bon modèle
    - Si train >> val : overfitting
    - Si les deux sont bas : underfitting
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i, target in enumerate(TARGETS):
        ax = axes[i]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            X,
            y[target],
            train_sizes=train_sizes,
            cv=cv,
            scoring='r2',
            n_jobs=-1
        )

        # Moyenne et écart-type
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # Plot
        ax.plot(train_sizes_abs, train_mean, 'o-', label='Train', color='blue')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        ax.plot(train_sizes_abs, val_mean, 'o-', label='Validation', color='orange')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')

        ax.set_xlabel('Taille du training set')
        ax.set_ylabel('R²')
        ax.set_title(target)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Diagnostic overfitting
        gap = train_mean[-1] - val_mean[-1]
        if gap > 0.1:
            ax.text(0.05, 0.95, f'Gap: {gap:.2f} (overfitting)',
                   transform=ax.transAxes, fontsize=9, color='red', va='top')

    plt.suptitle('Learning Curves - Détection Overfitting', fontsize=12)
    plt.tight_layout()
    plt.show()


# =============================================================================
# HYPERPARAMETER SEARCH CURVES
# =============================================================================

def plot_hyperparameter_search(results_df: pd.DataFrame):
    """
    Affiche les courbes R² train/val pour chaque combinaison d'hyperparamètres.

    Pour chaque target, affiche R² en fonction de max_depth,
    avec une ligne par valeur de min_samples_leaf.
    """
    targets = results_df['target'].unique()
    min_samples_leafs = sorted(results_df['min_samples_leaf'].unique())

    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4))
    if len(targets) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(min_samples_leafs)))

    for i, target in enumerate(targets):
        ax = axes[i]
        target_data = results_df[results_df['target'] == target]

        for j, min_leaf in enumerate(min_samples_leafs):
            subset = target_data[target_data['min_samples_leaf'] == min_leaf]
            subset = subset.sort_values('max_depth')

            depths = subset['max_depth'].values
            train_r2 = subset['train_R2'].values
            val_r2 = subset['val_R2'].values

            # Courbe train (ligne pleine)
            ax.plot(depths, train_r2, 'o-', color=colors[j],
                   label=f'min_leaf={min_leaf} (train)', linewidth=2)
            # Courbe val (ligne pointillée)
            ax.plot(depths, val_r2, 's--', color=colors[j],
                   label=f'min_leaf={min_leaf} (val)', linewidth=2, alpha=0.7)

        ax.set_xlabel('max_depth')
        ax.set_ylabel('R²')
        ax.set_title(target)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(target_data['max_depth'].unique()))

    # Légende commune
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)

    plt.suptitle('R² Train vs Validation par hyperparamètres\n(lignes pleines = train, pointillées = val)', fontsize=12)
    plt.tight_layout()
    plt.show()
