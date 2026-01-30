# Evaluation Metrics - EY Water Quality Challenge

## 1. Métriques Principales

### 1.1 RMSE (Root Mean Square Error)

**Formule**:
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Caractéristiques**:
- Pénalise fortement les grandes erreurs (erreur au carré)
- Même unité que la variable cible
- Sensible aux outliers

**Implémentation**:
```python
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# ou
rmse = mean_squared_error(y_true, y_pred, squared=False)
```

### 1.2 MAE (Mean Absolute Error)

**Formule**:
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Caractéristiques**:
- Erreur moyenne en valeur absolue
- Même unité que la variable cible
- Moins sensible aux outliers que RMSE

**Implémentation**:
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

### 1.3 R² (Coefficient de Détermination)

**Formule**:
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**Caractéristiques**:
- Proportion de variance expliquée
- Valeur entre -∞ et 1 (1 = parfait)
- Indépendant de l'échelle

**Implémentation**:
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

---

## 2. Comparaison des Métriques

| Métrique | Sensibilité Outliers | Interprétabilité | Usage |
|----------|---------------------|------------------|-------|
| RMSE | Haute | Moyenne | Optimisation |
| MAE | Moyenne | Haute | Reporting |
| R² | Moyenne | Haute | Comparaison |
| MAPE | Faible | Haute | Business |

### 2.1 Quand utiliser quelle métrique?

- **RMSE**: Quand les grandes erreurs sont plus coûteuses
- **MAE**: Quand toutes les erreurs ont le même coût
- **R²**: Pour comparer des modèles sur différents targets
- **MAPE**: Quand on veut une erreur en pourcentage (attention aux valeurs proches de 0)

---

## 3. Métriques Multi-Target

### 3.1 Score Global

Pour combiner les scores des 3 targets:

**Option A: Moyenne Simple**
```python
score_global = (rmse_alkalinity + rmse_conductivity + rmse_phosphorus) / 3
```

**Option B: Moyenne Pondérée**
```python
weights = {'alkalinity': 0.33, 'conductivity': 0.33, 'phosphorus': 0.34}
score_global = sum(w * rmse for w, rmse in zip(weights.values(), rmses))
```

**Option C: RMSE Normalisé (recommandé)**
```python
# Normaliser par l'écart-type de chaque target
nrmse_alk = rmse_alk / std_alk
nrmse_cond = rmse_cond / std_cond
nrmse_phos = rmse_phos / std_phos
score_global = (nrmse_alk + nrmse_cond + nrmse_phos) / 3
```

### 3.2 Implémentation Complète

```python
def evaluate_multi_target(y_true, y_pred, target_names):
    """
    Évalue les prédictions multi-target.

    Args:
        y_true: DataFrame avec les vraies valeurs
        y_pred: DataFrame avec les prédictions
        target_names: Liste des noms de targets

    Returns:
        DataFrame avec les métriques par target
    """
    results = []

    for target in target_names:
        yt = y_true[target]
        yp = y_pred[target]

        results.append({
            'target': target,
            'rmse': np.sqrt(mean_squared_error(yt, yp)),
            'mae': mean_absolute_error(yt, yp),
            'r2': r2_score(yt, yp),
            'std_true': yt.std(),
            'nrmse': np.sqrt(mean_squared_error(yt, yp)) / yt.std()
        })

    return pd.DataFrame(results)
```

---

## 4. Métriques de Validation Spatiale

### 4.1 Performance par Site

Évaluer si le modèle a des biais géographiques:

```python
def evaluate_by_site(y_true, y_pred, site_ids):
    """Calcule les métriques par site."""
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'site': site_ids
    })

    return df.groupby('site').apply(
        lambda x: pd.Series({
            'rmse': np.sqrt(mean_squared_error(x['y_true'], x['y_pred'])),
            'mae': mean_absolute_error(x['y_true'], x['y_pred']),
            'n_samples': len(x)
        })
    )
```

### 4.2 Variance Inter-Sites vs Intra-Sites

```python
# Comparer la variance des erreurs entre sites
site_rmses = evaluate_by_site(y_true, y_pred, site_ids)['rmse']
cv_site_rmse = site_rmses.std() / site_rmses.mean()  # Coefficient de variation

# Un CV élevé indique une performance hétérogène entre sites
```

---

## 5. Analyse des Résidus

### 5.1 Distribution des Erreurs

```python
residuals = y_true - y_pred

# Statistiques
print(f"Mean residual: {residuals.mean():.4f}")  # Devrait être ~0
print(f"Std residual: {residuals.std():.4f}")
print(f"Skewness: {residuals.skew():.4f}")  # Devrait être ~0
print(f"Kurtosis: {residuals.kurtosis():.4f}")  # Devrait être ~0 (normal)
```

### 5.2 Résidus vs Prédictions

Vérifier l'homoscédasticité (variance constante des erreurs):
```python
import matplotlib.pyplot as plt

plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
```

### 5.3 Résidus vs Features

Identifier si certaines features sont mal modélisées:
```python
for feature in important_features:
    plt.scatter(X[feature], residuals, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs {feature}')
    plt.show()
```

---

## 6. Intervalles de Confiance

### 6.1 Bootstrap des Métriques

```python
from sklearn.utils import resample

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000):
    """
    Calcule un intervalle de confiance pour une métrique.
    """
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = resample(range(n), n_samples=n)
        score = metric_func(y_true.iloc[idx], y_pred.iloc[idx])
        scores.append(score)

    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'ci_lower': np.percentile(scores, 2.5),
        'ci_upper': np.percentile(scores, 97.5)
    }
```

---

## 7. Métriques de Monitoring

### 7.1 Stabilité du Modèle

Variance des scores à travers les folds CV:
```python
cv_scores = cross_val_score(model, X, y, cv=spatial_cv, groups=sites)
stability = cv_scores.std() / abs(cv_scores.mean())  # CV du score

# Stabilité < 0.1 = bon
# Stabilité > 0.2 = instable
```

### 7.2 Robustesse aux Seeds

```python
scores_by_seed = []
for seed in range(10):
    model.set_params(random_state=seed)
    score = cross_val_score(model, X, y, cv=cv).mean()
    scores_by_seed.append(score)

print(f"Score mean: {np.mean(scores_by_seed):.4f}")
print(f"Score std: {np.std(scores_by_seed):.4f}")
```

---

## 8. Tableau de Bord des Métriques

```python
def create_metrics_dashboard(y_true, y_pred, target_names, site_ids=None):
    """
    Crée un rapport complet des métriques.
    """
    report = {}

    # Métriques globales par target
    for target in target_names:
        yt, yp = y_true[target], y_pred[target]
        report[target] = {
            'rmse': np.sqrt(mean_squared_error(yt, yp)),
            'mae': mean_absolute_error(yt, yp),
            'r2': r2_score(yt, yp),
            'mape': np.mean(np.abs((yt - yp) / yt)) * 100
        }

    # Score global normalisé
    nrmses = [report[t]['rmse'] / y_true[t].std() for t in target_names]
    report['global'] = {
        'mean_nrmse': np.mean(nrmses),
        'mean_r2': np.mean([report[t]['r2'] for t in target_names])
    }

    return report
```

---

## 9. Seuils de Performance

| Niveau | R² | Interprétation |
|--------|-----|----------------|
| Excellent | > 0.9 | Prédictions très précises |
| Bon | 0.7 - 0.9 | Modèle utile |
| Acceptable | 0.5 - 0.7 | Amélioration possible |
| Faible | < 0.5 | Modèle peu fiable |

**Note**: Ces seuils sont indicatifs. La qualité acceptable dépend du contexte et de la variabilité naturelle des données.
