# -*- coding: utf-8 -*-
"""
Configuration du projet - EY Water Quality Challenge
=====================================================

Ce fichier contient toutes les constantes du projet.
Pourquoi? Pour avoir un seul endroit où changer les paramètres.
"""

# =============================================================================
# LES 3 TARGETS (variables à prédire)
# =============================================================================

# Noms exacts des colonnes dans les fichiers CSV
TARGETS = [
    'Total Alkalinity',              # Alcalinité totale
    'Electrical Conductance',        # Conductivité électrique
    'Dissolved Reactive Phosphorus'  # Phosphore réactif dissous
]

# Descriptions pour les graphiques
TARGET_INFO = {
    'Total Alkalinity': {
        'short_name': 'TA',
        'unit': 'mg/L CaCO3',
        'description': 'Capacité tampon de l\'eau (résistance aux changements de pH)'
    },
    'Electrical Conductance': {
        'short_name': 'EC',
        'unit': 'µS/cm',
        'description': 'Concentration totale en minéraux dissous'
    },
    'Dissolved Reactive Phosphorus': {
        'short_name': 'DRP',
        'unit': 'µg/L',
        'description': 'Nutriment biodisponible (risque d\'algues si trop élevé)'
    }
}

# =============================================================================
# LES FEATURES (variables explicatives)
# =============================================================================

# Features du benchmark (celles utilisées par EY dans leur exemple)
BENCHMARK_FEATURES = ['swir22', 'NDMI', 'MNDWI', 'pet']

# Toutes les features Landsat disponibles
LANDSAT_FEATURES = ['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']

# Features TerraClimate disponibles (pet dans le benchmark, mais il y en a d'autres)
TERRACLIMATE_FEATURES = ['pet']  # Potential Evapotranspiration

# Colonnes de localisation (ne pas utiliser comme features!)
LOCATION_COLS = ['Latitude', 'Longitude', 'Sample Date']

# =============================================================================
# REPRODUCTIBILITÉ
# =============================================================================

# Seed pour que les résultats soient reproductibles
# (toujours le même résultat si on relance le code)
RANDOM_SEED = 42

# =============================================================================
# VALIDATION CROISÉE
# =============================================================================

# Nombre de folds pour la cross-validation
N_FOLDS = 5

# Ratio train/test pour le split simple
TEST_SIZE = 0.3  # 30% pour le test, comme le benchmark

# =============================================================================
# PARAMÈTRES DU MODÈLE (Random Forest du benchmark)
# =============================================================================

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,    # Nombre d'arbres
    'random_state': RANDOM_SEED,
    'n_jobs': -1            # Utiliser tous les CPU
}

# =============================================================================
# AFFICHAGE
# =============================================================================

def show_config():
    """Affiche la configuration actuelle."""
    print("=" * 50)
    print("CONFIGURATION DU PROJET")
    print("=" * 50)
    print(f"Targets: {TARGETS}")
    print(f"Features benchmark: {BENCHMARK_FEATURES}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"N folds: {N_FOLDS}")
    print(f"Test size: {TEST_SIZE}")
    print("=" * 50)


if __name__ == "__main__":
    show_config()
