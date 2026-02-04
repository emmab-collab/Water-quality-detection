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

# Features du benchmark ORIGINAL (celles utilisées par EY dans leur exemple)
# On les garde pour référence, mais on va utiliser plus de features
BENCHMARK_FEATURES_ORIGINAL = ['swir22', 'NDMI', 'MNDWI', 'pet']

# Toutes les features Landsat disponibles (6 bandes + 4 indices)
LANDSAT_BANDS = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']
LANDSAT_INDICES = ['NDVI', 'NDWI', 'NDMI', 'MNDWI']
LANDSAT_FEATURES = LANDSAT_BANDS + LANDSAT_INDICES

# Features TerraClimate disponibles (10 variables)
TERRACLIMATE_FEATURES = [
    'pet',   # Potential Evapotranspiration (évapotranspiration potentielle)
    'aet',   # Actual Evapotranspiration (évapotranspiration réelle)
    'ppt',   # Precipitation (précipitations)
    'tmax',  # Maximum Temperature (température max)
    'tmin',  # Minimum Temperature (température min)
    'soil',  # Soil Moisture (humidité du sol)
    'def',   # Climate Water Deficit (déficit hydrique = PET - AET)
    'pdsi',  # Palmer Drought Severity Index (indice de sécheresse)
    'vpd',   # Vapor Pressure Deficit (déficit de pression de vapeur)
    'ws',    # Wind Speed (vitesse du vent)
]

# =============================================================================
# NOS FEATURES (améliorées par rapport au benchmark)
# =============================================================================

# Toutes les features disponibles dans les données
ALL_FEATURES = [
    # Bandes spectrales Landsat (6)
    'blue',     # Bande bleue - pénètre l'eau profonde, turbidité
    'green',    # Bande verte - chlorophylle/algues
    'red',      # Bande rouge - chlorophylle, végétation
    'nir',      # Proche infrarouge - détecte végétation/algues
    'swir16',   # Infrarouge ondes courtes 1 - humidité
    'swir22',   # Infrarouge ondes courtes 2 - minéraux/turbidité
    # Indices spectraux (4)
    'NDVI',     # Indice de végétation
    'NDWI',     # Indice d'eau (Green-NIR)
    'NDMI',     # Indice d'humidité (NIR-SWIR)
    'MNDWI',    # Indice de détection d'eau modifié (Green-SWIR)
    # Climat TerraClimate (10)
    'pet',      # Évapotranspiration potentielle
    'aet',      # Évapotranspiration réelle
    'ppt',      # Précipitations
    'tmax',     # Température max
    'tmin',     # Température min
    'soil',     # Humidité du sol
    'def',      # Déficit hydrique climatique
    'pdsi',     # Indice de sécheresse Palmer
    'vpd',      # Déficit de pression de vapeur
    'ws',       # Vitesse du vent
]

# Alias pour rétrocompatibilité
BENCHMARK_FEATURES = ALL_FEATURES

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
    print(f"Features benchmark original: {BENCHMARK_FEATURES_ORIGINAL}")
    print(f"Nos features (améliorées): {BENCHMARK_FEATURES}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"N folds: {N_FOLDS}")
    print(f"Test size: {TEST_SIZE}")
    print("=" * 50)


if __name__ == "__main__":
    show_config()
