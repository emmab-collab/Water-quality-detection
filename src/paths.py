# -*- coding: utf-8 -*-
"""
Chemins du projet - EY Water Quality Challenge
==============================================

Ce fichier centralise tous les chemins vers les dossiers et fichiers.
Pourquoi? Pour ne pas avoir à recopier les chemins partout dans le code.

Si tu déplaces un dossier, tu changes ici et tout le reste fonctionne.
"""

from pathlib import Path
from datetime import datetime

# =============================================================================
# RACINE DU PROJET
# =============================================================================

# Trouver automatiquement la racine du projet
# __file__ = ce fichier (paths.py)
# .parent = dossier parent (src/)
# .parent = dossier parent (PROJET EY/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# FICHIERS DE DONNÉES
# =============================================================================

# Données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # Données brutes
PROCESSED_DIR = DATA_DIR / "processed"  # Données traitées

# --- TRAINING ---
# Water quality (targets + coordonnées)
WATER_QUALITY_FILE = RAW_DIR / "water_quality_training_dataset.csv"

# Features Landsat (6 bandes + 4 indices : blue, green, red, nir, swir16, swir22, NDVI, NDWI, NDMI, MNDWI)
LANDSAT_FILE = PROCESSED_DIR / "landsat_features_training_complete.csv"

# Features TerraClimate (pet)
TERRACLIMATE_FILE = PROCESSED_DIR / "terraclimate_features_training.csv"

# --- SUBMISSION (TEST) ---
# Template de soumission (coordonnées + dates)
SUBMISSION_TEMPLATE = RAW_DIR / "submission_template.csv"

# Features Landsat pour submission
LANDSAT_SUBMISSION_FILE = PROCESSED_DIR / "landsat_features_validation_complete.csv"

# Features TerraClimate pour submission
TERRACLIMATE_SUBMISSION_FILE = PROCESSED_DIR / "terraclimate_features_validation.csv"

# Alias pour rétrocompatibilité
LANDSAT_VAL_FILE = LANDSAT_SUBMISSION_FILE
TERRACLIMATE_VAL_FILE = TERRACLIMATE_SUBMISSION_FILE

# =============================================================================
# DOSSIERS DU PROJET
# =============================================================================

# Code source
SRC_DIR = PROJECT_ROOT / "src"

# Notebooks Jupyter
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Documentation
DOCS_DIR = PROJECT_ROOT / "docs"

# Résultats
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"         # Modèles entraînés (.pkl)
FIGURES_DIR = OUTPUTS_DIR / "figures"       # Graphiques (.png)
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"  # Fichiers de soumission (.csv)

# =============================================================================
# FONCTIONS UTILES
# =============================================================================

def create_folders():
    """
    Crée tous les dossiers du projet s'ils n'existent pas.
    À lancer une fois au début du projet.
    """
    folders = [
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        SUBMISSIONS_DIR
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("[OK] Dossiers créés:")
    for folder in folders:
        print(f"  - {folder}")


def get_submission_path(name):
    """
    Génère un chemin pour sauvegarder une soumission avec la date/heure.

    Exemple:
        path = get_submission_path("random_forest_v1")
        # -> outputs/submissions/submission_random_forest_v1_20240115_143052.csv
    """
    # Créer le dossier si nécessaire
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Ajouter un timestamp pour ne pas écraser les anciens fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{name}_{timestamp}.csv"

    return SUBMISSIONS_DIR / filename


def get_model_path(name):
    """
    Génère un chemin pour sauvegarder un modèle.

    Exemple:
        path = get_model_path("lightgbm_alkalinity")
        # -> outputs/models/lightgbm_alkalinity.pkl
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"{name}.pkl"


def get_figure_path(name):
    """
    Génère un chemin pour sauvegarder une figure.

    Exemple:
        path = get_figure_path("correlation_matrix")
        # -> outputs/figures/correlation_matrix.png
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / f"{name}.png"


# =============================================================================
# VÉRIFICATION
# =============================================================================

def check_data_files():
    """
    Vérifie que les fichiers de données existent.
    Utile pour s'assurer que tout est bien configuré.
    """
    files_to_check = [
        # Training
        ("Water Quality (training)", WATER_QUALITY_FILE),
        ("Landsat (training)", LANDSAT_FILE),
        ("TerraClimate (training)", TERRACLIMATE_FILE),
        # Submission
        ("Submission Template", SUBMISSION_TEMPLATE),
        ("Landsat (submission)", LANDSAT_SUBMISSION_FILE),
        ("TerraClimate (submission)", TERRACLIMATE_SUBMISSION_FILE),
    ]

    print("Vérification des fichiers de données:")
    print("=" * 50)

    all_ok = True
    for name, path in files_to_check:
        if path.exists():
            print(f"[OK] {name}")
        else:
            print(f"[MANQUANT] {name}")
            print(f"    Attendu: {path}")
            all_ok = False

    print("=" * 50)
    if all_ok:
        print("Tous les fichiers sont présents!")
    else:
        print("Certains fichiers sont manquants.")

    return all_ok


def show_paths():
    """
    Affiche tous les chemins configurés.
    Utile pour vérifier que tout est correct.
    """
    print("=" * 60)
    print("CHEMINS DU PROJET")
    print("=" * 60)
    print(f"Racine du projet : {PROJECT_ROOT}")
    print()
    print("Données TRAINING:")
    print(f"  - Water Quality : {WATER_QUALITY_FILE}")
    print(f"  - Landsat       : {LANDSAT_FILE}")
    print(f"  - TerraClimate  : {TERRACLIMATE_FILE}")
    print()
    print("Données SUBMISSION:")
    print(f"  - Template      : {SUBMISSION_TEMPLATE}")
    print(f"  - Landsat       : {LANDSAT_SUBMISSION_FILE}")
    print(f"  - TerraClimate  : {TERRACLIMATE_SUBMISSION_FILE}")
    print()
    print("Dossiers de sortie:")
    print(f"  - Modèles       : {MODELS_DIR}")
    print(f"  - Figures       : {FIGURES_DIR}")
    print(f"  - Soumissions   : {SUBMISSIONS_DIR}")
    print("=" * 60)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Si tu lances ce fichier directement: python src/paths.py
    show_paths()
    print()
    check_data_files()
