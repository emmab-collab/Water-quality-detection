# -*- coding: utf-8 -*-
"""
Module de chargement des données.

Utilisation:
    from src.data.load_data import load_all
    X, y, sites, df = load_all(wq_path, landsat_path, terra_path)
"""

# Rendre les fonctions accessibles directement
from src.data.load_data import (
    load_all,
    load_water_quality,
    load_landsat,
    load_terraclimate,
    merge_all_data,
    check_missing,
    fill_missing,
    get_X_y,
    get_site_ids
)
