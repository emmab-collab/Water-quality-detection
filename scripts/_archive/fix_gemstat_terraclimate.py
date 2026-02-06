"""
Fix TerraClimate extraction for GEMStat data
Using Zarr access (like the original notebook)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import pystac_client
import planetary_computer as pc
from tqdm import tqdm
import os

# Variables TerraClimate
TERRACLIMATE_VARIABLES = ['pet', 'aet', 'ppt', 'tmax', 'tmin', 'soil', 'def', 'pdsi', 'vpd', 'ws']
TEMPORAL_VARS = ['ppt', 'soil', 'def', 'vpd']
N_LAGS = 3

def load_terraclimate_dataset():
    """Charge le dataset TerraClimate via Zarr."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields["xarray:open_kwargs"],
        )

    print(f"Dataset charge! Variables: {list(ds.data_vars)}")
    return ds


def filter_terraclimate(ds, var, start_date="2010-01-01", end_date="2016-12-31"):
    """Filtre le dataset pour une variable et la zone Afrique du Sud."""
    ds_filtered = ds[var].sel(
        time=slice(start_date, end_date),
        lat=slice(-21.72, -35.18),
        lon=slice(14.97, 32.79)
    )

    df = ds_filtered.to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df = df.rename(columns={"lat": "Latitude", "lon": "Longitude", "time": "year_month"})
    df['year_month'] = df['year_month'].dt.to_period('M')

    return df


def build_kdtree(climate_df):
    """Construit un KD-Tree pour trouver les points les plus proches."""
    unique_coords = climate_df[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
    coords_radians = np.radians(unique_coords.values)
    tree = cKDTree(coords_radians)
    return tree, unique_coords


def extract_with_temporal_features(sites_df, climate_df, var_name, tree, unique_coords, n_lags=3):
    """Extrait une variable avec ses agregations temporelles."""
    sites_df = sites_df.copy().reset_index(drop=True)

    # Trouver les coordonnees les plus proches
    site_coords = np.radians(sites_df[['Latitude', 'Longitude']].values)
    _, indices = tree.query(site_coords, k=1)
    sites_df['nearest_lat'] = unique_coords.iloc[indices]['Latitude'].values
    sites_df['nearest_lon'] = unique_coords.iloc[indices]['Longitude'].values

    # Convertir les dates (format DD-MM-YYYY)
    sites_df['Sample Date'] = pd.to_datetime(sites_df['Sample Date'], format='%d-%m-%Y', errors='coerce')
    sites_df['year_month'] = sites_df['Sample Date'].dt.to_period('M')

    # Pivot pour acces rapide
    climate_pivot = climate_df.pivot_table(
        index=['Latitude', 'Longitude'],
        columns='year_month',
        values=var_name,
        aggfunc='first'
    )

    # Moyenne saisonniere
    climate_df['month'] = climate_df['year_month'].apply(lambda x: x.month)
    seasonal_mean = climate_df.groupby(['Latitude', 'Longitude', 'month'])[var_name].mean().reset_index()
    seasonal_mean = seasonal_mean.rename(columns={var_name: f'{var_name}_seasonal_mean'})

    results = []

    for idx, row in sites_df.iterrows():
        lat, lon = row['nearest_lat'], row['nearest_lon']
        ym = row['year_month']
        month = ym.month

        result = {}

        try:
            current_val = climate_pivot.loc[(lat, lon), ym] if ym in climate_pivot.columns else np.nan
            result[var_name] = current_val

            lag_values = [current_val] if not pd.isna(current_val) else []
            for lag in range(1, n_lags + 1):
                lag_ym = ym - lag
                lag_val = climate_pivot.loc[(lat, lon), lag_ym] if lag_ym in climate_pivot.columns else np.nan
                result[f'{var_name}_lag{lag}'] = lag_val
                if not pd.isna(lag_val):
                    lag_values.append(lag_val)

            result[f'{var_name}_sum{n_lags+1}'] = np.sum(lag_values) if lag_values else np.nan
            result[f'{var_name}_mean{n_lags+1}'] = np.mean(lag_values) if lag_values else np.nan

            seasonal = seasonal_mean[(seasonal_mean['Latitude'] == lat) &
                                     (seasonal_mean['Longitude'] == lon) &
                                     (seasonal_mean['month'] == month)]
            if len(seasonal) > 0 and not pd.isna(current_val):
                seasonal_val = seasonal[f'{var_name}_seasonal_mean'].values[0]
                result[f'{var_name}_anomaly'] = current_val - seasonal_val
            else:
                result[f'{var_name}_anomaly'] = np.nan

        except (KeyError, IndexError):
            result[var_name] = np.nan
            for lag in range(1, n_lags + 1):
                result[f'{var_name}_lag{lag}'] = np.nan
            result[f'{var_name}_sum{n_lags+1}'] = np.nan
            result[f'{var_name}_mean{n_lags+1}'] = np.nan
            result[f'{var_name}_anomaly'] = np.nan

        results.append(result)

    return pd.DataFrame(results)


def extract_simple(sites_df, climate_df, var_name, tree, unique_coords):
    """Extrait une variable simple (sans agregations temporelles)."""
    sites_df = sites_df.copy().reset_index(drop=True)

    site_coords = np.radians(sites_df[['Latitude', 'Longitude']].values)
    _, indices = tree.query(site_coords, k=1)
    sites_df['nearest_lat'] = unique_coords.iloc[indices]['Latitude'].values
    sites_df['nearest_lon'] = unique_coords.iloc[indices]['Longitude'].values

    sites_df['Sample Date'] = pd.to_datetime(sites_df['Sample Date'], format='%d-%m-%Y', errors='coerce')
    sites_df['year_month'] = sites_df['Sample Date'].dt.to_period('M')

    result = sites_df.merge(
        climate_df[['Latitude', 'Longitude', 'year_month', var_name]],
        left_on=['nearest_lat', 'nearest_lon', 'year_month'],
        right_on=['Latitude', 'Longitude', 'year_month'],
        how='left'
    )

    return pd.DataFrame({var_name: result[var_name].values})


def main():
    print("=" * 60)
    print("FIX TERRACLIMATE EXTRACTION FOR GEMSTAT")
    print("=" * 60)

    # Charger les donnees GEMStat
    gemstat = pd.read_csv('data/raw/gemstat_eastern_cape.csv')
    print(f"\nObservations GEMStat: {len(gemstat)}")

    # Verifier la plage de dates
    dates = pd.to_datetime(gemstat['Sample Date'], format='%d-%m-%Y')
    print(f"Plage de dates: {dates.min()} a {dates.max()}")

    # Charger TerraClimate via Zarr
    print("\n1. Connexion a TerraClimate (Zarr)...")
    ds = load_terraclimate_dataset()

    # Initialiser le DataFrame
    result_df = gemstat[['Latitude', 'Longitude', 'Sample Date']].copy()

    # Cache pour les donnees climatiques
    tc_cache = {}

    print("\n2. Extraction des variables...")

    for var in tqdm(TERRACLIMATE_VARIABLES, desc="Variables"):
        # Filtrer (2010 pour avoir les lags pour 2011)
        climate_df = filter_terraclimate(ds, var, start_date="2010-01-01", end_date="2016-12-31")
        tc_cache[var] = climate_df

        tree, unique_coords = build_kdtree(climate_df)

        if var in TEMPORAL_VARS:
            var_df = extract_with_temporal_features(
                gemstat, climate_df, var, tree, unique_coords, n_lags=N_LAGS
            )
        else:
            var_df = extract_simple(gemstat, climate_df, var, tree, unique_coords)

        for col in var_df.columns:
            result_df[col] = var_df[col].values

    print(f"\nExtraction terminee: {len(result_df)} lignes, {len(result_df.columns)} colonnes")

    # Statistiques
    print("\n3. Statistiques:")
    for var in TERRACLIMATE_VARIABLES:
        n_valid = result_df[var].notna().sum()
        pct = n_valid / len(result_df) * 100
        print(f"  {var}: {n_valid}/{len(result_df)} ({pct:.1f}%)")

    # Sauvegarder les features TerraClimate
    terra_output = 'data/processed/gemstat_terraclimate_features.csv'
    result_df.to_csv(terra_output, index=False)
    print(f"\n[OK] Sauvegarde: {terra_output}")

    # Mettre a jour gemstat_features.csv
    print("\n4. Mise a jour de gemstat_features.csv...")
    features = pd.read_csv('data/processed/gemstat_features.csv')

    # Colonnes TerraClimate a supprimer
    terra_cols = ['ppt', 'tmax', 'tmin', 'soil', 'def', 'vpd', 'aet', 'pet']
    for col in terra_cols:
        if col in features.columns:
            features = features.drop(columns=[col])

    # Supprimer aussi les colonnes avec suffixes temporels si elles existent
    cols_to_drop = [c for c in features.columns if any(c.startswith(v) for v in TEMPORAL_VARS)]
    features = features.drop(columns=cols_to_drop, errors='ignore')

    # Merger les nouvelles features
    terra_features = result_df.drop(columns=['Latitude', 'Longitude', 'Sample Date'])
    for col in terra_features.columns:
        features[col] = terra_features[col].values

    features.to_csv('data/processed/gemstat_features.csv', index=False)
    print("[OK] gemstat_features.csv mis a jour!")

    # Verification finale
    print("\n5. Verification finale des valeurs manquantes:")
    missing = features.isnull().sum()
    has_missing = False
    for col in features.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing[col]/len(features)*100:.1f}%)")
            has_missing = True

    if not has_missing:
        print("  [OK] Aucune valeur manquante!")

    print("\n" + "=" * 60)
    print("TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()
