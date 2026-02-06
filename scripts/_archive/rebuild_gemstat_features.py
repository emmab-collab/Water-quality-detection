"""
Rebuild GEMStat features with all components
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import time
import rasterio
from rasterio.windows import from_bounds
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import pystac_client
import planetary_computer as pc
from tqdm import tqdm

# =============================================================================
# DEM
# =============================================================================
def extract_dem(catalog, lat, lon, buffer_deg=0.005):
    results = {'elevation': np.nan, 'slope': np.nan, 'aspect': np.nan}
    try:
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
        search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
        items = list(search.items())
        if len(items) == 0:
            return results
        item = items[0]
        signed_asset = pc.sign(item.assets["data"])
        with rasterio.open(signed_asset.href) as src:
            dst_crs = src.crs
            transformed_bbox = transform_bounds(CRS.from_epsg(4326), dst_crs, *bbox)
            window = from_bounds(*transformed_bbox, src.transform)
            data = src.read(1, window=window)
            if data.size == 0:
                return results
            valid_data = data[data > -1000]
            if valid_data.size == 0:
                return results
            results['elevation'] = float(np.mean(valid_data))
            if data.size >= 9:
                dy, dx = np.gradient(data, 30)
                slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                results['slope'] = float(np.nanmean(np.degrees(slope_rad)))
                aspect_rad = np.arctan2(-dx, dy)
                aspect_deg = (np.degrees(aspect_rad) + 360) % 360
                results['aspect'] = float(np.nanmean(aspect_deg))
    except:
        pass
    return results

# =============================================================================
# WORLDCOVER
# =============================================================================
WORLDCOVER_CLASSES = {
    10: 'lc_tree', 20: 'lc_shrubland', 30: 'lc_grassland', 40: 'lc_cropland',
    50: 'lc_builtup', 60: 'lc_bare', 80: 'lc_water', 90: 'lc_wetland'
}

def extract_worldcover(catalog, lat, lon, buffer_meters=500):
    results = {name: 0.0 for name in WORLDCOVER_CLASSES.values()}
    try:
        delta_lat = buffer_meters / 111000
        delta_lon = buffer_meters / (111000 * np.cos(np.radians(lat)))
        bbox = [lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat]
        search = catalog.search(collections=["esa-worldcover"], bbox=bbox)
        items = list(search.items())
        if len(items) == 0:
            return results
        item = items[0]
        signed_asset = pc.sign(item.assets["map"])
        with rasterio.open(signed_asset.href) as src:
            dst_crs = src.crs
            transformed_bbox = transform_bounds(CRS.from_epsg(4326), dst_crs, *bbox)
            window = from_bounds(*transformed_bbox, src.transform)
            data = src.read(1, window=window)
            if data.size == 0:
                return results
            total_pixels = data.size
            unique, counts = np.unique(data, return_counts=True)
            for class_code, class_name in WORLDCOVER_CLASSES.items():
                if class_code in unique:
                    idx = np.where(unique == class_code)[0][0]
                    results[class_name] = (counts[idx] / total_pixels) * 100
    except:
        pass
    return results

# =============================================================================
# SOILGRIDS
# =============================================================================
SOILGRIDS_API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRIDS_PROPERTIES = {
    'phh2o': 'soil_ph', 'clay': 'soil_clay', 'sand': 'soil_sand',
    'soc': 'soil_soc', 'cec': 'soil_cec', 'nitrogen': 'soil_nitrogen'
}

def extract_soilgrids(lat, lon):
    results = {name: np.nan for name in SOILGRIDS_PROPERTIES.values()}
    try:
        params = {
            'lon': lon, 'lat': lat,
            'property': list(SOILGRIDS_PROPERTIES.keys()),
            'depth': ['0-5cm'], 'value': ['mean']
        }
        response = requests.get(SOILGRIDS_API_URL, params=params, timeout=30)
        if response.status_code != 200:
            return results
        data = response.json()
        if 'properties' in data and 'layers' in data['properties']:
            for layer in data['properties']['layers']:
                prop_name = layer['name']
                if prop_name in SOILGRIDS_PROPERTIES:
                    feature_name = SOILGRIDS_PROPERTIES[prop_name]
                    for depth_data in layer['depths']:
                        if depth_data['label'] == '0-5cm':
                            value = depth_data['values'].get('mean')
                            if value is not None:
                                results[feature_name] = float(value)
                            break
    except:
        pass
    return results

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("REBUILD GEMSTAT FEATURES")
    print("=" * 60)

    # Charger les donnees
    gemstat = pd.read_csv('data/raw/gemstat_eastern_cape.csv')
    print(f"\nObservations: {len(gemstat)}")

    # Sites uniques
    sites = gemstat[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
    print(f"Sites uniques: {len(sites)}")

    # Connexion Planetary Computer
    print("\n1. Connexion Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    print("[OK]")

    # DEM
    print("\n2. Extraction DEM...")
    dem_results = []
    for idx, row in tqdm(sites.iterrows(), total=len(sites), desc="DEM"):
        result = extract_dem(catalog, row['Latitude'], row['Longitude'])
        result['Latitude'] = row['Latitude']
        result['Longitude'] = row['Longitude']
        dem_results.append(result)
    dem_df = pd.DataFrame(dem_results)

    # WorldCover
    print("\n3. Extraction WorldCover...")
    wc_results = []
    for idx, row in tqdm(sites.iterrows(), total=len(sites), desc="WorldCover"):
        result = extract_worldcover(catalog, row['Latitude'], row['Longitude'])
        result['Latitude'] = row['Latitude']
        result['Longitude'] = row['Longitude']
        wc_results.append(result)
    wc_df = pd.DataFrame(wc_results)

    # SoilGrids
    print("\n4. Extraction SoilGrids...")
    soil_results = []
    for idx, row in tqdm(sites.iterrows(), total=len(sites), desc="SoilGrids"):
        result = extract_soilgrids(row['Latitude'], row['Longitude'])
        result['Latitude'] = row['Latitude']
        result['Longitude'] = row['Longitude']
        soil_results.append(result)
        time.sleep(0.3)
    soil_df = pd.DataFrame(soil_results)

    # Fusionner les features par site
    print("\n5. Fusion des features par site...")
    site_features = sites.copy()
    site_features = site_features.merge(dem_df, on=['Latitude', 'Longitude'], how='left')
    site_features = site_features.merge(wc_df, on=['Latitude', 'Longitude'], how='left')
    site_features = site_features.merge(soil_df, on=['Latitude', 'Longitude'], how='left')
    print(f"Features par site: {site_features.shape}")

    # Charger TerraClimate
    print("\n6. Chargement TerraClimate...")
    terra_df = pd.read_csv('data/processed/gemstat_terraclimate_features.csv')
    print(f"TerraClimate: {terra_df.shape}")

    # Fusion finale
    print("\n7. Fusion finale...")
    gemstat_features = gemstat.copy()

    # Ajouter features par site
    gemstat_features = gemstat_features.merge(
        site_features,
        on=['Latitude', 'Longitude'],
        how='left'
    )

    # Ajouter TerraClimate (par observation)
    terra_cols = [c for c in terra_df.columns if c not in ['Latitude', 'Longitude', 'Sample Date']]
    for col in terra_cols:
        gemstat_features[col] = terra_df[col].values

    print(f"Dataset final: {gemstat_features.shape}")

    # Sauvegarder
    gemstat_features.to_csv('data/processed/gemstat_features.csv', index=False)
    print("\n[OK] Sauvegarde: gemstat_features.csv")

    # Verification
    print("\n8. Verification des valeurs manquantes:")
    missing = gemstat_features.isnull().sum()
    has_missing = False
    for col in gemstat_features.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing[col]/len(gemstat_features)*100:.1f}%)")
            has_missing = True
    if not has_missing:
        print("  [OK] Aucune valeur manquante!")

    print("\n" + "=" * 60)
    print("TERMINE")
    print("=" * 60)

if __name__ == "__main__":
    main()
