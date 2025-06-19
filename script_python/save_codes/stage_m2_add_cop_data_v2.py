# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:12:20 2025

@author: vanleene valentine
"""

#%% Librairies import
import geopandas as gpd
import os
import glob
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree

#%% 
# Points du maillage
mesh_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/ks7_v5_1.3_4326.shp'
gdf_mesh = gpd.read_file(mesh_path) # 4326
# Points de la frontière ouverte
cl_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/clv_chaincut_complete_largedomain_1.3_v2_4326.shp'
gdf_cl = gpd.read_file(cl_path) # 4326
# Points correspondants aux bouées copernicus
buoy_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/buoy.shp'
gdf_buoy = gpd.read_file(buoy_path) # 4326

#%% Loop selon tempête (martin, xynthia, celine)
tempest = ['martin', 'xynthia', 'celine']
type_data_list = ['oper', 'wave']
start_times = ["1999-12-26", "2010-02-25", "2024-10-27"]
end_times = ["1999-12-29", "2010-03-01", "2024-10-30"]
mesh_lat = gdf_mesh.latitude.values
mesh_lon = gdf_mesh.longitude.values
mesh_coords = np.c_[gdf_mesh.geometry.x, gdf_mesh.geometry.y]
buoy_coords = np.c_[gdf_buoy.geometry.x, gdf_buoy.geometry.y]

gdf_cl = gdf_cl[gdf_cl.geometry.notnull()]
gdf_cl = gdf_cl[gdf_cl.geometry.is_valid]
gdf_cl = gdf_cl[~gdf_cl.geometry.is_empty]
cl_coords = np.c_[gdf_cl.geometry.x, gdf_cl.geometry.y]

# Construction d’un arbre KDTree 
tree = cKDTree(mesh_coords)
_, closest_indices = tree.query(buoy_coords)  # indices des points maillage les + proches
tree2 = cKDTree(mesh_coords)
_2, closest_indices2 = tree2.query(cl_coords)  # indices des points maillage les + proches

for ti, storm in enumerate(tempest):
    print(f'{storm}')
    ds = xr.open_dataset(f'../datas_stage_m2/{storm}/data_stream-oper_stepType-instant.nc')
    oper_param = ['u10', 'v10', 'msl']
    for param in oper_param:
        gdf_mesh[f'{param}_{storm}'] = 0  # évite 'NaN' string
        
        # Vectorisation partielle
        for i, (lat, lon) in enumerate(zip(mesh_lat, mesh_lon)):
            param_point = ds[param].sel(latitude=lat, longitude=lon, method='nearest')
            param_filtered = param_point.sel(valid_time=slice(start_times[ti], end_times[ti]))
            mean_value = param_filtered.mean(dim='valid_time').values
            gdf_mesh.at[i, f'{param}_{storm}'] = mean_value
    ds = xr.open_dataset(f'../datas_stage_m2/{storm}/data_stream-wave_stepType-instant.nc')
    wave_param = ['mwd1', 'mwd', 'mwp', 'swh']
    for param in wave_param:
        # Initialiser avec NaN
        gdf_mesh[f'{param}_{storm}'] = 0

        # Boucles sur les bouées 
        for i, (lat, lon) in enumerate(zip(gdf_buoy.latitude, gdf_buoy.longitude)):
            # Extraire les données pour une bouée
            param_point = ds[param].sel(latitude=lat, longitude=lon, method='nearest')
            param_filtered = param_point.sel(valid_time=slice(start_times[ti], end_times[ti]))
            mean_value = param_filtered.mean(dim='valid_time').values

            # Trouver l'index du point de maillage le plus proche
            closest_idx = closest_indices[i]

            # Mise à jour du maillage
            gdf_mesh.at[closest_idx, f'{param}_{storm}'] = mean_value
        
        # Boucles sur les points OpenBoundary 
        for i, (lat, lon) in enumerate(zip(gdf_cl.latitude, gdf_cl.longitude)):
            # Extraire les données pour une bouée
            param_point = ds[param].sel(latitude=lat, longitude=lon, method='nearest')
            param_filtered = param_point.sel(valid_time=slice(start_times[ti], end_times[ti]))
            mean_value = param_filtered.mean(dim='valid_time').values

            # Trouver l'index du point de maillage le plus proche
            closest_idx2 = closest_indices2[i]

            # Mise à jour du maillage
            gdf_mesh.at[closest_idx2, f'{param}_{storm}'] = mean_value

#%% Exporting the new shapefile
exit_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/ks7_v5_1.3_4326_wind&swell.shp'
# In case the file already exists, it will be removed
for fichier in glob.glob(f'{exit_path}.*'):
    os.remove(fichier)
# Saving the GeoDataFrame as a .shp file
gdf_mesh.to_file(exit_path, driver='ESRI Shapefile')
