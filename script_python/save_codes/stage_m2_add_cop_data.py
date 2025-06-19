# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:56:26 2025

@author: vanleene valentine
"""

#%% Librairies import
import pandas as pd
import geopandas as gpd
import os
import glob
import xarray as xr

#%% 
# lecture maillage
mesh_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/ks7_v5_1.3_4326.shp'
gdf_mesh = gpd.read_file(mesh_path) # 4326
# lecture point open boundary et bouée
# Selection des point du maillage qui sont dans open boundary
cl_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/clv_chaincut_complete_largedomain_1.3_v2_4326.shp'
gdf_cl = gpd.read_file(cl_path) # 4326
# Selection des point des bouées
buoy_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/buoy.shp'
gdf_buoy = gpd.read_file(buoy_path) # 4326

#%% Loop selon tempête (martin, xynthia, celine)
# Lecture des données copernicus
tempest = ['martin', 'xynthia', 'celine']
type_data_list = ['oper', 'wave']
start_times = ["1999-12-26", "2010-02-25", "2024-10-27"]
end_times = ["1999-12-29", "2010-03-01", "2024-10-30"]
for ti in range(len(tempest)):
    for type_data in type_data_list:
        ds = xr.open_dataset(
            f'../datas_stage_m2/{tempest[ti]}/data_stream-{type_data}_stepType-instant.nc')
        if type_data == 'oper': # Ajout des données copernicus vent aux points du maillage
            oper_param = ['u10', 'v10', 'msl']
            for param in oper_param:
                gdf_mesh[f'{param}_{tempest[ti]}'] = 'NaN'
                for i in range(gdf_mesh.shape[0]):
                    param_point = ds[f'{param}'].sel(latitude=gdf_mesh.latitude.iloc[i], 
                                                     longitude=gdf_mesh.latitude.iloc[i], 
                                                     method='nearest')
                    # Mean
                    param_filtered = param_point.sel(valid_time=slice(start_times[ti], 
                                                                      end_times[ti]))
                    mean_value = param_filtered.mean(dim='valid_time')
                    gdf_mesh[f'{param}_{tempest[ti]}'].iloc[i] = mean_value.values 
                    
        elif type_data == 'wave': # Ajout des données copernicus houle aux points ob + bouées
            wave_param = ['mwd1', 'mwd', 'mwp', 'swh']
            for param in wave_param:
                gdf_mesh[f'{param}_{tempest[ti]}'] = 'NaN'
                for i in range(gdf_buoy.shape[0]):
                    param_point = ds[f'{param}'].sel(latitude=gdf_buoy.latitude.iloc[i], 
                                                     longitude=gdf_buoy.latitude.iloc[i], 
                                                     method='nearest')
                    # Mean
                    param_filtered = param_point.sel(valid_time=slice(start_times[ti], 
                                                                      end_times[ti]))
                    mean_value = param_filtered.mean(dim='valid_time')
                    for j in range(gdf_mesh.shape[0]):
                        # Trouver le point du maillage le plus proche
                        buoy_point = gdf_buoy.geometry.iloc[i]
                        distances = gdf_mesh.geometry.distance(buoy_point)
                        closest_index = distances.idxmin()
                        
                        # Stocker la valeur dans le maillage
                        gdf_mesh.at[closest_index, f'{param}_{tempest[ti]}'] = mean_value
                for i in range(gdf_cl.shape[0]):
                    param_point = ds[f'{param}'].sel(latitude=gdf_cl.latitude.iloc[i], 
                                                     longitude=gdf_cl.latitude.iloc[i], 
                                                     method='nearest')
                    # Mean
                    param_filtered = param_point.sel(valid_time=slice(start_times[ti], 
                                                                      end_times[ti]))
                    mean_value = param_filtered.mean(dim='valid_time')
                                
        else:
            print('Try again')

        
#%% Exporting the new shapefile
# exit_path = '../qgis_stage_m2/newDomain/swell_wind/ks7_v5_1.3_4326_wind&swell.shp'
# # In case the file already exists, it will be removed
# for fichier in glob.glob(f'{exit_path}.*'):
#     os.remove(fichier)
# # Saving the GeoDataFrame as a .shp file
# gdf.to_file(exit_path, driver='ESRI Shapefile')


