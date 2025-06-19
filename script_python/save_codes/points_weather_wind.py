# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 08:55:24 2025

@author: vanleene valentine
"""

# couche de point
# 3 points les plus proches
# idw

#%% Needed librairies import
import geopandas as gpd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob

#%%
# Open boundary points
points_filepath = '../qgis_stage_m2/newlargerdomain/OB_points_4326.shp'
# gdf_points = gpd.read_file(points_filepath)
# gdf_4326 = gdf_points.to_crs(epsg=4326)
gdf_4326 = gpd.read_file(points_filepath)
gdf_4326['lat'] = [point.y for point in gdf_4326.geometry]
gdf_4326['lon'] = [point.x for point in gdf_4326.geometry]

# Open swell data file
swell_filepath = '../datas_stage_m2/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1749021343992.nc'
swell_data = xr.open_dataset(swell_filepath)

# Open wind data file
weather_filepath = '../datas_stage_m2/data_weather/compiled_data/combined_data.nc'
weather_data = xr.open_dataset(weather_filepath)

#%% trouver le point le plus proche et télécharger les données
swell_params_map = {'Hs': 'VHM0', 'Dp': 'VPED', 'Tp': 'VTPK'}
weather_params_map = {'patm': 'msl', 'u10': 'u10', 'v10': 'v10'}
all_params = list(swell_params_map.keys()) + list(weather_params_map.keys())

# Initialisation de la base temporelle
swell_time_values = swell_data['time'].values
swell_time_df = pd.DataFrame({'time': pd.to_datetime(swell_time_values)})

weather_time_values = weather_data['time'].values
weather_time_df = pd.DataFrame({'time': pd.to_datetime(weather_time_values)})

# Boucle sur les paramètres
for param in all_params:
    # Choisir la bonne source de données et la bonne variable
    if param in swell_params_map:
        df_data = swell_time_df.copy()
        data_source = swell_data
        var_name = swell_params_map[param]
        out_folder = '../datas_stage_m2/data_swell_points2'
    else:
        df_data = weather_time_df.copy()
        data_source = weather_data
        var_name = weather_params_map[param]
        out_folder = '../datas_stage_m2/data_weather_points2'

    # Extraire les données pour chaque point
    for i, (lat, lon) in enumerate(zip(gdf_4326['lat'], gdf_4326['lon'])):
        param_point = data_source[var_name].sel(latitude=lat, longitude=lon, method='nearest')
        df_data[f'node_{gdf_4326.newid[i]}'] = param_point.values

    # Sauvegarder
    # df_data.set_index('time', inplace=True)
    output_path = os.path.join(out_folder, f'{param}_deepBoundary_new.txt')
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    df_data.to_csv(output_path, sep = ' ', index=False)

#%% Test to see if everything is working
def load_data(filename):
    """
    Loads time series data from a file, combines date and time into a single 
    datetime index, and returns a cleaned DataFrame.

    Parameters
    ----------
    filename : str
        Path to the file containing the data. The file should contain separate
        columns for date (as index) and time (as 'Time'), along with one or more
        data columns.

    Returns
    -------
    dataframe : pandas.DataFrame
        A DataFrame indexed by datetime, with the 'Time' column removed.
    """
    # print(filename)
    df = pd.read_csv(filename, sep=r'\s+')
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

base_path = '../datas_stage_m2/data_weather_points2/'
files = {
    'df_patm': 'patm_deepBoundary_new.txt',
    'df_u10': 'u10_deepBoundary_new.txt',
    'df_v10': 'v10_deepBoundary_new.txt'
    }
df_patm, df_u10, df_v10 = [load_data(base_path + fname) for fname in files.values()]

base_path = '../datas_stage_m2/data_swell_points2/'
files = {
    'df_hs': 'Hs_deepBoundary_new.txt',
    'df_dp': 'Dp_deepBoundary_new.txt',
    'df_tp': 'Tp_deepBoundary_new.txt'
    }
df_hs, df_dp, df_tp = [load_data(base_path + fname) for fname in files.values()]





