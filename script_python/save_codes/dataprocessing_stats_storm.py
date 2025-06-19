# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:45:28 2025

@author: vanleene valentine
"""

#%% Importation des librairies
import geopandas as gpd
import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyextremes import get_extremes
from pyextremes.plotting import plot_extremes
from scipy.signal import argrelmax, argrelmin, find_peaks
from datetime import datetime, timedelta

#%% Useful functions
def find_minimas_below_threshold(dataframe: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Finds local minimas using scipy find_peaks method
    Filter them to get those below given threshold only
    """
    minimas_indices = find_peaks(-1 * dataframe.values)[0]
    minimas = [value if (index in minimas_indices and value < threshold) else 0 for index, value in enumerate(dataframe.values)]
    return pd.Series(minimas)

def create_minimas_windows(dataframe: pd.DataFrame, outliers: pd.DataFrame, swh_threshold: int) -> list:
    # For each outlier, find previous and next local minimas
    window_list = []
    
    minimas_value_list = find_minimas_below_threshold(dataframe['Hs'], swh_threshold)
    minimas_index_list = minimas_value_list.to_numpy().nonzero()
    minimas = dataframe.iloc[minimas_index_list[0]]
    
    for date in outliers.index:
        try:
            low_boundary = minimas[minimas.index < date].iloc[-1].name  # Utiliser l'index pour accéder aux dates
        except:
            # Could not find date before the one concerned. Just taking the first value
            low_boundary = minimas.index[0]
        try:
            high_boundary = minimas[minimas.index > date].iloc[0].name  # Utiliser l'index pour accéder aux dates
        except:
            # Could not find date after the one concerned. Just taking the last value
            high_boundary = dataframe.index[-1]
        
        current_window = [low_boundary, high_boundary]
        if current_window in window_list:
            pass
        else:
            window_list.append(current_window)
    print("{} generated windows".format(len(window_list)))
    window_list = pd.DataFrame(window_list, columns=['start', 'end'])
    return window_list

# Conserver les tempêtes qui tombent dans les périodes fusionnées
def is_within_merged_periods(start, end, merged_periods):
    for m_start, m_end in merged_periods:
        if start >= m_start and end <= m_end:
            return (m_start, m_end)  # retourne la version élargie
    return None

#%%
# Ouverture du fichier de point (WGS83, openBoundary) 45 points en tout
points_filepath = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/open_boundary_points.shp'
gdf_points = gpd.read_file(points_filepath)

# Ouverture des fichier de données houle
swell_filepath = 'D:/stage_m2/datas_stage_m2/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1745831411603.nc'
swell_data = xr.open_dataset(swell_filepath)
swell_param = ['VHM0', 'VPED', 'VTPK']
weather_filepath = '../datas_stage_m2/data_wind/compiled_data/combined_data.nc'
weather_data = xr.open_dataset(weather_filepath)
weather_param = ['msl', 'u10', 'v10']
seuils = []

merged_all_points = [] 

for i, (lat, lon) in enumerate(zip(gdf_points['lat'], gdf_points['lon'])):
    # prendre le point le plus proche des données de houle
    Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')    
    df_Hs = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values),
        'Hs': Hs_point.values}).set_index('time')
    percentile_99 = df_Hs['Hs'].quantile(0.99)
    seuils.append(percentile_99)
s = pd.Series(seuils)
selected_Hs = s.mode()[0] # most frequent value

# Boucle por chaque point 
for i, (lat, lon) in enumerate(zip(gdf_points['lat'], gdf_points['lon'])):
    # prendre le point le plus proche des données de houle
    Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')    
    df_Hs = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values),
        'Hs': Hs_point.values}).set_index('time')
    
    df_param = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values)}).set_index('time')
    for param in swell_param:
        param_point = swell_data[param].sel(latitude=lat, longitude=lon, 
                                             method='nearest')  
        df_param[param] = param_point.values  
    
    w_point = weather_data['msl'].sel(latitude=lat, longitude=lon, method='nearest')
    dfw_param = pd.DataFrame({
        'time': pd.to_datetime(w_point.time.values)}).set_index('time')
    for param in weather_param:
        param_point = weather_data[param].sel(latitude=lat, longitude=lon, 
                                             method='nearest')  
        dfw_param[param] = param_point.values  
        
    extremes = get_extremes(df_Hs['Hs'], "POT", threshold=selected_Hs, r="24h")
    storm_windows = create_minimas_windows(df_Hs, extremes, percentile_99)
    
    if len(storm_windows) < 100 : 
        print('Next')
        continue # passe au point suivant
    
    records = []
    recordsw = []
    for start, end in storm_windows.values:
        param_filtered = df_param.loc[start:end]
        records.append({
            'start': start,
            'end': end,
            'Hs_mean': param_filtered['VHM0'].mean(),
            'Dp_mean': param_filtered['VPED'].mean(),
            'Tp_mean': param_filtered['VTPK'].mean()})
        try:
            paramw_filtered = dfw_param.loc[start:end]
            recordsw.append({
                'start': start,
                'end': end,
                'msl_min': paramw_filtered['msl'].min(),
                'u10_max': paramw_filtered['u10'].max(),
                'v10_max': paramw_filtered['v10'].max()})
        except:
            recordsw.append({
                'start': start,
                'end': end,
                'msl_min': 0,
                'u10_max': 0,
                'v10_max': 0})
    df_point = pd.DataFrame(records)
    dfw_point = pd.DataFrame(recordsw)
    merged_point = pd.merge(df_point, dfw_point, on=['start', 'end'])
    
    merged_all_points.append(merged_point)

#%%
# Discarding duplicates
unique_dfs = []
for df in merged_all_points:
    if not any(df.equals(existing_df) for existing_df in unique_dfs):
        unique_dfs.append(df)

#%%
# All storms periods
all_periods = pd.concat([
    df[['start', 'end']] for df in unique_dfs
]).drop_duplicates().sort_values(by='start').reset_index(drop=True)

# Fusionning storms if common dates (chevauchement)
merged_periods = []
current_start, current_end = all_periods.iloc[0]['start'], all_periods.iloc[0]['end']
for i in range(1, len(all_periods)):
    period_i = all_periods.iloc[i]
    if period_i['start'] <= current_end:  # chevauchement
        current_end = max(current_end, period_i['end'])
    else:
        merged_periods.append((current_start, current_end))
        current_start, current_end = period_i['start'], period_i['end']
merged_periods.append((current_start, current_end))

# Reconstruct dataframes with fused storm periods
merged_common = []
for df in unique_dfs:
    new_rows = []
    for _, row in df.iterrows():
        match = is_within_merged_periods(row['start'], row['end'], merged_periods)
        if match:
            new_row = row.copy()
            new_row['start'], new_row['end'] = match  # utiliser la période élargie
            new_rows.append(new_row)
    merged_common.append(pd.DataFrame(new_rows).drop_duplicates(subset=['start', 'end']))

#%%
all_storm_periods = [set(zip(df['start'], df['end'])) for df in merged_common]

# Finding common storms
common_storms = set.intersection(*all_storm_periods)

# Conserving only common storms for all points
merged_common2 = [
    df[df.apply(lambda row: (row['start'], row['end']) in common_storms, axis=1)]
    for df in merged_common]

#%% Graphical representation
swell_params = ['Hs_mean', 'Dp_mean', 'Tp_mean']
weather_params = ['msl_min', 'u10_max', 'v10_max']

for i, merged_point in enumerate(merged_all_points):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 10), sharex='col')
    
    # Colonne 1 : swell parameters
    for j, param in enumerate(swell_params):
        ax = axes[j, 0]
        ax.plot(merged_point['start'], merged_point[param], marker='o', 
                linestyle='-', color='lightseagreen')
        ax.set_ylabel(param, fontsize=18)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
    
    # Colonne 2 : weather parameters
    for j, param in enumerate(weather_params):
        ax = axes[j, 1]
        ax.plot(merged_point['start'], merged_point[param], marker='o', 
                linestyle='-', color='blue', linewidth=2.5, alpha=0.5)
        ax.set_ylabel(param, fontsize=18)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

    # Time ticks
    for ax in axes[-1, :]:
        ax.set_xlabel('Start of the storm', fontsize=18)
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(f'Swell and weather parameters at point {i+1}', fontsize=35, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    break
    