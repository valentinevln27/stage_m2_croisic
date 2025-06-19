# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:59:36 2025

@author: vanleene valentine
"""
#%% Librairies import
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

def is_within_merged_periods(start, end, merged_periods):
    for m_start, m_end in merged_periods:
        if start >= m_start and end <= m_end:
            return (m_start, m_end)  # return enlarged version
    return None

#%% Data extraction
# Open boundary points
points_filepath = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/open_boundary_points.shp'
gdf_points = gpd.read_file(points_filepath)

# Open swell data file
swell_filepath = 'D:/stage_m2/datas_stage_m2/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1745831411603.nc'
swell_data = xr.open_dataset(swell_filepath)

# Open wind data file
weather_filepath = '../datas_stage_m2/data_wind/compiled_data/combined_data.nc'
weather_data = xr.open_dataset(weather_filepath)

#%% Needed lists 
swell_param = ['VHM0', 'VPED', 'VTPK']
weather_param = ['msl', 'u10', 'v10']

#%% Determining the threshold with the 99 percentile of Hs
seuils = []
for i, (lat, lon) in enumerate(zip(gdf_points['lat'], gdf_points['lon'])):
    # prendre le point le plus proche des données de houle
    Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')    
    df_Hs = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values),
        'Hs': Hs_point.values}).set_index('time')
    percentile_95 = df_Hs['Hs'].quantile(0.95)
    seuils.append(percentile_95)
s = pd.Series(seuils)
counts, bins, _ = plt.hist(s, bins=20)
max_bin_index = np.argmax(counts)
selected_Hs = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
# selected_Hs = s.mode()[0] # most frequent value

#%% Identifiing the storms 
all_windows = []
valid_indices = []
# Storms periods for each point
for i, (lat, lon) in enumerate(zip(gdf_points['lat'], gdf_points['lon'])):
    # Taking the closest point  
    Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')
    df_Hs = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values),
        'Hs': Hs_point.values}).set_index('time')
       
    extremes_Hs = get_extremes(df_Hs['Hs'], "POT", threshold=selected_Hs, r="24h")
    storm_windows = create_minimas_windows(df_Hs, extremes_Hs, selected_Hs)
    
    if len(storm_windows) < 100 : 
        print('Next')
        continue # passing to next point
    else:
        all_windows.append(storm_windows)
        valid_indices.append(i)  # store index of valid point

# All storms periods
all_periods = pd.concat([
    df[['start', 'end']] for df in all_windows
    ]).drop_duplicates().sort_values(by='start').reset_index(drop=True)

# Combining storms if overlap
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
for df in all_windows:
    new_rows = []
    for _, row in df.iterrows():
        match = is_within_merged_periods(row['start'], row['end'], merged_periods)
        if match:
            new_row = row.copy()
            new_row['start'], new_row['end'] = match  # utiliser la période élargie
            new_rows.append(new_row)
    merged_common.append(pd.DataFrame(new_rows).drop_duplicates(subset=['start', 'end']))


# Identifiing the storms present in all dataframes of points
all_storm_periods = [set(zip(df['start'], df['end'])) for df in merged_common]
common_storms = set.intersection(*all_storm_periods) # Finding common storms
# Conserving only common storms for all points
common_storms_windows = [
    df[df.apply(lambda row: (row['start'], row['end']) in common_storms, axis=1)]
    for df in merged_common]

#%% Calculating paramters for each storm periods at each points location
filtered_gdf_points = gdf_points.iloc[valid_indices].reset_index(drop=True)
merged_all_points = [] 
for i, (lat, lon) in enumerate(zip(filtered_gdf_points['lat'], filtered_gdf_points['lon'])):
    # Extracting swell data for point i
    Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')      
    dfs_param = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values)}).set_index('time')   
    for param in swell_param:
        param_point = swell_data[param].sel(latitude=lat, longitude=lon, 
                                             method='nearest')  
        dfs_param[param] = param_point.values  
    
    # Extracting weather data for point i
    w_point = weather_data['msl'].sel(latitude=lat, longitude=lon, method='nearest')
    dfw_param = pd.DataFrame({
        'time': pd.to_datetime(w_point.time.values)}).set_index('time')
    for param in weather_param:
        param_point = weather_data[param].sel(latitude=lat, longitude=lon, 
                                             method='nearest')  
        dfw_param[param] = param_point.values  
        
    storm_windows = common_storms_windows[i]
    
    records_s = []
    records_w = []
    for start, end in storm_windows.values:
        # Calculating swell parameters for each storms
        params_filtered = dfs_param.loc[start:end]
        records_s.append({
            'start': start,
            'end': end,
            'Hs_mean': params_filtered['VHM0'].mean(),
            'Dp_mean': params_filtered['VPED'].mean(),
            'Tp_mean': params_filtered['VTPK'].mean()})
        try:
            paramw_filtered = dfw_param.loc[start:end]
            wind_speed = np.sqrt(paramw_filtered['u10']**2 + paramw_filtered['v10']**2)
            max_idx = wind_speed.idxmax()
            wind_dir_deg = (np.degrees(np.arctan2(-paramw_filtered['u10'], -paramw_filtered['v10'])) + 360) % 360
            records_w.append({
                'start': start,
                'end': end,
                'msl_min': paramw_filtered['msl'].min(),
                'wind_speed': max(wind_speed),
                'wind_dir': wind_dir_deg.mode()[0],
                'u10_max': paramw_filtered.loc[max_idx, 'u10'],
                'v10_max': paramw_filtered.loc[max_idx, 'v10']})
        except:
            records_w.append({
                'start': start,
                'end': end,
                'msl_min': np.nan,
                'wind_speed': np.nan,
                'wind_dir': np.nan,
                'u10_max': np.nan,
                'v10_max': np.nan})
    dfs_point = pd.DataFrame(records_s)
    dfw_point = pd.DataFrame(records_w)
    merged_point = pd.merge(dfs_point, dfw_point, on=['start', 'end'])
    merged_all_points.append(merged_point)

# filtered_gdf_points['parameters'] = merged_all_points

# Deleting duplicates
unique_dfs = []
for df in merged_all_points:
    if not any(df.equals(existing_df) for existing_df in unique_dfs):
        unique_dfs.append(df)

# Associating each unique_df to a list of index (id) of corresponding points
grouped_ids = [[] for _ in unique_dfs]
for i, df in enumerate(merged_all_points):  
    for j, unique_df in enumerate(unique_dfs):
        if df.equals(unique_df):
            grouped_ids[j].append(filtered_gdf_points.id[i])  
            break
        
#%% Graphical representation
swell_patm_params = ['Hs_mean', 'Dp_mean', 'Tp_mean', 'msl_min']
wind_params = ['wind_speed', 'wind_dir', 'u10_max', 'v10_max']
for i, merged_point in enumerate(unique_dfs):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 10), sharex='col')
    
    # Column 1 : swell parameters and patm
    for j, param in enumerate(swell_patm_params):
        ax = axes[j, 0]
        ax.plot(merged_point['start'], merged_point[param], marker='o', 
                linestyle='-', color='lightseagreen')
        ax.set_ylabel(param, fontsize=18)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
    
    # Column 2 : wind parameters
    for j, param in enumerate(wind_params):
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
    
    # Points ids
    min_id = min(grouped_ids[i])
    max_id = max(grouped_ids[i])
    
    # Legend
    fig.suptitle(f'Swell and weather parameters at points {min_id} to {max_id}',
                 fontsize=35, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # break
    
#%% Stats 
# Faire histo pour chaque paramètres pour tous les points confondus
swell_patm_params = ['Hs_mean', 'Dp_mean', 'Tp_mean', 'msl_min']
wind_params = ['wind_speed', 'wind_dir', 'u10_max', 'v10_max']
all_Hs = pd.concat([
    df[['start', 'end', 'Hs_mean']] for df in unique_dfs
    ]).drop_duplicates().sort_values(by='start').reset_index(drop=True)

# Histogramm
plt.figure(dpi=300)
counts, bins, _ = plt.hist(all_Hs['Hs_mean'], bins=20, color='mediumaquamarine', edgecolor='seagreen')
plt.title('Mean Hs')
# Finding most frequent bin
max_bin_index = np.argmax(counts)
most_frequent_bin_center = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
# Add the obtained value on the graph
plt.axvline(x=most_frequent_bin_center, color='forestgreen', linestyle='--', linewidth=2,
            label=f'Bin peak ≈ {most_frequent_bin_center:.2f} m')
# Legend
plt.xlabel('Mean Hs (m)')
plt.ylabel('Fréquence')
plt.legend()
plt.tight_layout()
plt.show()

#%%
Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')
df_Hs = pd.DataFrame({
    'time': pd.to_datetime(Hs_point.time.values),
    'Hs': Hs_point.values
}).set_index('time')

df_Hs.index = pd.to_datetime(df_Hs.index)

# Xynthia
df_filtered = df_Hs.loc['2010-02-20 00:00:00':'2010-02-28 23:59:59']
df_filtered2 = df_Hs.loc['2010-02-27 00:00:00':'2010-02-28 23:59:59']

# Calculer Hs pendant Xynthia
max_Hs = df_filtered2['Hs'].max()
print(f'Hs moyen du 27 au 28 février 2010 : {max_Hs:.2f} m')

plt.figure(dpi=300)
plt.plot(df_filtered['Hs'], label='Hs')
plt.axhline(y=selected_Hs, color='r', linestyle='--', 
            label=f'99th centile of Hs ({selected_Hs:.2f} m)')
plt.tick_params(axis='x', rotation=45)
plt.xlabel('Time')
plt.ylabel('Hs (m)')
plt.title('Hs during Xynthia and before')
plt.legend()
plt.tight_layout()
plt.show()

