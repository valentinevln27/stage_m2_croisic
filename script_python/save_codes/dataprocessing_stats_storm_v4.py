# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:02:39 2025

@author: vanleene valentine
"""

#%% Needed librairies import
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
def find_minimas_below_threshold(dataframe, threshold):
    """
    Finds local minimas in a DataFrame column using scipy's find_peaks (by 
    inverting the signal), then filters them to keep only those below the given
    threshold. Made by Benjamin HERVY.
    
    Parameters
    ----------
    dataframe : pd.DataFrame or pd.Series
        1D numerical data (e.g., a single column from a DataFrame or a Series).
    threshold : float
        The value below which a local minimum is considered.

    Returns
    -------
    minimas :  pd.Series
        A Series of the same length as input, containing the local minima values 
        below the threshold at their corresponding positions.

    """
    # Find local minimas (peaks in the inverted signal)
    minimas_indices = find_peaks(-1 * dataframe.values)[0]
    # Take only minimas below the threshold
    minimas = pd.Series([value if (index in minimas_indices and value < threshold) 
                         else 0 for index, value in enumerate(dataframe.values)])
    return minimas

def create_minimas_windows(dataframe, outliers, swh_threshold):
    """
    For each outlier, this script identifies the previous and next local minima 
    of significant wave height (Hs) that are below a given threshold, and uses 
    them to define a time window around the outlier.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A DataFrame with a datetime index and at least one column 'Hs' 
        (significant wave height).
    outliers : pd.DataFrame or pd.Series
        A DataFrame or Series containing outlier timestamps (datetime index 
        expected).
    swh_threshold : int
        Threshold under which a local minimum is considered significant.

    Returns
    -------
    window_list : list
        DataFrame containing start and end timestamps of windows around each 
        outlier, defined by local minima before and after the outlier.

    """    
    # Find local minimas below threshold
    minimas_value_list = find_minimas_below_threshold(dataframe['Hs'], swh_threshold)
    # Get indices (as positions) where minima exist
    minimas_index_list = minimas_value_list.to_numpy().nonzero()
    minimas = dataframe.iloc[minimas_index_list[0]]
    
    # Create list to store windows
    window_list = []
    for date in outliers.index:
        # Find previous minima before the storm period
        try:
            low_boundary = minimas[minimas.index < date].iloc[-1].name  
        except:
            # If the script could not find the date before the one concerned, it 
            # just take the first value.
            low_boundary = minimas.index[0]
        # Find next minima after the storm period
        try:
            high_boundary = minimas[minimas.index > date].iloc[0].name  
        except:
            # If the script could not find the date before the one concerned, it 
            # just take the last value.
            high_boundary = dataframe.index[-1]
            
        # Add the window if it doesn't already exist in the list (avoids duplicates)        
        current_window = [low_boundary, high_boundary]
        if current_window in window_list:
            pass
        else:
            window_list.append(current_window)
    print("{} generated windows".format(len(window_list)))
    # Return a dataframe of the windows
    window_list = pd.DataFrame(window_list, columns=['start', 'end'])
    return window_list

def is_within_merged_periods(start, end, merged_periods):
    """
    Checks if a given period [start, end] is fully contained within any of the 
    merged periods.

    Parameters
    ----------
    start : pd.Timestamp
        Start of the period to check.
    end : pd.Timestamp
        End of the period to check.
    merged_periods : list of tuple
        List of (start, end) tuples representing merged time periods, where each 
        element is a tuple of pd.Timestamp.

    Returns
    -------
    tuple or None
        Returns the (m_start, m_end) tuple of the merged period that contains 
        [start, end], or None if the period is not fully contained in any of the 
        merged periods.

    """
    for m_start, m_end in merged_periods:
        if start >= m_start and end <= m_end:
            return (m_start, m_end)  # return enlarged version
    return None

def find_maximas_above_threshold(dataframe, threshold):
    """
    Finds local maximas in a DataFrame column using scipy's find_peaks, then 
    filters them to keep only those above the given threshold. Modified function 
    from Benjamin HERVY.
    
    Parameters
    ----------
    dataframe : pd.DataFrame or pd.Series
        1D numerical data (e.g., a single column from a DataFrame or a Series).
    threshold : float
        The value above which a local maximum is considered.

    Returns
    -------
    maximas : pd.Series
        A Series of the same length as input, containing the local maxima values 
        above the threshold at their corresponding positions.

    """
    # Find local maximas (peaks in the inverted signal)
    maximas_indices = find_peaks(dataframe.values)[0]
    # Take only maximas below the threshold
    maximas = pd.Series([value if (index in maximas_indices and value > threshold) 
                         else 0 for index, value in enumerate(dataframe.values)])
    return maximas

def create_maximas_windows(dataframe, outliers, patm_threshold):
    """
    For each outlier, this script identifies the previous and next local maxima 
    of the atmospherical pressure (Patm) that are above a given threshold, and 
    uses them to define a time window around the outlier.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A DataFrame with a datetime index and at least one column 'Patm' 
        (atmospherical pressure).
    outliers : pd.DataFrame or pd.Series
        A DataFrame or Series containing outlier timestamps (datetime index 
        expected).
    swh_threshold : int
        Threshold above which a local maximum is considered significant.

    Returns
    -------
    window_list : list
        DataFrame containing start and end timestamps of windows around each 
        outlier, defined by local maxima before and after the outlier.

    """    
    # Find local maximas above threshold
    maximas_value_list = find_maximas_above_threshold(dataframe['Patm'], 
                                                      patm_threshold)
    # Get indices (as positions) where maxima exist
    maximas_index_list = maximas_value_list.to_numpy().nonzero()
    maximas = dataframe.iloc[maximas_index_list[0]]
    
    # Create list to store windows
    window_list = []
    for date in outliers.index:
        # Find previous maxima before the storm period
        try:
            low_boundary = maximas[maximas.index < date].iloc[-1].name
        except IndexError:
            # If the script could not find the date before the one concerned, it 
            # just take the first value.
            low_boundary = maximas.index[0]
        # Find next maxima after the storm period
        try:
            high_boundary = maximas[maximas.index > date].iloc[0].name
        except IndexError:
            # If the script could not find the date before the one concerned, it 
            # just take the last value.
            high_boundary = dataframe.index[-1]
            
        # Add the window if it doesn't already exist in the list (avoids duplicates)
        current_window = [low_boundary, high_boundary]
        if current_window not in window_list:
            window_list.append(current_window)
            
    print("{} generated windows".format(len(window_list)))
    # Return a dataframe of the windows
    window_list = pd.DataFrame(window_list, columns=['start', 'end'])
    return window_list

def find_overlapping_windows(df1, df2, tolerance_hours=6):
    """
    Trouve les tempêtes qui se chevauchent ou sont proches dans le temps 
    entre deux DataFrames de fenêtres de tempêtes.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Deux DataFrames avec colonnes 'start' et 'end' de type datetime.
    tolerance_hours : int
        Tolérance en heures à appliquer aux comparaisons de fenêtres.

    Returns
    -------
    overlapping : pd.DataFrame
        DataFrame des fenêtres de df1 qui ont une correspondance dans df2 
        selon les critères de chevauchement ou proximité.
    """
    tolerance = pd.Timedelta(hours=tolerance_hours)
    matches = []

    for _, row1 in df1.iterrows():
        start1 = row1['start'] - tolerance
        end1 = row1['end'] + tolerance

        for _, row2 in df2.iterrows():
            start2, end2 = row2['start'], row2['end']

            # Teste le chevauchement ou proximité
            if start1 <= end2 and end1 >= start2:
                merged_start = min(row1['start'], row2['start'])
                merged_end = max(row1['end'], row2['end'])
                matches.append({'start': merged_start, 'end': merged_end})
                break  # Une seule correspondance suffit

    return pd.DataFrame(matches)

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

#%% Determining the threshold with the 95 percentile of Hs
seuils = [] # List to store all calculated threshold
# Determining the threshold with the 95 percentile of Hs for each point
for i, (lat, lon) in enumerate(zip(gdf_points['lat'], gdf_points['lon'])):
    # Select the closest wave data point to the current location
    Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')    
    # Dataframe of Hs values
    df_Hs = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values),
        'Hs': Hs_point.values}).set_index('time')
    # Compute the 95th percentile of Hs at this location
    percentile_95 = df_Hs['Hs'].quantile(0.95)
    # Store the threshold
    seuils.append(percentile_95)

# Convert list of thresholds to a pandas Series to use an histogramm
s = pd.Series(seuils)
# Plot a histogram of the thresholds and get bin counts
plt.figure(dpi=300)
counts, bins, _ = plt.hist(s, bins=20, color='mediumaquamarine', edgecolor='seagreen')

# Identify the bin with the highest frequency (mode)
max_bin_index = np.argmax(counts)
# Select a representative Hs value: the center of the most populated bin
selected_Hs = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2

# Add legend
plt.title('95th percentile of significant wave height (Hs) at each point\n'
          'along the open boundary', fontweight='bold')
plt.axvline(selected_Hs, linestyle='--', linewidth=2, color='forestgreen', 
            label=f'Bin peak ≈ {selected_Hs:.2f}')
plt.xlabel('Significant wave height (m)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#%% Identifiing the storms 
all_windows = [] # Will store storm windows for each valid point
valid_indices = [] # Will store indices of points with enough storms
# Identifiing storms periods for each point
for i, (lat, lon) in enumerate(zip(gdf_points['lat'], gdf_points['lon'])):
    print(f'Point n°{i+1}')
    # Get the nearest significant wave height (Hs) data for the current point
    Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')
    df_Hs = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values),
        'Hs': Hs_point.values}).set_index('time')
    # Identify extreme wave height events using Peaks Over Threshold (POT) method
    extremes_Hs = get_extremes(df_Hs['Hs'], "POT", threshold=selected_Hs, r="24h")
    # Create windows around those wave height extremes using surrounding minima
    storm_windows1 = create_minimas_windows(df_Hs, extremes_Hs, selected_Hs)
    
    # Get the nearest atmospheric pressure (msl) data for the current point
    patm_point = weather_data['msl'].sel(latitude=lat, longitude=lon, method='nearest')
    df_patm = pd.DataFrame({
        'time': pd.to_datetime(patm_point.time.values),
        'Patm': patm_point.values}).set_index('time')
    # Identify low-pressure events (below 980 hPa)
    low_pressure_events = df_patm[df_patm['Patm'] < 99000]
    # Create windows around low-pressure events using surrounding maxima
    storm_windows2 = create_maximas_windows(df_patm, low_pressure_events, 99000)
    
    storm_windows = find_overlapping_windows(storm_windows1, storm_windows2, 
                                             tolerance_hours=3)

    # Filter out points with too few storms (due to a highter sea bottom)
    if len(storm_windows) < 12 : # May need adjustment depending on the threshold
        print('Next')
        continue # Skipping the point
    else:
        all_windows.append(storm_windows) # Save valid storm windows
        valid_indices.append(i) # Save index of valid point

# Combine storm windows from all valid points into one list
all_periods = pd.concat([
    df[['start', 'end']] for df in all_windows
    ]).drop_duplicates().sort_values(by='start').reset_index(drop=True)

# Merge overlapping storm periods into single continuous periods
merged_periods = []
current_start, current_end = all_periods.iloc[0]['start'], all_periods.iloc[0]['end']
for i in range(1, len(all_periods)):
    period_i = all_periods.iloc[i]
    if period_i['start'] <= current_end:  # If periods overlap
        current_end = max(current_end, period_i['end']) # Extend current period
    else:
        merged_periods.append((current_start, current_end))
        current_start, current_end = period_i['start'], period_i['end'] # Save and start new one
merged_periods.append((current_start, current_end)) # Add last period

# Reconstruct DataFrames where each storm is replaced by its merged version
merged_common = []
for df in all_windows:
    new_rows = []
    for _, row in df.iterrows():
        match = is_within_merged_periods(row['start'], row['end'], merged_periods)
        if match:
            new_row = row.copy()
            new_row['start'], new_row['end'] = match  # Replace with merged period
            new_rows.append(new_row)
    # Keep only unique merged windows
    merged_common.append(pd.DataFrame(new_rows).drop_duplicates(subset=['start', 'end']))

# Identify storms that are present in all merged DataFrames (i.e. all locations)
all_storm_periods = [set(zip(df['start'], df['end'])) for df in merged_common]
common_storms = set.intersection(*all_storm_periods) # Finding common storms
# Keep only the common storms across all points
common_storms_windows = [
    df[df.apply(lambda row: (row['start'], row['end']) in common_storms, axis=1)]
    for df in merged_common]

#%% Calculating paramters for each storm periods at each valid points location
# Select only valid points (those with enough identified storms)
filtered_gdf_points = gdf_points.iloc[valid_indices].reset_index(drop=True)
merged_all_points = []  # List to store storm parameter data for each point
# Loop over all valid points
for i, (lat, lon) in enumerate(zip(filtered_gdf_points['lat'], filtered_gdf_points['lon'])):
    # Extracting swell data (Hs, Dp, Tp) for the current point
    Hs_point = swell_data['VHM0'].sel(latitude=lat, longitude=lon, method='nearest')      
    dfs_param = pd.DataFrame({
        'time': pd.to_datetime(Hs_point.time.values)}).set_index('time') 
    swell_param = ['VHM0', 'VPED', 'VTPK']
    for param in swell_param:
        param_point = swell_data[param].sel(latitude=lat, longitude=lon, 
                                             method='nearest')  
        dfs_param[param] = param_point.values  
    
    # Extract weather data (pressure, wind) for the same point
    w_point = weather_data['msl'].sel(latitude=lat, longitude=lon, method='nearest')
    dfw_param = pd.DataFrame({
        'time': pd.to_datetime(w_point.time.values)}).set_index('time')
    weather_param = ['msl', 'u10', 'v10']
    for param in weather_param:
        param_point = weather_data[param].sel(latitude=lat, longitude=lon, 
                                             method='nearest')  
        dfw_param[param] = param_point.values  
    
    # Retrieve storm time windows detected at this point
    storm_windows = common_storms_windows[i]
    
    records_s = []  # List to store swell parameters for each storm
    records_w = []  # List to store weather parameters for each storm
    for start, end in storm_windows.values:
        # Select swell data during the storm window
        params_filtered = dfs_param.loc[start:end]
        # Compute average values of significant wave height, peak direction, 
        # and period
        records_s.append({
            'start': start,
            'end': end,
            'Average Hs (m)': params_filtered['VHM0'].mean(),
            'Average Dp (°)': params_filtered['VPED'].mean(),
            'Average Tp (s)': params_filtered['VTPK'].mean()})
        try:
            # Select weather data during the storm window
            paramw_filtered = dfw_param.loc[start:end]
            # Compute wind speed from u10 and v10 components
            wind_speed = np.sqrt(paramw_filtered['u10']**2 + paramw_filtered['v10']**2)
            # Get the index of the maximum wind speed
            max_idx = wind_speed.idxmax()
            # Calculate wind direction in degrees (meteorological convention)
            wind_dir = (180 + np.arctan2(paramw_filtered['u10'], 
                                         paramw_filtered['v10']) * 180 / np.pi) % 360
            # Store weather-related storm statistics
            records_w.append({
                'start': start,
                'end': end,
                'Minimal Patm (Pa)': paramw_filtered['msl'].min(),
                'Maximal wind speed (m/s)': max(wind_speed),
                'Most frequent wind direction (°)': wind_dir.mode()[0],
                'u10 corresponding to\nthe maximal speed(m/s)': paramw_filtered.loc[max_idx, 'u10'],
                'v10 corresponding to\nthe maximal speed(m/s)': paramw_filtered.loc[max_idx, 'v10']})
        except:
            # If data is missing or can't be computed, store NaNs
            records_w.append({
                'start': start,
                'end': end,
                'Minimal Patm (Pa)': np.nan,
                'Maximal wind speed (m/s)': np.nan,
                'Most frequent wind direction (°)': np.nan,
                'u10 corresponding to\nthe maximal speed (m/s)': np.nan,
                'v10 corresponding to\nthe maximal speed (m/s)': np.nan})
    
    # Create DataFrames for swell and weather parameters
    dfs_point = pd.DataFrame(records_s)
    dfw_point = pd.DataFrame(records_w)
    # Merge the two on the storm time windows
    merged_point = pd.merge(dfs_point, dfw_point, on=['start', 'end'])
    # Store all parameters results for the point
    merged_all_points.append(merged_point)

# Remove duplicates of storms's parameters across all point 
unique_dfs = []
for df in merged_all_points:
    if not any(df.equals(existing_df) for existing_df in unique_dfs):
        unique_dfs.append(df)

# Associate each unique DataFrame to the IDs of corresponding points
grouped_ids = [[] for _ in unique_dfs]
for i, df in enumerate(merged_all_points):  
    for j, unique_df in enumerate(unique_dfs):
        if df.equals(unique_df):
            grouped_ids[j].append(filtered_gdf_points.id[i])  
            break

#%% Graphical representation
# Define the list of swell and pressure parameters to be plotted
swell_patm_params = ['Average Hs (m)', 'Average Dp (°)', 'Average Tp (s)', 
                     'Minimal Patm (Pa)']
# Define the list of wind-related parameters to be plotted
wind_params = ['Maximal wind speed (m/s)', 'Most frequent wind direction (°)', 
               'u10 corresponding to\nthe maximal speed(m/s)', 
               'v10 corresponding to\nthe maximal speed(m/s)']
for i, merged_point in enumerate(unique_dfs):
    # Create a 4x2 grid of subplots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 17), sharex='col')
    
    # Plot swell and atmospheric pressure parameters in the left column
    for j, param in enumerate(swell_patm_params):
        ax = axes[j, 0]
        ax.plot(merged_point['start'], merged_point[param], marker='o', 
                linestyle='-', color='lightseagreen')
        ax.set_ylabel(param, fontsize=19)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
    
    # Plot wind parameters in the right column
    for j, param in enumerate(wind_params):
        ax = axes[j, 1]
        ax.plot(merged_point['start'], merged_point[param], marker='o', 
                linestyle='-', color='blue', linewidth=2.5, alpha=0.5)
        ax.set_ylabel(param, fontsize=19)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

    # Format the x-axis of the last row with labels and rotated ticks
    for ax in axes[-1, :]:
        ax.set_xlabel('Start of the storm', fontsize=19)
        ax.tick_params(axis='x', rotation=45)
    
    # Identify the range of point IDs corresponding to the current plot
    min_id = min(grouped_ids[i])
    max_id = max(grouped_ids[i])
    
    # Legend
    fig.suptitle(f'Swell and weather parameters at points {min_id} to {max_id}',
                 fontsize=35, fontweight='bold')
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.tight_layout()
    plt.show()

    # break
    
#%% Stats(most frequent values each having a plot)
# Get the list of all parameter columns
all_params = [col for col in unique_dfs[0].columns if col not in ['start', 'end']]
# Loop through each parameter to analyze and plot its distribution
for param in all_params:
    # Combine all parameter data across unique storm datasets and calculate the 
    # median of the parameter
    data_param = (
    pd.concat([df[['start', 'end', param]] for df in unique_dfs])
        .groupby(['start', 'end'], as_index=False).median()
        .sort_values(by='start').reset_index(drop=True))
    
    # Create a histogram of the parameter
    plt.figure(dpi=300)
    counts, bins, _ = plt.hist(data_param[param], bins=20, color='mediumaquamarine',
                               edgecolor='seagreen')
    plt.title(f'{param}')
        
    # Identify the two most frequent bins (peaks in the histogram)
    top2_indices = counts.argsort()[-2:][::-1]
    for i, idx in enumerate(top2_indices):
        bin_center = (bins[idx] + bins[idx + 1]) / 2 # Compute bin center
        plt.axvline(x=bin_center, linestyle='--', linewidth=2, color='forestgreen',
                    label=f'Bin peak {i+1} ≈ {bin_center:.2f}')    
    
    # Add legend
    plt.xlabel(f'{param}', fontweight='bold')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Graphical representation of the stats in a single plot
# Get the list of all parameter columns
all_params = [col for col in unique_dfs[0].columns if col not in ['start', 'end']]
# Create a grid of subplots for visualizing each parameter
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 14), dpi=300)
axes = axes.flatten() # Flatten the 2D array of axes for easier indexing

# Plot each parameters its histogram
for i, param in enumerate(all_params):
    ax = axes[i]
    
    # Selecting the parameter's data from all points
    data_param = (
        pd.concat([df[['start', 'end', param]] for df in unique_dfs])
        .groupby(['start', 'end'], as_index=False)
        .median()
        .sort_values(by='start')
        .reset_index(drop=True))
    
    # Calculation of the IQR and the number of bins using the Freedman-Diaconis method
    # Q1 = data_param[param].quantile(0.25)
    # Q3 = data_param[param].quantile(0.75)
    # IQR = Q3 - Q1
    # n = len(data_param[param])
    # bin_width = 2 * IQR / (n ** (1 / 3))  
    # # Number of bins
    # bin_count = int(np.ceil((data_param[param].max() - data_param[param].min()) / bin_width))
        
    # Histogram using a fixed number of bins 
    counts, bins, _ = ax.hist(data_param[param], bins=20, color='mediumaquamarine',
                              edgecolor='seagreen')
    
    # Identify and annotate the two most frequent bins (modes)
    top2_indices = counts.argsort()[-2:][::-1]
    for j, idx in enumerate(top2_indices):
        bin_center = (bins[idx] + bins[idx + 1]) / 2
        ax.axvline(x=bin_center, linestyle='--', linewidth=2, color='forestgreen',
                   label=f'Bin peak {j+1} ≈ {bin_center:.2f}')
    
    # Add legend 
    ax.set_title(f'{param}', fontsize=25, fontweight='bold')
    ax.set_xlabel(f'{param}', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=15)

plt.tight_layout()
plt.show()

#%% Stats on historical storms
# Load historical storm data of the Observatoire de la côte Nouvelle-Aquitaine
# from an Excel file 
file_storm = '../excel/storms.xlsx'
remarquable_storm = pd.read_excel(file_storm)

# Use the storm time windows detected at the first point as reference
storms_all_windows = common_storms_windows[0]

# Convert date columns to datetime format for accurate comparison
remarquable_storm['start'] = pd.to_datetime(remarquable_storm['start'])
remarquable_storm['end'] = pd.to_datetime(remarquable_storm['end'])
storms_all_windows['start'] = pd.to_datetime(storms_all_windows['start'])
storms_all_windows['end'] = pd.to_datetime(storms_all_windows['end'])

# Initialize lists to store matched storms and warning messages
matched_storms = []
warnings = []

# Compare each historical storm against the detected storm periods
for i, row in remarquable_storm.iterrows():
    r_start = row['start']
    r_end = row['end']
    r_name = row.get('storm', f'Storm_{i}') # Use name if available, else create default name

    found = False # Flag to check if the storm was matched
    for _, s_row in storms_all_windows.iterrows():
        s_start = s_row['start']
        s_end = s_row['end']

        # Case 1: The historical storm is entirely included within a detected storm period
        if s_start <= r_start and s_end >= r_end:
            matched_storms.append({
                'name': r_name,
                'start': r_start,
                'end': r_end,
                'status': 'included ou equal'
            })
            found = True
            break
                
        # Case 2: The historical storm partially overlaps with a detected storm period
        elif (s_start <= r_end and s_end >= r_start):
            warnings.append(
                f"⚠️ Storm '{r_name}' ({r_start.date()} to {r_end.date()}) "
                f"overlapping a detected period ({s_start.date()} to {s_end.date()})"
                " but isn't entirely included."
            )
            matched_storms.append({
                'name': r_name,
                'start': r_start,
                'end': r_end,
                'status': 'partial overlapping'
            })
            found = True
            break
    
    # If no match is found at all
    if not found:
        warnings.append(f"❌ Storm '{r_name}' ({r_start.date()} to {r_end.date()})"
                        "isn't detected into selected periods.")

# Convert matched storms list to a DataFrame for further use or export
df_matched = pd.DataFrame(matched_storms)
# Optionally save to CSV:
# df_matched.to_csv("tempetes_verifiees.csv", index=False)

# Print summary of comparison and warning messages
print('Comparison ended.')
for w in warnings:
    print(w)