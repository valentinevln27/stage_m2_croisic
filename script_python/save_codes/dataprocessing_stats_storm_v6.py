# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:47:52 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""

"""

#%% Needed librairies import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyextremes import get_extremes
from pyextremes.plotting import plot_extremes
from scipy.signal import argrelmax, argrelmin, find_peaks
from datetime import datetime, timedelta

#%% Useful functions
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
    df = pd.read_csv(filename, sep=r'\s+')
    df['time'] = pd.to_datetime(df.index.astype(str) + ' ' + df['Time'].astype(str))
    df.set_index('time', inplace=True)
    return df.drop(columns=['Time'])

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
    minimas_value_list = find_minimas_below_threshold(dataframe, swh_threshold)
    # Get indices (as positions) where minima exist
    minimas_index_list = minimas_value_list.to_numpy().nonzero()
    minimas = dataframe.iloc[minimas_index_list[0]]
    
    # Create list to store windows
    window_list = []
    for date in outliers.index:
        # Find previous minima before the storm period
        try:
            low_boundary = minimas[minimas.index < date].index[-1]
        except:
            # If the script could not find the date before the one concerned, it 
            # just take the first value.
            low_boundary = minimas.index[0]
        # Find next minima after the storm period
        try:
            high_boundary = minimas[minimas.index > date].index[0]
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
    maximas_value_list = find_maximas_above_threshold(dataframe, 
                                                      patm_threshold)
    # Get indices (as positions) where maxima exist
    maximas_index_list = maximas_value_list.to_numpy().nonzero()
    maximas = dataframe.iloc[maximas_index_list[0]]
    
    # Create list to store windows
    window_list = []
    for date in outliers.index:
        # Find previous maxima before the storm period
        try:
            low_boundary = maximas[maximas.index < date].index[-1]
        except IndexError:
            # If the script could not find the date before the one concerned, it 
            # just take the first value.
            low_boundary = maximas.index[0]
        # Find next maxima after the storm period
        try:
            high_boundary = maximas[maximas.index > date].index[0]
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

#%% Data extraction
# Parameters data
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

# Load historical storm data of the Observatoire de la côte Nouvelle-Aquitaine
# from an Excel file 
file_storm = '../excel/storms.xlsx'
remarkable_storm = pd.read_excel(file_storm)
# Convert date columns to datetime format for accurate comparison
remarkable_storm['start'] = pd.to_datetime(remarkable_storm['start'])
remarkable_storm['end'] = pd.to_datetime(remarkable_storm['end']) + pd.Timedelta(days=1)

#%% Determining the thresholds of Hs and Patm
# Créer des listes pour stocker les résultats
hs_means = []
dp_means = []
tp_means = []
patm_mins = []
ws_maxs = []
wd_maxs = []
u10_maxs = []
v10_maxs = []

# Parcours des tempêtes remarquables
for _, row in remarkable_storm.iterrows():
    start, end = row['start'], row['end']
    
    # Sélectionner les données pendant la tempête
    hs_storm = df_hs.loc[start:end]
    dp_storm = df_dp.loc[start:end]
    tp_storm = df_tp.loc[start:end]
    patm_storm = df_patm.loc[start:end]
    u10_storm = df_u10.loc[start:end]
    v10_storm = df_v10.loc[start:end]
    
    # Calculs de la vitesse et la direction du vent
    ws_storm = np.sqrt(u10_storm**2 + v10_storm**2)
    wd_storm = (180 + np.arctan2(u10_storm, v10_storm) * 180 / np.pi) % 360
    
    wind_storm = pd.concat([
    u10_storm.rename(columns={u10_storm.columns[0]: 'u10'}),
    v10_storm.rename(columns={v10_storm.columns[0]: 'v10'}),
    ws_storm.rename(columns={ws_storm.columns[0]: 'ws'}),
    wd_storm.rename(columns={wd_storm.columns[0]: 'wd'})
    ], axis=1)
    
    # Calculs stats
    hs_mean = hs_storm.max().values[0] if not hs_storm.empty else np.nan
    dp_mean = dp_storm.max().values[0] if not dp_storm.empty else np.nan
    tp_mean = tp_storm.max().values[0] if not dp_storm.empty else np.nan
    patm_min = patm_storm.min().values[0] if not patm_storm.empty else np.nan
    ws_max = wind_storm['ws'].max() if not wind_storm.empty else np.nan
    wd_max = wind_storm['wd'].max() if not wind_storm.empty else np.nan
    max_idx = wind_storm['ws'].idxmax()
    u10_max = wind_storm.loc[max_idx, 'u10'] if not wind_storm.empty else np.nan
    v10_max = wind_storm.loc[max_idx, 'v10'] if not wind_storm.empty else np.nan
    
    hs_means.append(hs_mean)
    dp_means.append(dp_mean)
    tp_means.append(tp_mean)
    patm_mins.append(patm_min)
    ws_maxs.append(ws_max)
    wd_maxs.append(wd_max)
    u10_maxs.append(u10_max)
    v10_maxs.append(v10_max)
    

# Ajouter les résultats au DataFrame des tempêtes
remarkable_storm['Hs_mean'] = hs_means
remarkable_storm['Dp_mean'] = dp_means
remarkable_storm['Tp_mean'] = tp_means
remarkable_storm['Patm_min'] = patm_mins
remarkable_storm['Wind_speed_max'] = ws_maxs
remarkable_storm['Wind_direction_max'] = wd_maxs
# u10 et v10 correspondants à la vitesse maximale
remarkable_storm['u10_max'] = u10_maxs
remarkable_storm['v10_max'] = v10_maxs

# Enlève les tempêtes avec une hauteur significative de la houle inférieure à 3 m
# remarkable_storm_fil = remarkable_storm.drop(index=[38,39,41,42,51,52]) 


