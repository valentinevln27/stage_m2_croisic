# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:50:25 2025

@author: benjamin
"""

from scipy.signal import argrelmax, argrelmin, find_peaks

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
    
    minimas_value_list = find_minimas_below_threshold(dataframe['swh'], swh_threshold)
    minimas_index_list = minimas_value_list.to_numpy().nonzero()
    minimas = dataframe.iloc[minimas_index_list[0]]
    
    for date in outliers.index:
        try:
            low_boundary = minimas[minimas.date < date].iloc[-1].date
        except:
            # Could not find date before the one concerned. Just taking the first value
            low_boundary = minimas.iloc[0].date
        try:
            high_boundary = minimas[minimas.date > date].iloc[0].date
        except:
            # Could not find date after the one concerned. Just taking the last value
            high_boundary = dataframe.iloc[-1].date
        
        current_window = [low_boundary, high_boundary]
        if current_window in window_list:
            pass
        else:
            window_list.append(current_window)
    print("{} generated windows".format(len(window_list)))
    return window_list


minima_windows = create_minimas_windows(df_wo_nan, model.extremes, 3)