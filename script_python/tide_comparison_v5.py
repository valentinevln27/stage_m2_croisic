# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:16:14 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""
This script reads predicted tidal gauge data and, for i signals, a signal is 
extracted from a specific point in a simulation with a given value of the 
Strickler coefficient (Ks). It is important to note that each simulation with a
Ks value contains six distinct points.

Then, for similar time steps, the predicted tide values from the tide gauge and 
the simulated values from a given point are gathered into a dataframe to conduct 
a comparative study. This analysis is based on the calculation of three error 
metrics: MAE (Mean Absolute Error), RMSE (Root Mean Square Error), and Pbiais 
(percentage bias).

Finally, the MAE, RMSE, and Pbiais results are displayed for each simulation. 
Specifically, for a simulation with a given Ks value, and for each storm studied.
"""

#%% Needed librairies imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# import utide
# from utide import solve, reconstruct

#%% Useful functions
def mae(obs, mod): # Calculate mae
    """
    Calculate the Mean Absolute Error (MAE) and its uncertainty.

    The MAE is a measure of the average magnitude of errors between observed 
    and modeled values, without considering their direction.

    Parameters
    ----------
    obs : array-like
        Observed values (reference measurements).
    mod : array-like
        Modeled values (simulated data).

    Returns
    -------
    maev : float
        The mean absolute error.
    maev_uncertainty : float
        The uncertainty of MAE, estimated as the standard deviation of the 
        residuals divided by the square root of the sample size.
    """
    n = len(obs)
    diff = obs - mod
    maev = sum(abs(diff)) / n
    maev_uncertainty = np.std(diff) / np.sqrt(len(diff))
    return maev, maev_uncertainty

def rmse(obs, mod): # Calculate rmse
    """
    Calculate the Root Mean Square Error (RMSE) and its uncertainty.

    RMSE measures the average magnitude of the error, giving more weight 
    to larger errors due to squaring differences before averaging.

    Parameters
    ----------
    obs : array-like
        Observed values (reference measurements).
    mod : array-like
        Modeled values (simulated data).

    Returns
    -------
    rmsev : float
        The root mean square error.
    rmsev_uncertainty : float
        The uncertainty of RMSE, estimated as the standard deviation of the 
        squared residuals divided by the square root of the sample size.
    """
    n = len(obs) 
    diff = obs - mod
    rmsev = np.sqrt(sum(diff**2) / n)
    rmsev_uncertainty = np.std(diff**2) / np.sqrt(len(diff**2))
    return rmsev, rmsev_uncertainty

def pbiais(obs, mod): # Calculate pbiais
    """
    Calculate the Percentage Bias (Pbiais).

    Percentage Bias indicates the average bias of the model in percentage terms.
    A positive bias means the model underestimates the observed values, while 
    a negative bias means the model overestimates them. In case the model overestimates
    the observed values, in the simulation we need to decrease h (so increasing u), so we 
    need to decrease the bottom friction by increasing Ks.

    Parameters
    ----------
    obs : array-like
        Observed values (reference measurements).
    mod : array-like
        Modeled values (simulated data).

    Returns
    -------
    p_biais : float
        The percentage bias, calculated as the mean relative difference 
        between modeled and observed values, expressed in percentage.
    """
    n = len(obs)
    diff = mod - obs 
    r = diff/obs
    p_biais = 100*np.mean(r)
    return p_biais

#%% Datas comparision
#%% Tide gauge data
# Extracting data from the file
storms = ['martin', 'xynthia', 'celine']
tg_code = ['99', '37', '62'] # In order, Le Croisic, Saint-Nazaire and les Sables d'Olonne
file_tg_metadata = '../datas_stage_m2/gauges_hf_vtd_vh_ZH.txt'
ks = 10 # The Strickler coefficient or combination number of the selected simulation
# Lists to store the results of the error metrics
MAE = [] 
RMSE = []
PBIAIS = [] 
for i in range(len(tg_code)):
    for j in range(len(storms)):    
        # Tide gauge datas
        file_path_tg = f'../excel/data_prediction/data_{tg_code[i]}_{storms[j]}.csv'
        df_tg = pd.read_csv(file_path_tg)
        df_tg['time'] = pd.to_datetime(df_tg['time']) 
        
        # Model datas
        file_path_model = f'C:/Users/vanle/Downloads/{tg_code[i]}_{storms[j]}.csv'
        df_pt = pd.read_csv(file_path_model, header=2, 
                            names=['date', 'time', 'u', 'v', 'sealevel', 'del'])
        df_pt['date'] = pd.to_datetime(df_pt['date']) 
        # Defining the time as the index
        df_pt.set_index('date', inplace=True)
        
        # Merging the tide gauge predictions with the simulation values
        df_merged = pd.merge(df_tg, df_pt, left_on='time', right_index=True, how='inner')
        # Computing MAE, RMSE, and Percentage Bias
        maev, maev_un = mae(df_merged[f'{storms[j]}'], df_merged['sealevel'])
        rmsev, rmsev_un = rmse(df_merged[f'{storms[j]}'], df_merged['sealevel'])
        p_biais = pbiais(df_merged[f'{storms[j]}'], df_merged['sealevel'])
        # Storing the results
        MAE.append(maev)
        RMSE.append(rmsev)
        PBIAIS.append(p_biais)


        # Graphical representation
        plt.figure(figsize=(10, 6), dpi=300)
        # Changing ticks size
        plt.rcParams['xtick.labelsize'] = 12 
        plt.rcParams['ytick.labelsize'] = 12 
        # Rotation of x-ticks
        plt.xticks(rotation=25)  
        
        # Plotting the predicted tide
        plt.plot(df_merged.index, df_merged.sealevel, color='red', linestyle='-', linewidth=0.5,
                 label='Model')
        plt.plot(df_merged.index, df_merged[f'{storms[j]}'] , color='royalblue', 
                 linestyle='-', linewidth=0.5,
                 label='Tide gauge')
        
        # Adding legend
        plt.title(f'Predicted tide at Le Croisic (combination of Ks {ks})', fontsize=20, 
                  fontweight='bold')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Predicted tide (m)', fontsize=15)
        
        # Graphical option and plotting
        plt.legend(fontsize=15)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7) 
        plt.tight_layout()
        plt.show()