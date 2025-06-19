# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:27:26 2025

@author: vanleene valentine
"""

#%% Needed librairies imports
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# import utide
from utide import solve, reconstruct

#%% Tide gauge data
# Extracting data from the file
file_path_tg = '../datas_stage_m2/croisicZH.txt'
df_tg = pd.read_csv(file_path_tg, delimiter=';', header=1, names=['time', 'zh'])

# Converting 'time' into datetime values
df_tg['time'] = pd.to_datetime(df_tg['time'], dayfirst=True)
# Defining the last date
time_limit = pd.Timestamp('2024-04-22 11:25:00')
# time_limit = pd.Timestamp('2024-01-04 23:45:00')
# Filter the rows with a timestamp beyond the previously defined last date.
df_tg = df_tg[df_tg['time'] <= time_limit]
# Defining the time as the index
df_tg.set_index('time', inplace=True)

# Replacing values > 7 m by nan: theses values are outliers
df_tg.loc[df_tg['zh'] > 7, 'zh'] = np.nan

# Calculating the predicted tide with Utide
cst_tg = solve(df_tg.index, df_tg.zh, lat=47.3, trend=None) # Latitude provided by the SHOM
predicted_tg = reconstruct(df_tg.index, cst_tg) # Calculate the predicted tide with utide
p_tg = pd.DataFrame(predicted_tg) # Convert predicted_tg into a DataFrame
p_tg.h = p_tg.h - cst_tg.mean

#%% Useful functions
# MAE, RMSE and PBIAIS
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

def best_point_pb(mae, rmse, p_biais, Storm):
    ct0 = min(p_biais, key=abs) # closest value to zero
    id_ct0 = p_biais.index(ct0)
    print(f'\nBest point (pbiais) for {Storm} (point n°{id_ct0+1}):')
    print(f'Mae value: {mae[id_ct0]:.2f}')
    print(f'Rmse value: {rmse[id_ct0]:.2f}')
    print(f'Pbiais value: {p_biais[id_ct0]:.2f}%')

def best_point_rmse(mae, rmse, p_biais, Storm):
    bp_rmse = min(rmse) 
    id_bp_rmse = rmse.index(bp_rmse)
    print(f'\nBest point (rmse) for {Storm} (point n°{id_bp_rmse+1}):')
    print(f'Mae value: {mae[id_bp_rmse]:.2f}')
    print(f'Rmse value: {rmse[id_bp_rmse]:.2f}')
    print(f'Pbiais value: {p_biais[id_bp_rmse]:.2f}%')
    
#%% Model data : For every points selected to compare with tide gauge datas
# Prediction of the tide for the storm period: 1999-12-26 to 1999-12-30
# Creating a new time range with the same frequency as original data
t_storm = pd.date_range(start='1999-12-26', end='1999-12-30 00:00:00', freq='5min')

# Predicting tide on the new time interval using UTide coefficients
predicted_storm = reconstruct(t_storm, cst_tg)
p_storm = pd.DataFrame(predicted_storm)
p_storm.h = p_storm.h - cst_tg.mean
p_storm.set_index('t_in', inplace=True)

storm = 'martin' 
MAE = [] 
RMSE = []
PBIAIS = [] 

for i in range(1, 7):
    file_path_pt = f'../bluekenue_stage/point/storm_meteo_point/{storm}/point_{i}_{storm}.ts1'
    df_pt = pd.read_csv(file_path_pt, sep=r'\s+', header=24, names=['sealevel'])
    df_pt['time'] = pd.date_range(start='1999-12-26', periods=len(df_pt), freq='5min')
    df_pt.set_index('time', inplace=True)
    
    cst_pt = solve(df_pt.index, df_pt.sealevel, lat=47.3, trend=True) # Latitude provided by the SHOM
    predicted_pt = reconstruct(df_pt.index, cst_pt) # Calculate the predicted tide with utide
    p_pt = pd.DataFrame(predicted_pt) # Convert predicted_tg into a DataFrame
    p_pt['sealevel'] = p_pt.h - cst_pt.mean
    p_pt.set_index('t_in', inplace=True)
    p_pt.drop('h', axis=1, inplace=True)
    
    # Merge simulation with storm tide prediction
    df_merged = pd.merge(p_storm, p_pt, left_index=True, right_index=True, how='inner')

    # Error metrics
    maev, maev_un = mae(df_merged['h'], df_merged['sealevel'])
    rmsev, rmsev_un = rmse(df_merged['h'], df_merged['sealevel'])
    p_biais_val = pbiais(df_merged['h'], df_merged['sealevel'])

    MAE.append(maev)
    RMSE.append(rmsev)
    PBIAIS.append(p_biais_val)

# Printing the min, max and closest to zero values from a simulation (of 6 points)
print(f'Indicators for {storm}:')
print(f'Min and max values of mae: {min(MAE):.2f}; {max(MAE):.2f}')
print(f'Min and max values of rmse: {min(RMSE):.2f}; {max(RMSE):.2f}')
print(f'Min and max values of pbiais: {min(PBIAIS):.2f}%; {max(PBIAIS):.2f}%')    
closest_to_zero = min(PBIAIS, key=abs)
print(f'Closest value to zero: {closest_to_zero:.2f}%')

best_point_pb(MAE, RMSE, PBIAIS, storm)
best_point_rmse(MAE, RMSE, PBIAIS, storm)

#%% Point 7
i=7
# Extracting the data of a point from the simulation
file_path_pt = f'../bluekenue_stage/point/storm_meteo_point/{storm}/point_1_{storm}.ts1'
df_pt = pd.read_csv(file_path_pt, sep=r'\s+', header=24, names=['sealevel'])
df_pt['time'] = pd.date_range(start='1999-12-26', periods=len(df_pt), freq='5min')
df_pt.set_index('time', inplace=True)
cst_pt = solve(df_pt.index, df_pt.sealevel, lat=47.3, trend=True) # Latitude provided by the SHOM
predicted_pt = reconstruct(df_pt.index, cst_pt) # Calculate the predicted tide with utide
p_pt = pd.DataFrame(predicted_pt) # Convert predicted_tg into a DataFrame
p_pt['sealevel'] = p_pt.h - cst_pt.mean
p_pt.set_index('t_in', inplace=True)
p_pt.drop('h', axis=1, inplace=True)

# Merge simulation with storm tide prediction
df_merged = pd.merge(p_storm, p_pt, left_index=True, right_index=True, how='inner')
# Computing MAE, RMSE, and Percentage Bias
maev, maev_un = mae(df_merged['h'], df_merged['sealevel'])
rmsev, rmsev_un = rmse(df_merged['h'], df_merged['sealevel'])
p_biais = pbiais(df_merged['h'], df_merged['sealevel'])

# Printing the min, max and closest to zero values from a simulation (of 6 points)
print(f'\nIndicators for {storm} point 7:')
print(f'Mae: {maev:.2f}')
print(f'Rmse: {rmsev:.2f}')
print(f'Pbiais: {p_biais:.2f}%')    

#%% Graphical representation with one point only
file_path_pt = f'../bluekenue_stage/point/storm_meteo_point/{storm}/point_1_{storm}.ts1'
df_pt = pd.read_csv(file_path_pt, sep=r'\s+', header=24, names=['sealevel'])
df_pt['time'] = pd.date_range(start='1999-12-26',periods=len(df_pt), freq='5min')
# Defining the time as the index
df_pt.set_index('time', inplace=True)

# Graphical representation
plt.figure(figsize=(10, 6), dpi=300)
# Changing ticks size
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
# Rotation of x-ticks
plt.xticks(rotation=25)  

# Plotting the predicted tide
plt.plot(df_pt.index, df_pt.sealevel, color='red', linestyle='-', linewidth=0.5,
         label='Model')
plt.plot(p_storm.index, p_storm.h , color='royalblue', 
         linestyle='-', linewidth=0.5,
         label='Tide gauge')

# Adding legend
plt.title(f'Predicted tide at Le Croisic during {storm}', fontsize=20, 
          fontweight='bold')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Predicted tide (m)', fontsize=15)

# Graphical option and plotting
plt.legend(fontsize=15)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7) 
plt.tight_layout()
plt.show()
