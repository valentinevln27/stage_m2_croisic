# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:17:05 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""
This script reads tidal gauge data and processes it. After treatment, the processed data 
is used to reconstruct the predicted tidal signal using UTide.

Next, for i signals, a signal is extracted from a specific point in a simulation with a 
given value of the Strickler coefficient (Ks). It is important to note that each simulation 
with a Ks value contains six distinct points.

Then, for similar time steps, the predicted tide values from the tide gauge and the 
simulated values from a given point are gathered into a dataframe to conduct a comparative 
study. This analysis is based on the calculation of three error metrics: MAE (Mean Absolute 
Error), RMSE (Root Mean Square Error), and Pbiais (percentage bias).

Finally, the MAE, RMSE, and Pbiais results are displayed for each simulation. Specifically, 
for a simulation with a given Ks value, the minimum and maximum values are presented, along 
with the Pbiais value closest to zero among the series of six points.
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
df_tg['zh'] = df_tg['zh']

# Calculating the predicted tide with Utide
cst_tg = solve(df_tg.index, df_tg.zh, lat=47.3, trend=None) # Latitude provided by the SHOM
predicted_tg = reconstruct(df_tg.index, cst_tg) # Calculate the predicted tide with utide
p_tg = pd.DataFrame(predicted_tg) # Convert predicted_tg into a DataFrame
p_tg.h = p_tg.h - cst_tg.mean

#%% Model data : For every points selected to compare with tide gauge datas
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

ks = 7 # The Strickler coefficient or combination number of the selected simulation
# Lists to store the results of the error metrics
ver = 5 # number version 
type_c = 'combi' # 'combi' or 'compa' or sensi
mesh = '1.3' # if type_c = 'sensi', can be '1.1' or '1.2' or '1.3' or '1.3_test'
MAE = [] 
RMSE = []
PBIAIS = [] 
for i in range(1, 7):
    # Extracting the data of a point from the simulation
    if type_c == 'sensi':
        file_path_pt = f'../bluekenue_stage/point/sensibility_point/sensi_ks{ks}_v{ver}_{mesh}/point_{i}_ks{ks}_v{ver}_{mesh}.ts1'
    elif type_c == 'compa':
        file_path_pt = f'../bluekenue_stage/point/{type_c}_point_v{ver}/ks{ks}_v{ver}/point_{i}_ks{ks}_{type_c}_v{ver}.ts1'
    else: 
        file_path_pt = f'../bluekenue_stage/point/{type_c}_point_v{ver}/ks{ks}_v{ver}/point_{i}_ks{ks}_v{ver}.ts1'
    df_pt = pd.read_csv(file_path_pt, sep=r'\s+', header=24, names=['sealevel'])
    df_pt['time'] = pd.date_range(start='2023-12-30',periods=len(df_pt), freq='5min')
    # Defining the time as the index
    df_pt.set_index('time', inplace=True)
    # Merging the tide gauge predictions with the simulation values
    df_merged = pd.merge(p_tg, df_pt, left_on='t_in', right_index=True, how='inner')
    # Computing MAE, RMSE, and Percentage Bias
    maev, maev_un = mae(df_merged['h'], df_merged['sealevel'])
    rmsev, rmsev_un = rmse(df_merged['h'], df_merged['sealevel'])
    p_biais = pbiais(df_merged['h'], df_merged['sealevel'])
    # Storing the results
    MAE.append(maev)
    RMSE.append(rmsev)
    PBIAIS.append(p_biais)
    # print(i)
    # print(maev_un)
    # print(rmsev_un)
    
# Printing the min, max and closest to zero values from a simulation (of 6 points)
print(f'Indicators for Ks{ks}:')
print(f'Min and max values of mae: {min(MAE):.2f}; {max(MAE):.2f}')
print(f'Min and max values of rmse: {min(RMSE):.2f}; {max(RMSE):.2f}')
print(f'Min and max values of pbiais: {min(PBIAIS):.2f}%; {max(PBIAIS):.2f}%')    
closest_to_zero = min(PBIAIS, key=abs)
print(f'Closest value to zero: {closest_to_zero:.2f}%')

def best_point_pb(mae, rmse, p_biais, KS):
    ct0 = min(p_biais, key=abs) # closest value to zero
    id_ct0 = p_biais.index(ct0)
    print(f'\nBest point (pbiais) for Ks{KS} (point n°{id_ct0+1}):')
    print(f'Mae value: {mae[id_ct0]:.2f}')
    print(f'Rmse value: {rmse[id_ct0]:.2f}')
    print(f'Pbiais value: {p_biais[id_ct0]:.2f}%')

best_point_pb(MAE, RMSE, PBIAIS, ks)

def best_point_rmse(mae, rmse, p_biais, KS):
    bp_rmse = min(rmse) 
    id_bp_rmse = rmse.index(bp_rmse)
    print(f'\nBest point (rmse) for Ks{KS} (point n°{id_bp_rmse+1}):')
    print(f'Mae value: {mae[id_bp_rmse]:.2f}')
    print(f'Rmse value: {rmse[id_bp_rmse]:.2f}')
    print(f'Pbiais value: {p_biais[id_bp_rmse]:.2f}%')

best_point_rmse(MAE, RMSE, PBIAIS, ks)

#%% Point 7
i=7
# Extracting the data of a point from the simulation
if type_c == 'sensi':
    file_path_pt = f'../bluekenue_stage/point/sensibility_point/sensi_ks{ks}_v{ver}_{mesh}/point_{i}_ks{ks}_v{ver}_{mesh}.ts1'
elif type_c == 'compa':
    file_path_pt = f'../bluekenue_stage/point/{type_c}_point_v{ver}/ks{ks}_v{ver}/point_{i}_ks{ks}_{type_c}_v{ver}.ts1'
else: 
    file_path_pt = f'../bluekenue_stage/point/{type_c}_point_v{ver}/ks{ks}_v{ver}/point_{i}_ks{ks}_v{ver}.ts1'
df_pt = pd.read_csv(file_path_pt, sep=r'\s+', header=24, names=['sealevel'])
df_pt['time'] = pd.date_range(start='2023-12-30',periods=len(df_pt), freq='5min')
# Defining the time as the index
df_pt.set_index('time', inplace=True)
# Merging the tide gauge predictions with the simulation values
df_merged = pd.merge(p_tg, df_pt, left_on='t_in', right_index=True, how='inner')
# Computing MAE, RMSE, and Percentage Bias
maev, maev_un = mae(df_merged['h'], df_merged['sealevel'])
rmsev, rmsev_un = rmse(df_merged['h'], df_merged['sealevel'])
p_biais = pbiais(df_merged['h'], df_merged['sealevel'])

# Printing the min, max and closest to zero values from a simulation (of 6 points)
print(f'\nIndicators for Ks{ks} point 7:')
print(f'Mae: {maev:.2f}')
print(f'Rmse: {rmsev:.2f}')
print(f'Pbiais: {p_biais:.2f}%')    

#%% Graphical representation with one point only
if type_c == 'sensi':
    file_path_pt = f'../bluekenue_stage/point/sensibility_point/sensi_ks{ks}_v{ver}_{mesh}/point_{i}_ks{ks}_v{ver}_{mesh}.ts1'
elif type_c == 'compa':
    file_path_pt = f'../bluekenue_stage/point/{type_c}_point_v{ver}/ks{ks}_v{ver}/point_{i}_ks{ks}_{type_c}_v{ver}.ts1'
else: 
    file_path_pt = f'../bluekenue_stage/point/{type_c}_point_v{ver}/ks{ks}_v{ver}/point_{i}_ks{ks}_v{ver}.ts1'
df_pt = pd.read_csv(file_path_pt, sep=r'\s+', header=24, names=['sealevel'])
df_pt['time'] = pd.date_range(start='2023-12-30',periods=len(df_pt), freq='5min')
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
plt.plot(df_merged.index, df_merged.sealevel, color='red', linestyle='-', linewidth=0.5,
         label='Model')
plt.plot(df_merged.index, df_merged.h , color='royalblue', 
         linestyle='-', linewidth=0.5,
         label='Tide gauge')

# Adding legend
plt.title(f'Predicted tide at Le Croisic (combination of Ks {ks})', fontsize=20, fontweight='bold')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Predicted tide (m)', fontsize=15)

# Graphical option and plotting
plt.legend(fontsize=15)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7) 
plt.tight_layout()
plt.show()
