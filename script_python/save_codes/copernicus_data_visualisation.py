# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:31:03 2025

@author: vanleene valentine
"""
#%%
"""
This script allows for the visualization of extracted Copernicus ERA5 data based 
on a selected parameter during a storm. In a second figure, the wind speed is 
calculated and displayed graphically.
"""

#%% Needed librairies import
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#%% Extracting data from a storm file
# Define the storm event and the type of dataset to use ('oper' for atmospheric or 'wave' for wave data)
storm = 'martin' # Options: 'martin', 'xynthia' or 'celine'
type_data = 'oper'  # Options: 'oper' (meteorological) or 'wave' (swell data)
# Open the corresponding NetCDF file for the selected storm and data type
ds = xr.open_dataset(
    f'../datas_stage_m2/{storm}/data_stream-{type_data}_stepType-instant.nc')

# Select the parameter to analyze 
# Options: 'u10', 'v10' or 'msl' if 'oper', 'swh', 'mwd' or 'mwp' for 'wave'
param = 'v10'  # Example : here is the 10-meter wind speed 
# Extract data for the parameter at a specific geographic location (nearest point)
param_point = ds[f'{param}'].sel(latitude=47.27, longitude=-3.28, method='nearest')
# For all storms, compute the mean over the full time series
mean_value = param_point.mean(dim='valid_time')
print(mean_value.values)

#%% Graphical representation for raw data of the selected parameter
plt.figure(figsize=(10, 4), dpi=300)
param_point.plot() # Plot the time series of the parameter
# Legend
plt.title(f'{param} during {storm} at (47.27°N, -3.38°E)')
plt.ylabel(f'{param} m/s') # The unit need to be manually changed
plt.xlabel('Time')
plt.grid(True)
plt.show()

#%% Graphical representation for wind speed
# Compute wind speed from the u and v wind components at the same location
u_point = ds['u10'].sel(latitude=47.27, longitude=-3.28, method='nearest')
v_point = ds['v10'].sel(latitude=47.27, longitude=-3.28, method='nearest')
# Calculate wind speed magnitude: sqrt(u² + v²)
wind_speed = np.sqrt((u_point)**2 + (v_point)**2)

# Plot the wind speed time series
plt.figure(figsize=(10, 4), dpi=300)
wind_speed.plot()
plt.title(f'Wind speed during {storm} at (47.27°N, -3.28°E)')
plt.ylabel('Wind speed (m/s)') 
plt.xlabel('Time')
plt.grid(True)
plt.show()