# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:29:27 2025

@author: vanleene valentine
"""

#%% Explaination of the script
"""
This script processes meteorological wind data from different GRIB files and 
compile them into a single NetCDF file.
"""

#%% Needed librairies import
from datetime import timedelta
from glob import glob
import cfgrib
import os
import pandas as pd
import xarray as xr

#%% Meteorological data files extraction
root_path = r'../datas_stage_m2/data_weather/raw_data' 
# Take all data.grib file in all subfolders
grib_files = sorted(glob(os.path.join(root_path, "*_to_*", "data.grib")))

#%% Combining all files
datasets = [] # List to store individual datasets
previous_end_time = None # Variable to track the end time of the previous file

for grib_file in grib_files:
    # print('File found') # Simple check to confirm progress
    ds = cfgrib.open_dataset(grib_file) # Open the GRIB file as an xarray dataset
    # Convert the time values to datetime format
    time_values = pd.to_datetime(ds.time.values) 

    if previous_end_time is not None:
        current_start_time = time_values[0]
        # Check if there is a gap or overlap between the current and previous files
        if current_start_time != previous_end_time + timedelta(hours=1):
            raise ValueError(f"Gap or overlap detected between files. Expected "
                             f"{previous_end_time + timedelta(hours=1)}, but got "
                             f"{current_start_time}")
            
    # Update the previous end time for the next iteration
    previous_end_time = time_values[-1]
    # Add the dataset to the list
    datasets.append(ds)

# Merge all datasets into a single dataset along the time 
combined = xr.concat(datasets, dim="time")

#%% Saving the compliled file
# Define the output file path
exit_path = '../datas_stage_m2/data_weather/compiled_data/combined_data.nc'
# If a NetCDF file already exists at the output path, delete it
if os.path.exists(exit_path):
    os.remove(exit_path)
# Save the combined dataset as a NetCDF file
combined.to_netcdf(exit_path)


