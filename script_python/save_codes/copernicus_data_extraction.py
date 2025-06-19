# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:20:25 2025

@author: vanleene valentine
"""

#%% Explaining the script
"""
This script comes from the ERA5 data API on the Copernicus website; I didn’t write 
it myself. On the site used to download ERA5 data, the API provides the code based 
on which data you want to download and in what format.

ERA5 data site : 
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview

Note: After downloading, all files were renamed.
"""

#%% Needed librairies import
import cdsapi 

#%% Data extraction
dataset = "reanalysis-era5-single-levels" 
request = {
    "product_type": ["reanalysis"],
    # Selecting the needed data
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure"
    ],
    # Selecting the time period 
    "year": [
        "1988", "1989" #, "1986",
        # "1987"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    # Selecting the format of the data
    "data_format": "grib",
    "download_format": "zip",
    # Selecting a specific area containing the points of the Deep/Open Boundary of 
    # the mesh (mesh used in my internship)
    "area": [46.73, -4.59, 45.73, -3.38]
    }

client = cdsapi.Client()
client.retrieve(dataset, request).download()


