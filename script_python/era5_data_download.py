# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 20:42:15 2025

@author: API ERA5
"""

#%% Explaining what the script does
"""
This script, provided by the ERA5 API, allows us to download the desired ERA5 data.
"""

#%% Needed librairy import
import cdsapi

#%% Extracting the data with the API
dataset = "reanalysis-era5-single-levels"
request = {
    # "product_type": ["reanalysis"],
    "product_type": ["ensemble_spread"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure"
    ],
    "year": [
        "1999","2000","2010", "2023"
    ],
    "month": [
        "01", "02", "03",
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
    "data_format": "grib",
    "download_format": "zip",
    "area": [48, -5.55, 44, 5]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
