# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 13:58:08 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""
This script extracts swell and weather parameters during three storms periods
(Lothar-Martin, Xynthia, Céline) using Candhis buoy data, Copernicus wave models 
(GOWR), and ERA5 reanalysis. It computes mean values for Hs, Tp and Dp as well 
as the minimal value of atmospherical pressure and the associated wind values 
(u10 and v10). The uncertainties for each values are also computed and given.
All outputs are given in a summary table per storm and buoy location.
"""

#%% Needed librairies imports
from datetime import datetime
import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

#%% Useful functions
def extract_mean_and_uncertainty(df, start, end, param, source):
    """
    Extracts the mean value of a swell parameter during a given period (storm) 
    and computes an uncertainty estimate.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series dataset (Candhis buoy data) containing the parameter of
        interest.
    df_un : pandas.DataFrame
        Not used in this function; included for compatibility.
    start : str or datetime
        Start date of the storm event.
    end : str or datetime
        End date of the storm event.
    param : str
        Name of the swell parameter to extract ('HM0', 'TP', 'THETAP').
    source : str
        Source of the data: 'candhis' or 'gowr'.

    Returns
    -------
    float
        Mean value of the swell parameter during the storm.
    float
        Estimated uncertainty (propagated or fixed depending on the source and 
                               swell parameter).
    """
    if df is None:
        return None, None
    df_storm = df[(df['Date'] >= start) & (df['Date'] <= end)]

    if param not in df_storm.columns:
        return None, None

    values = df_storm[param].dropna().astype(float)

    if values.empty:
        return None, None

    mean_val = values.mean()
    
    # Define uncertainty based on source and parameter
    if source == 'candhis': 
        # For candhis data, the uncertainties weren't given so we use a typical 
        # uncertainties given by buoy constructors
        if param == 'HM0':
            uncertainty = 0.005 * values
        elif param == 'TP':
            uncertainty = 0.5 # In secondes (s)
        elif param == 'THETAP':
            uncertainty = 5 # In degrees (°)
        else:
            uncertainty = 0

    elif source == 'gowr':
        if param == 'VHM0':
            uncertainty = 0.222 * values
        elif param == 'VTPK':
            uncertainty = 0.264 * values
        elif param == 'VPED':
            uncertainty = 37.8 # In degrees (°)
        else:
            uncertainty = 0

    else:
        uncertainty = 0

    if isinstance(uncertainty, float) or isinstance(uncertainty, int):
        propagated_uncertainty = uncertainty
    else:
        propagated_uncertainty = np.sqrt(np.sum(uncertainty ** 2)) / len(uncertainty)

    return mean_val, propagated_uncertainty


def get_param_mean(storm_idx, param, candhis_df, gowr_data, buoy_coords):
    """
    Returns the mean value and uncertainty of a swell parameter for a storm, 
    prioritizing Candhis data.

    Parameters
    ----------
    storm_idx : int
        Index of the storm in the predefined list.
    param : str
        Swell parameter to extract ('HM0' or 'VHM0').
    candhis_df : pandas.DataFrame or None
        Candhis buoy data if available.
    gowr_data : xarray.Dataset
        GOWR model data (Copernicus wave data).
    buoy_coords : list of float
        [latitude, longitude] of the buoy location.

    Returns
    -------
    float
        Mean value of the swell parameter.
    float
        Estimated uncertainty.
    """
    start = start_dates[storm_idx]
    end = end_dates[storm_idx]
    
    # Try Candhis data
    val, unc = extract_mean_and_uncertainty(candhis_df, start, end, param, 'candhis')
    if val is not None:
        return val, unc
    
    # Using GOWR data if Candhis unavailable
    else:
        lat, lon = buoy_coords
        subset = gowr_data.sel(time=slice(start, end))
        lat_idx = abs(subset['latitude'] - lat).argmin().item()
        lon_idx = abs(subset['longitude'] - lon).argmin().item()
        values = subset[param].isel(latitude=lat_idx, longitude=lon_idx).values
        values = values[~np.isnan(values)]
        if values.size == 0:
            return None, None
        mean_val = values.mean()
        if param == 'VHM0':
            unc = np.sqrt(np.sum((0.222 * values)**2)) / len(values)
        elif param == 'VTPK':
            unc = np.sqrt(np.sum((0.264 * values)**2)) / len(values)
        elif param == 'VPED':
            unc = 37.8
        else:
            unc = None
        return float(mean_val), float(unc)

def get_era5_uncertainty(spread_dataset, time_target, lat, lon, param):
    """
    Extracts the ERA5 ensemble spread (standard deviation) at a given time and 
    location.

    Parameters
    ----------
    spread_dataset : xarray.Dataset
        ERA5 spread dataset (standard deviation over ensemble members).
    time_target : datetime
        Timestamp of interest.
    lat : float
        Latitude of the point.
    lon : float
        Longitude of the point.
    param : str
        Weather parameter name ('msl', 'u10', 'v10').

    Returns
    -------
    float
        Spread value for the given parameter, location, and time.
    """
    subset = spread_dataset[param].sel(latitude=lat, longitude=lon, method='nearest')
    idx = np.argmin(np.abs(subset['time'].values - np.datetime64(time_target)))
    return float(subset.isel(time=idx).values)

def get_min_pressure_and_wind(df_era5, df_era5_spread, start, end, lat, lon):
    """
    Finds the minimum sea-level pressure and associated wind values during a storm.

    Parameters
    ----------
    df_era5 : xarray.Dataset
        ERA5 reanalysis dataset.
    df_era5_spread : xarray.Dataset
        ERA5 ensemble spread dataset.
    start : datetime
        Start time of the storm.
    end : datetime
        End time of the storm.
    lat : float
        Latitude of the point.
    lon : float
        Longitude of the point.

    Returns
    -------
    msl : float
        Minimum sea-level pressure.
    u10 : float
        Zonal wind component at the time of min pressure.
    v10 : float
        Meridional wind component at the time of min pressure.
    msl_err : float
        Uncertainty on msl.
    u10_err : float
        Uncertainty on u10.
    v10_err : float
        Uncertainty on v10.

    """
    
    subset = df_era5.sel(time=slice(start, end))
    subset_point = subset.sel(latitude=lat, longitude=lon, method='nearest')

    time_of_min_msl = subset_point['msl'].idxmin(dim='time')
    msl = float(subset_point['msl'].min())
    u10 = float(subset_point['u10'].sel(time=time_of_min_msl))
    v10 = float(subset_point['v10'].sel(time=time_of_min_msl))

    msl_err = get_era5_uncertainty(df_era5_spread, time_of_min_msl.values, lat, lon, 'msl')
    u10_err = get_era5_uncertainty(df_era5_spread, time_of_min_msl.values, lat, lon, 'u10')
    v10_err = get_era5_uncertainty(df_era5_spread, time_of_min_msl.values, lat, lon, 'v10')

    return msl, u10, v10, msl_err, u10_err, v10_err

#%% Extracting data and defining useful lists
# File paths for wave and weather datasets
data_era5 = '../datas_stage_m2/data_martin_xynthia_celine/data_era5/data.grib'
data_gowr = '../datas_stage_m2/data_martin_xynthia_celine/data_gowr/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1749652524428.nc'
candhis_PdFx = '../datas_stage_m2/data_martin_xynthia_celine/data_candhis/04403/Candhis_04403_2010_arch.csv'
candhis_PdFc = '../datas_stage_m2/data_martin_xynthia_celine/data_candhis/04403/Candhis_04403_2023_arch.csv'
candhis_IYNx = '../datas_stage_m2/data_martin_xynthia_celine/data_candhis/08504/Candhis_08504_2010_arch.csv'
candhis_IYNc = '../datas_stage_m2/data_martin_xynthia_celine/data_candhis/08504/Candhis_08504_2023_arch.csv'
buoy_coor = '../datas_stage_m2/data_martin_xynthia_celine/boueesCandhis.txt'

# Uncertainties
data_era5_err = '../datas_stage_m2/data_martin_xynthia_celine/data_era5/data_spread.grib'

# Load datasets
df_era5 = cfgrib.open_dataset(data_era5) 
df_era5_err = cfgrib.open_dataset(data_era5_err) 
df_gowr = xr.open_dataset(data_gowr)
df_PdFx = pd.read_csv(candhis_PdFx, sep=';')
df_PdFc = pd.read_csv(candhis_PdFc, sep=';')
df_IYNx = pd.read_csv(candhis_IYNx, sep=';')
df_IYNc = pd.read_csv(candhis_IYNc, sep=';')
buoy_coords = pd.read_csv(buoy_coor, sep=';')

# Convert date columns
for df in [df_PdFx, df_PdFc, df_IYNx, df_IYNc]:
    df['Date'] = pd.to_datetime(df['DateHeure'])

# Storm metadata
storm_names = ['Lothar-Martin', 'Xynthia', 'Celine']
start_dates = ['26/12/1999 00:00:00', '27/02/2010 00:00:00', '27/10/2023 00:00:00']
start_dates = [datetime.strptime(date, '%d/%m/%Y %H:%M:%S') for date in start_dates]
end_dates = ['28/12/1999 00:00:00', '01/03/2010 00:00:00', '29/10/2023 00:00:00']
end_dates = [datetime.strptime(date, '%d/%m/%Y %H:%M:%S') for date in end_dates]

# Parameter and buoy settings
buoy_params = ['HM0', 'TP', 'THETAP'] # In order, Hs, Tp and Dp
era5_params = ['msl', 'u10', 'v10'] # msl correspond to the atmospherical pressure
gowr_params = ['VHM0', 'VPED', 'VTPK'] # In order, Hs, Dp and Tp 

PdF_coords = [buoy_coords['lat'].iloc[11], buoy_coords['lon'].iloc[11]]
IYN_coords = [buoy_coords['lat'].iloc[8], buoy_coords['lon'].iloc[8]]

#%% Loop through storms and buoys to compute all values
results = []
for i, name in enumerate(storm_names):
    for label, coords, df_candhis in zip(['PdF', 'IYN'],
                                         [PdF_coords, IYN_coords],
                                         [[df_PdFx, df_PdFc], [df_IYNx, df_IYNc]]):
        # print(name) 
        # Select appropriate Candhis file for the storm
        if name == 'Xynthia':
            df_c = df_candhis[0]
        elif name == 'Celine':
            df_c = df_candhis[1]
        else:
            df_c = None
            
        # Retrieve swell parameters (Hs, Tp, Dp)
        Hs, Hs_err = get_param_mean(i, 'HM0' if df_c is not None else 'VHM0', df_c, 
                                    df_gowr, coords)
        Tp, Tp_err = get_param_mean(i, 'TP' if df_c is not None else 'VTPK', df_c, 
                                    df_gowr, coords)
        Dp, Dp_err = get_param_mean(i, 'THETAP' if df_c is not None and label == 'PdF' else 'VPED', 
                                    df_c, df_gowr, coords)
        
        # Retrieve ERA5 weather data
        msl, u10, v10, msl_err, u10_err, v10_err = get_min_pressure_and_wind(df_era5, 
                                                                             df_era5_err,
                                                                             start_dates[i], 
                                                                             end_dates[i], 
                                                                             *coords)
        
        # Store results
        results.append({
            'Tempête': name,
            'Bouée': label,
            'Hs': Hs, 'Hs_err': Hs_err,
            'Tp': Tp, 'Tp_err': Tp_err,
            'Dp': Dp, 'Dp_err': Dp_err,
            'MSL min': msl, 'MSL_err': msl_err,
            'u10': u10, 'u10_err': u10_err,
            'v10': v10, 'v10_err': v10_err
            })

# Final dataframe of results
df_final = pd.DataFrame(results)
