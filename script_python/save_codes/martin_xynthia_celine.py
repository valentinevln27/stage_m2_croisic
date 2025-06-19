# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:29:48 2025

@author: vanleene valentine
"""

#%% Librairies import
import pandas as pd
import cfgrib
import xarray as xr
from datetime import datetime

#%%
data_era5 = '../datas_stage_m2/data_martin_xynthia_celine/data_era5/data.grib'
data_gowr = '../datas_stage_m2/data_martin_xynthia_celine/data_gowr/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1749652524428.nc'
candhis_PdFx = '../datas_stage_m2/data_martin_xynthia_celine/data_candhis/04403/Candhis_04403_2010_arch.csv'
candhis_PdFc = '../datas_stage_m2/data_martin_xynthia_celine/data_candhis/04403/Candhis_04403_2023_arch.csv'
candhis_IYNx = '../datas_stage_m2/data_martin_xynthia_celine/data_candhis/08504/Candhis_08504_2010_arch.csv'
candhis_IYNc = '../datas_stage_m2/data_martin_xynthia_celine/data_candhis/08504/Candhis_08504_2023_arch.csv'
buoy_coor = '../datas_stage_m2/data_martin_xynthia_celine/boueesCandhis.txt'

df_era5 = cfgrib.open_dataset(data_era5) 
df_gowr = xr.open_dataset(data_gowr)
df_PdFx = pd.read_csv(candhis_PdFx, sep=';')
df_PdFc = pd.read_csv(candhis_PdFc, sep=';')
df_IYNx = pd.read_csv(candhis_IYNx, sep=';')
df_IYNc = pd.read_csv(candhis_IYNc, sep=';')
buoy_coords = pd.read_csv(buoy_coor, sep=';')
for df in [df_PdFx, df_PdFc, df_IYNx, df_IYNc]:
    df['Date'] = pd.to_datetime(df['DateHeure'])

storm_names = ['Lothar-Martin', 'Xynthia', 'Celine']
start_dates = ['26/12/1999 00:00:00', '27/02/2010 00:00:00', '27/10/2023 00:00:00']
start_dates = [datetime.strptime(date, '%d/%m/%Y %H:%M:%S') for date in start_dates]
end_dates = ['28/12/1999 00:00:00', '01/03/2010 00:00:00', '29/10/2023 00:00:00']
end_dates = [datetime.strptime(date, '%d/%m/%Y %H:%M:%S') for date in end_dates]

buoy_params = ['HM0', 'TP', 'THETAP']
era5_params = ['msl', 'u10', 'v10']
gowr_params = ['VHM0', 'VPED', 'VTPK']

PdF_coords = [buoy_coords['lat'].iloc[11], buoy_coords['lon'].iloc[11]]
IYN_coords = [buoy_coords['lat'].iloc[8], buoy_coords['lon'].iloc[8]]

#%% Useful functions
def extract_mean_param(df, start, end, param):
    if df is None:
        return None
    df_storm = df[(df['Date'] >= start) & (df['Date'] <= end)]
    if param in df_storm.columns:
        return df_storm[param].mean()
    else:
        return None

def get_param_mean(storm_idx, param, candhis_df, gowr_data, buoy_coords):
    start = start_dates[storm_idx]
    end = end_dates[storm_idx]

    val = extract_mean_param(candhis_df, start, end, param)
    
    if val is not None:
        return val
    else:
        lat, lon = buoy_coords
        subset = gowr_data.sel(time=slice(start, end))
        lat_idx = abs(subset['latitude'] - lat).argmin().item()
        lon_idx = abs(subset['longitude'] - lon).argmin().item()
        return float(subset[param].isel(latitude=lat_idx, longitude=lon_idx).mean())

def get_min_pressure_and_wind(df_era5, start, end, lat, lon):
    subset = df_era5.sel(time=slice(start, end))
    subset_point = subset.sel(latitude=lat, longitude=lon, method='nearest')

    time_of_min_msl = subset_point['msl'].idxmin(dim='time')
    msl = float(subset_point['msl'].min())
    u10 = float(subset_point['u10'].sel(time=time_of_min_msl))
    v10 = float(subset_point['v10'].sel(time=time_of_min_msl))

    return msl, u10, v10

#%%
results = []
for i, name in enumerate(storm_names):
    for label, coords, df_candhis in zip(['PdF', 'IYN'],
                                         [PdF_coords, IYN_coords],
                                         [[df_PdFx, df_PdFc], [df_IYNx, df_IYNc]]):
        print(name)
        if name == 'Xynthia':
            df_c = df_candhis[0]
        elif name == 'Celine':
            df_c = df_candhis[1]
        else:
            df_c = None  # Martin → pas de Candhis
            
        if df_c is not None:
            print('got it')
        Hs = get_param_mean(i, 'HM0' if df_c is not None else 'VHM0', df_c, df_gowr, coords)
        Tp = get_param_mean(i, 'TP' if df_c is not None else 'VTPK', df_c, df_gowr, coords)
        Dp = get_param_mean(i, 'THETAP' if df_c is not None and label == 'PdF' else 'VPED', df_c, df_gowr, coords)
        # Dp = get_param_mean(i, 'VPED', df_c, df_gowr, coords)
        msl, u10, v10 = get_min_pressure_and_wind(df_era5, start_dates[i], end_dates[i], *coords)

        results.append({
            'Tempête': name,
            'Bouée': label,
            'Hs': Hs,
            'Tp': Tp,
            'Dp': Dp,
            'MSL min': msl,
            'u10': u10,
            'v10': v10
            })

df_final = pd.DataFrame(results)    



