# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:35:13 2025

@author: vanle
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

#%%
file_path = 'C:/cours/master/M2/Stage/excel/data_prediction/data_prediction_all.xlsx'
df_tg = pd.read_excel(file_path)
df_tg.time = pd.to_datetime(df_tg.time)
df_tg['time'] = pd.to_datetime(df_tg['time']).dt.round('s')
storm = ['martin', 'xynthia', 'celine']
tg = ['le_croisic', 'saint_nazaire', 'sables_olonne']
nm = [3.30, 3.61, 3.31]
date_start = ['1999-12-23 00:00:00', '2010-02-24 00:00:00', '2023-10-24 00:00:00']
date_end = ['2000-01-01 23:00:00', '2010-03-03 23:00:00', '2023-11-01 23:00:00']

for i in range(3):
    start = pd.to_datetime(date_start[i])
    end = pd.to_datetime(date_end[i])
    data_filtered = df_tg[(df_tg['time'] >= start) & (df_tg['time'] <= end)]
    data_filtered.set_index('time', inplace=True)
    for j in range(3):
        data_filtered.loc[:, tg[j]] = data_filtered[tg[j]] - nm[j]
        
        # Saving the data
        # Define the output file path
        exit_path = f'C:/cours/master/M2/Stage/excel/data_prediction/data_{tg[j]}_{storm[i]}.csv'
        # If a NetCDF file already exists at the output path, delete it
        if os.path.exists(exit_path):
            os.remove(exit_path)
        # Save the combined dataset as a csv file
        data_filtered[f'{tg[j]}'].to_csv(exit_path)
    
    # Graphical representation
    plt.plot(data_filtered.index, data_filtered.le_croisic, 'r', label='CR')
    plt.plot(data_filtered.index, data_filtered.saint_nazaire, 'b', label='SN')
    plt.plot(data_filtered.index, data_filtered.sables_olonne, 'g', label='SO')
    plt.xticks(rotation=25)  
    plt.legend()
    plt.grid()
    plt.show()

    
