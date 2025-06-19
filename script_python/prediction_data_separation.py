# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:35:13 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""
In this script, we use an Excel file containing sea level predictions during the 
Lothar-Martin, Xynthia, and Celine storms at three different tide gauges: 
    - Le Croisic (CR), 
    - Saint-Nazaire (SN), 
    - and Les Sables-d’Olonne (SO).
    
The data are then separated by storm and by tide gauge. We also subtract the 
mean sea level to allow for a better comparison with the model data used in the 
tide_comparison script.
"""

#%% Needed librairies imports
from glob import glob
import matplotlib.pyplot as plt
import os
import pandas as pd

#%% Data extraction and useful lists
file_path = '../excel/data_prediction/data_prediction_all.xlsx'
df_tg = pd.read_excel(file_path)
df_tg.time = pd.to_datetime(df_tg.time)
df_tg['time'] = pd.to_datetime(df_tg['time']).dt.round('s')
storm = ['martin', 'xynthia', 'celine']
tg = ['le_croisic', 'saint_nazaire', 'sables_olonne']
code_tg = ['99', '37', '62']
nm = [3.30, 3.61, 3.31]
date_start = ['1999-12-23 00:00:00', '2010-02-24 00:00:00', '2023-10-24 00:00:00']
date_end = ['2000-01-01 23:00:00', '2010-03-03 23:00:00', '2023-11-01 23:00:00']

#%% Separating the data by storm and by tide gauge
for i in range(3):
    start = pd.to_datetime(date_start[i])
    end = pd.to_datetime(date_end[i])
    data_filtered = df_tg[(df_tg['time'] >= start) & (df_tg['time'] <= end)]
    data_filtered.set_index('time', inplace=True)
    for j in range(3):
        data_filtered.loc[:, tg[j]] = data_filtered[tg[j]] - nm[j]
        
        # Saving the data
        # Define the output file path
        exit_path = f'../excel/data_prediction/data_{code_tg[j]}_{storm[i]}.csv'
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

    
