# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 10:30:18 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""
This Python script analyzes and visualizes storm surge data at Le Croisic 
during the Xynthia storm event, comparing scenarios with and without a storm 
surge barrier.
"""

#%% Needed librairies imports
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

#%% Data extraction and surge
# Load model data without storm surge/flood barrier
df = pd.read_csv('C:/Users/vanle/Downloads/node_12345.csv')
df_meteo = pd.read_csv('C:/Users/vanle/Downloads/node_12345_meteo.csv')
df['complete_date'] = pd.to_datetime(df['DATE'])
# Load model data with storm surge/flood barrier
df_gate = pd.read_csv('C:/Users/vanle/Downloads/node_12335.csv')
df_gate_meteo = pd.read_csv('C:/Users/vanle/Downloads/node_12335_meteo.csv')
df_gate['complete_date'] = pd.to_datetime(df_gate['DATE'])

# Load reference prediction data during Xynthia at Le Croisic
df_cr = pd.read_csv('../datas_stage_m2/records/outputs/xynthia/99_xynthia.txt', 
                    sep=';')
df_cr['Time'] = pd.to_datetime(df_cr['Time'])

# Calculate surge: difference between forced (meteo) and unforced (model) free surface
df['surge'] = df_meteo[' FREE SURFACE'] - df[' FREE SURFACE']
df_gate['surge'] = df_gate_meteo[' FREE SURFACE'] - df_gate[' FREE SURFACE']

# Maximal surge
print(f"Maximal surge without barrier : {max(df['surge']):.2f} m")
print(f"Maximal surge with barrier : {max(df_gate['surge']):.2f} m")

#%% Surge graphical representation (model)
# Surge without barrier
plt.figure(dpi=300)
plt.plot(df['complete_date'], df['surge'])
# plt.title('Surge at Le Croisic without storm surge barrier')
plt.ylabel('Surge (m)')
plt.xticks(rotation=25)  
plt.grid()
plt.show()

# Surge with storm surge barrier
plt.figure(dpi=300)
plt.plot(df_gate['complete_date'], df_gate['surge'])
# plt.title('Surge at Le Croisic with storm surge barrier')
plt.ylabel('Surge (m)')
plt.xticks(rotation=25)  
plt.grid()
plt.show()

#%% Plot all water levels: predictions vs model with/without barrier
plt.figure(dpi=300)
plt.plot(df_cr['Time'], df_cr['z'] - 3.31, label='SHOM predictions')
plt.plot(df['complete_date'], df[' FREE SURFACE'], 
         label='Without storm surge barrier')
plt.plot(df_gate['complete_date'], df_gate[' FREE SURFACE'], 
         label='With storm surge barrier')
plt.ylabel('Water level (m)')
plt.xticks(rotation=25)  
plt.grid()
plt.legend()
plt.show()

#%% Plot all water levels within the model timespan
# Define common timespan
start_time = df['complete_date'].min()
end_time = df['complete_date'].max()

# Filter prediction data accordingly
df_cr_filtered = df_cr[(df_cr['Time'] >= start_time) & (df_cr['Time'] <= end_time)]

# Graphical representation
plt.figure(dpi=300)
plt.plot(df_cr_filtered['Time'], df_cr_filtered['z'] - 3.31, 
         label='SHOM predictions', linewidth=0.5)
plt.plot(df['complete_date'], df[' FREE SURFACE'], 
         label='Without storm surge barrier', linewidth=0.5)
plt.plot(df_gate['complete_date'], df_gate[' FREE SURFACE'], 
         label='With storm surge barrier', linewidth=0.5)
# plt.title('Sea level at Le Croisic with and without storm surge barrier')
plt.ylabel('Surge (m)')
plt.xticks(rotation=25)
plt.grid()
plt.legend()
plt.show()

#%% Compare surge with and without barrier
plt.figure(dpi=300)
plt.plot(df['complete_date'], df['surge'], label='Without storm surge barrier')
plt.plot(df_gate['complete_date'], df_gate['surge'], 
         label='With storm surge barrier')
# plt.title('Surge at Le Croisic with and without storm surge barrier')
plt.ylabel('Surge (m)')
plt.xticks(rotation=25) 
plt.grid()
plt.legend()
plt.show()

#%% Plot surge with zoom on maximal peak
# Identify time of maximal surge
idx_max = df['surge'].idxmax()
date_max = df.loc[idx_max, 'complete_date']

# Graphic
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

ax.plot(df['complete_date'], df['surge'], label='Without storm surge barrier')
ax.plot(df_gate['complete_date'], df_gate['surge'], 
        label='With storm surge barrier')
ax.set_ylabel('Surge (m)')
ax.tick_params(axis='x', rotation=25)
ax.legend()
ax.set_title("Surge at Le Croisic with and without storm surge barrier")

# Zoom of the period including the maximal surge
axins = inset_axes(ax, width="35%", height="40%", loc='lower right',
                   bbox_to_anchor=(-0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)
axins.plot(df['complete_date'], df['surge'], label='Without storm surge barrier')
axins.plot(df_gate['complete_date'], df_gate['surge'], 
           label='With storm surge barrier')
axins.xaxis.set_major_locator(mdates.HourLocator(interval=9))  
axins.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))  # format : day-hour

# Define zoom window (+-12h around peak)
start_zoom = date_max - pd.Timedelta(hours=12)
end_zoom = date_max + pd.Timedelta(hours=12)
axins.set_xlim(start_zoom, end_zoom)

# Set y-axis limits with small margin
ymin = min(df[df['complete_date'].between(start_zoom, end_zoom)]['surge'].min(),
           df_gate[df_gate['complete_date'].between(start_zoom, end_zoom)]['surge'].min())
ymax = max(df[df['complete_date'].between(start_zoom, end_zoom)]['surge'].max(),
           df_gate[df_gate['complete_date'].between(start_zoom, end_zoom)]['surge'].max())
axins.set_ylim(ymin - 0.035, ymax + 0.035)

# Link zoom window to main plot
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.show()

 


