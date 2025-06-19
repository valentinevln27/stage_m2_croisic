# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:13:11 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""
This script generates different combinations of coefficients based on three types of 
substrates (mud, sand, and rock). It then applies these combinations to a GeoDataFrame 
(gdf), assigning a specific coefficient (ks) to each entity. 
Finally the number of polygon per class is calculated.
"""

#%% Needed librairies imports
import pandas as pd
import geopandas as gpd
import os
import glob
import itertools

#%% Bed materials data extraction
# Extracting data from the file
# fullFilepath = '../bed_materials_2023/2154/modif/natures_fond_50000_2154_modif_decoupe_groupe_simplifie.shp'
fullFilepath = 'D:/nf_mourad/bed_materials_0225_single_parts.shp'
gdf = gpd.read_file(fullFilepath) # Extract the data as a GeoDataFrame

#%% Add a Ks value for all possible combinations
# Define possible values for each substrate type
mud_sand = [35, 45, 50, 55]       # Coefficients for mud and sand
gravel = [35, 40] # Coefficients for gravels
rock = [20, 30]  # Coefficients for rock

# Generating all possible combinations (each combination contains one element from each list)
combinations = [list(combo) for combo in itertools.product(mud_sand, gravel, rock)]
i = 1 # Counter to name the columns
for combo in combinations:
    print(combo)
    gdf[f'ks{i}'] = 0 # Add a new column in gdf to store ks values
    for idx, nf in enumerate(gdf['NF']):
        # Assigning the ks value based on the substrate type ('NF')
        if nf == 'NFSV': # If the substrate is "mud" or "sand"
            gdf.loc[idx, f'ks{i}'] = combo[0]
        elif nf == 'NFR': # If the substrate is "rock"
            gdf.loc[idx, f'ks{i}'] = combo[1]
        else: # If the substrate is "gravel"
            gdf.loc[idx,f'ks{i}'] = combo[2]
    i += 1  # Increment the counter for the next column

#%% Exporting the new shapefile
# exit_path = '../bed_materials_2023/2154/modif/natures_fond_50000_2154_modif_decoupe_simplifie_combi.shp'
exit_path = 'D:/nf_mourad/bed_materials_0225_single_parts_combi.shp'
# In case the file already exist, it will be removed
for fichier in glob.glob(f'{exit_path}.*'):
    os.remove(fichier)
# Saving the GeoDataFrame as a .shp file
gdf.to_file(exit_path, driver='ESRI Shapefile')



