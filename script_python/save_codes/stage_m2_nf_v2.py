# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:37:31 2025

@author: valentine vanleene
"""

#%% Explaining what the script does
"""
This script processes a shapefile containing spatial data about bed materials and classifies 
different substrate types into four broad categories. It then counts the number of polygons 
belonging to each category.
"""

#%% Needed librairies imports
import pandas as pd
import geopandas as gpd
import os
import glob

#%% Data extraction
# Path to the shapefile
fullFilePath = '../bed_materials_2023/2154/modif/natures_fond_50000_2154_modif_decoupe.shp'
gdf = gpd.read_file(fullFilePath) # Extract the data as a GeoDataFrame
gdf['NF'] = 'NF' # Add a new column in gdf to store the new category

# Grouping typelem in 4 classes
for i in range(gdf.shape[0]):
    if gdf.loc[i, 'typelem'].startswith('NFS'): # For all sand substrat
        gdf.loc[i, 'NF'] = 'NFSV'
    elif gdf.loc[i, 'typelem'].startswith('NFR'): # For all rock substrat
        gdf.loc[i, 'NF'] = 'NFR'
    elif gdf.loc[i, 'typelem'].startswith('NFV'): # For all mud substrat
        gdf.loc[i, 'NF'] = 'NFSV'
    else:
        gdf.loc[i, 'NF'] = 'NFCG' # For all gravel and pebbles substrat

# Calculating the number of polygon per class
# Count occurrences of each class in the 'NF' column
class_counts = gdf['NF'].value_counts()

# Extract counts safely (default to 0 if a class is missing)
nfr = class_counts.get('NFR', 0)
nfcg = class_counts.get('NFCG', 0)
nfs = class_counts.get('NFSV', 0)

#%% Saving the modified file
# exit_path = '../bed_materials_2023/2154/modif/natures_fond_50000_2154_modif_decoupe_groupe.shp'
# # In case the file already exist, it will be removed
# for fichier in glob.glob(f'{exit_path}.*'):
#     os.remove(fichier)
# # Saving the GeoDataFrame as a .shp file
# gdf.to_file(exit_path, driver='ESRI Shapefile')


