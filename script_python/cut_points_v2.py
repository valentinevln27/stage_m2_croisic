# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 09:54:22 2025

@author: vanleene valentine
"""

#%% Explaining what the script does
"""
In this script, we work with either the southern or northern section of the 
coastline around Le Croisic and the Guérande salt marshes.

The goal is to select specific points along the coastline in order to gradually 
increase the spacing between them, moving away from Le Croisic either to the north 
or to the south, depending on which coastline section is used.
"""

#%% Needed librairies import
import pandas as pd
import geopandas as gpd
import os
import glob

#%% Exporting the coastal lines
# One charged at a time
shapefile_path = "../qgis_stage_m2/newDomain/southern_part/clv_chain10_largedomain_south_v2.shp"
# shapefile_path = "../qgis_stage_m2/newDomain/clv_chain10_largedomain_north_v1.shp"
gdf = gpd.read_file(shapefile_path)

#%% Selecting the points
# Reverse the order of points to start from the last and go to the first
gdf = gdf[::-1].reset_index(drop=True) # Only needed for the southern coastline

# Define spacing steps for point selection
steps = [1, 2, 3, 5, 7, 10, 15, 30, 50, 70, 100, 150, 300, 500]  # "Keep 1 point every X points"
selected_gdfs = []

# Loop to apply point selection with increasing spacing
start_index = 0
for step in steps:
    selected = gdf.iloc[start_index::step][:6]  # Select 6 points spaced by the current step
    selected_gdfs.append(selected)
    start_index = selected.index.values[-1]  # Update the starting index for the next set

# Add the remaining points spaced by 5000 m
selected_gdfs.append(gdf.iloc[start_index::500]) # Final points spaced every 5000 meters

# Merge all selected points
selected_gdf = gpd.GeoDataFrame(pd.concat(selected_gdfs, ignore_index=True))

# Remove duplicate geometries (to avoid overlapping points)
selected_gdf = selected_gdf.drop_duplicates(subset='geometry')

# Remove the first or last point if needed (to clean the extremities)
selected_gdf = selected_gdf.iloc[2:]  #  Remove the first two southernmost points
# selected_gdf = selected_gdf.iloc[1:]  # Remove the last northern point

# Reverse again for southern coastline (if needed)
selected_gdf = selected_gdf[::-1].reset_index(drop=True) # Only for the southern coastline

#%% Saving the new shapefile
exit_path = "../qgis_stage_m2/newDomain/southern_part/clv_chaincut_largedomain_south_v2.shp"
# exit_path = "../qgis_stage_m2/newDomain/clv_chaincut_largedomain_north_v1.shp"
# In case the file already exist, it will be removed
for fichier in glob.glob(f'{exit_path}.*'):
    os.remove(fichier)
# Saving the GeoDataFrame as a .shp file
selected_gdf.to_file(exit_path, driver='ESRI Shapefile')
    
