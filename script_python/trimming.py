# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:20:19 2025

@author: valentine vanleene
"""

#%% Explaining what the script does
"""
This script trims a .slf layer from a mesh whose nodes contain bathymetry values. 
Thus, when the bathymetry value of a point exceeds a defined maximum value, it 
is replaced by this maximum value.
"""

#%% Needed librairies imports
import geopandas as gpd
import glob
import os

#%% Data extraction
# Path to the shapefile of the coastal line in points
path = '../bluekenue_stage/meshbathy/mesh_short_1.3_10m_500m_d25_50_sidw_rgealti2023.shp'
gdf = gpd.read_file(path) # Extract the data as a GeoDataFrame

max_bathy = -5 # m, trim value

# Trimming the bathymetry
for i in range(gdf.shape[0]):
    if gdf.iloc[i, 2] >= max_bathy:
        gdf.iloc[i, 2] = max_bathy
        
#%% Exporting the new shapefile
exit_path = '../bluekenue_stage/geo/rmesh_short_1.3_10m_500m_d25_50.shp'
# In case the file already exist, it will be removed
for fichier in glob.glob(f'{exit_path}.*'):
    os.remove(fichier)
# Saving the GeoDataFrame as a .shp file
gdf.to_file(exit_path, driver='ESRI Shapefile')