# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 09:21:38 2025

@author: vanleene valentine
"""

#%% Librairies import
import pandas as pd
import geopandas as gpd
import os
import glob
import xarray as xr
from tqdm import tqdm  # pour afficher la progression
import numpy as np
from scipy.spatial import cKDTree

#%% Lecture des données 
# lecture maillage
mesh_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/ks7_v5_1.3_4326.shp'
gdf_mesh = gpd.read_file(mesh_path) # 4326
# lecture point open boundary et bouée
# Selection des point du maillage qui sont dans open boundary
cl_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/clv_chaincut_complete_largedomain_1.3_v2_4326.shp'
gdf_cl = gpd.read_file(cl_path) # 4326
# Selection des point des bouées
buoy_path = 'D:/stage_m2/qgis_stage_m2/newDomain/swell_wind/buoy.shp'
gdf_buoy = gpd.read_file(buoy_path) # 4326

#%% 0 = no data
