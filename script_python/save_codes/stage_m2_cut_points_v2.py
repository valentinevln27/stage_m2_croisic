# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 09:54:22 2025

@author: vanleene valentine
"""
import pandas as pd
import geopandas as gpd
import os
import glob

# Charger le shapefile
shapefile_path = "../qgis_stage_m2/newDomain/southern_part/clv_chain10_largedomain_south_v2.shp"
# shapefile_path = "../qgis_stage_m2/newDomain/clv_chain10_largedomain_north_v1.shp"
gdf = gpd.read_file(shapefile_path)

# Renverser l'ordre des points pour commencer du dernier vers le premier
gdf = gdf[::-1].reset_index(drop=True) # Si le trait de côte est celui du sud

# Espacements et sélection des points
steps = [1, 2, 3, 5, 7, 10, 15, 30, 50, 70, 100, 150, 300, 500]  # Correspond à "supprimer 1 sur X"
selected_gdfs = []

# Boucle pour appliquer la sélection
start_index = 0
for step in steps:
    selected = gdf.iloc[start_index::step][:6]  # Prend 6 points en espaçant selon step
    selected_gdfs.append(selected)
    start_index = selected.index.values[-1]  # Ajuste l'index de départ pour la prochaine série

# Ajouter les derniers points espacés de 500m
selected_gdfs.append(gdf.iloc[start_index::500]) # Derniers points espacés de 5000m

# Fusionner tous les points sélectionnés
selected_gdf = gpd.GeoDataFrame(pd.concat(selected_gdfs, ignore_index=True))

# Supprimer les doublons géométriques (évite les points au même endroit)
selected_gdf = selected_gdf.drop_duplicates(subset='geometry')

# Supprimer le premier ou le dernier point après sélection
selected_gdf = selected_gdf.iloc[2:]  # Supprime les 2 premiers points sud
# selected_gdf = selected_gdf.iloc[1:]  # Supprime le dernier point nord

selected_gdf = selected_gdf[::-1].reset_index(drop=True) # pour sud uniquement

# Sauvegarder le nouveau shapefile
exit_path = "../qgis_stage_m2/newDomain/southern_part/clv_chaincut_largedomain_south_v2.shp"
# exit_path = "../qgis_stage_m2/newDomain/clv_chaincut_largedomain_north_v1.shp"
# In case the file already exist, it will be removed
for fichier in glob.glob(f'{exit_path}.*'):
    os.remove(fichier)
# Saving the GeoDataFrame as a .shp file
selected_gdf.to_file(exit_path, driver='ESRI Shapefile')
    
