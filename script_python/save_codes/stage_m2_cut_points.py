# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 09:54:22 2025

@author: vanleene valentine
"""

import geopandas as gpd
import os
import glob
from shapely.geometry import Point
#%%
# Charger le fichier shapefile
shapefile_path = "C:/cours/master/M2/Stage/qgis_stage_m2/newDomain/clv_chain10_largedomain_south_v1.shp"
gdf = gpd.read_file(shapefile_path)

#%%
# S'assurer que les points sont triés pour une sélection ordonnée
# gdf = gdf.sort_values(by=["geometry"])  # À adapter selon l'organisation de tes points
#%%
# Liste des distances et résultats
distances = [10, 20, 30, 50, 70, 100, 150, 300, 500]
selected_points = []
remaining_points = gdf.copy()

# Fonction pour sélectionner les points espacés d'une distance donnée
def select_points(gdf, spacing, count=6):
    selected = []
    last_point = None
    for _, row in gdf.iterrows():
        point = row.geometry
        if last_point is None or point.distance(last_point) >= spacing:
            selected.append(row)
            last_point = point
        if len(selected) == count:
            break
    return selected

#%%
# Sélection des points avec les distances spécifiées
for d in distances[:-1]:  # On garde 500m pour la fin
    points = select_points(remaining_points, d)
    selected_points.extend(points)
    remaining_points = remaining_points.drop([p.name for p in points])  # Supprimer les points sélectionnés

#%%
# Ajouter les derniers points espacés de 500m
while not remaining_points.empty:
    points = select_points(remaining_points, 500, 1)
    if not points:
        break
    selected_points.extend(points)
    remaining_points = remaining_points.drop([p.name for p in points])
    
#%%
# Créer un GeoDataFrame avec les points sélectionnés
selected_gdf = gpd.GeoDataFrame(selected_points, geometry="geometry", crs=gdf.crs)

# Sauvegarder le nouveau shapefile
exit_path = "C:/cours/master/M2/Stage/qgis_stage_m2/newDomain/clv_chaincut_largedomain_south_v1.shp"
# In case the file already exist, it will be removed
for fichier in glob.glob(f'{exit_path}.*'):
    os.remove(fichier)
# Saving the GeoDataFrame as a .shp file
selected_gdf.to_file(exit_path, driver='ESRI Shapefile')
    
