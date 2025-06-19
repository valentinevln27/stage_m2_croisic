# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:23:33 2025

@author: vanleene valentine
"""

#%% Needed libraries imports
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

#%% Data extraction
coast_path = '../qgis_stage_m2/newDomain/clv_chaincut_largedomain_v1.shp'
gdf = gpd.read_file(coast_path) 

# Sorting the GeoDataFrame by the 'id' column
gdf = gdf.sort_values(by='id', ascending=True)
# Resetting the index to ascending
gdf = gdf.reset_index(drop=True)

points_path = '../qgis_stage_m2/newDomain/openboundary_point.shp'
OB_points = gpd.read_file(points_path)

#%% Function to add interpolated points at fixed distances
def add_interpolated_points(start, end, step, reverse=False):
    """
    Generate points along a line from start to end with a fixed step distance.

    Parameters
    ----------
    start : tuple
        Coordinates (x, y) of the starting point.
    end : tuple
        Coordinates (x, y) of the ending point.
    step : float
        Fixed distance between generated points.
    reverse : bool, optional
        If True, reverses the order of generated points (default is False).

    Returns
    -------
    list
        A list of shapely.geometry.Point objects representing interpolated points.
    """
    points = []
    x0, y0 = start
    x1, y1 = end
    tot_dist = np.hypot(x1 - x0, y1 - y0)
    num_points = int(tot_dist // step)  # Number of points to add

    for i in range(1, num_points + 1):
        dist = i * step / tot_dist
        x = x0 + (x1 - x0) * dist
        y = y0 + (y1 - y0) * dist
        points.append(Point(x, y))
    
    return points[::-1] if reverse else points

#%% Generating additional points
coast_last = (gdf.iloc[-1].x, gdf.iloc[-1].y)
SW_point = (OB_points.x.iloc[1], OB_points.y.iloc[1])
NW_point = (OB_points.x.iloc[0], OB_points.y.iloc[0])
coast_first = (gdf.iloc[0].x, gdf.iloc[0].y)

# Points every 500m from coast_last to NW_point
northern_boundary = add_interpolated_points(coast_last, NW_point, 500)
# Points every 500m from NW_point to SW_point
western_boundary = add_interpolated_points(NW_point, SW_point, 500)
# Points every 500m from coast_first to SW_point, in reverse order
southern_boundary = add_interpolated_points(coast_first, SW_point, 500, reverse=True)

#%% Adding new points to GeoDataFrame
new_points = northern_boundary + western_boundary + [Point(SW_point)] + southern_boundary

data = [{'id': len(gdf) + i + 1, 'x': p.x, 'y': p.y, 'geometry': p} for i, p in enumerate(new_points)]
new_gdf = gpd.GeoDataFrame(data, geometry='geometry')

# Merge with original GeoDataFrame
gdf = pd.concat([gdf, new_gdf], ignore_index=True)

#%% Graphical representation
fig, ax = plt.subplots(figsize=(10,6), dpi=300)
# Defining the boundaries of the figure
margin_lat = 3500
margin_long = 1500
x_min, x_max = gdf['x'].min() - margin_lat, gdf['x'].max() + margin_lat
y_min, y_max = gdf['y'].min() - margin_long, gdf['y'].max() + margin_long
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Representing all the points
gdf.plot(marker='o', color='royalblue', markersize=10, ax=ax)

# Adding legend
plt.title('Le Croisic in points', fontsize=20, fontweight='bold')
plt.xlabel('Latitude (m)', fontsize=15)
plt.ylabel('Longitude (m)', fontsize=15)

# Graphical option and plotting
plt.tight_layout()
plt.grid()
plt.show()

#%% Exporting the new shapefile
# exit_path = '../qgis_stage_m2/newDomain/clv_chaincut_complete_largedomain_v1.shp'
# # In case the file already exists, it will be removed
# for fichier in glob.glob(f'{exit_path}.*'):
#     os.remove(fichier)
# # Saving the GeoDataFrame as a .shp file
# gdf.to_file(exit_path, driver='ESRI Shapefile')
