# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:33:36 2025

@author: vanleene valentine
"""

#%% Explaining the script
""" A changer
This script adds points to complete the studied domain defined by two offshore 
points. The, the eastern side of the domain is formed by a coastline construct
in QGIS. The final GeoDataFrame containing the added points is then exported as 
a .shp file.

The first point on the coastline (South) has an ID of 1, while the last one has 
an ID of m. The point at the northwestern corner of the rectangle has an ID of 
m+1, and the point at the southwestern corner has an ID of m+2.

For better visualization, a plot is generated at the end of the script to display 
the domain drawn with points, including the coastline, with the IDs of the 
mentioned points labeled.
"""

#%% Needed libraries imports
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from shapely.geometry import Point

#%% Needed functions
def add_interpolated_points(start, end, step_initial, max_dist, factor, reverse_id=False):
    """
    Generates a list of interpolated points along a straight line between two 
    coordinates.

    Parameters
    ----------
    start : tuple of float
        The starting point as (x, y) coordinates.
    end : tuple of float
        The ending point as (x, y) coordinates.
    step_initial : float
        The initial distance between two consecutive points.
    max_dist : float
        The maximum allowed spacing between points.
    factor : float
        The factor by which the spacing increases at each step.
    reverse_id : bool, optional
        If True, the resulting list of points is returned in reverse order.
        Useful when the direction of interpolation needs to be flipped (for the
        southern boundary).

    Returns
    -------
    list
        A list of interpolated Point objects from start to end, spaced increasingly up to `max_dist`,
        by a factor f and then uniformly if the remaining distance is too large.
    """
    points = [] # List to stock all points coordinates and corresponding id
    # Extract coordinates from the start and end points
    x0, y0 = start
    x1, y1 = end
    # Compute the total distance between start and end using the Euclidean norm
    dist = np.hypot(x1 - x0, y1 - y0)
    
    # If the distance is smaller than or equal to the initial step, just return 
    # the end point (no intermediate points needed)
    if dist <= step_initial:
        return [Point(x1, y1)]  # Directly add end point if very close

    # Phase 1: Add points with increasing spacing up to max_dist
    # Initialize distance counter
    current_dist = step_initial
    step_size = step_initial
    # While we haven't reached either the total distance or the max spacing limit
    while current_dist < min(dist, max_dist):
        # Calculate the point coordinates at current_dist along the line
        x = x0 + (x1 - x0) * current_dist / dist
        y = y0 + (y1 - y0) * current_dist / dist
        points.append(Point(x, y))
        # Move further along the line, increasing the step size
        current_dist += step_size
        step_size *= factor  # Gradually increase spacing

    # Phase 2: Uniform spacing if distance > max_dist
    if current_dist < dist:
        # Estimate how many segments we can fit in the remaining distance
        remaining_dist = dist - current_dist
        num_segments = round(remaining_dist / max_dist)
        # Compute a uniform step for the remaining distance 
        new_step = remaining_dist / max(1, num_segments) 
        # Add points with uniform spacing until reaching the end
        while current_dist < dist:
            x = x0 + (x1 - x0) * current_dist / dist
            y = y0 + (y1 - y0) * current_dist / dist
            points.append(Point(x, y))
            current_dist += new_step

    # Reverse the list of points if requested 
    return points[::-1] if reverse_id else points 

def add_fixed_spacing_points(start, end, step):
    """
    Generates evenly spaced points along a straight line between two coordinates.

    Parameters
    ----------
    start : tuple of float
        The starting point as (x, y) coordinates.
    end : tuple of float
        The ending point as (x, y) coordinates.
    step : float
        Desired distance between each point (in the same unit as the coordinates, e.g., meters).

    Returns
    -------
    points : list of shapely.geometry.Point
        A list of Point objects placed at regular intervals along the line from start to end.
        The end point itself is not included unless it coincides with the last step.
    """
    points = [] # List to stock all points coordinates and corresponding id
    # Extract coordinates from the start and end points
    x0, y0 = start
    x1, y1 = end
    # Compute the total distance between start and end using the Euclidean norm
    dist = np.hypot(x1 - x0, y1 - y0)

    # Compute the number of segments (steps), rounded to the nearest integer
    num_segments = round(dist / step)
    # Recalculate step size to evenly distribute points along the entire length
    new_step = dist / max(1, num_segments)  
    
    # Initialize distance counter
    current_dist = new_step
    # Add points at regular intervals along the line
    while current_dist < dist:
        # Calculate the new point position at the current distance
        x = x0 + (x1 - x0) * current_dist / dist
        y = y0 + (y1 - y0) * current_dist / dist
        points.append(Point(x, y))
        current_dist += new_step

    return points

#%% Data extraction
# Coastline points
coast_path = '../qgis_stage_m2/newDomain/clv_chaincut_largedomain_v2.shp'
gdf = gpd.read_file(coast_path) 
# Changing the column id's type
gdf['id'] = gdf['id'].astype(int)
# Sorting the GeoDataFrame by the 'id' column
gdf = gdf.sort_values(by='id', ascending=True)
# Resetting the index to ascending
gdf = gdf.reset_index(drop=True)
m = gdf['id'].iloc[-1]

# Open/Deep Boundary points form the corners of the domain
points_path = '../qgis_stage_m2/newDomain/openboundary_point_v2.shp'
OB_points = gpd.read_file(points_path)

#%% Generating additional points
# Get the last point of the coastline 
coast_last = (gdf.iloc[-1].x, gdf.iloc[-1].y)
# coast_last = gdf.loc[gdf['y'].idxmax(), ['x', 'y']].values 
# Define the southwest and northwest points of the open boundary
SW_point = (OB_points.x.iloc[1], OB_points.y.iloc[1]) # Southwest corner of the boundary
NW_point = (OB_points.x.iloc[0], OB_points.y.iloc[0]) # Northwest corner of the boundary
coast_first = (gdf.iloc[0].x, gdf.iloc[0].y)

# Generate points along the open boundary section
northern_boundary = add_interpolated_points(coast_last, NW_point, 500, 3000, 1.1)
western_boundary = add_fixed_spacing_points(NW_point, SW_point, 3000)
southern_boundary = add_interpolated_points(coast_first, SW_point, 500, 3000, 1.1, 
                                            reverse_id=True)  

#%% Adding new points to GeoDataFrame
# Concatenate all boundary points
new_points = northern_boundary + [Point(NW_point)] + western_boundary + southern_boundary

# Create a list of dictionaries representing each new point, assigning a unique
# ID continuing from the last index in the existing GeoDataFrame.
data = [{'id': len(gdf) + i + 1, 'x': p.x, 'y': p.y, 'geometry': p} for i, p in enumerate(new_points)]
# Convert the list of new points into a new GeoDataFrame
new_gdf = gpd.GeoDataFrame(data, geometry='geometry')

# Merge with original GeoDataFrame in order to complete the line of the domain
gdf = pd.concat([gdf, new_gdf], ignore_index=True)

#%% Graphical representation
fig, ax = plt.subplots(figsize=(10,6), dpi=300)
# Defining the boundaries of the figure
margin_lat = 50000
margin_long = 25000
x_min, x_max = gdf['x'].min() - margin_lat, gdf['x'].max() + margin_lat
y_min, y_max = gdf['y'].min() - margin_long, gdf['y'].max() + margin_long
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Representing all the points
gdf.plot(marker='o', color='royalblue', markersize=10, ax=ax)

# Defining the significant points (points with id=1,m,m+1 and m+2) and the positionning of 
# the text

m_1 = m + len(northern_boundary) + 1
m_2 = m_1 + len(western_boundary) + 1 
significant_points = {
    1: {'id_point':'1', 'xytext': (6000, -6000), 'ha': 'left', 'va': 'top'},   
    m: {'id_point':'m','xytext': (6000, 6000), 'ha': 'left', 'va': 'bottom'},
    m_1: {'id_point':'m+1','xytext': (-6000, 6000), 'ha': 'right', 'va': 'bottom'},  
    m_2: {'id_point':'m+2','xytext': (-6000, -6000), 'ha': 'right', 'va': 'top'}  
    }

# Add labels for points with id=1,m,m+1 and m+2
for _, row in gdf.iterrows():
    if row['id'] in significant_points:
        params = significant_points[row['id']]
        ax.annotate(
            text=f"ID: {row['id']}\n(point {params['id_point']})",  # Add a text
            xy=(row['x'], row['y']),  # Positining the point
            # Positioning the text
            xytext=(row['x'] + params['xytext'][0], row['y'] + params['xytext'][1]),
            fontsize=10, # Define the size of the labels
            color='black', # # Define the color of the labels
            ha=params['ha'], va=params['va'],  # Text alignment
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8)  # Add an arrow
            )

# Adding legend
plt.title('Le Croisic in points', fontsize=20, fontweight='bold')
plt.xlabel('Latitude (m)', fontsize=15)
plt.ylabel('Longitude (m)', fontsize=15)

# Graphical option and plotting
plt.tight_layout()
plt.grid()
plt.show()

# #%% Exporting the new shapefile
# exit_path = '../qgis_stage_m2/newDomain/clv_chaincut_complete_largedomain_1_1_v2.shp'
# # In case the file already exists, it will be removed
# for fichier in glob.glob(f'{exit_path}.*'):
#     os.remove(fichier)
# # Saving the GeoDataFrame as a .shp file
# gdf.to_file(exit_path, driver='ESRI Shapefile')
