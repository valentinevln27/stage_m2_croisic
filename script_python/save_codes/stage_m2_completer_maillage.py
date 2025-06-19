# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:03:19 2025

@author: valentine vanleene
"""

#%% Explaining what the script does
"""
This script adds points along a 17000-meter-long rectangle, with its eastern side forming 
a coastline. The GeoDataFrame containing the added points is then exported as a .shp file.

The first point on the coastline (South) has an ID of 1, while the last one has an ID of m. 
The point at the northwestern corner of the rectangle has an ID of m+1, and the point at 
the southwestern corner has an ID of m+2.

For better visualization, a plot is generated at the end of the script to display the 
rectangle drawn with points, including the coastline, with the IDs of the mentioned points 
labeled.
"""

#%% Needed librairies imports
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import os
import glob

#%% Data extraction
# Path to the shapefile of the coastal line in points
path = '../traits_de_cote/clv_short/chain10m_clv_short_poly_v1_smooth.shp'
gdf = gpd.read_file(path) # Extract the data as a GeoDataFrame

#%% Creating points id=m+1 and id=m+2
temp_gdf = gdf.copy() # gdf is a save of the file and only temp_gdf will be modified
coast_deep_length = 17000 # The length we want between the points id=m and id=m+1

# Calculating the position of the two points
WN_point_x = gdf['x'].iloc[-1] - coast_deep_length 
WN_point_y = gdf['y'].iloc[-1]
WS_point_x = WN_point_x
WS_point_y = gdf['y'].iloc[0]

# Saving the id of the point id=m
m = temp_gdf.shape[0]

#%% Useful function to add a point to the GeoDataFrame
def add_point(px, py, temp_geodf):
    """
    Adds a new point to the given GeoDataFrame.
    
    Parameters
    ----------
    px : float
        The x-coordinate of the new point.
    py : float
        The y-coordinate of the new point.
    temp_geodf : GeoDataFrame
        The GeoDataFrame to which the new point will be added.

    Returns
    -------
    temp_geodf : GeoDataFrame
        The updated GeoDataFrame including the newly added point.
    """
    # Creating a DataFrame for the point that we will add
    p = {'id': temp_geodf['id'].iloc[-1]+1, 'x': px, 'y': py, 
             'geometry': Point(px, py)} 
    p_df = pd.DataFrame([p]) 
    # Converting the DataFrame of the point into a GeoDataFrame
    new_geodf = gpd.GeoDataFrame(p_df, geometry='geometry')
    # Adding the point to the GeoDataFrame containing all points 
    temp_geodf = pd.concat([temp_geodf, new_geodf], ignore_index=True)
    return temp_geodf

#%% Add points between id=m and id=m+1 | NORTH BOUNDARY
def NorthBoundary(yN1, xNn, d0, dmax, f, geodf):
    """
    Adds points along the northern boundary between an initial and final x-coordinate.

    Parameters
    ----------
    yN1 : float
        The starting y-coordinate for the eastern-northern boundary point (id=m).
    xNn : float
        The target x-coordinate where the points should stop.
    d0 : float
        The initial spacing between points.
    dmax : float
        The maximum spacing allowed between points.
    f : float
        The growth factor for spacing between consecutive points.
    geodf : GeoDataFrame
        The GeoDataFrame to which the new points will be added.

    Returns
    -------
    geodf : GeoDataFrame
        The updated GeoDataFrame including the newly added northern boundary points.
    dnew : float
        The adjusted final spacing between the last two consecutive points.
    """
    d = d0 # The space between two consecutive points
    n = 2 # The power in the geometric sequence with a common ratio of n
    while d < dmax: # While the length between two points doesn't exceed a maximal length
        point_x = geodf['x'].iloc[-1] - d # Compute the x-coordinate
        geodf = add_point(point_x, yN1, geodf) # Add the new point
        d = (f**(n-2)) * d0 # Recalculate the space between two consecutive points
        n += 1 
    last_x = geodf['x'].iloc[-1] # Save the x-coordinate of the last point
    dist = last_x - xNn # Calculate the length between the two last points
    nb_segments = round(dist/dmax) # Calculate the remaining number of segment 
    dnew = dist/nb_segments # Change the length between two consecutive points
    while (last_x - dnew) > xNn: # For when the maximal length is exceeded
        point_x = last_x - dnew # Compute the x-coordinate
        geodf = add_point(point_x, yN1, geodf) # Add the new point
        last_x = geodf['x'].iloc[-1] # Save the x-coordinate of the last point
    return geodf, dnew

# Using the above function
nb_point_gdf = gdf.shape[0]
temp_gdf, dnew = NorthBoundary(temp_gdf['y'].iloc[nb_point_gdf-1], WN_point_x, 10, 500, 
                               1.3, temp_gdf)      

# Saving the id of the point id=m+1  
m_1 = temp_gdf.shape[0]

#%% Add points between id=m+1 and id=m+2 | OPEN/DEEPSEA BOUNDARY
L = WN_point_y - WS_point_y # Calculate the length between the points id=m+1 and id=m+2
# The length between the last two consecutive points in the GeoDataFrame
d2 = temp_gdf['x'].iloc[-2] - temp_gdf['x'].iloc[-1] 
nb_segment = round(L/d2) # The number of segment to add to the western side of the rectangle

# Adding the points of the western side of the rectangle
for i in range(1,nb_segment):
    point_y = temp_gdf['y'].iloc[-1] - dnew # Compute the y-coordinate
    temp_gdf = add_point(temp_gdf['x'].iloc[-1], point_y, temp_gdf) # Add the point

# Adding the southern-western point (id=m+2) into the GeoDataFrame
temp_gdf = add_point(WS_point_x, WS_point_y, temp_gdf)

# Saving the id of the point id=m+2
m_2 = temp_gdf.shape[0]

#%% Add points between id=m+2 and id=1 | SOUTH BOUNDARY
def SouthBoundary(xN1, yN1, xNn, d0, dmax, f, geodf):
    """
    Adds points along the southern boundary between an initial and final x-coordinate.

    Parameters
    ----------
    xN1 : float
        The starting x-coordinate for the western-southern boundary point (id=m+2).
    yN1 : float
        The starting y-coordinate for the western-southern boundary point (id=m+2).
    xNn : float
        The target x-coordinate where the points should stop.
    d0 : float
        The initial spacing between points.
    dmax : float
        The maximum spacing allowed between points.
    f : float
        The growth factor for spacing between consecutive points.
    geodf : GeoDataFrame
        The GeoDataFrame to which the new points will be added.

    Returns
    -------
    geodf : GeoDataFrame
        The updated GeoDataFrame including the newly added southern boundary points.
    """
    d = d0 # The space between two consecutive points
    n = 2 # The power in the geometric sequence with a common ratio of n
    list_x = [] # A list used to store the points of the southern side of the rectangle
    list_x.append(xN1) # Add the point id=1 for initialisation
    while d < dmax: # While the length between two points doesn't exceed a maximal length
        point_x = list_x[-1] - d # Compute the x-coordinate
        list_x.append(point_x) # Add the point to the list
        d = (f**(n-2)) * d0 # Recalculate the space between two consecutive points
        n += 1
    list_x.pop(0) # Remove the first element (initial point)
    last_x = list_x[-1] # Save the x-coordinate of the last point
    dist = last_x - xNn # Calculate the length between the two last points
    nb_segments = round(dist/dmax) # Calculate the remaining number of segment 
    dnew = dist/nb_segments # Change the length between two consecutive points
    while (last_x - dnew) >= xNn: # For when the maximal length is exceeded
        point_x = last_x - dnew # Compute the x-coordinate
        list_x.append(point_x) # Add the new point to the list
        last_x = list_x[-1] # Save the x-coordinate of the last point
    # list_x.pop()  # Remove the last point that would go beyond xNn
    # Reversing the list in order to have the point nearest point of point id=1 with the 
    # biggest id number
    reversed_list_x = list(reversed(list_x))
    # Adding all points to the GeoDataFrame
    for px in reversed_list_x:
        geodf = add_point(px, yN1, geodf)
    return geodf

# Using the above function
temp_gdf = SouthBoundary(temp_gdf['x'].iloc[0], temp_gdf['y'].iloc[0], 
                               WS_point_x, 10, 500, 1.3, temp_gdf)        

#%% Graphical representation
fig, ax = plt.subplots(figsize=(10,6), dpi=300)
# Defining the boundaries of the figure
margin_lat = 3500
margin_long = 1500
x_min, x_max = temp_gdf['x'].min() - margin_lat, temp_gdf['x'].max() + margin_lat
y_min, y_max = temp_gdf['y'].min() - margin_long, temp_gdf['y'].max() + margin_long
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Representing all the points
temp_gdf.plot(marker='o', color='royalblue', markersize=10, ax=ax)

# Defining the significant points (points with id=1,m,m+1 and m+2) and the positionning of 
# the text
significant_points = {
    1: {'id_point':'1', 'xytext': (300, -300), 'ha': 'left', 'va': 'top'},   
    m: {'id_point':'m','xytext': (300, 300), 'ha': 'left', 'va': 'bottom'},
    m_1: {'id_point':'m+1','xytext': (-300, 300), 'ha': 'right', 'va': 'bottom'},  
    m_2: {'id_point':'m+2','xytext': (-300, -300), 'ha': 'right', 'va': 'top'}  
    }

# Add labels for points with id=1,m,m+1 and m+2
for _, row in temp_gdf.iterrows():
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

#%% Exporting the new shapefile
exit_path = '../traits_de_cote/clv_short/chain10m_completeclv_short_vertices_v1_smooth.shp'
# In case the file already exist, it will be removed
for fichier in glob.glob(f'{exit_path}.*'):
    os.remove(fichier)
# Saving the GeoDataFrame as a .shp file
temp_gdf.to_file(exit_path, driver='ESRI Shapefile')