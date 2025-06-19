# Description of all scripts used during the internship

Note that all descriptions are provided at the beginning of each script.

## [bottom_nature](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/bottom_nature_v2.py)
This script (version 2) processes a shapefile containing spatial data about bed materials and classifies different substrate types into three broad categories. It then counts the number of polygons belonging to each category.

## [combinaitions](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/combinations_v2.py)
This script (version 2) generates different combinations of coefficients based on three types of substrates (mud/sand, gravel and rock). It then applies these combinations to a GeoDataFrame (gdf), assigning a specific coefficient (ks) to each entity. Finally the number of polygon per class is calculated.

## [compiling_grib_data](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/compiling_grib_data.py)
This script processes meteorological wind data from different GRIB files and compile them into a single NetCDF file. The result file of compiled data is used to analyse storms to identify a theorical storm.

## [cut_points](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/cut_points_v2.py)
In this script (version 2), we work with either the southern or northern section of the coastline around Le Croisic and the Guérande salt marshes.
<br>
The goal is to select specific points along the coastline in order to gradually increase the spacing between them, moving away from Le Croisic either to the north or to the south, depending on which coastline section is used.

## [era5_data_download](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/era5_data_download.py)
This script, provided by the ERA5 API, allows us to download the desired ERA5 data.

## [martin_xynthia_celine](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/martin_xynthia_celine_v2.py)
This script (version 2) extracts swell and weather parameters during three storms periods (Lothar-Martin, Xynthia, Céline) using Candhis buoy data, Copernicus wave models (GOWR), and ERA5 reanalysis. It computes mean values for Hs, Tp and Dp as well as the minimal value of atmospherical pressure and the associated wind values (u10 and v10). The uncertainties for each values are also computed and given. 
<br>
All outputs are given in a summary table per storm and buoy location.

## [prediction_data_separation](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/prediction_data_separation.py)
In this script, we use an Excel file containing sea level predictions during the Lothar-Martin, Xynthia, and Celine storms at three different tide gauges: 
    - Le Croisic (CR), 
    - Saint-Nazaire (SN), 
    - and Les Sables-d’Olonne (SO).
<br> 
The data are then separated by storm and by tide gauge. We also subtract the mean sea level to allow for a better comparison with the model data used in the tide_comparison script.

## [tide_comparison](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/tide_comparison_v5.py)
This script (version 5) reads predicted tidal gauge data and, for i signals, a signal is extracted from a specific point in a simulation with a given value of the Strickler coefficient (Ks). It is important to note that each simulation with a Ks value contains six distinct points.
<br>
Then, for similar time steps, the predicted tide values from the tide gauge and the simulated values from a given point are gathered into a dataframe to conduct a comparative study. This analysis is based on the calculation of three error metrics: MAE (Mean Absolute Error), RMSE (Root Mean Square Error), and Pbiais (percentage bias).
<br>
Finally, the MAE, RMSE, and Pbiais results are displayed for each simulation. Specifically, for a simulation with a given Ks value, and for each storm studied.

## [trimming](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/trimming.py)
This script trims a .slf layer from a mesh whose nodes contain bathymetry values. Thus, when the bathymetry value of a point exceeds a defined maximum value, it is replaced by this maximum value.