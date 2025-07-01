# Description des scripts utilisés pendant le stage

Toutes ces descriptions sont fournies au début de chaque script. Il est cependant à noter que les scripts sont tous écris et commentés en anglais.

## [bottom_nature](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/bottom_nature_v2.py)
Ce script (version 2) traite un fichier shapefile contenant des données spatiales sur les matériaux de fond et classe les différents types de substrats en trois grandes catégories. Il compte ensuite le nombre de polygones appartenant à chaque catégorie.

## [combinaitions](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/combinations_v2.py)
Ce script (version 2) génère différentes combinaisons de coefficients basées sur trois types de substrats (vase/sable, gravier et roche). Il applique ensuite ces combinaisons à une GeoDataFrame (gdf), en attribuant un coefficient spécifique (ks) à chaque entité. Enfin, le nombre de polygones par classe est calculé.

## [compiling_grib_data](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/compiling_grib_data.py)
Ce script traite les données de vent météorologiques issues de différents fichiers GRIB et les compile dans un seul fichier NetCDF. Le fichier compilé est utilisé pour analyser les tempêtes et identifier une tempête théorique.

## [completing_mesh_v3](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/completing_mesh_v3.py)
Ce script ajoute des points pour compléter le domaine étudié défini par deux points au large. Le côté est du domaine est formé par un trait de côte construit dans QGIS. La GeoDataFrame finale contenant les points ajoutés est ensuite exportée en tant que shapefile.
<br>
Le premier point sur le trait de côte (au sud) a un ID de 1, tandis que le dernier a un ID de m. Le point dans le coin nord-ouest du rectangle a un ID de m+1, et celui dans le coin sud-ouest a un ID de m+2.
<br>
Pour une meilleure visualisation, une figure est générée à la fin du script pour afficher le domaine dessiné avec les points, y compris le trait de côte, avec les IDs des points mentionnés.
<br>
📌 Remarque : certains paramètres peuvent nécessiter d’être modifiés, comme la distance maximale entre deux points.

## [cut_points](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/cut_points_v2.py)
Dans ce script (version 2), on travaille avec la section sud ou nord du trait de côte autour du Croisic et des marais salants de Guérande.
<br>
L’objectif est de sélectionner des points spécifiques le long du trait de côte afin d’augmenter progressivement l’espacement entre eux, en s’éloignant du Croisic vers le nord ou vers le sud, selon la section utilisée.

## [era5_data_download](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/era5_data_download.py)
Ce script, fourni par l’API ERA5, permet de télécharger les données ERA5 souhaitées.

## [martin_xynthia_celine](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/martin_xynthia_celine_v2.py)
Ce script (version 2) extrait les paramètres de houle et météorologiques pendant trois périodes de tempêtes (Lothar-Martin, Xynthia, Céline) à l’aide des données de bouées Candhis, des modèles de houle Copernicus (GOWR) et des réanalyses ERA5. Il calcule les valeurs moyennes pour Hs, Tp, et Dp, ainsi que la pression atmosphérique minimale et les valeurs de vent associées (u10 et v10). Les incertitudes pour chaque valeur sont également calculées et fournies.
<br>
Tous les résultats sont présentés dans un tableau récapitulatif par tempête et par localisation de bouée.

## [prediction_data_separation](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/prediction_data_separation.py)
Dans ce script, on utilise un fichier Excel contenant les prédictions de niveau marin pendant les tempêtes Lothar-Martin, Xynthia et Céline à trois marégraphes différents :

    Le Croisic (CR),

    Saint-Nazaire (SN),

    et Les Sables-d’Olonne (SO).
    <br>

Les données sont ensuite séparées par tempête et par marégraphe. On soustrait aussi le niveau moyen de la mer pour permettre une meilleure comparaison avec les données de simulation utilisées dans le script tide_comparison.

## [surge](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/surge.py)
Ce script analyse et visualise les données de surcote à partir des niveaux marins au Croisic pendant la tempête Xynthia, en comparant les scénarios avec et sans barrière anti-submersion.

## [tide_comparison](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/tide_comparison_v5.py)
Ce script (version 5) lit les données de marégraphes prédites et, pour i signaux, extrait un signal d’un point spécifique d’une simulation avec une valeur donnée du coefficient de Strickler (Ks). Il est important de noter que chaque simulation avec un Ks donné contient six points distincts.
<br>
Ensuite, pour des pas de temps similaires, les valeurs de marée prédites par le marégraphe et les valeurs simulées pour un point donné sont rassemblées dans un dataframe afin d’effectuer une étude comparative. Cette analyse repose sur le calcul de trois métriques d’erreur : MAE (erreur absolue moyenne), RMSE (racine de l’erreur quadratique moyenne) et Pbiais (biais en pourcentage).
<br>
Enfin, les résultats de MAE, RMSE et Pbiais sont affichés pour chaque simulation, spécifiquement pour une valeur de Ks donnée et pour chaque tempête étudiée.

## [trimming](https://gitlab.univ-nantes.fr/vanleene-v-1/croisic_stage/-/blob/main/script_python/trimming.py)
Ce script tronque une couche .slf issue d’un maillage dont les nœuds contiennent des valeurs de bathymétrie. Ainsi, lorsqu’une valeur de bathymétrie dépasse une valeur maximale définie, elle est remplacée par cette valeur maximale.