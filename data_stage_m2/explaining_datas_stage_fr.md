# Description des données brutes utilisées pendant le stage

## Métadonnées des marégraphes
Les métadonnées sur les marégraphes du Croisic, de Saint-Nazaire et des Sables d'Olonne, comprennent notamment les noms des maregraphes, leur identifiant SHOM, leurs coordonées et valeur de niveau moyen.

## Prédictions marégraphiques
Le dossier records comprend deux autres dossiers, inputs et outputs, et un fichier excel, data_prediction_all.
Ce dernier fichier comprend les prédictions marégraphiques du SHOM au Croisic, à Saint-Nazaire et aux Sables d'Olonne pendant les tempêtes Lothar-Martin, Xynthia et Céline.
Le dossier inputs de records contient ces prédictions regroupées par tempête.
Le dossier outputs de records comprend, quant à lui, les prédictions par tempête et par identifiant SHOM de marégraphe :
    - 99 : Le Croisic
    - 37 : Saint-Nazaire
    - 62 : les Sables d'Olonne
Pour obtenir ces données, les données du fichier data_prediction_all ont été séparées et soustraites par le niveau moyen de chaque marégraphe via le script python prediction_data_separation.

## Nature de fond
Dans le dossier qgis. Données issues du SHOM mais avec la couche déjà avec les polygones classés par 3 types de substrats (roche, graviers et cailloutis, sable et vase) via le script python bottom_nature.
Second groupe de fichiers qgis pareil mais avec les valeurs de longueur de rugosité suivant la combinaison et la classe. Couche obtenue via la script combinations avec en entrée la couche décrite précédemment.

## MNT
Aussi dans le dossier qgis

-> Pas oublier de donner les sources
