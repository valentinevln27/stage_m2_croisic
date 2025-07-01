# Description des données brutes utilisées pendant le stage

## Métadonnées des marégraphes
Les métadonnées sur les marégraphes du Croisic, de Saint-Nazaire et des Sables d'Olonne, comprennent notamment les noms des maregraphes, leur identifiant SHOM, leurs coordonées géographiques et leur valeur de niveau moyen.

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
Aussi dans le dossier qgis. Données du SHOM avec 2 résolutions : 1/500000 et 1/50000. MNT bathy de l'Atlantique, MNT topo-bathy du Morbihan, des pertuis Charentais et de la Gironde aval.
Le fichier mnt_group, regroupe ces mnt.

-> Pas oublier de donner les sources


Prédictions marégraphiques
Le dossier records contient deux sous-dossiers (inputs et outputs) ainsi qu’un fichier Excel nommé data_prediction_all.xlsx.
Ce fichier regroupe les prédictions marégraphiques fournies par le SHOM pour les sites du Croisic, de Saint-Nazaire et des Sables-d’Olonne, lors des tempêtes Lothar-Martin, Xynthia et Céline.
    Le dossier inputs contient ces prédictions organisées par tempête.
    Le dossier outputs regroupe quant à lui les prédictions classées à la fois par tempête et par identifiant SHOM du marégraphe :
        99 : Le Croisic
        37 : Saint-Nazaire
        62 : Les Sables-d’Olonne
Ces fichiers ont été générés à partir du fichier data_prediction_all.xlsx à l’aide du script prediction_data_separation.py, qui effectue un tri par marégraphe et par tempête, puis soustrait le niveau moyen propre à chaque site.

Nature de fond
Les données concernant la nature de fond se trouvent dans le dossier qgis. Elles proviennent du SHOM et ont été traitées à l’aide de scripts Python :
    Une première couche contient des polygones classés selon trois types de substrats : roche, graviers et cailloutis, sable et vase. Cette classification a été réalisée avec le script bottom_nature.py.
    Une seconde couche, obtenue à partir de la précédente via le script combinations.py, attribue une valeur de longueur de rugosité z₀ à chaque polygone, en fonction de sa classe et des combinaisons de substrats.

Modèle Numérique de Terrain (MNT)

Également présent dans le dossier qgis, le MNT utilisé provient du SHOM et se décline en deux résolutions spatiales : 1/500 000 et 1/50 000. Il comprend :
    un MNT bathymétrique de l’Atlantique,
    un MNT topo-bathymétrique pour les zones du Morbihan, des Pertuis Charentais et de la Gironde aval.
L’ensemble de ces fichiers est regroupé dans le fichier mnt_group.