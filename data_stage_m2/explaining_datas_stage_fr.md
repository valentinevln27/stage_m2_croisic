# Description des données brutes utilisées pendant le stage

## Métadonnées des marégraphes
Les métadonnées sur les marégraphes du Croisic, de Saint-Nazaire et des Sables d'Olonne, comprennent notamment les noms des maregraphes, leur identifiant SHOM, leurs coordonées géographiques et leur valeur de niveau moyen.

## Prédictions marégraphiques
<p style="text-align: justify">Le dossier records comprend deux sous-dossiers (inputs et outputs) et un fichier excel nommé `data_prediction_all.xlsx`. Ce fichier regroupe les prédictions marégraphiques fournies par le SHOM pour les sites du Croisic, de Saint-Nazaire et des Sables-d’Olonne, lors des tempêtes Lothar-Martin, Xynthia et Céline.
Le dossier inputs contient ces prédictions organisées par tempête.</p>

Le dossier outputs comprend, quant à lui, les prédictions classées à la fois par tempête et par identifiant SHOM du marégraphe :
- 99 : Le Croisic
- 37 : Saint-Nazaire
- 62 : les Sables d'Olonne
Ces fichiers outputs ont été générés à partir du fichier `data_prediction_all.xlsx` à l’aide du script `prediction_data_separation.py`, qui effectue un tri par marégraphe et par tempête, puis soustrait le niveau moyen propre à chaque site.

## Nature de fond
Les données concernant la nature de fond se trouvent dans le dossier qgis. Elles proviennent du SHOM et ont été traitées à l’aide de scripts Python :
- Une première couche contient des polygones classés selon trois types de substrats : roche, graviers et cailloutis, sable et vase. Cette classification a été réalisée avec le script bottom_nature.py. Elle résulte de la fusion de deux couches sources du SHOM : les cartes polygonales au 1/500 000 (Golfe de Gascogne) et au 1/50 000 (littoral métropolitain).
- Une seconde couche, obtenue à partir de la précédente via le script `combinations_v2.py`, attribue une valeur de longueur de rugosité z₀ à chaque polygone, en fonction de sa classe et des combinaisons de substrats. 

<p style="text-align: justify; text_indent:25px;">Lors de cette étude, , plusieurs variantes de cette seconde couche ont été créées, chacune associant des valeurs différentes de coefficients de Strickler (ks) selon quatre classes de substrat et différentes combinaisons de coefficients. Ces couches ne sont pas disponibles sur le dépôt GitHub, mais elles peuvent être facilement reproduites en exécutant le script `combinations_v2.py`, après avoir défini les valeurs de ks souhaitées.</p>

Les différents jeux de coefficients ks testés dans les simulations sont répertoriés dans le fichier de validation du modèle nommé `indicators_ks`.

## Modèle Numérique de Terrain (MNT)
Également présent dans le dossier qgis, le MNT utilisé provient du SHOM et se décline en deux résolutions spatiales : 100 et 20 mètres. Il comprend :
- un MNT bathymétrique de l’Atlantique (le seul d'une résolution de 100 mètres),
- trois MNT topo-bathymétriques distincts pour les zones du Morbihan, des Pertuis Charentais et de la Gironde aval.
- un MNT correspondant à la fusion de ces quatres MNT bathymétrique et topobathymétrique. Les valeurs prioritaires sont données d'abord au Morbihan, puis les pertuis charentais puis la gironde pour finir avec l'Atlantique
