# Description des données brutes utilisées pendant le stage

## Métadonnées des marégraphes
<p align="justify">Les <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/data_stage_m2/gauges_hf_vtd_vh_ZH.txt">métadonnées</a> sur les marégraphes du Croisic, de Saint-Nazaire et des Sables d'Olonne, comprennent notamment les noms des maregraphes, leur identifiant SHOM, leurs coordonées géographiques et leur valeur de niveau moyen.</p>

## Prédictions marégraphiques
<p align="justify">Le dossier records comprend deux sous-dossiers (inputs et outputs) et un fichier excel nommé <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/data_stage_m2/records/data_prediction_all.xlsx"><code>data_prediction_all.xlsx</code></a>. Ce fichier regroupe les prédictions marégraphiques fournies par le SHOM pour les sites du Croisic, de Saint-Nazaire et des Sables-d’Olonne, lors des tempêtes Lothar-Martin, Xynthia et Céline.</p> 

Le dossier inputs contient ces prédictions marégraphiques organisées par tempête. En colonne, se trouve les valeurs par marégraphe. <br>
Le dossier outputs comprend, quant à lui, les prédictions classées à la fois par tempête et par identifiant SHOM du marégraphe :
- 99 : Le Croisic
- 37 : Saint-Nazaire
- 62 : les Sables d'Olonne
<p align="justify">
Ces fichiers outputs ont été générés à partir du fichier <code>data_prediction_all.xlsx</code> à l’aide du script <code>prediction_data_separation.py</code>, qui effectue un tri par marégraphe et par tempête, puis soustrait le niveau moyen propre à chaque site.
</p>

## Nature de fond
Les données concernant la nature de fond se trouvent dans le dossier qgis. Elles proviennent du SHOM et ont été traitées à l’aide de scripts Python :
- <p align="justify">Une première couche contient des polygones classés selon trois types de substrats : roche, graviers et cailloutis, sable et vase. Cette classification a été réalisée avec le script <code>bottom_nature.py</code>. Elle résulte de la fusion de deux couches sources du SHOM : les cartes polygonales au 1/500 000 (Golfe de Gascogne) et au 1/50 000 (littoral métropolitain). </p>
- <p align="justify">Une seconde couche, obtenue à partir de la précédente via le script <code>combinations_v2.py</code>, attribue une valeur de longueur de rugosité z₀ à chaque polygone, en fonction de sa classe et des combinaisons de substrats. </p>

<p align="justify">
Lors de cette étude, plusieurs variantes de cette seconde couche ont été créées, chacune associant des valeurs différentes de coefficients de Strickler (ks) selon quatre classes de substrat et différentes combinaisons de coefficients. Ces couches ne sont pas disponibles sur le dépôt GitHub, mais elles peuvent être facilement reproduites en exécutant le script <code>combinations_v2.py</code>, après avoir défini les valeurs de ks souhaitées.
</p>

<p align="justify">Les différents jeux de coefficients ks testés dans les simulations sont répertoriés dans le fichier de validation du modèle nommé <code>indicators_ks</code>.</p>

## Modèle Numérique de Terrain (MNT)
Également présent dans le dossier qgis, le MNT utilisé provient du SHOM et se décline en deux résolutions spatiales : 100 et 20 mètres. Il comprend :
- un MNT bathymétrique de l’Atlantique (le seul d'une résolution de 100 mètres),
- trois MNT topo-bathymétriques distincts pour les zones du Morbihan, des Pertuis Charentais et de la Gironde aval.
- <p align="justify">un MNT résultant de la fusion de ces quatre modèles numériques de terrain bathymétriques et topo-bathymétriques. La priorité des valeurs est d’abord donnée au Morbihan, puis aux pertuis charentais, ensuite à la Gironde, pour enfin intégrer les données de l’Atlantique.</p>

--> Il faudra ajouter les liens hypertexte
