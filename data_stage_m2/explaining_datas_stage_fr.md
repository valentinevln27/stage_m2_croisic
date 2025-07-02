# Description des données brutes utilisées pendant le stage

## Métadonnées des marégraphes
<p align="justify">Les <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/data_stage_m2/gauges_hf_vtd_vh_ZH.txt">métadonnées</a> sur les marégraphes du Croisic, de Saint-Nazaire et des Sables d'Olonne, comprennent notamment les noms des maregraphes, leur identifiant SHOM, leurs coordonées géographiques et leur valeur de niveau moyen.</p>

## Prédictions marégraphiques
<p align="justify">Le dossier records comprend deux sous-dossiers (inputs et outputs) et un fichier excel nommé <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/data_stage_m2/records/data_prediction_all.xlsx"><code>data_prediction_all.xlsx</code></a>. Ce fichier regroupe les prédictions marégraphiques fournies par le <a href="https://maree.shom.fr/">SHOM</a> pour les sites du Croisic, de Saint-Nazaire et des Sables-d’Olonne, lors des tempêtes Lothar-Martin, Xynthia et Céline (avec une fenêtre de 3 jours avant et après les tempêtes).</p> 

Le dossier <a href="https://github.com/valentinevln27/stage_m2_croisic/tree/main/data_stage_m2/records/inputs">inputs</a> contient ces prédictions marégraphiques organisées par tempête. En colonne, se trouve les valeurs par marégraphe. <br>
Le dossier <a href="https://github.com/valentinevln27/stage_m2_croisic/tree/main/data_stage_m2/records/outputs">outputs</a> comprend, quant à lui, les prédictions classées à la fois par tempête et par identifiant SHOM du marégraphe :
- 99 : Le Croisic
- 37 : Saint-Nazaire
- 62 : les Sables d'Olonne
<p align="justify">
Ces fichiers outputs ont été générés à partir du fichier <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/data_stage_m2/records/data_prediction_all.xlsx"><code>data_prediction_all.xlsx</code></a> à l’aide du script <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/script_python/prediction_data_separation.py"><code>prediction_data_separation.py</code></a>, qui effectue un tri par marégraphe et par tempête, puis soustrait le niveau moyen propre à chaque site.
</p>

## Nature de fond (faut ajouter les couches donc pas encore les liens des couches, idem pour fichier excel de résultats)
Les données concernant la nature de fond se trouvent dans le dossier qgis. Elles proviennent du SHOM et ont été traitées à l’aide de scripts Python :
- <p align="justify">Une première couche contient des polygones classés selon trois types de substrats : roche, graviers et cailloutis, sable et vase. Cette classification a été réalisée avec le script <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/script_python/bottom_nature_v2.py"><code>bottom_nature.py</code></a>. Elle résulte de la fusion de deux couches sources du SHOM : les cartes polygonales au 1/500 000 (Golfe de Gascogne) et au 1/50 000 (littoral métropolitain) du <a href="https://data.shom.fr/donnees#001=eyJjIjpbLTY2MjgwNyw1ODIyOTI3XSwieiI6NiwiciI6MCwibCI6W3sidHlwZSI6IklOVEVSTkFMX0xBWUVSIiwiaWRlbnRpZmllciI6IlNFRElNX01PTkRJQUxFX1BZUl9QTkdfMzg1N19XTVRTIiwib3BhY2l0eSI6MSwidmlzaWJpbGl0eSI6dHJ1ZX0seyJ0eXBlIjoiSU5URVJOQUxfTEFZRVIiLCJpZGVudGlmaWVyIjoiTkRGX1BZUi1QTkdfV0xEXzM4NTdfV01UUyIsIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9LHsidHlwZSI6IklOVEVSTkFMX0xBWUVSIiwiaWRlbnRpZmllciI6IkZEQ19HRUJDT19QWVItUE5HXzM4NTdfV01UUyIsIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9XX0=">SHOM</a>. </p>
- <p align="justify">Une seconde couche, obtenue à partir de la précédente via le script <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/script_python/combinations_v2.py"><code>combinations_v2.py</code></a>, attribue une valeur de longueur de rugosité z₀ à chaque polygone, en fonction de sa classe et des combinaisons de substrats. </p>

<p align="justify">
Lors de cette étude, plusieurs variantes de cette seconde couche ont été créées, chacune associant des valeurs différentes de coefficients de Strickler (ks) selon quatre classes de substrat et différentes combinaisons de coefficients. Ces couches ne sont pas disponibles sur le dépôt GitHub, mais elles peuvent être facilement reproduites en exécutant le script <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/script_python/combinations_v2.py"><code>combinations_v2.py</code></a>, après avoir défini les valeurs de ks souhaitées.
</p>

<p align="justify">Les différents jeux de coefficients ks testés dans les simulations sont répertoriés dans le fichier de validation du modèle nommé <code>indicators_ks</code>.</p>

## Modèle Numérique de Terrain (MNT) (les mnt pas mis car passe pas sur le git)
<p align="justify">Également présent dans le dossier qgis, le MNT Atlantique utilisé — d'une résolution d'environ 100 mètres — provient du SHOM. Il a été découpé puis interpolé à l'aide de la méthode IDW (Inverse Distance Weighting).</p>

## Récapitulatif des données et traitement réalisés pour chaque modèle
<p align="justify">Afin de faciliter la compréhension, un tableau a été réalisé pour récapituler, pour chaque modèle utilisé, les données mobilisées, les traitements effectués ainsi que les options choisies (période de simulation, méthode de modélisation du frottement, etc.).</p> 
<p align="justify">Il est à noter que l’ensemble des fichiers associés aux différents modèles (données brutes, traitées, générées et résultats) ne sont pas tous fournis. Seuls ceux correspondant au dernier domaine simulé sont disponibles. Si l’espace de stockage disponible sur GitLab le permet, un fichier .zip contenant l’ensemble des fichiers, accompagné d’un tableau descriptif, sera ajouté. Dans le cas contraire, je pourrai fournir ces éléments sur demande via un lien FileSender.</p>

