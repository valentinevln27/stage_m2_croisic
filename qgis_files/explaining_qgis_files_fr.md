# Description des fichiers du dossier QGIS

## Matériaux de fonds
Le dossier [bed_materials](https://github.com/valentinevln27/stage_m2_croisic/tree/main/qgis_files/bed_materials) contient les fichiers finaux de la couche de nature de fond du domaine d'étude. Cette couche fournit, pour chaque zone, la classe de nature de fond (roche, graviers et cailloutis, sable et vase) ainsi que la valeur correspondante de la longueur de rugosité z₀, en fonction des différentes combinaisons possibles.


es données concernant la nature de fond se trouvent dans le dossier qgis. Elles proviennent du SHOM et ont été traitées à l’aide de scripts Python :
- <p align="justify">Une première couche contient des polygones classés selon trois types de substrats : roche, graviers et cailloutis, sable et vase. Cette classification a été réalisée avec le script <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/script_python/bottom_nature_v2.py"><code>bottom_nature.py</code></a>. Elle résulte de la fusion de deux couches sources du SHOM : les cartes polygonales au 1/500 000 (Golfe de Gascogne) et au 1/50 000 (littoral métropolitain) du <a href="https://data.shom.fr/donnees#001=eyJjIjpbLTY2MjgwNyw1ODIyOTI3XSwieiI6NiwiciI6MCwibCI6W3sidHlwZSI6IklOVEVSTkFMX0xBWUVSIiwiaWRlbnRpZmllciI6IlNFRElNX01PTkRJQUxFX1BZUl9QTkdfMzg1N19XTVRTIiwib3BhY2l0eSI6MSwidmlzaWJpbGl0eSI6dHJ1ZX0seyJ0eXBlIjoiSU5URVJOQUxfTEFZRVIiLCJpZGVudGlmaWVyIjoiTkRGX1BZUi1QTkdfV0xEXzM4NTdfV01UUyIsIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9LHsidHlwZSI6IklOVEVSTkFMX0xBWUVSIiwiaWRlbnRpZmllciI6IkZEQ19HRUJDT19QWVItUE5HXzM4NTdfV01UUyIsIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9XX0=">SHOM</a>. </p>
- <p align="justify">Une seconde couche, obtenue à partir de la précédente via le script <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/script_python/combinations_v2.py"><code>combinations_v2.py</code></a>, attribue une valeur de longueur de rugosité z₀ à chaque polygone, en fonction de sa classe et des combinaisons de substrats. </p>

<p align="justify">
Lors de cette étude, plusieurs variantes de cette seconde couche ont été créées, chacune associant des valeurs différentes de coefficients de Strickler (ks) selon quatre classes de substrat et différentes combinaisons de coefficients. Ces couches ne sont pas disponibles sur le dépôt GitHub, mais elles peuvent être facilement reproduites en exécutant le script <a href="https://github.com/valentinevln27/stage_m2_croisic/blob/main/script_python/combinations_v2.py"><code>combinations_v2.py</code></a>, après avoir défini les valeurs de ks souhaitées.
</p>

<p align="justify">Les différents jeux de coefficients ks testés dans les simulations sont répertoriés dans le fichier de validation du modèle nommé <code>indicators_ks</code>.</p>



## Frontières
Le dossier [outline_croisic](https://github.com/valentinevln27/stage_m2_croisic/tree/main/qgis_files/outline_croisic) comprend la ligne finale délimitant le domaine sans porte anti-submersion au niveau du chenal de Pen Bron.
Le dossier [outline_gate](https://github.com/valentinevln27/stage_m2_croisic/tree/main/qgis_files/outline_gate) fourni, quant à lui, la délimitation du domaine avec cette porte anti-submersion.
Finalement, le dossier outline_island regroupe les contours des diverses îles inclusent dans le domaine.

## Modèle Numérique de Terrain (MNT)
le MNT Atlantique utilisé — d'une résolution d'environ 100 mètres — provient du SHOM. Il a été découpé puis interpolé à l'aide de la méthode IDW (Inverse Distance Weighting).



## A ajouter
- les iles
- le mnt
- le/les maillages avec la bathy et combi de z0 -> version .csv des mailages
- les autres couches de matériaux de fond
