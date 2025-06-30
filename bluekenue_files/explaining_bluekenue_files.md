# Description des fichiers du dossier BlueKenue

## Frontières
Dans le dossier [boundaries](https://github.com/valentinevln27/stage_m2_croisic/tree/main/bluekenue_files/boundaries) contient les délimitations du domaine étudié avec et sans porte anti-submersion ainsi que celles des îles qui y sont incluses. Chaque titre de fichier des îles précise la distance entre deux points successifs et le numéro correspondant à l'île comme suit :
- 1 : ile d'Oléron
- 2 : ile de Ré
- 3 : ile d'Yeu
- 4 : ile de Noirmoutier
- 5 : ile d'Hoedic
- 6 : ile d'Houat
- 7 : Belle-Ile-en-Mer
- 8 : Groix

## Conditions aux limites
Le dossier [cli](https://github.com/valentinevln27/stage_m2_croisic/tree/main/bluekenue_files/cli) présente les fichiers définissant les conditions aux limites selon le facteur d'aggrandissement utilisé pour réaliser le maillage.

## Densités
Le dossier [densities](https://github.com/valentinevln27/stage_m2_croisic/tree/main/bluekenue_files/densities) regroupe les densités utilisées pour affiner le maillage à des emplacements spécifiques, comme la zone du Croisic et de la flèche de Pen Bron, les bouées Candhis, ou les marégraphes (du Croisic, de Saint-Nazaire et des Sables d'Olonne).

## Fichiers de géométrie
Le dossier [geoz0r5m](https://github.com/valentinevln27/stage_m2_croisic/tree/main/bluekenue_files/geoz0r5m), selon le facteur d’agrandissement appliqué, comprend les fichiers de géométrie. Ces derniers combinent différentes longueurs de rugosité z₀ en fonction de la nature du fond marin, ainsi qu'une bathymétrie rabotée à 5 mètres.

## Maillage
Le dossier [mesh](https://github.com/valentinevln27/stage_m2_croisic/tree/main/bluekenue_files/mesh) contient les maillages bruts (i.e sans bathymétrie et frottement de fond) générés sans porte anti-submersion, en fonction des facteurs d’agrandissement utilisés. Un maillage avec porte anti-submersion est également disponible pour le facteur 1,35.

## Points de comparaison
Le dossier [points](https://github.com/valentinevln27/stage_m2_croisic/tree/main/bluekenue_files/points) comprend les valeurs de sorties (i.e. l'élévation de la surface libre et la vitesse des courants) aux points utilisés pour comparer les données du modèle avec les prédictions marégraphiques fournies par le SHOM. 

