# Explication des données utilisées lors du stage

Les données présentées sont des données de prédictions marégraphiques du Croisic, de Saint-Nazaire et des Sables d'Olonne.

## Métadonnées des marégraphes
Les [métadonnées](https://github.com/valentinevln27/stage_m2_croisic/blob/main/data_stage_m2/gauges_hf_vtd_vh_ZH.txt) des marégraphes du Croisic, de Saint-Nazaire et des Sables-d'Olonne incluent notamment le nom de chaque marégraphe, son identifiant SHOM, ses coordonnées géographiques ainsi que sa valeur de niveau moyen.

## Prédictions par tempête et marégraphes
Dans le dossier [inputs](https://github.com/valentinevln27/stage_m2_croisic/tree/main/data_stage_m2/records/inputs) de records, les prédictions marégraphiques (en m) sont organisées par tempête pour chacun des marégraphes.
Dans le dossier [outputs](https://github.com/valentinevln27/stage_m2_croisic/tree/main/data_stage_m2/records/outputs) de records, les prédictions de hauteur du niveau de la mer ont été ajustées après soustraction du niveau moyen propre à chaque marégraphe via le script python prediction_data_separation. Ces prédictions sont disponibles pour chacun des marégraphes — 99 : Le Croisic, 37 : Saint-Nazaire, 62 : Les Sables-d'Olonne — et pour chaque tempête étudiée : [Lothar et Martin](https://github.com/valentinevln27/stage_m2_croisic/tree/main/data_stage_m2/records/outputs/martin), [Xynthia](https://github.com/valentinevln27/stage_m2_croisic/tree/main/data_stage_m2/records/outputs/xynthia), et [Céline](https://github.com/valentinevln27/stage_m2_croisic/tree/main/data_stage_m2/records/outputs/celine).

## Données météorologiques
Les composantes zonale (u10) et méridienne (v10) du vent (en m/s), ainsi que la pression atmosphérique (en Pa), et leurs incertitudes ne sont pas disponibles sur le dépôt Git en raison de la taille trop importante des fichiers. Cependant, ces données peuvent être facilement récupérées en exécutant le script Python [era5_data_download.py](https://github.com/valentinevln27/stage_m2_croisic/blob/main/script_python/era5_data_download.py), après avoir sélectionné le téléchargement des données ou de leurs incertitudes.
