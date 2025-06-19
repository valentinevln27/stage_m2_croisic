# Explication des données utilisées lors du stage

Les données présentées sont des données météo-océanique à certains points (bouées ou marégraphes) compris dans le domaine très large d'étude.


## Données des bouées Candhis
Les données des bouées candhis ont été téléchargées sur le site candhis pendant les campagnes au Plateau du Four et à l'Ile d'Yeu Nord. Cela permet d'avoir des données de houle (Hs, Tp, et seulement pour le Plateau du Four Dp).


## Données ERA5
Téléchargeable sur le site de copernicus, les données era5 ont été téléchargée sur une surface contenant les points des bouées candhis pendant les tempêtes étudiées. Le script python era5_data_download permet de télécharger les données via l'API.


## Données Global Ocean Wave Reanalysis (GOWR)
Comme pour les données era5, les données gowr ont été téléchargées sur une emprise contenant les points des bouées et pendant les tempêtes étudiées.


## Métadonnées des marégraphes
Ces métadonnées sur les marégraphes du croisic, de saint-nazaire et des sables d'olonne, comprennent notamment les noms des maregraphes, leur identifiant SHOM, leurs coordonées et valeur de niveau moyen.


## Résultats des calculs de données météo-océaniques des tempêtes
Ces données, obtenues via les données era5, gowr et candhis ainsi que le script python martin_xynthia_celine_v2, sont les valeurs moyennes de Hs, Dp et Tp pendant les tempêtes étudiées, ainsi que les pressions atmosphériques minimales et leurs composantes u10 et v10 associées à ces pressions minimales.
