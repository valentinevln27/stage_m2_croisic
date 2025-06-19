# Méthodologie de génération des fichiers de conditions aux limites et de géométrie pour les simulations hydrodynamiques

Pour réaliser les simulations hydrodynamiques, il est nécessaire de générer à la fois les fichiers de conditions aux limites et de géométrie.
Ce document décrit, étape par étape, la méthode employée dans le cadre de l'étude d'impact de la porte anti-submersion du Croisic, afin d’assurer la compréhension et la reproductibilité du travail effectué.


## Définition du trait de côte et du domaine
La délimitation du trait de côte a été réalisée grâce à une combinaison d’outils QGIS et Python. Les étapes détaillées de cette procédure sont présentées ci-dessous.

### Trait de côte principal
Ce trait de côte principal (Le Croisic et le marais de Guérande) a été construit avec QGIS à partir de données manuelles, selon les étapes suivantes :<br>
**QGIS :**
- Créer une couche de polylignes pour tracer le trait de côte principal.
- Générer une couche de sommets à partir de cette polyligne.
- Ajouter un champ d’index (id), croissant du sud vers le nord, en vérifiant l’ordre de numérotation.
- Créer une nouvelle polyligne connectant les sommets.
- Lisser la polyligne (3 itérations).
- Réaliser un chaînage tous les 10 m (via Excel ou un outil de calcul, selon la longueur totale).
- Ajouter les champs id, x, y (5 décimales pour x et y, entiers pour id).<br>
📌 Remarque : le chaînage désigne ici la génération régulière de points espacés le long d'une ligne.

### Trait de côte élargi pour les grands domaines (incluant les îles)
Pour les domaines larges, deux autres segments de trait de côte (au nord et au sud) ont été ajoutés.<br>
**QGIS :**
- Créer deux nouvelles couches représentant les côtes au nord et au sud du Croisic.
- Lisser chaque polyligne (3 itérations).
- Réaliser un chainage régulier tous les 10 m comme pour le trait de côte principal.

**Python :**
- Importer les couches de points chainés.
- Sélectionner les points de manière à créer un espacement croissant (géométrique) avec un facteur de 1,3, jusqu’à atteindre une distance de 5000 m entre deux points.
- Calculer la distance restante entre le dernier point sélectionné et l'extrémité du trait de côte.
- En déduire un espacement constant pour les derniers points.
- Exporter le tout en un shapefile.<br>
📌 Remarque : le script utilisé a été `stage_m2_cut_points_v2`. La première version du code n'a pas été utilisée pour les modèles et n'est pas disponible comme non utilisée.

**QGIS :**
- Combiner les trois segments de trait de côte (nord, sud, Croisic) pour créer une polyligne complète du domaine élargi.

### Fermeture du domaine
**Python :**
- Importer le trait de côte finalisé.
- Ajouter deux points id=m+1 et id=m+2 à l’ouest du domaine, formant une sorte de rectangle.
- Créer un chaînage depuis la côte vers ces points, avec un espacement croissant (facteur 1,3 jusqu'à 3000 m max pour le plus petit domaine large, sinon 5000 m).
- Compléter le reste du domaine avec des segments réguliers d'environ 3000 ou 5000 m selon le domaine (calcul de la distance restante et division par le nombre de segments à réaliser).
- Les index des points doivent aller du dernier point du trait de côte au nord vers id=m+1 puis de ce point à id=m+2 et enfin de ce dernier point au premier point au sud sur trait de côte.
- Sauvegarder le fichier des points en shapefile.<br>
📌 Remarque : le script utilisé a été `stage_m2_completer_maillage_v3`. Les autres versions ont été utilisées selon les modèles réalisés (la première pour le premier modèle et la seconde pour le premier domaine large contenant les îles).

**QGIS :**
- Importer tous les points et construire une ligne fermée.


## Construction du maillage
Le maillage a été généré à l’aide de BlueKenue, en utilisant le trait de côte fermé précédemment construit.

**BlueKenue :**
- Importer le contour du domaine au format .shp et l’enregistrer en tant que fichier i2s.
- Générer un nouveau maillage en mettant le coutour du domaine dans l'onglet Outline.
    ```
    Réglages dans le générateur 2D :
    - Taille maximale des éléments : 3000 ou 5000 m selon le domaine réalisé
    - Facteur d'agrandissement : 1.3
    ```
- Pour les trois plus petits domaines, ajouter des densités différentes au niveau des marais salants (50 m) et au niveau de la plage de Pen Bron jusqu'un peu au-delà du port au nord (25 m).
- Pour les domaines les plus larges, ajouter des densités différentes au niveau des marais salants (100 m) et au niveau de la plage de Pen Bron jusqu'un peu au-delà du port au nord, intégrant les bouées Candhis du Plateau du Four, de Saint-Nazaire et de l'Ile d'Yeu Nord (25m).


## Génération du fichier de géométrie
### Ajout de la bathymétrie
**QGIS :**
- Pour les domaines les plus larges, importer les modèles topo-bathymétriques préalablement téléchargés du Morbihan, des Pertuis Charentais et de l’aval de la Gironde.
- Pour les trois plus petits domaines, prendre le MNT de l’IGN du marais de Guérande et de ses alentours, puis assembler toutes les parties du MNT pour n’en avoir plus qu’un seul complet.
- Importer le modèle bathymétrique de l'Atlantique et découper la couche selon le domaine.
- Réaliser une interpolation de type IDW pour chaque MNT.

**GESTIONNAIRE DE DOCUMENT :**
- Faire une copie du maillage.

**OUTIL DE TRAITEMENT DE TEXTE :**
- Enlever les premières lignes de la copie réalisée et mettre x y del1 del2 à la place des commentaires supprimés.

**INVITE DE COMMANDE :**
- Convertir le fichier copié au format t3s en format csv. 
- Exemple : `mv mesh_1.3_10m_500m_d25_50.t3s mesh_1.3_10m_500m_d25_50.csv`

**QGIS :**
- Importer le maillage en version csv.
- Extraire la bathymétrie avec l'outil `Point Samplig Tool` aux nœuds du maillage.
- Pour les trois plus petits domaines (non utilisés dans le modèle final), ajouter une condition (shomrge23) pour sélectionner les valeurs de bathymétrie par priorité : Atlantique (mnt_façad) en premier, puis données IGN (rgealt2023) comme suit :
    ```
    case
    when  "mnt_façad" is not null then  "mnt_façad" 
    when  "mnt_façad"  is null and  "rgealt2023"  is not null then  "rgealt2023" 
    else NULL
    end 
    ```
- Pour les plus grands domaines, sélectionner la valeur de bathymétrie selon l’ordre de priorité suivant : Morbihan, Pertuis Charentais, Gironde, façade Atlantique.
- Ajouter une colonne id ($id+1).
- Enregistrer la couche de nœuds du maillage contenant la valeur de bathymétrie .

**PYTHON :**
- Importer le maillage avec les valeurs de bathymétrie.
- Pour toutes les profondeurs supérieures à 5 m, fixer cette valeur comme seuil.<br>
📌 Remarque : le script utilisé a été `rabotage`.

**BLUEKENUE :**
- Importer le maillage contenant l'information de bathymétrie.
- Enregistrer le fichier au format xyz.
- Créer un nouvel interpolateur, puis y ajouter le maillage avec la bathymétrie.
- Ajouter la bathymétrie via l'interpolateur au maillage avec les densités, soit en couche t3s (Tools > MapObject > select interpolator). ⚠️ Attention : bien sélectionner la bathymétrie rabotée avant d’utiliser MapObject.
- Créer un nouveau fichier slf avec ce maillage interpolé, puis ajouter la variable Bottom (unité : M).
- Enregistrer ce fichier.

### Ajout des coefficients de Strickler selon la nature de fond
Après préparation de la couche de nature du fond, et pour avoir des coefficients de Strickler différents selon le substrat, on ajoute les coefficients au fichier de géométrie.

**QGIS :**
- Ajouter les couches de nature du fond du monde et de la France métropolitaine préalablement téléchargées.
- Fusionner les deux couches en donnant priorité à celle de la France métropolitaine.
- Créer un tampon de 3000 à 5000 m (selon le domaine) à partir du tracé du domaine (couche de polylignes convertie en polygones).
- Modifier le tampon si nécessaire.
- Découper la couche de nature du fond à l’aide de cette couche polygone.
- Modifier cette couche pour que les zones de substrat dépassent le trait de côte et s’assurer que tous les points du maillage aient une valeur de nature du fond.

**PYTHON :**
- Importer cette dernière couche.
- Ajouter une colonne `NF` avec les classes suivantes :
Pour les 3 premiers modèles (les plus petits non utilisés dans le modèle final) :
    ```
    si 'typelem' (donnant la nature de fond à un polygone donné) commence par 'NFR', alors la valeur dans 'NF' sera 'NFR'
    si 'typelem' (donnant la nature de fond à un polygone donné) commence par 'NFC' ou 'NFG', alors la valeur dans 'NF' sera 'NFCG'
    si 'typelem' (donnant la nature de fond à un polygone donné) commence par 'NFS', alors la valeur dans 'NF' sera 'NFS'
    si 'typelem' (donnant la nature de fond à un polygone donné) commence par 'NFV', alors la valeur dans 'NF' sera 'NFV'
    ```
Pour les plus grands modèles :
```
    si 'typelem' (donnant la nature de fond à un polygone donné) commence par 'NFR', alors la valeur dans 'NF' sera 'NFR'
    si 'typelem' (donnant la nature de fond à un polygone donné) commence par 'NFC' ou 'NFG', alors la valeur dans 'NF' sera 'NFCG'
    si 'typelem' (donnant la nature de fond à un polygone donné) commence par 'NFS' ou 'NFV', alors la valeur dans 'NF' sera 'NFSV'
```
- Sauvegarder le fichier modifié au format shapefile.<br>
📌 Remarque : le script utilisé a été `stage_m2_nf_v2`. L'autre version a été utilisée pour les 3 plus petits domaines.

**QGIS :**
- Simplifier ce dernier fichier pour les deux plus petits domaines (non utilisés dans le modèle final).

**PYTHON :**
- Importer le fichier final de nature du fond.
- Construire les combinaisons possibles selon les valeurs de nature du fond choisies (les coefficients sont donnés par modèle dans le fichier Excel de résultats).
- Ajouter les combinaisons de Ks en colonnes et les valeurs correspondantes selon la nature du fond.
- Exporter le shapefile modifié.<br>
📌 Remarque : le script utilisé a été `stage_m2_combinaisons_v2`. L'autre version a été utilisée pour les trois plus petits domaines.

**QGIS :**
- Importer la couche de nature du fond avec les combinaisons.
- Sélectionner la couche contenant les points du maillage avec bathymétrie.
- Utiliser l'outil Point Sampling Tool pour extraire les valeurs de Ks correspondant à l'une des combinaisons, et enregistrer ces valeurs pour chaque point du maillage.

**BLUEKENUE :**
Pour chaque couche contenant une combinaison des valeurs du coefficient de Strickler :
- Importer la couche, en enregistrer une copie au format xyz.
- Importer le fichier de géométrie avec la bathymétrie rabotée à 5 m.
- Importer le maillage avec la même bathymétrie rabotée.
- Créer un nouvel interpolateur, glisser la couche xyz dans l’interpolateur, puis sauvegarder.
- Sélectionner le maillage ayant NodeType, puis utiliser Map Object pour ajouter les coefficients en choisissant l’interpolateur correspondant et en précisant l'unité M^(1/3)/s.
- Sélectionner le fichier de géométrie, puis ajouter une variable, cocher Copy node values from source et sélectionner le maillage et la combinaison. Ensuite, choisir Bottom friction et ajouter l'unité du coefficient de Strickler.
- Enregistrer le fichier de géométrie final.


## Génération du fichier de condition limite
La création du fichier de conditions limites a été réalisée uniquement avec BlueKenue.

**BLUEKENUE :**
- Créer le fichier cli en sélectionnant, au nord, le 10ᵉ point et au sud le 7ᵉ (ou les 2ᵉ point s'il s'agit des domaines contenant les îles) en partant des points du trait de côte. Cela permet de s’assurer que les conditions incluent uniquement des points en mer.
- Définir les conditions de frontière ouverte avec H prescrit et traceur libre.
- Enregistrer le fichier cli final.


## Lancement des simulations
Afin de lancer les simulations, quelques étapes sont nécessaires.

**APPLICATION DE TEXTE :**
- Créer un fichier cas définissant les paramètres du modèle.

**Invité de commande :**
- Se rendre dans le dossier où est enregistré le fichier cas.
- Lancer la simulation avec TELEMAC2d : `telemac2d.py nomdufichier.cas`

**BLUEKENUE :**
- Importer le fichier résultat de la simulation pour visualiser les résultats.
- Pour les trois premiers modèles (les plus petits), sélectionner 6 points au niveau du chenal de Pen Bron (près du marégraphe du Croisic) et un au centre du chenal, côté marais de Guérande (soit 7 points au total).
- Pour les plus grands modèles, sélectionner les 3 points les plus proches des marégraphes du Croisic, de Saint-Nazaire et des Sables-d’Olonne (soit 9 points au total).
-  Extraire les séries temporelles des points et les enregistrer.

**PYTHON :**
- Télécharger et importer les données marégraphiques du marégraphe du Croisic pour les trois plus petits modèles.
- Pour les plus grands modèles, récupérer et importer les prévisions marégraphiques du SHOM au Croisic, à Saint-Nazaire et aux Sables-d’Olonne.
- Afficher les données marégraphiques en retirant les périodes avec peu de données (à partir du 22 avril 2023 à 11h25).
- S'il s'agit des observations marégraphiques et non des prédictions, utiliser UTIDE pour calculer la marée prédite.
- Afficher la marée prédite (du SHOM ou obtenue avec UTIDE).
- Importer les données des points sélectionnés sur BlueKenue.
- Calculer la marée prédite pour les points simulés lorsque les paramètres météo-océaniques sont ajoutés.
- Comparer les deux signaux pour valider ou invalider le modèle grâce aux indicateurs RMSE, MAE et PBIAIS (formules données dans le rapport de stage).<br>
📌 Remarque : le script utilisé a été `stage_m2_tide_v5`. Les autres versions ont été utilisées pour ... .