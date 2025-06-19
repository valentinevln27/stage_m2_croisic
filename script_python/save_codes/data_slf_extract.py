# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:12:48 2025

@author: vanleene valentine
"""

#%%
"""
git clone https://github.com/ppKrauss/pputils.git
cd pputils
pip install .
"""

#%%
from ppSELAFIN import SELAFIN
import numpy as np
from scipy.spatial import cKDTree

# Charger le fichier .slf
slf = SELAFIN('ton_fichier.slf')
slf.readHeader()  # charge les infos de grille, variables, etc.

# Coordonnées des nœuds
x = slf.meshx
y = slf.meshy

# Créer un arbre pour rechercher le nœud le plus proche
tree = cKDTree(np.column_stack((x, y)))

# Coordonnée(s) cible(s) - à adapter
points = [(100.0, 200.0), (120.0, 210.0)]
indices = [tree.query(pt)[1] for pt in points]

# Liste des variables disponibles
print("Variables disponibles :", slf.varnames)

# Choisir la variable (ex: 'WATER DEPTH') et son index
varname = 'WATER DEPTH'
var_index = slf.varnames.index(varname)

# Lire les données pour chaque pas de temps
values_over_time = []

for t in range(slf.nbtimes):
    slf.readTimeStep(t)
    data = slf.values[var_index]
    extracted = [data[i] for i in indices]
    values_over_time.append(extracted)

# Résultat : liste [ [val1_pt1, val2_pt2], ... ]
print(values_over_time)