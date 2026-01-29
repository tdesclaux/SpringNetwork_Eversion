#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:09:34 2026

@author: terence
"""

import pandas as pd
import numpy as np
from itertools import combinations

# =====================
# 1. Chargement des données
# =====================
pos = pd.read_csv("data/data_01.csv")
spr = pd.read_csv("data/springs_array.csv",header=None)

# Positions au temps initial
pos0 = pos[pos["r_id"] == 0].reset_index(drop=True)

# Dictionnaire index -> position
positions = {
    i: np.array([row.x, row.y, row.z])
    for i, row in pos0.iterrows()
}

# Connectivité (colonnes 0 et 1)
springs = [(int(row.iloc[0]), int(row.iloc[1])) for _, row in spr.iterrows()]

# =====================
# 2. Vérification unicité des ressorts
# =====================
normalized = [(min(i, j), max(i, j)) for i, j in springs]
unique_springs = set(normalized)

if len(unique_springs) != len(springs):
    print("❌ Ressorts dupliqués détectés :", len(springs) - len(unique_springs))
else:
    print("✅ Tous les ressorts sont uniques")

# =====================
# 3. Détermination des longueurs caractéristiques
# =====================
lengths = np.array([
    np.linalg.norm(positions[i] - positions[j])
    for i, j in unique_springs
])

# Longueur de maille (plus petite distance non nulle)
L = np.min(lengths[lengths > 1e-8])

# Tolérance géométrique
tol = 1e-3

def classify_length(d):
    if abs(d - L) < tol:
        return "direct"
    elif abs(d - np.sqrt(2)*L) < tol:
        return "diag_face"
    elif abs(d - np.sqrt(3)*L) < tol:
        return "diag_volume"
    else:
        return "other"

spring_types = {
    (i, j): classify_length(np.linalg.norm(positions[i] - positions[j]))
    for i, j in unique_springs
}

# =====================
# 4. Vérification des voisins attendus
# =====================
errors = []

# Recherche des voisins géométriques attendus
for i, pi in positions.items():
    for j, pj in positions.items():
        if j <= i:
            continue

        d = np.linalg.norm(pi - pj)
        t = classify_length(d)

        if t in ("direct", "diag_face", "diag_volume"):
            key = (i, j)
            if key not in unique_springs:
                errors.append((i, j, t))

# =====================
# 5. Rapport final
# =====================
if not errors:
    print("✅ Tous les ressorts (directs + diagonaux) sont correctement connectés")
else:
    print(f"❌ {len(errors)} ressorts manquants détectés")
    for e in errors[:20]:
        print(f"   Ressort manquant entre {e[0]} et {e[1]} (type: {e[2]})")
    if len(errors) > 20:
        print("   ...")

