import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from scipy import special
from scipy.integrate import ode
import time
from tqdm import tqdm
import os
import matplotlib.colors as colors

color_plot = ['darkgreen', 'royalblue', 'orangered', 'darkorchid']
from scipy.optimize import fsolve
from numba import njit
from collections import defaultdict

import matplotlib as mpl
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["lines.markeredgecolor"] = "k"
mpl.rcParams["lines.markeredgewidth"] = 1
mpl.rcParams["figure.dpi"] = 250
mpl.rcParams['legend.frameon']= False
mpl.rcParams['legend.fontsize']= 20
mpl.rcParams['axes.labelsize'] = 20

from matplotlib import rc
#rc('font', family='serif')
rc('text', usetex=False)  # Correction ici
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
Color_python = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Parallel
import numpy as np
from numba import njit, prange, set_num_threads
from tqdm import tqdm
import numba

def simulation_spring_newtork(spring_network_type, NX, NY, NZ, fixed_indices, moved_indices, prescribed_indices, condition_type, pull_force, force_direction, Gamma, dt, intermediate_steps, relaxation_steps, path, data_filename):
    
    # -----------------------------
    # Create Elastic Network - PLATE
    # -----------------------------
    spacing = 1.0  # Espacement entre les nœuds
    
    # NX, NY, NZ = 7, 5, 3
    N = NX * NY * NZ
    
    X, Y, Z = np.zeros(N), np.zeros(N), np.zeros(N)
    
    # Génération d'une grille rectangulaire 3D
    for i in range(N):
        iz = i // (NX * NY)
        iy = (i % (NX * NY)) // NX
        ix = i % NX
        
        X[i] = ix * spacing
        Y[i] = iy * spacing
        Z[i] = iz * spacing
    
    k1 = 1
    springs = []
    
    def node_index(ix, iy, iz):
        return ix + NX * (iy + NY * iz)
    
    if spring_network_type == 'plate':
        # Création des ressorts entre nœuds voisins
        for iz in range(NZ):
            for iy in range(NY):
                for ix in range(NX):
                    i = node_index(ix, iy, iz)
        
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
        
                                # ignorer déplacement nul
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue
        
                                jx = ix + dx
                                jy = iy + dy
                                jz = iz + dz
        
                                # éviter hors limites
                                if not (0 <= jx < NX and 0 <= jy < NY and 0 <= jz < NZ):
                                    continue
        
                                j = node_index(jx, jy, jz)
        
                                # éviter les doublons : i < j
                                if j < i:
                                    continue
        
                                # distance au repos
                                xi, yi, zi = X[i], Y[i], Z[i]
                                xj, yj, zj = X[j], Y[j], Z[j]
                                L0 = np.sqrt((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2)
        
                                springs.append((i, j, L0, k1 / L0))
        
        springs_array = np.array(springs, dtype=np.float64)
        
    if spring_network_type=='cylinder':
        for iz in range(NZ):
            for iy in range(NY):
                for ix in range(NX):
                    i = node_index(ix, iy, iz)
                    for dx in [0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [0, 1]:
                                if dx == dy == dz == 0:
                                    continue
                                
                                # Conditions priodiques en X (cylindre)
                                jx = (ix + dx) % NX
                                jy = iy + dy
                                jz = iz + dz
                                
                                # Verifier limites Y et Z (pas periodiques)
                                if not (0 <= jy < NY and 0 <= jz < NZ):
                                    continue
                                
                                j = node_index(jx, jy, jz)
                                
                                # Eviter doublons
                                if j <= i:
                                    continue
                                
                                xi, yi, zi = X[i], Y[i], Z[i]
                                xj, yj, zj = X[j], Y[j], Z[j]
                                L0 = np.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
                                springs.append((i, j, L0, k1 / L0))

        springs_array = np.array(springs, dtype=np.float64)
    # -----------------------------
    # Elastic Forces
    # ----------------------------
    
    set_num_threads(max(1, os.cpu_count() - 1))
    
    @njit(parallel=True)
    def compute_elastic_energy_numba(r, springs):
        E = 0.0
        for s in prange(springs.shape[0]):
            i = int(springs[s, 0])
            j = int(springs[s, 1])
            L0, k = springs[s, 2], springs[s, 3]
            rij = r[j] - r[i]
            L = np.sqrt((rij ** 2).sum())
            dL = L - L0
            E += 0.5 * k * dL * dL
        return E
    
    @njit(parallel=True)
    def compute_spring_forces_numba(r, springs):
        N = r.shape[0]
        F_thread = np.zeros((numba.get_num_threads(), N, 3), dtype=np.float64)
    
        for s in prange(springs.shape[0]):
            tid = numba.get_thread_id()
            i = int(springs[s, 0])
            j = int(springs[s, 1])
            L0, k = springs[s, 2], springs[s, 3]
            rij = r[j] - r[i]
            L = np.sqrt((rij ** 2).sum())
            if L == 0:
                continue
            uij = rij / L
            force = k * (L - L0) * uij
            F_thread[tid, i] += force
            F_thread[tid, j] -= force
    
        F = np.zeros((N, 3), dtype=np.float64)
        for tid in range(F_thread.shape[0]):
            F += F_thread[tid]
        return F
    
    # -----------------------------
    # Repulsion with Cell List
    # -----------------------------
    
    # Plan médian (iz = 1)
    iz = int(NZ/2)
    centerline_indices = np.array([node_index(ix, iy, iz) for ix in range(NX) for iy in range(NY)], dtype=np.int64)
    
    @njit(parallel=True)
    def compute_repulsive_forces_centerline_numba(r, centerline_indices, r_cutoff, k_rep):
        N = centerline_indices.shape[0]
        F_total = np.zeros_like(r)
        epsilon = 1e-6
        r_cell = 0.3
    
        # Bounding box
        x_min = y_min = z_min = 1e10
        x_max = y_max = z_max = -1e10
        for idx in centerline_indices:
            xi, yi, zi = r[idx]
            if xi < x_min: x_min = xi
            if yi < y_min: y_min = yi
            if zi < z_min: z_min = zi
            if xi > x_max: x_max = xi
            if yi > y_max: y_max = yi
            if zi > z_max: z_max = zi
    
        x_min -= epsilon
        y_min -= epsilon
        z_min -= epsilon
        x_max += epsilon
        y_max += epsilon
        z_max += epsilon
    
        Lx, Ly, Lz = x_max - x_min, y_max - y_min, z_max - z_min
        n_cells_x = max(1, int(Lx / r_cell))
        n_cells_y = max(1, int(Ly / r_cell))
        n_cells_z = max(1, int(Lz / r_cell))
    
        head = -1 * np.ones((n_cells_x, n_cells_y, n_cells_z), dtype=np.int32)
        next_particle = -1 * np.ones(r.shape[0], dtype=np.int32)
    
        for idx in centerline_indices:
            cx = int((r[idx, 0] - x_min) / r_cell)
            cy = int((r[idx, 1] - y_min) / r_cell)
            cz = int((r[idx, 2] - z_min) / r_cell)
            if cx >= n_cells_x: cx = n_cells_x - 1
            if cy >= n_cells_y: cy = n_cells_y - 1
            if cz >= n_cells_z: cz = n_cells_z - 1
            next_particle[idx] = head[cx, cy, cz]
            head[cx, cy, cz] = idx
    
        thread_forces = np.zeros((numba.get_num_threads(), r.shape[0], 3), dtype=np.float64)
    
        for cx in prange(n_cells_x):
            for cy in range(n_cells_y):
                for cz in range(n_cells_z):
                    i = head[cx, cy, cz]
                    while i != -1:
                        for dx in [-1, 0, 1]:
                            nx = cx + dx
                            if nx < 0 or nx >= n_cells_x: continue
                            for dy in [-1, 0, 1]:
                                ny = cy + dy
                                if ny < 0 or ny >= n_cells_y: continue
                                for dz in [-1, 0, 1]:
                                    nz = cz + dz
                                    if nz < 0 or nz >= n_cells_z: continue
                                    j = head[nx, ny, nz]
                                    while j != -1:
                                        if j > i:
                                            rij = r[j] - r[i]
                                            d = np.sqrt((rij ** 2).sum())
                                            if d < r_cutoff and d > 1e-8:
                                                rep_force = k_rep * (r_cutoff - d) / d * rij
                                                tid = numba.get_thread_id()
                                                thread_forces[tid, i] -= rep_force
                                                thread_forces[tid, j] += rep_force
                                        j = next_particle[j]
                        i = next_particle[i]
    
        for tid in range(thread_forces.shape[0]):
            F_total += thread_forces[tid]
    
        return F_total
    
    # -----------------------------
    # Time Integration
    # -----------------------------
    r = np.stack([X.copy(), Y.copy(), Z.copy()], axis=1)
    
    
    #fixed_indices = [node_index(0, iy, 0) for iy in range(NY)]
    # prescribed_indices = [node_index(NX - 1, iy, 1) for iy in range(NY)]
    #moved_indices = [node_index(NX - 1, iy, iz) for iy in range(NY) for iz in range(NZ)]
    
    #pull_force = 0.02
    Fext = np.zeros_like(r)
    for i in moved_indices:
        if force_direction == 'x':
            Fext[i, 0] = pull_force  # force in x-direction
        elif force_direction == 'y':
            Fext[i, 1] = pull_force  # force in x-direction
        elif force_direction == 'z':
            Fext[i, 2] = pull_force  # force in x-direction
    
    
    def velocity_verlet_step_numba(r, v, dt, t, springs, r_cutoff, k_rep, gamma=0.0, fixed_idx=None, prescribed_idx=None, Fext=0):
        F = compute_spring_forces_numba(r, springs)
        if condition_type == 'force':
            F+= Fext    
        #F += compute_repulsive_forces_centerline_numba(r, centerline_indices, r_cutoff, k_rep)
        F_damped = F - gamma * v
    
        v += 0.5 * F_damped * dt
        r += v * dt
    
        F_new = compute_spring_forces_numba(r, springs)
        if condition_type == 'force':
            F_new+=Fext
        #F_new += compute_repulsive_forces_centerline_numba(r, centerline_indices, r_cutoff, k_rep)
        F_damped_new = F_new - gamma * v
        v += 0.5 * F_damped_new * dt
    
        if fixed_idx is not None:
            r[fixed_idx] = r0[fixed_idx]
            v[fixed_idx] = 0
    
        # if prescribed_idx is not None:
        #     r[prescribed_idx] = R_trajectories[t]
        #     v[prescribed_idx] = 0
    
        return r, v
    
    # -----------------------------
    # Simulation Setup
    # -----------------------------
    
    r = np.stack([X.copy(), Y.copy(), Z.copy()], axis=1)
    v = np.zeros_like(r)
    r0 = r.copy()
    
    # Bord fixe à ix = 0
    #fixed_indices = [node_index(0, iy, iz) for iy in range(NY) for iz in range(NZ)]
    # Bord prescrit à ix = NX-1, plan médian iz = 1
    #iz = int(NZ/2)
    #prescribed_indices = [node_index(NX - 1, iy, iz) for iy in range(NY)]
    
    #Gamma = 0.2
    #dt = 0.2
    
    # intermediate_steps = 10
    # relaxation_steps = 1000
    steps = intermediate_steps * relaxation_steps
    
    # # Trajectoires pour le bord prescrit
    # X_initial, X_final = (NX-1) * spacing, (NX-1) * spacing + 1
    # Y_initial, Y_final = 0.0, 0.0  # Pas de déplacement en Y
    # Z_initial, Z_final = spacing, spacing  # Reste au plan médian
    
    # R_trajectories = np.zeros((steps, NY, 3))
    # for i in range(NY):
    #     j = prescribed_indices[i]
    #     iz = j // (NX * NY)
    #     iy = (j % (NX * NY)) // NX
        
    #     for n in range(intermediate_steps):
    #         frac = n / (intermediate_steps - 1)
    #         R_trajectories[n * relaxation_steps:(n + 1) * relaxation_steps, i, 0] = X_initial + (X_final - X_initial) * frac
    #         R_trajectories[n * relaxation_steps:(n + 1) * relaxation_steps, i, 1] = iy * spacing + Y_initial + (Y_final - Y_initial) * frac
    #         R_trajectories[n * relaxation_steps:(n + 1) * relaxation_steps, i, 2] = Z_initial + (Z_final - Z_initial) * frac
            
    # -----------------------------
    # Run Simulation
    # -----------------------------
    
    r_cutoff = 0.5
    k_rep = 10.0
    Energy = np.zeros(steps)
    r_save = []
    step2save=[0, 999, 1999, 2999, 3999, 4999, 5999, 6999, 7999, 8999, 9999, 10999, 11999, 12999, 13999, 14999, 15999, 16999, 17999, 18999, 19999]
    
    
    print("Running simulation...")
    for step in tqdm(range(steps)):
        r, v = velocity_verlet_step_numba(r, v, dt, step, springs_array, r_cutoff, k_rep, Gamma, fixed_indices, prescribed_indices, Fext)
        Energy[step] = compute_elastic_energy_numba(r, springs_array)
        if step in step2save:
            r_save.append(r.copy())
            
    #path = r"C:\Users\DELL\Documents\M3S\Projet M3S\data"   # chemin du dossier
    #data_filename = "data_01.csv"                             # nom du fichier
    filepath = os.path.join(path, data_filename)
    
    os.makedirs(path, exist_ok=True)
    
    # Sauvegarde en CSV
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_id", "x", "y", "z"])  # en-têtes
        
        for i, r in enumerate(r_save):
            for pos in r:
                writer.writerow([i, pos[0], pos[1], pos[2]])
    
    np.savetxt(path + r"\springs_array.csv", springs_array, delimiter=",", comments='')
    
    print(f"Fichier sauvegardé dans : {filepath}")
    
    # -----------------------------
    # Plot Results
    # -----------------------------
    
    # Plot energy evolution
    plt.figure(figsize=(8, 4))
    plt.plot(-np.diff(Energy))
    plt.xlabel('Time step')
    plt.ylabel('Elastic Energy')
    plt.title('Energy Evolution')
    plt.grid(True, alpha=0.3)
    plt.loglog()
    plt.tight_layout()
    
    plt.show()
    
    
    # Plot energy evolution
    plt.figure(figsize=(8, 4))
    plt.plot(Energy)
    plt.xlabel('Time step')
    plt.ylabel('Elastic Energy')
    plt.title('Energy Evolution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
