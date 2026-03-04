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


# -----------------------------
# Create Elastic Network 
# -----------------------------

NX, NY, NZ = 10, 5, 1
N = NX * NY * NZ
iz_centerline=0 # Pour une plaque 2D 0, pour une plaque 3D, 1
X, Y, Z = np.zeros(N), np.zeros(N), np.zeros(N)

k1 = 1
springs = []

def node_index(ix, iy, iz):
    return ix + NX * (iy + NY * iz)
    
spacing = 1
spacing_z = 1

# Génération d'une grille rectangulaire 3D
for i in range(N):
    iz = i // (NX * NY)
    iy = (i % (NX * NY)) // NX
    ix = i % NX
        
    X[i] = ix * spacing
    Y[i] = iy * spacing
    Z[i] = iz * spacing_z
    
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

springs = []
# Création des ressorts de torsion entre nœuds voisins
for iz in range(NZ):
    for iy in range(NY):
        for ix in range(NX):

            i = node_index(ix, iy, iz)
            
            ip1x = node_index(ix+1, iy, iz)
            ip2x = node_index(ix+2, iy, iz)
            im1x = node_index(ix-1, iy, iz)
            im2x = node_index(ix-2, iy, iz)
            
            ip1y = node_index(ix, iy+1, iz)
            ip2y = node_index(ix, iy+2, iz)
            im1y = node_index(ix, iy-1, iz)
            im2y = node_index(ix, iy-2, iz)
            
            ip1xy = node_index(ix+1, iy+1, iz)
            ip2xy = node_index(ix+2, iy+2, iz)
            im1xy = node_index(ix-1, iy-1, iz)
            im2xy = node_index(ix-2, iy-2, iz)
            
            ip1yx = node_index(ix+1, iy-1, iz)
            ip2yx = node_index(ix+2, iy-2, iz)
            im1yx = node_index(ix-1, iy+1, iz)
            im2yx = node_index(ix-2, iy+2, iz)
            
            xi, xj = X[i], X[im1x]
            yi, yj = Y[i], Y[im1y]
            
            B = 0.5
            
            edge_type_x = 0
            if ix==0:
                xi, xj = X[i], X[ip1x]
                edge_type_x = 1
            if ix==1:
                edge_type_x = 2
            if ix==NX-2:
                edge_type_x = 3
            if ix==NX-1:
                edge_type_x = 4
            
            edge_type_y = 0
            if iy==0:
                yi, yj = Y[i], Y[ip1y]
                edge_type_y = 1
            if iy==1:
                edge_type_y = 2
            if iy==NY-2:
                edge_type_y = 3
            if iy==NY-1:
                edge_type_y = 4
                
                
            edge_type_xy = 0
            if ((ix == 0 and iy < NY-2) or (ix < NX-2 and iy == 0)):
                edge_type_xy = 1
            if ((ix == 1 and 0 < iy < NY-2) or (0 < ix < NX-2 and iy == 1)):
                edge_type_xy = 2
            if ((NX-1 > ix > 1 and iy == NY-2) or (ix == NX-2 and NY-1 > iy > 1)):
                edge_type_xy = 3
            if ((ix > 1 and iy == NY-1) or (ix == NX-1 and iy > 1)):
                edge_type_xy = 4   
            if ((ix <= 1 and iy >= NY-2) or (ix >= NX-2 and iy <= 1)):
                edge_type_xy = 5
            
            edge_type_yx = 0
            if ((ix == 0 and iy >= 2) or (ix <= NX-3 and iy == NY-1)):
                edge_type_yx = 1
            if ((ix == 1 and 1 < iy < NY-1) or (0 < ix < NX-2 and iy == NY-2)):
                edge_type_yx = 2
            if ((1 < ix < NX-1 and iy == 1) or (ix == NX-2 and NY-2 > iy > 0)):
                edge_type_yx = 3
            if ((1 < ix and iy == 0) or (ix == NX-1 and iy < NY-2)):
                edge_type_yx = 4   
            if ((ix <= 1 and iy <= 1) or (ix >= NX-2 and iy >= NY-2)):
                edge_type_yx = 5
                
            
            L0x = np.sqrt((xj-xi)**2)
            L0y = np.sqrt((yj-yi)**2)
            L0xy = np.sqrt((xj-xi)**2 + (yj-yi)**2)
            
            
            springs.append((i, ip1x, ip2x, im1x, im2x, L0x, B, edge_type_x))
            springs.append((i, ip1y, ip2y, im1y, im2y, L0y, B, edge_type_y))
            springs.append((i, ip1xy, ip2xy, im1xy, im2xy, L0xy, B, edge_type_xy))
            springs.append((i, ip1yx, ip2yx, im1yx, im2yx, L0xy, B, edge_type_yx))

bending_springs_array = np.array(springs, dtype=np.float64)

# Fait une liste propre des voisins
springs_cut = springs_array[:,0:2]
voisins = [[] for _ in range(2*N)]
for i, j in springs_cut:
    voisins[int(i)].append(int(j))
    voisins[int(j)].append(int(i))

# Fait plutot une matrice, pour Numba
Npart = len(voisins)
adj = np.zeros((Npart, Npart), dtype=np.bool_)
for i in range(Npart):
    for j in voisins[i]:
        adj[i, j] = True    

# Trouve la couche du milieu, du bas, du haut...
centerline_indices_list = []
Ncenterline_indices_list = [] # Maille nord de la maille i
Scenterline_indices_list = [] # Maille sud de la maille i
Ecenterline_indices_list = [] # Maille est de la maille i
Wcenterline_indices_list = [] # Maille ouest de la maille i
Tcenterline_indices_list = [] # Maille r+dr de la maille i
Bcenterline_indices_list = [] # Maille r-dr de la maille i
for network in range(1):
    for ix in range(NX):
        for iy in range(NY):
            centerline_indices_list.append(node_index(ix, iy, iz_centerline)+network*N)
            if ix+1 < NX:
                Ncenterline_indices_list.append(node_index(ix+1, iy, iz_centerline)+network*N)
            else:
                Ncenterline_indices_list.append(node_index(ix, iy, iz_centerline)+network*N)
            if ix > 0:
                Scenterline_indices_list.append(node_index(ix-1, iy, iz_centerline)+network*N)
            else:
                Scenterline_indices_list.append(node_index(ix, iy, iz_centerline)+network*N)
            if iy > 0:
                Wcenterline_indices_list.append(node_index(ix, iy-1, iz_centerline)+network*N)
            else:
                Wcenterline_indices_list.append(node_index(ix, iy, iz_centerline)+network*N)
            if iy+1 < NY:
                Ecenterline_indices_list.append(node_index(ix, iy+1, iz_centerline)+network*N)
            else:
                Ecenterline_indices_list.append(node_index(ix, iy, iz_centerline)+network*N)
            # Tcenterline_indices_list.append(node_index(ix,iy,2)+network*N)
            # Bcenterline_indices_list.append(node_index(ix,iy,0)+network*N)
centerline_indices=np.array(centerline_indices_list)
Ncenterline_indices=np.array(Ncenterline_indices_list)
Scenterline_indices=np.array(Scenterline_indices_list)
Ecenterline_indices=np.array(Ecenterline_indices_list)
Wcenterline_indices=np.array(Wcenterline_indices_list)
# Tcenterline_indices=np.array(Tcenterline_indices_list)
# Bcenterline_indices=np.array(Bcenterline_indices_list)

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



# ------------------------------------------------------
# Construction voisins centerline pour forces pression
# ------------------------------------------------------

max_neighbors = 4  # gauche, bas, droite, haut

neighbors = -np.ones((N, max_neighbors), dtype=np.int64)  # -1 si pas de voisin
n_neighbors = np.zeros(N, dtype=np.int64)

for ix in range(0, NX):
    for iy in range(NY):
        i = node_index(ix, iy, iz_centerline)
        J = []
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        for dx, dy in directions:
            jx, jy, jz = ix + dx, (iy + dy), iz_centerline
            if (0 <= jx < NX) and (0 <= jy < NY):
                j = node_index(jx, jy, jz)
                J.append(j)

        n_neighbors[i] = len(J)
        for k, nj in enumerate(J):
            neighbors[i, k] = nj
            
@njit(inline='always')
def cross(u, v):
    c0 = u[1]*v[2] - u[2]*v[1]
    c1 = u[2]*v[0] - u[0]*v[2]
    c2 = u[0]*v[1] - u[1]*v[0]
    return np.array((c0, c1, c2), dtype=np.float64)


@njit
def compute_normals(r,centerline_indices,Ncenterline_indices,Scenterline_indices,Ecenterline_indices,Wcenterline_indices):
    r_ns = r[Ncenterline_indices]-r[Scenterline_indices]
    r_we = r[Wcenterline_indices]-r[Ecenterline_indices]
    r_normal = np.cross(r_ns,r_we)
    # Calcule la norme
    normes = np.sqrt((r_normal * r_normal).sum(1)).reshape(r_normal.shape[0], 1)
    normes = np.maximum(normes, 1e-12) # Evite les 0
    N_norm_desordre = -r_normal / normes
    N_normalized = np.zeros_like(r)  # n_points x 3
    N_normalized[centerline_indices] = N_norm_desordre
    return N_normalized

@njit(parallel=True)
def compute_normal_numba(r, neighbors, n_neighbors):
    N_normalized = np.zeros_like(r)  # n_points x 3

    for i in range(r.shape[0]):
        k = n_neighbors[i]
        if k == 0:
            continue

        n = np.zeros(3, dtype=np.float64)
        vectors = np.zeros((k, 3), dtype=np.float64)

        for j in range(k):
            nj = neighbors[i, j]
            vectors[j, :] = r[nj] - r[i]

        for j in range(k):
            v1 = vectors[j]
            v2 = vectors[(j+1) % k]
            n += cross(v1, v2)
        n /= k
        norm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        if norm != 0.0:
            n /= norm

        N_normalized[i, :] = n

    return N_normalized
    

@njit(parallel=True)
def compute_bending_spring_forces(r, bending_springs_array,N_normalized):
    N = r.shape[0]
    F_thread = np.zeros((numba.get_num_threads(), N, 3), dtype=np.float64)
    
    for s in prange(bending_springs_array.shape[0]):
        tid = numba.get_thread_id()
        i = int(bending_springs_array[s, 0])
        ip1 = int(bending_springs_array[s, 1])
        ip2 = int(bending_springs_array[s, 2])
        im1 = int(bending_springs_array[s, 3])
        im2 = int(bending_springs_array[s, 4])
        L0, B = bending_springs_array[s, 5], bending_springs_array[s, 6]
        edge_type = bending_springs_array[s, 7]
        
        # tableau fixe taille 3
        f = np.zeros(3, dtype=np.float64)
        if edge_type == 0:
            f[:] = -B*(r[im2] - 4*r[im1] + 6*r[i] - 4*r[ip1] + r[ip2])/L0**3
        if edge_type == 1:
            f[:] = -B*(r[i] - 2*r[ip1] + r[ip2])/L0**3
        if edge_type == 2:
            f[:] = -B*(-2 *r[im1] + 5*r[i] - 4*r[ip1] + r[ip2])/L0**3
        if edge_type == 3:
            f[:] = -B*(-2 *r[ip1] + 5*r[i] - 4*r[im1] + r[im2])/L0**3
        if edge_type == 4:
            f[:] = -B*(r[i] - 2*r[im1] + r[im2])/L0**3
        
        n0 = N_normalized[i,0]
        n1 = N_normalized[i,1]
        n2 = N_normalized[i,2]
        
        dot = f[0]*n0 + f[1]*n1 + f[2]*n2
        
        F_thread[tid, i, 0] += dot * n0
        F_thread[tid, i, 1] += dot * n1
        F_thread[tid, i, 2] += dot * n2
        
    F = np.zeros((N, 3), dtype=np.float64)
    for tid in range(F_thread.shape[0]):
        F += F_thread[tid]
    return F

    
# -----------------------------
# Time Integration
# -----------------------------
r = np.stack([X.copy(), Y.copy(), Z.copy()], axis=1)


fixed_indices = [node_index(0, iy, iz) for iy in range(NY) for iz in range(NZ)] + [node_index(1, iy, iz) for iy in range(NY) for iz in range(NZ)]
prescribed_indices = None
moved_indices = None

def velocity_verlet_step_numba(r, v, dt, step, springs, bending_springs_array,neighbors, n_neighbors, r_cutoff, k_rep, p0, gamma=0.0, fixed_idx=None, prescribed_idx=None, Fext=0, relaxation_steps=1, intermediate_steps=1):
    # Forces élastiques
    F = compute_spring_forces_numba(r, springs)
    
    # Forces de pression
    N_normalized = compute_normals(r,centerline_indices,Ncenterline_indices,Scenterline_indices,Ecenterline_indices,Wcenterline_indices)
    F+= p0* N_normalized
    
    # Forces de flexion
    F_bending = compute_bending_spring_forces(r, bending_springs_array,N_normalized)
    F += F_bending
    
    F_damped = F - gamma * v
    v += 0.5 * F_damped * dt
    r += v * dt

    # Recalcul des forces
    F_new = compute_spring_forces_numba(r, springs)
    # Forces de pression
    N_normalized = compute_normals(r,centerline_indices,Ncenterline_indices,Scenterline_indices,Ecenterline_indices,Wcenterline_indices)
    F_new += p0* N_normalized
    F_bending_new = compute_bending_spring_forces(r, bending_springs_array,N_normalized)
    F_new += F_bending_new

    F_damped_new = F_new - gamma * v
    v += 0.5 * F_damped_new * dt

    if fixed_idx is not None:
        r[fixed_idx] = r0[fixed_idx]
        v[fixed_idx] = 0

    return r, v
# -----------------------------
# Simulation Setup
# -----------------------------

r = np.stack([X.copy(), Y.copy(), Z.copy()], axis=1)
v = np.zeros_like(r)
r0 = r.copy()

p0=0
dP = 0.015e-3
Gamma = 0.2 
dt = 0.2

intermediate_steps = 100
relaxation_steps = 10000
steps = intermediate_steps * relaxation_steps

# -----------------------------
# Run Simulation
# -----------------------------
r_cutoff = 0.5
k_rep = 10.0
Energy = np.zeros(steps)
r_save = []
P_save = []
Fext=0
step2save = np.arange(-1,steps,relaxation_steps)
step2save[0]=0

print("Running simulation...")
for step in tqdm(range(steps)):
    if step in step2save:
        r_save.append(r.copy())
        p0+=dP
        P_save.append(p0)
    r, v = velocity_verlet_step_numba(r, v, dt, step, springs_array, bending_springs_array, neighbors, n_neighbors , r_cutoff, k_rep, p0, Gamma, fixed_indices, prescribed_indices, Fext,relaxation_steps,intermediate_steps)
    Energy[step] = compute_elastic_energy_numba(r, springs_array)

        
str_args = "_NX=" + str(NX) + "_NY=" +str(NY) + "_NZ=" + str(NZ) +"_B=" + str(B) 
path = r"./data_bending"
data_filename = "data"+ str_args+ ".csv"

filepath = os.path.join(path, data_filename)

os.makedirs(path, exist_ok=True)

# Sauvegarde en CSV
with open(filepath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["r_id", "x", "y", "z"])  # en-têtes
    
    for i, r in enumerate(r_save):
        for pos in r:
            writer.writerow([i, pos[0], pos[1], pos[2]])
            

np.savetxt(path + r"/springs_array"+str_args+".csv", springs_array, delimiter=",", comments='')
np.savetxt(path + r"/bending_springs_array"+str_args+".csv", bending_springs_array, delimiter=",", comments='')
np.savetxt(path + r"/P_save"+str_args+".csv", P_save)


print(f"Fichier sauvegardé dans : {filepath}")

# Plot energy evolution
plt.figure(figsize=(8, 4))
plt.plot(Energy)
plt.xlabel('Time step')
plt.ylabel('Elastic Energy')
plt.title('Energy Evolution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


x = r[:, 0]; y = r[:, 1]; z = r[:, 2]
X = r0[:, 0]; Y = r0[:, 1]; Z = r0[:, 2]

slice_y = 2
IDX_PLOT = np.array([node_index(ix, slice_y, 0) for ix in range(NX)])


# -----------------------------
# PLOT COUPE
# -----------------------------
plt.figure(figsize=(6,6))
plt.plot(X[IDX_PLOT], Z[IDX_PLOT], '--', label='initial')
plt.plot(x[IDX_PLOT], z[IDX_PLOT], '-o', label='déformé')
plt.axis('equal')
plt.xlabel('x'); plt.ylabel('z')
plt.title(f'Coupe à y={slice_y}')
plt.legend()



for i_time in range(len(step2save)):
    r_time = r_save[i_time]
    xf, yf, zf = r_time[:, 0], r_time[:, 1], r_time[:, 2]
    
    plt.figure(figsize = (10,4))
    IDX_PLOT_ext2 = [node_index(ix, iy, 0) for ix in range(NX) for iy in [2]]    
    plt.plot(xf[IDX_PLOT_ext2], zf[IDX_PLOT_ext2], '.r-')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(i_time)
    #plt.gca().set_aspect('equal')
    plt.grid(visible=1)
    
    TailleZoomX=1000
    TailleZoomY=5000
    # plt.xlim(0,TailleZoomX)
    # plt.ylim(-TailleZoomY,TailleZoomY)
    