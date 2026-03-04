# %%

import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
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
rc('font', family='serif')
rc('text', usetex=True)  # Correction ici
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
Color_python = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# Parallel
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads



def node_index(Iaxial, Itheta, Iradius,Inetwork,Naxial,Ntheta,Ndr,Nnetwork):
    # Fonction pour calculer l'indicei de la maille située à la coordonnée ix,iy,iz,inetwork.
    i = Iaxial + Naxial*(Itheta % Ntheta) + Iradius * Naxial*Ntheta + Inetwork*Naxial*Ntheta*Ndr
    return i 

def node_index_Iax_Itheta_Idr_Inetwork(i,Naxial,Ntheta,Ndr,Nnetwork):
    N = Naxial*Ntheta*Ndr
    Inetwork = i // N # Division euclidienne par N
    i_moinsnetwork = i - Inetwork*N # Position en enlevant l'effet du network
    Iaxial = (i % Naxial) # Position selon l'axe du cylindre (anciennement X)
    Idr = i_moinsnetwork // (Naxial*Ntheta) # Position dans l'épaisseur 
    Itheta = (i_moinsnetwork - i_moinsnetwork % Naxial - Idr * (Naxial * Ntheta)) // Naxial # Position angulaire
    return Iaxial,Idr,Itheta,Inetwork

radiusExt = 5
radiusInt = 5*1.1
radius = radiusExt
Coef = radiusInt/radiusExt
dr=1.0

Naxial,Ntheta,Ndr,Nnetwork = 10, 60, 1, 2

dZ = 2*np.pi*radius/Ntheta
N = Naxial * Ntheta * Ndr
iz_centerline = 0
X, Y, Z = np.zeros(Nnetwork*N), np.zeros(Nnetwork*N), np.zeros(Nnetwork*N)

for i in range(N*Nnetwork):
    Iaxial,Idr,Itheta,Inetwork = node_index_Iax_Itheta_Idr_Inetwork(i,Naxial,Ntheta,Ndr,Nnetwork)
    if Inetwork==1:
        radius = radiusInt
        X[i] = (Naxial-1)*dZ-Iaxial*dZ 
    else:
        radius = radiusExt
        X[i] = Iaxial*dZ 

    Y[i] = (radius) * np.cos(2 * np.pi * Itheta / Ntheta)
    Z[i] = (radius) * np.sin(2 * np.pi * Itheta / Ntheta)

springs = []

# Séparation des deux groupes d'indices
fixed_indices = []
prescribed_indices_ext = [node_index(ix, iy, iz_centerline,0,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta) for ix in range(Naxial)] # Elastique extérieur
prescribed_indices_int1 = [node_index(ix, iy, iz_centerline,1,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta-20) for ix in range(Naxial)] # Elastique intérieur partie 1
prescribed_indices_int2 = [node_index(ix, iy, iz_centerline,1,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta-15,Ntheta-5) for ix in range(Naxial)] # Elastique intérieur partie 2
#prescribed_indices_int1=[]
prescribed_indices_init = prescribed_indices_ext + prescribed_indices_int1 + prescribed_indices_int2
# Paramètres pour elastique extérieur
R_initial_ext, R_final_ext = 5., 6.0

# Paramètres pour elastique intérieur
R_initial_int1, R_final_int1 = 5.5, 5.5
R_initial_int2, R_final_int2 = 5.5, 0.5

prescribed_indices_fin_int = [node_index(0, iy, iz_centerline,1,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(15,16)] # Elastique intérieur partie 2
prescribed_indices_fin_ext = [node_index(0, iy, iz_centerline,0,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(15,16)] # Elastique intérieur partie 2
prescribed_indices_fin = prescribed_indices_fin_int + prescribed_indices_fin_ext
prescribed_indices_fin=[]
prescribed_indices_X= [node_index(0, iy, iz_centerline,0,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta)]+[node_index(Naxial-1, iy, iz_centerline,1,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta)] # Tous les élastiques leur maille 0 reste en 0
X_prescribed = X[prescribed_indices_X]
X_fin_int= dZ
R_fin_int = 5.2
X_fin_ext= dZ
R_fin_ext = 5.8

# Paramètres
Gamma = 0.5
dt = 0.05
intermediate_steps = 2
relaxation_steps = 1000
initiation_steps = intermediate_steps * relaxation_steps

r_cutoff = dZ*0.9
k_rep = 5.0
B = 1e-3
k1 = 1.0

interaction_steps = 50000  # Nombre d'étapes pour la phase d'interaction
total_steps = initiation_steps + interaction_steps
step2save = np.arange(initiation_steps+1,total_steps,1e4)
step2save[0]=0



# %%


# -----------------------------
# Elastic Forces
# -----------------------------
set_num_threads(max(1, os.cpu_count() - 1))  # Set to number of logical cores or desired thread count
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
    F_thread = np.zeros((get_num_threads(), N, 3), dtype=np.float64)

    for s in prange(springs.shape[0]):
        tid = get_thread_id()
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

    # Reduce thread-local forces
    F = np.zeros((N, 3), dtype=np.float64)
    for tid in range(F_thread.shape[0]):
        F += F_thread[tid]
    return F

# -----------------------------
# Normales
# -----------------------------
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

# -----------------------------
# Contact
# -----------------------------
@njit(parallel=True, fastmath=True)
def compute_repulsive_forces_centerline_numba(r, centerline_indices, r_cutoff, k_rep, adj,N_normalized):
    Np = len(centerline_indices) # Nombre de particules à considérer
    F = np.zeros((r.shape[0], 3))
    
    r_cutoff2 = r_cutoff * r_cutoff

    for a in prange(Np):
        i = centerline_indices[a]
        riX = r[i, 0]
        riY = r[i, 1]
        riZ = r[i, 2]

        for b in range(a + 1, Np):

            j = centerline_indices[b]

            if adj[i, j]:
                continue

            dx = r[j, 0] - riX
            dy = r[j, 1] - riY
            dz = r[j, 2] - riZ

            d2 = dx*dx + dy*dy + dz*dz

            if d2 < r_cutoff2 and d2 > 1e-16:

                d = np.sqrt(d2)

                inv_d = 1.0 / d
                coeff = k_rep * (r_cutoff - d) * inv_d

                fx = coeff * dx
                fy = coeff * dy
                fz = coeff * dz
                
                
                # Normale moyenne
                niX = N_normalized[i, 0]
                niY = N_normalized[i, 1]
                niZ = N_normalized[i, 2]
                njX = N_normalized[j, 0]
                njY = N_normalized[j, 1]
                njZ = N_normalized[j, 2]
                Scal = niX*njX + niY*njY + niZ*njZ # Pour savoir si les vecteurs sont dnas le meme sens
                
                nx = 0.5 * (niX + np.sign(Scal)*njX)
                ny = 0.5 * (niY + np.sign(Scal)*njY)
                nz = 0.5 * (niZ + np.sign(Scal)*njZ)
                
                norm_n = np.sqrt(nx*nx + ny*ny + nz*nz)

                if norm_n > 1e-12:
                    nx /= norm_n
                    ny /= norm_n
                    nz /= norm_n

                    # Projection sur normale
                    dot = fx*nx + fy*ny + fz*nz

                    fx = dot * nx
                    fy = dot * ny
                    fz = dot * nz

                F[i, 0] -= fx
                F[i, 1] -= fy
                F[i, 2] -= fz

                F[j, 0] += fx
                F[j, 1] += fy
                F[j, 2] += fz

    return F


# -----------------------------
# Bending
# -----------------------------
@njit(parallel=True)
def compute_bending_spring_forces(r0, r, bending_springs_array,N_normalized):
    N = r.shape[0]
    F_thread = np.zeros((get_num_threads(), N, 3), dtype=np.float64)
    
    for s in prange(bending_springs_array.shape[0]):
        tid = get_thread_id()
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
            f[:] = -B*(r[im2] - 4*r[im1] + 6*r[i] - 4*r[ip1] + r[ip2])/L0**3 +B*(r0[im2] - 4*r0[im1] + 6*r0[i] - 4*r0[ip1] + r0[ip2])/L0**3
        if edge_type == 1:
            f[:] = -B*(r[i] - 2*r[ip1] + r[ip2])/L0**3 + B*(r0[i] - 2*r0[ip1] + r0[ip2])/L0**3
        if edge_type == 2:
            f[:] = -B*(-2 *r[im1] + 5*r[i] - 4*r[ip1] + r[ip2])/L0**3 +B*(-2 *r0[im1] + 5*r0[i] - 4*r0[ip1] + r0[ip2])/L0**3
        if edge_type == 3:
            f[:] = -B*(-2 *r[ip1] + 5*r[i] - 4*r[im1] + r[im2])/L0**3 + B*(-2 *r0[ip1] + 5*r0[i] - 4*r0[im1] + r0[im2])/L0**3
        if edge_type == 4:
            f[:] = -B*(r[i] - 2*r[im1] + r[im2])/L0**3 + B*(r0[i] - 2*r0[im1] + r0[im2])/L0**3
        
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
def velocity_verlet_step_numba(r0, r, v, dt, step, springs_array,bending_springs_array,centerline_indices,
                               Ncenterline_indices,Scenterline_indices,Ecenterline_indices,Wcenterline_indices,
                               r_cutoff, k_rep, Gamma=0.0, adj=[], fixed_indices=None, prescribed_indices=None, 
                               R_trajectories=[], enable_repulsion=False,prescribed_indices_X=[],X_prescribed=0,p0=0):
    F = compute_spring_forces_numba(r, springs_array)
    # Forces de pression
    N_normalized = compute_normals(r,centerline_indices,Ncenterline_indices,Scenterline_indices,Ecenterline_indices,Wcenterline_indices)
    F[centerline_indices] += p0* N_normalized
    if enable_repulsion:
        F += compute_repulsive_forces_centerline_numba(r, centerline_indices, r_cutoff, k_rep,adj,N_normalized)
    # Forces de flexion
    F_bending = compute_bending_spring_forces(r0, r, bending_springs_array,N_normalized)
    F += F_bending
    F_damped = F - Gamma * v

    v += 0.5 * F_damped * dt
    r += v * dt

    F_new = compute_spring_forces_numba(r, springs_array)
    # Forces de pression
    N_normalized = compute_normals(r,centerline_indices,Ncenterline_indices,Scenterline_indices,Ecenterline_indices,Wcenterline_indices)
    F_new[centerline_indices] += p0* N_normalized
    if enable_repulsion:
        F_new += compute_repulsive_forces_centerline_numba(r, centerline_indices, r_cutoff, k_rep,adj,N_normalized)
    F_bending_new = compute_bending_spring_forces(r0, r, bending_springs_array,N_normalized)
    F_new += F_bending_new
    F_damped_new = F_new - Gamma * v
    v += 0.5 * F_damped_new * dt

    if fixed_indices is not None:
        r[fixed_indices] = r0[fixed_indices]
        v[fixed_indices] = 0

    if prescribed_indices is not None:
        r[prescribed_indices] = R_trajectories[step]
        v[prescribed_indices] = 0
        
    if prescribed_indices_X is not None:
        r[prescribed_indices_X,0] = X_prescribed
        v[prescribed_indices_X,0] = 0

    return r, v

# %%
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Create Elastic Network
# -----------------------------
for inetwork in range(Nnetwork):
    for iz in range(Ndr):
        for iy in range(Ntheta):
            for ix in range(Naxial):
                i = node_index(ix, iy, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == dy == dz == 0:
                                continue
                            
                            # Conditions priodiques en Y (cylindre)
                            jx = ix + dx
                            jy = (iy + dy) % Ntheta
                            jz = iz + dz
                            
                            # Verifier limites X et Z (pas periodiques)
                            if not (0 <= jx < Naxial and 0 <= jz < Ndr):
                                continue
                            
                            j = node_index(jx, jy, jz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                            
                            # Eviter doublons
                            if j <= i:
                                continue
                            
                            xi, yi, zi = X[i], Y[i], Z[i]
                            xj, yj, zj = X[j], Y[j], Z[j]
                            L0 = np.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
                            springs.append((i, j, L0, k1 / L0))
                            
springs_array = np.array(springs, dtype=np.float64)

springs = []
# Création des ressorts de torsion entre nœuds voisins
for inetwork in range(Nnetwork):
    for iz in range(Ndr):
        for iy in range(Ntheta):
            for ix in range(Naxial):
    
                i = node_index(ix, iy, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                
                ip1x = node_index(ix+1, iy, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                ip2x = node_index(ix+2, iy, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                im1x = node_index(ix-1, iy, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                im2x = node_index(ix-2, iy, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                
                ip1y = node_index(ix, iy+1, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                ip2y = node_index(ix, iy+2, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                im1y = node_index(ix, iy-1, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                im2y = node_index(ix, iy-2, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                
                ip1xy = node_index(ix+1, iy+1, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                ip2xy = node_index(ix+2, iy+2, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                im1xy = node_index(ix-1, iy-1, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                im2xy = node_index(ix-2, iy-2, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                
                ip1yx = node_index(ix+1, iy-1, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                ip2yx = node_index(ix+2, iy-2, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                im1yx = node_index(ix-1, iy+1, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                im2yx = node_index(ix-2, iy+2, iz,inetwork,Naxial,Ntheta,Ndr,Nnetwork)
                

                # yi, yj = Y[i], Y[im1y]
                
               
                edge_type_x = 0
                if ix==0:
                    # xi, xj = X[i], X[ip1x]
                    edge_type_x = 1
                # else:
                    # xi, xj = X[i], X[im1x]
                if ix==1:
                    edge_type_x = 2
                if ix==Naxial-2:
                    edge_type_x = 3
                if ix==Naxial-1:
                    edge_type_x = 4
                
                edge_type_y = 0
                # if iy==0:
                #     yi, yj = Y[i], Y[ip1y]
                #     edge_type_y = 1
                # if iy==1:
                #     edge_type_y = 2
                # if iy==Ntheta-2:
                #     edge_type_y = 3
                # if iy==Ntheta-1:
                #     edge_type_y = 4
                    
                    
                edge_type_xy = 0
                # if ((ix == 0 and iy < Ntheta-2) or (ix < Naxial-2 and iy == 0)):
                if (ix == 0):
                    edge_type_xy = 1
                # if ((ix == 1 and 0 < iy < Ntheta-2) or (0 < ix < Naxial-2 and iy == 1)):
                if (ix == 1):
                    edge_type_xy = 2
                if (ix == Naxial-2):
                    edge_type_xy = 3
                if (ix == Naxial-1):
                    edge_type_xy = 4   
                # if ((Naxial-1 > ix > 1 and iy == Ntheta-2) or (ix == Naxial-2 and Ntheta-1 > iy > 1)):
                    # edge_type_xy = 3
                # if ((ix > 1 and iy == Ntheta-1) or (ix == Naxial-1 and iy > 1)):
                    # edge_type_xy = 4   
                # if ((ix <= 1 and iy >= Ntheta-2) or (ix >= Naxial-2 and iy <= 1)):
                    # edge_type_xy = 5
                
                edge_type_yx = 0
                if ((ix == 0)):
                    edge_type_yx = 1
                if ((ix == 1)):
                    edge_type_yx = 2
                if (ix == Naxial-2):
                    edge_type_yx = 3
                if (ix == Naxial-1):
                    edge_type_yx = 4   

                # if ((ix == 0 and iy >= 2) or (ix <= Naxial-3 and iy == Ntheta-1)):
                #     edge_type_yx = 1
                # if ((ix == 1 and 1 < iy < Ntheta-1) or (0 < ix < Naxial-2 and iy == Ntheta-2)):
                #     edge_type_yx = 2
                # if ((1 < ix < Naxial-1 and iy == 1) or (ix == Naxial-2 and Ntheta-2 > iy > 0)):
                #     edge_type_yx = 3
                # if ((1 < ix and iy == 0) or (ix == Naxial-1 and iy < Ntheta-2)):
                #     edge_type_yx = 4   
                # if ((ix <= 1 and iy <= 1) or (ix >= Naxial-2 and iy >= Ntheta-2)):
                #     edge_type_yx = 5
                    
                
                L0x = dZ
                L0y = dZ
                L0xy = dZ*np.sqrt(2)
                
                
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
for inetwork in range(Nnetwork):
    for ix in range(Naxial):
        for iy in range(Ntheta):
            centerline_indices_list.append(node_index(ix, iy, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork))
            if ix+1 < Naxial:
                Ncenterline_indices_list.append(node_index(ix+1, iy, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            else:
                Ncenterline_indices_list.append(node_index(ix, iy, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            if ix > 0:
                Scenterline_indices_list.append(node_index(ix-1, iy, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            else:
                Scenterline_indices_list.append(node_index(ix, iy, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            # if iy > 0:
            Wcenterline_indices_list.append(node_index(ix, iy-1, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            # else:
            # Wcenterline_indices_list.append(node_index(ix, iy, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            # if iy+1 < Ntheta:
            Ecenterline_indices_list.append(node_index(ix, iy+1, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            # else:
            # Ecenterline_indices_list.append(node_index(ix, iy, iz_centerline,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            # Tcenterline_indices_list.append(node_index(ix,iy,2,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
            # Bcenterline_indices_list.append(node_index(ix,iy,0,,inetwork,Naxial,Ntheta,Ndr,Nnetwork) )
centerline_indices=np.array(centerline_indices_list)
Ncenterline_indices=np.array(Ncenterline_indices_list)
Scenterline_indices=np.array(Scenterline_indices_list)
Ecenterline_indices=np.array(Ecenterline_indices_list)
Wcenterline_indices=np.array(Wcenterline_indices_list)
# Tcenterline_indices=np.array(Tcenterline_indices_list)
# Bcenterline_indices=np.array(Bcenterline_indices_list)

        
# -----------------------------
# Simulation Setup
# -----------------------------

r = np.stack([X.copy(), Y.copy(), Z.copy()], axis=1)
v = np.zeros_like(r)
r0 = r.copy()
x0 = X.copy()




# Initialisation
R_trajectories_init = np.zeros((initiation_steps, len(prescribed_indices_init), 3))

# Boucle sur tous les indices prescrits
for i, j in enumerate(prescribed_indices_init):
    I = j // (Naxial * Ntheta)
    J = (j - j % Naxial - I * (Naxial * Ntheta)) // Naxial
    
    # Déterminer quel groupe et quels paramètres utiliser
    if i < len(prescribed_indices_ext):
        # Groupe 1
        R_initial, R_final = R_initial_ext, R_final_ext
    elif i <  len(prescribed_indices_ext)+len(prescribed_indices_int1):
        # Groupe 2
        R_initial, R_final = R_initial_int1, R_final_int1
    else:
        # Groupe 3
        R_initial, R_final = R_initial_int2, R_final_int2
    
    # Génération de la trajectoire
    for n in range(intermediate_steps):
        R_trajectories_init[n * relaxation_steps:(n + 1) * relaxation_steps, i, 0] = x0[j]
        R_trajectories_init[n * relaxation_steps:(n + 1) * relaxation_steps, i, 1] = (R_initial + (R_final - R_initial) * n / (intermediate_steps - 1)) * np.cos(2 * np.pi * J / Ntheta)
        R_trajectories_init[n * relaxation_steps:(n + 1) * relaxation_steps, i, 2] = (R_initial + (R_final - R_initial) * n / (intermediate_steps - 1)) * np.sin(2 * np.pi * J / Ntheta)
    
    
R_trajectories_fin = np.zeros((initiation_steps+interaction_steps, len(prescribed_indices_fin), 3))
# Boucle sur tous les indices prescrits
for i, j in enumerate(prescribed_indices_fin):
    I = j // (Naxial * Ntheta)
    J = (j - j % Naxial - I * (Naxial * Ntheta)) // Naxial
    
    # Déterminer quel groupe et quels paramètres utiliser
    if (i < len(prescribed_indices_fin_int)):
        # Intérieur
        X = X_fin_int
        R = R_fin_int
    else:
        # Extérieur
        X = X_fin_ext
        R = R_fin_ext
    # Génération de la trajectoire
    for n in range(initiation_steps,interaction_steps+initiation_steps,1):
        R_trajectories_fin[n, i, 0] = X 
        R_trajectories_fin[n, i, 1] = R * np.cos(2 * np.pi * J / Ntheta)
        R_trajectories_fin[n, i, 2] = R * np.sin(2 * np.pi * J / Ntheta)

# -----------------------------
# Run Simulation
# -----------------------------

Energy = np.zeros(total_steps)
r_save = []
P_save=[]
p0=0
r_save.append(r0)
print("Running simulation...")
for step in tqdm(range(total_steps)):

    if step in step2save:
        r_save.append(r.copy())
        P_save.append(p0)

    if step < initiation_steps:
        r, v = velocity_verlet_step_numba(r0, r, v, dt, step, springs_array,bending_springs_array,centerline_indices,
                                          Ncenterline_indices,Scenterline_indices,Ecenterline_indices,Wcenterline_indices,
                                          r_cutoff, k_rep, Gamma, adj, fixed_indices, prescribed_indices_init, R_trajectories_init, False,
                                          prescribed_indices_X,X_prescribed,p0)
    else:
        Gamma = 0.2
        r, v = velocity_verlet_step_numba(r0, r, v, dt, step, springs_array,bending_springs_array,centerline_indices,
                                          Ncenterline_indices,Scenterline_indices,Ecenterline_indices,Wcenterline_indices,
                                          r_cutoff, k_rep, Gamma, adj, fixed_indices, prescribed_indices_fin, R_trajectories_fin, True,
                                          prescribed_indices_X,X_prescribed,p0)

    Energy[step] = compute_elastic_energy_numba(r, springs_array)
    
# Gère les sauvegardes       
str_args = "2DRessortsTorsion_Naxial=" + str(Naxial) + "_Ntheta=" +str(Ntheta) + "_Nnetwork=2" + "_B=" + str(B) + "_k=" + str(k1)
path = r"./results"
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

print(f"Fichier sauvegarde dans : {filepath}")
# %%


###################
# -----------------------------
# Plot Results
# -----------------------------

for i_time in range(0,len(step2save),1):
    r_time = r_save[i_time]
    xf, yf, zf = r_time[:, 0], r_time[:, 1], r_time[:, 2]
    

    plt.figure(figsize = (10,4))
    for slice_number in  [0]:
        # slice_number=0
        for inetwork in range(Nnetwork):
            # IDX_PLOT_ext2 = [node_index(slice_number, iy, 2,inetwork,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta+1)]    
            # IDX_PLOT_ext = [node_index(slice_number, iy, 0,inetwork,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta+1)]
            IDX_PLOT_int = [node_index(slice_number, iy, 0,inetwork,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta+1)]
            # IDX_PLOT_ext = [node_index(slice_number, iy, 0,1,Naxial,Ntheta,Ndr,Nnetwork) for iy in range(Ntheta+1)]
            #plt.plot(radius * np.cos(np.linspace(0, 2*np.pi, 300)),radius * np.sin(np.linspace(0, 2*np.pi, 300)))
            #plt.plot(radius * np.cos(np.linspace(0, 2*np.pi, 300)),radius * np.sin(np.linspace(0, 2*np.pi, 300)))
            # plt.plot(yf[IDX_PLOT_ext], zf[IDX_PLOT_ext], '.g-')
            if (inetwork == 1):
                plt.plot(yf[IDX_PLOT_int], zf[IDX_PLOT_int], '.k-')
            else:
                plt.plot(yf[IDX_PLOT_int], zf[IDX_PLOT_int], '.r-')
            # plt.plot(yf[IDX_PLOT_ext2], zf[IDX_PLOT_ext2], '.r-')
            # plt.plot(yf[    IDX_PLOT_intB], zf[IDX_PLOT_intB], '.b-')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(i_time)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.grid(visible=1)
    
    TailleZoom=10
    plt.xlim(-TailleZoom,TailleZoom)
    plt.ylim(-TailleZoom,TailleZoom)


# Plot energy evolution
plt.figure(figsize=(8, 4))
plt.plot(Energy)
plt.xlabel('Time step')
plt.ylabel('Elastic Energy')
plt.title('Energy Evolution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
    
