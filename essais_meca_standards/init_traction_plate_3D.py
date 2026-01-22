from coeur_reseau_ressort import simulation_spring_newtork
import numpy as np



spring_network_type = 'plate' #plate #tube
NX, NY, NZ = 10, 7, 3

#########################################################

N = NX * NY * NZ
spacing = 1
    
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

##########################################################
    
    
    

condition_type = 'force' #displacement #force
pull_force = 0

if condition_type == 'force':
    pull_force = 0.002
    force_direction = 'x' # x, y, z 
    fixed_indices = [node_index(0, iy, iz) for iy in range(NY) for iz in range(NZ)]
    prescribed_indices = []
    moved_indices = [node_index(NX - 1, iy, iz) for iy in range(NY) for iz in range(NZ)]
    
if condition_type == 'displacement':
    fixed_indices = [node_index(0, iy, iz) for iy in range(NY) for iz in range(NZ)]
    prescribed_indices = [node_index(NX - 1, iy, iz) for iy in range(NY)]


Gamma = 0.2
dt = 0.2
intermediate_steps = 10
relaxation_steps = 1000

path = r"C:\Users\DELL\Documents\M3S\Projet M3S\mettre_sur_github\data"
data_filename = "data_01.csv" 
simulation_spring_newtork(spring_network_type, NX, NY, NZ, fixed_indices, moved_indices, prescribed_indices, condition_type, pull_force, force_direction, Gamma, dt, intermediate_steps, relaxation_steps, path, data_filename)













