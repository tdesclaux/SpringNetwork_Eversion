from coeur_reseau_ressort import simulation_spring_newtork

spring_network_type = 'tube' #plate #tube
NX, NY, NZ = 10, 7, 3
    
##########################################################
    
def node_index(ix, iy, iz):
    return ix + NX * (iy + NY * iz)

##########################################################

condition_type = 'displacement' #displacement #force
pull_force = 0
force_direction = ''

if condition_type == 'force':
    pull_force = 0.002
    force_direction = 'x' # x, y, z 
    fixed_indices = [node_index(0, iy, iz) for iy in range(NY) for iz in range(NZ)]
    prescribed_indices = []
    moved_indices = [node_index(NX - 1, iy, iz) for iy in range(NY) for iz in range(NZ)]
    
if condition_type == 'displacement':
    fixed_indices = [node_index(0, iy, iz) for iy in range(NY) for iz in range(NZ)]
    prescribed_indices = [node_index(NX - 1, iy, 1) for iy in range(NY)]
    moved_indices = []


Gamma = 0.2
dt = 0.2
intermediate_steps = 10
relaxation_steps = 1000

path = r"C:\Users\DELL\Documents\M3S\SpringNetwork_Eversion\essais_meca_standards\data"
data_filename = "data_01.csv" 
simulation_spring_newtork(spring_network_type, NX, NY, NZ, fixed_indices, moved_indices, prescribed_indices, condition_type, pull_force, force_direction, Gamma, dt, intermediate_steps, relaxation_steps, path, data_filename)













