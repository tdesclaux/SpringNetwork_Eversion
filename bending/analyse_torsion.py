import numpy as np
import matplotlib.pyplot as plt
import csv
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
rc('font', family='serif')
rc('text', usetex=True)  # Correction ici
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
Color_python = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# Parallel
NX, NY, NZ = 10, 5, 1


def node_index(ix, iy, iz):
    return ix + NX * (iy + NY * iz)

def produce_r_save(filepath):

    r_save = []
    n_step=0

    data = defaultdict(list)
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            r_id = int(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            data[r_id].append([x,y,z])
    for r_id in(data.keys()):
        xyz_current = np.array(data[r_id])
        r_save.append(xyz_current)
        n_step +=1
    return r_save,n_step

path = r"./data_torquespring"
ib = 0
plt.figure()
colB = [[0,0.8,0],[1,0.5,0],[1,0,0]]
for B in [0.05,0.1,0.5]:
    filepath_r = path + r"/data_NX=10_NY=5_NZ=1_B="+str(B)+".csv"
    filepath_F = path + r"/P_save_NX=10_NY=5_NZ=1_B="+str(B)+".csv"
    # dZ=0.5
    # filepath_r = path + r"/data_NX=10_NY=5_NZ=3_dZ="+str(dZ)+".csv"
    # filepath_F = path + r"/P_save_NX=10_NY=5_NZ=3_dZ="+str(dZ)+".csv"
    
    # Lit la force
    P_save = [0.]
    with open(filepath_F, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
    
        for row in reader:
            F = float(row[0])
            P_save.append(F)
    
    r_save,n_step = produce_r_save(filepath_r)
    
    # PLOTS :
    delta=[0.]
    for step in range(15,20):
        r = r_save[step]
        r0 = r_save[0]
        
        x = r[:, 0]; y = r[:, 1]; z = r[:, 2]
        X = r0[:, 0]; Y = r0[:, 1]; Z = r0[:, 2]
        
        
        
    
        slice_y = 2
        IDX_delta = node_index(NX-1,slice_y,0)
        IDX_PLOT = np.array([node_index(ix, slice_y, 0) for ix in range(NX)])
        IDX_PLOT2 = np.array([node_index(ix, slice_y, 1) for ix in range(NX)])
        IDX_PLOT3 = np.array([node_index(ix, slice_y, 2) for ix in range(NX)])
        if (step>0):
            delta.append(z[IDX_delta])
        
        # plt.plot(Y[IDX_PLOT], Z[IDX_PLOT], '--', label='initial')
        plt.plot(x[IDX_PLOT], z[IDX_PLOT]/P_save[step]*B, '-',color=colB[ib],label=f'déformé incrément={step}')
        # plt.axis('equal')
        plt.xlabel('x (m)'); plt.ylabel('$u_z B/P$ (m$^5$)')
        plt.grid(True, alpha=0.3)
    ib = ib+1

plt.show()
