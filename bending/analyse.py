import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

NX, NY, NZ = 10, 3, 3
spacing_z = 1


def node_index(ix, iy, iz):
    return ix + NX * (iy + NY * iz)

def analyse(filepath):

    r_save = []

    NX, NY, NZ = 50, 3, 3

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

    return r_save

path = r"C:\Users\Lab\Documents\M3S_2025\Test 19.02\Bending 19.02\data"

filepath_r = path + rf"\data_NX={NX}_NY={NY}_NZ={NZ}_deltaZ={spacing_z}.csv"
filepath_F = path + r"\F_save.csv"


r_save = analyse(filepath_r)


# PLOTS :
step = 10

r = r_save[step]
r0 = r_save[0]

x = r[:, 0]; y = r[:, 1]; z = r[:, 2]
X = r0[:, 0]; Y = r0[:, 1]; Z = r0[:, 2]

slice_y = 5
IDX_PLOT = np.array([node_index(ix, slice_y, 1) for ix in range(NX)])

plt.figure()
# plt.plot(Y[IDX_PLOT], Z[IDX_PLOT], '--', label='initial')
plt.plot(x[IDX_PLOT], y[IDX_PLOT], '-o',label=f'déformé incrément={step}')
# plt.axis('equal')
plt.xlabel('x'); plt.ylabel('z')
plt.legend()
plt.show()