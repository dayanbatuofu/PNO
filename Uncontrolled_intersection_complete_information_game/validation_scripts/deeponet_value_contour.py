import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import copy
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

#________________________________________________________________________________________

file = 'test_data/deeponet_visualiztion.mat'
data = scipy.io.loadmat(file)

#____________________________________________________________________________________________________

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

plt.rc('font', **font)

branch_V1 = data['branch_V1']
trunk_V1 = data['trunk_V1']
d1 = data['x1']
d2 = data['x2']

N_num = branch_V1.shape[1]

branch_avg = np.zeros((1, N_num)).flatten()

for i in range(N_num):
    branch_avg[i] = np.mean(np.abs(branch_V1[:, i]))

branch_avg1 = copy.deepcopy(branch_avg)
branch_idx = np.argsort(branch_avg1)[::-1]
N_select = 0

trunk_value1 = trunk_V1[:, :, branch_idx[N_select]]

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

# Configure Plot
title = '$t_{' + str(N_select+1) + '}$ Contour for |$b_{' + str(N_select+1) + '}$|'
axs.set_title(title, fontweight='bold')

theta1 = 1
theta2 = 1

start1 = patches.Rectangle((15, 15), 5, 5, linewidth=3, edgecolor='k', facecolor='none')
axs.add_patch(start1)

train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - 1 * 0.75), 3 + theta1 * 0.75 + 0.75,
                           3 + 1 * 0.75 + 0.75, linewidth=3, edgecolor='k', facecolor='none')
axs.add_patch(train1)
train2 = patches.Rectangle((35 - 1 * 0.75, 35 - theta2 * 0.75), 3 + 1 * 0.75 + 0.75,
                            3 + theta2 * 0.75 + 0.75, linewidth=3, edgecolor='k', facecolor='none')
axs.add_patch(train2)
axs.set_xlim(15, 40)
axs.set_xlabel('d1', fontweight='bold')
axs.set_ylim(15, 40)
axs.set_ylabel('d2', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

a = plt.contourf(d1, d2, trunk_value1, 10, cmap=plt.cm.Spectral, fontsize=14)
b = plt.contour(d1, d2, trunk_value1, 10, colors='black', linewidths=2, linestyles='solid')

cbar = fig.colorbar(a, ax=axs)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_weight('bold')

plt.clabel(b, inline=True, fontsize=16, fmt='%1.2f', manual=True)

plt.show()
