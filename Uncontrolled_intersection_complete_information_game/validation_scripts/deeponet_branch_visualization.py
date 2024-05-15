import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import copy

#________________________________________________________________________________________

file = 'test_data/deeponet_visualiztion.mat'
data = scipy.io.loadmat(file)
title = '|$b_k$| Tendency with Index'

#____________________________________________________________________________________________________

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

plt.rc('font', **font)

branch_V1 = data['branch_V1']
branch_V2 = data['branch_V2']
N_num = branch_V1.shape[1]

branch_avg = np.zeros((1, N_num)).flatten()
branch_std = np.zeros((1, N_num)).flatten()

for i in range(N_num):
    branch_avg[i] = np.mean(np.abs(branch_V1[:, i]))
    branch_std[i] = np.std(np.abs(branch_V1[:, i]))

branch_avg1 = copy.deepcopy(np.sort(branch_avg)[::-1])
branch_avg2 = copy.deepcopy(branch_avg)
branch_idx = np.argsort(branch_avg2)[::-1]
idx = np.arange(1, 65, step=1)

fig, axs = plt.subplots(1, 1, figsize=(8, 6))

# Configure Plot
axs.set_title(title, fontweight='bold')

plt.errorbar(idx, branch_avg1, linewidth=5.0, yerr=branch_std, fmt="-", ecolor="red", elinewidth=2, capsize=4, capthick=2)

axs.set_xlim(0, 64)
axs.set_xlabel('index', fontweight='bold')
axs.set_ylabel('|$b_k$|', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

plt.show()


