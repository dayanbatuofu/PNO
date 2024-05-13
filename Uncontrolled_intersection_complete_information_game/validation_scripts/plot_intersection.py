"""
An uncontrolled intersection dataset for theta function
===========================
In this example, we visualize the grid world for different theta setting
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib import cm
from matplotlib.collections import LineCollection


# %%
# Load the dataset
# ----------------
# Training samples are 20x20

data_path = './train_data/intersection_param_fun_400.mat'
Param_fun = scio.loadmat(data_path)

param_type = ['theta_11', 'theta_15', 'theta_51', 'theta_55']
name = ['1, 1', '1, 5', '5, 1', '5, 5']

# Which sample to view
N_choice = 0

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

plt.rc('font', **font)
fig, axs = plt.subplots(1, 1, figsize=(6, 6))

param_fun = Param_fun[str(param_type[N_choice])]

# plot grid
for i in range(20):
    x = i
    plt.hlines(i, 0, 20, color="gray", linewidth=2)

for i in range(20):
    y = i
    plt.vlines(i, 0, 20, color="gray", linewidth=2)

# white represent theta=0 while black represents theta=1
tmp = np.ones((20, 20))
param_fun = tmp - param_fun
axs.imshow(param_fun, cmap='gray', origin='lower', extent=(0, 20, 0, 20))
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
axs.set_xlim(0, 20)
axs.set_ylim(0, 20)
title = 'Input Function at $\u03B8=($' + str(name[N_choice]) + '$)$'
axs.set_title(title, fontweight='bold')

plt.tight_layout()
plt.show()

