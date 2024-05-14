import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


#________________________________________________________________________________________

activation = ['tanh', 'sin', 'relu']

policy = ['1_1', '2_2', '3_3', '4_4', '5_5']
# policy = ['1_2', '1_3', '1_4', '1_5', '2_3', '2_4', '2_5', '3_4', '3_5', '4_5']
# policy = ['2_1', '3_1', '4_1', '5_1', '3_2', '4_2', '5_2', '4_3', '5_3', '5_4']

name = ['1, 1', '2, 2', '3, 3', '4, 4', '5, 5']
# name = ['1, 2', '1, 3', '1, 4', '1, 5', '2, 3', '2, 4', '2, 5', '3, 4', '3, 5', '4, 5']
# name = ['2, 1', '3, 1', '4, 1', '5, 1', '3, 2', '4, 2', '5, 2', '4, 3', '5, 3', '5, 4']

N_policy = 1
N_activation = 2
agent1 = True
agent2 = False

choice = 2

if choice == 0:
    file = 'test_data/data_test_' + str(policy[N_policy]) + '_600_5k.mat'
    # file = 'nop_test_data/data_test_' + str(policy[N_policy]) + '_600_nc.mat'
    data = scipy.io.loadmat(file)
    title = 'GT $\u03B8=($' + str(name[N_policy]) + '$)$'
elif choice == 1:
    file = 'closed_loop/HNO/closedloop_traj_hno_initial_' + str(policy[N_policy]) + '_tanh_5k.mat'
    # file = 'closed_loop/HNO/closedloop_traj_hno_initial_' + str(policy[N_policy]) + '_tanh_nc_5k.mat'
    data = scipy.io.loadmat(file)
    title = 'Hybrid $\u03B8=($' + str(name[N_policy]) + '$)$'
else:
    # file = 'closed_loop/PNO/' + str(activation[N_activation]) + '/closedloop_traj_pno_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '_5k.mat'
    file = 'closed_loop/PNO/' + str(activation[N_activation]) + '/closedloop_traj_pno_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '_nc.mat'
    data = scipy.io.loadmat(file)
    title = 'Pontryagin $\u03B8=($' + str(name[N_policy]) + '$)$'

index = 2
theta1 = int(name[N_policy][0])
theta2 = int(name[N_policy][3:])

# collision area
bl = (35 - theta1 * 0.75, 35 - theta2 * 0.75)
tr = (38.75, 38.75)

def pointInRect(bl, tr, points):
    for p in points:
        if (p[0] >= bl[0] and p[0] <= tr[0] and p[1] >= bl[1] and p[1] <= tr[1]):
            return True
    return False

#____________________________________________________________________________________________________

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

plt.rc('font', **font)

X = data['X']
V = data['V']
T = data['t']
# U = data['U']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]
X0 = X[:, idx0]
count = 0
traj_list = []

fig, axs = plt.subplots(1, 1, figsize=(8, 6))

if agent1 == True:
    # norm = plt.Normalize(np.min(V[0, :]), 0)
    norm = plt.Normalize(-1500, 0)
if agent2 == True:
    norm = plt.Normalize(np.min(V[1, :]), 0)

# for n in range(1, 2):
for n in range(1, len(idx0) + 1):
    if n == len(idx0):
        d1 = X[0, idx0[n - 1]:]
        d2 = X[2, idx0[n - 1]:]
        V1 = V[0, idx0[n - 1]:]
        V2 = V[1, idx0[n - 1]:]
        t = T[0, idx0[n - 1]:]

        pairs = zip(d1, d2)
        if pointInRect(bl, tr, pairs):
            count += 1
            print(n - 1)
            traj_list.append(n - 1)
    else:
        d1 = X[0, idx0[n - 1]: idx0[n]]
        d2 = X[2, idx0[n - 1]: idx0[n]]
        V1 = V[0, idx0[n - 1]: idx0[n]]
        V2 = V[1, idx0[n - 1]: idx0[n]]
        t = T[0, idx0[n - 1]: idx0[n]]

        pairs = zip(d1, d2)
        if pointInRect(bl, tr, pairs):
            count += 1
            print(n - 1)
            traj_list.append(n - 1)

    points = np.array([d1, d2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize/Plot Value 1 ColorBar
    lc = LineCollection(segments, cmap='PuOr', norm=norm)  # PuOr, viridis
    if agent1 == True:
        lc.set_array(V1)
    if agent2 == True:
        lc.set_array(V2)
    line = axs.add_collection(lc)

# Configure Plot
axs.set_title(title, fontweight='bold')

start1 = patches.Rectangle((15, 15), 5, 5, linewidth=0.5, edgecolor='k', facecolor='none')
intersection1 = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
axs.add_patch(intersection1)
axs.add_patch(start1)

train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - 1 * 0.75), 3 + theta1 * 0.75 + 0.75,
                           3 + 1 * 0.75 + 0.75, linewidth=1, edgecolor='k', facecolor='none')
axs.add_patch(train1)
train2 = patches.Rectangle((35 - 1 * 0.75, 35 - theta2 * 0.75), 3 + 1 * 0.75 + 0.75,
                            3 + theta2 * 0.75 + 0.75, linewidth=1, edgecolor='k', facecolor='none')
axs.add_patch(train2)
axs.set_xlim(15, 40)
axs.set_xlabel('d1', fontweight='bold')
axs.set_ylim(15, 40)
axs.set_ylabel('d2', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
cbar = fig.colorbar(line, ax=axs)

for l in cbar.ax.yaxis.get_ticklabels():
    l.set_weight('bold')

for i in range(len(traj_list)):
    axs.plot(X0[0, traj_list[i]], X0[2, traj_list[i]], marker='o', color='red', markersize=5)

print("Total Collision: %d" %count)
plt.show()