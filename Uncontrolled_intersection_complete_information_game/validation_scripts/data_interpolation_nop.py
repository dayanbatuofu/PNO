import numpy as np
import scipy.io
import torch
N_states = 2

def U_star(X_aug):
    '''Control as a function of the costate.'''
    # If we keep collision function L in the value cost function V, we consider dH/du = 0 and get the U*
    B = np.array([[0], [1]])
    A = X_aug[2 * N_states:3 * N_states]
    U1 = np.matmul(B.T, A) / 2
    A = X_aug[5 * N_states:6 * N_states]
    U2 = np.matmul(B.T, A) / 2

    max_acc = 10
    min_acc = -5
    U1[np.where(U1 > max_acc)] = max_acc
    U1[np.where(U1 < min_acc)] = min_acc
    U2[np.where(U2 > max_acc)] = max_acc
    U2[np.where(U2 < min_acc)] = min_acc

    return U1, U2

def generate(data):
    t_bar = np.linspace(0.0, 3.0, num=5000)
    t_step = t_bar[1] - t_bar[0]
    X_bar = np.zeros((4, t_bar.shape[0]))
    V_bar = np.zeros((2, t_bar.shape[0]))
    t_bar = t_bar.reshape(1, -1)

    t = data['t']  # time is from train_data
    X = data['X']
    V = data['V']
    data.update({'t0': data['t']})
    idx0 = np.nonzero(np.equal(data.pop('t0'), 0))[1]
    print(len(idx0))

    t_OUT = np.empty((1, 0))
    X_OUT = np.empty((4, 0))
    V_OUT = np.empty((2, 0))

    for n in range(1, len(idx0) + 1):
        if n == len(idx0):
            t_e = t[:, idx0[n - 1]:]
            X_e = X[:, idx0[n - 1]:]
            V_e = V[:, idx0[n - 1]:]
        else:
            t_e = t[:, idx0[n - 1]: idx0[n]]
            X_e = X[:, idx0[n - 1]: idx0[n]]
            V_e = V[:, idx0[n - 1]: idx0[n]]

        i = 0
        j = 0
        time = 0
        while time < 3.0:
            while t_e[0][i] <= time:
                i += 1
            X_bar[0][j] = (time - t_e[0][i - 1]) * (X_e[0][i] - X_e[0][i - 1]) / (t_e[0][i] - t_e[0][i - 1]) + X_e[0][i - 1]
            X_bar[1][j] = (time - t_e[0][i - 1]) * (X_e[1][i] - X_e[1][i - 1]) / (t_e[0][i] - t_e[0][i - 1]) + X_e[1][i - 1]
            X_bar[2][j] = (time - t_e[0][i - 1]) * (X_e[2][i] - X_e[2][i - 1]) / (t_e[0][i] - t_e[0][i - 1]) + X_e[2][i - 1]
            X_bar[3][j] = (time - t_e[0][i - 1]) * (X_e[3][i] - X_e[3][i - 1]) / (t_e[0][i] - t_e[0][i - 1]) + X_e[3][i - 1]

            V_bar[0][j] = (time - t_e[0][i - 1]) * (V_e[0][i] - V_e[0][i - 1]) / (t_e[0][i] - t_e[0][i - 1]) + V_e[0][i - 1]
            V_bar[1][j] = (time - t_e[0][i - 1]) * (V_e[1][i] - V_e[1][i - 1]) / (t_e[0][i] - t_e[0][i - 1]) + V_e[1][i - 1]

            time = time + t_step
            j += 1

        X_bar[0][-1] = X_e[0][-1]
        X_bar[1][-1] = X_e[1][-1]
        X_bar[2][-1] = X_e[2][-1]
        X_bar[3][-1] = X_e[3][-1]

        V_bar[0][-1] = V_e[0][-1]
        V_bar[1][-1] = V_e[1][-1]

        t_OUT = np.hstack((t_OUT, t_bar))
        X_OUT = np.hstack((X_OUT, X_bar))
        V_OUT = np.hstack((V_OUT, V_bar))
        pass

    new_data = dict()

    new_data.update({'lb_1': np.min(X_OUT[:N_states], axis=1, keepdims=True),
                     'ub_1': np.max(X_OUT[:N_states], axis=1, keepdims=True),
                     'lb_2': np.min(X_OUT[N_states:2 * N_states], axis=1, keepdims=True),
                     'ub_2': np.max(X_OUT[N_states:2 * N_states], axis=1, keepdims=True),
                     'V_min_1': np.min(V_OUT[-2:-1, :]), 'V_max_1': np.max(V_OUT[-2:-1, :]),
                     'V_min_2': np.min(V_OUT[-1, :]), 'V_max_2': np.max(V_OUT[-1, :]),
                     't': t_OUT, 'X': X_OUT, 'V': V_OUT})

    return new_data

model = ['hno', 'pno']
activation = ['tanh', 'sine', 'relu']
# policy = ['1_1', '2_2', '3_3', '4_4', '5_5', '1_2', '1_3', '1_4', '1_5', '2_3', '2_4', '2_5', '3_4', '3_5', '4_5',
#           '2_1', '3_1', '4_1', '5_1', '3_2', '4_2', '5_2', '4_3', '5_3', '5_4']

policy = ['2_2', '3_3', '4_4', '5_5', '1_2', '1_3', '1_4', '1_5', '2_3', '2_4', '2_5', '3_4', '3_5', '4_5',
          '2_1', '3_1', '4_1', '5_1', '3_2', '4_2', '5_2', '4_3', '5_3', '5_4']

for i in range(len(policy)):

    N_model = 1
    N_policy = i
    N_activation = 2

    # path = './test_data/data_test_' + str(policy[N_policy]) + '_600.mat'
    path = 'closed_loop/HNO/closedloop_traj_' + str(model[N_model]) + '_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '.mat'
    # path = 'closed_loop/HNO/closedloop_traj_' + str(model[N_model]) + '_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '_nc.mat'
    # path = 'closed_loop/PNO/' + str(activation[N_activation]) + '/closedloop_traj_' + str(model[N_model]) + '_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '.mat'
    # path = 'closed_loop/PNO/' + str(activation[N_activation]) + '/closedloop_traj_' + str(model[N_model]) + '_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '_nc.mat'
    data = scipy.io.loadmat(path)

    data.update({'t0': data['t']})
    idx0 = np.nonzero(np.equal(data.pop('t0'), 0))[1]

    t_OUT = data['t'][:, idx0[0]: idx0[1]]
    X_OUT = data['X'][:, idx0[0]: idx0[1]]
    V_OUT = data['V'][:, idx0[0]: idx0[1]]

    new_data = generate(data)
    # save_path = './test_data/data_test_' + str(policy[N_policy]) + '_600_5k.mat'
    save_path = 'closed_loop/HNO/closedloop_traj_' + str(model[N_model]) + '_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '_5k.mat'
    # save_path = 'closed_loop/HNO/closedloop_traj_' + str(model[N_model]) + '_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '_nc_5k.mat'
    # save_path = 'closed_loop/PNO/' + str(activation[N_activation]) + '/closedloop_traj_' + str(model[N_model]) + '_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '_5k.mat'
    # save_path = 'closed_loop/PNO/' + str(activation[N_activation]) + '/closedloop_traj_' + str(model[N_model]) + '_initial_' + str(policy[N_policy]) + '_' + str(activation[N_activation]) + '_nc_5k.mat'
    scipy.io.savemat(save_path, new_data)
