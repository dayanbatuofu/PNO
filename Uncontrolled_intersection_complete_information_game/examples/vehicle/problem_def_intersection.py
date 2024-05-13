import numpy as np
import torch
import scipy.io
from examples.problem_def_template import config_prototype, problem_prototype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class config_NN(config_prototype):
    def __init__(self, N_states, time_dependent):
        self.N_layers = 3
        self.N_neurons = 64
        self.layers = self.build_layers(N_states,
                                        time_dependent,
                                        self.N_layers,
                                        self.N_neurons)

        self.random_seeds = {'train': 1122, 'generate': 1122}

        self.ODE_solver = 'RK23'
        # Accuracy level of BVP data
        self.data_tol = 1e-3  # 1e-03
        # Max number of nodes to use in BVP
        self.max_nodes = 2500  # 800000
        # Time horizon
        self.t1 = 3.

        # Time subintervals to use in time marching
        Nt = 10  # 10
        self.tseq = np.linspace(0., self.t1, Nt + 1)[1:]

        # Time step for integration and sampling
        self.dt = 1e-01
        # Standard deviation of measurement noise
        self.sigma = np.pi * 1e-02

        # Which dimensions to plot when predicting value function V(0,x)?
        # (unspecified dimensions are held at mean value)
        self.plotdims = [0, 3]

        # Number of training trajectories
        self.Ns = {'train': 1600, 'val': 600, 'test': 600}

        ##### Options for training #####
        # Number of data points to use in first training rounds
        # Set to None to use whole data set
        self.batch_size = None  # 200

        # Maximum factor to increase data set size each round
        self.Ns_scale = 2
        # Number of candidate points to pick from when selecting large gradient
        # points during adaptive sampling
        self.Ns_cand = 2
        # Maximum size of batch size to use
        self.Ns_max = 8192

        # Convergence tolerance parameter (see paper)
        self.conv_tol = 1e-03

        # maximum and minimum number of training rounds
        self.max_rounds = 1
        self.min_rounds = 1

        # List or array of weights on gradient term, length = max_rounds
        self.weight_A = [1.]  # 1
        # List or array of weights on control learning term, not used in paper
        self.weight_U = [0.]  # 0.1

        # Dictionary of options to be passed to L-BFGS-B optimizer
        # Leave empty for default values
        self.BFGS_opts = {}


class setup_problem(problem_prototype):
    def __init__(self):
        self.N_states = 2
        self.t1 = 3.

        # Parameter setting for the equation X_dot = Ax+Bu
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0], [1]])

        # Initial condition bounds (different initial setting)
        # Currently, we give the initial position is [15m, 20m]
        # initial velocity is [18m/s, 25m/s]
        self.X0_lb = np.array([[15.], [18.], [15.], [18.]])
        self.X0_ub = np.array([[20.], [25.], [20.], [25.]])

        self.beta = 10000  # 10000
        self.theta1 = 1  # [1, 5]
        self.theta2 = 1  # [1, 5]

        # weight for terminal lose
        self.alpha = 1e-06  # 1e-06

        # Length for each vehicle
        self.L1 = 3
        self.L2 = 3

        # Length for each vehicle
        self.W1 = 1.5
        self.W2 = 1.5

        # Road length setting
        self.R1 = 70  # 71 or 62, 73
        self.R2 = 70

        self.max_acc = 10
        self.min_acc = -5

        data_path = '../validation_scripts/train_data/intersection_param_fun_400.mat'
        self.input_fun = scipy.io.loadmat(data_path)

    def U_star(self, X_aug):
        '''Control as a function of the costate.'''
        # If we keep collision function L in the value cost function V, we consider dH/du = 0 and get the U*
        A = X_aug[2 * self.N_states:3 * self.N_states]
        U1 = np.matmul(self.B.T, A) / 2
        A = X_aug[5 * self.N_states:6 * self.N_states]
        U2 = np.matmul(self.B.T, A) / 2

        U1[np.where(U1 > self.max_acc)] = self.max_acc
        U1[np.where(U1 < self.min_acc)] = self.min_acc
        U2[np.where(U2 > self.max_acc)] = self.max_acc
        U2[np.where(U2 < self.min_acc)] = self.min_acc

        return U1, U2

    def U_NN(self, t, X_aug, model, param_fun):
        d1 = (2.0 * (X_aug[0, :] - 15) / (105 - 15) - 1).reshape(1, -1)
        v1 = (2.0 * (X_aug[1, :] - 15) / (32 - 15) - 1).reshape(1, -1)
        d2 = (2.0 * (X_aug[2, :] - 15) / (105 - 15) - 1).reshape(1, -1)
        v2 = (2.0 * (X_aug[3, :] - 15) / (32 - 15) - 1).reshape(1, -1)
        label1 = torch.zeros((1, 1))
        label2 = torch.ones((1, 1))

        X = np.vstack((d1, v1, d2, v2))
        X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
        t = torch.tensor(3-t, dtype=torch.float32, requires_grad=True).reshape(1, -1)
        coords_1 = torch.cat((t, X), dim=1)
        coords_2 = torch.cat((t, (torch.cat((X[:, 2:], X[:, :2]), dim=1))), dim=1)
        coords_vn = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
        coords_1 = torch.cat((coords_1, label1), dim=1)
        coords_2 = torch.cat((coords_2, label2), dim=1)
        coords_cn = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
        param_fun = param_fun.unsqueeze(0)
        model_input = {'coords_vn': coords_vn.to(device),
                       'coords_cn': coords_cn.to(device),
                       'input_fun': param_fun.to(device)}
        model_output = model(model_input)
        costate = model_output['model_out_cn']

        # calculate the costate for agent 1 and 2
        lam11_2 = costate[:, :1, 1:2]  # lambda_11
        lam22_2 = costate[:, 1:, -1:]  # lambda_22

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        # Agent 1's action, be careful about the order of u1>0 and u1<0
        u1_star = 0.5 * lam11_2 * 10

        # Agent 2's action, be careful about the order of u2>0 and u2<0
        u2_star = 0.5 * lam22_2 * 10

        u1_star[torch.where(u1_star > max_acc)] = max_acc
        u1_star[torch.where(u1_star < min_acc)] = min_acc
        u2_star[torch.where(u2_star > max_acc)] = max_acc
        u2_star[torch.where(u2_star < min_acc)] = min_acc

        return u1_star, u2_star

    # Boundary function for BVP
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:2 * self.N_states]
            XT = X_aug_T[:2 * self.N_states]
            AT = X_aug_T[2 * self.N_states:6 * self.N_states]
            VT = X_aug_T[6 * self.N_states:]

            # Boundary setting for lambda(T) when it is the final time T
            dFdXT = np.concatenate((np.array([self.alpha]),
                                    np.array([-2 * (XT[1] - X0[1])]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([self.alpha]),
                                    np.array([-2 * (XT[3] - X0[3])])))

            # Terminal cost in the value function, see the new version of HJI equation
            F = -np.array((self.alpha * XT[0] - (XT[1] - X0[1]) ** 2, self.alpha * XT[2] - (XT[3] - X0[3]) ** 2))

            return np.concatenate((X0 - X0_in, AT - dFdXT, VT - F))

        return bc

    def v_dynamics(self, t, X_aug, model, theta1, theta2):
        '''Evaluation of the augmented dynamics at a vector of time instances'''
        X_aug = X_aug.reshape(-1, 1)
        # Control as a function of the costate

        if theta1 == 1 and theta2 == 1:
            param_fun_P1 = torch.tensor(self.input_fun['theta_aa'], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1)
            param_fun_P2 = torch.tensor(self.input_fun['theta_aa'], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1)
            param_fun = torch.cat((param_fun_P1, param_fun_P2), dim=0)
        elif theta1 == 1 and theta2 == 5:
            param_fun_P1 = torch.tensor(self.input_fun['theta_ana'], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1)
            param_fun_P2 = torch.tensor(self.input_fun['theta_naa'], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1)
            param_fun = torch.cat((param_fun_P1, param_fun_P2), dim=0)
        elif theta1 == 5 and theta2 == 1:
            param_fun_P1 = torch.tensor(self.input_fun['theta_naa'], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1)
            param_fun_P2 = torch.tensor(self.input_fun['theta_ana'], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1)
            param_fun = torch.cat((param_fun_P1, param_fun_P2), dim=0)
        else:
            param_fun_P1 = torch.tensor(self.input_fun['theta_nana'], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1)
            param_fun_P2 = torch.tensor(self.input_fun['theta_nana'], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1)
            param_fun = torch.cat((param_fun_P1, param_fun_P2), dim=0)

        # U1, U2 = self.U_star(X_aug)
        U1, U2 = self.U_NN(t, X_aug, model, param_fun)
        U1 = U1.squeeze(0).detach().cpu().numpy()
        U2 = U2.squeeze(0).detach().cpu().numpy()

        # State for each vehicle
        X1 = X_aug[:self.N_states]
        X2 = X_aug[self.N_states:2 * self.N_states]

        # State space function: X_dot = Ax+Bu
        dXdt_1 = np.matmul(self.A, X1) + np.matmul(self.B, U1)
        dXdt_2 = np.matmul(self.A, X2) + np.matmul(self.B, U2)

        # lambda in Hamiltonian equation
        A_11 = X_aug[2 * self.N_states:3 * self.N_states]
        A_12 = X_aug[3 * self.N_states:4 * self.N_states]
        A_21 = X_aug[4 * self.N_states:5 * self.N_states]
        A_22 = X_aug[5 * self.N_states:6 * self.N_states]

        # Sigmoid function: sigmoid(x1_in)*inverse_sigmoid(x1_out)*sigmoid(x2_in)*inverse_sigmoid(x2_out)
        x1 = torch.tensor(X1[0], requires_grad=True, dtype=torch.float32)  # including x1,v1
        x2 = torch.tensor(X2[0], requires_grad=True, dtype=torch.float32)  # including x1,v1

        x1_in = (x1 - self.R1 / 2 + theta1 * self.W2 / 2) * 5  # 3
        x1_out = -(x1 - self.R1 / 2 - self.W2 / 2 - self.L1) * 5
        x2_in = (x2 - self.R2 / 2 + theta2 * self.W1 / 2) * 5
        x2_out = -(x2 - self.R2 / 2 - self.W1 / 2 - self.L2) * 5

        Collision_F_x = self.beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(x2_in) * torch.sigmoid(x2_out)

        Collision_F_x_sum = torch.sum(Collision_F_x)
        Collision_F_x_sum.requires_grad_()

        dL1dx1 = torch.autograd.grad(Collision_F_x_sum, x1, create_graph=True)[0].detach().numpy()
        dL1dx2 = torch.autograd.grad(Collision_F_x_sum, x2, create_graph=True)[0].detach().numpy()
        dL2dx1 = torch.autograd.grad(Collision_F_x_sum, x1, create_graph=True)[0].detach().numpy()
        dL2dx2 = torch.autograd.grad(Collision_F_x_sum, x2, create_graph=True)[0].detach().numpy()

        dL1dv1 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        dL1dv2 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        dL2dv1 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        dL2dv2 = np.zeros(dL2dx2.shape[0], dtype=np.int32)

        # lambda_dot in PMP equation
        dAdt_11 = -np.matmul(self.A.T, A_11) + np.array([dL1dx1, dL1dv1])
        dAdt_12 = -np.matmul(self.A.T, A_12) + np.array([dL1dx2, dL1dv2])
        dAdt_21 = -np.matmul(self.A.T, A_21) + np.array([dL2dx1, dL2dv1])
        dAdt_22 = -np.matmul(self.A.T, A_22) + np.array([dL2dx2, dL2dv2])

        return np.vstack((dXdt_1, dXdt_2, dAdt_11, dAdt_12, dAdt_21, dAdt_22)).reshape(-1)
