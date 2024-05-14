import torch
from torch.utils.data import Dataset
import scipy.io
import os
import numpy as np

class IntersectionHJI_Supervised(Dataset):
    def __init__(self, seed=0, rank=0):

        super().__init__()
        torch.manual_seed(0)

        current_dir = os.path.dirname(os.path.abspath(__file__))

        data_path11 = current_dir + '/validation_scripts/train_data/data_train_1_1_500.mat'
        data_path12 = current_dir + '/validation_scripts/train_data/data_train_1_5_500.mat'
        data_path13 = current_dir + '/validation_scripts/train_data/data_train_5_1_500.mat'
        data_path14 = current_dir + '/validation_scripts/train_data/data_train_5_5_500.mat'

        train_data11 = scipy.io.loadmat(data_path11)
        train_data12 = scipy.io.loadmat(data_path12)
        train_data13 = scipy.io.loadmat(data_path13)
        train_data14 = scipy.io.loadmat(data_path14)
        self.train_data11 = train_data11
        self.train_data12 = train_data12
        self.train_data13 = train_data13
        self.train_data14 = train_data14

        data_path21 = current_dir + '/validation_scripts/train_data/data_train_1_1_500_spare.mat'
        data_path22 = current_dir + '/validation_scripts/train_data/data_train_1_5_500_spare.mat'
        data_path23 = current_dir + '/validation_scripts/train_data/data_train_5_1_500_spare.mat'
        data_path24 = current_dir + '/validation_scripts/train_data/data_train_5_5_500_spare.mat'

        train_data21 = scipy.io.loadmat(data_path21)
        train_data22 = scipy.io.loadmat(data_path22)
        train_data23 = scipy.io.loadmat(data_path23)
        train_data24 = scipy.io.loadmat(data_path24)
        self.train_data21 = train_data21
        self.train_data22 = train_data22
        self.train_data23 = train_data23
        self.train_data24 = train_data24

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = current_dir + '/validation_scripts/train_data/intersection_param_fun_400.mat'

        self.input_fun = scipy.io.loadmat(data_path)

        # Set the seed
        torch.manual_seed(seed)

        self.rank = rank

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        self.t_train11 = torch.tensor(self.train_data11['t'], dtype=torch.float32).flip(1)
        self.X_train11 = torch.tensor(self.train_data11['X'], dtype=torch.float32)
        self.A_train11 = torch.tensor(self.train_data11['A'], dtype=torch.float32)
        self.V_train11 = torch.tensor(self.train_data11['V'], dtype=torch.float32)
        self.t_train12 = torch.tensor(self.train_data12['t'], dtype=torch.float32).flip(1)
        self.X_train12 = torch.tensor(self.train_data12['X'], dtype=torch.float32)
        self.A_train12 = torch.tensor(self.train_data12['A'], dtype=torch.float32)
        self.V_train12 = torch.tensor(self.train_data12['V'], dtype=torch.float32)
        self.t_train13 = torch.tensor(self.train_data13['t'], dtype=torch.float32).flip(1)
        self.X_train13 = torch.tensor(self.train_data13['X'], dtype=torch.float32)
        self.A_train13 = torch.tensor(self.train_data13['A'], dtype=torch.float32)
        self.V_train13 = torch.tensor(self.train_data13['V'], dtype=torch.float32)
        self.t_train14 = torch.tensor(self.train_data14['t'], dtype=torch.float32).flip(1)
        self.X_train14 = torch.tensor(self.train_data14['X'], dtype=torch.float32)
        self.A_train14 = torch.tensor(self.train_data14['A'], dtype=torch.float32)
        self.V_train14 = torch.tensor(self.train_data14['V'], dtype=torch.float32)

        self.t_train21 = torch.tensor(self.train_data21['t'], dtype=torch.float32).flip(1)
        self.X_train21 = torch.tensor(self.train_data21['X'], dtype=torch.float32)
        self.A_train21 = torch.tensor(self.train_data21['A'], dtype=torch.float32)
        self.V_train21 = torch.tensor(self.train_data21['V'], dtype=torch.float32)
        self.t_train22 = torch.tensor(self.train_data22['t'], dtype=torch.float32).flip(1)
        self.X_train22 = torch.tensor(self.train_data22['X'], dtype=torch.float32)
        self.A_train22 = torch.tensor(self.train_data22['A'], dtype=torch.float32)
        self.V_train22 = torch.tensor(self.train_data22['V'], dtype=torch.float32)
        self.t_train23 = torch.tensor(self.train_data23['t'], dtype=torch.float32).flip(1)
        self.X_train23 = torch.tensor(self.train_data23['X'], dtype=torch.float32)
        self.A_train23 = torch.tensor(self.train_data23['A'], dtype=torch.float32)
        self.V_train23 = torch.tensor(self.train_data23['V'], dtype=torch.float32)
        self.t_train24 = torch.tensor(self.train_data24['t'], dtype=torch.float32).flip(1)
        self.X_train24 = torch.tensor(self.train_data24['X'], dtype=torch.float32)
        self.A_train24 = torch.tensor(self.train_data24['A'], dtype=torch.float32)
        self.V_train24 = torch.tensor(self.train_data24['V'], dtype=torch.float32)

        self.lb = torch.tensor([[15], [15], [15], [15]], dtype=torch.float32)
        self.ub = torch.tensor([[105], [32], [105], [32]], dtype=torch.float32)

        self.X_train11 = 2.0 * (self.X_train11 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train12 = 2.0 * (self.X_train12 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train13 = 2.0 * (self.X_train13 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train14 = 2.0 * (self.X_train14 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train21 = 2.0 * (self.X_train21 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train22 = 2.0 * (self.X_train22 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train23 = 2.0 * (self.X_train23 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train24 = 2.0 * (self.X_train24 - self.lb) / (self.ub - self.lb) - 1.

        self.X_train = torch.cat((self.X_train11, self.X_train21,
                                  self.X_train12, self.X_train22,
                                  self.X_train13, self.X_train23,
                                  self.X_train14, self.X_train24), dim=1)
        self.t_train = torch.cat((self.t_train11, self.t_train21,
                                  self.t_train12, self.t_train22,
                                  self.t_train13, self.t_train23,
                                  self.t_train14, self.t_train24), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)

        coords_1 = torch.cat((self.t_train.T, coords_1), dim=1)
        coords_2 = torch.cat((self.t_train.T, coords_2), dim=1)

        # set up ground truth for values and costates
        groundtruth_values1 = torch.cat((self.V_train11[0, :].reshape(-1, 1), self.V_train21[0, :].reshape(-1, 1),
                                         self.V_train12[0, :].reshape(-1, 1), self.V_train22[0, :].reshape(-1, 1),
                                         self.V_train13[0, :].reshape(-1, 1), self.V_train23[0, :].reshape(-1, 1),
                                         self.V_train14[0, :].reshape(-1, 1), self.V_train24[0, :].reshape(-1, 1)), dim=0)
        groundtruth_values2 = torch.cat((self.V_train11[1, :].reshape(-1, 1), self.V_train21[1, :].reshape(-1, 1),
                                         self.V_train12[1, :].reshape(-1, 1), self.V_train22[1, :].reshape(-1, 1),
                                         self.V_train13[1, :].reshape(-1, 1), self.V_train23[1, :].reshape(-1, 1),
                                         self.V_train14[1, :].reshape(-1, 1), self.V_train24[1, :].reshape(-1, 1)), dim=0)
        groundtruth_values = torch.cat((groundtruth_values1, groundtruth_values2), dim=0)

        groundtruth_costates1 = torch.cat((self.A_train11[:4, :].T, self.A_train21[:4, :].T,
                                           self.A_train12[:4, :].T, self.A_train22[:4, :].T,
                                           self.A_train13[:4, :].T, self.A_train23[:4, :].T,
                                           self.A_train14[:4, :].T, self.A_train24[:4, :].T), dim=0)
        groundtruth_costates2 = torch.cat((self.A_train11[4:, :].T, self.A_train21[4:, :].T,
                                           self.A_train12[4:, :].T, self.A_train22[4:, :].T,
                                           self.A_train13[4:, :].T, self.A_train23[4:, :].T,
                                           self.A_train14[4:, :].T, self.A_train24[4:, :].T), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        theta_11 = torch.tensor(self.input_fun['theta_11'], dtype=torch.float32).clone()
        theta_15 = torch.tensor(self.input_fun['theta_15'], dtype=torch.float32).clone()
        theta_51 = torch.tensor(self.input_fun['theta_51'], dtype=torch.float32).clone()
        theta_55 = torch.tensor(self.input_fun['theta_55'], dtype=torch.float32).clone()

        num_input = coords_1.shape[0] // 4

        theta_11_hji = theta_11.unsqueeze(0).repeat(num_input, 1, 1).flatten(start_dim=1)
        theta_15_hji = theta_15.unsqueeze(0).repeat(num_input, 1, 1).flatten(start_dim=1)
        theta_51_hji = theta_51.unsqueeze(0).repeat(num_input, 1, 1).flatten(start_dim=1)
        theta_55_hji = theta_55.unsqueeze(0).repeat(num_input, 1, 1).flatten(start_dim=1)

        input_fun = torch.cat((theta_11_hji, theta_15_hji, theta_51_hji, theta_55_hji,
                               theta_11_hji, theta_51_hji, theta_15_hji, theta_55_hji), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)
        return {'coords': coords, 'input_fun': input_fun}, \
               {'groundtruth_values': groundtruth_values,
                'groundtruth_costates': groundtruth_costates}


class IntersectionHJI_Hybrid(Dataset):
    def __init__(self, numpoints, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3, num_src_samples=1000,
                 seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.numpoints = numpoints
        self.num_states = 4

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.full_count = counter_end
        self.alpha = 1e-6

        # Set the seed
        torch.manual_seed(seed)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path11 = current_dir + '/validation_scripts/train_data/data_train_1_1_500.mat'
        data_path12 = current_dir + '/validation_scripts/train_data/data_train_1_5_500.mat'
        data_path13 = current_dir + '/validation_scripts/train_data/data_train_5_1_500.mat'
        data_path14 = current_dir + '/validation_scripts/train_data/data_train_5_5_500.mat'

        train_data11 = scipy.io.loadmat(data_path11)
        train_data12 = scipy.io.loadmat(data_path12)
        train_data13 = scipy.io.loadmat(data_path13)
        train_data14 = scipy.io.loadmat(data_path14)
        self.train_data11 = train_data11
        self.train_data12 = train_data12
        self.train_data13 = train_data13
        self.train_data14 = train_data14

        data_path21 = current_dir + '/validation_scripts/train_data/data_train_1_1_500_spare.mat'
        data_path22 = current_dir + '/validation_scripts/train_data/data_train_1_5_500_spare.mat'
        data_path23 = current_dir + '/validation_scripts/train_data/data_train_5_1_500_spare.mat'
        data_path24 = current_dir + '/validation_scripts/train_data/data_train_5_5_500_spare.mat'

        train_data21 = scipy.io.loadmat(data_path21)
        train_data22 = scipy.io.loadmat(data_path22)
        train_data23 = scipy.io.loadmat(data_path23)
        train_data24 = scipy.io.loadmat(data_path24)
        self.train_data21 = train_data21
        self.train_data22 = train_data22
        self.train_data23 = train_data23
        self.train_data24 = train_data24

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = current_dir + '/validation_scripts/train_data/intersection_param_fun_400.mat'

        self.input_fun = scipy.io.loadmat(data_path)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # supervised learning data
        self.t_train11 = torch.tensor(self.train_data11['t'], dtype=torch.float32).flip(1)
        self.X_train11 = torch.tensor(self.train_data11['X'], dtype=torch.float32)
        self.A_train11 = torch.tensor(self.train_data11['A'], dtype=torch.float32)
        self.V_train11 = torch.tensor(self.train_data11['V'], dtype=torch.float32)
        self.t_train12 = torch.tensor(self.train_data12['t'], dtype=torch.float32).flip(1)
        self.X_train12 = torch.tensor(self.train_data12['X'], dtype=torch.float32)
        self.A_train12 = torch.tensor(self.train_data12['A'], dtype=torch.float32)
        self.V_train12 = torch.tensor(self.train_data12['V'], dtype=torch.float32)
        self.t_train13 = torch.tensor(self.train_data13['t'], dtype=torch.float32).flip(1)
        self.X_train13 = torch.tensor(self.train_data13['X'], dtype=torch.float32)
        self.A_train13 = torch.tensor(self.train_data13['A'], dtype=torch.float32)
        self.V_train13 = torch.tensor(self.train_data13['V'], dtype=torch.float32)
        self.t_train14 = torch.tensor(self.train_data14['t'], dtype=torch.float32).flip(1)
        self.X_train14 = torch.tensor(self.train_data14['X'], dtype=torch.float32)
        self.A_train14 = torch.tensor(self.train_data14['A'], dtype=torch.float32)
        self.V_train14 = torch.tensor(self.train_data14['V'], dtype=torch.float32)

        self.t_train21 = torch.tensor(self.train_data21['t'], dtype=torch.float32).flip(1)
        self.X_train21 = torch.tensor(self.train_data21['X'], dtype=torch.float32)
        self.A_train21 = torch.tensor(self.train_data21['A'], dtype=torch.float32)
        self.V_train21 = torch.tensor(self.train_data21['V'], dtype=torch.float32)
        self.t_train22 = torch.tensor(self.train_data22['t'], dtype=torch.float32).flip(1)
        self.X_train22 = torch.tensor(self.train_data22['X'], dtype=torch.float32)
        self.A_train22 = torch.tensor(self.train_data22['A'], dtype=torch.float32)
        self.V_train22 = torch.tensor(self.train_data22['V'], dtype=torch.float32)
        self.t_train23 = torch.tensor(self.train_data23['t'], dtype=torch.float32).flip(1)
        self.X_train23 = torch.tensor(self.train_data23['X'], dtype=torch.float32)
        self.A_train23 = torch.tensor(self.train_data23['A'], dtype=torch.float32)
        self.V_train23 = torch.tensor(self.train_data23['V'], dtype=torch.float32)
        self.t_train24 = torch.tensor(self.train_data24['t'], dtype=torch.float32).flip(1)
        self.X_train24 = torch.tensor(self.train_data24['X'], dtype=torch.float32)
        self.A_train24 = torch.tensor(self.train_data24['A'], dtype=torch.float32)
        self.V_train24 = torch.tensor(self.train_data24['V'], dtype=torch.float32)

        self.lb = torch.tensor([[15], [15], [15], [15]], dtype=torch.float32)
        self.ub = torch.tensor([[105], [32], [105], [32]], dtype=torch.float32)

        self.X_train11 = 2.0 * (self.X_train11 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train12 = 2.0 * (self.X_train12 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train13 = 2.0 * (self.X_train13 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train14 = 2.0 * (self.X_train14 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train21 = 2.0 * (self.X_train21 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train22 = 2.0 * (self.X_train22 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train23 = 2.0 * (self.X_train23 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train24 = 2.0 * (self.X_train24 - self.lb) / (self.ub - self.lb) - 1.

        self.X_train = torch.cat((self.X_train11, self.X_train21,
                                  self.X_train12, self.X_train22,
                                  self.X_train13, self.X_train23,
                                  self.X_train14, self.X_train24), dim=1)
        self.t_train = torch.cat((self.t_train11, self.t_train21,
                                  self.t_train12, self.t_train22,
                                  self.t_train13, self.t_train23,
                                  self.t_train14, self.t_train24), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)

        coords_1_supervised = torch.cat((self.t_train.T, coords_1), dim=1)
        coords_2_supervised = torch.cat((self.t_train.T, coords_2), dim=1)

        # set up ground truth for values and costates
        groundtruth_values1 = torch.cat((self.V_train11[0, :].reshape(-1, 1), self.V_train21[0, :].reshape(-1, 1),
                                         self.V_train12[0, :].reshape(-1, 1), self.V_train22[0, :].reshape(-1, 1),
                                         self.V_train13[0, :].reshape(-1, 1), self.V_train23[0, :].reshape(-1, 1),
                                         self.V_train14[0, :].reshape(-1, 1), self.V_train24[0, :].reshape(-1, 1)), dim=0)
        groundtruth_values2 = torch.cat((self.V_train11[1, :].reshape(-1, 1), self.V_train21[1, :].reshape(-1, 1),
                                         self.V_train12[1, :].reshape(-1, 1), self.V_train22[1, :].reshape(-1, 1),
                                         self.V_train13[1, :].reshape(-1, 1), self.V_train23[1, :].reshape(-1, 1),
                                         self.V_train14[1, :].reshape(-1, 1), self.V_train24[1, :].reshape(-1, 1)), dim=0)
        groundtruth_values = torch.cat((groundtruth_values1, groundtruth_values2), dim=0)

        groundtruth_costates1 = torch.cat((self.A_train11[:4, :].T, self.A_train21[:4, :].T,
                                           self.A_train12[:4, :].T, self.A_train22[:4, :].T,
                                           self.A_train13[:4, :].T, self.A_train23[:4, :].T,
                                           self.A_train14[:4, :].T, self.A_train24[:4, :].T), dim=0)
        groundtruth_costates2 = torch.cat((self.A_train11[4:, :].T, self.A_train21[4:, :].T,
                                           self.A_train12[4:, :].T, self.A_train22[4:, :].T,
                                           self.A_train13[4:, :].T, self.A_train23[4:, :].T,
                                           self.A_train14[4:, :].T, self.A_train24[4:, :].T), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        # HJI data(sample entire state space)
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates for both agents
        coords_1 = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)

        # slowly grow time values from start time
        # this currently assumes start_time = 0 and max time value is tMax
        time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                self.counter / self.full_count))

        coords_1_hji = torch.cat((time, coords_1), dim=1)
        coords_2_hji = torch.cat((time, coords_2), dim=1)

        # make sure we always have training samples at the initial time
        coords_1_hji[-self.N_src_samples:, 0] = start_time
        coords_2_hji[-self.N_src_samples:, 0] = start_time

        coords_1_hji = coords_1_hji.repeat(4, 1)
        coords_2_hji = coords_2_hji.repeat(4, 1)

        # set up boundary condition: V(T) = alpha*X(T) - (V(T) - V(0))^2
        boundary_values_1 = self.alpha * ((coords_1_hji[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                            ((coords_1_hji[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values_2 = self.alpha * ((coords_2_hji[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                            ((coords_2_hji[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=0)

        dirichlet_mask = (coords_1_hji[:, 0, None] == start_time)

        if self.counter < self.full_count:
            self.counter += 1

        coords_1 = torch.cat((coords_1_supervised, coords_1_hji), dim=0)
        coords_2 = torch.cat((coords_2_supervised, coords_2_hji), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)

        theta_11 = torch.tensor(self.input_fun['theta_11'], dtype=torch.float32).clone()
        theta_15 = torch.tensor(self.input_fun['theta_15'], dtype=torch.float32).clone()
        theta_51 = torch.tensor(self.input_fun['theta_51'], dtype=torch.float32).clone()
        theta_55 = torch.tensor(self.input_fun['theta_55'], dtype=torch.float32).clone()

        num_sl = coords_1_supervised.shape[0] // 4
        num_hl = coords_1_hji.shape[0] // 4

        theta_11_sl = theta_11.unsqueeze(0).repeat(num_sl, 1, 1).flatten(start_dim=1)
        theta_15_sl = theta_15.unsqueeze(0).repeat(num_sl, 1, 1).flatten(start_dim=1)
        theta_51_sl = theta_51.unsqueeze(0).repeat(num_sl, 1, 1).flatten(start_dim=1)
        theta_55_sl = theta_55.unsqueeze(0).repeat(num_sl, 1, 1).flatten(start_dim=1)

        theta_11_hl = theta_11.unsqueeze(0).repeat(num_hl, 1, 1).flatten(start_dim=1)
        theta_15_hl = theta_15.unsqueeze(0).repeat(num_hl, 1, 1).flatten(start_dim=1)
        theta_51_hl = theta_51.unsqueeze(0).repeat(num_hl, 1, 1).flatten(start_dim=1)
        theta_55_hl = theta_55.unsqueeze(0).repeat(num_hl, 1, 1).flatten(start_dim=1)

        input_fun = torch.cat((theta_11_sl, theta_15_sl, theta_51_sl, theta_55_sl,
                               theta_11_hl, theta_15_hl, theta_51_hl, theta_55_hl,
                               theta_11_sl, theta_51_sl, theta_15_sl, theta_55_sl,
                               theta_11_hl, theta_51_hl, theta_15_hl, theta_55_hl), dim=0)

        return {'coords': coords, 'input_fun': input_fun}, \
               {'groundtruth_values': groundtruth_values, 'groundtruth_costates': groundtruth_costates,
                'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class IntersectionHJ_Pontryagin(Dataset):
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3,
                 pretrain_iters=2000, num_src_samples=1000, seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.numcostate = 1000  
        self.num_states = 4
        self.num_vio = int(0.2 * self.numpoints)
        self.num_end = 200
        self.n_sample = self.numpoints - self.num_vio - self.num_end

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.counter_checkpoint = 10
        self.counter_epoch = 1
        self.counter_next = 0
        self.alpha = 1e-6
        self.rollout_horizon = torch.linspace(0, 3, steps=31)
        self.dt = self.rollout_horizon[1] - self.rollout_horizon[0]

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = current_dir + '/validation_scripts/train_data/intersection_param_fun_400.mat'
        self.input_fun = scipy.io.loadmat(data_path)

        # Set the seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        if self.pretrain:
            # only sample in time around the initial condition
            self.n_sample = self.numpoints - self.num_vio - self.num_end
            time = torch.ones(self.numpoints, 1) * start_time
            coords_11 = torch.cat((torch.zeros(self.num_vio, 1).uniform_(-0.638, -0.472),
                                   torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                                   torch.zeros(self.num_vio, 1).uniform_(-0.638, -0.472),
                                   torch.zeros(self.num_vio, 1).uniform_(-1, 1)), dim=1)
            coords_12 = torch.zeros(self.n_sample, self.num_states).uniform_(-1, 1)
            coords_13 = torch.cat((torch.zeros(self.num_end, 1).uniform_(-0.638, -0.472),
                                   torch.zeros(self.num_end, 1).uniform_(-1, 1),
                                   torch.zeros(self.num_end, 1).uniform_(-0.638, -0.472),
                                   torch.zeros(self.num_end, 1).uniform_(-1, 1)), dim=1)
            coords1_vn = torch.cat((coords_11, coords_12, coords_13), dim=0)
            coords2_vn = torch.cat((coords1_vn[:, 2:], coords1_vn[:, :2]), dim=1)
            coords1_vn = torch.cat((time, coords1_vn), dim=1)
            coords2_vn = torch.cat((time, coords2_vn), dim=1)

            time_cn = torch.ones(self.numcostate, 1) * start_time
            coords1_cn = torch.zeros(self.numcostate, self.num_states).uniform_(-1, 1)
            coords2_cn = torch.cat((coords1_cn[:, 2:], coords1_cn[:, :2]), dim=1)
            label1 = torch.zeros(self.numcostate, 1)
            label2 = torch.ones(self.numcostate, 1)
            coords1_cn = torch.cat((time_cn, coords1_cn, label1), dim=1)
            coords2_cn = torch.cat((time_cn, coords2_cn, label2), dim=1)

            coords1_vn = coords1_vn.repeat(4, 1)
            coords2_vn = coords2_vn.repeat(4, 1)
            coords1_cn = coords1_cn.repeat(4, 1)
            coords2_cn = coords2_cn.repeat(4, 1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            if not self.counter % self.counter_checkpoint and (self.counter + 1):
                self.numpoints = self.numpoints
                self.num_vio = int(0.2 * self.numpoints)
                self.n_sample = self.numpoints - self.num_vio - self.num_end
                print(self.numpoints)

            if not self.counter % self.counter_epoch and (self.counter + 1):
                coords_11 = torch.cat((torch.zeros(self.num_vio, 1).uniform_(-0.638, -0.472),
                                       torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                                       torch.zeros(self.num_vio, 1).uniform_(-0.638, -0.472),
                                       torch.zeros(self.num_vio, 1).uniform_(-1, 1)), dim=1)
                coords_12 = torch.zeros(self.n_sample, self.num_states).uniform_(-1, 1)
                coords_13 = torch.cat((torch.zeros(self.num_end, 1).uniform_(-0.638, -0.472),
                                       torch.zeros(self.num_end, 1).uniform_(-1, 1),
                                       torch.zeros(self.num_end, 1).uniform_(-0.638, -0.472),
                                       torch.zeros(self.num_end, 1).uniform_(-1, 1)), dim=1)
                self.coords1_vn = torch.cat((coords_11, coords_12, coords_13), dim=0)
                self.coords2_vn = torch.cat((self.coords1_vn[:, 2:], self.coords1_vn[:, :2]), dim=1)

            if not self.counter % self.counter_epoch and (self.counter + 1):
                if not self.counter % self.counter_checkpoint and (self.counter + 1):
                    self.counter_next = self.counter + self.counter_checkpoint
                    print(self.counter_next)
                else:
                    pass
                self.time_horizon = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                                    self.counter_next / self.full_count))

            time = self.time_horizon

            coords1_vn = torch.cat((time, self.coords1_vn), dim=1)
            coords2_vn = torch.cat((time, self.coords2_vn), dim=1)

            # make sure we always have training samples at the initial time
            coords1_vn[-self.N_src_samples:, 0] = start_time
            coords2_vn[-self.N_src_samples:, 0] = start_time

            time_cn = torch.ones(self.numcostate, 1) * 3
            coords1_cn = torch.zeros(self.numcostate, self.num_states).uniform_(-1, 1)
            coords2_cn = torch.cat((coords1_cn[:, 2:], coords1_cn[:, :2]), dim=1)
            label1 = torch.zeros(self.numcostate, 1)
            label2 = torch.ones(self.numcostate, 1)
            coords1_cn = torch.cat((time_cn, coords1_cn, label1), dim=1)
            coords2_cn = torch.cat((time_cn, coords2_cn, label2), dim=1)

            coords1_vn = coords1_vn.repeat(4, 1)
            coords2_vn = coords2_vn.repeat(4, 1)

            coords1_cn = coords1_cn.repeat(4, 1)
            coords2_cn = coords2_cn.repeat(4, 1)

        # set up boundary condition for value and costate
        boundary_values1_vn = self.alpha * ((coords1_vn[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                              ((coords1_vn[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values2_vn = self.alpha * ((coords2_vn[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                              ((coords2_vn[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values_vn = torch.cat((boundary_values1_vn, boundary_values2_vn), dim=0)

        boundary_values1_cn = torch.cat((self.alpha * torch.ones((4*self.numcostate, 1)),
                                         -2 * ((coords1_cn[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18),
                                         torch.zeros((4*self.numcostate, 1)),
                                         torch.zeros((4*self.numcostate, 1))), dim=1)
        boundary_values2_cn = torch.cat((torch.zeros((4*self.numcostate, 1)),
                                         torch.zeros((4*self.numcostate, 1)),
                                         self.alpha * torch.ones((4*self.numcostate, 1)),
                                         -2 * ((coords2_cn[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18)), dim=1)

        boundary_values_cn = torch.cat((boundary_values1_cn, boundary_values2_cn), dim=0)

        # set up parameter input
        theta_11 = torch.tensor(self.input_fun['theta_11'], dtype=torch.float32).clone()
        theta_15 = torch.tensor(self.input_fun['theta_15'], dtype=torch.float32).clone()
        theta_51 = torch.tensor(self.input_fun['theta_51'], dtype=torch.float32).clone()
        theta_55 = torch.tensor(self.input_fun['theta_55'], dtype=torch.float32).clone()

        theta_11_vn = theta_11.unsqueeze(0).repeat(self.numpoints, 1, 1).flatten(start_dim=1)
        theta_15_vn = theta_15.unsqueeze(0).repeat(self.numpoints, 1, 1).flatten(start_dim=1)
        theta_51_vn = theta_51.unsqueeze(0).repeat(self.numpoints, 1, 1).flatten(start_dim=1)
        theta_55_vn = theta_55.unsqueeze(0).repeat(self.numpoints, 1, 1).flatten(start_dim=1)

        input_fun = torch.cat((theta_11_vn, theta_15_vn, theta_51_vn, theta_55_vn,
                               theta_11_vn, theta_51_vn, theta_15_vn, theta_55_vn), dim=0)

        if self.pretrain:
            dirichlet_mask_vn = torch.ones(coords1_vn.shape[0], 1) > 0
            dirichlet_mask_cn = torch.ones(coords1_cn.shape[0], 1) > 0
            dirichlet_mask_cn = dirichlet_mask_cn.repeat(1, 4)
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask_vn = (coords1_vn[:, 0, None] == start_time)
            dirichlet_mask_cn = (coords1_cn[:, 0, None] == start_time).repeat(1, 4)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        coords_vn = torch.cat((coords1_vn, coords2_vn), dim=0)
        coords_cn = torch.cat((coords1_cn, coords2_cn), dim=0)

        return {'coords_vn': coords_vn,
                'coords_cn': coords_cn,
                'input_fun': input_fun}, \
               {'boundary_values_vn': boundary_values_vn,
                'boundary_values_cn': boundary_values_cn,
                'dirichlet_mask_vn': dirichlet_mask_vn,
                'dirichlet_mask_cn': dirichlet_mask_cn,
                'numcostate': self.numcostate,
                'numrollout': self.rollout_horizon.shape[0],
                'numpoints': self.numpoints,
                'counter': self.counter,
                'counter_end': self.full_count,
                'costate_gt': 0,
                'value_gt': 0,
                'num_cn': torch.tensor([0, 0, 0, 0]),
                'dt': self.dt}

