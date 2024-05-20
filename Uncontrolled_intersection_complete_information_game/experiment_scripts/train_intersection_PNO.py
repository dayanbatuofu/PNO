# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio
import loss_functions, modules_pno, action_functions, traj_functions, bound_functions, value_functions, sampling_functions
import training_pno

from torch.utils.data import DataLoader
import configargparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')

"""
training epoch --num_epochs:
50300 for PNO 
"""
p.add_argument('--num_epochs', type=int, default=50300,
               help='Number of epochs to train for.')

# p.add_argument('--num_epochs_warmstart', type=int, default=110000,
#                help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='tanh', required=False, choices=['tanh', 'relu', 'sine', 'gelu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=3, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--value_num_nl', type=int, default=64, required=False, help='Number of neurons per value net hidden layer.')
p.add_argument('--costate_num_nl', type=int, default=64, required=False, help='Number of neurons per costate net hidden layer.')
p.add_argument('--branch_num_nl', type=int, default=64, required=False, help='Number of neurons per branch net hidden layer.')

"""
training epoch ---pretrain_iters: 
50000 for PNO 
"""
p.add_argument('--pretrain_iters', type=int, default=50000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')

"""
training epoch --counter_end:
300 for PNO 
"""
p.add_argument('--counter_end', type=int, default=300, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')  # 10000

# p.add_argument('--counter_end_warmstart', type=int, default=100000, required=False, help='Defines the linear step for curriculum training starting from the initial time')

p.add_argument('--collisionR', type=float, default=0.25, required=False, help='Collision radius between vehicles')
p.add_argument('--minWith', type=str, default='target', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=True, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets

source_coords = [0., 0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload
  # opt.counter_start = 0

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs

Weight = (10, 1)

HJ_Pontryagin_dataset = dataio.IntersectionHJ_Pontryagin(numpoints=30000,
                                                         pretrain=opt.pretrain, tMin=opt.tMin,
                                                         tMax=opt.tMax, counter_start=opt.counter_start,
                                                         counter_end=opt.counter_end,
                                                         pretrain_iters=opt.pretrain_iters, seed=opt.seed,
                                                         num_src_samples=opt.num_src_samples)

HJ_Pontryagin_dataloader = DataLoader(HJ_Pontryagin_dataset, shuffle=True, batch_size=opt.batch_size,
                                      pin_memory=True, num_workers=0)

model = modules_pno.SingleBVPNet(value_in_features=5, value_out_features=64, costate_in_features=6,
                                 costate_out_features=4, branch_in_features=400, branch_out_features=64,
                                 type=opt.model, mode=opt.mode, value_hidden_features=opt.value_num_nl,
                                 costate_hidden_features=opt.costate_num_nl, branch_hidden_features=opt.branch_num_nl,
                                 num_hidden_layers=opt.num_hl, final_layer_factor=1.)

model.to(device)

loss_fn_pontryagin = loss_functions.initialize_HJ_Pontryagin(HJ_Pontryagin_dataset, Weight, device)

action_fn_pontryagin = action_functions.initialize_pontryagin_sampling(HJ_Pontryagin_dataset, device)

traj_fn_pontryagin = traj_functions.initialize_pontryagin_sampling(HJ_Pontryagin_dataset, device)

bound_fn_pontryagin = bound_functions.initialize_pontryagin_sampling(HJ_Pontryagin_dataset, device)

value_fn_pontryagin = value_functions.initialize_pontryagin_sampling(HJ_Pontryagin_dataset, device)

sampling_fn_pontryagin = sampling_functions.initialize_pontryagin_sampling(HJ_Pontryagin_dataset, device)

path = 'experiment_HJ_Pontryagin_' + opt.model + '_pno/'
root_path = os.path.join(opt.logging_root, path)

training_pno.train(model=model, train_dataloader=HJ_Pontryagin_dataloader, epochs=opt.num_epochs,
                   lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn_pontryagin, action_fn=action_fn_pontryagin,
                   traj_fn=traj_fn_pontryagin, bound_fn=bound_fn_pontryagin, value_fn=value_fn_pontryagin,
                   sampling_fn=sampling_fn_pontryagin, clip_grad=opt.clip_grad, use_lbfgs=opt.use_lbfgs,
                   validation_fn=None, start_epoch=opt.checkpoint_toload)

