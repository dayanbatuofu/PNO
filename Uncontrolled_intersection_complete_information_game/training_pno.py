'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import random
from examples.choose_problem_intersection import system, problem, config
from scipy.integrate import solve_ivp
import scipy.io as scio
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, action_fn,
          traj_fn, bound_fn, value_fn, sampling_fn, summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False,
          use_lbfgs=False, loss_schedules=None, validation_fn=None, start_epoch=0):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.9, verbose=True, min_lr=1e-6,
                                                           patience=5000)

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.9, verbose=True, min_lr=1e-6,
                                                               patience=5000)

    # Load the checkpoint if required
    if start_epoch > 0:
        # Load the model and start training from that point onwards
        model_path = os.path.join(model_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.train()
        optim.load_state_dict(checkpoint['optimizer'])
        optim.param_groups[0]['lr'] = lr
        assert(start_epoch == checkpoint['epoch'])
    else:
        # Start training from scratch
        if os.path.exists(model_dir):
            val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
            if val == 'y':
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs - start_epoch) as pbar:
        train_losses = []
        bcs_losses_hji = []
        bcs_losses_costate = []
        losses_diff_vn = []
        losses_diff_cn = []
        values_diff = []
        HJI_weight = []
        LR = []
        costate_data = dict()
        costate_data.update({'coords_cn': 0,
                             'coords_vn': 0,
                             'costate_gt': 0,
                             'value_gt': 0,
                             'boundary_values_cn': 0,
                             'dirichlet_mask_cn': 0,
                             'input_fun': 0,
                             'num_cn': 0})

        for epoch in range(start_epoch, epochs):
            if not (epoch - 50000) % 50 and epoch and epoch >= 50000:
                # Saving the optimizer state is important to produce consistent results
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict()}
                torch.save(checkpoint,
                       os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'bcs_losses_hji_epoch_%04d.txt' % epoch),
                           np.array(bcs_losses_hji))
                np.savetxt(os.path.join(checkpoints_dir, 'bcs_losses_costate_epoch_%04d.txt' % epoch),
                           np.array(bcs_losses_costate))
                np.savetxt(os.path.join(checkpoints_dir, 'losses_diff_vn_epoch_%04d.txt' % epoch),
                           np.array(losses_diff_vn))
                np.savetxt(os.path.join(checkpoints_dir, 'losses_diff_cn_epoch_%04d.txt' % epoch),
                           np.array(losses_diff_cn))
                np.savetxt(os.path.join(checkpoints_dir, 'values_diff_epoch_%04d.txt' % epoch),
                           np.array(values_diff))
                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch)

            # self-supervised learning
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.to(device) for key, value in model_input.items()}
                gt = {key: value.to(device) for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                counter = int(gt['counter'])
                counter_end = int(gt['counter_end'])

                if counter == 0:
                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        if loss_name == 'weight':
                            if loss == 1:
                                hji_weight = 0
                            else:
                                hji_weight = loss
                            continue
                        if loss_name == 'dirichlet_vn':
                            bcs_loss_hji = loss.mean()
                            single_loss = bcs_loss_hji
                        if loss_name == 'dirichlet_cn':
                            bcs_loss_costate = loss.mean()
                            single_loss = bcs_loss_costate
                        if loss_name == 'costate_difference_vn':
                            loss_diff_vn = loss.mean()
                            single_loss = loss_diff_vn
                        if loss_name == 'costate_difference_cn':
                            loss_diff_cn = loss.mean()
                            single_loss = loss_diff_cn
                        if loss_name == 'value_difference':
                            value_diff = loss.mean()
                            single_loss = value_diff
                        else:
                            single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps),
                                              total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    bcs_losses_hji.append(bcs_loss_hji.item())
                    bcs_losses_costate.append(bcs_loss_costate.item())
                    losses_diff_vn.append(loss_diff_vn.item())
                    losses_diff_cn.append(loss_diff_cn.item())
                    values_diff.append(value_diff.item())
                    HJI_weight.append(hji_weight)
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                        optim.step()

                    scheduler.step(train_loss)

                    lr_scheduler = optim.state_dict()['param_groups'][0]['lr']
                    LR.append(lr_scheduler)

                else:
                    numrollout = int(gt['numrollout']) - 1
                    numcostate = int(gt['numcostate'])
                    numpoints = int(gt['numpoints'])
                    N_sample = numcostate*(numrollout + 1)

                    if counter == counter_end:
                        model_input['coords_cn'] = costate_data['coords_cn']
                        model_input['coords_vn'] = costate_data['coords_vn']
                        model_input['input_fun'] = costate_data['input_fun']
                        gt['costate_gt'] = costate_data['costate_gt']
                        gt['value_gt'] = costate_data['value_gt']
                        gt['num_cn'] = costate_data['num_cn']

                    elif not counter % 10 or counter == 1:

                        # dynamical sampling for initial state applied to costate net
                        if counter == 1:
                            coords_data_save = {'coords': model_input['coords_cn'].squeeze().detach().cpu().numpy()}
                            save_path = 'cn_data_sine/coords_cn_data_' + str(counter) + '.mat'
                            scio.savemat(save_path, coords_data_save)

                        if not counter == 1:
                            coords_vn_tmp = model_input['coords_vn']
                            input_fun_tmp = model_input['input_fun']
                            model_input['coords_vn'] = costate_data['coords_vn']
                            model_input['coords_cn'] = costate_data['coords_cn']
                            model_input['input_fun'] = costate_data['input_fun']
                            gt['num_cn'] = costate_data['num_cn']
                            model_output = model(model_input)
                            model_input['coords_cn'] = sampling_fn(model_output, gt, counter)['coords_cn']
                            model_input['coords_vn'] = coords_vn_tmp
                            model_input['input_fun'] = input_fun_tmp

                        coords1_update_11 = model_input['coords_cn'][:, :numcostate, :].squeeze(0)
                        coords1_update_15 = model_input['coords_cn'][:, numcostate:2*numcostate, :].squeeze(0)
                        coords1_update_51 = model_input['coords_cn'][:, 2*numcostate:3*numcostate, :].squeeze(0)
                        coords1_update_55 = model_input['coords_cn'][:, 3*numcostate:4*numcostate, :].squeeze(0)
                        coords2_update_11 = model_input['coords_cn'][:, 4*numcostate:5*numcostate, :].squeeze(0)
                        coords2_update_15 = model_input['coords_cn'][:, 5*numcostate:6*numcostate, :].squeeze(0)
                        coords2_update_51 = model_input['coords_cn'][:, 6*numcostate:7*numcostate, :].squeeze(0)
                        coords2_update_55 = model_input['coords_cn'][:, 7*numcostate:, :].squeeze(0)

                        coords1_mask_update_11 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
                        coords1_mask_update_15 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
                        coords1_mask_update_51 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
                        coords1_mask_update_55 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
                        coords2_mask_update_11 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
                        coords2_mask_update_15 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
                        coords2_mask_update_51 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
                        coords2_mask_update_55 = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)

                        for num in range(numrollout):  # generate closed-loop trajectories at t=3.0s
                            model_output = model(model_input)
                            action = action_fn(model_output)
                            with torch.no_grad():
                                coords_new = traj_fn(model_output, gt, action, num)
                                model_input['coords_cn'] = coords_new['coords_cn']
                                coords1_next_11 = coords_new['coords_cn'][:, :numcostate, :].squeeze(0)
                                coords1_next_15 = coords_new['coords_cn'][:, numcostate:2*numcostate, :].squeeze(0)
                                coords1_next_51 = coords_new['coords_cn'][:, 2*numcostate:3*numcostate, :].squeeze(0)
                                coords1_next_55 = coords_new['coords_cn'][:, 3*numcostate:4*numcostate, :].squeeze(0)
                                coords2_next_11 = coords_new['coords_cn'][:, 4*numcostate:5*numcostate, :].squeeze(0)
                                coords2_next_15 = coords_new['coords_cn'][:, 5*numcostate:6*numcostate, :].squeeze(0)
                                coords2_next_51 = coords_new['coords_cn'][:, 6*numcostate:7*numcostate, :].squeeze(0)
                                coords2_next_55 = coords_new['coords_cn'][:, 7*numcostate:, :].squeeze(0)
                                coords1_update_11 = torch.cat((coords1_update_11, coords1_next_11), dim=0)
                                coords1_update_15 = torch.cat((coords1_update_15, coords1_next_15), dim=0)
                                coords1_update_51 = torch.cat((coords1_update_51, coords1_next_51), dim=0)
                                coords1_update_55 = torch.cat((coords1_update_55, coords1_next_55), dim=0)
                                coords2_update_11 = torch.cat((coords2_update_11, coords2_next_11), dim=0)
                                coords2_update_15 = torch.cat((coords2_update_15, coords2_next_15), dim=0)
                                coords2_update_51 = torch.cat((coords2_update_51, coords2_next_51), dim=0)
                                coords2_update_55 = torch.cat((coords2_update_55, coords2_next_55), dim=0)
                                
                                coords1_mask_next_11 = coords_new['coords_mask'][:, :numcostate, :]
                                coords1_mask_next_15 = coords_new['coords_mask'][:, numcostate:2*numcostate, :]
                                coords1_mask_next_51 = coords_new['coords_mask'][:, 2*numcostate:3*numcostate, :]
                                coords1_mask_next_55 = coords_new['coords_mask'][:, 3*numcostate:4*numcostate, :]
                                coords2_mask_next_11 = coords_new['coords_mask'][:, 4*numcostate:5*numcostate, :]
                                coords2_mask_next_15 = coords_new['coords_mask'][:, 5*numcostate:6*numcostate, :]
                                coords2_mask_next_51 = coords_new['coords_mask'][:, 6*numcostate:7*numcostate, :]
                                coords2_mask_next_55 = coords_new['coords_mask'][:, 7*numcostate:, :]
                                coords1_mask_update_11 = torch.cat((coords1_mask_update_11, coords1_mask_next_11), dim=1)
                                coords1_mask_update_15 = torch.cat((coords1_mask_update_15, coords1_mask_next_15), dim=1)
                                coords1_mask_update_51 = torch.cat((coords1_mask_update_51, coords1_mask_next_51), dim=1)
                                coords1_mask_update_55 = torch.cat((coords1_mask_update_55, coords1_mask_next_55), dim=1)
                                coords2_mask_update_11 = torch.cat((coords2_mask_update_11, coords2_mask_next_11), dim=1)
                                coords2_mask_update_15 = torch.cat((coords2_mask_update_15, coords2_mask_next_15), dim=1)
                                coords2_mask_update_51 = torch.cat((coords2_mask_update_51, coords2_mask_next_51), dim=1)
                                coords2_mask_update_55 = torch.cat((coords2_mask_update_55, coords2_mask_next_55), dim=1)
                            # gc.collect()
                            # torch.cuda.empty_cache()

                        # add boundary points
                        coords1_cn = torch.cat((coords1_update_11,
                                                coords1_update_15,
                                                coords1_update_51,
                                                coords1_update_55), dim=0)
                        coords2_cn = torch.cat((coords2_update_11,
                                                coords2_update_15,
                                                coords2_update_15,
                                                coords2_update_15), dim=0)
                        coords_cn = torch.cat((coords1_cn, coords2_cn), dim=0)

                        coords1_mask = torch.cat((coords1_mask_update_11,
                                                  coords1_mask_update_15,
                                                  coords1_mask_update_51,
                                                  coords1_mask_update_55), dim=1)
                        coords2_mask = torch.cat((coords2_mask_update_11,
                                                  coords2_mask_update_15,
                                                  coords2_mask_update_51,
                                                  coords2_mask_update_55), dim=1)
                        coords_mask = torch.cat((coords1_mask, coords2_mask), dim=1)

                        # record the remaining state for each type
                        num_11 = coords1_mask_update_11.sum().unsqueeze(0)
                        num_15 = coords1_mask_update_15.sum().unsqueeze(0)
                        num_51 = coords1_mask_update_51.sum().unsqueeze(0)
                        num_55 = coords1_mask_update_55.sum().unsqueeze(0)

                        print('remaining data:', int(num_11), int(num_15), int(num_51), int(num_55))

                        gt['num_cn'] = torch.cat((num_11, num_15, num_51, num_55), dim=0)
                        gt_update = bound_fn(coords1_cn, coords2_cn)

                        costate = gt_update['boundary_values_cn']
                        cn_index = costate.shape[0] // 2
                        costate1 = costate[:cn_index]
                        costate2 = costate[cn_index:]
                        state = coords_cn[:cn_index, 1:-1]
                        dirichlet_mask_cn = gt_update['dirichlet_mask_cn']

                        X_pred_T = state[dirichlet_mask_cn].detach().cpu().numpy().reshape(4*numcostate, 4).T
                        A1_pred_T = costate1[dirichlet_mask_cn].detach().cpu().numpy().reshape(4*numcostate, 4).T
                        A2_pred_T = costate2[dirichlet_mask_cn].detach().cpu().numpy().reshape(4*numcostate, 4).T
                        A_pred_T = np.vstack((A1_pred_T, A2_pred_T))
                        data_ode = {}
                        start_time = time.time()

                        """
                        use for trajectory verification
                        """
                        # p_num = numcostate*(numrollout+1)
                        # d1_tmp = torch.cat((((state[:p_num, 0:1] + 1) * (105 - 15) / 2 + 15).reshape(31, numcostate).T,
                        #                    ((state[p_num:2*p_num, 0:1] + 1) * (105 - 15) / 2 + 15).reshape(31,numcostate).T,
                        #                    ((state[2*p_num:3*p_num, 0:1] + 1) * (105 - 15) / 2 + 15).reshape(31, numcostate).T,
                        #                    ((state[3*p_num:, 0:1] + 1) * (105 - 15) / 2 + 15).reshape(31, numcostate).T), dim=0)
                        # d2_tmp = torch.cat((((state[:p_num, 2:3] + 1) * (105 - 15) / 2 + 15).reshape(31, numcostate).T,
                        #                    ((state[p_num:2*p_num, 2:3] + 1) * (105 - 15) / 2 + 15).reshape(31,numcostate).T,
                        #                    ((state[2*p_num:3*p_num, 2:3] + 1) * (105 - 15) / 2 + 15).reshape(31, numcostate).T,
                        #                    ((state[3*p_num:, 2:3] + 1) * (105 - 15) / 2 + 15).reshape(31, numcostate).T), dim=0)
                        # v1_tmp = torch.cat((((state[:p_num, 1:2] + 1) * (32 - 15) / 2 + 15).reshape(31, numcostate).T,
                        #                    ((state[p_num:2*p_num, 1:2] + 1) * (32 - 15) / 2 + 15).reshape(31,numcostate).T,
                        #                    ((state[2*p_num:3*p_num, 1:2] + 1) * (32 - 15) / 2 + 15).reshape(31, numcostate).T,
                        #                    ((state[3*p_num:, 1:2] + 1) * (32 - 15) / 2 + 15).reshape(31, numcostate).T), dim=0)
                        # v2_tmp = torch.cat((((state[:p_num, 3:4] + 1) * (32 - 15) / 2 + 15).reshape(31, numcostate).T,
                        #                    ((state[p_num:2*p_num, 3:4] + 1) * (32 - 15) / 2 + 15).reshape(31,numcostate).T,
                        #                    ((state[2*p_num:3*p_num, 3:4] + 1) * (32 - 15) / 2 + 15).reshape(31, numcostate).T,
                        #                    ((state[3*p_num:, 3:4] + 1) * (32 - 15) / 2 + 15).reshape(31, numcostate).T), dim=0)
                        # d1_tmp = d1_tmp.detach().cpu().numpy()
                        # d2_tmp = d2_tmp.detach().cpu().numpy()
                        # v1_tmp = v1_tmp.detach().cpu().numpy()
                        # v2_tmp = v2_tmp.detach().cpu().numpy()

                        for idx in range(0, 4*numcostate):
                            if 0 <= idx < numcostate:
                                theta1, theta2 = 1, 1
                            elif numcostate <= idx < 2*numcostate:
                                theta1, theta2 = 1, 5
                            elif 2*numcostate <= idx < 3*numcostate:
                                theta1, theta2 = 5, 1
                            else:
                                theta1, theta2 = 5, 5
                            d1 = (X_pred_T[0, idx] + 1) * (105 - 15) / 2 + 15
                            v1 = (X_pred_T[1, idx] + 1) * (32 - 15) / 2 + 15
                            d2 = (X_pred_T[2, idx] + 1) * (105 - 15) / 2 + 15
                            v2 = (X_pred_T[3, idx] + 1) * (32 - 15) / 2 + 15
                            xT = np.vstack((d1, v1, d2, v2)).reshape(-1, 1)
                            yT = A_pred_T[:, idx].reshape(-1, 1)

                            X_aug = np.vstack((xT, yT)).reshape(-1)
                            t_eval = np.linspace(0.0, 3.0, num=numrollout+1)
                            t_span = np.array([t_eval[-1], 0.0])
                            t_eval = t_eval[::-1]

                            # solve final value problem
                            SOL = solve_ivp(problem.v_dynamics, t_span, X_aug, method='RK45',
                                            t_eval=t_eval, args=(model, theta1, theta2), rtol=1e-03)
                            # SOL = solve_ivp(problem.v_dynamics, t_span, X_aug, method='DOP853',
                            #                 t_eval=t_eval, args=(model, theta1, theta2), rtol=1e-03)
                            A_ivp = SOL.y
                            t_ivp = SOL.t.reshape(1, -1)
                            if 'A' in data_ode.keys():
                                data_ode['A'] = np.hstack((data_ode['A'], np.flip(A_ivp[4:], axis=1)))
                                data_ode['t'] = np.hstack((data_ode['t'], np.flip(t_ivp, axis=1)))
                                data_ode['X'] = np.hstack((data_ode['X'], np.flip(A_ivp[:4], axis=1)))
                            else:
                                data_ode['A'] = np.flip(A_ivp[4:], axis=1)
                                data_ode['t'] = np.flip(t_ivp, axis=1)
                                data_ode['X'] = np.flip(A_ivp[:4], axis=1)

                        """
                        use for trajectory verification
                        """
                        # d1_test = data_ode['X'][0, :].reshape(4*numcostate, 31)
                        # v1_test = data_ode['X'][1, :].reshape(4*numcostate, 31)
                        # d2_test = data_ode['X'][2, :].reshape(4*numcostate, 31)
                        # v2_test = data_ode['X'][3, :].reshape(4*numcostate, 31)

                        final_time = time.time() - start_time
                        print('ode cost time:', final_time)

                        data_ode.update({'t0': data_ode['t']})
                        idx0 = np.nonzero(np.equal(data_ode.pop('t0'), 0))[1]
                        A1_11 = np.empty((0, 4))
                        A1_15 = np.empty((0, 4))
                        A1_51 = np.empty((0, 4))
                        A1_55 = np.empty((0, 4))
                        A2_11 = np.empty((0, 4))
                        A2_15 = np.empty((0, 4))
                        A2_51 = np.empty((0, 4))
                        A2_55 = np.empty((0, 4))
                        for idx in range(numrollout+1):
                            A1_11 = np.vstack((A1_11, data_ode['A'][:4, idx0 + idx][:, :numcostate].T))
                            A1_15 = np.vstack((A1_15, data_ode['A'][:4, idx0 + idx][:, numcostate:2*numcostate].T))
                            A1_51 = np.vstack((A1_51, data_ode['A'][:4, idx0 + idx][:, 2*numcostate:3*numcostate].T))
                            A1_55 = np.vstack((A1_55, data_ode['A'][:4, idx0 + idx][:, 3*numcostate:].T))
                            A2_11 = np.vstack((A2_11, data_ode['A'][4:, idx0 + idx][:, :numcostate].T))
                            A2_15 = np.vstack((A2_15, data_ode['A'][4:, idx0 + idx][:, numcostate:2*numcostate].T))
                            A2_51 = np.vstack((A2_51, data_ode['A'][4:, idx0 + idx][:, 2*numcostate:3*numcostate].T))
                            A2_55 = np.vstack((A2_55, data_ode['A'][4:, idx0 + idx][:, 3*numcostate:].T))

                        A1 = torch.tensor(np.vstack((A1_11, A1_15, A1_51, A1_55)), dtype=torch.float32).unsqueeze(0).to(device)
                        A2 = torch.tensor(np.vstack((A2_11, A2_15, A2_51, A2_55)), dtype=torch.float32).unsqueeze(0).to(device)

                        gt['costate_gt'] = torch.cat((A1, A2), dim=1)
                        model_input['coords_cn'] = coords_cn.unsqueeze(0)
                        V_cn = value_fn(model_input, gt)
                        gt['value_gt'] = torch.cat((V_cn[:, :4*N_sample, :],
                                                    V_cn[:, 4*N_sample:, :]), dim=1)

                        # remove the state beyond the space
                        N_shape = int(2*(num_11+num_15+num_51+num_55))
                        model_input['coords_cn'] = model_input['coords_cn'][torch.cat([coords_mask]*6, dim=2)].reshape(N_shape, 6).unsqueeze(0)
                        gt['costate_gt'] = gt['costate_gt'][torch.cat([coords_mask]*4, dim=2)].reshape(N_shape, 4).unsqueeze(0)
                        gt['value_gt'] = gt['value_gt'][coords_mask].reshape(N_shape, 1).unsqueeze(0)

                        inputfun1_add_11 = model_input['input_fun'][:, 0, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun1_add_15 = model_input['input_fun'][:, numpoints, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun1_add_51 = model_input['input_fun'][:, 2*numpoints, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun1_add_55 = model_input['input_fun'][:, 3*numpoints, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun2_add_11 = model_input['input_fun'][:, 4*numpoints, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun2_add_51 = model_input['input_fun'][:, 5*numpoints, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun2_add_15 = model_input['input_fun'][:, 6*numpoints, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun2_add_55 = model_input['input_fun'][:, 7*numpoints, :].repeat(1, N_sample, 1).squeeze(0)
                        inputfun1_add = torch.cat((inputfun1_add_11,
                                                   inputfun1_add_15,
                                                   inputfun1_add_51,
                                                   inputfun1_add_55), dim=0)
                        inputfun2_add = torch.cat((inputfun2_add_11,
                                                   inputfun2_add_51,
                                                   inputfun2_add_15,
                                                   inputfun2_add_55), dim=0)

                        cut_index = coords_mask.shape[1]//2
                        input_mask1 = torch.cat([coords_mask]*400, dim=2)[:, :cut_index, :]
                        input_mask2 = torch.cat([coords_mask]*400, dim=2)[:, cut_index:, :]
                        inputfun1_add = inputfun1_add.unsqueeze(0)[input_mask1].reshape(N_shape//2, 400)
                        inputfun2_add = inputfun2_add.unsqueeze(0)[input_mask2].reshape(N_shape//2, 400)

                        inputfun1_pre = model_input['input_fun'][:, :4*numpoints, :].squeeze(0)
                        inputfun2_pre = model_input['input_fun'][:, 4*numpoints:, :].squeeze(0)
                        inputfun_vn = torch.cat((inputfun1_add, inputfun1_pre,
                                                 inputfun2_add, inputfun2_pre), dim=0)
                        model_input['input_fun'] = inputfun_vn.unsqueeze(0)

                        N_sample_new = model_input['coords_cn'].shape[1] // 2

                        coords1_cn = model_input['coords_cn'][:, :N_sample_new, :-1].squeeze(0)
                        coords2_cn = model_input['coords_cn'][:, N_sample_new:, :-1].squeeze(0)
                        coords1_vn = model_input['coords_vn'][:, :4*numpoints, :].squeeze(0)
                        coords2_vn = model_input['coords_vn'][:, 4*numpoints:, :].squeeze(0)
                        coords_vn = torch.cat((coords1_cn, coords1_vn,
                                               coords2_cn, coords2_vn), dim=0)
                        model_input['coords_vn'] = coords_vn.unsqueeze(0)

                        costate_data['coords_cn'] = model_input['coords_cn']
                        costate_data['coords_vn'] = model_input['coords_vn']
                        costate_data['input_fun'] = model_input['input_fun']
                        costate_data['costate_gt'] = gt['costate_gt']
                        costate_data['value_gt'] = gt['value_gt']
                        costate_data['num_cn'] = gt['num_cn']

                        # don't need this term
                        # costate_data['boundary_values_cn'] = gt['boundary_values_cn']
                        # costate_data['dirichlet_mask_cn'] = gt['dirichlet_mask_cn']
                    else:
                        model_input['coords_cn'] = costate_data['coords_cn']
                        model_input['coords_vn'] = costate_data['coords_vn']
                        model_input['input_fun'] = costate_data['input_fun']
                        gt['costate_gt'] = costate_data['costate_gt']
                        gt['value_gt'] = costate_data['value_gt']
                        gt['num_cn'] = costate_data['num_cn']

                    num_gradient = 3000  # 5000

                    for _ in range(num_gradient):
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)

                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            if loss_name == 'weight':
                                if loss == 1:
                                    hji_weight = 0
                                else:
                                    hji_weight = loss
                                continue
                            if loss_name == 'dirichlet_vn':
                                bcs_loss_hji = loss.mean()
                                single_loss = bcs_loss_hji
                            if loss_name == 'dirichlet_cn':
                                bcs_loss_costate = loss.mean()
                                single_loss = bcs_loss_costate
                            if loss_name == 'costate_difference_vn':
                                loss_diff_vn = loss.mean()
                                single_loss = loss_diff_vn
                            if loss_name == 'costate_difference_cn':
                                loss_diff_cn = loss.mean()
                                single_loss = loss_diff_cn
                            if loss_name == 'value_difference':
                                value_diff = loss.mean()
                                single_loss = value_diff
                            else:
                                single_loss = loss.mean()

                            if loss_schedules is not None and loss_name in loss_schedules:
                                writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps),
                                                  total_steps)
                                single_loss *= loss_schedules[loss_name](total_steps)

                            writer.add_scalar(loss_name, single_loss, total_steps)
                            train_loss += single_loss

                        train_losses.append(train_loss.item())
                        bcs_losses_hji.append(bcs_loss_hji.item())
                        bcs_losses_costate.append(bcs_loss_costate.item())
                        losses_diff_vn.append(loss_diff_vn.item())
                        losses_diff_cn.append(loss_diff_cn.item())
                        values_diff.append(value_diff.item())
                        HJI_weight.append(hji_weight)
                        writer.add_scalar("total_train_loss", train_loss, total_steps)

                        if not total_steps % steps_til_summary:
                            torch.save(model.state_dict(),
                                       os.path.join(checkpoints_dir, 'model_current.pth'))
                            # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                        if not use_lbfgs:
                            optim.zero_grad()
                            train_loss.backward()

                            if clip_grad:
                                if isinstance(clip_grad, bool):
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                            optim.step()

                        scheduler.step(train_loss)

                        lr_scheduler = optim.state_dict()['param_groups'][0]['lr']
                        LR.append(lr_scheduler)

            pbar.update(1)

            if counter == 0:
                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.3f, bcs loss hji %0.2f, bcs loss costate %0.2f, loss diff_cn %0.2f, hji weight %0.2f, lr %0.6f"
                                % (epoch, train_loss, bcs_loss_hji, bcs_loss_costate, loss_diff_cn, hji_weight, lr_scheduler))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

            else:
                if not total_steps % 10:
                    tqdm.write("Epoch %d, Total loss %0.3f, bcs loss hji %0.2f, bcs loss costate %0.2f, loss diff_cn %0.2f, hji weight %0.2f, lr %0.6f"
                        % (epoch, train_loss, bcs_loss_hji, bcs_loss_costate, loss_diff_cn, hji_weight, lr_scheduler))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

            total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'bcs_losses_hji_final.txt'),
                   np.array(bcs_losses_hji))
        np.savetxt(os.path.join(checkpoints_dir, 'bcs_losses_costate_final.txt'),
                   np.array(bcs_losses_costate))
        np.savetxt(os.path.join(checkpoints_dir, 'losses_diff_vn_final.txt'),
                   np.array(losses_diff_vn))
        np.savetxt(os.path.join(checkpoints_dir, 'losses_diff_cn_final.txt'),
                   np.array(losses_diff_cn))
        np.savetxt(os.path.join(checkpoints_dir, 'values_diff_final.txt'),
                   np.array(values_diff))
        np.savetxt(os.path.join(checkpoints_dir, 'hji_weight_final.txt'),
                   np.array(HJI_weight))
        np.savetxt(os.path.join(checkpoints_dir, 'learning rate.txt'),
                   np.array(LR))

class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
