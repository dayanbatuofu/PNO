import torch
import os
import scipy.io

torch.manual_seed(0)

def initialize_pontryagin_sampling(dataset, device):
    def intersection_sampling(model_input, gt):
        costates_gt = gt['costate_gt'].squeeze(0)
        cut_index = costates_gt.shape[0] // 2
        numcostate = int(gt['numcostate'])
        state = model_input['coords_cn'].squeeze(0)
        dt = gt['dt']
        numrollout = int(gt['numrollout'])
        point_num = numcostate*numrollout

        d1 = torch.cat((((state[:point_num, 1:2] + 1) * (105 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[point_num:2*point_num, 1:2] + 1) * (105 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[2*point_num:3*point_num, 1:2] + 1) * (105 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[3*point_num:cut_index, 1:2] + 1) * (105 - 15) / 2 + 15).reshape(numrollout, numcostate).T), dim=0)
        v1 = torch.cat((((state[:point_num, 2:3] + 1) * (32 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[point_num:2*point_num, 2:3] + 1) * (32 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[2*point_num:3*point_num, 2:3] + 1) * (32 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[3*point_num:cut_index, 2:3] + 1) * (32 - 15) / 2 + 15).reshape(numrollout, numcostate).T), dim=0)
        d2 = torch.cat((((state[:point_num, 3:4] + 1) * (105 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[point_num:2*point_num, 3:4] + 1) * (105 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[2*point_num:3*point_num, 3:4] + 1) * (105 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[3*point_num:cut_index, 3:4] + 1) * (105 - 15) / 2 + 15).reshape(numrollout, numcostate).T), dim=0)
        v2 = torch.cat((((state[:point_num, 4:5] + 1) * (32 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[point_num:2*point_num, 4:5] + 1) * (32 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[2*point_num:3*point_num, 4:5] + 1) * (32 - 15) / 2 + 15).reshape(numrollout, numcostate).T,
                        ((state[3*point_num:cut_index, 4:5] + 1) * (32 - 15) / 2 + 15).reshape(numrollout, numcostate).T), dim=0)
        u1 = 0.5 * costates_gt[:cut_index, 1:2]
        u2 = 0.5 * costates_gt[cut_index:, -1:]

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        u1 = torch.cat((u1[:point_num].reshape(numrollout, numcostate).T,
                        u1[point_num:2*point_num].reshape(numrollout, numcostate).T,
                        u1[2*point_num:3*point_num].reshape(numrollout, numcostate).T,
                        u1[3*point_num:].reshape(numrollout, numcostate).T), dim=0)

        u2 = torch.cat((u2[:point_num].reshape(numrollout, numcostate).T,
                        u2[point_num:2*point_num].reshape(numrollout, numcostate).T,
                        u2[2*point_num:3*point_num].reshape(numrollout, numcostate).T,
                        u2[3*point_num:].reshape(numrollout, numcostate).T), dim=0)

        R1 = torch.tensor([70.], dtype=torch.float32).to(device)
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)
        alpha = torch.tensor([1e-6], dtype=torch.float32).to(device)
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)

        V1 = torch.zeros((4*numcostate, numrollout)).to(device)
        Loss1 = torch.zeros((4*numcostate, numrollout)).to(device)
        Loss1_tmp = torch.zeros((4*numcostate, numrollout)).to(device)
        V2 = torch.zeros((4*numcostate, numrollout)).to(device)
        Loss2 = torch.zeros((4*numcostate, numrollout)).to(device)
        Loss2_tmp = torch.zeros((4*numcostate, numrollout)).to(device)

        for i in range(numcostate):
            for j in range(numrollout):
                x1 = d1[i][j]
                x2 = d2[i][j]
                x1_in = (x1 - R1 / 2 + 1 * W2 / 2) * 5
                x1_out = -(x1 - R1 / 2 - W2 / 2 - L1) * 5
                x2_in = (x2 - R2 / 2 + 1 * W1 / 2) * 5
                x2_out = -(x2 - R2 / 2 - W1 / 2 - L2) * 5
                Loss1_tmp[i][j] = (u1[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                    x2_in) * torch.sigmoid(x2_out)) * dt
                Loss2_tmp[i][j] = (u2[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                    x2_in) * torch.sigmoid(x2_out)) * dt

        for i in range(numcostate, 2*numcostate):
            for j in range(numrollout):
                x1 = d1[i][j]
                x2 = d2[i][j]
                x1_in = (x1 - R1 / 2 + 1 * W2 / 2) * 5
                x1_out = -(x1 - R1 / 2 - W2 / 2 - L1) * 5
                x2_in = (x2 - R2 / 2 + 5 * W1 / 2) * 5
                x2_out = -(x2 - R2 / 2 - W1 / 2 - L2) * 5
                Loss1_tmp[i][j] = (u1[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                    x2_in) * torch.sigmoid(x2_out)) * dt
                Loss2_tmp[i][j] = (u2[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                    x2_in) * torch.sigmoid(x2_out)) * dt

        for i in range(2*numcostate, 3*numcostate):
            for j in range(numrollout):
                x1 = d1[i][j]
                x2 = d2[i][j]
                x1_in = (x1 - R1 / 2 + 5 * W2 / 2) * 5
                x1_out = -(x1 - R1 / 2 - W2 / 2 - L1) * 5
                x2_in = (x2 - R2 / 2 + 1 * W1 / 2) * 5
                x2_out = -(x2 - R2 / 2 - W1 / 2 - L2) * 5
                Loss1_tmp[i][j] = (u1[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                    x2_in) * torch.sigmoid(x2_out)) * dt
                Loss2_tmp[i][j] = (u2[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                    x2_in) * torch.sigmoid(x2_out)) * dt

        for i in range(3*numcostate, 4*numcostate):
            for j in range(numrollout):
                x1 = d1[i][j]
                x2 = d2[i][j]
                x1_in = (x1 - R1 / 2 + 5 * W2 / 2) * 5
                x1_out = -(x1 - R1 / 2 - W2 / 2 - L1) * 5
                x2_in = (x2 - R2 / 2 + 5 * W1 / 2) * 5
                x2_out = -(x2 - R2 / 2 - W1 / 2 - L2) * 5
                Loss1_tmp[i][j] = (u1[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                    x2_in) * torch.sigmoid(x2_out)) * dt
                Loss2_tmp[i][j] = (u2[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                    x2_in) * torch.sigmoid(x2_out)) * dt

        for i in range(4*numcostate):
            for j in range(numrollout):
                Loss1[i][j] = torch.sum(Loss1_tmp[i][j:]).to(device)
                Loss2[i][j] = torch.sum(Loss2_tmp[i][j:]).to(device)

        for i in range(4*numcostate):
            for j in range(numrollout):
                V1[i][j] = alpha * d1[i][-1] - (v1[i][-1] - 18) ** 2 - Loss1[i][j]
                V2[i][j] = alpha * d2[i][-1] - (v2[i][-1] - 18) ** 2 - Loss2[i][j]

        V1 = torch.cat((V1[:numcostate, :].T.reshape(-1, 1),
                        V1[numcostate:2*numcostate, :].T.reshape(-1, 1),
                        V1[2*numcostate:3*numcostate, :].T.reshape(-1, 1),
                        V1[3*numcostate:, :].T.reshape(-1, 1)), dim=0)
        V2 = torch.cat((V2[:numcostate, :].T.reshape(-1, 1),
                        V2[numcostate:2*numcostate, :].T.reshape(-1, 1),
                        V2[2*numcostate:3*numcostate, :].T.reshape(-1, 1),
                        V2[3*numcostate:, :].T.reshape(-1, 1)), dim=0)
        Value = torch.cat((V1, V2), dim=0).unsqueeze(0)

        return Value

    return intersection_sampling
