import torch
import diff_operators
import scipy.io as scio

torch.manual_seed(0)

def initialize_pontryagin_sampling(dataset, device):
    def intersection_sampling(model_output, gt, counter):
        x = model_output['model_in_vn']
        y = model_output['model_out_vn']
        costate_pred = model_output['model_out_cn']
        cn_index = costate_pred.shape[1] // 2
        cut_index = x.shape[1] // 2
        num_x0 = gt['numcostate']
        cn_11_idx = int(gt['num_cn'][0])
        cn_15_idx = int(gt['num_cn'][0]) + int(gt['num_cn'][1])
        cn_51_idx = int(gt['num_cn'][0]) + int(gt['num_cn'][1]) + int(gt['num_cn'][2])

        x1 = x[:, :cut_index]
        x2 = x[:, cut_index:]
        x1_cn = x1[:, :cn_index]
        x2_cn = x2[:, :cn_index]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)

        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1_whole = dv_1[..., 0, 0].squeeze()[:cn_index]
        dvdt1_11 = dvdt_1_whole[:cn_11_idx][:num_x0]
        dvdt1_15 = dvdt_1_whole[cn_11_idx:cn_15_idx][:num_x0]
        dvdt1_51 = dvdt_1_whole[cn_15_idx:cn_51_idx][:num_x0]
        dvdt1_55 = dvdt_1_whole[cn_51_idx:][:num_x0]
        dvdt_1 = torch.cat((dvdt1_11, dvdt1_15, dvdt1_51, dvdt1_55), dim=0)
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1_whole = dvdx_1[:cn_index, :1] / ((105 - 15) / 2)  # lamda_11
        lam11_2_whole = dvdx_1[:cn_index, 1:2] / ((32 - 15) / 2)  # lamda_11
        lam12_1_whole = dvdx_1[:cn_index, 2:3] / ((105 - 15) / 2)  # lamda_12
        lam12_2_whole = dvdx_1[:cn_index, 3:4] / ((32 - 15) / 2)  # lamda_12

        lam11_1_11 = lam11_1_whole[:cn_11_idx, :][:num_x0, :]
        lam11_1_15 = lam11_1_whole[cn_11_idx:cn_15_idx, :][:num_x0, :]
        lam11_1_51 = lam11_1_whole[cn_15_idx:cn_51_idx, :][:num_x0, :]
        lam11_1_55 = lam11_1_whole[cn_51_idx:, :][:num_x0, :]
        lam11_2_11 = lam11_2_whole[:cn_11_idx, :][:num_x0, :]
        lam11_2_15 = lam11_2_whole[cn_11_idx:cn_15_idx, :][:num_x0, :]
        lam11_2_51 = lam11_2_whole[cn_15_idx:cn_51_idx, :][:num_x0, :]
        lam11_2_55 = lam11_2_whole[cn_51_idx:, :][:num_x0, :]
        lam12_1_11 = lam12_1_whole[:cn_11_idx, :][:num_x0, :]
        lam12_1_15 = lam12_1_whole[cn_11_idx:cn_15_idx, :][:num_x0, :]
        lam12_1_51 = lam12_1_whole[cn_15_idx:cn_51_idx, :][:num_x0, :]
        lam12_1_55 = lam12_1_whole[cn_51_idx:, :][:num_x0, :]
        lam12_2_11 = lam12_2_whole[:cn_11_idx, :][:num_x0, :]
        lam12_2_15 = lam12_2_whole[cn_11_idx:cn_15_idx, :][:num_x0, :]
        lam12_2_51 = lam12_2_whole[cn_15_idx:cn_51_idx, :][:num_x0, :]
        lam12_2_55 = lam12_2_whole[cn_51_idx:, :][:num_x0, :]
        lam11_1 = torch.cat((lam11_1_11, lam11_1_15, lam11_1_51, lam11_1_55), dim=0)
        lam11_2 = torch.cat((lam11_2_11, lam11_2_15, lam11_2_51, lam11_2_55), dim=0)
        lam12_1 = torch.cat((lam12_1_11, lam12_1_15, lam12_1_51, lam12_1_55), dim=0)
        lam12_2 = torch.cat((lam12_2_11, lam12_2_15, lam12_2_51, lam12_2_55), dim=0)

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2_whole = dv_2[..., 0, 0].squeeze()[:cn_index]
        dvdt2_11 = dvdt_2_whole[:cn_11_idx][:num_x0]
        dvdt2_15 = dvdt_2_whole[cn_11_idx:cn_15_idx][:num_x0]
        dvdt2_51 = dvdt_2_whole[cn_15_idx:cn_51_idx][:num_x0]
        dvdt2_55 = dvdt_2_whole[cn_51_idx:][:num_x0]
        dvdt_2 = torch.cat((dvdt2_11, dvdt2_15, dvdt2_51, dvdt2_55), dim=0)
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1_whole = dvdx_2[:cn_index, 2:3] / ((105 - 15) / 2)  # lamda_21
        lam21_2_whole = dvdx_2[:cn_index, 3:4] / ((32 - 15) / 2)  # lamda_21
        lam22_1_whole = dvdx_2[:cn_index, :1] / ((105 - 15) / 2)  # lamda_22
        lam22_2_whole = dvdx_2[:cn_index, 1:2] / ((32 - 15) / 2)  # lamda_22

        lam21_1_11 = lam21_1_whole[:cn_11_idx, :][:num_x0, :]
        lam21_1_15 = lam21_1_whole[cn_11_idx:cn_15_idx, :][:num_x0, :]
        lam21_1_51 = lam21_1_whole[cn_15_idx:cn_51_idx, :][:num_x0, :]
        lam21_1_55 = lam21_1_whole[cn_51_idx:, :][:num_x0, :]
        lam21_2_11 = lam21_2_whole[:cn_11_idx, :][:num_x0, :]
        lam21_2_15 = lam21_2_whole[cn_11_idx:cn_15_idx, :][:num_x0, :]
        lam21_2_51 = lam21_2_whole[cn_15_idx:cn_51_idx, :][:num_x0, :]
        lam21_2_55 = lam21_2_whole[cn_51_idx:, :][:num_x0, :]
        lam22_1_11 = lam22_1_whole[:cn_11_idx, :][:num_x0, :]
        lam22_1_15 = lam22_1_whole[cn_11_idx:cn_15_idx, :][:num_x0, :]
        lam22_1_51 = lam22_1_whole[cn_15_idx:cn_51_idx, :][:num_x0, :]
        lam22_1_55 = lam22_1_whole[cn_51_idx:, :][:num_x0, :]
        lam22_2_11 = lam22_2_whole[:cn_11_idx, :][:num_x0, :]
        lam22_2_15 = lam22_2_whole[cn_11_idx:cn_15_idx, :][:num_x0, :]
        lam22_2_51 = lam22_2_whole[cn_15_idx:cn_51_idx, :][:num_x0, :]
        lam22_2_55 = lam22_2_whole[cn_51_idx:, :][:num_x0, :]
        lam21_1 = torch.cat((lam21_1_11, lam21_1_15, lam21_1_51, lam21_1_55), dim=0)
        lam21_2 = torch.cat((lam21_2_11, lam21_2_15, lam21_2_51, lam21_2_55), dim=0)
        lam22_1 = torch.cat((lam22_1_11, lam22_1_15, lam22_1_51, lam22_1_55), dim=0)
        lam22_2 = torch.cat((lam22_2_11, lam22_2_15, lam22_2_51, lam22_2_55), dim=0)

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        # Agent 1's action, be careful about the order of u1>0 and u1<0
        u1 = 0.5 * lam11_2 * 10
        u1_11 = u1[num_x0:2*num_x0, :]
        u1_15 = u1[num_x0:2*num_x0, :]
        u1_51 = u1[2*num_x0:3*num_x0, :]
        u1_55 = u1[3*num_x0:, :]
        u1 = torch.cat((u1_11, u1_15, u1_51, u1_55), dim=0)

        # Agent 2's action, be careful about the order of u2>0 and u2<0
        u2 = 0.5 * lam22_2 * 10
        u2_11 = u2[:num_x0, :]
        u2_15 = u2[num_x0:2*num_x0, :]
        u2_51 = u2[2*num_x0:3*num_x0, :]
        u2_55 = u2[3*num_x0:, :]
        u2 = torch.cat((u2_11, u2_15, u2_51, u2_55), dim=0)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # unnormalize the state for agent 1
        d11_11_cn = (x1_cn[:, :cn_11_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_15_cn = (x1_cn[:, cn_11_idx:cn_15_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_51_cn = (x1_cn[:, cn_15_idx:cn_51_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_55_cn = (x1_cn[:, cn_51_idx:, 1:2] + 1) * (105 - 15) / 2 + 15
        v11_11_cn = (x1_cn[:, :cn_11_idx, 2:3] + 1) * (32 - 15) / 2 + 15
        v11_15_cn = (x1_cn[:, cn_11_idx:cn_15_idx, 2:3] + 1) * (32 - 15) / 2 + 15
        v11_51_cn = (x1_cn[:, cn_15_idx:cn_51_idx, 2:3] + 1) * (32 - 15) / 2 + 15
        v11_55_cn = (x1_cn[:, cn_51_idx:, 2:3] + 1) * (32 - 15) / 2 + 15
        v11 = torch.cat((v11_11_cn[:, :num_x0, :],
                         v11_15_cn[:, :num_x0, :],
                         v11_51_cn[:, :num_x0, :],
                         v11_55_cn[:, :num_x0, :]), dim=1)

        # unnormalize the state for agent 2
        d12_11_cn = (x1_cn[:, :cn_11_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_15_cn = (x1_cn[:, cn_11_idx:cn_15_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_51_cn = (x1_cn[:, cn_15_idx:cn_51_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_55_cn = (x1_cn[:, cn_51_idx:, 3:4] + 1) * (105 - 15) / 2 + 15
        v12_11_cn = (x1_cn[:, :cn_11_idx, 4:5] + 1) * (32 - 15) / 2 + 15
        v12_15_cn = (x1_cn[:, cn_11_idx:cn_15_idx, 4:5] + 1) * (32 - 15) / 2 + 15
        v12_51_cn = (x1_cn[:, cn_15_idx:cn_51_idx, 4:5] + 1) * (32 - 15) / 2 + 15
        v12_55_cn = (x1_cn[:, cn_51_idx:, 4:5] + 1) * (32 - 15) / 2 + 15
        v12 = torch.cat((v12_11_cn[:, :num_x0, :],
                         v12_15_cn[:, :num_x0, :],
                         v12_51_cn[:, :num_x0, :],
                         v12_55_cn[:, :num_x0, :]), dim=1)

        # unnormalize the state for agent 1
        d21_11_cn = (x2[:, :cn_11_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_15_cn = (x2_cn[:, cn_11_idx:cn_15_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_51_cn = (x2_cn[:, cn_15_idx:cn_51_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_55_cn = (x2_cn[:, cn_51_idx:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21_11_cn = (x2_cn[:, :cn_11_idx, 4:5] + 1) * (32 - 15) / 2 + 15
        v21_15_cn = (x2_cn[:, cn_11_idx:cn_15_idx, 4:5] + 1) * (32 - 15) / 2 + 15
        v21_51_cn = (x2_cn[:, cn_15_idx:cn_51_idx, 4:5] + 1) * (32 - 15) / 2 + 15
        v21_55_cn = (x2_cn[:, cn_51_idx:, 4:5] + 1) * (32 - 15) / 2 + 15
        v21 = torch.cat((v21_11_cn[:, :num_x0, :],
                         v21_15_cn[:, :num_x0, :],
                         v21_51_cn[:, :num_x0, :],
                         v21_55_cn[:, :num_x0, :]), dim=1)

        # unnormalize the state for agent 2
        d22_11_cn = (x2[:, :cn_11_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_15_cn = (x2_cn[:, cn_11_idx:cn_15_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_51_cn = (x2_cn[:, cn_15_idx:cn_51_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_55_cn = (x2_cn[:, cn_51_idx:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22_11_cn = (x2_cn[:, :cn_11_idx, 2:3] + 1) * (32 - 15) / 2 + 15
        v22_15_cn = (x2_cn[:, cn_11_idx:cn_15_idx, 2:3] + 1) * (32 - 15) / 2 + 15
        v22_51_cn = (x2_cn[:, cn_15_idx:cn_51_idx, 2:3] + 1) * (32 - 15) / 2 + 15
        v22_55_cn = (x2_cn[:, cn_51_idx:, 2:3] + 1) * (32 - 15) / 2 + 15
        v22 = torch.cat((v22_11_cn[:, :num_x0, :],
                         v22_15_cn[:, :num_x0, :],
                         v22_51_cn[:, :num_x0, :],
                         v22_55_cn[:, :num_x0, :]), dim=1)

        # calculate the collision area lower and upper bounds for a-a
        x11_in_11_cn = ((d11_11_cn[:, :num_x0, :] - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_11_cn = (-(d11_11_cn[:, :num_x0, :] - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_11_cn = ((d12_11_cn[:, :num_x0, :] - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_11_cn = (-(d12_11_cn[:, :num_x0, :] - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_11_cn = torch.sigmoid(x11_in_11_cn) * torch.sigmoid(x11_out_11_cn)
        sigmoid12_11_cn = torch.sigmoid(x12_in_11_cn) * torch.sigmoid(x12_out_11_cn)
        loss_instant1_11_cn = beta * sigmoid11_11_cn * sigmoid12_11_cn

        x21_in_11_cn = ((d21_11_cn[:, :num_x0, :] - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_11_cn = (-(d21_11_cn[:, :num_x0, :] - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_11_cn = ((d22_11_cn[:, :num_x0, :] - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_11_cn = (-(d22_11_cn[:, :num_x0, :] - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_11_cn = torch.sigmoid(x21_in_11_cn) * torch.sigmoid(x21_out_11_cn)
        sigmoid22_11_cn = torch.sigmoid(x22_in_11_cn) * torch.sigmoid(x22_out_11_cn)
        loss_instant2_11_cn = beta * sigmoid21_11_cn * sigmoid22_11_cn

        # calculate the collision area lower and upper bounds for a-na
        x11_in_15_cn = ((d11_15_cn[:, :num_x0, :] - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_15_cn = (-(d11_15_cn[:, :num_x0, :] - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_15_cn = ((d12_15_cn[:, :num_x0, :] - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_15_cn = (-(d12_15_cn[:, :num_x0, :] - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_15_cn = torch.sigmoid(x11_in_15_cn) * torch.sigmoid(x11_out_15_cn)
        sigmoid12_15_cn = torch.sigmoid(x12_in_15_cn) * torch.sigmoid(x12_out_15_cn)
        loss_instant1_15_cn = beta * sigmoid11_15_cn * sigmoid12_15_cn

        x21_in_15_cn = ((d21_15_cn[:, :num_x0, :] - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_15_cn = (-(d21_15_cn[:, :num_x0, :] - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_15_cn = ((d22_15_cn[:, :num_x0, :] - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_15_cn = (-(d22_15_cn[:, :num_x0, :] - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_15_cn = torch.sigmoid(x21_in_15_cn) * torch.sigmoid(x21_out_15_cn)
        sigmoid22_15_cn = torch.sigmoid(x22_in_15_cn) * torch.sigmoid(x22_out_15_cn)
        loss_instant2_15_cn = beta * sigmoid21_15_cn * sigmoid22_15_cn

        # calculate the collision area lower and upper bounds for na-a
        x11_in_51_cn = ((d11_51_cn[:, :num_x0, :] - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_51_cn = (-(d11_51_cn[:, :num_x0, :] - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_51_cn = ((d12_51_cn[:, :num_x0, :] - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_51_cn = (-(d12_51_cn[:, :num_x0, :] - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_51_cn = torch.sigmoid(x11_in_51_cn) * torch.sigmoid(x11_out_51_cn)
        sigmoid12_51_cn = torch.sigmoid(x12_in_51_cn) * torch.sigmoid(x12_out_51_cn)
        loss_instant1_51_cn = beta * sigmoid11_51_cn * sigmoid12_51_cn

        x21_in_51_cn = ((d21_51_cn[:, :num_x0, :] - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_51_cn = (-(d21_51_cn[:, :num_x0, :] - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_51_cn = ((d22_51_cn[:, :num_x0, :] - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_51_cn = (-(d22_51_cn[:, :num_x0, :] - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_51_cn = torch.sigmoid(x21_in_51_cn) * torch.sigmoid(x21_out_51_cn)
        sigmoid22_51_cn = torch.sigmoid(x22_in_51_cn) * torch.sigmoid(x22_out_51_cn)
        loss_instant2_51_cn = beta * sigmoid21_51_cn * sigmoid22_51_cn

        # calculate the collision area lower and upper bounds for na-na
        x11_in_55_cn = ((d11_55_cn[:, :num_x0, :] - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_55_cn = (-(d11_55_cn[:, :num_x0, :] - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_55_cn = ((d12_55_cn[:, :num_x0, :] - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_55_cn = (-(d12_55_cn[:, :num_x0, :] - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_55_cn = torch.sigmoid(x11_in_55_cn) * torch.sigmoid(x11_out_55_cn)
        sigmoid12_55_cn = torch.sigmoid(x12_in_55_cn) * torch.sigmoid(x12_out_55_cn)
        loss_instant1_55_cn = beta * sigmoid11_55_cn * sigmoid12_55_cn

        x21_in_55_cn = ((d21_55_cn[:, :num_x0, :] - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_55_cn = (-(d21_55_cn[:, :num_x0, :] - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_55_cn = ((d22_55_cn[:, :num_x0, :] - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_55_cn = (-(d22_55_cn[:, :num_x0, :] - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_55_cn = torch.sigmoid(x21_in_55_cn) * torch.sigmoid(x21_out_55_cn)
        sigmoid22_55_cn = torch.sigmoid(x22_in_55_cn) * torch.sigmoid(x22_out_55_cn)
        loss_instant2_55_cn = beta * sigmoid21_55_cn * sigmoid22_55_cn

        loss_instant1 = torch.cat((loss_instant1_11_cn, loss_instant1_15_cn, loss_instant1_51_cn, loss_instant1_55_cn), dim=0)
        loss_instant2 = torch.cat((loss_instant2_11_cn, loss_instant2_15_cn, loss_instant2_51_cn, loss_instant2_55_cn), dim=0)

        # calculate instantaneous loss
        loss_fun_1 = 0.1 * (u1 ** 2 + loss_instant1)
        loss_fun_2 = 0.1 * (u2 ** 2 + loss_instant2)

        # calculate hamiltonian, -H = (dV/dx)^T * f - (dV/dz)^T * L
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        diff_constraint_hom_1 = torch.abs(dvdt_1 + ham_1)
        diff_constraint_hom_2 = torch.abs(dvdt_2 + ham_2)
        
        threshold1_11 = torch.mean(diff_constraint_hom_1[:num_x0])
        threshold1_15 = torch.mean(diff_constraint_hom_1[num_x0:2*num_x0])
        threshold1_51 = torch.mean(diff_constraint_hom_1[2*num_x0:3*num_x0])
        threshold1_55 = torch.mean(diff_constraint_hom_1[3*num_x0:])
        threshold2_11 = torch.mean(diff_constraint_hom_2[:num_x0])
        threshold2_15 = torch.mean(diff_constraint_hom_2[num_x0:2*num_x0])
        threshold2_51 = torch.mean(diff_constraint_hom_2[2*num_x0:3*num_x0])
        threshold2_55 = torch.mean(diff_constraint_hom_2[3*num_x0:])
        
        index_hjpde1_11 = torch.where(diff_constraint_hom_1[:num_x0] >= threshold1_11)[0]
        index_hjpde2_11 = torch.where(diff_constraint_hom_2[:num_x0] >= threshold2_11)[0]
        index_hjpde1_15 = torch.where(diff_constraint_hom_1[num_x0:2*num_x0] >= threshold1_15)[0]
        index_hjpde2_15 = torch.where(diff_constraint_hom_2[num_x0:2*num_x0] >= threshold2_15)[0]
        index_hjpde1_51 = torch.where(diff_constraint_hom_1[2*num_x0:3*num_x0] >= threshold1_51)[0]
        index_hjpde2_51 = torch.where(diff_constraint_hom_2[2*num_x0:3*num_x0] >= threshold2_51)[0]
        index_hjpde1_55 = torch.where(diff_constraint_hom_1[3*num_x0:] >= threshold1_55)[0]
        index_hjpde2_55 = torch.where(diff_constraint_hom_2[3*num_x0:] >= threshold2_55)[0]

        if index_hjpde1_11.shape[0] > index_hjpde2_11.shape[0]:
            index_remain_11 = index_hjpde1_11
        else:
            index_remain_11 = index_hjpde2_11
            
        if index_hjpde1_15.shape[0] > index_hjpde2_15.shape[0]:
            index_remain_15 = index_hjpde1_15
        else:
            index_remain_15 = index_hjpde2_15

        if index_hjpde1_51.shape[0] > index_hjpde2_51.shape[0]:
            index_remain_51 = index_hjpde1_51
        else:
            index_remain_51 = index_hjpde2_51

        if index_hjpde1_55.shape[0] > index_hjpde2_55.shape[0]:
            index_remain_55 = index_hjpde1_55
        else:
            index_remain_55 = index_hjpde2_55

        x1_cn = model_output['model_in_cn'][:, :cn_index, :]
        x2_cn = model_output['model_in_cn'][:, cn_index:, :]

        coords1_first_11 = x1_cn[:, :cn_11_idx, :][:, :num_x0, :][:, index_remain_11, :]
        coords2_first_11 = x2_cn[:, :cn_11_idx, :][:, :num_x0, :][:, index_remain_11, :]
        coords1_first_15 = x1_cn[:, cn_11_idx:cn_15_idx, :][:, :num_x0, :][:, index_remain_15, :]
        coords2_first_15 = x2_cn[:, cn_11_idx:cn_15_idx, :][:, :num_x0, :][:, index_remain_15, :]
        coords1_first_51 = x1_cn[:, cn_15_idx:cn_51_idx, :][:, :num_x0, :][:, index_remain_51, :]
        coords2_first_51 = x2_cn[:, cn_15_idx:cn_51_idx, :][:, :num_x0, :][:, index_remain_51, :]
        coords1_first_55 = x1_cn[:, cn_51_idx:, :][:, :num_x0, :][:, index_remain_55, :]
        coords2_first_55 = x2_cn[:, cn_51_idx:, :][:, :num_x0, :][:, index_remain_55, :]

        numpoints_11 = num_x0 - index_remain_11.shape[0]
        numpoints_15 = num_x0 - index_remain_15.shape[0]
        numpoints_51 = num_x0 - index_remain_51.shape[0]
        numpoints_55 = num_x0 - index_remain_55.shape[0]

        time_second_11 = torch.ones(numpoints_11, 1)*3
        coords1_second_11 = torch.zeros(numpoints_11, 4).uniform_(-1, 1)
        coords2_second_11 = torch.cat((coords1_second_11[:, 2:], coords1_second_11[:, :2]), dim=1)
        label1_11 = torch.zeros(numpoints_11, 1)
        label2_11 = torch.ones(numpoints_11, 1)
        coords1_second_11 = torch.cat((time_second_11, coords1_second_11, label1_11), dim=1).unsqueeze(0).to(device)
        coords2_second_11 = torch.cat((time_second_11, coords2_second_11, label2_11), dim=1).unsqueeze(0).to(device)

        time_second_15 = torch.ones(numpoints_15, 1)*3
        coords1_second_15 = torch.zeros(numpoints_15, 4).uniform_(-1, 1)
        coords2_second_15 = torch.cat((coords1_second_15[:, 2:], coords1_second_15[:, :2]), dim=1)
        label1_15 = torch.zeros(numpoints_15, 1)
        label2_15 = torch.ones(numpoints_15, 1)
        coords1_second_15 = torch.cat((time_second_15, coords1_second_15, label1_15), dim=1).unsqueeze(0).to(device)
        coords2_second_15 = torch.cat((time_second_15, coords2_second_15, label2_15), dim=1).unsqueeze(0).to(device)

        time_second_51 = torch.ones(numpoints_51, 1)*3
        coords1_second_51 = torch.zeros(numpoints_51, 4).uniform_(-1, 1)
        coords2_second_51 = torch.cat((coords1_second_51[:, 2:], coords1_second_51[:, :2]), dim=1)
        label1_51 = torch.zeros(numpoints_51, 1)
        label2_51 = torch.ones(numpoints_51, 1)
        coords1_second_51 = torch.cat((time_second_51, coords1_second_51, label1_51), dim=1).unsqueeze(0).to(device)
        coords2_second_51 = torch.cat((time_second_51, coords2_second_51, label2_51), dim=1).unsqueeze(0).to(device)

        time_second_55 = torch.ones(numpoints_55, 1)*3
        coords1_second_55 = torch.zeros(numpoints_55, 4).uniform_(-1, 1)
        coords2_second_55 = torch.cat((coords1_second_55[:, 2:], coords1_second_55[:, :2]), dim=1)
        label1_55 = torch.zeros(numpoints_55, 1)
        label2_55 = torch.ones(numpoints_55, 1)
        coords1_second_55 = torch.cat((time_second_55, coords1_second_55, label1_55), dim=1).unsqueeze(0).to(device)
        coords2_second_55 = torch.cat((time_second_55, coords2_second_55, label2_55), dim=1).unsqueeze(0).to(device)

        coords1_cn = torch.cat((coords1_first_11, coords1_second_11,
                                coords1_first_15, coords1_second_15,
                                coords1_first_51, coords1_second_51,
                                coords1_first_55, coords1_second_55), dim=1)
        coords2_cn = torch.cat((coords2_first_11, coords2_second_11,
                                coords2_first_15, coords2_second_15,
                                coords2_first_51, coords2_second_51,
                                coords2_first_55, coords2_second_55), dim=1)
        coords_cn = torch.cat((coords1_cn, coords2_cn), dim=1)

        coords_data = {'coords_cn': coords_cn}

        print('new sampling points:', int(numpoints_11), int(numpoints_15), int(numpoints_51), int(numpoints_55))

        coords_data_save = {'coords': coords_cn.squeeze().detach().cpu().numpy()}
        save_path = 'cn_data_sine/coords_cn_data_' + str(counter) + '.mat'
        scio.savemat(save_path, coords_data_save)

        return coords_data

    return intersection_sampling
