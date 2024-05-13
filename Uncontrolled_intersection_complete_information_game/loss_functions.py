import torch
import diff_operators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_intersection_HJI_supervised(dataset, Weight, alpha):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]   # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]   # (meta_batch_size, num_points, 1); agent 2's value

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # supervised learning for values
        value1_difference = y1 - alpha * groundtruth_values[:, :y1.shape[1]]
        value2_difference = y2 - alpha * groundtruth_values[:, y2.shape[1]:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1, lam11_2, lam12_1, lam12_2), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam22_1, lam22_2), dim=1)
        costate1_difference = costate1_prediction - alpha * groundtruth_costates[:, :y1.shape[1]].squeeze()
        costate2_difference = costate2_prediction - alpha * groundtruth_costates[:, y2.shape[1]:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # A factor of (weight1, weight2) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2}
    return intersection_hji


def initialize_intersection_HJI_hyrid(dataset, Weight, alpha):
    def intersection_hji(model_output, gt):
        weight1, weight2, weight3, weight4 = Weight
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        source_boundary_values = gt['source_boundary_values']
        dirichlet_mask = gt['dirichlet_mask']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2
        supervised_index = groundtruth_values.shape[1] // 2
        hji_index = source_boundary_values.shape[1] // 2
        num_sl = supervised_index // 4
        num_hl = hji_index // 4

        y1 = y[:, :cut_index]   # (meta_batch_size, num_points, 1); agent 1's value
        y2 = y[:, cut_index:]   # (meta_batch_size, num_points, 1); agent 2's value
        x1 = x[:, :cut_index]
        x2 = x[:, cut_index:]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision loss weight

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = 0.5 * lam11_2 * (1/alpha)

        # Agent 2's action
        u2 = 0.5 * lam22_2 * (1/alpha)

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # unnormalize the state for agent 1
        d11_11_sl = (x1[:, :num_sl, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_15_sl = (x1[:, num_sl:2*num_sl, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_51_sl = (x1[:, 2*num_sl:3*num_sl, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_55_sl = (x1[:, 3*num_sl:4*num_sl, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_55_hl = (x1[:, 4*num_sl+3*num_hl:, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (x1[:, :, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12_11_sl = (x1[:, :num_sl, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_15_sl = (x1[:, num_sl:2*num_sl, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_51_sl = (x1[:, 2*num_sl:3*num_sl, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_55_sl = (x1[:, 3*num_sl:4*num_sl, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_11_hl = (x1[:, 4*num_sl:4*num_sl+num_hl, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_15_hl = (x1[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_51_hl = (x1[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_55_hl = (x1[:, 4*num_sl+3*num_hl:, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (x1[:, :, 4:5] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x11_in_11_sl = ((d11_11_sl - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_11_sl = (-(d11_11_sl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_11_sl = ((d12_11_sl - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_11_sl = (-(d12_11_sl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_in_11_hl = ((d11_11_hl - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_11_hl = (-(d11_11_hl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_11_hl = ((d12_11_hl - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_11_hl = (-(d12_11_hl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_11_sl = torch.sigmoid(x11_in_11_sl) * torch.sigmoid(x11_out_11_sl)
        sigmoid12_11_sl = torch.sigmoid(x12_in_11_sl) * torch.sigmoid(x12_out_11_sl)
        loss_instant1_11_sl = beta * sigmoid11_11_sl * sigmoid12_11_sl
        sigmoid11_11_hl = torch.sigmoid(x11_in_11_hl) * torch.sigmoid(x11_out_11_hl)
        sigmoid12_11_hl = torch.sigmoid(x12_in_11_hl) * torch.sigmoid(x12_out_11_hl)
        loss_instant1_11_hl = beta * sigmoid11_11_hl * sigmoid12_11_hl

        x11_in_15_sl = ((d11_15_sl - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_15_sl = (-(d11_15_sl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_15_sl = ((d12_15_sl - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_15_sl = (-(d12_15_sl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_in_15_hl = ((d11_15_hl - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_15_hl = (-(d11_15_hl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_15_hl = ((d12_15_hl - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_15_hl = (-(d12_15_hl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_15_sl = torch.sigmoid(x11_in_15_sl) * torch.sigmoid(x11_out_15_sl)
        sigmoid12_15_sl = torch.sigmoid(x12_in_15_sl) * torch.sigmoid(x12_out_15_sl)
        loss_instant1_15_sl = beta * sigmoid11_15_sl * sigmoid12_15_sl
        sigmoid11_15_hl = torch.sigmoid(x11_in_15_hl) * torch.sigmoid(x11_out_15_hl)
        sigmoid12_15_hl = torch.sigmoid(x12_in_15_hl) * torch.sigmoid(x12_out_15_hl)
        loss_instant1_15_hl = beta * sigmoid11_15_hl * sigmoid12_15_hl

        x11_in_51_sl = ((d11_51_sl - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_51_sl = (-(d11_51_sl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_51_sl = ((d12_51_sl - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_51_sl = (-(d12_51_sl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_in_51_hl = ((d11_51_hl - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_51_hl = (-(d11_51_hl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_51_hl = ((d12_51_hl - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_51_hl = (-(d12_51_hl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_51_sl = torch.sigmoid(x11_in_51_sl) * torch.sigmoid(x11_out_51_sl)
        sigmoid12_51_sl = torch.sigmoid(x12_in_51_sl) * torch.sigmoid(x12_out_51_sl)
        loss_instant1_51_sl = beta * sigmoid11_51_sl * sigmoid12_51_sl
        sigmoid11_51_hl = torch.sigmoid(x11_in_51_hl) * torch.sigmoid(x11_out_51_hl)
        sigmoid12_51_hl = torch.sigmoid(x12_in_51_hl) * torch.sigmoid(x12_out_51_hl)
        loss_instant1_51_hl = beta * sigmoid11_51_hl * sigmoid12_51_hl

        x11_in_55_sl = ((d11_55_sl - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_55_sl = (-(d11_55_sl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_55_sl = ((d12_55_sl - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_55_sl = (-(d12_55_sl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_in_55_hl = ((d11_55_hl - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_55_hl = (-(d11_55_hl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_55_hl = ((d12_55_hl - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_55_hl = (-(d12_55_hl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_55_sl = torch.sigmoid(x11_in_55_sl) * torch.sigmoid(x11_out_55_sl)
        sigmoid12_55_sl = torch.sigmoid(x12_in_55_sl) * torch.sigmoid(x12_out_55_sl)
        loss_instant1_55_sl = beta * sigmoid11_55_sl * sigmoid12_55_sl
        sigmoid11_55_hl = torch.sigmoid(x11_in_55_hl) * torch.sigmoid(x11_out_55_hl)
        sigmoid12_55_hl = torch.sigmoid(x12_in_55_hl) * torch.sigmoid(x12_out_55_hl)
        loss_instant1_55_hl = beta * sigmoid11_55_hl * sigmoid12_55_hl

        # unnormalize the state for agent 1
        d21_11_sl = (x2[:, :num_sl, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_15_sl = (x2[:, num_sl:2*num_sl, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_51_sl = (x2[:, 2*num_sl:3*num_sl, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_55_sl = (x2[:, 3*num_sl:4*num_sl, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_55_hl = (x2[:, 4*num_sl+3*num_hl:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (x2[:, :, 4:5] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22_11_sl = (x2[:, :num_sl, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_15_sl = (x2[:, num_sl:2*num_sl, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_51_sl = (x2[:, 2*num_sl:3*num_sl, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_55_sl = (x2[:, 3*num_sl:4*num_sl, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_11_hl = (x2[:, 4*num_sl:4*num_sl+num_hl, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_15_hl = (x2[:, 4*num_sl+num_hl:4*num_sl+2*num_hl, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_51_hl = (x2[:, 4*num_sl+2*num_hl:4*num_sl+3*num_hl, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_55_hl = (x2[:, 4*num_sl+3*num_hl:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (x2[:, :, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x21_in_11_sl = ((d21_11_sl - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_11_sl = (-(d21_11_sl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_11_sl = ((d22_11_sl - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_11_sl = (-(d22_11_sl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_in_11_hl = ((d21_11_hl - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_11_hl = (-(d21_11_hl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_11_hl = ((d22_11_hl - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_11_hl = (-(d22_11_hl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_11_sl = torch.sigmoid(x21_in_11_sl) * torch.sigmoid(x21_out_11_sl)
        sigmoid22_11_sl = torch.sigmoid(x22_in_11_sl) * torch.sigmoid(x22_out_11_sl)
        loss_instant2_11_sl = beta * sigmoid21_11_sl * sigmoid22_11_sl
        sigmoid21_11_hl = torch.sigmoid(x21_in_11_hl) * torch.sigmoid(x21_out_11_hl)
        sigmoid22_11_hl = torch.sigmoid(x22_in_11_hl) * torch.sigmoid(x22_out_11_hl)
        loss_instant2_11_hl = beta * sigmoid21_11_hl * sigmoid22_11_hl
        
        x21_in_15_sl = ((d21_15_sl - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_15_sl = (-(d21_15_sl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_15_sl = ((d22_15_sl - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_15_sl = (-(d22_15_sl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_in_15_hl = ((d21_15_hl - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_15_hl = (-(d21_15_hl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_15_hl = ((d22_15_hl - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_15_hl = (-(d22_15_hl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_15_sl = torch.sigmoid(x21_in_15_sl) * torch.sigmoid(x21_out_15_sl)
        sigmoid22_15_sl = torch.sigmoid(x22_in_15_sl) * torch.sigmoid(x22_out_15_sl)
        loss_instant2_15_sl = beta * sigmoid21_15_sl * sigmoid22_15_sl
        sigmoid21_15_hl = torch.sigmoid(x21_in_15_hl) * torch.sigmoid(x21_out_15_hl)
        sigmoid22_15_hl = torch.sigmoid(x22_in_15_hl) * torch.sigmoid(x22_out_15_hl)
        loss_instant2_15_hl = beta * sigmoid21_15_hl * sigmoid22_15_hl

        x21_in_51_sl = ((d21_51_sl - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_51_sl = (-(d21_51_sl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_51_sl = ((d22_51_sl - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_51_sl = (-(d22_51_sl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_in_51_hl = ((d21_51_hl - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_51_hl = (-(d21_51_hl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_51_hl = ((d22_51_hl - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_51_hl = (-(d22_51_hl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_51_sl = torch.sigmoid(x21_in_51_sl) * torch.sigmoid(x21_out_51_sl)
        sigmoid22_51_sl = torch.sigmoid(x22_in_51_sl) * torch.sigmoid(x22_out_51_sl)
        loss_instant2_51_sl = beta * sigmoid21_51_sl * sigmoid22_51_sl
        sigmoid21_51_hl = torch.sigmoid(x21_in_51_hl) * torch.sigmoid(x21_out_51_hl)
        sigmoid22_51_hl = torch.sigmoid(x22_in_51_hl) * torch.sigmoid(x22_out_51_hl)
        loss_instant2_51_hl = beta * sigmoid21_51_hl * sigmoid22_51_hl

        x21_in_55_sl = ((d21_55_sl - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_55_sl = (-(d21_55_sl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_55_sl = ((d22_55_sl - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_55_sl = (-(d22_55_sl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_in_55_hl = ((d21_55_hl - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_55_hl = (-(d21_55_hl - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_55_hl = ((d22_55_hl - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_55_hl = (-(d22_55_hl - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_55_sl = torch.sigmoid(x21_in_55_sl) * torch.sigmoid(x21_out_55_sl)
        sigmoid22_55_sl = torch.sigmoid(x22_in_55_sl) * torch.sigmoid(x22_out_55_sl)
        loss_instant2_55_sl = beta * sigmoid21_55_sl * sigmoid22_55_sl
        sigmoid21_55_hl = torch.sigmoid(x21_in_55_hl) * torch.sigmoid(x21_out_55_hl)
        sigmoid22_55_hl = torch.sigmoid(x22_in_55_hl) * torch.sigmoid(x22_out_55_hl)
        loss_instant2_55_hl = beta * sigmoid21_55_hl * sigmoid22_55_hl

        # calculate instantaneous loss
        loss_instant1 = torch.cat((loss_instant1_11_sl, loss_instant1_15_sl, loss_instant1_51_sl, loss_instant1_55_sl,
                                   loss_instant1_11_hl, loss_instant1_15_hl, loss_instant1_51_hl, loss_instant1_55_hl), dim=0)
        loss_instant2 = torch.cat((loss_instant2_11_sl, loss_instant2_15_sl, loss_instant2_51_sl, loss_instant2_55_sl,
                                   loss_instant2_11_hl, loss_instant2_15_hl, loss_instant2_51_hl, loss_instant2_55_hl), dim=0)
        loss_fun_1 = alpha * (u1 ** 2 + loss_instant1)
        loss_fun_2 = alpha * (u2 ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        diff_constraint_hom_1 = dvdt_1 + ham_1
        diff_constraint_hom_2 = dvdt_2 + ham_2
        diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # supervised learning for values
        value1_difference = y1[:, :supervised_index] - alpha * groundtruth_values[:, :supervised_index]
        value2_difference = y2[:, :supervised_index] - alpha * groundtruth_values[:, supervised_index:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1[:supervised_index, :],
                                         lam11_2[:supervised_index, :],
                                         lam12_1[:supervised_index, :],
                                         lam12_2[:supervised_index, :]), dim=1)
        costate2_prediction = torch.cat((lam21_1[:supervised_index, :],
                                         lam21_2[:supervised_index, :],
                                         lam22_1[:supervised_index, :],
                                         lam22_2[:supervised_index, :]), dim=1)
        costate1_difference = costate1_prediction - alpha * groundtruth_costates[:, :supervised_index].squeeze()
        costate2_difference = costate2_prediction - alpha * groundtruth_costates[:, supervised_index:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # boundary condition check
        dirichlet_1 = y1[:, supervised_index:][dirichlet_mask] - alpha * source_boundary_values[:, :hji_index][dirichlet_mask]
        dirichlet_2 = y2[:, supervised_index:][dirichlet_mask] - alpha * source_boundary_values[:, hji_index:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (weight1, weight2, weight3, weight4) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2,
                'dirichlet': torch.abs(dirichlet).sum() / weight3,  
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight4}
    return intersection_hji

def initialize_HJ_Pontryagin(dataset, Weight, device):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        boundary_values_vn = gt['boundary_values_vn']
        boundary_values_cn = gt['boundary_values_cn']
        x = model_output['model_in_vn']
        y = model_output['model_out_vn']
        dirichlet_mask_vn = gt['dirichlet_mask_vn']
        dirichlet_mask_cn = gt['dirichlet_mask_cn']
        values_gt = gt['value_gt']
        costates_gt = gt['costate_gt']
        costate_pred = model_output['model_out_cn']
        cn_index = costate_pred.shape[1] // 2
        cut_index = x.shape[1] // 2
        cn_11_idx = int(gt['num_cn'].squeeze()[0])
        cn_15_idx = int(gt['num_cn'].squeeze()[0]) + int(gt['num_cn'].squeeze()[1])
        cn_51_idx = int(gt['num_cn'].squeeze()[0]) + int(gt['num_cn'].squeeze()[1]) + int(gt['num_cn'].squeeze()[2])

        y1 = y[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = y[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        x1 = x[:, :cut_index]
        x2 = x[:, cut_index:]

        if torch.all(dirichlet_mask_vn):
            num_vn = cut_index // 4
            vn_index = cut_index
            x1_cn = x1[:, :0]
            x2_cn = x2[:, :0]
            x1_vn = x1
            x2_vn = x2
        else:
            num_vn = (cut_index - cn_index) // 4
            vn_index = cut_index - cn_index
            x1_cn = x1[:, :cn_index]
            x2_cn = x2[:, :cn_index]
            x1_vn = x1[:, cn_index:]
            x2_vn = x2[:, cn_index:]

        costate1_pred = costate_pred[:, :cn_index]  # (meta_batch_size, num_points, 1); agent 1's costate
        costate2_pred = costate_pred[:, cn_index:]  # (meta_batch_size, num_points, 1); agent 2's costate

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)

        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:4] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:4] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

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

        # Agent 2's action, be careful about the order of u2>0 and u2<0
        u2 = 0.5 * lam22_2 * 10

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # unnormalize the state for agent 1
        d11_11_cn = (x1_cn[:, :cn_11_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_15_cn = (x1_cn[:, cn_11_idx:cn_15_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_51_cn = (x1_cn[:, cn_15_idx:cn_51_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_55_cn = (x1_cn[:, cn_51_idx:, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_11_vn = (x1_vn[:, :num_vn, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_15_vn = (x1_vn[:, num_vn:2*num_vn, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_51_vn = (x1_vn[:, 2*num_vn:3*num_vn, 1:2] + 1) * (105 - 15) / 2 + 15
        d11_55_vn = (x1_vn[:, 3*num_vn:, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (x1[:, :, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12_11_cn = (x1_cn[:, :cn_11_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_15_cn = (x1_cn[:, cn_11_idx:cn_15_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_51_cn = (x1_cn[:, cn_15_idx:cn_51_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_55_cn = (x1_cn[:, cn_51_idx:, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_11_vn = (x1_vn[:, :num_vn, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_15_vn = (x1_vn[:, num_vn:2*num_vn, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_51_vn = (x1_vn[:, 2*num_vn:3*num_vn, 3:4] + 1) * (105 - 15) / 2 + 15
        d12_55_vn = (x1_vn[:, 3*num_vn:, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (x1[:, :, 4:5] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 1
        d21_11_cn = (x2_cn[:, :cn_11_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_15_cn = (x2_cn[:, cn_11_idx:cn_15_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_51_cn = (x2_cn[:, cn_15_idx:cn_51_idx, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_55_cn = (x2_cn[:, cn_51_idx:, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_11_vn = (x2_vn[:, :num_vn, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_15_vn = (x2_vn[:, num_vn:2*num_vn, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_51_vn = (x2_vn[:, 2*num_vn:3*num_vn, 3:4] + 1) * (105 - 15) / 2 + 15
        d21_55_vn = (x2_vn[:, 3*num_vn:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (x2[:, :, 4:5] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22_11_cn = (x2_cn[:, :cn_11_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_15_cn = (x2_cn[:, cn_11_idx:cn_15_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_51_cn = (x2_cn[:, cn_15_idx:cn_51_idx, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_55_cn = (x2_cn[:, cn_51_idx:, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_11_vn = (x2_vn[:, :num_vn, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_15_vn = (x2_vn[:, num_vn:2*num_vn, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_51_vn = (x2_vn[:, 2*num_vn:3*num_vn, 1:2] + 1) * (105 - 15) / 2 + 15
        d22_55_vn = (x2_vn[:, 3*num_vn:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (x2[:, :, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds for a-a
        x11_in_11_cn = ((d11_11_cn - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_11_cn = (-(d11_11_cn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_11_cn = ((d12_11_cn - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_11_cn = (-(d12_11_cn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_in_11_vn = ((d11_11_vn - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_11_vn = (-(d11_11_vn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_11_vn = ((d12_11_vn - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_11_vn = (-(d12_11_vn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_11_cn = torch.sigmoid(x11_in_11_cn) * torch.sigmoid(x11_out_11_cn)
        sigmoid12_11_cn = torch.sigmoid(x12_in_11_cn) * torch.sigmoid(x12_out_11_cn)
        loss_instant1_11_cn = beta * sigmoid11_11_cn * sigmoid12_11_cn
        sigmoid11_11_vn = torch.sigmoid(x11_in_11_vn) * torch.sigmoid(x11_out_11_vn)
        sigmoid12_11_vn = torch.sigmoid(x12_in_11_vn) * torch.sigmoid(x12_out_11_vn)
        loss_instant1_11_vn = beta * sigmoid11_11_vn * sigmoid12_11_vn

        x21_in_11_cn = ((d21_11_cn - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_11_cn = (-(d21_11_cn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_11_cn = ((d22_11_cn - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_11_cn = (-(d22_11_cn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_in_11_vn = ((d21_11_vn - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_11_vn = (-(d21_11_vn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_11_vn = ((d22_11_vn - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_11_vn = (-(d22_11_vn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_11_cn = torch.sigmoid(x21_in_11_cn) * torch.sigmoid(x21_out_11_cn)
        sigmoid22_11_cn = torch.sigmoid(x22_in_11_cn) * torch.sigmoid(x22_out_11_cn)
        loss_instant2_11_cn = beta * sigmoid21_11_cn * sigmoid22_11_cn
        sigmoid21_11_vn = torch.sigmoid(x21_in_11_vn) * torch.sigmoid(x21_out_11_vn)
        sigmoid22_11_vn = torch.sigmoid(x22_in_11_vn) * torch.sigmoid(x22_out_11_vn)
        loss_instant2_11_vn = beta * sigmoid21_11_vn * sigmoid22_11_vn

        # calculate the collision area lower and upper bounds for a-na
        x11_in_15_cn = ((d11_15_cn - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_15_cn = (-(d11_15_cn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_15_cn = ((d12_15_cn - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_15_cn = (-(d12_15_cn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_in_15_vn = ((d11_15_vn - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_15_vn = (-(d11_15_vn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_15_vn = ((d12_15_vn - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_15_vn = (-(d12_15_vn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_15_cn = torch.sigmoid(x11_in_15_cn) * torch.sigmoid(x11_out_15_cn)
        sigmoid12_15_cn = torch.sigmoid(x12_in_15_cn) * torch.sigmoid(x12_out_15_cn)
        loss_instant1_15_cn = beta * sigmoid11_15_cn * sigmoid12_15_cn
        sigmoid11_15_vn = torch.sigmoid(x11_in_15_vn) * torch.sigmoid(x11_out_15_vn)
        sigmoid12_15_vn = torch.sigmoid(x12_in_15_vn) * torch.sigmoid(x12_out_15_vn)
        loss_instant1_15_vn = beta * sigmoid11_15_vn * sigmoid12_15_vn

        x21_in_15_cn = ((d21_15_cn - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_15_cn = (-(d21_15_cn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_15_cn = ((d22_15_cn - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_15_cn = (-(d22_15_cn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_in_15_vn = ((d21_15_vn - R1 / 2 + 1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_15_vn = (-(d21_15_vn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_15_vn = ((d22_15_vn - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_15_vn = (-(d22_15_vn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_15_cn = torch.sigmoid(x21_in_15_cn) * torch.sigmoid(x21_out_15_cn)
        sigmoid22_15_cn = torch.sigmoid(x22_in_15_cn) * torch.sigmoid(x22_out_15_cn)
        loss_instant2_15_cn = beta * sigmoid21_15_cn * sigmoid22_15_cn
        sigmoid21_15_vn = torch.sigmoid(x21_in_15_vn) * torch.sigmoid(x21_out_15_vn)
        sigmoid22_15_vn = torch.sigmoid(x22_in_15_vn) * torch.sigmoid(x22_out_15_vn)
        loss_instant2_15_vn = beta * sigmoid21_15_vn * sigmoid22_15_vn

        # calculate the collision area lower and upper bounds for na-a
        x11_in_51_cn = ((d11_51_cn - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_51_cn = (-(d11_51_cn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_51_cn = ((d12_51_cn - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_51_cn = (-(d12_51_cn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_in_51_vn = ((d11_51_vn - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_51_vn = (-(d11_51_vn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_51_vn = ((d12_51_vn - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_51_vn = (-(d12_51_vn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_51_cn = torch.sigmoid(x11_in_51_cn) * torch.sigmoid(x11_out_51_cn)
        sigmoid12_51_cn = torch.sigmoid(x12_in_51_cn) * torch.sigmoid(x12_out_51_cn)
        loss_instant1_51_cn = beta * sigmoid11_51_cn * sigmoid12_51_cn
        sigmoid11_51_vn = torch.sigmoid(x11_in_51_vn) * torch.sigmoid(x11_out_51_vn)
        sigmoid12_51_vn = torch.sigmoid(x12_in_51_vn) * torch.sigmoid(x12_out_51_vn)
        loss_instant1_51_vn = beta * sigmoid11_51_vn * sigmoid12_51_vn

        x21_in_51_cn = ((d21_51_cn - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_51_cn = (-(d21_51_cn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_51_cn = ((d22_51_cn - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_51_cn = (-(d22_51_cn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_in_51_vn = ((d21_51_vn - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_51_vn = (-(d21_51_vn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_51_vn = ((d22_51_vn - R2 / 2 + 1 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_51_vn = (-(d22_51_vn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_51_cn = torch.sigmoid(x21_in_51_cn) * torch.sigmoid(x21_out_51_cn)
        sigmoid22_51_cn = torch.sigmoid(x22_in_51_cn) * torch.sigmoid(x22_out_51_cn)
        loss_instant2_51_cn = beta * sigmoid21_51_cn * sigmoid22_51_cn
        sigmoid21_51_vn = torch.sigmoid(x21_in_51_vn) * torch.sigmoid(x21_out_51_vn)
        sigmoid22_51_vn = torch.sigmoid(x22_in_51_vn) * torch.sigmoid(x22_out_51_vn)
        loss_instant2_51_vn = beta * sigmoid21_51_vn * sigmoid22_51_vn

        # calculate the collision area lower and upper bounds for na-na
        x11_in_55_cn = ((d11_55_cn - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_55_cn = (-(d11_55_cn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_55_cn = ((d12_55_cn - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_55_cn = (-(d12_55_cn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_in_55_vn = ((d11_55_vn - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out_55_vn = (-(d11_55_vn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in_55_vn = ((d12_55_vn - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out_55_vn = (-(d12_55_vn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11_55_cn = torch.sigmoid(x11_in_55_cn) * torch.sigmoid(x11_out_55_cn)
        sigmoid12_55_cn = torch.sigmoid(x12_in_55_cn) * torch.sigmoid(x12_out_55_cn)
        loss_instant1_55_cn = beta * sigmoid11_55_cn * sigmoid12_55_cn
        sigmoid11_55_vn = torch.sigmoid(x11_in_55_vn) * torch.sigmoid(x11_out_55_vn)
        sigmoid12_55_vn = torch.sigmoid(x12_in_55_vn) * torch.sigmoid(x12_out_55_vn)
        loss_instant1_55_vn = beta * sigmoid11_55_vn * sigmoid12_55_vn

        x21_in_55_cn = ((d21_55_cn - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_55_cn = (-(d21_55_cn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_55_cn = ((d22_55_cn - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_55_cn = (-(d22_55_cn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_in_55_vn = ((d21_55_vn - R1 / 2 + 5 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out_55_vn = (-(d21_55_vn - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in_55_vn = ((d22_55_vn - R2 / 2 + 5 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out_55_vn = (-(d22_55_vn - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21_55_cn = torch.sigmoid(x21_in_55_cn) * torch.sigmoid(x21_out_55_cn)
        sigmoid22_55_cn = torch.sigmoid(x22_in_55_cn) * torch.sigmoid(x22_out_55_cn)
        loss_instant2_55_cn = beta * sigmoid21_55_cn * sigmoid22_55_cn
        sigmoid21_55_vn = torch.sigmoid(x21_in_55_vn) * torch.sigmoid(x21_out_55_vn)
        sigmoid22_55_vn = torch.sigmoid(x22_in_55_vn) * torch.sigmoid(x22_out_55_vn)
        loss_instant2_55_vn = beta * sigmoid21_55_vn * sigmoid22_55_vn

        loss_instant1 = torch.cat((loss_instant1_11_cn, loss_instant1_15_cn, loss_instant1_51_cn, loss_instant1_55_cn,
                                   loss_instant1_11_vn, loss_instant1_15_vn, loss_instant1_51_vn, loss_instant1_55_vn), dim=0)
        loss_instant2 = torch.cat((loss_instant2_11_cn, loss_instant2_15_cn, loss_instant2_51_cn, loss_instant2_55_cn,
                                   loss_instant2_11_vn, loss_instant2_15_vn, loss_instant2_51_vn, loss_instant2_55_vn), dim=0)

        # calculate instantaneous loss
        loss_fun_1 = 0.1 * (u1 ** 2 + loss_instant1)
        loss_fun_2 = 0.1 * (u2 ** 2 + loss_instant2)

        # calculate hamiltonian, -H = (dV/dx)^T * f - (dV/dz)^T * L
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check, HJI = dV/dt - H = -dV/dt + (dV/dx)^T * f - (dV/dz)^T * L because we invert the time
        if torch.all(dirichlet_mask_vn):
            # boundary condition check
            dirichlet1_vn = y1[dirichlet_mask_vn] - 0.1 * boundary_values_vn[:, :vn_index][dirichlet_mask_vn]
            dirichlet2_vn = y2[dirichlet_mask_vn] - 0.1 * boundary_values_vn[:, vn_index:][dirichlet_mask_vn]
            dirichlet_vn = torch.cat((dirichlet1_vn, dirichlet2_vn), dim=0)

            dirichlet1_cn = costate1_pred[dirichlet_mask_cn] - 0.1 * boundary_values_cn[:, :cn_index][dirichlet_mask_cn]
            dirichlet2_cn = costate2_pred[dirichlet_mask_cn] - 0.1 * boundary_values_cn[:, cn_index:][dirichlet_mask_cn]
            dirichlet_cn = torch.cat((dirichlet1_cn, dirichlet2_cn), dim=0)

            diff_constraint_hom_1 = torch.Tensor([0])
            diff_constraint_hom_2 = torch.Tensor([0])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

            # compute value difference
            values1_difference = torch.Tensor([0])
            values2_difference = torch.Tensor([0])
            values_difference = torch.cat((values1_difference, values2_difference), dim=0)

            # compute costate difference
            costate1_difference = torch.Tensor([0])
            costate2_difference = torch.Tensor([0])
            costates_difference_vn = torch.cat((costate1_difference, costate2_difference), dim=0)
            costates_difference_cn = torch.cat((costate1_difference, costate2_difference), dim=0)
        else:
            dirichlet1_vn = y1[:, cn_index:][dirichlet_mask_vn] - 0.1 * boundary_values_vn[:, :vn_index][dirichlet_mask_vn]
            dirichlet2_vn = y2[:, cn_index:][dirichlet_mask_vn] - 0.1 * boundary_values_vn[:, vn_index:][dirichlet_mask_vn]
            dirichlet_vn = torch.cat((dirichlet1_vn, dirichlet2_vn), dim=0)

            dirichlet1_cn = torch.Tensor([0])
            dirichlet2_cn = torch.Tensor([0])
            dirichlet_cn = torch.cat((dirichlet1_cn, dirichlet2_cn), dim=0)

            diff_constraint_hom_1 = dvdt_1 + ham_1
            diff_constraint_hom_2 = dvdt_2 + ham_2
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

            # compute value difference
            value1_difference = y1[:, :cn_index] - 0.1 * values_gt[:, :cn_index]
            value2_difference = y2[:, :cn_index] - 0.1 * values_gt[:, cn_index:]
            values_difference = torch.cat((value1_difference, value2_difference), dim=1)

            # compute costate difference
            costate1_prediction = torch.cat((lam11_1[:cn_index, :],
                                             lam11_2[:cn_index, :],
                                             lam12_1[:cn_index, :],
                                             lam12_2[:cn_index, :]), dim=1)
            costate2_prediction = torch.cat((lam21_1[:cn_index, :],
                                             lam21_2[:cn_index, :],
                                             lam22_1[:cn_index, :],
                                             lam22_2[:cn_index, :]), dim=1)
            costate1_difference_vn = costate1_prediction - 0.1 * costates_gt[:, :cn_index].squeeze()
            costate2_difference_vn = costate2_prediction - 0.1 * costates_gt[:, cn_index:].squeeze()
            costates_difference_vn = torch.cat((costate1_difference_vn, costate2_difference_vn), dim=0)

            costate1_difference_cn = costate_pred[:, :cn_index].squeeze() - 0.1 * costates_gt[:, :cn_index].squeeze()
            costate2_difference_cn = costate_pred[:, cn_index:].squeeze() - 0.1 * costates_gt[:, cn_index:].squeeze()
            costates_difference_cn = torch.cat((costate1_difference_cn, costate2_difference_cn), dim=0)

        weight_ratio = torch.abs(diff_constraint_hom).sum() * weight1 / torch.abs(dirichlet_vn).sum()

        weight_ratio = weight_ratio.detach()

        if weight_ratio == 0:
            hjpde_weight = 1
        else:
            hjpde_weight = float(weight_ratio)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet_vn': torch.abs(dirichlet_vn).sum(),
                'dirichlet_cn': torch.abs(dirichlet_cn).sum() / 2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / 50,
                'costate_difference_vn': torch.abs(costates_difference_vn).sum() / 10,
                'costate_difference_cn': torch.abs(costates_difference_cn).sum() / 5,
                'value_difference': torch.abs(values_difference).sum() / 50,
                'weight': hjpde_weight}

    return intersection_hji

