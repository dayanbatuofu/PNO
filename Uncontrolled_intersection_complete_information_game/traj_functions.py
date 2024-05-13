import torch
import copy

torch.manual_seed(0)

def initialize_pontryagin_sampling(dataset, device):
    def intersection_sampling(model_output, gt, u_star, num):
        start_time = 0.
        dt = gt['dt']
        numcostate = int(gt['numcostate'])

        u1_11 = u_star['u1_star'][:, :numcostate, :]
        u2_11 = u_star['u2_star'][:, :numcostate, :]
        u1_15 = u_star['u1_star'][:, numcostate:2*numcostate, :]
        u2_15 = u_star['u2_star'][:, numcostate:2*numcostate, :]
        u1_51 = u_star['u1_star'][:, 2*numcostate:3*numcostate, :]
        u2_51 = u_star['u2_star'][:, 2*numcostate:3*numcostate, :]
        u1_55 = u_star['u1_star'][:, 3*numcostate:4*numcostate, :]
        u2_55 = u_star['u2_star'][:, 3*numcostate:4*numcostate, :]

        max_d = torch.tensor([105.], dtype=torch.float32).to(device)
        max_v = torch.tensor([32.], dtype=torch.float32).to(device)
        min_d = torch.tensor([15.], dtype=torch.float32).to(device)
        min_v = torch.tensor([15.], dtype=torch.float32).to(device)

        v1_opt_11 = (model_output['model_in_cn'][:, :numcostate, 2:3] + 1) * (32 - 15) / 2 + 15 + u1_11 * dt
        v2_opt_11 = (model_output['model_in_cn'][:, :numcostate, 4:5] + 1) * (32 - 15) / 2 + 15 + u2_11 * dt
        v1_cur_11 = (model_output['model_in_cn'][:, :numcostate, 2:3] + 1) * (32 - 15) / 2 + 15
        v2_cur_11 = (model_output['model_in_cn'][:, :numcostate, 4:5] + 1) * (32 - 15) / 2 + 15
        d1_opt_11 = (model_output['model_in_cn'][:, :numcostate, 1:2] + 1) * (105 - 15) / 2 + 15 + v1_cur_11 * dt
        d2_opt_11 = (model_output['model_in_cn'][:, :numcostate, 3:4] + 1) * (105 - 15) / 2 + 15 + v2_cur_11 * dt

        v1_opt_15 = (model_output['model_in_cn'][:, numcostate:2*numcostate, 2:3] + 1) * (32 - 15) / 2 + 15 + u1_15 * dt
        v2_opt_15 = (model_output['model_in_cn'][:, numcostate:2*numcostate, 4:5] + 1) * (32 - 15) / 2 + 15 + u2_15 * dt
        v1_cur_15 = (model_output['model_in_cn'][:, numcostate:2*numcostate, 2:3] + 1) * (32 - 15) / 2 + 15
        v2_cur_15 = (model_output['model_in_cn'][:, numcostate:2*numcostate, 4:5] + 1) * (32 - 15) / 2 + 15
        d1_opt_15 = (model_output['model_in_cn'][:, numcostate:2*numcostate, 1:2] + 1) * (105 - 15) / 2 + 15 + v1_cur_15 * dt
        d2_opt_15 = (model_output['model_in_cn'][:, numcostate:2*numcostate, 3:4] + 1) * (105 - 15) / 2 + 15 + v2_cur_15 * dt

        v1_opt_51 = (model_output['model_in_cn'][:, 2*numcostate:3*numcostate, 2:3] + 1) * (32 - 15) / 2 + 15 + u1_51 * dt
        v2_opt_51 = (model_output['model_in_cn'][:, 2*numcostate:3*numcostate, 4:5] + 1) * (32 - 15) / 2 + 15 + u2_51 * dt
        v1_cur_51 = (model_output['model_in_cn'][:, 2*numcostate:3*numcostate, 2:3] + 1) * (32 - 15) / 2 + 15
        v2_cur_51 = (model_output['model_in_cn'][:, 2*numcostate:3*numcostate, 4:5] + 1) * (32 - 15) / 2 + 15
        d1_opt_51 = (model_output['model_in_cn'][:, 2*numcostate:3*numcostate, 1:2] + 1) * (105 - 15) / 2 + 15 + v1_cur_51 * dt
        d2_opt_51 = (model_output['model_in_cn'][:, 2*numcostate:3*numcostate, 3:4] + 1) * (105 - 15) / 2 + 15 + v2_cur_51 * dt

        v1_opt_55 = (model_output['model_in_cn'][:, 3*numcostate:4*numcostate, 2:3] + 1) * (32 - 15) / 2 + 15 + u1_55 * dt
        v2_opt_55 = (model_output['model_in_cn'][:, 3*numcostate:4*numcostate, 4:5] + 1) * (32 - 15) / 2 + 15 + u2_55 * dt
        v1_cur_55 = (model_output['model_in_cn'][:, 3*numcostate:4*numcostate, 2:3] + 1) * (32 - 15) / 2 + 15
        v2_cur_55 = (model_output['model_in_cn'][:, 3*numcostate:4*numcostate, 4:5] + 1) * (32 - 15) / 2 + 15
        d1_opt_55 = (model_output['model_in_cn'][:, 3*numcostate:4*numcostate, 1:2] + 1) * (105 - 15) / 2 + 15 + v1_cur_55 * dt
        d2_opt_55 = (model_output['model_in_cn'][:, 3*numcostate:4*numcostate, 3:4] + 1) * (105 - 15) / 2 + 15 + v2_cur_55 * dt

        d1_11_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        d1_15_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        d1_51_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        d1_55_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        v1_11_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        v1_15_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        v1_51_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        v1_55_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        d2_11_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        d2_15_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        d2_51_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        d2_55_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        v2_11_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        v2_15_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        v2_51_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)
        v2_55_mask = torch.ones((1, numcostate, 1), dtype=torch.bool).to(device)

        d1_11_mask[torch.where(d1_opt_11 > max_d)] = False
        d1_11_mask[torch.where(d1_opt_11 < min_d)] = False
        d2_11_mask[torch.where(d2_opt_11 > max_d)] = False
        d2_11_mask[torch.where(d2_opt_11 < min_d)] = False
        v1_11_mask[torch.where(v1_opt_11 > max_v)] = False
        v1_11_mask[torch.where(v1_opt_11 < min_v)] = False
        v2_11_mask[torch.where(v2_opt_11 > max_v)] = False
        v2_11_mask[torch.where(v2_opt_11 < min_v)] = False

        d1_15_mask[torch.where(d1_opt_15 > max_d)] = False
        d1_15_mask[torch.where(d1_opt_15 < min_d)] = False
        d2_15_mask[torch.where(d2_opt_15 > max_d)] = False
        d2_15_mask[torch.where(d2_opt_15 < min_d)] = False
        v1_15_mask[torch.where(v1_opt_15 > max_v)] = False
        v1_15_mask[torch.where(v1_opt_15 < min_v)] = False
        v2_15_mask[torch.where(v2_opt_15 > max_v)] = False
        v2_15_mask[torch.where(v2_opt_15 < min_v)] = False

        d1_51_mask[torch.where(d1_opt_51 > max_d)] = False
        d1_51_mask[torch.where(d1_opt_51 < min_d)] = False
        d2_51_mask[torch.where(d2_opt_51 > max_d)] = False
        d2_51_mask[torch.where(d2_opt_51 < min_d)] = False
        v1_51_mask[torch.where(v1_opt_51 > max_v)] = False
        v1_51_mask[torch.where(v1_opt_51 < min_v)] = False
        v2_51_mask[torch.where(v2_opt_51 > max_v)] = False
        v2_51_mask[torch.where(v2_opt_51 < min_v)] = False

        d1_55_mask[torch.where(d1_opt_55 > max_d)] = False
        d1_55_mask[torch.where(d1_opt_55 < min_d)] = False
        d2_55_mask[torch.where(d2_opt_55 > max_d)] = False
        d2_55_mask[torch.where(d2_opt_55 < min_d)] = False
        v1_55_mask[torch.where(v1_opt_55 > max_v)] = False
        v1_55_mask[torch.where(v1_opt_55 < min_v)] = False
        v2_55_mask[torch.where(v2_opt_55 > max_v)] = False
        v2_55_mask[torch.where(v2_opt_55 < min_v)] = False

        # find the index: the state should be removed
        mask_11_idx = torch.argmin(torch.cat((d1_11_mask.sum().unsqueeze(0), v1_11_mask.sum().unsqueeze(0),
                                              d2_11_mask.sum().unsqueeze(0), v2_11_mask.sum().unsqueeze(0)), dim=0))
        mask_15_idx = torch.argmin(torch.cat((d1_15_mask.sum().unsqueeze(0), v1_15_mask.sum().unsqueeze(0),
                                               d2_15_mask.sum().unsqueeze(0), v2_15_mask.sum().unsqueeze(0)), dim=0))
        mask_51_idx = torch.argmin(torch.cat((d1_51_mask.sum().unsqueeze(0), v1_51_mask.sum().unsqueeze(0),
                                               d2_51_mask.sum().unsqueeze(0), v2_51_mask.sum().unsqueeze(0)), dim=0))
        mask_55_idx = torch.argmin(torch.cat((d1_55_mask.sum().unsqueeze(0), v1_55_mask.sum().unsqueeze(0),
                                                d2_55_mask.sum().unsqueeze(0), v2_55_mask.sum().unsqueeze(0)), dim=0))

        mask_11 = torch.cat((d1_11_mask, v1_11_mask, d2_11_mask, v2_11_mask), dim=2)[:, :, mask_11_idx].unsqueeze(2)
        mask_15 = torch.cat((d1_15_mask, v1_15_mask, d2_15_mask, v2_15_mask), dim=2)[:, :, mask_15_idx].unsqueeze(2)
        mask_51 = torch.cat((d1_51_mask, v1_51_mask, d2_51_mask, v2_51_mask), dim=2)[:, :, mask_51_idx].unsqueeze(2)
        mask_55 = torch.cat((d1_55_mask, v1_55_mask, d2_55_mask, v2_55_mask), dim=2)[:, :, mask_55_idx].unsqueeze(2)

        d1_scaled_opt_11 = (2 * (d1_opt_11 - 15) / (105 - 15) - 1).to(device)
        v1_scaled_opt_11 = (2 * (v1_opt_11 - 15) / (32 - 15) - 1).to(device)
        d2_scaled_opt_11 = (2 * (d2_opt_11 - 15) / (105 - 15) - 1).to(device)
        v2_scaled_opt_11 = (2 * (v2_opt_11 - 15) / (32 - 15) - 1).to(device)

        d1_scaled_opt_15 = (2 * (d1_opt_15 - 15) / (105 - 15) - 1).to(device)
        v1_scaled_opt_15 = (2 * (v1_opt_15 - 15) / (32 - 15) - 1).to(device)
        d2_scaled_opt_15 = (2 * (d2_opt_15 - 15) / (105 - 15) - 1).to(device)
        v2_scaled_opt_15 = (2 * (v2_opt_15 - 15) / (32 - 15) - 1).to(device)

        d1_scaled_opt_51 = (2 * (d1_opt_51 - 15) / (105 - 15) - 1).to(device)
        v1_scaled_opt_51 = (2 * (v1_opt_51 - 15) / (32 - 15) - 1).to(device)
        d2_scaled_opt_51 = (2 * (d2_opt_51 - 15) / (105 - 15) - 1).to(device)
        v2_scaled_opt_51 = (2 * (v2_opt_51 - 15) / (32 - 15) - 1).to(device)

        d1_scaled_opt_55 = (2 * (d1_opt_55 - 15) / (105 - 15) - 1).to(device)
        v1_scaled_opt_55 = (2 * (v1_opt_55 - 15) / (32 - 15) - 1).to(device)
        d2_scaled_opt_55 = (2 * (d2_opt_55 - 15) / (105 - 15) - 1).to(device)
        v2_scaled_opt_55 = (2 * (v2_opt_55 - 15) / (32 - 15) - 1).to(device)

        if num == 29:
            time_opt = (torch.ones(numcostate, 1) * start_time).unsqueeze(0).to(device)
        else:
            time_opt = (model_output['model_in_cn'][:, :numcostate, :1] - dt).to(device)

        coords1_opt_11 = torch.cat((d1_scaled_opt_11, v1_scaled_opt_11, d2_scaled_opt_11, v2_scaled_opt_11), dim=2)
        coords2_opt_11 = torch.cat((d2_scaled_opt_11, v2_scaled_opt_11, d1_scaled_opt_11, v1_scaled_opt_11), dim=2)

        coords1_opt_15 = torch.cat((d1_scaled_opt_15, v1_scaled_opt_15, d2_scaled_opt_15, v2_scaled_opt_15), dim=2)
        coords2_opt_15 = torch.cat((d2_scaled_opt_15, v2_scaled_opt_15, d1_scaled_opt_15, v1_scaled_opt_15), dim=2)

        coords1_opt_51 = torch.cat((d1_scaled_opt_51, v1_scaled_opt_51, d2_scaled_opt_51, v2_scaled_opt_51), dim=2)
        coords2_opt_51 = torch.cat((d2_scaled_opt_51, v2_scaled_opt_51, d1_scaled_opt_51, v1_scaled_opt_51), dim=2)

        coords1_opt_55 = torch.cat((d1_scaled_opt_55, v1_scaled_opt_55, d2_scaled_opt_55, v2_scaled_opt_55), dim=2)
        coords2_opt_55 = torch.cat((d2_scaled_opt_55, v2_scaled_opt_55, d1_scaled_opt_55, v1_scaled_opt_55), dim=2)

        coords1_opt = torch.cat((coords1_opt_11, coords1_opt_15, coords1_opt_51, coords1_opt_55), dim=1)
        coords2_opt = torch.cat((coords2_opt_11, coords2_opt_15, coords2_opt_51, coords2_opt_55), dim=1)

        label1 = torch.zeros(1, numcostate, 1).to(device)
        label2 = torch.ones(1, numcostate, 1).to(device)
        label1 = label1.repeat(1, 4, 1)
        label2 = label2.repeat(1, 4, 1)

        time_opt = time_opt.repeat(1, 4, 1)

        coords1_costate = torch.cat((time_opt, coords1_opt, label1), dim=2)
        coords2_costate = torch.cat((time_opt, coords2_opt, label2), dim=2)
        coords_costate = torch.cat((coords1_costate, coords2_costate), dim=1)

        coords_mask = torch.cat((mask_11,
                                 mask_15,
                                 mask_51,
                                 mask_55,
                                 mask_11,
                                 mask_15,
                                 mask_51,
                                 mask_55), dim=1)

        coords_data = {'coords_cn': coords_costate,
                       'coords_mask': coords_mask}
        
        return coords_data

    return intersection_sampling
