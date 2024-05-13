import torch
import os
import scipy.io

torch.manual_seed(0)

def initialize_pontryagin_sampling(dataset, device):
    def intersection_sampling(coords1_final, coords2_final):
        alpha = torch.tensor([1e-6]).to(device)
        start_time = 0.
        numcostate = coords1_final.shape[0]

        boundary_values1_vn = alpha * ((coords1_final[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                               ((coords1_final[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values2_vn = alpha * ((coords2_final[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                               ((coords2_final[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values_vn = torch.cat((boundary_values1_vn,
                                        boundary_values2_vn), dim=0)

        boundary_values1_cn = torch.cat((alpha * torch.ones((numcostate, 1)).to(device),
                                         -2 * ((coords1_final[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18),
                                         torch.zeros((numcostate, 1)).to(device),
                                         torch.zeros((numcostate, 1)).to(device)), dim=1)
        boundary_values2_cn = torch.cat((torch.zeros((numcostate, 1)).to(device),
                                         torch.zeros((numcostate, 1)).to(device),
                                         alpha * torch.ones((numcostate, 1)).to(device),
                                         -2 * ((coords2_final[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18)), dim=1)

        boundary_values_cn = torch.cat((boundary_values1_cn, boundary_values2_cn), dim=0)

        dirichlet_mask_cn = (coords1_final[:, 0, None] == start_time).repeat(1, 4)
        dirichlet_mask_vn = (coords1_final[:, 0, None] == start_time)

        gt_update = {'boundary_values_vn': boundary_values_vn,
                     'boundary_values_cn': boundary_values_cn,
                     'dirichlet_mask_vn': dirichlet_mask_vn,
                     'dirichlet_mask_cn': dirichlet_mask_cn}

        return gt_update

    return intersection_sampling
