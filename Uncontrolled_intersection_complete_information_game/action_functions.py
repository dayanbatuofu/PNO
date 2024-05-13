import torch
import diff_operators

torch.manual_seed(0)

def initialize_pontryagin_sampling(dataset, device):
    def intersection_sampling(model_output):
        x = model_output['model_in_cn']
        costate = model_output['model_out_cn']
        cut_index = x.shape[1] // 2

        # calculate the costate for agent 1 and 2
        lam11_2 = costate[:, :cut_index, 1:2]  # lambda_11
        lam22_2 = costate[:, cut_index:, -1:]  # lambda_22

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

        u_star = {'u1_star': u1_star,
                  'u2_star': u2_star}

        return u_star

    return intersection_sampling
