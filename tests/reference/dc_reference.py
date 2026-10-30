# Frozen official numerical oracle. MIT; see THIRD_PARTY.md.
# Tobeginwith/DC-Merge@089af77a378562272461e0f041217ec440a4a230
import torch
from tqdm.auto import tqdm

def dc_merge(task_vectors, config):
    """
    DC-Merge
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing DC-Merge...")

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            trunc_vecs = []
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
                vec = task_vector.vector[key].to(device)

                if (len(task_vector.vector[key].shape) == 2) and ("text_projection" not in key):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    low_rank_per_task = int(s.shape[0] * sv_reduction)

                    sum_u[:, i * low_rank_per_task : (i + 1) * low_rank_per_task] = u[:, :low_rank_per_task]
                    sum_v[i * low_rank_per_task : (i + 1) * low_rank_per_task, :] = v[:low_rank_per_task, :]
                    
                    trunc_recon = u[:, :low_rank_per_task] @ torch.diag(s[:low_rank_per_task]) @ v[:low_rank_per_task, :]
                    trunc_vecs.append(trunc_recon)

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                
                cover_space_u = u_u @ v_u
                cover_space_vT = u_v @ v_v
                
                Ms = [torch.linalg.multi_dot((cover_space_u.T, trunc_vecs[i].to(device), cover_space_vT.T, )) for i in range(len(config.DATASETS))]
                filtered_Ms = keep_topk_percent(Ms, 1e-3)
                agg_M = ties_small(filtered_Ms)
                mask_M = torch.zeros_like(agg_M)
                d_per_task = mask_M.shape[0] // len(config.DATASETS)
                for i in range(len(config.DATASETS)):
                    mask_M[i * d_per_task : (i+1) * d_per_task, i * d_per_task : (i+1) * d_per_task] = 1
                
                new_vector[key] = torch.linalg.multi_dot((cover_space_u, agg_M * mask_M, cover_space_vT, ))
                
    return new_vector


def keep_topk_percent(tensor_list, percent=0.1):
    new_list = []
    for i, t in enumerate(tensor_list):
        flat_abs = t.abs().view(-1)
        k = max(1, int(percent * flat_abs.numel()))
        threshold = torch.topk(flat_abs, k).values.min()
        mask = t.abs() >= threshold
        new_t = t * mask
        new_list.append(new_t)
    return new_list


def ties_small(mat_list):
    stacked = torch.stack(mat_list, dim=0)
    summed = stacked.sum(dim=0)
    summed_sign = torch.sign(summed)
    elem_sign = torch.sign(stacked)
    mask = (elem_sign == summed_sign.unsqueeze(0))
    count = mask.sum(dim=0)
    selected = stacked * mask
    res = selected.sum(dim=0) / count.clamp(min=1)
    res = torch.where(summed == 0, torch.zeros_like(res), res)

    return res


