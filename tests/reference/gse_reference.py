# Frozen GlobalRawEnergyTopR workspace snapshot before refactoring.
# Source SHA256: f23db653980ac78ab072aeceb52f5ea85b93ddc774318fc1239c8f60b9d01c6d
import torch

def _component_selection_ranks(singular_values, config, selection_mode):
    assert selection_mode == "global_raw_top_r"
    return _global_raw_energy_top_r_ranks(singular_values, config.method.eps)

def _global_raw_energy_top_r_ranks(singular_values, eps):
    candidates = []
    for task_index, singular_value in enumerate(singular_values):
        if singular_value.square().sum() <= eps:
            continue
        for component_index, value in enumerate(singular_value.square()):
            candidates.append((float(value), component_index, task_index))

    max_rank = singular_values[0].shape[0]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = candidates[: min(max_rank, len(candidates))]
    ranks = [0 for _ in singular_values]
    for _, _, task_index in selected:
        ranks[task_index] += 1
    return ranks

def _polar_merge_with_stats(u_list, s_list, v_list, ranks, device, eps):
    total_rank = sum(ranks)
    if total_rank == 0:
        return None, {
            "total_rank": 0,
            "before_norm": 0.0,
            "after_norm": 0.0,
            "deformation_rate": None,
        }

    sum_u = torch.zeros(
        u_list[0].shape[0], total_rank, device=device, dtype=u_list[0].dtype
    )
    sum_s = torch.zeros(total_rank, device=device, dtype=s_list[0].dtype)
    sum_v = torch.zeros(
        total_rank, v_list[0].shape[1], device=device, dtype=v_list[0].dtype
    )
    offset = 0
    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        next_offset = offset + rank
        if rank > 0:
            sum_u[:, offset:next_offset] = u[:, :rank]
            sum_s[offset:next_offset] = s[:rank]
            sum_v[offset:next_offset, :] = v[:rank, :]
        offset = next_offset

    polar_u_left, _, polar_u_right = torch.linalg.svd(sum_u, full_matrices=False)
    polar_v_left, _, polar_v_right = torch.linalg.svd(sum_v, full_matrices=False)
    polar_u = polar_u_left @ polar_u_right
    polar_v_t = polar_v_left @ polar_v_right
    before = (sum_u * sum_s.unsqueeze(0)) @ sum_v
    after = (polar_u * sum_s.unsqueeze(0)) @ polar_v_t
    before_norm = torch.linalg.norm(before)
    after_norm = torch.linalg.norm(after)
    if before_norm <= eps:
        deformation_rate = None
    else:
        deformation_rate = float(
            (torch.linalg.norm(after - before) / before_norm).item()
        )
    return after, {
        "total_rank": int(total_rank),
        "before_norm": float(before_norm.item()),
        "after_norm": float(after_norm.item()),
        "deformation_rate": deformation_rate,
    }

def _average_non_matrix_parameter(task_vectors, key, device):
    for i, task_vector in enumerate(task_vectors):
        vec = task_vector.vector[key].to(device)
        if i == 0:
            new_value = vec.clone()
        else:
            new_value += (vec - new_value) / (i + 1)
    return new_value

def compute_component_selection_tsv_merge(
    task_vectors, config, selection_mode, method_name, return_diagnostics=False
):
    device = config.device
    eps = float(config.method.eps)
    diagnostics = (
        _init_component_selection_diagnostics(config, method_name, selection_mode)
        if return_diagnostics
        else None
    )
    print(f"Computing {method_name}...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            first_vec = task_vectors[0].vector[key]
            if len(first_vec.shape) != 2 or "text_projection" in key:
                new_vector[key] = _average_non_matrix_parameter(
                    task_vectors, key, device
                )
                continue

            u_list, s_list, v_list = [], [], []
            for task_vector in task_vectors:
                vec = task_vector.vector[key].to(device)
                u, s, v = torch.linalg.svd(vec, full_matrices=False)
                u_list.append(u)
                s_list.append(s)
                v_list.append(v)

            ranks = _component_selection_ranks(s_list, config, selection_mode)
            merged, merge_stats = _polar_merge_with_stats(
                u_list, s_list, v_list, ranks, device, eps
            )
            if merged is None:
                new_vector[key] = torch.zeros_like(first_vec, device=device)
            else:
                new_vector[key] = merged
            if return_diagnostics:
                _update_component_selection_diagnostics(
                    diagnostics, key, s_list, ranks, merge_stats, config
                )

    if return_diagnostics:
        return new_vector, _finalize_component_selection_diagnostics(diagnostics)
    return new_vector
