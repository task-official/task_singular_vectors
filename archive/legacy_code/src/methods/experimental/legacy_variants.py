"""Preserved local research variants, moved from src/utils/TSVM_utils.py.

The protected TSV-M entry delegates to the canonical kernel. Other historical
variants retain their original numerical rules; do not silently normalize them.
"""
import torch
import math


def compute_svd_dict(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector in the provided datasets and stores the results in a dictionary.

    Args:
        task_vectors (list): A list of task vector objects, where each task vector contains a dictionary of matrices to be decomposed.
        config (object): Configuration object containing the list of datasets under the attribute `DATASETS`.

    Returns:
        dict: A dictionary where each key is a dataset name and the value is another dictionary containing the SVD components ('u', 's', 'v') for each matrix in the task vector.
        If a matrix is not 2-dimensional or contains 'text_projection' in its key, it is stored under the key 'dim1' without decomposition.
    """
    sv_reduction = 1 / len(config.DATASETS)
    with torch.no_grad():
        svd_dict = {}
        for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
            svd_dict[dataset] = {}
            print(f"Computing SVD for {dataset}...")
            for key in task_vector.vector:
                svd_dict[dataset][key] = {}
                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(
                        task_vector.vector[key], full_matrices=False
                    )
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    temp_u = torch.zeros_like(u)
                    # select only the first reduced_index_s columns of u and place them
                    temp_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    svd_dict[dataset][key]["u"] = temp_u

                    temp_s = torch.zeros_like(s)
                    temp_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]

                    svd_dict[dataset][key]["s"] = temp_s  # s_reduced

                    # select only the first reduced_index_s rows of v and place them
                    temp_v = torch.zeros_like(v)
                    temp_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                    svd_dict[dataset][key]["v"] = temp_v

                    # temp_mat = temp_u @ torch.diag_embed(temp_s) @ temp_v
                else:
                    svd_dict[dataset][key]["dim1"] = task_vector.vector[key]
    return svd_dict


def sum_svd_dict(svd_dict, config):
    """
    Sums the Singular Value Decomposition (SVD) components from multiple datasets and computes a new vector.

    Args:
        svd_dict (dict): A dictionary containing SVD components for multiple datasets. The structure of the dictionary is expected to be:
                         {
                             dataset_name: {
                                 key: {
                                     "u": tensor,
                                     "s": tensor,
                                     "v": tensor,
                                     "dim1": tensor (optional)
                                 }
                             }
                         }
        config (object): A configuration object that contains a list of dataset names under the attribute `DATASETS`.

    Returns:
        dict: A dictionary containing the merged SVD components or averaged "dim1" values for each key.
    """
    print("Summing SVD...")
    new_vector = {}
    for key in svd_dict[config.DATASETS[0]]:
        if "u" in svd_dict[config.DATASETS[0]][key].keys():
            sum_u = sum([svd_dict[dataset][key]["u"] for dataset in config.DATASETS])
            sum_s = sum([svd_dict[dataset][key]["s"] for dataset in config.DATASETS])
            sum_v = sum([svd_dict[dataset][key]["v"] for dataset in config.DATASETS])

            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            new_vector[key] = torch.linalg.multi_dot(
                (
                    u_u,
                    v_u,
                    torch.diag(sum_s),
                    u_v,
                    v_v,
                )
            )
        else:
            for i, dataset in enumerate(config.DATASETS, start=1):
                if i == 1:
                    new_vector[key] = svd_dict[dataset][key]["dim1"]
                else:
                    new_vector[key] += (
                        svd_dict[dataset][key]["dim1"] - new_vector[key]
                    ) / i
    return new_vector


###############
##### LOSSLESS Orthogonalization
def compute_and_sum_svd_mem_reduction_lossless(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector and merge the results.

    This function performs the following steps:
    1. Iterates over each layer in the task vectors.
    2. For each layer, it computes the SVD of the corresponding matrix if it is a 2D tensor excluding "text_projection".
    3. Concatenate the U_i, S_i, and V_i matrices from the SVD across all tasks.
    4. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    5. After concatenating the SVD components, recomputes the SVD of the summed U and V matrices and constructs the merged layer.

    Args:
        task_vectors (list): A list of task vectors, where each task vector is a dictionary containing the vectors for each task.
        config (object): A configuration object containing the device and dataset information.

    Returns:
        dict: A dictionary containing the new vectors after summing the SVD components.
    """
    # becareful wit vit-l on 20 task it does not fit in GPU or in 64 GB RAM (try without last layer)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):

                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros(
                            u.shape[0], u.shape[1] * config.num_tasks, device=device
                        )
                        sum_s = torch.zeros(
                            s.shape[0] * config.num_tasks, device=device
                        )
                        sum_v = torch.zeros(
                            v.shape[0] * config.num_tasks, v.shape[1], device=device
                        )
                    reduced_index_s = s.shape[0]

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )

    return new_vector


###############
##### LOSSLESS EIGENDECOMP
def compute_and_sum_svd_mem_reduction_lossless_eigen(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector and merge the results.

    This function performs the following steps:
    1. Iterates over each layer in the task vectors.
    2. For each layer, it computes the SVD of the corresponding matrix if it is a 2D tensor excluding "text_projection".
    3. Concatenate the U_i, S_i, and V_i matrices from the SVD across all tasks.
    4. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    5. After concatenating the SVD components, recomputes the eigendecomposition of the summed U and V matrices and constructs the merged layer.

    Args:
        task_vectors (list): A list of task vectors, where each task vector is a dictionary containing the vectors for each task.
        config (object): A configuration object containing the device and dataset information.

    Returns:
        dict: A dictionary containing the new vectors after merging the SVD components.
    """
    # becareful wit vit-l on 20 task it does not fit in GPU or in 64 GB RAM (try without last layer)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):

                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros(
                            u.shape[0], u.shape[1] * config.num_tasks, device=device
                        )
                        sum_s = torch.zeros(
                            s.shape[0] * config.num_tasks, device=device
                        )
                        sum_v = torch.zeros(
                            v.shape[0] * config.num_tasks, v.shape[1], device=device
                        )
                    reduced_index_s = s.shape[0]

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                sum_s, indices = torch.sort(sum_s, stable=True)

                sum_u = torch.index_select(sum_u, 1, indices)
                l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
                u_orth = (
                    q_u
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
                    @ q_u.mT
                )

                sum_v = torch.index_select(sum_v, 0, indices)

                l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
                v_orth = (
                    q_v
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
                    @ q_v.mT
                )

                new_vector[key] = torch.linalg.multi_dot(  # bool_mask *
                    (
                        u_orth,
                        torch.diag(sum_s),
                        v_orth,
                    )
                )

    return new_vector


###############
#### TSV Merge Orthogonalization


###############
#### TSV Merge Eigendecomp
def compute_and_sum_svd_mem_reduction_2(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the eigendecomposition is performed instead of the SVD for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                sum_s, indices = torch.sort(sum_s, stable=True)

                sum_u = torch.index_select(sum_u, 1, indices)
                l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
                u_orth = (
                    q_u
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
                    @ q_u.mT
                )

                sum_v = torch.index_select(sum_v, 0, indices)

                l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
                v_orth = (
                    q_v
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
                    @ q_v.mT
                )

                new_vector[key] = torch.linalg.multi_dot(  # bool_mask *
                    (
                        sum_u,
                        u_orth,
                        torch.diag(sum_s),
                        v_orth,
                        sum_v,
                    )
                )

    return new_vector


def _adaptive_rank_from_energy(s, energy_threshold, eps):
    energy = s.square()
    total_energy = energy.sum()
    if total_energy <= eps:
        return 0
    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
    threshold = torch.tensor(energy_threshold, device=s.device)
    return int(torch.searchsorted(cumulative_energy, threshold).item()) + 1


def _base_tsv_rank(s, config):
    return int(s.shape[0] / len(config.DATASETS))


def _normalized_svd_energy(s, eps):
    total_energy = s.square().sum()
    if total_energy <= eps:
        return None
    return s.square() / total_energy


def _adaptive_rank_budget(singular_values, config, energy_threshold, eps):
    ranks = [
        _adaptive_rank_from_energy(s, energy_threshold, eps) for s in singular_values
    ]
    base_rank = _base_tsv_rank(singular_values[0], config)
    budget = base_rank * len(config.DATASETS)
    if budget == 0:
        return [0 for _ in ranks]

    nonzero_tasks = [
        i for i, s in enumerate(singular_values) if s.square().sum() > eps
    ]
    min_rank = 1 if len(nonzero_tasks) <= budget else 0
    ranks = [
        max(min_rank, rank) if i in nonzero_tasks else 0
        for i, rank in enumerate(ranks)
    ]
    normalized_energies = [_normalized_svd_energy(s, eps) for s in singular_values]

    while sum(ranks) > budget:
        candidates = [
            (float(normalized_energies[i][ranks[i] - 1]), i)
            for i in nonzero_tasks
            if ranks[i] > min_rank
        ]
        if not candidates:
            break
        _, task_index = min(candidates, key=lambda item: item[0])
        ranks[task_index] -= 1

    while sum(ranks) < budget:
        candidates = [
            (float(normalized_energies[i][ranks[i]]), i)
            for i in nonzero_tasks
            if (
                normalized_energies[i] is not None
                and ranks[i] < singular_values[i].shape[0]
            )
        ]
        if not candidates:
            break
        _, task_index = max(candidates, key=lambda item: (item[0], -item[1]))
        ranks[task_index] += 1

    return ranks


def _strict_adaptive_rank_budget(
    singular_values, num_tasks, energy_threshold, eps
):
    normalized_energies = [_normalized_svd_energy(s, eps) for s in singular_values]
    active_tasks = [
        task_index
        for task_index, energy in enumerate(normalized_energies)
        if energy is not None
    ]
    base_rank = int(singular_values[0].shape[0] / num_tasks)
    budget = base_rank * num_tasks
    ranks = [0 for _ in singular_values]
    if budget == 0 or len(active_tasks) == 0:
        return ranks

    if budget < len(active_tasks):
        ordered_tasks = sorted(
            active_tasks,
            key=lambda task_index: (
                -float(normalized_energies[task_index][0]),
                task_index,
            ),
        )
        for task_index in ordered_tasks[:budget]:
            ranks[task_index] = 1
        return ranks

    for task_index in active_tasks:
        ranks[task_index] = _adaptive_rank_from_energy(
            singular_values[task_index], energy_threshold, eps
        )

    while sum(ranks) > budget:
        candidates = [
            (float(normalized_energies[task_index][ranks[task_index] - 1]), task_index)
            for task_index in active_tasks
            if ranks[task_index] > 1
        ]
        if not candidates:
            raise RuntimeError("Adaptive rank budget cannot be reduced to target.")
        _, task_index = min(candidates, key=lambda item: (item[0], item[1]))
        ranks[task_index] -= 1

    while sum(ranks) < budget:
        candidates = [
            (float(normalized_energies[task_index][ranks[task_index]]), task_index)
            for task_index in active_tasks
            if ranks[task_index] < singular_values[task_index].shape[0]
        ]
        if not candidates:
            raise RuntimeError("Adaptive rank budget cannot be increased to target.")
        _, task_index = max(candidates, key=lambda item: (item[0], -item[1]))
        ranks[task_index] += 1

    return ranks


def _global_normalized_energy_top_r_ranks(singular_values, eps):
    normalized_energies = [_normalized_svd_energy(s, eps) for s in singular_values]
    candidates = []
    for task_index, energy in enumerate(normalized_energies):
        if energy is None:
            continue
        for component_index, value in enumerate(energy):
            candidates.append(
                (float(value), component_index, task_index)
            )

    max_rank = singular_values[0].shape[0]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = candidates[: min(max_rank, len(candidates))]
    ranks = [0 for _ in singular_values]
    for _, _, task_index in selected:
        ranks[task_index] += 1
    return ranks


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


def _component_selection_ranks(singular_values, config, selection_mode):
    eps = float(config.method.eps)
    energy_threshold = float(getattr(config.method, "energy_threshold", 0.95))
    active = [s.square().sum() > eps for s in singular_values]
    if selection_mode == "tsvm":
        base_rank = _base_tsv_rank(singular_values[0], config)
        return [base_rank if is_active else 0 for is_active in active]
    if selection_mode == "adaptive":
        return _strict_adaptive_rank_budget(
            singular_values,
            len(config.DATASETS),
            energy_threshold,
            eps,
        )
    if selection_mode == "global_top_r":
        return _global_normalized_energy_top_r_ranks(singular_values, eps)
    if selection_mode == "global_raw_top_r":
        return _global_raw_energy_top_r_ranks(singular_values, eps)
    if selection_mode == "per_task_95":
        return [
            _adaptive_rank_from_energy(s, energy_threshold, eps)
            for s in singular_values
        ]
    if selection_mode == "full":
        return [s.shape[0] if is_active else 0 for s, is_active in zip(singular_values, active)]
    raise ValueError(f"Unknown component selection mode: {selection_mode}")


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


def _init_component_selection_diagnostics(config, method_name, selection_mode):
    return {
        "method": method_name,
        "selection_mode": selection_mode,
        "datasets": list(config.DATASETS),
        "rank": {"per_layer": {}, "total_per_layer": {}},
        "selection_energy": {
            "per_layer": {},
            "selected_energy_by_task": [0.0 for _ in config.DATASETS],
            "total_energy_by_task": [0.0 for _ in config.DATASETS],
        },
        "procrustes": {"per_layer": {}},
        "component_width": {"sum_selected_components": 0, "sum_max_rank": 0},
    }


def _update_component_selection_diagnostics(
    diagnostics, key, singular_values, ranks, merge_stats, config
):
    max_rank = singular_values[0].shape[0]
    base_rank = int(max_rank / len(config.DATASETS))
    total_rank = sum(ranks)
    diagnostics["rank"]["per_layer"][key] = {
        "r": int(max_rank),
        "base_rank": int(base_rank),
        "ranks": {
            dataset: int(rank)
            for dataset, rank in zip(diagnostics["datasets"], ranks)
        },
        "total_rank": int(total_rank),
    }
    diagnostics["rank"]["total_per_layer"][key] = int(total_rank)
    diagnostics["component_width"]["sum_selected_components"] += int(total_rank)
    diagnostics["component_width"]["sum_max_rank"] += int(max_rank)

    layer_selection = {}
    for task_index, (dataset, s, rank) in enumerate(
        zip(diagnostics["datasets"], singular_values, ranks)
    ):
        total_energy = s.square().sum()
        selected_energy = s[:rank].square().sum()
        total_value = float(total_energy.item())
        selected_value = float(selected_energy.item())
        layer_selection[dataset] = (
            selected_value / total_value if total_value > 0 else 0.0
        )
        diagnostics["selection_energy"]["selected_energy_by_task"][task_index] += selected_value
        diagnostics["selection_energy"]["total_energy_by_task"][task_index] += total_value
    diagnostics["selection_energy"]["per_layer"][key] = layer_selection

    diagnostics["procrustes"]["per_layer"][key] = {
        "r": int(max_rank),
        "total_rank": int(total_rank),
        "width_ratio": float(total_rank / max_rank) if max_rank > 0 else 0.0,
        **merge_stats,
    }


def _finalize_component_selection_diagnostics(diagnostics):
    selected = diagnostics["selection_energy"].pop("selected_energy_by_task")
    total = diagnostics["selection_energy"].pop("total_energy_by_task")
    task_scores = {
        dataset: selected_value / total_value if total_value > 0 else 0.0
        for dataset, selected_value, total_value in zip(
            diagnostics["datasets"], selected, total
        )
    }
    diagnostics["selection_energy"]["task_scores"] = task_scores
    values = list(task_scores.values())
    diagnostics["selection_energy"]["avg_score"] = sum(values) / len(values)
    diagnostics["selection_energy"]["min_score"] = min(values)
    sum_rank = diagnostics["component_width"]["sum_selected_components"]
    sum_max_rank = diagnostics["component_width"]["sum_max_rank"]
    diagnostics["component_width"]["normalized_width"] = (
        sum_rank / sum_max_rank if sum_max_rank > 0 else 0.0
    )
    return diagnostics


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


def _orthogonalized_tsv_merge(
    u_list, s_list, v_list, ranks, device, return_reconstructions=False
):
    total_rank = sum(ranks)
    if total_rank == 0:
        if return_reconstructions:
            zero_reconstructions = [
                torch.zeros(
                    u_list[0].shape[0],
                    v_list[0].shape[1],
                    device=device,
                    dtype=u_list[0].dtype,
                )
                for _ in ranks
            ]
            return None, zero_reconstructions
        return None

    sum_u = torch.zeros(
        u_list[0].shape[0],
        total_rank,
        device=device,
        dtype=u_list[0].dtype,
    )
    sum_s = torch.zeros(total_rank, device=device, dtype=s_list[0].dtype)
    sum_v = torch.zeros(
        total_rank,
        v_list[0].shape[1],
        device=device,
        dtype=v_list[0].dtype,
    )

    offset = 0
    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        if rank == 0:
            continue
        next_offset = offset + rank
        sum_u[:, offset:next_offset] = u[:, :rank]
        sum_s[offset:next_offset] = s[:rank]
        sum_v[offset:next_offset, :] = v[:rank, :]
        offset = next_offset

    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
    merged = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
    if not return_reconstructions:
        return merged

    u_orth = u_u @ v_u
    v_orth = u_v @ v_v
    reconstructions = []
    offset = 0
    for rank in ranks:
        next_offset = offset + rank
        if rank == 0:
            reconstructions.append(
                torch.zeros(
                    u_list[0].shape[0],
                    v_list[0].shape[1],
                    device=device,
                    dtype=u_list[0].dtype,
                )
            )
        else:
            reconstructions.append(
                torch.linalg.multi_dot(
                    (
                        u_orth[:, offset:next_offset],
                        torch.diag(sum_s[offset:next_offset]),
                        v_orth[offset:next_offset, :],
                    )
                )
            )
        offset = next_offset
    return merged, reconstructions


def _top_eigenvectors(matrix, rank):
    _, eigenvectors = torch.linalg.eigh(matrix)
    return eigenvectors[:, -rank:]


def _mean_adaptive_rank(ranks, max_rank):
    nonzero_ranks = [rank for rank in ranks if rank > 0]
    if len(nonzero_ranks) == 0:
        return 0
    return min(max_rank, math.ceil(sum(nonzero_ranks) / len(nonzero_ranks)))


def _adaptive_subspace_merge(
    u_list, s_list, v_list, ranks, config, weighted, return_reconstructions=False
):
    device = config.device
    eps = float(config.method.eps)
    common_rank = _mean_adaptive_rank(
        ranks, min(u_list[0].shape[0], v_list[0].shape[1])
    )
    if common_rank == 0:
        if return_reconstructions:
            zero_reconstructions = [
                torch.zeros(
                    u_list[0].shape[0],
                    v_list[0].shape[1],
                    device=device,
                    dtype=u_list[0].dtype,
                )
                for _ in ranks
            ]
            return None, zero_reconstructions, common_rank
        return None

    g_u = torch.zeros(
        u_list[0].shape[0],
        u_list[0].shape[0],
        device=device,
        dtype=u_list[0].dtype,
    )
    g_v = torch.zeros(
        v_list[0].shape[1],
        v_list[0].shape[1],
        device=device,
        dtype=v_list[0].dtype,
    )
    task_weight = 1 / len(config.DATASETS)

    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        if rank == 0:
            continue
        u_rank = u[:, :rank]
        v_rank = v[:rank, :].mT
        if weighted:
            energy = s[:rank].square()
            total_energy = energy.sum()
            if total_energy <= eps:
                continue
            g_u += task_weight * (
                (u_rank * energy.unsqueeze(0)) @ u_rank.mT
            ) / total_energy
            g_v += task_weight * (
                (v_rank * energy.unsqueeze(0)) @ v_rank.mT
            ) / total_energy
        else:
            g_u += task_weight * (u_rank @ u_rank.mT) / rank
            g_v += task_weight * (v_rank @ v_rank.mT) / rank

    p = _top_eigenvectors(g_u, common_rank)
    q = _top_eigenvectors(g_v, common_rank)
    c_merge = torch.zeros(
        common_rank,
        common_rank,
        device=device,
        dtype=u_list[0].dtype,
    )
    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        if rank == 0:
            continue
        candidate = (u[:, :rank] * s[:rank].unsqueeze(0)) @ v[:rank, :]
        c_merge += p.mT @ candidate @ q

    merged = p @ c_merge @ q.mT
    if not return_reconstructions:
        return merged

    reconstructions = []
    for u, s, v, rank in zip(u_list, s_list, v_list, ranks):
        if rank == 0:
            reconstructions.append(
                torch.zeros(
                    u_list[0].shape[0],
                    v_list[0].shape[1],
                    device=device,
                    dtype=u_list[0].dtype,
                )
            )
            continue
        candidate = (u[:, :rank] * s[:rank].unsqueeze(0)) @ v[:rank, :]
        task_coefficients = p.mT @ candidate @ q
        reconstructions.append(p @ task_coefficients @ q.mT)

    return merged, reconstructions, common_rank


def _average_non_matrix_parameter(task_vectors, key, device):
    for i, task_vector in enumerate(task_vectors):
        vec = task_vector.vector[key].to(device)
        if i == 0:
            new_value = vec.clone()
        else:
            new_value += (vec - new_value) / (i + 1)
    return new_value


def _init_merge_diagnostics(config, method_name):
    return {
        "method": method_name,
        "datasets": list(config.DATASETS),
        "reconstruction": {
            "squared_error_by_task": [0.0 for _ in config.DATASETS],
            "squared_norm_by_task": [0.0 for _ in config.DATASETS],
        },
        "sti": {
            "per_layer": {},
        },
        "rank": {
            "per_layer": {},
            "total_per_layer": {},
        },
    }


def _compute_layer_sti(reconstructions, eps):
    left_vectors = []
    right_vectors = []
    singular_values = []
    for reconstruction in reconstructions:
        u, s, v = torch.linalg.svd(reconstruction, full_matrices=False)
        rank = int((s > eps).sum().item())
        if rank == 0:
            continue
        left_vectors.append(u[:, :rank])
        right_vectors.append(v[:rank, :].mT)
        singular_values.append(s[:rank])

    if len(singular_values) == 0:
        return 0.0

    sum_u = torch.cat(left_vectors, dim=1)
    sum_v = torch.cat(right_vectors, dim=1)
    sum_s = torch.cat(singular_values)
    identity = torch.eye(sum_s.shape[0], device=sum_s.device, dtype=sum_s.dtype)
    interference = (
        (sum_u.mT @ sum_u - identity)
        @ torch.diag(sum_s)
        @ (sum_v.mT @ sum_v - identity)
    )
    return float(torch.sum(torch.abs(interference)).item())


def _selection_energy_values(original_vectors, selection_ranks, eps):
    values = []
    for original_vector, rank in zip(original_vectors, selection_ranks):
        _, s, _ = torch.linalg.svd(original_vector, full_matrices=False)
        total_energy = s.square().sum()
        if total_energy <= eps:
            values.append((0.0, 0.0))
        else:
            values.append(
                (
                    float(s[:rank].square().sum().item()),
                    float(total_energy.item()),
                )
            )
    return values


def _update_merge_diagnostics(
    diagnostics,
    key,
    original_vectors,
    reconstructions,
    ranks,
    base_rank=None,
    selection_ranks=None,
    subspace_rank=None,
    max_rank=None,
):
    eps = 1e-12
    for task_index, (original_vector, reconstruction) in enumerate(
        zip(original_vectors, reconstructions)
    ):
        squared_error = (original_vector - reconstruction).square().sum()
        squared_norm = original_vector.square().sum()
        diagnostics["reconstruction"]["squared_error_by_task"][task_index] += float(
            squared_error.item()
        )
        diagnostics["reconstruction"]["squared_norm_by_task"][task_index] += float(
            squared_norm.item()
        )

    diagnostics["sti"]["per_layer"][key] = _compute_layer_sti(reconstructions, eps)
    diagnostics["rank"]["per_layer"][key] = {
        "r": int(max_rank if max_rank is not None else min(original_vectors[0].shape)),
        "ranks": {
            dataset: int(rank) for dataset, rank in zip(diagnostics["datasets"], ranks)
        },
        "total_rank": int(sum(ranks)),
    }
    diagnostics["rank"]["total_per_layer"][key] = int(sum(ranks))
    if base_rank is not None:
        diagnostics["rank"]["per_layer"][key]["base_rank"] = int(base_rank)

    if selection_ranks is not None:
        values = _selection_energy_values(original_vectors, selection_ranks, eps)
        selection = diagnostics.setdefault(
            "selection_energy",
            {
                "selected_energy_by_task": [0.0 for _ in diagnostics["datasets"]],
                "total_energy_by_task": [0.0 for _ in diagnostics["datasets"]],
            },
        )
        for task_index, (selected_energy, total_energy) in enumerate(values):
            selection["selected_energy_by_task"][task_index] += selected_energy
            selection["total_energy_by_task"][task_index] += total_energy

    if subspace_rank is not None:
        subspace = diagnostics.setdefault("subspace", {"per_layer": {}})
        r = int(max_rank if max_rank is not None else min(original_vectors[0].shape))
        subspace["per_layer"][key] = {
            "q": int(subspace_rank),
            "r": r,
            "q_over_r": float(subspace_rank / r) if r > 0 else 0.0,
        }


def _finalize_merge_diagnostics(diagnostics):
    datasets = diagnostics["datasets"]
    squared_errors = diagnostics["reconstruction"].pop("squared_error_by_task")
    squared_norms = diagnostics["reconstruction"].pop("squared_norm_by_task")
    task_scores = {}
    for dataset, squared_error, squared_norm in zip(
        datasets, squared_errors, squared_norms
    ):
        if squared_norm == 0:
            score = 0.0
        else:
            score = 1.0 - squared_error / squared_norm
        task_scores[dataset] = score
    diagnostics["reconstruction"]["task_scores"] = task_scores
    scores = list(task_scores.values())
    diagnostics["reconstruction"]["avg_score"] = sum(scores) / len(scores)
    diagnostics["reconstruction"]["min_score"] = min(scores)

    diagnostics["sti"]["total"] = sum(diagnostics["sti"]["per_layer"].values())

    if "selection_energy" in diagnostics:
        selected_energy = diagnostics["selection_energy"].pop(
            "selected_energy_by_task"
        )
        total_energy = diagnostics["selection_energy"].pop("total_energy_by_task")
        task_scores = {}
        for dataset, selected, total in zip(datasets, selected_energy, total_energy):
            task_scores[dataset] = selected / total if total > 0 else 0.0
        diagnostics["selection_energy"]["task_scores"] = task_scores
        scores = list(task_scores.values())
        diagnostics["selection_energy"]["avg_score"] = sum(scores) / len(scores)
        diagnostics["selection_energy"]["min_score"] = min(scores)

    return diagnostics


###############
#### Adaptive TSV Merge Rank
def compute_adaptive_tsv_merge(task_vectors, config, return_diagnostics=False):
    """
    Reallocates TSV-M's per-layer rank budget across tasks using normalized
    singular-value energy, then uses the same Procrustes reconstruction as TSV-M.
    """
    device = config.device
    energy_threshold = float(config.method.energy_threshold)
    eps = float(config.method.eps)
    if energy_threshold <= 0 or energy_threshold > 1:
        raise ValueError("method.energy_threshold must be in (0, 1].")

    diagnostics = (
        _init_merge_diagnostics(config, "AdaptiveTSV") if return_diagnostics else None
    )
    print("Computing Adaptive TSV Rank...")
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
            original_vectors = []
            for task_vector in task_vectors:
                vec = task_vector.vector[key].to(device)
                if return_diagnostics:
                    original_vectors.append(vec)
                u, s, v = torch.linalg.svd(vec, full_matrices=False)
                u_list.append(u)
                s_list.append(s)
                v_list.append(v)

            ranks = _adaptive_rank_budget(s_list, config, energy_threshold, eps)
            if return_diagnostics:
                merged, reconstructions = _orthogonalized_tsv_merge(
                    u_list,
                    s_list,
                    v_list,
                    ranks,
                    device,
                    return_reconstructions=True,
                )
            else:
                merged = _orthogonalized_tsv_merge(u_list, s_list, v_list, ranks, device)
            if merged is None:
                new_vector[key] = torch.zeros_like(first_vec, device=device)
            else:
                print(f"Adaptive TSV Rank for {key}: ranks={ranks}")
                new_vector[key] = merged
            if return_diagnostics:
                _update_merge_diagnostics(
                    diagnostics,
                    key,
                    original_vectors,
                    reconstructions,
                    ranks,
                    base_rank=_base_tsv_rank(s_list[0], config),
                    selection_ranks=ranks,
                )

    if return_diagnostics:
        return new_vector, _finalize_merge_diagnostics(diagnostics)
    return new_vector


###############
#### Adaptive TSV Merge Subspace
def compute_adaptive_tsv_subspace(
    task_vectors, config, weighted=False, return_diagnostics=False
):
    """
    Builds shared left/right subspaces from 95%-energy task subspaces and merges
    task vectors in the shared coordinate system. When weighted=True, singular
    value energy weights the projection sums.
    """
    device = config.device
    energy_threshold = float(config.method.energy_threshold)
    eps = float(config.method.eps)
    if energy_threshold <= 0 or energy_threshold > 1:
        raise ValueError("method.energy_threshold must be in (0, 1].")

    method_label = "EnergySubspace" if weighted else "Subspace"
    method_name = "AdaptiveSubspaceTSV" if weighted else "SubspaceTSV"
    diagnostics = (
        _init_merge_diagnostics(config, method_name) if return_diagnostics else None
    )
    print(f"Computing Adaptive TSV {method_label}...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            first_vec = task_vectors[0].vector[key]
            if len(first_vec.shape) != 2 or "text_projection" in key:
                new_vector[key] = _average_non_matrix_parameter(
                    task_vectors, key, device
                )
                continue

            u_list, s_list, v_list, ranks = [], [], [], []
            original_vectors = []
            for task_vector in task_vectors:
                vec = task_vector.vector[key].to(device)
                if return_diagnostics:
                    original_vectors.append(vec)
                u, s, v = torch.linalg.svd(vec, full_matrices=False)
                rank = _adaptive_rank_from_energy(s, energy_threshold, eps)
                u_list.append(u)
                s_list.append(s)
                v_list.append(v)
                ranks.append(rank)

            if return_diagnostics:
                merged, reconstructions, common_rank = _adaptive_subspace_merge(
                    u_list,
                    s_list,
                    v_list,
                    ranks,
                    config,
                    weighted,
                    return_reconstructions=True,
                )
            else:
                merged = _adaptive_subspace_merge(
                    u_list, s_list, v_list, ranks, config, weighted
                )
            if merged is None:
                new_vector[key] = torch.zeros_like(first_vec, device=device)
            else:
                print(f"Adaptive TSV {method_label} for {key}: ranks={ranks}")
                new_vector[key] = merged
            if return_diagnostics:
                _update_merge_diagnostics(
                    diagnostics,
                    key,
                    original_vectors,
                    reconstructions,
                    ranks,
                    subspace_rank=common_rank,
                    max_rank=min(first_vec.shape),
                )

    if return_diagnostics:
        return new_vector, _finalize_merge_diagnostics(diagnostics)
    return new_vector


###############
#### Rank Reduction TV
def compute_and_sum_svd_mem_reduction_rank_reduction(task_vectors, config):
    """
    Compute and sum the Singular Value Decomposition (SVD) of task vectors with rank reduction.

    This function performs SVD on the vectors in `task_vectors` and reduces their rank based on the
    number of tasks specified in `config.DATASETS`. The reduced vectors are then summed together.

    Args:
        task_vectors (list): A list of task vector objects. Each object should have a `vector` attribute
                             which is a dictionary where keys are vector names and values are tensors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and summation.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v,
                    )
                )
    return new_vector


def compute_and_sum_svd_mem_reduction_dummy(task_vectors, config):
    """To perform dummy operations."""
    sv_reduction = 1 / 8
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i in range(0, 8):
                if (
                    len(task_vectors[0].vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    if i == 0:
                        u, s, v = torch.linalg.svd(
                            task_vectors[0].vector[key], full_matrices=False
                        )
                        reduced_index_s = int(s.shape[0] * sv_reduction)

                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u)
                        sum_s = torch.zeros_like(s)
                        sum_v = torch.zeros_like(v)

                        # select only the first reduced_index_s columns of u and place them
                        sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                            :, :reduced_index_s
                        ]
                        sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                            :reduced_index_s
                        ]
                        # select only the first reduced_index_s rows of v and place them
                        sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                            :reduced_index_s, :
                        ]
                    else:
                        # generate u vectors orthogonal to the previous ones
                        # generate v vectors orthogonal to the previous ones
                        print("dummy")
                        u = torch.nn.functional.normalize(
                            torch.randn_like(sum_u), p=2, dim=-2
                        )
                        v = torch.nn.functional.normalize(
                            torch.randn_like(sum_v), p=2, dim=-1
                        )

                        # select only the first reduced_index_s columns of u and place them
                        sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                            :, :reduced_index_s
                        ]
                        sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                            :reduced_index_s
                        ]
                        # select only the first reduced_index_s rows of v and place them
                        sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                            :reduced_index_s, :
                        ]

                else:
                    if i == 0:
                        new_vector[key] = task_vectors[0].vector[key]
                    # else:
                    #     new_vector[key] += (
                    #         task_vector.vector[key] - new_vector[key]
                    #     ) / (i + 1)

            if (
                len(task_vectors[0].vector[key].shape) == 2
                and "text_projection" not in key
            ):

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v,
                    )
                )

    return new_vector

from src.compat.kernels import compute_and_sum_svd_mem_reduction
