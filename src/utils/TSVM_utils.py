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
def compute_and_sum_svd_mem_reduction(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

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


def _adaptive_task_weights(config, device):
    if len(config.method.task_weights) == 0:
        weights = torch.ones(len(config.DATASETS), device=device)
    else:
        if len(config.method.task_weights) != len(config.DATASETS):
            raise ValueError(
                "method.task_weights must be empty or match the number of datasets."
            )
        weights = torch.tensor(config.method.task_weights, device=device)
    if torch.any(weights < 0):
        raise ValueError("method.task_weights must be non-negative.")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("method.task_weights must contain at least one positive value.")
    return weights / weight_sum


def _adaptive_rank_from_energy(s, energy_threshold, eps):
    energy = s.square()
    total_energy = energy.sum()
    if total_energy <= eps:
        return 0
    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
    threshold = torch.tensor(energy_threshold, device=s.device)
    return int(torch.searchsorted(cumulative_energy, threshold).item()) + 1


def _adaptive_candidate_rank(s, config, energy_threshold, eps):
    energy = s.square().sum()
    if energy <= eps:
        return 0

    if config.method.rank_selection == "adaptive":
        return _adaptive_rank_from_energy(s, energy_threshold, eps)
    if config.method.rank_selection == "fixed":
        return max(1, int(s.shape[0] / len(config.DATASETS)))

    raise ValueError("method.rank_selection must be one of: adaptive, fixed.")


def _adaptive_merge_coefficients(coefficients, weights, sign_merge, sign_threshold, eps):
    weighted_coefficients = coefficients * weights.view(-1, 1, 1)
    signed_sum = weighted_coefficients.sum(dim=0)

    if sign_merge == "mean":
        return signed_sum

    absolute_sum = (coefficients.abs() * weights.view(-1, 1, 1)).sum(dim=0)
    agreement = signed_sum.abs() / (absolute_sum + eps)

    if sign_merge == "soft":
        return signed_sum * agreement
    if sign_merge == "hard":
        return signed_sum * (agreement >= sign_threshold)
    if sign_merge == "thresholded_soft":
        return signed_sum * agreement * (agreement >= sign_threshold)

    raise ValueError(
        "method.sign_merge must be one of: mean, soft, hard, thresholded_soft."
    )


###############
#### Adaptive TSV Merge
def compute_adaptive_tsv_merge(task_vectors, config):
    """
    Merges task vectors with adaptive per-task TSV ranks and sign-agreement
    filtering in a shared low-rank coordinate system.
    """
    device = config.device
    energy_threshold = float(config.method.energy_threshold)
    final_rank_fraction = float(config.method.final_rank_fraction)
    sign_merge = config.method.sign_merge
    sign_threshold = float(config.method.sign_threshold)
    eps = float(config.method.eps)
    weights = _adaptive_task_weights(config, device)
    if energy_threshold <= 0 or energy_threshold > 1:
        raise ValueError("method.energy_threshold must be in (0, 1].")
    if final_rank_fraction <= 0:
        raise ValueError("method.final_rank_fraction must be positive.")

    print("Computing Adaptive TSV Merge...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            first_vec = task_vectors[0].vector[key]
            if len(first_vec.shape) != 2 or "text_projection" in key:
                for i, task_vector in enumerate(task_vectors):
                    vec = task_vector.vector[key].to(device)
                    if i == 0:
                        new_vector[key] = weights[i] * vec
                    else:
                        new_vector[key] += weights[i] * vec
                continue

            candidates = []
            candidate_ranks = []
            for task_vector in task_vectors:
                vec = task_vector.vector[key].to(device)
                u, s, v = torch.linalg.svd(vec, full_matrices=False)
                rank = _adaptive_candidate_rank(s, config, energy_threshold, eps)
                candidate_ranks.append(rank)
                if rank == 0:
                    candidates.append(torch.zeros_like(vec))
                else:
                    candidates.append(
                        (u[:, :rank] * s[:rank].unsqueeze(0)) @ v[:rank, :]
                    )

            candidate_count = sum(candidate_ranks)
            if candidate_count == 0:
                new_vector[key] = torch.zeros_like(first_vec, device=device)
                continue

            layer_rank = min(first_vec.shape)
            requested_rank = max(1, int(layer_rank * final_rank_fraction))
            common_rank = min(layer_rank, candidate_count, requested_rank)
            print(
                f"Adaptive TSV for {key}: candidate ranks={candidate_ranks}, q={common_rank}"
            )

            weighted_candidates = [
                torch.sqrt(weights[i]) * candidate
                for i, candidate in enumerate(candidates)
            ]
            left_basis_source = torch.cat(weighted_candidates, dim=1)
            right_basis_source = torch.cat(
                [candidate.mT for candidate in weighted_candidates], dim=1
            )

            p, _, _ = torch.linalg.svd(left_basis_source, full_matrices=False)
            q, _, _ = torch.linalg.svd(right_basis_source, full_matrices=False)
            p = p[:, :common_rank]
            q = q[:, :common_rank]

            coefficients = torch.stack(
                [p.mT @ candidate @ q for candidate in candidates], dim=0
            )
            merged_coefficients = _adaptive_merge_coefficients(
                coefficients, weights, sign_merge, sign_threshold, eps
            )
            new_vector[key] = p @ merged_coefficients @ q.mT

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
