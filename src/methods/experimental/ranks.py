"""Historical normalized-energy rank allocation; task-order ties are retained."""
import math
import torch

def _adaptive_rank_from_energy(s, energy_threshold, eps):
    energy = s.square()
    total_energy = energy.sum()
    if total_energy <= eps:
        return 0
    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
    threshold = torch.tensor(energy_threshold, device=s.device)
    return min(s.numel(), int(torch.searchsorted(cumulative_energy, threshold).item()) + 1)


def _normalized_svd_energy(s, eps):
    total_energy = s.square().sum()
    if total_energy <= eps:
        return None
    return s.square() / total_energy


def _adaptive_rank_budget(singular_values, num_tasks, energy_threshold, eps):
    ranks = [
        _adaptive_rank_from_energy(s, energy_threshold, eps) for s in singular_values
    ]
    base_rank = int(singular_values[0].shape[0] / num_tasks)
    budget = base_rank * num_tasks
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


def _mean_adaptive_rank(ranks, max_rank):
    nonzero_ranks = [rank for rank in ranks if rank > 0]
    if len(nonzero_ranks) == 0:
        return 0
    return min(max_rank, math.ceil(sum(nonzero_ranks) / len(nonzero_ranks)))
