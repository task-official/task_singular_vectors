"""One deterministic train/validation partition, shared with legacy loaders."""
import torch


def validation_partition(dataset, fraction=.1, maximum=5000, seed=0):
    if not 0 < fraction < 1 or (maximum is not None and maximum <= 0):
        raise ValueError("Invalid validation fraction/maximum")
    size = int(len(dataset) * fraction)
    if maximum is not None:
        size = min(size, maximum)
    if not 0 < size < len(dataset):
        raise ValueError("Validation partition must leave nonempty train and validation sets")
    return torch.utils.data.random_split(
        dataset, [len(dataset)-size, size], generator=torch.Generator().manual_seed(seed))
