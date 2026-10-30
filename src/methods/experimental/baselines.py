"""Historical random dummy baseline and individual checkpoint evaluation."""
from dataclasses import dataclass
import torch
from src.core.numerics import cached_merge, is_matrix
from src.core.types import MergeOutput
from .eigen_tsv import EigenParams


@dataclass(frozen=True)
class IndividualParams:
    reconstructed: bool = False

    def __post_init__(self):
        if not isinstance(self.reconstructed, bool):
            raise ValueError("reconstructed must be a boolean")


def validate_individual(inputs, params):
    if len(inputs.task_order) != 1:
        raise ValueError("individual requires exactly one task")


@torch.no_grad()
def individual(inputs, params, context, fitted_state=None):
    validate_individual(inputs, params)
    source = inputs.finetuned[inputs.task_order[0]]
    weights = {}
    for name in inputs.parameter_order:
        base, tuned = inputs.base.tensor(name), source.tensor(name)
        weights[name] = (base + (tuned-base) if params.reconstructed and base.is_floating_point() else tuned).clone()
    return MergeOutput(weights, {"method": "individual", "reconstructed": params.reconstructed})


@torch.no_grad()
def merge(inputs, params, context, fitted_state=None):
    def build():
        generator = torch.Generator(device="cpu").manual_seed(context.runtime.seed)
        update = {}
        for name in inputs.parameter_order:
            base = inputs.base.tensor(name)
            if not base.is_floating_point():
                continue
            delta = inputs.delta(inputs.task_order[0], name)
            if not is_matrix(name, delta):
                update[name] = delta.clone()
                continue
            u, s, v = torch.linalg.svd(delta, full_matrices=False)
            left, values, right = torch.zeros_like(u), torch.zeros_like(s), torch.zeros_like(v)
            rank = int(s.numel() / 8)
            for index in range(8):
                if index:
                    u = torch.nn.functional.normalize(torch.empty_like(left).normal_(generator=generator), p=2, dim=-2)
                    v = torch.nn.functional.normalize(torch.empty_like(right).normal_(generator=generator), p=2, dim=-1)
                start, end = index*rank, (index+1)*rank
                left[:, start:end], values[start:end], right[start:end, :] = u[:, :rank], s[:rank], v[:rank, :]
            update[name] = torch.linalg.multi_dot((left, torch.diag(values), right))
        return update, {"method": "historical_dummy", "random_slots": 7,
                        "rng_policy": "isolated_cpu_generator_v2", "seed": context.runtime.seed}
    return cached_merge(inputs, params, context, "dummy_rng_isolated_v2", build)
