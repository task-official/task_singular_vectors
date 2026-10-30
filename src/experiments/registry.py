"""The single wiring table for method kernels and validation recipes."""
from __future__ import annotations

from functools import lru_cache

from src.core.types import MethodDefinition


@lru_cache(maxsize=1)
def methods() -> dict[str, MethodDefinition]:
    from src.methods import arithmetic, consensus, dc_merge, gse_tsv, iso_cts, sera, ties, tsvm
    from src.selection import consensus as consensus_selection
    from src.selection import search
    from src.selection import sera as sera_selection
    from src.methods.experimental import adaptive_tsv, subspace_tsv, component_selection, eigen_tsv, single_task_svd, baselines

    upstream = "https://github.com/AntoAndGar/task_singular_vectors/tree/a6c97188f7aa0bb20753de4f2939be89dd64f1c3"
    return {
        "zeroshot": MethodDefinition("zeroshot", arithmetic.merge_zeroshot,
                                    arithmetic.ZeroShotParams, search.select, "upstream_a6c9718", upstream),
        "weight_average": MethodDefinition("weight_average", arithmetic.merge_average,
                                         arithmetic.AverageParams, search.select, "upstream_a6c9718", upstream),
        "task_arithmetic": MethodDefinition("task_arithmetic", arithmetic.merge_task_arithmetic,
                                          arithmetic.TaskArithmeticParams, search.select, "upstream_a6c9718", upstream),
        "ties": MethodDefinition("ties", ties.merge, ties.TiesParams, search.select,
                                "tsv_global_ties_a6c9718", upstream),
        "consensus_ta": MethodDefinition("consensus_ta", consensus.merge, consensus.ConsensusParams,
                                        consensus_selection.select, "tall_strict_gt_v1",
                                        upstream.replace("/tree/", "/blob/") + "/src/utils/tallmask_utils.py"),
        "tsvm": MethodDefinition("tsvm", tsvm.merge, tsvm.TsvmParams, search.select,
                                "upstream_a6c9718", upstream),
        "gse_tsv": MethodDefinition("gse_tsv", gse_tsv.merge, gse_tsv.GseTsvParams, search.select,
                                   "global_raw_energy_top_r_device_v2", "local:GlobalRawEnergyTopR"),
        "iso_cts": MethodDefinition("iso_cts", iso_cts.merge, iso_cts.IsoCtsParams, search.select,
                                   "iso_d34010af8ba1_v1", "https://github.com/danielm1405/iso-merging/blob/d34010af8ba129ef4d17a91ca9943b413fe06f8c/src/utils/iso.py"),
        "dc_merge": MethodDefinition("dc_merge", dc_merge.merge, dc_merge.DcMergeParams, search.select,
                                    "dc_fft_089af77a3785_v1", "https://github.com/Tobeginwith/DC-Merge/blob/089af77a378562272461e0f041217ec440a4a230/vision_fft_merge/src/utils/DC_utils.py"),
        "sera": MethodDefinition("sera", sera.merge, sera.SeraParams, sera_selection.select,
                                "paper_2609_24517v1_local_v2", "https://arxiv.org/html/2609.24517v1", sera.validate_inputs),
        "adaptive_tsv": MethodDefinition("adaptive_tsv", adaptive_tsv.merge, adaptive_tsv.AdaptiveParams,
                                         search.select, "historical_endpoint_bounded_v2", "archive:AdaptiveTSV"),
        "subspace_tsv": MethodDefinition("subspace_tsv", subspace_tsv.merge, subspace_tsv.SubspaceParams,
                                         search.select, "historical_endpoint_bounded_v2", "archive:SubspaceTSV"),
        "component_selection": MethodDefinition("component_selection", component_selection.merge, component_selection.ComponentParams,
                                         search.select, "historical_endpoint_bounded_v2", "archive:component-selection"),
        "eigen_tsv": MethodDefinition("eigen_tsv", eigen_tsv.merge, eigen_tsv.EigenParams,
                                      search.select, "historical_v1", "archive:TSVM_2"),
        "single_task_svd": MethodDefinition("single_task_svd", single_task_svd.merge, single_task_svd.RankParams,
                                      search.select, "historical_padded_v1", "archive:eval_single_task_for_rank.py", single_task_svd.validate_inputs),
        "dummy": MethodDefinition("dummy", baselines.merge, eigen_tsv.EigenParams,
                                   search.select, "historical_rng_isolated_v2", "archive:dummy"),
        "individual": MethodDefinition("individual", baselines.individual, baselines.IndividualParams,
                                   search.select, "direct_or_explicit_reconstructed_v1", "local:individual", baselines.validate_individual),
    }


def get_method(method_id: str) -> MethodDefinition:
    try:
        return methods()[method_id]
    except KeyError:
        raise ValueError(f"Unknown method {method_id!r}; choose from {', '.join(methods())}") from None
