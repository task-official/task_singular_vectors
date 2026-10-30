"""Predeclared single-task rank curves; test values never choose a rank."""
from src.evaluation.single_task import run_individuals

DEFAULT_FRACTIONS = (.01, .02, .03, .04, .05, .07, .1, .15, .2, .3, .5)


def run_rank_study(spec, fractions=DEFAULT_FRACTIONS, *, run=None):
    from src.core.numerics import validate_fraction
    values = tuple(fractions)
    if not values or len(set(values)) != len(values):
        raise ValueError("Rank fractions must be nonempty and unique")
    for value in values:
        validate_fraction(value, "rank_fraction")
    results = []
    for value in values:
        options = {} if run is None else {"run": run}
        results.extend(run_individuals(spec, method="single_task_svd",
                                       parameters={"rank_fraction": value}, **options))
    return results
