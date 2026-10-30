"""Training helpers are loaded only when requested by training code."""
__all__ = ["parse_arguments", "initialize_wandb", "wandb_log"]


def __getattr__(name):
    if name == "parse_arguments":
        from .args import parse_arguments
        return parse_arguments
    if name in ("initialize_wandb", "wandb_log"):
        from . import logging
        return getattr(logging, name)
    raise AttributeError(name)
