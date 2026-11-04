"""Opt-in SUN397 factory including the GIFs in the local author-data export.

The legacy SUN397 loader remains unchanged. Class directories retain its
sorted label order and display-name conversion. Animated images contribute
only their first frame, decoded to RGB without changing the source file.
"""
from pathlib import Path
from typing import Any, Callable, Mapping

from src.core.types import RuntimeSpec
from src.datasets.splits import validation_partition


def first_frame_rgb(path):
    from PIL import Image

    with open(path, "rb") as source:
        with Image.open(source) as image:
            image.seek(0)
            return image.convert("RGB")


def supported_image(path):
    from torchvision.datasets.folder import IMG_EXTENSIONS

    return str(path).lower().endswith(IMG_EXTENSIONS + (".gif",))


def build_dataset(split_spec: Mapping[str, Any], data_root: Path,
                  preprocess: Callable | None, runtime: RuntimeSpec):
    """Return the requested split and legacy-compatible ordered class names."""
    from torchvision.datasets import ImageFolder

    split = split_spec["split"]
    if split not in ("validation", "test"):
        raise ValueError(f"Unsupported SUN397 evaluation split: {split}")
    dataset = ImageFolder(str(Path(data_root) / str(split_spec["path"])),
                          transform=preprocess, loader=first_frame_rgb,
                          is_valid_file=supported_image)
    if any(len(name) <= 2 or name[1] != "_" for name in dataset.classes):
        raise ValueError("SUN397 requires flattened class directories such as a_abbey")
    names = tuple(name[2:].replace("_", " ") for name in dataset.classes)
    if len(set(names)) != len(names):
        raise ValueError("SUN397 class directories produce duplicate display names")
    if split == "validation":
        _, dataset = validation_partition(
            dataset, fraction=float(split_spec.get("val_fraction", .1)),
            maximum=int(split_spec.get("max_val_samples", 5000)),
            seed=int(split_spec.get("seed", 0)))
    return dataset, names
