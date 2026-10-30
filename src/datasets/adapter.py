"""Explicit, offline evaluation datasets.

Only this adapter understands the historical ``TaskVal`` naming convention.
New datasets can register a factory in DATASET_ADAPTERS. Factories return a
dataset and its ordered class names; they must not download data.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import hashlib
import inspect
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from src.core.types import RuntimeSpec, TaskSpec

DATASET_ADAPTER_VERSION = "offline-splits-v3-hf-observed-rows"


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_digest(*objects):
    """Bind evaluation caches to loaders, factories and dataset classes as well as bytes."""
    paths = set(Path(__file__).parent.glob("*.py"))
    for obj in objects:
        for cls in (obj, *getattr(obj, "__mro__", ())):
            try:
                filename = inspect.getsourcefile(cls)
            except (TypeError, OSError):
                continue
            if filename and Path(filename).is_file():
                paths.add(Path(filename))
    return hashlib.sha256(_json_bytes([(p.name, file_digest(p)) for p in sorted(paths)])).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _tensor_digest(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(_json_bytes(tuple(tensor.shape)))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _hf_image_sources(dataset):
    """Read actual ordered HF rows without decoding/transforming image pixels.

    cache_files alone omits in-memory row indices and external image bytes.
    Inspect only the two columns used by _HFDiskImages, retaining row order.
    """
    from datasets import Image as HFImage
    image_key = next((key for key in ("image", "img", "img_bytes") if key in dataset.column_names), None)
    label_key = next((key for key in ("label", "labels") if key in dataset.column_names), None)
    if image_key is None or label_key is None:
        raise ValueError("hf_disk requires an image/img/img_bytes column and integer label/labels")
    if dataset.format.get("type") is not None:
        raise ValueError("hf_disk requires unformatted source rows; remove saved format/transforms during preparation")
    raw = dataset.with_format(None)
    if isinstance(raw.features.get(image_key), HFImage):
        raw = raw.cast_column(image_key, HFImage(decode=False))
    raw = raw.select_columns([image_key, label_key])
    for row in raw:
        image, label = row[image_key], row[label_key]
        if isinstance(label, bool) or not isinstance(label, Integral):
            raise ValueError("hf_disk labels must be integers")
        path = None
        if isinstance(image, dict):
            if image.get("bytes") is not None:
                image = image["bytes"]
            elif image.get("path"):
                path = Path(image["path"]).resolve()
            else:
                raise ValueError("hf_disk image requires bytes or an existing path")
        if path is not None:
            digest = file_digest(path)
        elif isinstance(image, bytes):
            digest = hashlib.sha256(image).hexdigest()
        else:
            raise ValueError("hf_disk images require encoded bytes or image path/bytes records")
        yield digest, int(label), path


def sample_membership(dataset, root, specification):
    """Known physical sample IDs, in row order; opaque factories may expose sample_ids."""
    if isinstance(dataset, Subset):
        parent = sample_membership(dataset.dataset, root, specification)
        return None if parent is None else tuple(parent[int(index)] for index in dataset.indices)
    explicit = getattr(dataset, "sample_ids", None)
    if explicit is not None:
        values = tuple("provided:" + _json_bytes(value).decode() for value in explicit)
        if len(values) != len(dataset):
            raise ValueError("sample_ids must identify every dataset row")
        return values
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        known = tuple("file:" + str(path) if path is not None else None
                      for _, _, path in _hf_image_sources(hf_dataset))
        # Embedded bytes do not identify physical source rows across independent
        # exports. Do not invent IDs or impose an undeclared pixel-dedup policy.
        return known if any(value is not None for value in known) else None
    samples = getattr(dataset, "samples", getattr(dataset, "_samples", None))
    if samples is not None:
        return tuple("file:" + str(Path(name).resolve()) for name, _ in samples)
    images = getattr(dataset, "_image_files", getattr(dataset, "_images", None))
    if images is not None:
        return tuple("file:" + str(Path(name).resolve()) for name in images)
    if isinstance(dataset, TensorDataset) and specification.get("adapter") == "tensor_file":
        namespace = "tensor-file:" + str((Path(root) / str(specification["path"])).resolve())
    else:
        location = getattr(dataset, "root", None)
        partition = getattr(dataset, "train", getattr(dataset, "split", getattr(dataset, "_split", None)))
        if location is None or partition is None:
            return None
        namespace = f"{type(dataset).__module__}.{type(dataset).__name__}:{Path(location).resolve()}:{partition}"
    return tuple(f"{namespace}:{index}" for index in range(len(dataset)))


def dataset_digest(dataset: Dataset, root: str | Path) -> str:
    """Bind cache identity to membership, labels and available source bytes.

    Opaque custom datasets must supply a stable ``identity``. A length alone is
    deliberately insufficient: it would reuse accuracy after changing images.
    """
    root = Path(root).resolve()
    if isinstance(dataset, Subset):
        return hashlib.sha256(_json_bytes({
            "parent": dataset_digest(dataset.dataset, root),
            "indices": list(map(int, dataset.indices)),
        })).hexdigest()
    if isinstance(dataset, TensorDataset):
        return hashlib.sha256(_json_bytes([_tensor_digest(t) for t in dataset.tensors])).hexdigest()
    samples = getattr(dataset, "samples", getattr(dataset, "_samples", None))
    if samples is None:
        image_files = getattr(dataset, "_image_files", getattr(dataset, "_images", None))
        image_labels = getattr(dataset, "_labels", None)
        if image_files is not None and image_labels is not None:
            if len(image_files) != len(image_labels):
                raise ValueError("Image path/label count mismatch")
            samples = zip(image_files, image_labels)
    if samples is not None:
        digest = hashlib.sha256()
        for filename, target in samples:
            path = Path(filename).resolve()
            try:
                relative = str(path.relative_to(root))
            except ValueError:
                relative = str(path)
            digest.update(_json_bytes([relative, int(target), file_digest(path)]))
        return digest.hexdigest()
    files = getattr(dataset, "_FILES", None)
    base_folder = getattr(dataset, "_base_folder", None)
    physical_split = getattr(dataset, "_split", None)
    if isinstance(files, Mapping) and base_folder is not None and physical_split in files:
        # torchvision PCAM stores images/labels in local HDF5 files rather than
        # sample lists. Hash those files as well as the selected subset indices.
        names = [Path(base_folder) / value[0] for value in files[physical_split].values()]
        return hashlib.sha256(_json_bytes([(path.name, file_digest(path)) for path in names])).hexdigest()
    data = getattr(dataset, "data", None)
    labels = getattr(dataset, "targets", getattr(dataset, "labels", None))
    if data is not None and labels is not None:
        return hashlib.sha256(_json_bytes([_tensor_digest(data), _tensor_digest(labels)])).hexdigest()
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        digest = hashlib.sha256()
        for image_digest, label, _ in _hf_image_sources(hf_dataset):
            digest.update(_json_bytes([image_digest, label]))
        return digest.hexdigest()
    identity = getattr(dataset, "identity", None)
    if identity is not None:
        return hashlib.sha256(_json_bytes(identity)).hexdigest()
    raise ValueError(f"Cannot identify dataset {type(dataset).__name__}; expose samples/data or a stable identity")


def _legacy(spec: Mapping[str, Any], root: Path, preprocess: Callable | None, runtime: RuntimeSpec):
    from src.datasets.registry import get_dataset, registry

    profile = spec.get("profile", "legacy_local")
    if profile != "legacy_local":
        raise ValueError(f"Unknown legacy dataset profile {profile!r}; supply explicit split adapters for a different protocol")
    name = str(spec["name"])
    split = str(spec["split"])
    if name.endswith("Val"):
        raise ValueError("Use a base task name and explicit split, not a Val suffix")
    if name == "FER2013":
        # _hf_disk applies data_root; pass a relative path, including when the
        # configured root is itself relative (the default is "datasets").
        path = Path("fer2013_dataset") / ("train" if split == "validation" else "test")
        base, names = _hf_disk({"path": str(path)}, root, preprocess, runtime)
        if split == "validation":
            base = _validation_subset(base, spec)
        return base, names
    if split == "validation" and name + "Val" in registry:
        if any(key in spec for key in ("val_fraction", "max_val_samples", "seed")):
            raise ValueError(f"{name} uses explicit validation files; partition options do not apply")
    options = dict(location=str(root), batch_size=runtime.batch_size,
                   num_workers=runtime.num_workers, download=False,
                   val_fraction=float(spec.get("val_fraction", .1)),
                   max_val_samples=int(spec.get("max_val_samples", 5000)), seed=int(spec.get("seed", 0)))
    if split == "test":
        bundle = get_dataset(name, preprocess, **options)
    else:
        bundle = get_dataset(name + "Val", preprocess, **options)
    def mapping(dataset):
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return getattr(dataset, "class_to_idx", None)
    train_mapping = mapping(getattr(bundle, "train_dataset", None))
    evaluation_mapping = mapping(bundle.test_dataset)
    if train_mapping is not None and evaluation_mapping is not None and train_mapping != evaluation_mapping:
        raise ValueError(f"{name}: class-to-label mapping differs between training and {split}")
    return bundle.test_dataset, tuple(bundle.classnames)


def _validation_subset(dataset: Dataset, spec: Mapping[str, Any]) -> Subset:
    from src.datasets.splits import validation_partition
    return validation_partition(dataset, float(spec.get("val_fraction", .1)),
                                int(spec.get("max_val_samples", 5000)), int(spec.get("seed", 0)))[1]


def _imagefolder(spec: Mapping[str, Any], root: Path, preprocess: Callable | None, runtime: RuntimeSpec):
    from torchvision.datasets import ImageFolder
    path = root / str(spec["path"])
    dataset = ImageFolder(str(path), transform=preprocess)
    names = tuple(spec.get("class_names", dataset.classes))
    if len(names) != len(dataset.classes) or len(set(names)) != len(names):
        raise ValueError("ImageFolder class_names must name each sorted class directory exactly once")
    return dataset, names


def _tensor_file(spec: Mapping[str, Any], root: Path, preprocess: Callable | None, runtime: RuntimeSpec):
    path = root / str(spec["path"])
    data = torch.load(path, map_location="cpu")
    images, labels = data["images"], data["labels"]
    if len(images) != len(labels):
        raise ValueError(f"Image/label count mismatch in {path}")
    names = spec.get("class_names", data.get("class_names"))
    if names is None:
        raise ValueError("tensor_file requires ordered class_names")
    return TensorDataset(images, labels), tuple(names)


class _HFDiskImages(Dataset):
    def __init__(self, dataset, preprocess):
        self.hf_dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        import io
        from PIL import Image
        row = self.hf_dataset[index]
        image = row.get("image", row.get("img", row.get("img_bytes")))
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, dict):
            image = Image.open(io.BytesIO(image["bytes"])) if image.get("bytes") is not None else Image.open(image["path"])
        if not isinstance(image, Image.Image):
            raise ValueError("hf_disk rows require an image/img/img_bytes field")
        image = image.convert("RGB")
        return self.preprocess(image) if self.preprocess else image, row.get("label", row.get("labels"))


def _hf_disk(spec: Mapping[str, Any], root: Path, preprocess: Callable | None, runtime: RuntimeSpec):
    from datasets import Dataset as HFDataset, load_from_disk
    path = root / str(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"Offline dataset missing: {path}; prepare it explicitly before evaluation")
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, HFDataset):
        raise ValueError("hf_disk.path must point to one saved Dataset, not a DatasetDict; select its split during preparation")
    feature = dataset.features.get("label", dataset.features.get("labels"))
    names = spec.get("class_names", getattr(feature, "names", None))
    if names is None and "fer2013" in str(path).lower():
        names = ("angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral")
    if names is None:
        raise ValueError("hf_disk requires ordered class_names or a ClassLabel feature")
    return _HFDiskImages(dataset, preprocess), tuple(names)


DATASET_ADAPTERS: dict[str, Callable] = {
    "legacy": _legacy, "imagefolder": _imagefolder,
    "tensor_file": _tensor_file, "hf_disk": _hf_disk,
}


@dataclass
class EvaluationTask:
    id: str
    split: str
    loader: DataLoader
    class_names: tuple[Any, ...]
    identity: dict[str, Any]
    membership: frozenset[str] | None = None

    @property
    def size(self) -> int:
        return len(self.loader.dataset)


def inspect_task(task: TaskSpec, split: str, data_root: str, preprocess: Callable | None, runtime: RuntimeSpec) -> EvaluationTask:
    if split not in ("validation", "test"):
        raise ValueError(f"Unsupported evaluation split: {split}")
    description = dict(getattr(task, split))
    declared = description.setdefault("split", split)
    if declared != split:
        raise ValueError(f"{task.id}: requested {split}, dataset specification declares {declared}")
    adapter = description.get("adapter", "legacy")
    if adapter not in DATASET_ADAPTERS:
        raise ValueError(f"Unknown dataset adapter {adapter!r}; register it in DATASET_ADAPTERS")
    description.setdefault("name", task.id)
    root = Path(data_root).expanduser()
    dataset, names = DATASET_ADAPTERS[adapter](description, root, preprocess, runtime)
    if len(dataset) == 0 or not names:
        raise ValueError(f"{task.id}/{split} has no samples or classes")
    if task.class_names and tuple(task.class_names) != tuple(names):
        raise ValueError(f"{task.id}: configured class order differs from dataset class order")
    members = sample_membership(dataset, root, description)
    known_members = tuple(value for value in members if value is not None) if members is not None else ()
    libraries = {}
    for package in ("torchvision", "datasets", "Pillow", "h5py"):
        try:
            libraries[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            libraries[package] = "not_installed"
    identity = {
        "libraries": libraries,
        "task": task.id, "split": split, "adapter": adapter,
        "adapter_version": DATASET_ADAPTER_VERSION,
        "implementation_sha256": implementation_digest(DATASET_ADAPTERS[adapter], type(dataset)),
        "specification": description, "class_names": names,
        "sample_digest": dataset_digest(dataset, root), "size": len(dataset),
        "membership_digest": hashlib.sha256(_json_bytes(sorted(set(known_members)))).hexdigest() if known_members else None,
        "membership_status": "unavailable" if not known_members else ("complete" if len(known_members) == len(dataset) else "partial"),
    }
    loader = DataLoader(dataset, batch_size=runtime.batch_size, num_workers=runtime.num_workers,
                        shuffle=False, drop_last=False)
    return EvaluationTask(task.id, split, loader, tuple(names), identity,
                          frozenset(known_members) if known_members else None)
