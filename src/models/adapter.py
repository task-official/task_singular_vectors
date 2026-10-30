"""Offline CLIP construction and supplied classification-head validation.

No pretrained weights or heads are generated here. The base checkpoint is the
source of model parameters; model construction uses pretrained=None explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
import hashlib
import inspect
import json
import math
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.core.types import ExperimentSpec
from src.datasets.adapter import file_digest

ADAPTER_VERSION = "openclip-offline-v2-stable-preprocess"
SUPPORTED_MODELS = ("ViT-B-32", "ViT-B-16", "ViT-L-14")


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def preprocessing_identity(preprocess: Any) -> dict[str, Any]:
    """Describe transform settings and implementation without object addresses.

    ``repr(Compose(...))`` includes function addresses, so it cannot identify
    preprocessing across processes. Inspect the actual ordered transform tree
    instead. Unsupported opaque state fails explicitly rather than using repr.
    """
    active: set[int] = set()

    def implementation(value):
        description = {"module": value.__module__, "qualname": value.__qualname__}
        try:
            source = inspect.getsourcefile(value)
        except TypeError:
            source = None
        if source is not None and Path(source).is_file():
            description["module_sha256"] = file_digest(source)
        else:
            raise TypeError(f"Preprocessing implementation has no inspectable source: {description}")
        description["source_sha256"] = hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
        if inspect.isfunction(value):
            # Several lambdas may share a qualified name and even a source
            # line. Bytecode and constants distinguish their computations.
            from types import CodeType
            def code_state(code):
                def constant(item):
                    if isinstance(item, CodeType):
                        return code_state(item)
                    if isinstance(item, (tuple, frozenset)):
                        return [constant(part) for part in sorted(item, key=repr)] if isinstance(item, frozenset) else [constant(part) for part in item]
                    return {"type": type(item).__name__, "value": repr(item)}
                return {"bytecode": code.co_code.hex(), "constants": [constant(item) for item in code.co_consts],
                        "names": code.co_names, "variables": code.co_varnames,
                        "freevars": code.co_freevars, "cellvars": code.co_cellvars,
                        "arguments": (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount),
                        "flags": code.co_flags}
            description["code_sha256"] = hashlib.sha256(json.dumps(code_state(value.__code__), sort_keys=True).encode()).hexdigest()
        return description

    def encode(value):
        if isinstance(value, Enum):
            return {"enum": implementation(type(value)), "value": encode(value.value)}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Preprocessing state must be finite")
            return value
        if id(value) in active:
            raise TypeError("Cyclic preprocessing state cannot be fingerprinted")
        active.add(id(value))
        try:
            if isinstance(value, Mapping):
                if any(not isinstance(key, str) for key in value):
                    raise TypeError("Preprocessing mappings require string keys")
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return {"sequence": type(value).__name__, "items": [encode(item) for item in value]}
            if isinstance(value, (set, frozenset)):
                return {"set": sorted((encode(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))}
            if isinstance(value, torch.Tensor):
                from src.core.checkpoints import tensor_fingerprint
                return {"tensor_sha256": tensor_fingerprint({"value": value})}
            if isinstance(value, partial):
                return {"partial": encode(value.func), "args": encode(value.args), "keywords": encode(value.keywords)}
            if inspect.isfunction(value):
                return {"function": implementation(value),
                        "defaults": encode(value.__defaults__), "kwdefaults": encode(value.__kwdefaults__),
                        "closure": encode(tuple(cell.cell_contents for cell in value.__closure__ or ()))}
            if callable(value) and hasattr(value, "__dict__"):
                return {"transform": implementation(type(value)), "state": encode(vars(value))}
            raise TypeError(f"Unsupported preprocessing state type: {type(value).__module__}.{type(value).__qualname__}")
        finally:
            active.remove(id(value))

    return {"format": "structured-preprocess-v1", "transform": encode(preprocess)}


@dataclass
class SuppliedHead:
    weight: torch.Tensor
    bias: torch.Tensor
    normalize: bool
    class_names: tuple[Any, ...]
    identity: dict[str, Any]


def load_head(path: str | Path) -> SuppliedHead:
    """Load a trusted author head or a tensor-only head prepared by the user."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Classification head missing: {path}; prepare it before running")

    def normalization_flag(value, *, metadata=False):
        if metadata and isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        if isinstance(value, bool):
            return value
        raise ValueError(f"Head {path}: normalize must be a boolean (true/false in safetensors metadata)")

    names: tuple[Any, ...] = ()
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            content = {key: reader.get_tensor(key) for key in reader.keys()}
            info = reader.metadata() or {}
        normalize = normalization_flag(info.get("normalize", "true"), metadata=True)
        if "class_names" in info:
            import json
            names = tuple(json.loads(info["class_names"]))
    else:
        content = torch.load(path, map_location="cpu")
        normalize = normalization_flag(getattr(content, "normalize", True))
        if isinstance(content, Mapping):
            normalize = normalization_flag(content.get("normalize", True))
            names = tuple(content.get("class_names", ()))
            if "state_dict" in content:
                content = content["state_dict"]
        elif isinstance(content, nn.Module):
            names = tuple(getattr(content, "class_names", ()))
            content = content.state_dict()
        else:
            raise ValueError(f"Unsupported classification head: {path}")
    if "weight" not in content:
        raise ValueError(f"Head {path} has no weight tensor")
    weight = content["weight"].detach().cpu()
    if weight.ndim != 2 or not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError(f"Head {path} requires a finite two-dimensional floating weight")
    bias = content.get("bias", torch.zeros(weight.shape[0], dtype=weight.dtype)).detach().cpu()
    if bias.shape != (weight.shape[0],) or not torch.isfinite(bias).all():
        raise ValueError(f"Head {path} bias does not match its classes")
    return SuppliedHead(weight, bias, normalize, names, {
        "sha256": file_digest(path), "shape": list(weight.shape),
        "normalize": normalize, "class_names": names,
    })


class _OfflineEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.encode_image(images)


class OpenClipAdapter:
    """A single reusable encoder and small task heads; no method-dependent logic."""
    def __init__(self, spec: ExperimentSpec, parameter_names: Sequence[str]):
        if spec.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported CLIP architecture {spec.model}; register a model adapter explicitly")
        if spec.pretrained_tag != "openai":
            raise ValueError("This adapter's preprocessing is pinned to OpenAI CLIP; use a separate adapter for other families")
        if spec.runtime.compute_dtype != "float32":
            raise ValueError("The reference evaluation adapter currently requires float32")
        if str(spec.runtime.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; explicitly select runtime.device=cpu if intended")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.model, pretrained=None,
        )
        names = tuple(parameter_names)
        # Match the serialized ImageEncoder layout, including legacy unused text
        # embeddings; do not silently discard keys during strict state loading.
        if not any(name.startswith("model.transformer.") for name in names) and hasattr(model, "transformer"):
            delattr(model, "transformer")
        self.encoder = _OfflineEncoder(model).to(spec.runtime.device).float().eval()
        self.device = torch.device(spec.runtime.device)
        self.preprocess = preprocess
        self.heads = {task.id: load_head(task.head) for task in spec.tasks}
        feature_dim = getattr(model.visual, "output_dim", None)
        for task_id, head in self.heads.items():
            if feature_dim is not None and head.weight.shape[1] != feature_dim:
                raise ValueError(f"{task_id}: head feature dimension {head.weight.shape[1]} != encoder {feature_dim}")
        self.identity = {
            "adapter": ADAPTER_VERSION, "architecture": spec.model,
            "implementation_sha256": file_digest(Path(__file__)),
            "pretrained_family": spec.pretrained_tag,
            "preprocess": preprocessing_identity(preprocess),
            "open_clip": _package_version("open-clip-torch"),
            "torch": torch.__version__, "torchvision": _package_version("torchvision"),
            "pillow": _package_version("Pillow"),
            "compute_dtype": spec.runtime.compute_dtype,
        }
        self._device_heads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._bound_class_names: dict[str, tuple[Any, ...]] = {}

    def head_identity(self, task_id: str, class_names: Sequence[Any], declared_names: Sequence[Any] = ()) -> dict[str, Any]:
        head = self.heads[task_id]
        names = tuple(class_names)
        if head.weight.shape[0] != len(class_names):
            raise ValueError(f"{task_id}: head has {head.weight.shape[0]} classes, dataset has {len(class_names)}")
        if head.class_names and tuple(head.class_names) != tuple(class_names):
            raise ValueError(f"{task_id}: head class order differs from dataset")
        if declared_names and tuple(declared_names) != tuple(class_names):
            raise ValueError(f"{task_id}: declared class order differs from dataset")
        if task_id in self._bound_class_names and self._bound_class_names[task_id] != names:
            raise ValueError(f"{task_id}: class order differs between evaluation splits")
        self._bound_class_names[task_id] = names
        return {
            **head.identity,
            "dataset_class_names": tuple(class_names),
            "class_order_status": "verified_against_head_metadata" if head.class_names else (
                "user_declared" if declared_names else "unverified_author_head"
            ),
        }

    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        self.encoder.load_state_dict(weights, strict=True)
        self.encoder.eval()

    def logits(self, images: torch.Tensor, task_id: str) -> torch.Tensor:
        head = self.heads[task_id]
        if task_id not in self._device_heads:
            self._device_heads[task_id] = (
                head.weight.to(self.device, dtype=torch.float32),
                head.bias.to(self.device, dtype=torch.float32),
            )
        features = self.encoder(images.to(self.device))
        if head.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        weight, bias = self._device_heads[task_id]
        return torch.nn.functional.linear(features, weight, bias)


class ExistingEncoderAdapter(OpenClipAdapter):
    """Use an already trained encoder without constructing/downloading a model."""
    def __init__(self, encoder, task, runtime):
        self.encoder = encoder.to(runtime.device).eval()
        self.device = torch.device(runtime.device)
        self.preprocess = encoder.val_preprocess
        self.heads = {task.id: load_head(task.head)}
        self.identity = {"adapter": "existing-encoder-v2-stable-preprocess",
                         "preprocess": preprocessing_identity(self.preprocess),
                         "open_clip": _package_version("open-clip-torch"),
                         "torch": torch.__version__, "torchvision": _package_version("torchvision"),
                         "pillow": _package_version("Pillow"),
                         "implementation_sha256": file_digest(Path(__file__))}
        self._device_heads = {}
        self._bound_class_names = {}

    def logits(self, images, task_id):
        # Old ImageEncoder.forward is identical to the offline wrapper's forward.
        return super().logits(images, task_id)
