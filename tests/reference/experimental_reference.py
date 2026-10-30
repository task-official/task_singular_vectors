"""Load the immutable pre-migration research snapshot for regression tests only."""
import importlib.util
import hashlib
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "archive/legacy_code/src/methods/experimental/legacy_variants.py"
if hashlib.sha256(path.read_bytes()).hexdigest() != "b410e564694a9b2a99eacb2749dd3c95875acb44b460775ae2aad98c870d2409":
    raise RuntimeError("Frozen research reference was modified")
spec = importlib.util.spec_from_file_location("frozen_research_snapshot", path)
legacy_variants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy_variants)
