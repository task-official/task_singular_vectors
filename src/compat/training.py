"""Bridge the unchanged training loop to the common offline inference evaluator."""
from pathlib import Path

from src.core.types import RuntimeSpec, TaskSpec


def eval_single_dataset(image_encoder, dataset_name, args):
    from src.datasets.adapter import inspect_task
    from src.evaluation.evaluator import Evaluator
    from src.models.adapter import ExistingEncoderAdapter
    split = "validation" if dataset_name.endswith("Val") else "test"
    task_id = dataset_name[:-3] if split == "validation" else dataset_name
    head = str(Path(args.save_dir) / f"head_{task_id}Val.pt")
    description = {"adapter": "legacy", "name": task_id, "split": split, "profile": "legacy_local"}
    task_spec = TaskSpec(task_id, description if split == "validation" else {},
                         description if split == "test" else {}, head)
    runtime = RuntimeSpec(device=str(args.device), batch_size=args.batch_size,
                          num_workers=getattr(args, "num_workers", 0))
    adapter = ExistingEncoderAdapter(image_encoder, task_spec, runtime)
    task = inspect_task(task_spec, split, args.data_location, adapter.preprocess, runtime)
    task.identity["head"] = adapter.head_identity(task_id, task.class_names)
    metrics = Evaluator(adapter, (task,), runtime).evaluate(image_encoder.state_dict())
    return {"top1": metrics.avg_top1}
