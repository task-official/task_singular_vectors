import json
import os
from pprint import pprint

from src.eval.eval import eval_single_dataset
from src.models.task_vectors import NonLinearTaskVector
from src.utils.args import parse_arguments
from src.utils.variables_and_paths import (
    DATASETS_8,
    get_finetuned_path,
    get_zeroshot_path,
    get_single_task_accuracies_path,
)

args = parse_arguments()
args.save_dir = os.path.join(args.model_location, args.model)
os.makedirs(args.save_dir, exist_ok=True)

if args.finetuning_mode != "standard":
    raise ValueError(
        "This script expects --finetuning-mode=standard"
    )

pretrained_checkpoint = get_zeroshot_path(
    args.model_location,
    "MNIST",
    args.model,
)

accuracies = {}

for dataset in DATASETS_8:
    print("\n" * 2)
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    finetuned_checkpoint = get_finetuned_path(
        args.model_location,
        dataset,
        args.model,
    )

    task_vector = NonLinearTaskVector(
        args.model,
        pretrained_checkpoint,
        finetuned_checkpoint,
    )

    image_encoder = task_vector.apply_to(
        pretrained_checkpoint,
        scaling_coef=1.0,
        args=args,
    )

    for split in ["val", "test"]:
        eval_dataset = (
            dataset if split == "test"
            else f"{dataset}Val"
        )

        print("=" * 100)
        print(f"Evaluating on {eval_dataset}")

        result = eval_single_dataset(
            image_encoder,
            eval_dataset,
            args,
        )

        accuracies[eval_dataset] = float(result["top1"])

save_path = get_single_task_accuracies_path(args.model)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "w") as handle:
    json.dump(accuracies, handle, indent=4)

pprint(accuracies)
print("File saved at:", save_path)
