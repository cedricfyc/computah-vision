"""Train or evaluate waste detector.

it is reconstructed content of my Kaggle notebook .

dependecies:
ultralytics==8.4.208
albumentations==2.0.8

pip install -r requirements.txt
python waste_detection.py train --dataset "/path/to/combined-dataset" --device "0,1"
python waste_detection.py evaluate --dataset "/path/to/combined-dataset" --weights "5th_30 epochs_2gpu_chosen/best.pt" --split val --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from uuid import uuid4


CLASSES = ("biodegradable", "cardboard", "glass", "metal", "paper", "plastic")


def dataset_config(dataset: Path, output: Path) -> Path:
    """Check the prepared YOLO dataset and create a portable configuration."""
    dataset = dataset.expanduser().resolve()
    for split in ("train", "valid", "test"):
        for part in ("images", "labels"):
            folder = dataset / split / part
            if not folder.is_dir():
                raise FileNotFoundError(f"Required dataset folder is missing: {folder}")

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "dataset_config.yaml"
    config = {
        "path": str(dataset),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(CLASSES)),
    }
    # JSON is valid YAML, so no additional YAML writer is needed.
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def train(args: argparse.Namespace, config_path: Path, output: Path) -> None:
    """Start a new run using the recorded fifth-experiment settings."""
    import albumentations as A
    from ultralytics import YOLO

    model = YOLO("yolo26m.pt")  # Pretrained weights
    model.train(
        data=str(config_path),
        epochs=30,
        patience=10,
        imgsz=512,
        batch=8,
        device=args.device,
        seed=0,
        optimizer="auto",
        project=str(output),
        name="train",
        save_period=5,
        # The source dataset already has training-only augmented copies.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        augmentations=[A.NoOp(p=0.0)],
    )


def evaluate(args: argparse.Namespace, config_path: Path, output: Path) -> None:
    """Evaluate a selected checkpoint and save aggregate and class metrics."""
    from ultralytics import YOLO

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    model = YOLO(str(weights))
    if len(model.names) != len(CLASSES):
        raise ValueError(f"Expected {len(CLASSES)} model classes, found {len(model.names)}")
    actual_names = [model.names[i] for i in range(len(CLASSES))]
    if actual_names != list(CLASSES):
        raise ValueError(f"Checkpoint class mapping does not match: {actual_names}")

    run_name = f"evaluation_{args.split}_{uuid4().hex[:8]}"
    results = model.val(
        data=str(config_path),
        split=args.split,
        imgsz=512,
        device=args.device,
        max_det=358,
        plots=True,
        project=str(output),
        name=run_name,
    )

    # Ultralytics reports the actual directory used for its plots.
    result_dir = Path(results.save_dir)
    summary = {key: float(value) for key, value in results.results_dict.items()}
    (result_dir / "overall_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    with (result_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("class", "precision", "recall", "ap50", "ap50_95"))
        for result_index, class_id in enumerate(results.box.ap_class_index):
            precision, recall, ap50, ap = results.box.class_result(result_index)
            writer.writerow((model.names[int(class_id)], precision, recall, ap50, ap))

    print(f"Metrics and plots saved to: {result_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--dataset", required=True, type=Path, help="Folder containing train, valid, and test")
    parser.add_argument("--output", default=Path("runs/waste_detection"), type=Path, help="Writable output folder")
    parser.add_argument("--device", default="0", help="GPU index, e.g. 0 or 0,1; use cpu for CPU")
    parser.add_argument("--weights", type=Path, help="Selected .pt checkpoint (required for evaluation)")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Evaluation split; default: val")
    args = parser.parse_args()

    if args.mode == "evaluate" and args.weights is None:
        parser.error("--weights is required for evaluation")
    if args.mode == "train" and args.weights is not None:
        parser.error("--weights is only used for evaluation; train starts from yolo26m.pt")

    output = args.output.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    if output == dataset or dataset in output.parents:
        parser.error("Choose an output folder outside the input dataset")
    config_path = dataset_config(dataset, output)

    if args.mode == "train":
        train(args, config_path, output)
    else:
        evaluate(args, config_path, output)


if __name__ == "__main__":
    main()
