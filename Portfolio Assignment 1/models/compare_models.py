"""Reproduce the fifth-versus-seventh model validation comparison.

Both checkpoints are evaluated on the same raw validation images and labels.
This script reconstructs the comparison setup. 

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, is_zipfile

CLASSES = (
    "biodegradable", "cardboard", "glass", "metal", "paper", "plastic"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_config(dataset: Path, output: Path, expected_count: int) -> tuple[Path, int]:
    """Validate the extracted dataset and write a separate YOLO configuration."""
    import yaml

    dataset = dataset.expanduser().resolve()
    source_yaml = dataset / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Dataset data.yaml not found: {source_yaml}")
    with source_yaml.open(encoding="utf-8") as file:
        metadata = yaml.safe_load(file)
    names = metadata.get("names") if isinstance(metadata, dict) else None
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(CLASSES))]
    if names != list(CLASSES):
        raise ValueError(f"Unexpected class order in {source_yaml}: {names}")

    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            directory = dataset / split / kind
            if not directory.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {directory}")

    images = sorted(
        p for p in (dataset / "valid" / "images").iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) != expected_count:
        raise ValueError(
            f"Expected {expected_count} validation images; found {len(images)}. "
            "Check that the extracted raw_same_split dataset was selected."
        )
    missing = [p.name for p in images if not (dataset / "valid" / "labels" / f"{p.stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation labels, e.g. {missing[:3]}")

    config = {
        "path": str(dataset),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": list(CLASSES),
    }
    output.mkdir(parents=True)
    path = output / "comparison_data.yaml"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, len(images)


def checkpoint_path(path: Path, temp_dir: Path, label: str) -> Path:
    """Use a .pt file, or copy a PyTorch checkpoint saved with a .zip suffix."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix.lower() == ".pt":
        return path
    if path.suffix.lower() != ".zip" or not is_zipfile(path):
        raise ValueError(f"Expected a .pt checkpoint or PyTorch checkpoint saved as .zip: {path}")
    with ZipFile(path) as archive:
        if not any(name.endswith("/data.pkl") for name in archive.namelist()):
            raise ValueError(
                f"{path} is not a direct PyTorch checkpoint. Extract the archive "
                "and pass the actual .pt file instead."
            )
    copy = temp_dir / f"{label}.pt"
    shutil.copy2(path, copy)
    return copy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Extracted raw_same_split dataset folder")
    parser.add_argument("--fifth", type=Path, required=True, help="Fifth model checkpoint (.pt or PyTorch .zip)")
    parser.add_argument("--seventh", type=Path, required=True, help="Seventh model checkpoint (.pt or PyTorch .zip)")
    parser.add_argument("--output", type=Path, default=Path("runs/comparison_5_7"), help="New output directory")
    parser.add_argument("--device", default="0", help="GPU index or cpu; default: 0")
    parser.add_argument("--expected-images", type=int, default=2093, help="Expected validation image count")
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == dataset or dataset in output.parents:
        parser.error("Choose an output directory outside the dataset")
    if output.exists():
        parser.error(f"Output already exists; choose a new path: {output}")
    for label, path in (("fifth", args.fifth), ("seventh", args.seventh)):
        if not path.expanduser().is_file():
            parser.error(f"{label} checkpoint does not exist: {path}")

    import ultralytics
    from ultralytics import YOLO

    config_path, count = dataset_config(dataset, output, args.expected_images)
    settings = {"imgsz": 512, "batch": 8, "device": args.device,
                "conf": 0.001, "iou": 0.7, "max_det": 300}
    sources = {"fifth_model": args.fifth, "seventh_model": args.seventh}
    provenance = {
        "ultralytics_version": ultralytics.__version__,
        "dataset": str(dataset),
        "validation_images": count,
        "validation_settings": settings,
        "checkpoints": {
            name: {"path": str(p.expanduser().resolve()), "sha256": sha256(p.expanduser())}
            for name, p in sources.items()
        },
    }
    (output / "comparison_settings.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    summary = {}
    class_rows = []
    temp_dir = output / "_temporary_weights"
    temp_dir.mkdir()
    try:
        for name, source in sources.items():
            print(f"\nEvaluating {name} on {count} validation images...", flush=True)
            weights = checkpoint_path(source, temp_dir, name)
            model = YOLO(str(weights))
            actual_classes = [model.names[i] for i in range(len(model.names))]
            if actual_classes != list(CLASSES):
                raise ValueError(f"{name} has an unexpected class order: {actual_classes}")
            metrics = model.val(
                data=str(config_path), split="val", plots=True,
                project=str(output), name=name, **settings,
            )
            summary[name] = {key: float(value) for key, value in metrics.results_dict.items()}
            for index, class_id in enumerate(metrics.box.ap_class_index):
                precision, recall, ap50, ap = metrics.box.class_result(index)
                class_rows.append({
                    "model": name, "class": model.names[int(class_id)],
                    "precision": float(precision), "recall": float(recall),
                    "ap50": float(ap50), "ap50_95": float(ap),
                })
    finally:
        # Only remove temporary files created by this script in the new output folder.
        for name in sources:
            (temp_dir / f"{name}.pt").unlink(missing_ok=True)
        temp_dir.rmdir()

    (output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "per_class_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "class", "precision", "recall", "ap50", "ap50_95"])
        writer.writeheader()
        writer.writerows(class_rows)
    print("\nOverall validation results:")
    for name, values in summary.items():
        print(f"{name}: precision={values['metrics/precision(B)']:.3f}, "
              f"recall={values['metrics/recall(B)']:.3f}, "
              f"AP50={values['metrics/mAP50(B)']:.3f}, "
              f"AP50-95={values['metrics/mAP50-95(B)']:.3f}")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
