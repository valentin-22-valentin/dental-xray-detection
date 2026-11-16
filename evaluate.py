"""Shared held-out evaluation, including synchronized end-to-end latency."""
import argparse
import time
from pathlib import Path
from PIL import Image
import torch
from common.utils import load_config, read_json, fingerprint, ROOT
from common.inference import Detector
from common.evaluation import save_evaluation


def evaluate(method, weights, cfg, split="test", output=None, label=None):
    root = Path(cfg["prepared_dir"])
    model = Detector(method, weights, cfg)
    model.check_dataset(root)
    gt_path = root / f"{split}.json"
    gt = read_json(gt_path)
    def sync():
        if str(model.device).startswith("cuda"):
            torch.cuda.synchronize(model.device)
    # Warm-up excluded; disk I/O also excluded. Pre/postprocessing and transfers included.
    with Image.open(root / "images" / split / gt["images"][0]["file_name"]) as im:
        warmup = im.convert("RGB")
    for _ in range(3):
        model.predict_square(warmup)
    sync()
    predictions, elapsed = [], 0.0
    for im in gt["images"]:
        with Image.open(root / "images" / split / im["file_name"]) as pic:
            image = pic.convert("RGB")
        sync()
        start = time.perf_counter()
        found = model.predict_square(image)
        sync()
        elapsed += time.perf_counter()-start
        predictions.extend({**p, "image_id": im["id"]} for p in found)
    ev = cfg["evaluation"]
    metadata = {"method": method, "split": split, "manifest_sha256": fingerprint(root / "manifest.json"), "ground_truth_sha256": fingerprint(gt_path), "weights_sha256": fingerprint(weights), "device": model.device, "image_size": model.size, "score_floor": ev["score_floor"], "display_threshold": ev["display_threshold"], "max_detections": ev["max_detections"], "nms_iou": 0.5, "images": len(gt["images"]), "latency_ms_per_image": elapsed/len(gt["images"])*1000, "weights_mb": Path(weights).stat().st_size/1024**2, "torch": str(torch.__version__)}
    network = model.model.model if method == "yolo" else model.model
    metadata["parameters_millions"] = sum(p.numel() for p in network.parameters()) / 1e6
    run_dir = Path(weights).resolve().parents[1] if method == "yolo" else Path(weights).resolve().parent
    summary_path = run_dir / "training_summary.json"
    metadata["training_seconds"] = read_json(summary_path)["seconds"] if summary_path.exists() else None
    metadata["label"] = label or run_dir.name
    save_evaluation(gt_path, predictions, metadata, output or run_dir / f"{split}_evaluation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["yolo","faster_rcnn"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--split", choices=["val","test"], default="test")
    parser.add_argument("--output")
    parser.add_argument("--label", help="name shown in the comparison (default: name of the run folder)")
    a = parser.parse_args()
    evaluate(a.method, a.weights, load_config(a.config), a.split, a.output, a.label)
