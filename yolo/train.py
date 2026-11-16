"""Run from project root: python -m yolo.train"""
import argparse
import time
from pathlib import Path
from common.utils import ROOT, load_config, read_json, device_name, fingerprint, write_json, seed_everything, write_dataset_yaml


def train(cfg, model_name=None, run_name="yolo"):
    from ultralytics import YOLO
    seed_everything(cfg["seed"])
    root = Path(cfg["prepared_dir"])
    manifest = read_json(root / "manifest.json")
    if manifest["image_size"] != cfg["image_size"]:
        raise ValueError("Az image_size eltér az előkészített adatokétól.")
    run = ROOT / "runs" / run_name
    if run.exists():
        raise FileExistsError(f"A futás már létezik: {run}. Archiváld/nevezd át új tanítás előtt.")
    write_dataset_yaml(root, manifest["categories"])
    start = time.perf_counter()
    model_name = model_name or cfg["yolo"]["model"]
    model = YOLO(model_name)
    model.train(data=str(root / "dataset.yaml"), epochs=cfg["epochs"], imgsz=cfg["image_size"], batch=cfg["yolo"]["batch_size"], device=device_name(cfg), workers=cfg["workers"], seed=cfg["seed"], deterministic=True, project=str(ROOT / "runs"), name=run_name, exist_ok=False, optimizer="SGD", lr0=cfg["yolo"]["lr"], patience=0,
                # Same fixed square inputs; disable YOLO-only augmentations for the baseline.
                mosaic=0.0, mixup=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, multi_scale=False, plots=True)
    actual_run = Path(model.trainer.save_dir)
    write_json(actual_run / "metadata.json", {"manifest_sha256": fingerprint(root / "manifest.json"), "image_size": cfg["image_size"], "categories": manifest["categories"], "config": cfg})
    write_json(actual_run / "training_summary.json", {"seconds": time.perf_counter()-start, "device": device_name(cfg), "model": model_name})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", help="Előtanított súlyok, pl. yolov8s.pt vagy yolo26s.pt (alapértelmezés: config.json)")
    parser.add_argument("--name", default="yolo", help="Futás neve a runs/ mappában, pl. yolo26s")
    args = parser.parse_args()
    train(load_config(args.config), args.model, args.name)
