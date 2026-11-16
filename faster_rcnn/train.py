"""Run from project root: python -m faster_rcnn.train"""
import argparse
import csv
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from common.utils import load_config, read_json, seed_everything, device_name, fingerprint, write_json, ROOT
from common.evaluation import compute_metrics
from faster_rcnn.dataset import DentexDataset, collate
from faster_rcnn.model import build_model


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    predictions = []
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for target, out in zip(targets, outputs):
            for box, label, score in zip(out["boxes"].cpu().tolist(), out["labels"].cpu().tolist(), out["scores"].cpu().tolist()):
                x1,y1,x2,y2 = box
                if x2 > x1 and y2 > y1:
                    predictions.append({"image_id": int(target["image_id"]), "category_id": label-1, "bbox": [x1,y1,x2-x1,y2-y1], "score": score})
    return predictions


def train(cfg):
    seed_everything(cfg["seed"])
    device = torch.device(device_name(cfg))
    root = Path(cfg["prepared_dir"])
    manifest = read_json(root / "manifest.json")
    if manifest["image_size"] != cfg["image_size"]:
        raise ValueError("Az image_size eltér az előkészített adatokétól.")
    run = ROOT / "runs" / "faster_rcnn"
    if run.exists():
        raise FileExistsError(f"A futás már létezik: {run}. Archiváld/ nevezd át új tanítás előtt.")
    run.mkdir(parents=True)
    ev, fc = cfg["evaluation"], cfg["faster_rcnn"]
    model = build_model(len(manifest["categories"]), cfg["image_size"], fc["pretrained"], ev["score_floor"], ev["max_detections"]).to(device)
    train_loader = DataLoader(DentexDataset(root,"train"), batch_size=fc["batch_size"], shuffle=True, num_workers=cfg["workers"], collate_fn=collate)
    val_loader = DataLoader(DentexDataset(root,"val"), batch_size=fc["batch_size"], num_workers=cfg["workers"], collate_fn=collate)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=fc["lr"], momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg["epochs"]//3), gamma=0.1)
    best, start = -1.0, time.perf_counter()
    write_json(run / "config.json", cfg)
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_map50_95", "val_map50", "lr", "elapsed_seconds"])
        writer.writeheader()
        for epoch in range(1, cfg["epochs"]+1):
            model.train()
            total = 0.0
            for images, targets in train_loader:
                images = [im.to(device) for im in images]
                targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
                loss = sum(model(images, targets).values())
                if not math.isfinite(loss.item()):
                    raise RuntimeError("Nem véges loss; ellenőrizd az adatokat és csökkentsd az lr értékét.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                total += loss.item()
            metrics = compute_metrics(root / "val.json", predict_loader(model,val_loader,device), ev["display_threshold"], ev["max_detections"])
            score = metrics["map50_95"]
            if score is None:
                raise ValueError("Nincs értékelhető validációs annotáció.")
            writer.writerow({"epoch": epoch, "train_loss": total/len(train_loader), "val_map50_95": score, "val_map50": metrics["map50"], "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter()-start})
            f.flush()
            scheduler.step()
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "categories": manifest["categories"], "image_size": cfg["image_size"], "manifest_sha256": fingerprint(root / "manifest.json"), "config": cfg, "val_map50_95": score}
            torch.save(checkpoint, run / "last.pth")
            if score > best:
                best = score
                torch.save(checkpoint, run / "best.pth")
            print(f"Epoch {epoch}/{cfg['epochs']}: loss={total/len(train_loader):.4f}, val mAP={score:.4f}", flush=True)
    write_json(run / "training_summary.json", {"seconds": time.perf_counter()-start, "best_val_map50_95": best, "device": str(device), "torch": torch.__version__})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    train(load_config(parser.parse_args().config))
