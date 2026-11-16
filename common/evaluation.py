"""Identical COCO AP and IoU=0.5 precision/recall for both methods."""
import contextlib
import io
import math
from collections import defaultdict

from common.utils import read_json, write_json


def iou(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    overlap = max(0, min(x+w, xx+ww)-max(x,xx)) * max(0, min(y+h, yy+hh)-max(y,yy))
    union = w*h + ww*hh - overlap
    return overlap/union if union else 0.0


def operating_point(gt, predictions, threshold):
    truth = defaultdict(list)
    for a in gt["annotations"]:
        truth[(a["image_id"], a["category_id"])].append(a["bbox"])
    used, tp, fp = set(), 0, 0
    for p in sorted(predictions, key=lambda p: p["score"], reverse=True):
        if p["score"] < threshold:
            continue
        key = (p["image_id"], p["category_id"])
        choices = [(iou(p["bbox"], box), idx) for idx, box in enumerate(truth[key]) if (key, idx) not in used]
        overlap, idx = max(choices, default=(0, -1))
        if overlap >= 0.5:
            tp += 1
            used.add((key,idx))
        else:
            fp += 1
    fn = len(gt["annotations"])-tp
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0, "tp": tp, "fp": fp, "fn": fn}


def compute_metrics(gt_path, predictions, threshold=0.25, max_detections=100):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = read_json(gt_path)
    image_ids = {x["id"] for x in gt["images"]}
    category_ids = {x["id"] for x in gt["categories"]}
    for p in predictions:
        if p["image_id"] not in image_ids or p["category_id"] not in category_ids:
            raise ValueError("A predikció ismeretlen kép-/osztályazonosítót tartalmaz.")
        if not all(math.isfinite(x) for x in [*p["bbox"], p["score"]]) or len(p["bbox"]) != 4 or min(p["bbox"][2:]) <= 0 or not 0 <= p["score"] <= 1:
            raise ValueError("Érvénytelen predikció.")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(gt_path))
        if predictions:
            detected = coco.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {"images": gt["images"], "categories": gt["categories"], "annotations": []}
            detected.createIndex()
        evaluation = COCOeval(coco, detected, "bbox")
        evaluation.params.maxDets = [1, 10, max_detections]
        evaluation.evaluate()
        evaluation.accumulate()
    # T x R x K x A x M; area=all, largest detection budget.
    precision = evaluation.eval["precision"][:, :, :, 0, -1]
    def average(values):
        valid = values[values >= 0]
        return float(np.mean(valid)) if valid.size else None
    metrics = {"map50_95": average(precision), "map50": average(precision[0]), "map75": average(precision[5])}
    metrics["per_class_map50_95"] = {next(c["name"] for c in gt["categories"] if c["id"] == cat): average(precision[:,:,i]) for i, cat in enumerate(evaluation.params.catIds)}
    metrics.update(operating_point(gt, predictions, threshold))
    return metrics


def save_evaluation(gt_path, predictions, metadata, output):
    metrics = compute_metrics(gt_path, predictions, metadata["display_threshold"], metadata["max_detections"])
    write_json(output, {"metadata": metadata, "metrics": metrics, "predictions": predictions})
    print(metrics)
