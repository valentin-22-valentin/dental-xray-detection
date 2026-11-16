"""Annotate a new panoramic radiograph and export detections in original coordinates."""
import argparse
from pathlib import Path
from PIL import Image
from common.utils import load_config, write_json, draw_prediction
from common.inference import Detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["yolo","faster_rcnn"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="runs/prediction.png")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--threshold", type=float, default=0.25)
    a = parser.parse_args()
    if not 0 <= a.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    model = Detector(a.method, a.weights, load_config(a.config))
    with Image.open(a.image) as image:
        predictions = model.predict_original(image)
    visible = [p for p in predictions if p["score"] >= a.threshold]
    draw_prediction(a.image, visible, model.categories, a.output, a.threshold)
    write_json(Path(a.output).with_suffix(".json"), {"image": str(Path(a.image).resolve()), "method": a.method, "threshold": a.threshold, "categories": model.categories, "detections": visible})
    print(f"Elmentve: {a.output}; {len(visible)} jelölés.")
    for p in visible:
        print(f"{model.categories[p['category_id']]['name']}: {p['score']:.3f}, bbox={p['bbox']}")
    if not visible:
        print("Nincs a küszöb feletti találat; ebből nem következik, hogy a kép egészséges.")
