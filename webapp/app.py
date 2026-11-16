"""Web interface for running the trained detectors side by side on a radiograph.

Run from the project root:  python -m webapp.app   ->  http://127.0.0.1:5000
"""
import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image, UnidentifiedImageError

from common.utils import load_config, project_path, read_json

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_MODELS = [
    {"id": "yolov8s", "name": "YOLOv8s", "method": "yolo", "weights": "runs/yolo/weights/best.pt"},
    {"id": "faster_rcnn", "name": "Faster R-CNN", "method": "faster_rcnn", "weights": "runs/faster_rcnn/best.pth"},
]
MIN_SCORE = 0.05  # lower scores are dropped server side, the slider filters the rest
MAX_UPLOAD_MB = 25

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CFG = None
MODELS = {}
_detectors = {}
# Inference always runs on the same thread: the first CUDA call on a new thread is slow,
# and this also keeps requests from using the GPU at the same time.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")


def setup(config_path="config.json"):
    global CFG
    CFG = load_config(config_path)
    for spec in CFG.get("models", DEFAULT_MODELS):
        MODELS[spec["id"]] = {**spec, "path": project_path(spec["weights"])}


def _load(model_id):
    if model_id not in _detectors:
        from common.inference import Detector
        spec = MODELS[model_id]
        model = Detector(spec["method"], spec["path"], CFG)
        blank = Image.new("RGB", (model.size, model.size), (114, 114, 114))
        for _ in range(3):  # warm-up
            model.predict_square(blank)
        _detectors[model_id] = model
    return _detectors[model_id]


def _infer(model_id, image):
    import torch
    model = _load(model_id)
    cuda = str(model.device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    found = model.predict_original(image)
    if cuda:
        torch.cuda.synchronize(model.device)
    return {
        "model": model_id,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "device": str(model.device),
        "detections": [{**d, "score": round(d["score"], 4)} for d in found if d["score"] >= MIN_SCORE],
    }


def detector(model_id):
    return _worker.submit(_load, model_id).result()


def run_dir(spec):
    path = spec["path"].resolve()
    return path.parents[1] if spec["method"] == "yolo" else path.parent


@lru_cache(maxsize=1)
def test_split():
    """Test images with their ground-truth boxes (in 1024x1024 letterbox coordinates)."""
    path = Path(CFG["prepared_dir"]) / "test.json"
    if not path.is_file():
        return [], {}, []
    gt = read_json(path)
    boxes = {im["file_name"]: [] for im in gt["images"]}
    names = {im["id"]: im["file_name"] for im in gt["images"]}
    for a in gt["annotations"]:
        boxes[names[a["image_id"]]].append({"category_id": a["category_id"], "bbox": a["bbox"]})
    files = sorted(boxes, key=lambda n: int(Path(n).stem) if Path(n).stem.isdigit() else n)
    return files, boxes, gt["categories"]


def sample_path(name):
    if name not in test_split()[0]:
        abort(404)
    return Path(CFG["prepared_dir"]) / "images" / "test" / name


def categories():
    cats = test_split()[2]
    if cats:
        return cats
    for spec in MODELS.values():
        if spec["path"].is_file():
            return detector(spec["id"]).categories
    return []


def content_box(image):
    """Bounding box of the radiograph inside the grey (114) letterbox padding."""
    grey = image.convert("L")
    return grey.point(lambda v: 0 if abs(v - 114) < 2 else 255).getbbox() or (0, 0, *image.size)


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.get("/api/info")
def info():
    return jsonify({
        "models": [{"id": m["id"], "name": m["name"], "method": m["method"], "available": m["path"].is_file()} for m in MODELS.values()],
        "samples": test_split()[0],
        "categories": categories(),
        "min_score": MIN_SCORE,
        "default_threshold": CFG["evaluation"]["display_threshold"],
    })


@app.get("/api/results")
def results():
    """Test-set metrics saved by evaluate.py (runs/<model>/test_evaluation.json)."""
    rows = []
    for spec in MODELS.values():
        path = run_dir(spec) / "test_evaluation.json"
        if not path.is_file():
            continue
        report = read_json(path)
        meta, metrics = report["metadata"], report["metrics"]
        rows.append({
            "name": spec["name"],
            "images": meta["images"],
            "annotations": metrics["tp"] + metrics["fn"],
            "device": meta["device"],
            "threshold": meta["display_threshold"],
            "map50_95": metrics["map50_95"],
            "map50": metrics["map50"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "latency_ms": meta["latency_ms_per_image"],
            "training_min": meta["training_seconds"] / 60 if meta.get("training_seconds") else None,
            "params_m": meta.get("parameters_millions"),
            "per_class": metrics["per_class_map50_95"],
        })
    return jsonify(rows)


@app.get("/api/samples/<name>")
def sample(name):
    return send_file(sample_path(name))


@lru_cache(maxsize=256)
def thumbnail_bytes(name):
    with Image.open(sample_path(name)) as im:
        im = im.convert("L")
        im = im.crop(content_box(im))
        im.thumbnail((320, 160))
        buffer = io.BytesIO()
        im.save(buffer, "JPEG", quality=80)
    return buffer.getvalue()


@app.get("/api/thumbs/<name>")
def thumbnail(name):
    return send_file(io.BytesIO(thumbnail_bytes(name)), mimetype="image/jpeg", max_age=3600)


@app.post("/api/predict")
def predict():
    sample_name = request.form.get("sample")
    if sample_name:
        with Image.open(sample_path(sample_name)) as im:
            image = im.convert("RGB")
        ground_truth = test_split()[1][sample_name]
        x1, y1, x2, y2 = content_box(image)
        view = [x1, y1, x2 - x1, y2 - y1]
    elif "image" in request.files:
        try:
            with Image.open(request.files["image"].stream) as im:
                image = im.convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify({"error": "The uploaded file is not a readable image."}), 400
        ground_truth = None
        view = [0, 0, image.width, image.height]
    else:
        return jsonify({"error": "Pick a test image or upload a radiograph."}), 400

    wanted = [m for m in request.form.get("models", "").split(",") if m]
    unknown = [m for m in wanted if m not in MODELS or not MODELS[m]["path"].is_file()]
    if not wanted or unknown:
        return jsonify({"error": f"Model not available: {', '.join(unknown) or '-'}"}), 400

    output = [_worker.submit(_infer, model_id, image).result() for model_id in wanted]
    return jsonify({"width": image.width, "height": image.height, "view": view, "ground_truth": ground_truth, "results": output})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large (max {MAX_UPLOAD_MB} MB)."}), 413


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--preload", action="store_true", help="load all available models at startup")
    args = parser.parse_args()
    setup(args.config)
    for spec in MODELS.values():
        state = "ok" if spec["path"].is_file() else "missing"
        print(f"  {spec['name']:<14} {spec['weights']}  [{state}]")
        if args.preload and spec["path"].is_file():
            detector(spec["id"])
    print(f"Running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
