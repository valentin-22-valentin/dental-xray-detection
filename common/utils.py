import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path="config.json"):
    cfg = read_json(project_path(path))
    cfg["dataset_root"] = str(project_path(cfg["dataset_root"]).resolve())
    cfg["prepared_dir"] = str(project_path(cfg["prepared_dir"]).resolve())
    if int(cfg["epochs"]) < 1 or int(cfg["workers"]) < 0:
        raise ValueError("epochs >= 1 és workers >= 0 szükséges.")
    for method in ("yolo", "faster_rcnn"):
        if cfg[method]["batch_size"] < 1 or cfg[method]["lr"] <= 0:
            raise ValueError(f"Érvénytelen batch_size/lr: {method}")
    ev = cfg["evaluation"]
    if not 0 < ev["score_floor"] <= ev["display_threshold"] <= 1 or ev["max_detections"] < 10:
        raise ValueError("0 < score_floor <= display_threshold <= 1; max_detections >= 10 szükséges.")
    return cfg


def write_dataset_yaml(prepared, categories):
    """Refresh the local YOLO path after copying the project to another machine."""
    prepared = Path(prepared).resolve()
    yaml = f"path: {json.dumps(prepared.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {c['id']}: {json.dumps(c['name'])}\n" for c in categories)
    (prepared / "dataset.yaml").write_text(yaml, encoding="utf-8")


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def device_name(config):
    import torch
    value = config["device"]
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(value).isdigit():
        return f"cuda:{value}"
    return value


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def square_image(image, size):
    from PIL import Image
    image = image.convert("RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    return canvas, (nw / width, nh / height, left, top)


def draw_prediction(image_path, detections, categories, output, threshold):
    from PIL import Image, ImageDraw
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ["#e63946", "#ff9f1c", "#2a9d8f", "#4361ee"]
    for det in detections:
        if det["score"] < threshold:
            continue
        x, y, w, h = det["bbox"]
        color = colors[det["category_id"] % len(colors)]
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        text = f'{categories[det["category_id"]]["name"]}: {det["score"]:.2f}'
        draw.text((x, max(0, y - 12)), text, fill=color)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
