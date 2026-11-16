"""Convert fully annotated DENTEX data once for BOTH detectors. No downloads."""
import argparse
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from common.utils import load_config, read_json, write_json, square_image, fingerprint, write_dataset_yaml


def find_annotations(root, explicit):
    if explicit:
        path = Path(explicit)
        path = path if path.is_absolute() else root / path
        data = read_json(path)
        if "categories_3" not in data:
            raise ValueError("The given JSON is not a DENTEX diagnosis annotation file (categories_3 missing).")
        return path, data
    matches = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = read_json(path)
            if "categories_3" in data and data.get("images") and data.get("annotations"):
                matches.append((path, data))
        except (ValueError, OSError):
            continue
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one diagnosis JSON, found: {[str(p) for p, _ in matches]}. Set the annotations field in config.json.")
    return matches[0]


def prepare(cfg):
    root = Path(cfg["dataset_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {root}. Download and extract it, then set dataset_root in config.json.")
    output = Path(cfg["prepared_dir"])
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new prepared_dir to prepare the data again.")
    source, raw = find_annotations(root, cfg.get("annotations"))
    cats = sorted(raw["categories_3"], key=lambda c: c["id"])
    mapping = {c["id"]: i for i, c in enumerate(cats)}
    if len(mapping) != 4:
        raise ValueError("Expected four DENTEX diagnosis classes.")
    categories = [{"id": i, "name": c["name"]} for i, c in enumerate(cats)]
    size = int(cfg["image_size"])
    vf, tf = float(cfg["val_fraction"]), float(cfg["test_fraction"])
    if size < 64 or size % 32 or not (0 < vf < 1 and 0 < tf < 1 and vf + tf < 1):
        raise ValueError("image_size must be >= 64 and a multiple of 32; val/test fractions must be > 0 with a sum < 1.")
    images = sorted(raw["images"], key=lambda x: x["id"])
    by_id = {im["id"]: im for im in images}
    if len(images) != len(by_id) or len(images) < 3:
        raise ValueError("At least three images with unique ids are required.")
    files = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            files[p.name].append(p)
    resolved, hashes = {}, {}
    for im in images:
        name = im["file_name"]
        direct = [source.parent / name, root / name]
        candidates = list(dict.fromkeys(p.resolve() for p in direct if p.is_file()))
        if not candidates:
            candidates = files[Path(name).name]
        if len(candidates) != 1:
            raise ValueError(f"Missing or ambiguous image: {name}, candidates: {candidates}")
        path = candidates[0]
        with Image.open(path) as pic:
            if pic.size != (im["width"], im["height"]):
                raise ValueError(f"Image size differs from the annotation: {path}")
            digest = hashlib.sha256(pic.convert("RGB").tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate image: {name} and {hashes[digest]}; remove one before splitting.")
        hashes[digest] = name
        resolved[im["id"]] = path
    annotations = defaultdict(list)
    for a in raw["annotations"]:
        if a.get("image_id") not in by_id or a.get("category_id_3") not in mapping:
            raise ValueError(f"Invalid image/category reference: {a.get('id')}")
        if a.get("iscrowd", 0):
            raise ValueError("Crowd annotations are not supported in the shared YOLO format.")
        im = by_id[a["image_id"]]
        x, y, w, h = a["bbox"]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > im["width"]+1 or y+h > im["height"]+1:
            raise ValueError(f"Invalid bbox: {a.get('id')}, {a['bbox']}")
        annotations[a["image_id"]].append(a)
    # Fixed random image split. Patient identifiers are not supplied by the archive.
    random.Random(cfg["seed"]).shuffle(images)
    nv, nt = max(1, round(len(images)*vf)), max(1, round(len(images)*tf))
    if nv + nt >= len(images):
        raise ValueError("Too few images for the requested split fractions.")
    splits = {"val": images[:nv], "test": images[nv:nv+nt], "train": images[nv+nt:]}
    manifest = {"source_annotations": str(source), "source_sha256": fingerprint(source), "seed": cfg["seed"], "image_size": size, "categories": categories, "splits": {}, "counts": {}}
    for split, subset in splits.items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        coco = {"images": [], "annotations": [], "categories": categories}
        counts = Counter()
        for im in subset:
            ident = im["id"]
            name = f"{ident}.png"
            with Image.open(resolved[ident]) as pic:
                canvas, (sx, sy, left, top) = square_image(pic, size)
                canvas.save(image_dir / name)
            coco["images"].append({"id": ident, "file_name": name, "width": size, "height": size})
            labels = []
            for a in annotations[ident]:
                x, y, w, h = a["bbox"]
                x, y = x*sx+left, y*sy+top
                w, h = min(w*sx, size-x), min(h*sy, size-y)
                cls = mapping[a["category_id_3"]]
                counts[cls] += 1
                coco["annotations"].append({"id": len(coco["annotations"])+1, "image_id": ident, "category_id": cls, "bbox": [x,y,w,h], "area": w*h, "iscrowd": 0})
                labels.append(f"{cls} {(x+w/2)/size:.9f} {(y+h/2)/size:.9f} {w/size:.9f} {h/size:.9f}")
            (label_dir / f"{ident}.txt").write_text("\n".join(labels), encoding="utf-8")
        write_json(output / f"{split}.json", coco)
        manifest["splits"][split] = [im["id"] for im in subset]
        manifest["counts"][split] = {c["name"]: counts[c["id"]] for c in categories}
        missing = [c["name"] for c in categories if counts[c["id"]] == 0]
        if missing:
            print(f"WARNING: classes missing from the {split} split: {missing}. More data or another seed is needed.")
    write_json(output / "manifest.json", manifest)
    write_dataset_yaml(output, categories)
    print(f"Done: {output}; images: { {k: len(v) for k,v in splits.items()} }")
    print(manifest["counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    prepare(load_config(args.config))
