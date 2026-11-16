"""Create a minimal source ZIP or a complete trained project with prepared data."""
import argparse
import json
import zipfile
from pathlib import Path
from common.utils import ROOT, fingerprint, load_config, read_json

SOURCE_FILES = [
    "README.md", ".gitignore", "config.json", "report_metadata.json",
    "requirements.txt", "requirements-report.txt", "prepare_data.py",
    "evaluate.py", "predict.py", "compare.py", "build_report.py", "package_project.py",
]


def select_files(mode):
    files = {ROOT / name for name in SOURCE_FILES}
    for directory in ("common", "yolo", "faster_rcnn", "tests"):
        files.update(p for p in (ROOT / directory).rglob("*.py") if "__pycache__" not in p.parts)
    if mode == "full":
        for directory in ("docs", "reports", "runs/demo"):
            files.update(p for p in (ROOT / directory).rglob("*") if p.is_file())
        files.update(p for p in (ROOT / "runs/comparison").iterdir() if p.is_file())
        files.add(ROOT / "output/pdf/fogaszati_ai_beszamolo.pdf")
        for method in ("yolo", "faster_rcnn"):
            run = ROOT / "runs" / method
            files.update(p for p in run.iterdir() if p.is_file() and p.suffix in (".json", ".yaml", ".csv", ".png"))
        files.update([ROOT / "runs/yolo/weights/best.pt", ROOT / "runs/faster_rcnn/best.pth"])
        prepared = Path(load_config()["prepared_dir"])
        if not prepared.is_relative_to(ROOT):
            raise ValueError("Packaging requires prepared_dir inside the project directory.")
        files.update(p for p in prepared.rglob("*") if p.is_file() and p.suffix != ".cache")
        for path in [ROOT / "yolov8s.pt", ROOT / "weights/yolo26n.pt"]:
            if path.is_file():
                files.add(path)
    return files


def package(mode, output):
    files = select_files(mode)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            manifest[relative] = {"bytes": path.stat().st_size, "sha256": fingerprint(path)}
            archive.write(path, relative)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():
            raise RuntimeError("ZIP integrity check failed")
        names = set(archive.namelist())
        if mode == "full":
            prepared = Path(load_config()["prepared_dir"]).relative_to(ROOT)
            for split in ("train", "val", "test"):
                data = read_json(ROOT / prepared / f"{split}.json")
                for im in data["images"]:
                    assert (prepared / "images" / split / im["file_name"]).as_posix() in names
            assert "runs/yolo/metadata.json" in names
        assert len([n for n in names if n.lower().endswith("readme.md")]) == 1
    print(f"Ready: {output}; {len(files)} files; {output.stat().st_size/1024**2:.2f} MB; integrity verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["source", "full"], default="full")
    parser.add_argument("--output")
    args = parser.parse_args()
    default = f"output/fogaszati_ai_{'forras' if args.mode == 'source' else 'teljes'}.zip"
    package(args.mode, ROOT / (args.output or default))
