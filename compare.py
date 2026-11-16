"""Generate CSV + Markdown report for two or more models, rejecting incompatible evaluations."""
import argparse
import csv
from pathlib import Path
from common.utils import read_json

DEFAULT_REPORTS = ["runs/yolo/test_evaluation.json", "runs/faster_rcnn/test_evaluation.json"]
SHARED_KEYS = ["split", "manifest_sha256", "ground_truth_sha256", "device", "image_size", "score_floor", "display_threshold", "max_detections", "nms_iou", "images"]


def label_of(report):
    return report["metadata"].get("label") or report["metadata"]["method"]


def compare(paths, output):
    reports = [read_json(p) for p in paths]
    if len(reports) < 2:
        raise ValueError("Legalább két kiértékelési jelentés szükséges.")
    labels = [label_of(r) for r in reports]
    if len(set(labels)) != len(labels):
        raise ValueError(f"A modellnevek nem egyediek: {labels}. Használd az evaluate.py --label kapcsolóját.")
    for key in SHARED_KEYS:
        if len({str(r["metadata"][key]) for r in reports}) != 1:
            raise ValueError(f"Nem összehasonlítható futások: eltérő {key}.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["model", "method", "map50_95", "map50", "map75", "precision", "recall", "f1", "latency_ms_per_image", "training_seconds", "parameters_millions", "weights_mb"]
    rows = [{"model": label, "method": r["metadata"]["method"], **{k: r["metrics"].get(k, r["metadata"].get(k)) for k in fields[2:]}} for label, r in zip(labels, reports)]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v):
        return "N/A" if v is None else f"{v:.4f}" if isinstance(v, (float, int)) else str(v)

    meta = reports[0]["metadata"]
    lines = [f"# {' vs. '.join(labels)}", "", f"Halmaz: {meta['split']}; képek: {meta['images']}; eszköz: {meta['device']}", "",
             "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    lines += ["| " + " | ".join(fmt(row[k]) for k in fields) + " |" for row in rows]
    lines += ["", "## Osztályonkénti mAP@0.50:0.95", "", "| Osztály | " + " | ".join(labels) + " |", "| --- |" + " --- |" * len(labels)]
    for name in reports[0]["metrics"]["per_class_map50_95"]:
        lines.append(f"| {name} | " + " | ".join(fmt(r["metrics"]["per_class_map50_95"].get(name)) for r in reports) + " |")
    lines += ["", "Az AP értékek 0–1 skálán vannak; a nagyobb jobb. A késleltetésnél a kisebb jobb.", "Precision/recall/F1: osztályhelyes párosítás, IoU ≥ 0.5, a rögzített display_threshold mellett.", "A késleltetés batch=1, három bemelegítő futás után, lemezolvasás nélkül, elő-/utófeldolgozással és GPU-szinkronizálással mérve.", "A weights_mb a mentett fájl mérete; a Faster R-CNN fájl az optimalizáló állapotát is tartalmazza, ezért ez nem tiszta modellméret-összehasonlítás.", "Eltérő optimalizálók/hálózatok esetén azonos epoch nem jelent azonos számítási költséget. A következtetést a pontosság, tanítási idő és késleltetés együtt adja.", "A teszthalmazt ne használd hiperparaméterek vagy küszöbértékek kiválasztására.", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Jelentés: {output / 'comparison.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", default=DEFAULT_REPORTS, help="Kiértékelési JSON-ok (evaluate.py kimenetei)")
    parser.add_argument("--output", default="runs/comparison")
    a = parser.parse_args()
    compare(a.reports, a.output)
