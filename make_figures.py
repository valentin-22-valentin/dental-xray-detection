"""Draw the README figures from saved training logs and test evaluations (no training needed)."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import ROOT, read_json

MODELS = [
    ("YOLOv8s", ROOT / "runs/yolo", "results.csv", "metrics/mAP50-95(B)", "#2a78d6"),
    ("Faster R-CNN", ROOT / "runs/faster_rcnn", "history.csv", "val_map50_95", "#eb6834"),
]
INK, MUTED, GRID = "#1d1d1b", "#6a6a64", "#e6e6e2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def per_class(output):
    reports = [(name, read_json(run / "test_evaluation.json"), color) for name, run, _, _, color in MODELS]
    classes = list(reports[0][1]["metrics"]["per_class_map50_95"])
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (name, report, color) in enumerate(reports):
        values = [report["metrics"]["per_class_map50_95"][c] for c in classes]
        xs = [x + (i - 0.5) * (width + 0.02) for x in range(len(classes))]
        bars = ax.bar(xs, values, width, color=color, label=name)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=9.5, color=INK)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylabel("mAP50–95 (test)")
    ax.set_ylim(0, 0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, colors=INK)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Detection quality per class", loc="left", fontsize=13, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(output / "class_ap.png")
    plt.close(fig)


def validation_curves(output):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, run, log, key, color in MODELS:
        rows = read_csv(run / log)
        epochs = [int(float(r["epoch"])) for r in rows]
        values = [float(r[key]) for r in rows]
        ax.plot(epochs, values, color=color, linewidth=2, label=name)
        best = max(range(len(values)), key=values.__getitem__)
        ax.plot(epochs[best], values[best], "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{name}: best {values[best]:.3f} @ epoch {epochs[best]}", (epochs[best], values[best]),
                    xytext=(8, 8), textcoords="offset points", fontsize=9.5, color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50–95 (validation)")
    ax.set_ylim(0, 0.36)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Validation mAP during training", loc="left", fontsize=13, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png")
    plt.close(fig)


def speed_accuracy(output):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for name, run, _, _, color in MODELS:
        report = read_json(run / "test_evaluation.json")
        latency, score = report["metadata"]["latency_ms_per_image"], report["metrics"]["map50_95"]
        ax.plot(latency, score, "o", color=color, markersize=11, markeredgecolor="white", markeredgewidth=2, label=name)
        ax.annotate(f"{name}\n{latency:.0f} ms · mAP {score:.3f}", (latency, score), xytext=(12, -4),
                    textcoords="offset points", fontsize=10, color=INK, va="center")
    ax.set_xlim(0, 260)
    ax.set_ylim(0.1, 0.36)
    ax.set_xlabel("Latency per image on RTX 3050 (ms, lower is better)")
    ax.set_ylabel("mAP50–95 (test)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Speed vs. accuracy", loc="left", fontsize=13, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(output / "speed_accuracy.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/figures")
    out = ROOT / parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    per_class(out)
    validation_curves(out)
    speed_accuracy(out)
    print(f"Figures written to {out}")
