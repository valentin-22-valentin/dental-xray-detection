"""Small synthetic dataset checks; no DENTEX data or pretrained weights needed."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from PIL import Image
from common.utils import ROOT, load_config, read_json, write_json, square_image, write_dataset_yaml
from common.evaluation import compute_metrics
from prepare_data import prepare
from compare import compare


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        raw = {"images": [], "annotations": [], "categories_3": [{"id": i, "name": name} for i,name in enumerate(["Impacted","Caries","Periapical lesion","Deep caries"])]}
        for i in range(20):
            Image.new("RGB",(120,60),(i*10,30,50)).save(source / f"{i}.png")
            raw["images"].append({"id": i, "file_name": f"{i}.png", "width": 120,"height": 60})
            for cls in range(4):
                raw["annotations"].append({"id": i*4+cls, "image_id": i, "category_id_3": cls, "bbox": [cls*20+1,5,15,25]})
        write_json(source / "train.json", raw)
        self.cfg = {"dataset_root": str(source), "annotations": "", "prepared_dir": str(self.root / "prepared"), "image_size": 64, "seed": 42, "val_fraction": 0.15,"test_fraction": 0.15}

    def tearDown(self):
        self.temp.cleanup()

    def test_preparation_and_metrics(self):
        prepare(self.cfg)
        root = Path(self.cfg["prepared_dir"])
        manifest = read_json(root / "manifest.json")
        splits = [set(x) for x in manifest["splits"].values()]
        self.assertEqual(len(set.union(*splits)),20)
        self.assertTrue(all(not a & b for i,a in enumerate(splits) for b in splits[i+1:]))
        cfg2 = {**self.cfg, "prepared_dir": str(self.root / "second")}
        prepare(cfg2)
        self.assertEqual(manifest, read_json(Path(cfg2["prepared_dir"]) / "manifest.json"))
        gt = read_json(root / "test.json")
        predictions = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.99} for a in gt["annotations"]]
        metrics = compute_metrics(root / "test.json",predictions)
        self.assertAlmostEqual(metrics["map50_95"],1.0)
        self.assertEqual(metrics["f1"],1.0)
        empty = compute_metrics(root / "test.json",[])
        self.assertEqual(empty["map50_95"],0.0)
        self.assertEqual(empty["recall"],0.0)
        duplicated = compute_metrics(root / "test.json",predictions+predictions)
        self.assertEqual(duplicated["precision"],0.5)
        for im in gt["images"]:
            labels = (root / "labels" / "test" / f"{im['id']}.txt").read_text().splitlines()
            for line, ann in zip(labels,[a for a in gt["annotations"] if a["image_id"] == im["id"]]):
                cls,xc,yc,w,h = map(float,line.split())
                x,y,bw,bh = ann["bbox"]
                self.assertEqual(int(cls),ann["category_id"])
                self.assertAlmostEqual(xc*64,x+bw/2,places=5)
                self.assertAlmostEqual(yc*64,y+bh/2,places=5)
                self.assertAlmostEqual(w*64,bw,places=5)
                self.assertAlmostEqual(h*64,bh,places=5)
        metadata = {"method":"yolo", "split":"test", "manifest_sha256":"x", "ground_truth_sha256":"g", "device":"cpu", "image_size":64, "score_floor":0.001, "display_threshold":0.25,"max_detections":100,"nms_iou":0.5,"images":3,"latency_ms_per_image":1,"weights_mb":10}
        report = {"metadata":metadata,"metrics":metrics}
        write_json(self.root / "y.json",report)
        second = copy.deepcopy(report)
        second["metadata"]["method"] = "faster_rcnn"
        write_json(self.root / "f.json",second)
        compare([self.root / "y.json",self.root / "f.json"],self.root / "report")
        self.assertTrue((self.root / "report" / "comparison.csv").is_file())
        second["metadata"]["manifest_sha256"] = "different"
        write_json(self.root / "f.json",second)
        with self.assertRaises(ValueError):
            compare([self.root / "y.json",self.root / "f.json"],self.root / "report")

    def test_duplicate_rejected(self):
        source = Path(self.cfg["dataset_root"])
        (source / "1.png").write_bytes((source / "0.png").read_bytes())
        with self.assertRaisesRegex(ValueError,"Duplikált"):
            prepare(self.cfg)

    def test_invalid_box_rejected(self):
        path = Path(self.cfg["dataset_root"]) / "train.json"
        data = read_json(path)
        data["annotations"][0]["bbox"] = [1,1,-5,20]
        write_json(path,data)
        with self.assertRaisesRegex(ValueError,"bbox"):
            prepare(self.cfg)

    def test_letterbox(self):
        canvas,(sx,sy,left,top) = square_image(Image.new("RGB",(120,60)),64)
        self.assertEqual(canvas.size,(64,64))
        self.assertEqual((left,top),(0,16))
        self.assertAlmostEqual(sx,sy)

    def test_portable_paths(self):
        cfg = read_json(ROOT / "config.json")
        cfg.update({"dataset_root": "source", "prepared_dir": "copied data"})
        write_json(self.root / "portable.json", cfg)
        with patch("common.utils.ROOT", self.root):
            loaded = load_config("portable.json")
        self.assertEqual(Path(loaded["dataset_root"]), self.root / "source")
        copied = self.root / "copied data"
        copied.mkdir()
        write_dataset_yaml(copied, [{"id": 0, "name": "Caries"}])
        yaml = (copied / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn(copied.as_posix(), yaml)
        self.assertNotIn(ROOT.as_posix(), yaml)
        self.assertIn('0: "Caries"', yaml)


if __name__ == "__main__":
    unittest.main()
