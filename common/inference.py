from pathlib import Path
import torch
from torchvision.transforms.functional import to_tensor
from common.utils import read_json, fingerprint, square_image


class Detector:
    def __init__(self, method, weights, cfg):
        from common.utils import device_name
        self.method, self.cfg = method, cfg
        self.device = device_name(cfg)
        self.ev = cfg["evaluation"]
        if method == "yolo":
            from ultralytics import YOLO
            self.metadata = read_json(Path(weights).resolve().parents[1] / "metadata.json")
            self.model = YOLO(str(weights))
            expected = [c["name"] for c in self.metadata["categories"]]
            if [self.model.names[i] for i in range(len(expected))] != expected:
                raise ValueError("The classes in the YOLO weights do not match the run metadata.")
        else:
            from faster_rcnn.model import build_model
            # Load only your own locally produced checkpoints.
            self.metadata = torch.load(weights, map_location="cpu", weights_only=True)
            self.model = build_model(len(self.metadata["categories"]), self.metadata["image_size"], False, self.ev["score_floor"], self.ev["max_detections"])
            self.model.load_state_dict(self.metadata["model"])
            self.model.to(self.device).eval()
        self.categories = self.metadata["categories"]
        self.size = self.metadata["image_size"]

    def check_dataset(self, prepared):
        if fingerprint(Path(prepared) / "manifest.json") != self.metadata["manifest_sha256"]:
            raise ValueError("The model was trained on a different data split; the test comparison would not be valid.")

    @torch.inference_mode()
    def predict_square(self, image):
        if self.method == "yolo":
            result = self.model.predict(image, imgsz=self.size, device=self.device, conf=self.ev["score_floor"], iou=0.5, max_det=self.ev["max_detections"], agnostic_nms=False, augment=False, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().tolist()
            labels = result.boxes.cls.int().cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
        else:
            result = self.model([to_tensor(image).to(self.device)])[0]
            boxes = result["boxes"].cpu().tolist()
            labels = (result["labels"]-1).cpu().tolist()
            scores = result["scores"].cpu().tolist()
        predictions = []
        for (x1,y1,x2,y2), label, score in zip(boxes, labels, scores):
            if x2 > x1 and y2 > y1:
                predictions.append({"category_id": label, "bbox": [x1,y1,x2-x1,y2-y1], "score": score})
        return predictions

    def predict_original(self, image):
        image = image.convert("RGB")
        square, (sx,sy,left,top) = square_image(image, self.size)
        predictions = []
        for det in self.predict_square(square):
            x,y,w,h = det["bbox"]
            x1,y1 = max(0,(x-left)/sx), max(0,(y-top)/sy)
            x2,y2 = min(image.width,(x+w-left)/sx), min(image.height,(y+h-top)/sy)
            if x2 > x1 and y2 > y1:
                predictions.append({**det, "bbox": [x1,y1,x2-x1,y2-y1]})
        return predictions
