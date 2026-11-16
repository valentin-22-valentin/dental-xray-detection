from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from common.utils import read_json


class DentexDataset(Dataset):
    def __init__(self, prepared, split):
        self.root = Path(prepared)
        raw = read_json(self.root / f"{split}.json")
        self.images = raw["images"]
        self.split = split
        self.annotations = {im["id"]: [] for im in self.images}
        for a in raw["annotations"]:
            self.annotations[a["image_id"]].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = self.images[index]
        with Image.open(self.root / "images" / self.split / im["file_name"]) as image:
            tensor = to_tensor(image.convert("RGB"))
        ann = self.annotations[im["id"]]
        boxes = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]] for a in ann], dtype=torch.float32).reshape(-1,4)
        target = {"boxes": boxes, "labels": torch.tensor([a["category_id"]+1 for a in ann], dtype=torch.int64), "image_id": torch.tensor(im["id"])}
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))
