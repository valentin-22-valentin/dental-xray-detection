from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes, image_size, pretrained=False, score_floor=0.001, max_detections=100):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None, min_size=image_size, max_size=image_size,
        box_score_thresh=score_floor, box_nms_thresh=0.5,
        box_detections_per_img=max_detections,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes + 1)
    return model
