#!/usr/bin/env python3

# Third-party imports
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

# Constants
COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
    "trafficlight", "firehydrant", "streetsign", "stopsign", "parkingmeter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
    "hat", "backpack", "umbrella", "shoe", "eyeglasses", "handbag", "tie", "suitcase", 
    "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", 
    "skateboard", "surfboard", "tennisracket", "bottle", "plate", "wineglass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
    "carrot", "hotdog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", 
    "mirror", "diningtable", "window", "desk", "toilet", "door", "tvmonitor", "laptop", 
    "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", 
    "toothbrush", "hairbrush"
]

ASSOCIATION_LABELS = {
    1: 5,
    2: 0,
    3: 2,
    4: 4,
    6: 1,
    8: 7,
}


class MaskRCNNModel:
    def __init__(self) -> None:
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        
    def predict(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = transforms.Compose([ transforms.ToTensor() ])
        with torch.no_grad():
            output = self.model(transform(image).unsqueeze(0))

        boxes, labels, scores = self.applyNMS(output[0]['boxes'], output[0]['labels'], output[0]['scores'], IoUThreshold=0.5)
        boxes, labels, scores = self.applyConfThreshold(boxes, labels, scores, threshold=0.75)
        boxes, labels, scores = self.applyLabelFilter(boxes, labels, scores)
        labels = [ASSOCIATION_LABELS[int(label)] for label in labels]

        return boxes, labels, scores
    
    def applyNMS(self, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, IoUThreshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' Apply NMS, if two boxes overlap more than the threshold, the one with the lower score/confidence is removed '''
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=IoUThreshold)
        return boxes[keep], labels[keep], scores[keep]
    
    def applyConfThreshold(self, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        ''' Apply confidence threshold '''
        boxes  = [box for box, conf in zip(boxes, scores) if conf > threshold]
        labels = [label for label, conf in zip(labels, scores) if conf > threshold]
        scores = [conf for conf in scores if conf > threshold]
        return boxes, labels, scores
    
    def applyLabelFilter(self, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        ''' Apply label filter '''
        boxes = [box for box, label in zip(boxes, labels) if int(label) in ASSOCIATION_LABELS.keys()]
        labels = [label for label in labels if int(label) in ASSOCIATION_LABELS.keys()]
        scores = [conf for conf, label in zip(scores, labels) if int(label) in ASSOCIATION_LABELS.keys()]
        return boxes, labels, scores
