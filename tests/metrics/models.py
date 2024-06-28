import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn,
    SSD300_VGG16_Weights, ssd300_vgg16,
    RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn,
)
from ultralytics import YOLO



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



class YOLOModel:
    def __init__(self) -> None:
        self.model = YOLO('yolov5x6u.pt')
    
    def __call__(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model.predict(image, conf=0.1, classes=[0, 1, 2, 3, 5, 7])  # person 0, bicycle 1, car 2, motorbike 3, bus 5, truck 7
        return output[0].boxes.xyxy, output[0].boxes.cls, output[0].boxes.conf


class FasterRCNNModel:
    def __init__(self) -> None:
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
    
    def __call__(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        with torch.no_grad():
            output = self.model(transform(image).unsqueeze(0))
        return output[0]['boxes'], output[0]['labels'], output[0]['scores']
    
    
class MaskRCNNModel:
    def __init__(self) -> None:
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        
    def __call__(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        with torch.no_grad():
            output = self.model(transform(image).unsqueeze(0))
        return output[0]['boxes'], output[0]['labels'], output[0]['scores']


class SSDModel:
    def __init__(self) -> None:
        self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        self.model.eval()
    
    def __call__(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        with torch.no_grad():
            output = self.model(transform(image).unsqueeze(0))
        return output[0]['boxes'], output[0]['labels'], output[0]['scores']


class RetinaNetModel:
    def __init__(self) -> None:
        self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
    
    def __call__(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        with torch.no_grad():
            output = self.model(transform(image).unsqueeze(0))
        return output[0]['boxes'], output[0]['labels'], output[0]['scores']