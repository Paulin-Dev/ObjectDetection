#!/usr/bin/env python3

# Built-in imports
import json
from os.path import dirname, join, realpath

# Third-party imports
import cv2
import numpy as np
from PIL import Image
import torch

# Constants
ROOT_DIR   = dirname(realpath(__file__))
ASSETS_DIR = join(ROOT_DIR, 'assets')
RESULTS_DIR = join(ROOT_DIR, 'results')
LABELS = {
    0: 'Bicycle',
    1: 'Bus',
    2: 'Car',
    3: 'Motor scooter',
    4: 'Motorbike',
    5: 'Person', # Pedestrian
    6: 'Scooter',
    7: 'Truck'
}


class Utils:
    def __init__(self) -> None:
        self.mask = self.loadMask(join(ASSETS_DIR, 'mask.jpg'))
        self.labels = LABELS

    def loadImage(self, path: str) -> Image:
        return Image.open(path, 'r', formats=['JPEG']).convert('RGB')
    
    def loadMask(self, path: str) -> np.ndarray: 
        return np.array(Image.open(path, 'r', formats=['PNG', 'JPEG']).convert('RGB')) > 127

    def applyMask(self, image) -> np.ndarray:
        return np.array(image) * self.mask

    def countObjects(self, labels: torch.Tensor) -> None:
        return {self.labels[label]: labels.count(label) for label in set(labels)}

    def drawBoundingBoxes(self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, categories: dict = {}, color: str = (255, 0, 0), scores: torch.Tensor = None) -> np.ndarray:
        output = np.array(image)
        for index, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            cv2.putText(output, f'{categories[int(label)] if categories != {} else ""} {round(float(scores[index]), 2) if scores is not None else ""}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return output
    
    def saveStats(self, filename: str, count: dict) -> None:
        with open(join(RESULTS_DIR, 'data.json'), 'a+') as f:
            f.seek(0)

            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            
            data[filename[-18:-4]] = count
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4)
  