#!/usr/bin/env python3

# Built-in imports
import json

# Local imports
from metrics import computeAllMetrics
from models import YOLOModel, FasterRCNNModel, MaskRCNNModel, SSDModel, RetinaNetModel, COCO_LABELS
from plotter import plotConfusionMatrix

# Third-party imports
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision



def loadAnnotations(path: str) -> dict:
    output = {'images': [], 'categories': []}
    with open(path, 'r') as file:
        data = json.load(file)
        
        output['categories'] = {category['id']: category['name'] for category in data['categories']}
        
        for img in data['images']:
            bbox, categories = [], []
            for annotation in data['annotations']:
                if annotation['image_id'] == img['id']:
                    bbox.append([annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0] + annotation['bbox'][2], annotation['bbox'][1] + annotation['bbox'][3]])
                    categories.append(annotation['category_id'])
                else:
                    if bbox:
                        break
                    
            if bbox:
                output['images'].append({
                    'id': img['id'],
                    'filename': img['file_name'].split('/')[1],
                    'bbox': bbox,
                    'categories': categories,
                })

    return output
            

def loadImage(path: str) -> Image:
    return Image.open(path, 'r', formats=['JPEG']).convert('RGB')


def loadMask(path: str) -> np.ndarray: 
    return np.array(Image.open(path, 'r', formats=['PNG', 'JPEG']).convert('RGB')) > 127


def applyMask(image, mask) -> np.ndarray:
    return np.array(image) * mask


def drawBoundingBoxes(image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, categories: dict = {}, color: str = (255, 0, 0), scores: torch.Tensor = None) -> np.ndarray:
    output = np.array(image)
    for index, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.putText(output, f'{categories[int(label)] if categories != {} else ""} {round(float(scores[index]), 2) if scores is not None else ""}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return output



ASSOCIATION_LABELS_YOLO = {
    0: 5,
    1: 0,
    2: 2,
    3: 4,
    5: 1,
    7: 7,
}

ASSOCIATION_LABELS_COCO = {
    1: 5,
    2: 0,
    3: 2,
    4: 4,
    6: 1,
    8: 7,
}


def main() -> None:
    
    associationDict = ASSOCIATION_LABELS_YOLO

    data = loadAnnotations('./coco/result.json')
    mask = loadMask('./assets/mask.jpg')

  
    model = MaskRCNNModel()
    APPLY_MASK = True

    allPredBoxes = []
    allPredConfidences = []
    allPredClasses = []
    
    allGroundTruthBoxes = []
    allGroundTruthClasses = []
    
    MAX_IMAGES = 20
    # SHOW_IMG   = np.random.randint(0, MAX_IMAGES)
    SHOW_IMG   = 999
    

    for index, img in enumerate(data['images']):
        
        # if index == 0:
        #     # save it
        #     Image.fromarray(applyMask(loadImage(f'./coco/images/{img["filename"]}'), mask)).save('masked.jpg')
        #     exit()
         
        if index == MAX_IMAGES:
            break
        
        print(f'Processing image {index + 1}/{len(data["images"])} ({img["filename"]})', end='\r')

        original = loadImage(f'./coco/images/{img["filename"]}')
        if APPLY_MASK:
            original_masked = applyMask(original, mask)
        else:
            original_masked = original
        
        boxes, classes, confidences = model(original_masked)
        
        # Apply NMS, (iou threshold = if two boxes overlap more than the threshold, the one with the lower confidence is removed)
        keep = torchvision.ops.nms(boxes, confidences, iou_threshold=0.5)
        boxes = boxes[keep]
        classes = classes[keep]
        confidences = confidences[keep]
        
        # Remove boxes that are not in the association dictionary
        boxes = [box for box, label in zip(boxes, classes) if int(label) in associationDict.keys()]
        classes = [cls for cls in classes if int(cls) in associationDict.keys()]
        confidences = [conf for conf, label in zip(confidences, classes) if int(label) in associationDict.keys()]
        
        # Remove boxes with confidence lower than 0.5
        confidence = 0.75
        boxes = [box for box, conf in zip(boxes, confidences) if conf > confidence]
        classes = [cls for cls, conf in zip(classes, confidences) if conf > confidence]
        confidences = [conf for conf in confidences if conf > confidence]

        allPredBoxes.append(boxes)
        allPredClasses.append(classes)
        allPredConfidences.append(confidences)
        
        # Remove boxes that are not in the association dictionary
        img['bbox'] = [box for box, label in zip(img['bbox'], img['categories']) if label in associationDict.values()]
        img['categories'] = [label for label in img['categories'] if label in associationDict.values()]
        
        allGroundTruthBoxes.append(img['bbox'])
        allGroundTruthClasses.append(img['categories'])
 
        original_with_boxes = drawBoundingBoxes(original, boxes, classes, COCO_LABELS, (255, 0, 0), scores=confidences)
        original_with_boxes = drawBoundingBoxes(original_with_boxes, img['bbox'], img['categories'], data['categories'], (0, 255, 0))

        # show random image
        if index == SHOW_IMG:
            Image.fromarray(original_with_boxes).show()
            # save
            Image.fromarray(original_with_boxes).save('OUTPUUUUT.jpg')
    


    metrics, confusionMatrix = computeAllMetrics(
        predBoxes=allPredBoxes,
        predConf=allPredConfidences,
        predClasses=allPredClasses,
        groundTruthBoxes=allGroundTruthBoxes,
        groundTruthClasses=allGroundTruthClasses,
        associationDict=associationDict,
        nbClasses=len(data['categories']),
        IoUthreshold=0.5,
    )

    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("F1 Score:", metrics['f1'])
    print("mAP@50:", metrics['mAP50'])
    print("mAP@50-95:", metrics['mAP50_95'])
    print(f'TP: {metrics["TP"]}, FP: {metrics["FP"]}, FN: {metrics["FN"]}')
    
    #data['categories'][len(data['categories'])] = 'Non existent'
    plotConfusionMatrix(confusionMatrix, data['categories'], save=True)



if __name__ == '__main__':
    main()