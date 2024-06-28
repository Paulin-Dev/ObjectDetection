
# Third-party imports
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def calculateIoU(box1: torch.Tensor, box2: torch.Tensor) -> float:
    ''' Computes the Intersection over Union (IoU) for two boxes '''

    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Intersection rectangle
    x_inter_min = max(x_min1, x_min2)
    y_inter_min = max(y_min1, y_min2)
    x_inter_max = min(x_max1, x_max2)
    y_inter_max = min(y_max1, y_max2)

    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)

    area_inter = inter_width * inter_height
    
    # Area of both bounding boxes
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    # IoU
    return area_inter / (area1 + area2 - area_inter)


def calculateIoUMatrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> np.ndarray:
    ''' Computes an IoU matrix for all pairs of predicted and ground truth boxes '''

    iou_matrix = np.zeros((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = calculateIoU(box1, box2)
    return iou_matrix


def matchBoxes(boxes1: torch.Tensor, boxes2: torch.Tensor, IoUthreshold: float = 0.5) -> list[tuple[int, int, float]]:
    ''' Matches predicted and ground truth boxes using the Hungarian algorithm based on IoU '''

    iou_matrix = calculateIoUMatrix(boxes1, boxes2)
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Convert IoU to cost by negating it

    output = []
    for i, j in zip(row_ind, col_ind):
        iou_value = iou_matrix[i, j]
        if iou_value >= IoUthreshold:
            output.append((i, j, iou_value))
        else:
            output.append((i, None, 0.0))
            output.append((None, j, 0.0))
    
    # Add unmatched boxes
    for i in range(len(boxes1)):
        if i not in set(row_ind):
            output.append((i, None, 0.0))
    
    for j in range(len(boxes2)):
        if j not in set(col_ind):
            output.append((None, j, 0.0))

    return output


def computeMetricsPerClass(matches: list[tuple[int, int, float]], realClasses: list, predClasses: list, associationDict: dict = {}, nbClasses: int = 8) -> dict:
    ''' Computes TP, FP, and FN for each class '''

    metricsPerClass = { i: { 'TP': 0, 'FP': 0, 'FN': 0 } for i in range(nbClasses) }

    for match in matches:
        
        # Correct detection (IoU >= threshold)
        if match[0] is not None and match[1] is not None:
                
            # Correct classification
            if realClasses[match[0]] == associationDict.get(int(predClasses[match[1]]), None):
                metricsPerClass[realClasses[match[0]]]['TP'] += 1
            
            # Incorrect classification
            else:
                metricsPerClass[realClasses[match[0]]]['FP'] += 1
        
        # Missed detection
        elif match[0] is not None:
            metricsPerClass[realClasses[match[0]]]['FN'] += 1
            
        # False positive
        elif match[1] is not None:
            metricsPerClass[associationDict.get(int(predClasses[match[1]]), None)]['FP'] += 1
    
    return metricsPerClass


def computeConfusionMatrix(matches: list[tuple[int, int, float]], realClasses: list, predClasses: list, associationDict: dict = {}, nbClasses: int = 8) -> np.ndarray:
    ''' Constructs the confusion matrix based on matches '''
    
    confusion_matrix = np.zeros((nbClasses+1, nbClasses+1), dtype=int)
    
    for match in matches:
        if match[0] is not None and match[1] is not None:
            real_class = realClasses[match[0]]
            pred_class = associationDict.get(int(predClasses[match[1]]), None)
            
            # Increment confusion matrix
            confusion_matrix[real_class, pred_class] += 1
        
        elif match[1] is not None:
            pred_class = associationDict.get(int(predClasses[match[1]]), None)
            confusion_matrix[nbClasses, pred_class] += 1
  
    return confusion_matrix



def computePrecisionRecall(metrics: dict) -> tuple[dict, dict]:
    ''' Calculates precision and recall for each class '''

    precision = {}
    recall = {}

    for class_id, metrics in metrics.items():
        TP = metrics['TP']
        FP = metrics['FP']
        FN = metrics['FN']
        
        if TP + FP > 0:
            precision[class_id] = TP / (TP + FP) if TP + FP > 0 else 0.0
        else:
            precision[class_id] = 0.0
        
        if TP + FN > 0:
            recall[class_id] = TP / (TP + FN) if TP + FN > 0 else 0.0
        else:
            recall[class_id] = 0.0
    
    return precision, recall


def computeAPFromPrecisionRecall(precisions: np.ndarray, recalls: np.ndarray) -> float:
    ''' Computes the Average Precision (AP) from precision and recall '''
    
    # Combine precision and recall into sorted pairs
    sorted_pairs = sorted(zip(recalls, precisions))
    recalls, precisions = zip(*sorted_pairs)

    # Handle missing precision value for recall 0
    if recalls[0] != 0:
        recalls = np.concatenate(([0.0], recalls))
        precisions = np.concatenate(([1.0], precisions))

    # Handle missing precision value for recall 1
    if recalls[-1] != 1:
        recalls = np.concatenate((recalls, [1.0]))
        precisions = np.concatenate((precisions, [0.0]))
    
    unique_recalls = []
    max_precisions = []
    for recall in sorted(set(recalls)):
        recall = np.round(recall, 2)
            
        if recall in unique_recalls:
            continue
            
        unique_recalls.append(recall)
        max_precisions.append(np.mean([p for r, p in zip(recalls, precisions) if np.round(r, 2) == recall]))
        
    # from plotter import plotPrecisionRecallCurve
    # plotPrecisionRecallCurve(unique_recalls, max_precisions)

    ap = 0.0
    for i in range(1, len(unique_recalls)):
        ap += max_precisions[i] * (unique_recalls[i] - unique_recalls[i-1])
    return ap


def computeAPForClasses(
        predBoxes: list[torch.Tensor],
        predConf: list[torch.Tensor],
        predClasses: list[torch.Tensor],
        groundTruthBoxes: list[list],
        groundTruthClasses: list[list],
        associationDict: dict = {},
        nbClasses: int = 8,
        IoUthresholds: np.ndarray = np.arange(0.50, 1.00, 0.05)
    ) -> tuple[dict, dict, dict, dict]:
    ''' Computes the Average Precision (AP) for each class '''

    AP_per_class = {iou_thresh: {} for iou_thresh in IoUthresholds}
    
    classes = set(sum(groundTruthClasses, []))
    # classes = set([int(item) for sublist in predClasses for item in sublist])

    # Loop through IoU thresholds
    for iou_thresh in IoUthresholds:
        print(f'\n----- Processing IoU threshold {iou_thresh} -----')

        class_precision_recall = {class_id: ([], []) for class_id in classes}
        
        # Loop though images
        for img in range(len(groundTruthBoxes)):
            print(f'Image {img + 1}/{len(groundTruthBoxes)}', end='\r')

            # Loop through confidence thresholds
            for conf_threshold in np.arange(0.25, 0.85, 0.1): # 0.25 - 0.75

                # Filter predictions by confidence threshold
                filtered_predictions = [
                    (box, cls)
                    for box, cls, conf in zip(predBoxes[img], predClasses[img], predConf[img])
                    if conf >= conf_threshold
                ]
                
                filtered_pred_boxes, filtered_pred_classes = zip(*filtered_predictions) if filtered_predictions else ([], [])
                
                # Match filtered predictions with ground truth boxes
                filtered_matches = matchBoxes(groundTruthBoxes[img], filtered_pred_boxes, iou_thresh)
                filtered_metrics = computeMetricsPerClass(filtered_matches, groundTruthClasses[img], filtered_pred_classes, associationDict, nbClasses)

                filtered_precision, filtered_recall = computePrecisionRecall(filtered_metrics)
                
                # Append precision and recall for each class
                for class_id in classes:
                    class_precision_recall[class_id][0].append(filtered_precision.get(class_id, 0.0))
                    class_precision_recall[class_id][1].append(filtered_recall.get(class_id, 0.0))

        # Compute AP for the current class
        for class_id, (precisions, recalls) in class_precision_recall.items():
            AP_per_class[iou_thresh][class_id] = computeAPFromPrecisionRecall(np.array(precisions), np.array(recalls)),
    
    print()
    
    return AP_per_class


def computeMAP(classesAP: dict, fromIoU: float, toIoU: float = None) -> float:
    ''' Computes the mean Average Precision (mAP) '''
    
    if toIoU is None:
        toIoU = fromIoU

    # Get mAP from specific IoU range
    return np.mean([AP for iou_thresh, AP_per_class in classesAP.items() if fromIoU <= iou_thresh <= toIoU for AP in AP_per_class.values()])


def computePrecision(metrics: dict) -> float:
    ''' Computes the precision '''
   
    return metrics['TP'] / (metrics['TP'] + metrics['FP']) if metrics['TP'] + metrics['FP'] > 0 else 0.0
    


def computeRecall(metrics: dict) -> float:
    ''' Computes the recall '''

    return metrics['TP'] / (metrics['TP'] + metrics['FN']) if metrics['TP'] + metrics['FN'] > 0 else 0.0


def computeF1Score(precision: float, recall: float) -> float:
    ''' Computes the F1 score '''

    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0


def computeAllMetrics(
    predBoxes: list[torch.Tensor],
    predConf: list[torch.Tensor],
    predClasses: list[torch.Tensor],
    groundTruthBoxes: list[list],
    groundTruthClasses: list[list],
    associationDict: dict = {},
    nbClasses: int = 8,
    IoUthreshold: float = 0.5
) -> tuple[dict, np.ndarray]:
    
    confusionMatrix = np.zeros((nbClasses+1, nbClasses+1), dtype=int)
    totalMetrics = { 'TP': 0, 'FP': 0, 'FN': 0 }

    for img in range(len(groundTruthBoxes)):
        matches = matchBoxes(groundTruthBoxes[img], predBoxes[img], IoUthreshold)
        metrics = computeMetricsPerClass(matches, groundTruthClasses[img], predClasses[img], associationDict, nbClasses)
        
        for key in metrics.keys():
            for metric in metrics[key]:
                totalMetrics[metric] += metrics[key][metric]

        confusionMatrix += computeConfusionMatrix(matches, groundTruthClasses[img], predClasses[img], associationDict, nbClasses)
    
    AP_per_class = computeAPForClasses(
        predBoxes, 
        predConf,
        predClasses, 
        groundTruthBoxes, 
        groundTruthClasses, 
        associationDict, 
        nbClasses,
        IoUthresholds=np.array([0.5])
    )

    precision, recall = computePrecision(totalMetrics), computeRecall(totalMetrics)

    output = {
        'mAP50': computeMAP(AP_per_class, 0.5),
        'mAP50_95': computeMAP(AP_per_class, 0.5, 0.95),
        'precision': precision,
        'recall': recall,
        'f1': computeF1Score(precision, recall),
        'TP': totalMetrics['TP'],
        'FP': totalMetrics['FP'],
        'FN': totalMetrics['FN'],
    }

    return output, confusionMatrix