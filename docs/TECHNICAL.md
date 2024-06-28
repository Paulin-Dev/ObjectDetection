# Technical Overview

## Metrics
<!-- 
https://labelyourdata.com/articles/object-detection-metrics -->

- ### Intersection over Union (IoU)

    Measures the overlap between the predicted bounding box and the ground truth bounding box.

    $IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}$

    IoU is used to determine if a prediction is a true positive or a false positive based on a threshold (commonly 0.5).

- ### Precision

    Measures the accuracy o f the positive predictions.  

    $Precision = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$

- ### Recall

    Measures the model's ability to find all the relevant instances in the dataset.

    $Recall = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$

- ### F1 Score

    Harmonic mean of precision and recall, providing a balance between the two.

    $F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

 
- ### Average Precision (AP)

    Measures the precision across different recall levels.

    Calculated as the area under the precision-recall curve.

    Often computed at different IoU thresholds (e.g., AP@0.5, AP@0.75).

    $AP = \int_{0}^{1} \text{Precision}(r) \, dr$


- ### Mean Average Precision (mAP)

    Average of AP across all classes and IoU thresholds.

    A comprehensive measure of the overall performance of the object detection model.

    $mAP = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i$



