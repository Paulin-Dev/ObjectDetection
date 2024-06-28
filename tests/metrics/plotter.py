import matplotlib.pyplot as plt
import numpy as np



def plotConfusionMatrix(confusion_matrix: np.ndarray, allClasses: dict, save: bool = False):
    
    classes = list(allClasses.keys())

    fig, ax = plt.subplots(figsize=(10, 8))

    #im = ax.matshow(confusion_matrix, cmap='Blues')
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Set axis labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(allClasses.values(), rotation=45)
    
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(allClasses.values())

    # Add color bar
    fig.colorbar(im)
        
    # Set labels for axes
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Loop over data dimensions and create text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = 'white' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black'
            ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color=text_color)

    if save:
        plt.savefig('confusion_matrix.png')
    else:
        plt.show()
    


def plotPrecisionRecallCurve(recalls, precisions) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve')
    plt.grid(True)
    plt.show()