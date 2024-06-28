#!/usr/bin/env python3

# Built-in imports
from os import listdir, remove
from os.path import basename, dirname, join, realpath
from time import sleep

# Local imports
from models import MaskRCNNModel
from utils import Utils

# Third-party imports
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Constants
CURRENT_DIR = dirname(realpath(__file__))
IMAGES_DIR = '/images'


model = MaskRCNNModel()
utils = Utils()
analyzed_images = []


class ImageHandler(FileSystemEventHandler):
    def on_created(self, event) -> None:
        if event.is_directory:
            return
        
        filepath = event.src_path
        
        if basename(filepath).lower().endswith('.jpg'):
            analyzeImage(filepath)
            cleanupImages()


def cleanupImages() -> None:
    global analyzed_images
    for file_path in analyzed_images:
        if file_path != analyzed_images[-1]:
            remove(file_path)
    analyzed_images = [analyzed_images[-1]]
            


def analyzeImage(filePath: str) -> None:
    try:
        # Load the image
        original = utils.loadImage(filePath)
        
        # Apply mask
        image = utils.applyMask(original)
        
        # Perform object detection
        boxes, labels, scores = model.predict(image)
        
        # Count objects
        count = utils.countObjects(labels)

        output = utils.drawBoundingBoxes(original, boxes, labels, categories=utils.labels, scores=scores)
        Image.fromarray(output).save(filePath)
        
        # Save stats to JSON file
        utils.saveStats(filePath, count)
        
    except:
        pass

    finally:
        # Add image to analyzed images
        analyzed_images.append(filePath)


def monitor(folder: str) -> None:
    
    observer = Observer()
    observer.schedule(ImageHandler(), folder, recursive=False)
    observer.start()
    
    for filename in listdir(folder):
        filepath = join(folder, filename)
        if basename(filepath).lower().endswith('.jpg'):
            analyzeImage(filepath)
    cleanupImages()
    
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    monitor(IMAGES_DIR)
