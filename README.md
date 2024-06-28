# ObjectDetection
This repository contains scripts and resources aimed at developing an AI-based system to detect objects from city cameras.

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#-usage)
- [Metrics](#-metrics)
- [Contributing](#-contributing)


## ✨ Features
- Fetching dated images from an online camera every few seconds.
- Retrieving images labelled on **Label Studio** from an exported file (different formats supported: JSON, COCO and YOLO).
- ...


## ⚙️ Installation 

> You first need to have **Docker Compose** installed (make sure the Docker daemon is running).  
> The easiest and recommended way is to install Docker Desktop. Refer to the [Docker documentation](https://docs.docker.com/get-docker/).

Open a new terminal and follow these steps to set up the project:

1. **Clone the repository**  
    ```bash
    git clone https://github.com/Paulin-Dev/ObjectDetection.git
    ```

2. **Navigate to the project directory**  
    ```bash
    cd ObjectDetection
    ```

3. **Clone the environment file**  
    ```bash
    cp .env.example .env
    ```
    > In the next section, you will have a better understanding of each variable and its purpose, allowing you to customize the configuration.


## 🚀 Usage 
> In Docker Desktop, you can view the containers with their status and logs.

❗ *`VARIABLE`* means that this variable can be customized in the **.env** file.

### Fetcher
You can fetch *`LOOP`* images, from a *`URL`*, every *`SLEEP_TIME`* seconds using the following command:
```bash
docker-compose up -d fetcher
```

### Retriever
This assumes you are using [Label Studio](https://labelstud.io/) for labelling and that you have already uploaded and labelled your images.

On **Label Studio**, in your project, you can find the *`PROJECT_ID`* in your browser URL. The *`ACCESS_TOKEN`* is visible in the "Account & Settings" tab (top right corner).

Back to your project, click "Export" and "Create New Snapshot". Then download it in one of the following formats: JSON, COCO or YOLO.
Move the downloaded file/folder to the root directory of this GitHub repository, and unzip the folder (for COCO and YOLO formats). You can now set/change the *`EXPORTED_FORMAT`* and *`EXPORTED_PATH`* in the **.env** file.

You can now retrieve your labelled images using the following command:
```bash
docker-compose up -d retriever
```
> NB: Change the *`REQUEST_INTERVAL`* if you are being rate limited.  
> Images will be downloaded to the "JSON_images" directory at the project's root for JSON format, and to the "images" directory within exported directories for COCO and YOLO formats.

### Object Detector
You can run the detector using the following command:
```bash
docker-compose up -d object-detector
```

### Dashboard
You can run the dashboard using the following command:
```bash
docker-compose up -d dashboard
```

## 📊 Metrics
<!-- 
|     Model                            | IoU  | Precision | Recall | F1 Score | AP | mAP<sup>50</sup>  | 
| ------------------------------------ | :--: | :-------: | :----: | :------: | :---: | :---: | 
| YOLOv10                              | 0.85 |   0.90    |  0.87  |   0.80   |
| Faster R-CNN                         | 0.90 |   0.85    |  0.87  |   0.85   |
| SSD (Single Shot MultiBox Detector)  | 0.80 |   0.95    |  0.87  |   0.75   |
| Haar Cascade Classifier              | 0.75 |   0.80    |  0.77  |   0.70   |
| MMOD (max-margin object-detection)   | 0.85 |   0.90    |  0.87  |   0.80   | -->


## 🤝 Contributing 
Guidelines for contributing to the project can be found in the [CONTRIBUTING.md](https://github.com/Paulin-Dev/ObjectDetection/blob/main/docs/CONTRIBUTING.md) file.
