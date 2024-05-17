# ObjectDetection
This repository contains scripts and resources aimed at developing an AI-based system to detect objects from city cameras.

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)


## ✨ Features
- Fetching dated images from an online camera every few seconds.
- Retrieving images labelled on **Label Studio** from an exported file (different formats supported: JSON, COCO and YOLO).
- ...


## ⚙️ Installation 

> You first need to have **Docker Compose** installed.  
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
> NB: *`VARIABLE`* means that this variable can be customized in the **.env** file.

### Fetcher
You can fetch images *`LOOP`* times, from a *`URL`*, every *`SLEEP_TIME`* seconds using the following command:
```bash
docker-compose up -d fetcher
```

### Retriever
This assumes you are using [Label Studio](https://labelstud.io/) for labelling and that you have already imported and labelled your images.

On **Label Studio**, in your project, you can find the *`PROJECT_ID`* in your browser URL. The *`ACCESS_TOKEN`* is visible in the "Account & Settings" tab (top right corner).

Back to your project, click "Export" and "Create New Snapshot". Then download it in one of the following formats: JSON, COCO or YOLO.
Move the downloaded file/folder in this project directory, and unzip the folder (for COCO and YOLO formats). You can now set the *`EXPORTED_FORMAT`* and *`EXPORTED_PATH`* in the **.env** file.

You can now retrieve your labelled images using the following command:
```bash
docker-compose up -d retriever
```
> NB: Change the *`REQUEST_INTERVAL`* if you are being rate limited.

## 🤝 Contributing 
Guidelines for contributing to the project can be found in the `docs/CONTRIBUTING.md` file.
