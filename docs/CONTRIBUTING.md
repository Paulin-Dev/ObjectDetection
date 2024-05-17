# Contributing Guidelines

Thank you for considering contributing to this project! 🙌

## 📋 Table of Contents

- [Overview](#-overview)
- [Services](#%EF%B8%8F-services)
  - [Fetcher](#fetcher)
  - [Retriever](#retriever)
- [Reporting Bugs](#-reporting-bugs)
- [Suggesting Features](#-suggesting-features)
- [Submitting Code Changes](#-submitting-code-changes)
- [Code Style](#-code-style)


## 🌐 Overview

This project aims to provide a user-friendly experience by relying on a single `docker-compose.yml` file to manage the services and an `.env` file to store the environment variables. 

By using this approach, the project shifts the complexity from the CLI to configuration files and can be easily deployed on any machine with Docker and Docker Compose installed.

> Note that only the `.env.example` file should be committed to the repository.

## 🛠️ Services

The project is divided into services, each service having its own directory inside the `Docker` directory. 

Each service is responsible for a specific task and will run in a separate container. 

You can use any language to implement a service, as long as it can be containerized with all its dependencies, which have to be specified in a file like `requirements.txt` for Python or `package.json` for NodeJS.

Each service should have its own `Dockerfile` as well as its section in the `docker-compose.yml` file, which will be used to build the image and run the container respectively.

Every useless file can be added to the `.gitignore` file to prevent it from being committed.

### Fetcher
Bash script that fetches images from a given URL and stores them in a directory. 

- Reads exif data such as the size of the image and the time it was taken, making sure it's not corrupted or already present in the directory. 
- Needs to have a volume to store data on the host machine.
- Retries if the image is corrupted or if it's the same as the previous one.

*Debian image with the following packages :*
```bash
wget
libimage-exiftool-perl
```

### Retriever
JavaScript script that reads exported files from Label Studio and uses the Label Studio API to download the related images.

- Different formats are supported: JSON, COCO and YOLO.
- Needs to have a volume to store data on the host machine.
- Checks if the image is already present in the directory.

*NodeJS image with no additional packages.*



## 🐞 Reporting Bugs

If you encounter a bug in the project, please open a new issue on GitHub and include as much detail as possible. Describe the steps to reproduce the bug, what you expected to happen, and what actually happened. Screenshots, error messages, and code snippets are all helpful.

## 💡 Suggesting Features

Have an idea for a new feature or improvement? We'd love to hear it! Open a new issue on GitHub and provide a clear description of the feature, along with any relevant context or use cases.

## 💻 Submitting Code Changes

1. Fork the repository and create a new branch for your changes.
2. Ensure your code follows the coding standards and conventions.
3. Test your changes thoroughly.
4. Submit a pull request with a clear description of the changes you've made and why they're necessary.
5. Be responsive to feedback and be prepared to make further changes if requested.

> Don't forget to update the `README.md` file and `CONTRIBUTING.md` file with any relevant information about your changes.

## 🎨 Code Style

- Follow the coding style and conventions used throughout the project.
- Use meaningful variable and function names.
- Write clear and concise comments to explain your code.
- Keep lines of code reasonably short and avoid overly complex logic.


