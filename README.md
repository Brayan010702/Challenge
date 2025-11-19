# Machine Learning Engineer Challenge

## Overview

Welcome to the **Software Engineer (ML & Computer Vision)** Application Challenge. In this, you will have the opportunity to demonstrate your skills and knowledge in machine learning, computer vision and cloud.

## Problem

In this challenge, you are tasked with developing an **object detection system** capable of identifying multiple classes of objects in industrial or workplace environments — such as people, forklifts, and safety helmets — using **computer vision techniques**.

Your goal is to:
1. **Train** and **evaluate** a YOLO-based model (e.g., YOLOv8 or YOLO11) using the provided dataset.
2. **Analyze** the dataset quality, class balance, and visual variability.
3. **Deploy** the trained model as an inference-ready artifact that can later be served through a FastAPI endpoint.

---

### 🎯 Main Problem to Solve
The primary problem is to **accurately detect and localize multiple object categories** in challenging, real-world conditions — for instance:
- Different lighting conditions or camera angles.
- Partially occluded objects (e.g., people behind forklifts).
- Varying object scales and positions in the scene.

This requires a robust model that can generalize well from the available training data.


## Challenge

### Data


The dataset required to train the model is not included in this repository due to its size.
You can download it from the following Google Drive link:

🔗 [Download dataset](https://drive.google.com/drive/folders/1XEU1eUnQOf5alRipPJ37myGn2C2jeZmY?usp=sharing)

Important:
Once downloaded, the data folder must be placed in the root directory of the project (./data).
The repository structure should remain exactly as provided, since the notebooks and scripts depend on this path.

### Instructions

1. Create a repository in **github** and copy all the challenge content into it. Remember that the repository must be **public**.

2. Use the **main** branch for any official release that we should review. It is highly recommended to use [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) development practices. **NOTE: do not delete your development branches.**
   
3. Please, do not change the structure of the challenge (names of folders and files).
   
4. All the documentation and explanations that you have to give us must go in the `challenge.md` file inside `docs` folder.

5. After completing the challenge, upload your project to a public GitHub repository and ensure it contains all the required files. Then, send the link to your repository as a backup to the following email address: matias.rojas@latam.com, ricardo.valdivia@latam.com

***NOTE: We recommend to send the challenge even if you didn't manage to finish all the parts.***

### Context:

The goal of this challenge is to operationalize a computer vision model that can detect multiple object classes in workplace or industrial environments (e.g., people, forklifts, helmets).
To achieve this, you will work within a pre-configured Jupyter notebook that contains placeholders for each stage of the machine learning workflow — from data exploration to model evaluation.

Ultimately, the objective is to prepare a trained and validated model that could later be exposed through an API service for real-time image inference.
This notebook represents the data science foundation that will later be integrated into a production pipeline.

*We recommend reading the entire challenge (all its parts) before you start developing.*

### Part I

In this part of the challenge, you will work exclusively within the provided Jupyter notebook.
Your goal is to complete the notebook by filling in the placeholder sections to develop, train, and evaluate an object detection model.

Instructions
Carefully review the notebook and fill in all placeholders (marked with TODO comments).
These include:
> -	Dataset exploration and visualization.
> -	Hyperparameter definitions (e.g., EPOCHS, IMGSZ, BATCH, DEVICE).
> -	Model training (model.train(...)).
> -	Model evaluation and metric reporting (model.val(...)).
> -	Written analysis and interpretation of results.
> -	If you encounter any code errors or inconsistencies, fix them as needed so the notebook runs end-to-end without breaking.
> -	The notebook already provides a baseline YOLO architecture.

You may choose the variant that performs best (e.g., yolov8n.pt or yolo11n.pt).


- Apply good programming and machine learning practices, such as:
- Clean, readable code and proper documentation.
- Sensible use of random seeds, configurations, and reproducibility.
- Organized outputs (plots, metrics, exported artifacts).
- Ensure your notebook runs successfully from top to bottom and produces the expected outputs in the artifacts/ directory.

### Part II

Deploy the model in an `API` with `FastAPI` using the `api.py` file.

- The `API` should pass the tests by running `make api-test`.

> **Note:** 
> - **You cannot** use other framework.

### Part III

Deploy the `API` in your favorite cloud provider (we recomend to use GCP).

- Put the `API`'s url in the `Makefile` (`line 26`).

> **Note:** 
> - **It is important that the API is deployed until we review the tests.**

### Part IV

We are looking for a proper `CI/CD` implementation for this development.

- Create a new folder called `.github` and copy the `workflows` folder that we provided inside it.
- Complete both `ci.yml` and `cd.yml`(consider what you did in the previous parts).