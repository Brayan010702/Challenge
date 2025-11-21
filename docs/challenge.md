# Challenge Documentation

## Overview

This project implements an object detection system for industrial/workplace safety applications using YOLO11. The system detects 17 object classes including people, forklifts, safety helmets, and other industrial equipment.

## Part I: Model Training and Evaluation

**Training Environment**: Google Colab (Tesla T4 GPU) was used for model training to leverage free GPU resources and accelerate the training process.

### Dataset Analysis

The dataset contains images with 17 classes related to industrial safety:
- **High frequency classes**: forklift (24,213 samples), person (20,480 samples)
- **Low frequency classes**: gloves, traffic light, van (< 30 samples each)
- **Key challenge**: Severe class imbalance (ratio > 2000:1 between most and least common classes)

### Hyperparameter Selection

| Parameter | Value | Justification |
|-----------|-------|---------------|
| EPOCHS | 20 | Balance between convergence and training time on Colab |
| IMGSZ | 640 | Standard YOLO size, good balance for industrial scenes |
| BATCH | 16 | Optimized for Colab T4 GPU memory (16GB) |
| DEVICE | cuda | Colab GPU acceleration |
| MODEL | YOLO11n | Lightweight model suitable for deployment |

### Results

- **mAP50-95**: 15.3%
- **Best performing classes**: forklift, person (classes with most training data)
- **Worst performing classes**: gloves, traffic light, van (classes with < 100 samples)

### Improvement Proposals

1. **Address class imbalance**: Use Focal Loss and oversampling of rare classes
2. **Higher resolution**: Train at 1024px for better small object detection
3. **Extended training**: 80-100 epochs with cosine annealing scheduler
4. **Larger model**: YOLO11s or YOLO11m for better accuracy

## Part II: FastAPI Implementation

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns model status (`{"status": "model_loaded"}`) |
| `/predict` | POST | Accepts image file, returns detections |

### Response Format

```json
{
  "detections": [
    {
      "cls_id": 0,
      "bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 200,
        "y2": 300
      }
    }
  ]
}
```

### Local Testing

```bash
# Start server
uvicorn challenge.api:app --reload --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

## Part III: Cloud Deployment

### Platform

- **Provider**: Google Cloud Platform (GCP)
- **Service**: Cloud Run (serverless containers)
- **Region**: us-central1

### Deployment Configuration

- **Memory**: 1Gi
- **CPU**: 1
- **Authentication**: Public (unauthenticated access allowed)

### Production URL

```text
https://challenge-api-334447306714.us-central1.run.app
```

### Deployment Command

```bash
gcloud run deploy challenge-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1
```

## Part IV: CI/CD

### Continuous Integration (ci.yml)

- **Trigger**: Push or PR to `develop` or `main` branches
- **Jobs**:
  1. **Lint Job**:
     - Code quality checks with `ruff`
     - Detects potential errors and style issues
  2. **Test Job** (runs after lint passes):
     - Verifies code imports correctly
     - Starts API server locally
     - Tests `/health` endpoint functionality

This approach ensures code quality before deployment and validates that the API works correctly in a clean environment.

### Continuous Delivery (cd.yml)

- **Trigger**: Push to `main` branch
- **Steps**:
  1. Checkout code
  2. Authenticate to GCP
  3. Deploy to Cloud Run

### Required Secrets

For CD to work automatically, configure `GCP_SA_KEY` secret in GitHub repository settings with a Service Account JSON key.

## Project Structure

```text
challenge/
├── challenge/
│   ├── api.py              # FastAPI application
│   ├── exploration.ipynb   # Model development notebook
│   └── artifacts/          # Trained model weights
├── tests/                  # Test suite
├── data/                   # Dataset (not in repo)
├── workflows/              # CI/CD workflow templates
└── .github/workflows/      # Active GitHub Actions
```


## How to Run Tests

```bash
# Model tests (requires trained model and data)
make model-test

# API tests (requires running API server)
make api-test
```