# Machine Learning Assignment 3
By Maha Qaiser 22i-2348

This assignment involves building a machine learning pipeline for predicting housing prices using the **California Housing** dataset from `sklearn.datasets`. The project applies various regression techniques and training optimizations including:
- Batch, Stochastic, and Mini-Batch Gradient Descent  
- L1/L2 Regularization  
- Early Stopping  
- Model Deployment via Hugging Face  
- Inference script and web interface with Flask

![Screenshot 2025-04-13 231029](https://github.com/user-attachments/assets/3e2c1eb5-1871-4f49-be49-b13a58dd6d6f)

Hugging Face Link: https://huggingface.co/mahaqj/ml_assignment_3
W&B Link: https://wandb.ai/mahaqj-/california-housing-prediction

---
## Files Included
- `best_model.joblib`: Trained Mini-Batch Linear Regression model with Ridge regularization  
- `scaler.joblib`: Fitted `StandardScaler` for input preprocessing  
- `inference.py`: CLI-based script to load model + scaler and predict housing prices based on user input  
---

## Dataset & Features
**Source**: California Housing Dataset (`sklearn.datasets.fetch_california_housing`)  
**Target Variable**: Median House Value (Price)
**Input Features:**
- `MedInc`: Median income of households (×10,000 USD)  
- `HouseAge`: Median age of the houses (years)  
- `AveRooms`: Average number of rooms per household  
- `AveBedrms`: Average number of bedrooms per household  
- `Population`: Total population in the block group  
- `AveOccup`: Average number of occupants per household  
- `Latitude`: Geographical latitude  
- `Longitude`: Geographical longitude

---

## Model Overview

- **Model Type:** Mini-Batch Linear Regression  
- **Preprocessing:** StandardScaler  
- **Regularization:** Ridge (L2)  
- **Early Stopping:** Enabled to prevent overfitting

---

## How to Run Inference

### 1. Clone the Repository
```bash
git clone https://huggingface.co/mahaqj/ml_assignment_3
cd ml_assignment_3
```

### 2. Install Required Libraries
```bash
pip install joblib numpy scikit-learn huggingface_hub
```

### 3. Run the Script
```bash
python inference.py
```

You’ll be prompted to enter the following 8 features:
- Avg. Rooms  
- Avg. Bedrooms  
- Population  
- Household  
- Median Income  
- Latitude  
- Longitude  
- Housing Median Age  

The model will return a predicted median house value.

---

## Load the Model in Python

```python
import joblib
import requests
from io import BytesIO

# URLs to download model and scaler from Hugging Face
model_url = "https://huggingface.co/mahaqj/ml_assignment_3/resolve/main/best_model.joblib"
scaler_url = "https://huggingface.co/mahaqj/ml_assignment_3/resolve/main/scaler.joblib"

# Load model
model = joblib.load(BytesIO(requests.get(model_url).content))

# Load scaler
scaler = joblib.load(BytesIO(requests.get(scaler_url).content))
```
