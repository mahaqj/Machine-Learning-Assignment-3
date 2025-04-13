# Machine Learning Assignment 3
By Maha Qaiser 22i-2348

This assignment involves building a machine learning pipeline for predicting housing prices using the **California Housing** dataset from `sklearn.datasets`. The project applies various regression techniques and training optimizations including:
- Batch, Stochastic, and Mini-Batch Gradient Descent  
- L1/L2 Regularization  
- Early Stopping  
- Model Deployment via Hugging Face  
- Inference script and web interface with Flask

![Screenshot 2025-04-13 231029](https://github.com/user-attachments/assets/3e2c1eb5-1871-4f49-be49-b13a58dd6d6f)

### Run the Web App
```bash
python app.py
```

Then open your browser and go to [http://localhost:5000](http://localhost:5000) to view the web interface.

---

## Links

- **Hugging Face Link**: [Mini-Batch Regression Model](https://huggingface.co/mahaqj/ml_assignment_3)
- **W&B Link**: [Training Dashboard](https://wandb.ai/mahaqj-/california-housing-prediction)

## Hugging Face Files
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
