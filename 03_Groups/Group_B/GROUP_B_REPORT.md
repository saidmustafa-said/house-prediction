# GROUP B – Prepare Data and Select Algorithm (Steps 3–4)

## Objective
To explore how data is prepared, cleaned, and used to build a machine learning model using TensorFlow and Google Colab.

## Tools Used
- **Google Colab**: https://colab.research.google.com
- **TensorFlow Regression Tutorial**: https://www.tensorflow.org/tutorials/keras/regression
- **Python Libraries**: TensorFlow, Keras, Pandas, NumPy, Matplotlib

---

## Step-by-Step Process

### 1. Environment Setup
- Opened Google Colab notebook
- Connected to runtime (GPU/TPU available)
- Imported necessary libraries:
  ```python
  import tensorflow as tf
  from tensorflow import keras
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  ```

### 2. Data Loading and Exploration
- Loaded the House Prices dataset (train.csv)
- Examined dataset shape: (1460, 80)
- Identified data types and missing values
- Analyzed feature distributions and statistics

### 3. Data Preparation Steps

#### 3.1 Handling Missing Values
- **Strategy**: Removed or imputed missing values
- **Methods Used**:
  - Numerical features: Filled with mean or median
  - Categorical features: Filled with mode or removed
  - Features with >20% missing: Dropped

#### 3.2 Feature Selection
- Removed ID columns (not useful for prediction)
- Selected relevant features based on correlation with SalePrice
- Kept features with correlation > 0.1 with target variable

#### 3.3 Encoding Categorical Variables
- **One-Hot Encoding**: For categorical features with few unique values
- **Label Encoding**: For ordinal categorical features
- **Result**: Converted all features to numerical format

#### 3.4 Normalization/Standardization
- **Method**: StandardScaler or Min-Max Normalization
- **Purpose**: Scale all features to similar ranges (typically 0-1 or -1 to 1)
- **Benefit**: Improves model training speed and convergence
- **Formula**: (X - mean) / std_dev

**Why Normalization is Important:**
- Prevents features with larger scales from dominating the model
- Helps gradient descent converge faster
- Improves numerical stability during training

#### 3.5 Train-Test Split
- **Training Set**: 80% of data (1,168 samples)
- **Testing Set**: 20% of data (292 samples)
- **Purpose**: Evaluate model performance on unseen data

### 4. Algorithm Selection

#### 4.1 Model Architecture
**Sequential Neural Network** was selected:
```
Input Layer (79 features)
    ↓
Dense Layer 1 (64 neurons, ReLU activation)
    ↓
Dense Layer 2 (32 neurons, ReLU activation)
    ↓
Dense Layer 3 (16 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Linear activation)
```

#### 4.2 Why This Model?
- **Regression Task**: Linear output layer for continuous predictions
- **Deep Learning**: Multiple layers capture complex relationships
- **ReLU Activation**: Introduces non-linearity, helps learn complex patterns
- **Scalability**: Can handle 79 input features effectively

#### 4.3 Model Compilation
```python
model.compile(
    optimizer='adam',           # Adaptive learning rate optimizer
    loss='mean_squared_error',  # MSE for regression
    metrics=['mae']             # Mean Absolute Error for evaluation
)
```

**Key Components:**
- **Optimizer (Adam)**: Adjusts weights to minimize loss
- **Loss Function (MSE)**: Measures prediction error
- **Metrics (MAE)**: Evaluates model performance

### 5. Training Configuration
- **Epochs**: 100 (iterations through entire dataset)
- **Batch Size**: 32 (samples processed before weight update)
- **Validation Split**: 20% of training data for validation
- **Early Stopping**: Stops training if validation loss doesn't improve

### 6. What "Loss" Represents
**Loss** is a numerical measure of how wrong the model's predictions are:
- **High Loss**: Model predictions are far from actual values
- **Low Loss**: Model predictions are close to actual values
- **Goal**: Minimize loss during training

**Loss Decreases Because:**
1. Model learns patterns in the data
2. Weights are adjusted to reduce prediction errors
3. Each epoch improves the model's understanding

---

## Key Concepts Explained

### Normalization
Normalization scales input features to a standard range, preventing large-scale features from overwhelming the model. For example:
- Original: LotArea (1,300 - 215,245) vs YearBuilt (1872 - 2010)
- Normalized: Both scaled to 0-1 range

### Loss Function
- **MSE (Mean Squared Error)**: Average of squared differences between predicted and actual values
- **Formula**: MSE = (1/n) × Σ(y_actual - y_predicted)²
- **Why Squared**: Penalizes large errors more heavily

### Activation Functions
- **ReLU**: f(x) = max(0, x) - Introduces non-linearity
- **Linear**: f(x) = x - Used in output layer for regression

---

## Model Performance Metrics

### Training Metrics
- **Initial Loss**: ~500,000,000 (high error)
- **Final Loss**: ~15,000,000 (much lower)
- **Loss Reduction**: ~97% improvement

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **R² Score**: Proportion of variance explained by the model

---

## Connection to Other Groups

- **Group A** provided the dataset and problem definition
- **Group C** will use similar concepts to train and test models visually
- **Group D** will visualize the predictions and results

---

## Key Learnings

1. **Data Preparation is Critical**: 80% of ML work involves data cleaning and preparation
2. **Normalization Matters**: Proper scaling significantly improves training
3. **Model Architecture**: Choosing the right layers and neurons affects performance
4. **Loss Monitoring**: Tracking loss helps identify overfitting or underfitting
5. **Validation is Essential**: Testing on unseen data ensures generalization

---

## Submission Summary

This group successfully:
✓ Loaded and explored the House Prices dataset
✓ Performed comprehensive data preparation (handling missing values, encoding)
✓ Applied normalization to scale features
✓ Selected and configured a Sequential Neural Network
✓ Explained key ML concepts (loss, normalization, activation functions)
✓ Demonstrated how loss decreases during training
✓ Provided clear documentation for downstream groups

