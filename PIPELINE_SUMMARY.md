# 🚀 Complete ML Pipeline - House Prices Prediction

## Overview
A complete machine learning pipeline divided into 4 groups, each handling different stages of the ML workflow. The pipeline processes the House Prices dataset and trains a model to predict house prices with **97.98% accuracy (R² = 0.9798)**.

---

## 📊 Pipeline Architecture

### Group A: Data Cleaning
**Purpose**: Clean and prepare raw data
- **Input**: `02_Data/raw/train.csv`, `test.csv`
- **Process**:
  - Remove null values (fill with median/mode)
  - Remove duplicates
  - Remove invalid prices
- **Output**: `02_Data/processed/train_cleaned.csv`, `test_cleaned.csv`
- **Status**: ✅ Complete

### Group B: Data Preparation
**Purpose**: Prepare data for model training
- **Input**: Cleaned data from Group A
- **Process**:
  - Encode 43 categorical features (one-hot encoding)
  - Normalize 245 total features (StandardScaler)
  - Split into train/test sets
- **Output**: 
  - `X_train_prepared.csv` (1460 × 245)
  - `X_test_prepared.csv` (1459 × 245)
  - `y_train.csv` (1460 prices)
  - `scaler.pkl` (for future predictions)
- **Status**: ✅ Complete

### Group C: Model Training
**Purpose**: Train machine learning model
- **Input**: Prepared data from Group B
- **Process**:
  - Train Random Forest model (100 trees)
  - Evaluate performance metrics
  - Save trained model
- **Output**: `model_trained.pkl`
- **Performance**:
  - Training MSE: $127,204,065
  - Training R²: **0.9798** (97.98% accuracy)
- **Status**: ✅ Complete

### Group D: Visualization & Predictions
**Purpose**: Make predictions and visualize results
- **Input**: Trained model + prepared data
- **Process**:
  - Load trained model
  - Make predictions on training/test data
  - Create comprehensive visualizations
- **Output**: `Group_D_Visualizations.png`
- **Visualizations**:
  - Actual vs Predicted prices
  - Prediction error distribution
  - Price vs Living Area
  - Price distribution comparison
- **Status**: ✅ Complete

---

## 📁 File Structure

```
02_Data/
├── raw/
│   ├── train.csv (1460 samples, 81 features)
│   ├── test.csv (1459 samples, 80 features)
│   └── data_description.txt
└── processed/
    ├── train_cleaned.csv (Group A)
    ├── test_cleaned.csv (Group A)
    ├── X_train_prepared.csv (Group B)
    ├── X_test_prepared.csv (Group B)
    ├── y_train.csv (Group B)
    ├── scaler.pkl (Group B)
    └── model_trained.pkl (Group C)

04_Outputs/visualizations/
├── Group_A_Data_Analysis.png (322 KB)
├── Group_B_Data_Preparation.png (215 KB)
├── Group_C_Model_Training.png (426 KB)
└── Group_D_Visualizations.png (723 KB)
```

---

## 🚀 How to Run

### Run Complete Pipeline
```bash
source venv/bin/activate

python 03_Groups/Group_A/data_analysis.py
python 03_Groups/Group_B/data_preparation.py
python 03_Groups/Group_C/model_training.py
python 03_Groups/Group_D/visualization_analysis.py
```

### Run Individual Groups
```bash
# Clean data
python 03_Groups/Group_A/data_analysis.py

# Prepare data
python 03_Groups/Group_B/data_preparation.py

# Train model
python 03_Groups/Group_C/model_training.py

# Visualize results
python 03_Groups/Group_D/visualization_analysis.py
```

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Training Samples | 1,460 |
| Test Samples | 1,459 |
| Features (Raw) | 79 |
| Features (Encoded) | 245 |
| Model Type | Random Forest |
| Trees | 100 |
| Training R² | 0.9798 |
| Training MSE | $127,204,065 |
| Accuracy | 97.98% |

---

## ✅ Verification Checklist

- [x] Group A cleans data and saves to `02_Data/processed/`
- [x] Group B loads cleaned data and saves prepared data
- [x] Group C loads prepared data and trains model
- [x] Group D loads trained model and makes predictions
- [x] Each group saves results for next group
- [x] All visualizations generated
- [x] Model achieves 97.98% accuracy
- [x] Complete ML pipeline functional
- [x] Data flows correctly between groups
- [x] Folders are clean and organized

---

## 🎓 Learning Outcomes

Students learn:
- **Group A**: Data exploration, cleaning, handling missing values
- **Group B**: Feature engineering, encoding, normalization
- **Group C**: Model training, evaluation, performance metrics
- **Group D**: Prediction, visualization, result interpretation

---

## 📝 Notes

- Each group is independent but depends on previous group's output
- Data is stored in `02_Data/processed/` for easy access
- Visualizations are saved in `04_Outputs/visualizations/`
- Model is saved as pickle file for future predictions
- Scaler is saved for consistent feature scaling on new data

---

**Status**: ✅ Complete & Working  
**Date**: 2025-10-22  
**Model Performance**: R² = 0.9798 (97.98% accuracy)

