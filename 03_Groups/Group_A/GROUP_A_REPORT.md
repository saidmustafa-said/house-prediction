# GROUP A – Define Objective and Collect Data (Steps 1–2)

## Objective
To explore where real-world ML data comes from, understand what it represents, and define what problem the model will solve.

## Tools Used
- **Kaggle**: https://www.kaggle.com/datasets
- **UCI Machine Learning Repository**: https://archive.ics.uci.edu/

## Dataset Selected
**House Prices – Advanced Regression Techniques**
- **Source**: Kaggle
- **Link**: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- **Backup Option**: UCI Wine Quality Dataset (https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

---

## Step-by-Step Process

### 1. Dataset Exploration
- Accessed the Kaggle dataset page
- Downloaded the dataset files (train.csv, test.csv, data_description.txt)
- Examined the first few rows and columns in the dataset

### 2. Problem Definition

**Problem Statement:**
The objective is to predict the sale price of houses based on their characteristics and features. This is a **regression problem** where we need to build a model that can estimate house prices given various input features.

### 3. Dataset Structure Analysis

#### Features (Input Variables)
The dataset contains 79 features describing various aspects of houses:

**Categorical Features:**
- Neighborhood: Physical locations within Ames city limits
- HouseStyle: Style of dwelling (e.g., 1Story, 2Story, etc.)
- RoofStyle: Type of roof (e.g., Flat, Gable, Hip, etc.)
- ExteriorCond: Exterior material condition
- Foundation: Type of foundation
- HeatingQC: Heating quality and condition
- CentralAir: Central air conditioning (Yes/No)
- Electrical: Electrical system type
- KitchenQual: Kitchen quality
- GarageType: Garage location type
- PavedDrive: Paved driveway (Yes/No/Partial)

**Numerical Features:**
- LotArea: Lot size in square feet
- YearBuilt: Original construction year
- YearRemodAdd: Remodel date
- MasVnrArea: Masonry veneer area in square feet
- BsmtFinSF1: Type 1 finished basement area
- BsmtUnfSF: Unfinished basement area
- TotalBsmtSF: Total basement area
- 1stFlrSF: First floor square feet
- 2ndFlrSF: Second floor square feet
- LivArea: Above grade living area square feet
- GarageCars: Size of garage in car capacity
- GarageArea: Size of garage in square feet
- PoolArea: Pool area in square feet
- YrSold: Year property was sold

#### Target Variable (Output)
- **SalePrice**: The price at which the house was sold (in dollars)

### 4. Dataset Dimensions
- **Training Set**: 1,460 houses with 80 features (including SalePrice)
- **Test Set**: 1,459 houses with 79 features (SalePrice to be predicted)
- **Total Features**: 79 input features + 1 target variable

### 5. Data Characteristics
- **Data Type**: Mixed (numerical and categorical)
- **Missing Values**: Some features have missing values that need to be handled
- **Price Range**: Varies from approximately $34,900 to $755,000
- **Time Period**: Houses sold between 2006 and 2010

---

## Problem Interpretation

### What is the Problem (Objective)?
To build a machine learning regression model that can accurately predict house sale prices based on 79 different features including location, size, condition, and other characteristics. This is a **supervised learning** problem where we have labeled data (known prices) to train the model.

### What are the Input Features?
The input features include:
- **Structural features**: Size (LotArea, TotalBsmtSF, LivArea), Year built, Number of rooms
- **Location features**: Neighborhood, Street type
- **Condition features**: Overall quality, Exterior condition, Roof condition
- **Amenities**: Garage type/size, Pool area, Porch area
- **Utilities**: Central air, Heating type, Electrical system

### What is the Output Variable?
The output variable is **SalePrice** – the actual selling price of the house in dollars. This is a continuous numerical value that the model needs to predict.

---

## Key Insights

1. **Real-World Relevance**: This dataset represents actual house sales data from Ames, Iowa, making it highly relevant for practical ML applications.

2. **Feature Diversity**: The dataset includes both numerical and categorical features, requiring data preprocessing and encoding techniques.

3. **Regression Task**: Unlike classification (predicting categories), this is a regression task where we predict continuous values.

4. **Data Quality**: The dataset is well-structured with clear feature descriptions, making it ideal for learning ML workflows.

---

## Connection to Other Groups

- **Group B** will use this dataset to prepare the data and select appropriate algorithms
- **Group C** will train and test models using this dataset
- **Group D** will visualize and interpret the results using this dataset

---

## Submission Summary

This group successfully:
✓ Identified and accessed the House Prices dataset from Kaggle
✓ Analyzed the dataset structure and features
✓ Defined the ML problem as a regression task
✓ Documented all 79 input features and the target variable
✓ Provided clear problem statement and dataset interpretation
✓ Prepared comprehensive documentation for other groups

