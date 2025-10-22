"""
GROUP A - Data Collection, Exploration & Cleaning
This script:
1. Loads the House Prices dataset from Kaggle
2. Analyzes the data structure
3. Cleans data (removes nulls, handles missing values)
4. Saves cleaned data for Group B to use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory if it doesn't exist
os.makedirs('02_Data/processed', exist_ok=True)

print("=" * 80)
print("GROUP A: DATA COLLECTION, EXPLORATION & CLEANING")
print("=" * 80)

# Load training data from raw folder
train_data = pd.read_csv('02_Data/raw/train.csv')
test_data = pd.read_csv('02_Data/raw/test.csv')

print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Training set shape: {train_data.shape}")
print(f"Test set shape: {test_data.shape}")
print(f"Total features: {train_data.shape[1] - 1} (excluding SalePrice)")

print("\n2. FIRST FEW ROWS")
print("-" * 80)
print(train_data.head())

print("\n3. DATA TYPES")
print("-" * 80)
print(train_data.dtypes)

print("\n4. MISSING VALUES ANALYSIS")
print("-" * 80)
missing = train_data.isnull().sum()
missing_percent = (missing / len(train_data)) * 100
missing_df = pd.DataFrame({
    'Feature': missing.index,
    'Missing_Count': missing.values,
    'Percentage': missing_percent.values
})
missing_df = missing_df[missing_df['Missing_Count'] >
                        0].sort_values('Missing_Count', ascending=False)
print(missing_df)

print("\n5. TARGET VARIABLE STATISTICS (SalePrice)")
print("-" * 80)
print(train_data['SalePrice'].describe())

print("\n6. NUMERICAL FEATURES STATISTICS")
print("-" * 80)
numerical_features = train_data.select_dtypes(
    include=[np.number]).columns.tolist()
print(f"Number of numerical features: {len(numerical_features)}")
print("\nNumerical features:")
for i, feature in enumerate(numerical_features[:10], 1):
    print(f"  {i}. {feature}")
print(f"  ... and {len(numerical_features) - 10} more")

print("\n7. CATEGORICAL FEATURES")
print("-" * 80)
categorical_features = train_data.select_dtypes(
    include=['object']).columns.tolist()
print(f"Number of categorical features: {len(categorical_features)}")
print("\nCategorical features:")
for i, feature in enumerate(categorical_features, 1):
    unique_count = train_data[feature].nunique()
    print(f"  {i}. {feature}: {unique_count} unique values")

print("\n8. CORRELATION WITH SALEPRICE (Top 15)")
print("-" * 80)
# Only correlate numerical columns
correlation = train_data.select_dtypes(include=[np.number]).corr()[
    'SalePrice'].sort_values(ascending=False)
print(correlation.head(16))

print("\n9. PROBLEM DEFINITION")
print("-" * 80)
print("""
OBJECTIVE: Predict house sale prices based on 79 features

PROBLEM TYPE: Regression (predicting continuous numerical values)

INPUT FEATURES (79 total):
  - Structural: LotArea, YearBuilt, YearRemodAdd, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea
  - Location: Neighborhood, Street, LotShape, LandContour
  - Condition: OverallQual, OverallCond, ExterCond, BsmtCond, RoofCond
  - Amenities: GarageCars, GarageArea, PoolArea, Fireplaces, Porch areas
  - Utilities: CentralAir, Heating, Electrical, Plumbing
  - And many more...

OUTPUT VARIABLE: SalePrice (in dollars)
  - Range: ${:,.0f} to ${:,.0f}
  - Mean: ${:,.0f}
  - Median: ${:,.0f}
""".format(
    train_data['SalePrice'].min(),
    train_data['SalePrice'].max(),
    train_data['SalePrice'].mean(),
    train_data['SalePrice'].median()
))

print("\n10. KEY INSIGHTS")
print("-" * 80)
print(f"""
✓ Dataset contains {len(train_data)} houses with {len(train_data.columns)} features
✓ Target variable (SalePrice) ranges from ${train_data['SalePrice'].min():,.0f} to ${train_data['SalePrice'].max():,.0f}
✓ {len(numerical_features)} numerical features and {len(categorical_features)} categorical features
✓ Some features have missing values that need to be handled
✓ Strong correlations exist between certain features and price
✓ This is a supervised learning regression problem
✓ Data is from Ames, Iowa housing market (2006-2010)
""")

print("\n" + "=" * 80)
print("GROUP A ANALYSIS COMPLETE")
print("=" * 80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SalePrice Distribution
axes[0, 0].hist(train_data['SalePrice'], bins=50,
                edgecolor='black', color='skyblue')
axes[0, 0].set_title('Distribution of House Prices',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Sale Price ($)')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Top 10 Correlations
top_corr = correlation.head(11)[1:]  # Exclude SalePrice itself
top_corr.plot(kind='barh', ax=axes[0, 1], color='coral')
axes[0, 1].set_title(
    'Top 10 Features Correlated with SalePrice', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Correlation Coefficient')

# Plot 3: Missing Values
if len(missing_df) > 0:
    missing_df.head(10).plot(x='Feature', y='Percentage',
                             kind='barh', ax=axes[1, 0], color='lightcoral')
    axes[1, 0].set_title('Top 10 Features with Missing Values',
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Percentage Missing (%)')
else:
    axes[1, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
    axes[1, 0].set_title('Missing Values', fontsize=12, fontweight='bold')

# Plot 4: Feature Type Distribution
feature_types = pd.Series({
    'Numerical': len(numerical_features),
    'Categorical': len(categorical_features)
})
axes[1, 1].pie(feature_types, labels=feature_types.index,
               autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
axes[1, 1].set_title('Feature Type Distribution',
                     fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('04_Outputs/visualizations/Group_A_Data_Analysis.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as '04_Outputs/visualizations/Group_A_Data_Analysis.png'")

print("\n" + "=" * 80)
print("DATA CLEANING & PREPARATION")
print("=" * 80)

# Clean training data
print("\n6. CLEANING TRAINING DATA")
print("-" * 80)

# Make a copy for cleaning
train_cleaned = train_data.copy()

# Handle missing values
print("Handling missing values...")
for col in train_cleaned.columns:
    if train_cleaned[col].isnull().sum() > 0:
        if train_cleaned[col].dtype in ['float64', 'int64']:
            # Fill numerical columns with median
            train_cleaned[col].fillna(
                train_cleaned[col].median(), inplace=True)
        else:
            # Fill categorical columns with mode
            train_cleaned[col].fillna(
                train_cleaned[col].mode()[0], inplace=True)

print(f"✓ Missing values handled")
print(f"  Remaining nulls: {train_cleaned.isnull().sum().sum()}")

# Remove duplicates
initial_rows = len(train_cleaned)
train_cleaned = train_cleaned.drop_duplicates()
print(
    f"✓ Duplicates removed: {initial_rows - len(train_cleaned)} rows removed")

# Remove rows with invalid SalePrice
train_cleaned = train_cleaned[train_cleaned['SalePrice'] > 0]
print(f"✓ Invalid prices removed")

print(f"\nCleaned training data shape: {train_cleaned.shape}")

# Clean test data
print("\n7. CLEANING TEST DATA")
print("-" * 80)

test_cleaned = test_data.copy()

# Handle missing values in test data
for col in test_cleaned.columns:
    if test_cleaned[col].isnull().sum() > 0:
        if test_cleaned[col].dtype in ['float64', 'int64']:
            test_cleaned[col].fillna(test_cleaned[col].median(), inplace=True)
        else:
            test_cleaned[col].fillna(test_cleaned[col].mode()[0], inplace=True)

print(f"✓ Missing values handled")
print(f"  Remaining nulls: {test_cleaned.isnull().sum().sum()}")

# Remove duplicates
initial_rows = len(test_cleaned)
test_cleaned = test_cleaned.drop_duplicates()
print(f"✓ Duplicates removed: {initial_rows - len(test_cleaned)} rows removed")

print(f"\nCleaned test data shape: {test_cleaned.shape}")

# Save cleaned data for Group B
print("\n8. SAVING CLEANED DATA")
print("-" * 80)

train_cleaned.to_csv('02_Data/processed/train_cleaned.csv', index=False)
test_cleaned.to_csv('02_Data/processed/test_cleaned.csv', index=False)

print(f"✓ Cleaned training data saved: 02_Data/processed/train_cleaned.csv")
print(f"✓ Cleaned test data saved: 02_Data/processed/test_cleaned.csv")
print(f"\n✓ Data is ready for Group B (Data Preparation)")
