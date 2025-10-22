"""
GROUP A - Data Collection and Exploration
This script analyzes the House Prices dataset to understand its structure,
features, and the prediction problem.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("=" * 80)
print("GROUP A: DEFINE OBJECTIVE AND COLLECT DATA")
print("=" * 80)

# Load training data from new structure
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

print("\n4. MISSING VALUES")
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
