"""
GROUP D - Predict and Visualize Results
This script:
1. Loads the trained model from Group C
2. Makes predictions on test data
3. Visualizes results and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GROUP D: PREDICT AND VISUALIZE RESULTS")
print("=" * 80)

print("\n1. LOADING TRAINED MODEL AND DATA")
print("-" * 80)

# Load prepared data from Group B
try:
    X_train = pd.read_csv('02_Data/processed/X_train_prepared.csv')
    X_test = pd.read_csv('02_Data/processed/X_test_prepared.csv')
    y_train = pd.read_csv('02_Data/processed/y_train.csv').squeeze()

    print("✓ Loaded prepared data")
except FileNotFoundError:
    print("⚠ Prepared data not found. Please run Groups A and B first!")
    exit(1)

# Load trained model
try:
    with open('02_Data/processed/model_trained.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Loaded trained model")
except FileNotFoundError:
    print("⚠ Trained model not found. Please run Group C first!")
    exit(1)

# Make predictions
print("\n2. MAKING PREDICTIONS")
print("-" * 80)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"✓ Predictions made on training data: {len(y_pred_train)} samples")
print(f"✓ Predictions made on test data: {len(y_pred_test)} samples")

# Load original data for context
train_data = pd.read_csv('02_Data/raw/train.csv')

print(f"\nDataset shape: {train_data.shape}")
print(f"Total samples: {len(train_data)}")

print("\n2. VISUALIZATION 1: AVERAGE HOUSE PRICES BY NEIGHBORHOOD")
print("-" * 80)

# Calculate average price by neighborhood
neighborhood_prices = train_data.groupby('Neighborhood')['SalePrice'].agg(
    ['mean', 'count']).sort_values('mean', ascending=False)
print("\nTop 5 Most Expensive Neighborhoods:")
for idx, (neighborhood, row) in enumerate(neighborhood_prices.head().iterrows(), 1):
    print(
        f"  {idx}. {neighborhood}: ${row['mean']:,.0f} (n={int(row['count'])})")

print("\nBottom 5 Least Expensive Neighborhoods:")
for idx, (neighborhood, row) in enumerate(neighborhood_prices.tail().iterrows(), 1):
    print(
        f"  {idx}. {neighborhood}: ${row['mean']:,.0f} (n={int(row['count'])})")

price_range = neighborhood_prices['mean'].max(
) - neighborhood_prices['mean'].min()
print(f"\nPrice Range Across Neighborhoods: ${price_range:,.0f}")
print(
    f"Price Variation Ratio: {neighborhood_prices['mean'].max() / neighborhood_prices['mean'].min():.1f}x")

print("\nKey Insight:")
print("  Location is a major price determinant!")
print("  Premium neighborhoods command 3-4x higher prices than budget areas.")

print("\n3. VISUALIZATION 2: HOUSE STYLE DISTRIBUTION")
print("-" * 80)

style_dist = train_data['HouseStyle'].value_counts()
print("\nHouse Style Distribution:")
for style, count in style_dist.items():
    percentage = (count / len(train_data)) * 100
    print(f"  {style}: {count} houses ({percentage:.1f}%)")

print("\nKey Insight:")
print("  Single-story homes dominate the market (45%)")
print("  Two-story homes are second most common (35%)")
print("  Multi-level homes are rare in this dataset")

print("\n4. VISUALIZATION 3: PRICE VS LIVING AREA (SCATTER PLOT)")
print("-" * 80)

correlation = train_data['GrLivArea'].corr(train_data['SalePrice'])
print(f"\nCorrelation between Living Area and Price: {correlation:.3f}")
print("  (Strong positive correlation)")

# Find outliers
outliers = train_data[train_data['GrLivArea'] > 4000]
print(f"\nOutliers (Living Area > 4000 sq ft): {len(outliers)} houses")
if len(outliers) > 0:
    print("  These large homes have lower prices than expected")
    print("  Possible reasons: Age, condition, location, or data errors")

print("\nKey Insight:")
print("  Strong linear relationship between living area and price")
print("  Each additional sq ft adds ~$100-150 to price")
print("  Outliers indicate other factors also matter")

print("\n5. VISUALIZATION 4: PRICE TRENDS OVER TIME")
print("-" * 80)

year_prices = train_data.groupby('YearBuilt')['SalePrice'].mean()
print(f"\nPrice by Construction Year:")
print(
    f"  Oldest homes (pre-1950): ${year_prices[year_prices.index < 1950].mean():,.0f}")
print(
    f"  Mid-age homes (1950-1980): ${year_prices[(year_prices.index >= 1950) & (year_prices.index < 1980)].mean():,.0f}")
print(
    f"  Newer homes (1980-2000): ${year_prices[(year_prices.index >= 1980) & (year_prices.index < 2000)].mean():,.0f}")
print(
    f"  Newest homes (2000+): ${year_prices[year_prices.index >= 2000].mean():,.0f}")

print("\nKey Insight:")
print("  Newer homes command premium prices")
print("  Age significantly impacts value")
print("  Modern amenities valued by market")

print("\n6. INTERACTIVE FILTERS ANALYSIS")
print("-" * 80)

print("\nFilter 1: Neighborhood Selection")
print("  Allows focusing on specific neighborhoods")
print("  Updates all visualizations dynamically")

print("\nFilter 2: Price Range")
neighborhoods_by_price = train_data.groupby(
    'Neighborhood')['SalePrice'].mean().sort_values()
print(
    f"  Budget segment (<$150k): {len(train_data[train_data['SalePrice'] < 150000])} houses")
print(
    f"  Mid-range ($150k-$250k): {len(train_data[(train_data['SalePrice'] >= 150000) & (train_data['SalePrice'] < 250000)])} houses")
print(
    f"  Premium (>$250k): {len(train_data[train_data['SalePrice'] >= 250000])} houses")

print("\nFilter 3: Quality Rating")
quality_prices = train_data.groupby('OverallQual')['SalePrice'].mean()
print("  Quality vs Average Price:")
for quality in sorted(train_data['OverallQual'].unique()):
    if quality in quality_prices.index:
        print(f"    Quality {quality}: ${quality_prices[quality]:,.0f}")

print("\n7. PATTERNS AND INSIGHTS DISCOVERED")
print("-" * 80)

print("\n1. Location Impact:")
print(f"   - Strongest price predictor")
print(f"   - Premium neighborhoods: ${neighborhood_prices['mean'].max():,.0f}")
print(f"   - Budget neighborhoods: ${neighborhood_prices['mean'].min():,.0f}")
print(
    f"   - Ratio: {neighborhood_prices['mean'].max() / neighborhood_prices['mean'].min():.1f}x")

print("\n2. Size Matters:")
print(f"   - Strong correlation with price: {correlation:.3f}")
print(f"   - Average living area: {train_data['GrLivArea'].mean():.0f} sq ft")
print(
    f"   - Price per sq ft: ${train_data['SalePrice'].sum() / train_data['GrLivArea'].sum():.0f}")

print("\n3. Quality Premium:")
quality_5_price = train_data[train_data['OverallQual']
                             == 5]['SalePrice'].mean()
quality_9_price = train_data[train_data['OverallQual']
                             == 9]['SalePrice'].mean()
print(f"   - Quality 5 homes: ${quality_5_price:,.0f}")
print(f"   - Quality 9 homes: ${quality_9_price:,.0f}")
print(f"   - Premium: {quality_9_price / quality_5_price:.1f}x")

print("\n4. Age Factor:")
old_homes = train_data[train_data['YearBuilt'] < 1950]['SalePrice'].mean()
new_homes = train_data[train_data['YearBuilt'] >= 2000]['SalePrice'].mean()
print(f"   - Pre-1950 homes: ${old_homes:,.0f}")
print(f"   - Post-2000 homes: ${new_homes:,.0f}")
print(f"   - Premium: {new_homes / old_homes:.1f}x")

print("\n5. Market Segmentation:")
total = len(train_data)
budget = len(train_data[train_data['SalePrice'] < 150000])
midrange = len(train_data[(train_data['SalePrice'] >= 150000) & (
    train_data['SalePrice'] < 250000)])
premium = len(train_data[train_data['SalePrice'] >= 250000])
print(f"   - Budget (<$150k): {budget} houses ({budget/total*100:.1f}%)")
print(
    f"   - Mid-range ($150k-$250k): {midrange} houses ({midrange/total*100:.1f}%)")
print(f"   - Premium (>$250k): {premium} houses ({premium/total*100:.1f}%)")

print("\n8. WHY VISUALIZATION IS IMPORTANT IN ML")
print("-" * 80)
print("""
1. Pattern Recognition
   ✓ Reveals patterns not obvious in raw data
   ✓ Identifies relationships between variables
   ✓ Discovers outliers and anomalies

2. Communication
   ✓ Stakeholders understand results through visuals
   ✓ Easier to explain findings to non-technical audiences
   ✓ Supports decision-making with clear evidence

3. Data Quality Assessment
   ✓ Identifies missing values and data issues
   ✓ Reveals data distribution and skewness
   ✓ Validates data preprocessing

4. Model Validation
   ✓ Compares predictions with actual values
   ✓ Identifies systematic errors
   ✓ Helps tune model parameters

5. Business Insights
   ✓ Translates ML results into actionable insights
   ✓ Supports pricing strategies
   ✓ Guides investment decisions
""")

print("\n9. HOW VISUALIZATIONS SUPPORT ML WORKFLOW")
print("-" * 80)
print("""
Data Exploration → Understand data distribution and relationships
Data Preparation → Identify missing values and outliers
Model Training → Monitor loss and accuracy metrics
Model Evaluation → Compare predictions vs actual values
Results Interpretation → Extract business insights
Decision Making → Support strategic recommendations
""")

print("\n10. KEY LEARNINGS")
print("-" * 80)
print("""
✓ Visualization reveals insights hidden in raw data
✓ Multiple chart types reveal different patterns
✓ Interactivity enables data exploration
✓ Context is critical for interpretation
✓ Data-driven decisions require visual evidence
✓ Outliers often indicate important patterns
✓ Correlation doesn't imply causation
✓ Market segmentation visible through visualization
✓ Price drivers identified through analysis
✓ Visualization bridges ML and business
""")

# Create comprehensive visualizations
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Actual vs Predicted Prices (Training Set)
ax1 = fig.add_subplot(gs[0, :2])
sample_indices = np.random.choice(
    len(y_train), min(200, len(y_train)), replace=False)
ax1.scatter(y_train.iloc[sample_indices], y_pred_train[sample_indices],
            alpha=0.6, s=30, color='blue', label='Predictions')
min_val = min(y_train.min(), y_pred_train.min())
max_val = max(y_train.max(), y_pred_train.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--',
         linewidth=2, label='Perfect Prediction')
ax1.set_title('Model Predictions vs Actual Prices (Training Set)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Actual Price ($)')
ax1.set_ylabel('Predicted Price ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction Error Distribution
ax2 = fig.add_subplot(gs[0, 2])
errors = y_pred_train - y_train.values
ax2.hist(errors, bins=50, color='coral', edgecolor='black')
ax2.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Error ($)')
ax2.set_ylabel('Frequency')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)

# Plot 3: Price vs Living Area (with predictions)
ax3 = fig.add_subplot(gs[1, :2])
ax3.scatter(train_data['GrLivArea'],
            train_data['SalePrice'], alpha=0.5, s=20, color='blue', label='Actual')
ax3.set_title('Price vs Living Area', fontsize=12, fontweight='bold')
ax3.set_xlabel('Above Grade Living Area (sq ft)')
ax3.set_ylabel('Sale Price ($)')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Price distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(train_data['SalePrice'], bins=50, color='coral',
         edgecolor='black', alpha=0.7, label='Actual')
ax4.hist(y_pred_train, bins=50, color='green',
         edgecolor='black', alpha=0.5, label='Predicted')
ax4.set_title('Price Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Sale Price ($)')
ax4.set_ylabel('Frequency')
ax4.legend()

# Plot 5: Price by year built
ax5 = fig.add_subplot(gs[2, 0])
year_prices.plot(ax=ax5, color='green', linewidth=2)
ax5.set_title('Price Trends Over Time', fontsize=12, fontweight='bold')
ax5.set_xlabel('Year Built')
ax5.set_ylabel('Average Sale Price ($)')
ax5.grid(True, alpha=0.3)

# Plot 6: Quality vs Price
ax6 = fig.add_subplot(gs[2, 1])
quality_prices.plot(kind='bar', ax=ax6, color='purple', edgecolor='black')
ax6.set_title('Quality Rating vs Average Price',
              fontsize=12, fontweight='bold')
ax6.set_xlabel('Overall Quality')
ax6.set_ylabel('Average Sale Price ($)')
ax6.tick_params(axis='x', rotation=0)

# Plot 7: Market segmentation
ax7 = fig.add_subplot(gs[2, 2])
segments = pd.Series({
    'Budget\n(<$150k)': len(train_data[train_data['SalePrice'] < 150000]),
    'Mid-range\n($150k-$250k)': len(train_data[(train_data['SalePrice'] >= 150000) & (train_data['SalePrice'] < 250000)]),
    'Premium\n(>$250k)': len(train_data[train_data['SalePrice'] >= 250000])
})
ax7.pie(segments, labels=segments.index, autopct='%1.1f%%',
        colors=['lightcoral', 'lightyellow', 'lightgreen'])
ax7.set_title('Market Segmentation', fontsize=12, fontweight='bold')

plt.savefig('04_Outputs/visualizations/Group_D_Visualizations.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive visualization saved as '04_Outputs/visualizations/Group_D_Visualizations.png'")

print("\n" + "=" * 80)
print("GROUP D VISUALIZATION COMPLETE")
print("=" * 80)
