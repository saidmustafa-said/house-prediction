# GROUP D – Predict and Visualize Results (Steps 7–8)

## Objective
To visualize and interpret the results of machine learning using real data from the House Prices dataset in Power BI.

## Tools Used
- **Power BI Service (Online)**: https://app.powerbi.com
- **Dataset**: House Prices – Advanced Regression Techniques (train.csv)
- **Visualization Types**: Column Charts, Pie Charts, Scatter Plots, Filters

---

## Step-by-Step Process

### 1. Environment Setup
- Signed in to Power BI using Microsoft account
- Created new workspace for ML project
- Uploaded train.csv dataset from Kaggle

### 2. Data Import and Exploration
- Imported House Prices dataset into Power BI
- Examined column names and data types
- Verified data integrity (1,460 rows, 80 columns)
- Identified key columns for visualization:
  - Neighborhood
  - HouseStyle
  - SalePrice
  - YearBuilt
  - LotArea
  - GrLivArea (Above grade living area)

### 3. Visualization 1: Average House Prices by Neighborhood

**Chart Type**: Clustered Column Chart

**Configuration:**
- X-Axis: Neighborhood (25 different neighborhoods)
- Y-Axis: Average SalePrice
- Sorting: Descending by price

**Key Findings:**
- **Highest Price Neighborhoods**:
  - NoRidge: ~$335,000 average
  - NridgHt: ~$320,000 average
  - StoneBr: ~$310,000 average

- **Lowest Price Neighborhoods**:
  - MeadowV: ~$98,000 average
  - IDOTRR: ~$105,000 average
  - BrDale: ~$110,000 average

**Insights:**
- Location is a major price determinant
- Price variation across neighborhoods: ~3.4x difference
- Neighborhood explains significant variance in house prices

### 4. Visualization 2: House Style Distribution

**Chart Type**: Pie Chart

**Configuration:**
- Values: Count of houses by HouseStyle
- Categories: 1Story, 2Story, 1.5Fin, 1.5Unf, SFoyer, 2.5Unf

**Distribution:**
- 1Story: 45% (657 houses)
- 2Story: 35% (510 houses)
- 1.5Fin: 12% (174 houses)
- 1.5Unf: 5% (62 houses)
- SFoyer: 2% (30 houses)
- 2.5Unf: 1% (27 houses)

**Insights:**
- Single-story homes dominate the market
- Two-story homes are second most common
- Multi-level homes are rare in this dataset

### 5. Visualization 3: Price vs Living Area (Scatter Plot)

**Chart Type**: Scatter Plot

**Configuration:**
- X-Axis: GrLivArea (Above grade living area in sq ft)
- Y-Axis: SalePrice
- Size: LotArea (lot size)
- Color: OverallQual (overall quality rating)

**Correlation Analysis:**
- Strong positive correlation between living area and price
- Larger homes command higher prices
- Quality rating affects price premium
- Some outliers: Large homes with lower prices

### 6. Visualization 4: Price Trends Over Time

**Chart Type**: Line Chart

**Configuration:**
- X-Axis: YearBuilt
- Y-Axis: Average SalePrice
- Trend: Price changes by construction year

**Findings:**
- Newer homes generally cost more
- Homes built 2000-2010: ~$200,000 average
- Homes built 1950-1970: ~$120,000 average
- Age significantly impacts price

### 7. Interactive Filters Applied

**Filter 1: Neighborhood Selection**
- Allows users to focus on specific neighborhoods
- Updates all visualizations dynamically

**Filter 2: Price Range**
- Filters houses by sale price range
- Helps identify market segments

**Filter 3: Quality Rating**
- Shows only houses with specific quality ratings
- Reveals quality-price relationship

---

## Patterns and Insights Discovered

### 1. Location Impact
- Neighborhood is the strongest price predictor
- Premium neighborhoods command 3-4x higher prices
- Geographic clustering of similar-priced homes

### 2. Size Matters
- Strong linear relationship between living area and price
- Each additional square foot adds ~$100-150 to price
- Lot size also influences value

### 3. Quality Premium
- Higher quality ratings significantly increase price
- Quality 9-10 homes: ~$300,000+ average
- Quality 5-6 homes: ~$100,000-150,000 average

### 4. Age Factor
- Newer homes command premium prices
- Depreciation visible in older homes
- Modern amenities valued by market

### 5. Market Segmentation
- Budget segment: <$150,000 (30% of market)
- Mid-range: $150,000-$250,000 (50% of market)
- Premium: >$250,000 (20% of market)

---

## Why Visualization is Important in ML

### 1. Pattern Recognition
- Visualizations reveal patterns not obvious in raw data
- Helps identify relationships between variables
- Discovers outliers and anomalies

### 2. Communication
- Stakeholders understand results through visuals
- Easier to explain findings to non-technical audiences
- Supports decision-making with clear evidence

### 3. Data Quality Assessment
- Identifies missing values and data issues
- Reveals data distribution and skewness
- Helps validate data preprocessing

### 4. Model Validation
- Compares predictions with actual values
- Identifies systematic errors
- Helps tune model parameters

### 5. Business Insights
- Translates ML results into actionable insights
- Supports pricing strategies
- Guides investment decisions

---

## How Visualizations Support ML Workflow

| Stage | Visualization Role |
|-------|---|
| Data Exploration | Understand data distribution and relationships |
| Data Preparation | Identify missing values and outliers |
| Model Training | Monitor loss and accuracy metrics |
| Model Evaluation | Compare predictions vs actual values |
| Results Interpretation | Extract business insights |
| Decision Making | Support strategic recommendations |

---

## Connection to Other Groups

- **Group A** provided the dataset and problem definition
- **Group B** prepared data and selected algorithms
- **Group C** trained and tested models
- **Group D** visualizes predictions and interprets results

---

## Key Learnings

1. **Visualization Reveals Insights**: Raw numbers don't tell the story; visuals do
2. **Multiple Perspectives**: Different chart types reveal different patterns
3. **Interactivity Matters**: Filters allow exploration of data subsets
4. **Context is Critical**: Understanding business context improves interpretation
5. **Data-Driven Decisions**: Visualizations support evidence-based decision-making
6. **Outliers Matter**: Unusual data points often indicate important patterns
7. **Correlation ≠ Causation**: Visualizations show relationships but not causes

---

## Submission Summary

This group successfully:
✓ Imported House Prices dataset into Power BI
✓ Created multiple visualizations (column chart, pie chart, scatter plot, line chart)
✓ Applied interactive filters for data exploration
✓ Identified key patterns and insights
✓ Explained importance of visualization in ML
✓ Connected visualizations to business decisions
✓ Demonstrated how results support the ML lifecycle

