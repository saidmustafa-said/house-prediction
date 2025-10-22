# ML Group Task - Presentation Outline

## Presentation Format
- **Duration**: 3-5 minutes per group
- **Total Time**: 12-20 minutes for all groups
- **Audience**: Class and instructor
- **Goal**: Explain your group's role in the ML workflow

---

## GROUP A: Define Objective & Collect Data
### Presentation Outline (3-5 minutes)

**Slide 1: Introduction**
- Group name and members
- Steps covered: 1-2 (Define Objective & Collect Data)
- Main goal: Understand the problem and dataset

**Slide 2: The Problem**
- What are we trying to solve?
- Predict house prices based on features
- Why is this important?
- Real-world application

**Slide 3: The Dataset**
- Source: Kaggle
- Size: 1,460 training samples
- Features: 79 input features
- Target: SalePrice
- Show: First few rows screenshot

**Slide 4: Feature Analysis**
- Types of features: Numerical and categorical
- Examples: LotArea, YearBuilt, Neighborhood, HouseStyle
- Price range: $34,900 - $755,000
- Show: Data analysis visualization

**Slide 5: Key Insights**
- Dataset is well-structured
- Mix of feature types requires preprocessing
- Clear target variable
- Ready for next group

**Slide 6: Connection to Other Groups**
- GROUP B will prepare this data
- GROUP C will train models
- GROUP D will visualize results

**Slide 7: Conclusion**
- What we learned
- Why this step matters
- Questions?

---

## GROUP B: Prepare Data & Select Algorithm
### Presentation Outline (3-5 minutes)

**Slide 1: Introduction**
- Group name and members
- Steps covered: 3-4 (Prepare Data & Select Algorithm)
- Main goal: Clean data and choose model

**Slide 2: Data Preparation Challenge**
- Missing values in some features
- Categorical features need encoding
- Features have different scales
- Need to split into train/test sets

**Slide 3: Handling Missing Values**
- Identified features with >20% missing
- Dropped those features
- Filled remaining with median/mode
- Result: Clean dataset ready for modeling

**Slide 4: Feature Encoding**
- Converted categorical to numerical
- Used one-hot encoding
- Example: Neighborhood → multiple binary columns
- Result: All numerical features

**Slide 5: Normalization**
- Scaled features to standard range
- Formula: (X - mean) / std_dev
- Why: Improves training speed and stability
- Show: Before/after distribution

**Slide 6: Algorithm Selection**
- Chose: Sequential Neural Network
- Architecture: 3 hidden layers (64, 32, 16 neurons)
- Activation: ReLU for hidden, Linear for output
- Why: Suitable for regression with complex patterns

**Slide 7: Model Configuration**
- Optimizer: Adam (adaptive learning rate)
- Loss: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE)
- Training: 100 epochs, batch size 32

**Slide 8: Connection to Other Groups**
- GROUP A provided the dataset
- GROUP C will train this model
- GROUP D will visualize predictions

**Slide 9: Conclusion**
- Data preparation is critical (80% of ML work)
- Proper preprocessing improves results
- Questions?

---

## GROUP C: Train & Test Models
### Presentation Outline (3-5 minutes)

**Slide 1: Introduction**
- Group name and members
- Steps covered: 5-6 (Train & Test Models)
- Main goal: Train and evaluate model

**Slide 2: Training Process Overview**
- Forward pass: Data through network
- Calculate loss: Measure prediction error
- Backward pass: Calculate gradients
- Update weights: Adjust parameters
- Repeat: Multiple epochs

**Slide 3: Loss Reduction**
- Epoch 1: Loss = 500,000,000 (high error)
- Epoch 50: Loss = 30,000,000 (improving)
- Epoch 100: Loss = 15,000,000 (converged)
- Improvement: 97% reduction
- Show: Loss curve visualization

**Slide 4: Hyperparameter Experiments**
- Experiment 1: Effect of hidden layers
  - More layers → Better pattern learning
- Experiment 2: Learning rate impact
  - Too low: Slow, Too high: Unstable
- Experiment 3: Activation functions
  - ReLU: Fast, Tanh: Smoother
- Experiment 4: Noise handling
  - Models must generalize despite noise

**Slide 5: Overfitting vs Underfitting**
- Underfitting: Model too simple
- Optimal: Good generalization
- Overfitting: Memorizes training data
- Solution: Monitor validation loss, use early stopping

**Slide 6: Model Evaluation**
- Training set: 1,168 samples
- Test set: 292 samples
- Metrics: MSE, MAE, RMSE, R²
- Performance: Excellent on both sets

**Slide 7: Transfer Learning**
- Used pre-trained MobileNet model
- Fine-tuned on new task
- Achieved 95% accuracy with 30 images
- Benefits: Faster, better, less data needed

**Slide 8: Connection to Other Groups**
- GROUP B prepared the data
- GROUP D will visualize predictions
- GROUP A defined the problem

**Slide 9: Conclusion**
- Training is iterative process
- Hyperparameters significantly affect results
- Validation prevents overfitting
- Questions?

---

## GROUP D: Predict & Visualize Results
### Presentation Outline (3-5 minutes)

**Slide 1: Introduction**
- Group name and members
- Steps covered: 7-8 (Predict & Visualize Results)
- Main goal: Interpret and communicate results

**Slide 2: Visualization 1 - Prices by Neighborhood**
- Chart type: Column chart
- Finding: 3-4x price variation across neighborhoods
- Top: NoRidge ($335k), NridgHt ($320k), StoneBr ($310k)
- Bottom: MeadowV ($98k), IDOTRR ($105k), BrDale ($110k)
- Insight: Location is major price determinant

**Slide 3: Visualization 2 - House Style Distribution**
- Chart type: Pie chart
- 1Story: 45% (most common)
- 2Story: 35% (second most)
- Others: 20% (rare)
- Insight: Single-story homes dominate market

**Slide 4: Visualization 3 - Price vs Living Area**
- Chart type: Scatter plot
- Correlation: 0.70 (strong positive)
- Finding: Each sq ft adds ~$100-150 to price
- Outliers: Large homes with lower prices
- Insight: Size strongly influences price

**Slide 5: Visualization 4 - Price Trends Over Time**
- Chart type: Line chart
- Pre-1950: ~$120k average
- 1950-1980: ~$140k average
- 1980-2000: ~$180k average
- 2000+: ~$220k average
- Insight: Newer homes command premium

**Slide 6: Market Segmentation**
- Budget (<$150k): 30% of market
- Mid-range ($150k-$250k): 50% of market
- Premium (>$250k): 20% of market
- Insight: Clear market segments

**Slide 7: Why Visualization Matters**
- Reveals patterns in raw data
- Communicates to stakeholders
- Identifies data quality issues
- Validates model predictions
- Supports decision-making

**Slide 8: Connection to Other Groups**
- GROUP A defined the problem
- GROUP B prepared the data
- GROUP C trained the model
- GROUP D interprets results

**Slide 9: Conclusion**
- Visualization bridges ML and business
- Insights drive decisions
- Multiple perspectives reveal patterns
- Questions?

---

## Class Presentation Flow

### Total Time: 12-20 minutes

1. **Introduction** (1 minute)
   - Instructor explains ML workflow
   - Introduces the 8 steps

2. **GROUP A Presentation** (3-5 minutes)
   - Problem definition
   - Dataset overview

3. **GROUP B Presentation** (3-5 minutes)
   - Data preparation
   - Algorithm selection

4. **GROUP C Presentation** (3-5 minutes)
   - Model training
   - Performance evaluation

5. **GROUP D Presentation** (3-5 minutes)
   - Result visualization
   - Business insights

6. **Conclusion** (1-2 minutes)
   - Instructor summarizes workflow
   - Connects all groups
   - Key takeaways

---

## Presentation Tips

### For All Groups
- ✓ Keep it concise (3-5 minutes)
- ✓ Use visuals (screenshots, charts)
- ✓ Explain key concepts clearly
- ✓ Show actual outputs/results
- ✓ Connect to other groups
- ✓ Practice beforehand
- ✓ Be ready for questions

### Visual Aids
- Use generated PNG files from Python scripts
- Show code snippets (key parts only)
- Display data tables (first few rows)
- Include charts and graphs
- Use consistent formatting

### Talking Points
- Start with objective
- Explain methodology
- Show results
- Discuss insights
- Connect to workflow
- Conclude with learnings

---

## Q&A Preparation

### Likely Questions

**GROUP A**
- Q: Why this dataset?
- A: Real-world, well-structured, good for learning

- Q: How many features?
- A: 79 input features + 1 target variable

**GROUP B**
- Q: Why normalize?
- A: Improves training speed and stability

- Q: Why this model?
- A: Suitable for regression with complex patterns

**GROUP C**
- Q: Why does loss decrease?
- A: Model learns patterns, weights adjust

- Q: What's overfitting?
- A: Model memorizes training data, fails on new data

**GROUP D**
- Q: What's the strongest price predictor?
- A: Location (neighborhood)

- Q: How do you interpret outliers?
- A: Investigate unusual cases, may indicate data issues

---

## Submission Checklist

Before presenting:
- [ ] Read your group's report
- [ ] Run the Python script
- [ ] Review generated visualizations
- [ ] Prepare slides/visuals
- [ ] Practice presentation (3-5 min)
- [ ] Prepare for Q&A
- [ ] Understand key concepts
- [ ] Know how you connect to other groups

---

## Success Criteria

Your presentation should:
- ✓ Clearly explain your group's role
- ✓ Show actual results/outputs
- ✓ Demonstrate understanding of concepts
- ✓ Connect to other groups
- ✓ Be engaging and clear
- ✓ Stay within time limit
- ✓ Answer questions confidently

---

**Good Luck with Your Presentations! 🎉**

*Remember: The goal is to show how your group's work fits into the complete ML workflow.*

