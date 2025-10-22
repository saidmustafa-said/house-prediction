"""
GROUP B - Data Preparation and Algorithm Selection
This script demonstrates data cleaning, preprocessing, and model selection
for the House Prices prediction problem.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GROUP B: PREPARE DATA AND SELECT ALGORITHM")
print("=" * 80)

# Load data from new structure
train_data = pd.read_csv('02_Data/raw/train.csv')
test_data = pd.read_csv('02_Data/raw/test.csv')

print("\n1. INITIAL DATA INSPECTION")
print("-" * 80)
print(f"Training set shape: {train_data.shape}")
print(
    f"Missing values:\n{train_data.isnull().sum().sum()} total missing values")

# Separate features and target
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

print("\n2. HANDLING MISSING VALUES")
print("-" * 80)

# Strategy: Drop columns with >20% missing, fill others with median/mode
missing_percent = (X.isnull().sum() / len(X)) * 100
cols_to_drop = missing_percent[missing_percent > 20].index.tolist()
print(f"Dropping {len(cols_to_drop)} columns with >20% missing values:")
for col in cols_to_drop:
    print(f"  - {col}: {missing_percent[col]:.1f}% missing")

X = X.drop(cols_to_drop, axis=1)

# Fill remaining missing values
numerical_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(include=['object']).columns

X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
X[categorical_cols] = X[categorical_cols].fillna(
    X[categorical_cols].mode().iloc[0])

print(
    f"\nAfter handling missing values: {X.isnull().sum().sum()} missing values remain")

print("\n3. ENCODING CATEGORICAL VARIABLES")
print("-" * 80)
print(f"Categorical features to encode: {len(categorical_cols)}")

# One-hot encoding for categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"Features after encoding: {X_encoded.shape[1]}")
print(f"New shape: {X_encoded.shape}")

print("\n4. FEATURE SCALING (NORMALIZATION)")
print("-" * 80)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

print("Scaling applied using StandardScaler:")
print("  Formula: (X - mean) / std_dev")
print(f"\nSample statistics after scaling:")
print(f"  Mean: {X_scaled.mean().mean():.6f} (should be ~0)")
print(f"  Std Dev: {X_scaled.std().mean():.6f} (should be ~1)")

print("\n5. TRAIN-TEST SPLIT")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(
    f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(
    f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"Features: {X_train.shape[1]}")

print("\n6. ALGORITHM SELECTION: SEQUENTIAL NEURAL NETWORK")
print("-" * 80)
print("""
Model Architecture:
  Input Layer: {} features
    ↓
  Dense Layer 1: 64 neurons, ReLU activation
    ↓
  Dense Layer 2: 32 neurons, ReLU activation
    ↓
  Dense Layer 3: 16 neurons, ReLU activation
    ↓
  Output Layer: 1 neuron, Linear activation (for regression)

Why This Model?
  ✓ Regression task requires continuous output (Linear activation)
  ✓ Multiple layers capture complex non-linear relationships
  ✓ ReLU activation introduces non-linearity
  ✓ Can handle 79+ input features effectively
  ✓ Deep learning suitable for this problem complexity
""".format(X_train.shape[1]))

print("\n7. MODEL COMPILATION CONFIGURATION")
print("-" * 80)
print("""
Optimizer: Adam
  - Adaptive learning rate optimizer
  - Adjusts learning rate for each parameter
  - Combines advantages of AdaGrad and RMSprop

Loss Function: Mean Squared Error (MSE)
  - Formula: MSE = (1/n) × Σ(y_actual - y_predicted)²
  - Suitable for regression problems
  - Penalizes large errors more heavily

Metrics: Mean Absolute Error (MAE)
  - Formula: MAE = (1/n) × Σ|y_actual - y_predicted|
  - Easier to interpret than MSE
  - Represents average prediction error in dollars
""")

print("\n8. TRAINING CONFIGURATION")
print("-" * 80)
print("""
Epochs: 100
  - Number of times the model sees the entire dataset
  - More epochs = more learning (but risk of overfitting)

Batch Size: 32
  - Number of samples processed before weight update
  - Smaller batches = more frequent updates
  - Larger batches = faster training

Validation Split: 20%
  - 20% of training data used for validation
  - Helps detect overfitting during training

Early Stopping: Yes
  - Stops training if validation loss doesn't improve
  - Prevents overfitting and saves training time
""")

print("\n9. WHAT IS LOSS?")
print("-" * 80)
print("""
Loss is a numerical measure of how wrong the model's predictions are:

  High Loss → Predictions far from actual values
  Low Loss → Predictions close to actual values

Why Loss Decreases During Training:
  1. Model learns patterns in the data
  2. Weights are adjusted to reduce prediction errors
  3. Each epoch improves the model's understanding
  4. Gradient descent moves toward optimal solution

Example Loss Progression:
  Epoch 1:   Loss = 500,000,000 (very high error)
  Epoch 10:  Loss = 100,000,000 (improving)
  Epoch 50:  Loss = 30,000,000  (much better)
  Epoch 100: Loss = 15,000,000  (converged)

This represents ~97% improvement in prediction accuracy!
""")

print("\n10. DATA PREPARATION SUMMARY")
print("-" * 80)
print(f"""
✓ Loaded {len(train_data)} training samples
✓ Removed {len(cols_to_drop)} features with excessive missing values
✓ Handled missing values in remaining features
✓ Encoded {len(categorical_cols)} categorical features
✓ Applied StandardScaler normalization
✓ Split into {len(X_train)} training and {len(X_test)} test samples
✓ Selected Sequential Neural Network model
✓ Configured with Adam optimizer and MSE loss
✓ Ready for training!
""")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Feature distribution before scaling
axes[0, 0].hist(X_encoded.iloc[:, 0], bins=30,
                color='skyblue', edgecolor='black')
axes[0, 0].set_title('Feature Distribution (Before Scaling)',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Feature Value')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Feature distribution after scaling
axes[0, 1].hist(X_scaled.iloc[:, 0], bins=30,
                color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Feature Distribution (After Scaling)',
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Scaled Feature Value')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Target variable distribution
axes[1, 0].hist(y, bins=50, color='coral', edgecolor='black')
axes[1, 0].set_title('Target Variable Distribution (SalePrice)',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Sale Price ($)')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Train-Test Split
split_data = pd.Series({
    'Training Set': len(X_train),
    'Test Set': len(X_test)
})
axes[1, 1].pie(split_data, labels=split_data.index,
               autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
axes[1, 1].set_title('Train-Test Split', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('04_Outputs/visualizations/Group_B_Data_Preparation.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as '04_Outputs/visualizations/Group_B_Data_Preparation.png'")

print("\n" + "=" * 80)
print("GROUP B PREPARATION COMPLETE")
print("=" * 80)
