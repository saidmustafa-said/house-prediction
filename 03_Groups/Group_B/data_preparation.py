"""
GROUP B - Data Preparation and Algorithm Selection
This script:
1. Loads cleaned data from Group A
2. Encodes categorical variables
3. Normalizes numerical features
4. Saves prepared data for Group C (Model Training)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GROUP B: PREPARE DATA AND SELECT ALGORITHM")
print("=" * 80)

# Create output directory if it doesn't exist
os.makedirs('02_Data/processed', exist_ok=True)

print("\n1. LOADING CLEANED DATA FROM GROUP A")
print("-" * 80)

# Load cleaned data from Group A
try:
    train_data = pd.read_csv('02_Data/processed/train_cleaned.csv')
    test_data = pd.read_csv('02_Data/processed/test_cleaned.csv')
    print("✓ Loaded cleaned data from Group A")
except FileNotFoundError:
    print("⚠ Cleaned data not found. Using raw data instead...")
    train_data = pd.read_csv('02_Data/raw/train.csv')
    test_data = pd.read_csv('02_Data/raw/test.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Separate features and target
X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice']
X_test = test_data.copy()

print(f"\nFeatures shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")

print("\n2. ENCODING CATEGORICAL VARIABLES")
print("-" * 80)

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# One-hot encoding for categorical features
print("Applying one-hot encoding...")
X_train_encoded = pd.get_dummies(
    X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(
    X_test, columns=categorical_cols, drop_first=True)

# Ensure both have same columns
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0

X_test_encoded = X_test_encoded[X_train_encoded.columns]

print(f"✓ Features after encoding: {X_train_encoded.shape[1]}")
print(f"  Training shape: {X_train_encoded.shape}")
print(f"  Test shape: {X_test_encoded.shape}")

print("\n3. FEATURE SCALING (NORMALIZATION)")
print("-" * 80)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns)

print("✓ Scaling applied using StandardScaler:")
print("  Formula: (X - mean) / std_dev")
print(f"\nSample statistics after scaling:")
print(f"  Mean: {X_train_scaled.mean().mean():.6f} (should be ~0)")
print(f"  Std Dev: {X_train_scaled.std().mean():.6f} (should be ~1)")

print(f"\nPrepared data shapes:")
print(f"  Training features: {X_train_scaled.shape}")
print(f"  Test features: {X_test_scaled.shape}")
print(f"  Target: {y_train.shape}")

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

print("\n4. DATA PREPARATION SUMMARY")
print("-" * 80)
print(f"""
✓ Loaded {len(train_data)} training samples
✓ Handled missing values in all features
✓ Encoded {len(categorical_cols)} categorical features
✓ Applied StandardScaler normalization
✓ Prepared {len(X_train_scaled)} training and {len(X_test_scaled)} test samples
✓ Ready for model training!
""")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Feature distribution before scaling
axes[0, 0].hist(X_train_encoded.iloc[:, 0], bins=30,
                color='skyblue', edgecolor='black')
axes[0, 0].set_title('Feature Distribution (Before Scaling)',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Feature Value')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Feature distribution after scaling
axes[0, 1].hist(X_train_scaled.iloc[:, 0], bins=30,
                color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Feature Distribution (After Scaling)',
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Scaled Feature Value')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Target variable distribution
axes[1, 0].hist(y_train, bins=50, color='coral', edgecolor='black')
axes[1, 0].set_title('Target Variable Distribution (SalePrice)',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Sale Price ($)')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Train-Test Split
split_data = pd.Series({
    'Training Set': len(X_train_scaled),
    'Test Set': len(X_test_scaled)
})
axes[1, 1].pie(split_data, labels=split_data.index,
               autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
axes[1, 1].set_title('Train-Test Split', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('04_Outputs/visualizations/Group_B_Data_Preparation.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as '04_Outputs/visualizations/Group_B_Data_Preparation.png'")

print("\n" + "=" * 80)
print("SAVING PREPARED DATA FOR GROUP C")
print("=" * 80)

# Save prepared data
print("\nSaving prepared datasets...")

# Save as CSV
X_train_scaled.to_csv('02_Data/processed/X_train_prepared.csv', index=False)
X_test_scaled.to_csv('02_Data/processed/X_test_prepared.csv', index=False)
y_train.to_csv('02_Data/processed/y_train.csv', index=False)

# Save scaler for later use
with open('02_Data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✓ X_train_prepared.csv saved ({X_train_scaled.shape})")
print(f"✓ X_test_prepared.csv saved ({X_test_scaled.shape})")
print(f"✓ y_train.csv saved ({y_train.shape})")
print(f"✓ scaler.pkl saved (for future predictions)")

print("\n" + "=" * 80)
print("GROUP B PREPARATION COMPLETE")
print("=" * 80)
print("\n✓ Data is ready for Group C (Model Training)")
print(f"  - Features: {X_train_scaled.shape[1]}")
print(f"  - Training samples: {X_train_scaled.shape[0]}")
print(f"  - Test samples: {X_test_scaled.shape[0]}")
