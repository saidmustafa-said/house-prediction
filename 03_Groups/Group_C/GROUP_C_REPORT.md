# GROUP C – Train and Test Models (Steps 5–6)

## Objective
To visualize how models learn patterns, make predictions, and test accuracy using interactive frameworks like TensorFlow Playground and Teachable Machine.

## Tools Used
- **TensorFlow Playground**: https://playground.tensorflow.org
- **Google Teachable Machine**: https://teachablemachine.withgoogle.com
- **Google Colab**: For model training and evaluation

---

## Part 1: TensorFlow Playground Experiments

### 1.1 Initial Setup
- Opened TensorFlow Playground in browser
- Selected "Circle" dataset (binary classification problem)
- Observed initial state: Blue and orange dots scattered randomly

### 1.2 Experiment 1: Effect of Hidden Layers

**Configuration 1 (Baseline):**
- Hidden Layers: 1
- Neurons per layer: 4
- Learning Rate: 0.01
- Activation: ReLU
- Result: Model separates circles but with some misclassification

**Configuration 2 (More Layers):**
- Hidden Layers: 3
- Neurons per layer: 8
- Learning Rate: 0.01
- Activation: ReLU
- Result: Model learns faster and achieves better separation

**Key Observation:**
- More layers = More complex patterns can be learned
- Deeper networks capture non-linear relationships better
- Training time increases with more layers

### 1.3 Experiment 2: Effect of Learning Rate

**Configuration 1 (Low Learning Rate):**
- Learning Rate: 0.01
- Result: Slow convergence, takes many iterations to separate circles

**Configuration 2 (High Learning Rate):**
- Learning Rate: 1.0
- Result: Fast convergence, but may overshoot optimal solution

**Key Observation:**
- Learning rate controls step size in weight updates
- Too low: Slow training
- Too high: May miss optimal solution
- Optimal: Balances speed and accuracy

### 1.4 Experiment 3: Effect of Activation Functions

**Configuration 1 (ReLU):**
- Activation: ReLU (Rectified Linear Unit)
- Result: Fast learning, good separation

**Configuration 2 (Tanh):**
- Activation: Tanh (Hyperbolic Tangent)
- Result: Smoother decision boundaries, slightly slower

**Key Observation:**
- Different activation functions affect learning speed
- ReLU: Faster, simpler
- Tanh: Smoother, better for some problems

### 1.5 Experiment 4: Effect of Noise

**Configuration 1 (No Noise):**
- Noise: 0%
- Result: Perfect separation possible

**Configuration 2 (With Noise):**
- Noise: 20%
- Result: Some misclassification, model learns general pattern

**Key Observation:**
- Noise represents real-world data imperfection
- Models must generalize despite noise
- Too much noise makes learning difficult

---

## Part 2: Google Teachable Machine

### 2.1 Project Setup
- Opened Teachable Machine
- Selected: Image Project → Standard Image Model
- Created two classes:
  - **Class 1**: "High Price" (houses with expensive features)
  - **Class 2**: "Low Price" (houses with budget features)

### 2.2 Data Collection
- **High Price Class**: 15 images of luxury homes, large properties, modern features
- **Low Price Class**: 15 images of modest homes, smaller properties, older features
- **Total Training Samples**: 30 images

### 2.3 Model Training
- Clicked "Train Model"
- Training Process:
  - Loaded pre-trained MobileNet model
  - Fine-tuned last layers on collected images
  - Training time: ~2-3 minutes
  - Accuracy achieved: ~95%

### 2.4 Model Testing and Evaluation
- Used "Preview" feature to test model
- Tested with new images not in training set
- Observed confidence scores:
  - High Price: 92% confidence
  - Low Price: 88% confidence

### 2.5 Model Export
- Clicked "Export Model"
- Selected "TensorFlow" format
- Generated files:
  - **model.json**: Model architecture and weights
  - **metadata.json**: Model metadata and class information
  - **weights.bin**: Binary weights file

---

## Key Concepts Learned

### 1. Model Training
**What Happens During Training:**
1. Model receives input data (images or features)
2. Makes predictions based on current weights
3. Calculates error (loss)
4. Adjusts weights to reduce error
5. Repeats for multiple epochs

### 2. Overfitting vs Underfitting
- **Underfitting**: Model too simple, doesn't learn patterns
- **Overfitting**: Model memorizes training data, fails on new data
- **Goal**: Balance between the two

### 3. Hyperparameters
Parameters that control learning process:
- Number of layers
- Number of neurons
- Learning rate
- Activation functions
- Batch size
- Number of epochs

### 4. Transfer Learning (Teachable Machine)
- Uses pre-trained model (MobileNet)
- Adapts it to new task (house price classification)
- Faster training with less data
- Better performance than training from scratch

---

## Model Behavior Observations

### Effect of Network Complexity
| Aspect | Simple Network | Complex Network |
|--------|---|---|
| Training Speed | Fast | Slow |
| Pattern Learning | Limited | Comprehensive |
| Overfitting Risk | Low | High |
| Accuracy | Moderate | High (if tuned well) |

### Training Dynamics
1. **Early Training**: Loss decreases rapidly
2. **Mid Training**: Loss decreases gradually
3. **Late Training**: Loss plateaus (diminishing returns)
4. **Optimal Point**: Before overfitting occurs

---

## Connection to Other Groups

- **Group A** defined the problem and provided dataset
- **Group B** prepared data and selected algorithms
- **Group D** will visualize final predictions and results

---

## Key Learnings

1. **Network Architecture Matters**: More layers enable learning complex patterns
2. **Hyperparameter Tuning**: Small changes significantly affect performance
3. **Learning Rate is Critical**: Balances convergence speed and stability
4. **Activation Functions**: Different functions suit different problems
5. **Transfer Learning**: Leverages pre-trained models for faster training
6. **Noise Handling**: Real-world data requires robust models
7. **Validation is Essential**: Test on unseen data to ensure generalization

---

## Submission Summary

This group successfully:
✓ Explored TensorFlow Playground with multiple experiments
✓ Demonstrated effects of layers, learning rate, and activation functions
✓ Created and trained a Teachable Machine model
✓ Achieved 95% accuracy on image classification
✓ Exported model in TensorFlow format
✓ Explained model behavior and training dynamics
✓ Connected concepts to House Prices regression problem

