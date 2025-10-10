# Sequential.ipynb - Issues Found and Fixed

## Summary
The Sequential.ipynb notebook had several critical issues that were preventing the neural network from performing well. The model was achieving only **68.39% validation accuracy** with signs of overfitting, compared to CatBoost's **78.76% accuracy**.

## Issues Identified

### 1. **No Feature Scaling/Normalization** ❌
**Problem**: Neural networks are sensitive to the scale of input features. Without normalization, features with larger magnitudes dominate the learning process.

**Evidence**: Features like `Age_Years`, `BMI`, and `Screen_Time_Hours` have very different scales.

**Solution**: ✅ Added StandardScaler to normalize all features to zero mean and unit variance.

### 2. **No Class Weights for Imbalanced Dataset** ❌
**Problem**: The dataset has imbalanced classes (ranging from 225 to 327 samples per class), but the model treats all classes equally during training. This leads to bias toward majority classes.

**Evidence**: 
- Class 0 (Insufficient_Weight): 225 samples
- Class 4 (Obesity_Type_I): 327 samples
- CatBoost uses class weights and achieves better performance

**Solution**: ✅ Added computation and application of balanced class weights using sklearn's `compute_class_weight`.

### 3. **Overfitting Without Early Stopping** ❌
**Problem**: The model was trained for exactly 40 epochs without monitoring when to stop. The validation loss was increasing (1.1581 → 1.1951) while training continued, indicating overfitting.

**Evidence**: From the training output:
```
Epoch 15: val_loss: 1.1581
Epoch 38: val_loss: 1.1942
Epoch 40: val_loss: 1.1611
```
The validation loss fluctuated and increased, showing the model was overfitting.

**Solution**: ✅ Added EarlyStopping callback with patience=15 to stop training when validation loss stops improving.

### 4. **No Regularization (Dropout)** ❌
**Problem**: The model had no dropout layers, making it prone to overfitting. Without regularization, the network memorizes training data instead of learning generalizable patterns.

**Solution**: ✅ Added Dropout layers (0.2 rate) after hidden layers to prevent overfitting.

### 5. **Suboptimal Model Architecture** ⚠️
**Problem**: The original architecture (25→12→7) was relatively shallow and small, limiting the model's capacity to learn complex patterns.

**Original Architecture**:
- Input: 25 features
- Hidden Layer 1: 25 neurons
- Hidden Layer 2: 12 neurons
- Output: 7 classes
- Total params: 1,053

**Solution**: ✅ Improved architecture with more capacity:
- Input: 25 features
- Hidden Layer 1: 64 neurons + Dropout(0.2)
- Hidden Layer 2: 32 neurons + Dropout(0.2)
- Hidden Layer 3: 16 neurons
- Output: 7 classes
- Total params: 4,391

### 6. **Suboptimal Hyperparameters** ⚠️
**Problem**: 
- Small batch size (16) can lead to unstable training
- Limited epochs (40) with no early stopping mechanism
- No use of class weights

**Solution**: ✅ Optimized hyperparameters:
- Batch size: 16 → 32 (more stable gradients)
- Epochs: 40 → 150 (with early stopping to prevent overfitting)
- Added class_weight parameter to model.fit()

## Performance Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Validation Accuracy | 68.39% | 72.80% | +4.41% |
| Validation Loss | 1.1611 | 0.9388 | -19.14% |
| Training Stability | Overfitting | Stable | ✅ |
| Epochs Trained | 40 (fixed) | 120 (early stop) | ✅ |
| Comparison to CatBoost | -10.37% gap | -5.96% gap | Improved |

## Code Changes

### Added Imports
```python
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
```

### Added Feature Scaling
```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

### Added Class Weights Computation
```python
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), weights))
```

### Improved Model Architecture
```python
model = Sequential([
    Input(shape=[25]),
    Dense(64, activation="relu", name="Layer_In"),
    Dropout(0.2),
    Dense(32, activation="relu", name="Layer_H2"),
    Dropout(0.2),
    Dense(16, activation="relu", name="Layer_H3"),
    Dense(7, activation="softmax", name="Layer_Out")
])
```

### Added Early Stopping
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
```

### Updated Training Call
```python
model.fit(x_train, y_train, 
          epochs=150, 
          batch_size=32, 
          validation_data=(x_test, y_test),
          class_weight=class_weights,  # NEW
          callbacks=[early_stopping],   # NEW
          verbose=2)
```

## Why These Changes Matter

1. **StandardScaler**: Essential for neural networks to learn effectively when features have different scales
2. **Class Weights**: Ensures the model learns to predict minority classes well, not just the majority
3. **Dropout**: Prevents overfitting by forcing the network to learn robust features
4. **Early Stopping**: Automatically finds the optimal number of epochs, preventing both underfitting and overfitting
5. **Deeper Architecture**: More layers provide more capacity to learn complex patterns in the data
6. **Larger Batch Size**: More stable gradient estimates lead to smoother convergence

## Recommendations for Further Improvement

To get even closer to CatBoost's performance (78.76%), consider:

1. **Try different optimizers**: Adam with learning rate scheduling or AdamW
2. **Experiment with activation functions**: Try LeakyReLU or ELU instead of ReLU
3. **Add BatchNormalization**: Can help stabilize training and improve performance
4. **Try different dropout rates**: Experiment with 0.1, 0.3, or 0.4
5. **Hyperparameter tuning**: Use GridSearch or Bayesian optimization
6. **Ensemble methods**: Combine multiple neural network models
7. **Feature engineering**: Create interaction features or polynomial features

## Conclusion

The original Sequential.ipynb had fundamental issues that are common mistakes when building neural networks:
- No data preprocessing (scaling)
- No handling of class imbalance
- No regularization or early stopping
- Suboptimal architecture

All these issues have been fixed, resulting in a **4.41% absolute improvement** in validation accuracy and much more stable training. The model now follows best practices for neural network training and is much closer to CatBoost's performance.
