"""
================================================================================
TRAINING MODEL BACKPROPAGATION UNTUK PREDIKSI KUALITAS WINE
================================================================================
Script untuk:
1. Load dan preprocess dataset
2. Membuat model neural network
3. Training model dengan backpropagation
4. Evaluasi model
5. Simpan model dan scaler
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import warnings
warnings.filterwarnings('ignore')

# Membuat folder 'static' jika belum ada untuk menyimpan hasil visualisasi
if not os.path.exists('static/img'):
    os.makedirs('static/img')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("TRAINING MODEL BACKPROPAGATION - WINE QUALITY PREDICTION")
print("=" * 80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[1] LOADING DATASET...")
# PERBAIKAN: Menggunakan path lokal Windows (relatif)
dataset = pd.read_csv('data/WineQT.csv')
print(f"✓ Dataset loaded: {dataset.shape[0]} samples, {dataset.shape[1]} features")

# ============================================================================
# 2. EXPLORASI DATA
# ============================================================================
print("\n[2] EXPLORASI DATA...")
print("\n📊 INFO DATASET:")
print(dataset.info())

print("\n📈 STATISTIK DATASET:")
print(dataset.describe())

print("\n🔍 FIRST 5 ROWS:")
print(dataset.head())

print("\n❓ MISSING VALUES:")
print(dataset.isnull().sum())

# ============================================================================
# 3. PREPROCESSING DATA
# ============================================================================
print("\n[3] DATA PREPROCESSING...")

# Hilangkan kolom Id jika ada (tidak digunakan untuk training)
if 'Id' in dataset.columns:
    dataset = dataset.drop('Id', axis=1)

# Ambil fitur (X) dan target (y)
X = dataset.iloc[:, :-1].values  # Semua kolom kecuali quality
y = dataset.iloc[:, -1].values   # Kolom quality

# Ubah target menjadi klasifikasi BINARY
# quality >= 6 → 1 (GOOD), quality < 6 → 0 (BAD)
y_binary = (y >= 6).astype(int)

print(f"✓ Target distribution:")
unique, counts = np.unique(y_binary, return_counts=True)
for label, count in zip(unique, counts):
    label_name = "GOOD" if label == 1 else "BAD"
    percentage = (count / len(y_binary)) * 100
    print(f"  {label_name}: {count} samples ({percentage:.2f}%)")

# ============================================================================
# 4. NORMALISASI DATA
# ============================================================================
print("\n[4] NORMALISASI DATA...")

# Inisialisasi scaler
scaler_X = StandardScaler()

# Fit dan transform X
X_scaled = scaler_X.fit_transform(X)

print(f"✓ Features dinormalisasi menggunakan StandardScaler")
print(f"  - Mean: {X_scaled.mean():.4f}")
print(f"  - Std: {X_scaled.std():.4f}")

# ============================================================================
# 5. SPLIT DATA TRAIN-TEST
# ============================================================================
print("\n[5] SPLIT DATA TRAIN-TEST...")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")
print(f"✓ Input features: {X_train.shape[1]}")

# ============================================================================
# 6. MEMBUAT MODEL BACKPROPAGATION
# ============================================================================
print("\n[6] MEMBUAT MODEL NEURAL NETWORK...")

model = Sequential([
    # Input layer + Hidden layer 1
    Dense(64, activation='relu', input_dim=X_train.shape[1], 
          name='hidden_layer_1'),
    Dropout(0.2),
    
    # Hidden layer 2
    Dense(32, activation='relu', name='hidden_layer_2'),
    Dropout(0.2),
    
    # Hidden layer 3
    Dense(16, activation='relu', name='hidden_layer_3'),
    Dropout(0.1),
    
    # Output layer
    Dense(1, activation='sigmoid', name='output_layer')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture:")
model.summary()

# ============================================================================
# 7. TRAINING MODEL (BACKPROPAGATION)
# ============================================================================
print("\n[7] TRAINING MODEL (BACKPROPAGATION)...")

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Training model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print("\n✓ Model training completed!")

# ============================================================================
# 8. EVALUASI MODEL
# ============================================================================
print("\n[8] EVALUASI MODEL...")

# Prediksi pada test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Evaluasi
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n📊 TEST SET EVALUATION:")
print(f"  - Test Loss: {test_loss:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n🔍 CONFUSION MATRIX:")
print(cm)

# Classification Report
print(f"\n📈 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, 
      target_names=['BAD (0)', 'GOOD (1)']))

# ============================================================================
# 9. VISUALISASI HASIL TRAINING
# ============================================================================
print("\n[9] VISUALISASI HASIL TRAINING...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Training & Evaluation Results', fontsize=16, fontweight='bold')

# Plot 1: Training Loss vs Validation Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Loss Over Epochs', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training Accuracy vs Validation Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Accuracy Over Epochs', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['BAD', 'GOOD'], yticklabels=['BAD', 'GOOD'])
axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# Plot 4: Model Metrics
metrics = ['Accuracy', 'Loss']
values = [test_accuracy, test_loss]
colors = ['#2ecc71', '#e74c3c']
axes[1, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 1].set_title('Test Set Metrics', fontweight='bold')
axes[1, 1].set_ylabel('Value')
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
# PERBAIKAN: Path penyimpanan gambar lokal
plt.savefig('static/img/training_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: static/img/training_results.png")
plt.close()

# ============================================================================
# 10. SIMPAN MODEL DAN SCALER
# ============================================================================
print("\n[10] MENYIMPAN MODEL DAN SCALER...")

# PERBAIKAN: Simpan di folder saat ini (lokal)
model.save('model.keras')
print("✓ Model saved: model.keras")

# PERBAIKAN: Simpan scaler di folder saat ini (lokal)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
print("✓ Scaler saved: scaler.pkl")

# ============================================================================
# 11. RINGKASAN
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\n📋 SUMMARY:")
print(f"  ✓ Dataset: {dataset.shape[0]} samples")
print(f"  ✓ Target: Binary Classification (GOOD/BAD)")
print(f"  ✓ Model: Neural Network (4 layers)")
print(f"  ✓ Training Loss: {history.history['loss'][-1]:.4f}")
print(f"  ✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  ✓ Model saved: model.keras")
print(f"  ✓ Scaler saved: scaler.pkl")
print("\n" + "=" * 80)
print("Siap untuk digunakan di Flask Web App!")
print("=" * 80)